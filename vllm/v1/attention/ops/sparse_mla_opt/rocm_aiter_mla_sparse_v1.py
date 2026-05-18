# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized ROCm sparse-MLA decode kernel (v1).

Optimizations vs. the baseline `_sparse_attn_decode_ragged_kernel`:

1. BLOCK_H = 16 instead of 1.  The baseline launches one program per
   (query, head), which forces `tl.dot` to a degenerate M=1 shape.  The
   AMD matrix engine (mfma_*) needs M >= 16, so M=1 falls back to a
   scalarised dot product and discards almost all the MFMA throughput.
   With BLOCK_H = 16 every dot becomes a clean 16x16x16 MFMA tile, and
   the same K-block is shared by 16 heads instead of being reloaded 16
   times.

2. BLOCK_K = 32 — two MFMA N-tiles per K block, twice the work amortised
   per K load.

3. Split-K with a small reduce kernel.  For B=1 the partial grid is only
   (1, 64/16) = (1, 4) — far short of the 256 CUs on gfx950.  Split-K
   gives 4 * SPLIT_K programs per query so we can saturate the GPU
   without growing register pressure on the inner kernel.

4. Adaptive SPLIT_K (chosen on the host) — large for long contexts
   (B=1 full), small for early-decode where the K loop is tiny.

The kernel uses the same FP8 (uint8 + e8m0 group scale) cache layout as
the baseline: each token = 448 bytes nope (fp8) + 64 bf16 rope = 576
bytes data, followed by `block_size * 8` scale bytes per block.
"""

from __future__ import annotations

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


# Layout constants (same as baseline).
TOKEN_BYTES = tl.constexpr(576)  # 448 fp8 nope + 64*2 bf16 rope
SCALE_BYTES_PER_TOKEN = tl.constexpr(8)  # 7 e8m0 scales padded to 8


# ---------------------------------------------------------------------------
# Partial kernel — flash-attention online softmax over a sliced K range.
# ---------------------------------------------------------------------------
@triton.jit
def _v1_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    # scratch buffers
    scratch_m_ptr,
    scratch_l_ptr,
    scratch_acc_nope_ptr,
    scratch_acc_rope_ptr,
    # strides / params
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    num_head_blocks,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_hs = tl.program_id(1)

    pid_split = pid_hs // num_head_blocks
    pid_h = pid_hs - pid_split * num_head_blocks

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)
    k_offsets = tl.arange(0, BLOCK_K)

    # ------------------------------------------------------------------
    # Load Q once.  Q-nope is bf16 and shape [BLOCK_H, NOPE_BLOCK];
    # NOPE_BLOCK is the next pow2 of NOPE_DIM so it lines up with the
    # K-block we will materialise below.
    # ------------------------------------------------------------------
    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main (SWA) loop.
    # ------------------------------------------------------------------
    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    for k_start in tl.range(pid_split * BLOCK_K, main_len, BLOCK_K * SPLIT_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos,
                       mask=in_range, other=0)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot - block_idx * main_block_size
        cache_block_ptr = (
            main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        )
        token_data_ptr = cache_block_ptr + pos_in_block * TOKEN_BYTES
        token_scale_ptr = (
            cache_block_ptr
            + main_block_size * TOKEN_BYTES
            + pos_in_block * SCALE_BYTES_PER_TOKEN
        )

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        # exp2(x - 127) materialises the e8m0 group scale.
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
            q_rope, tl.trans(k_rope)
        )
        scores *= scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        p_bf16 = p.to(tl.bfloat16)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(p_bf16, k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p_bf16, k_rope)
        m_i = m_new
        l_i = l_new

    # ------------------------------------------------------------------
    # Extra (top-k) loop.
    # ------------------------------------------------------------------
    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(pid_split * BLOCK_K, extra_len, BLOCK_K * SPLIT_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(extra_indices_ptr + extra_start + k_pos,
                           mask=in_range, other=0)
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot - block_idx * extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * TOKEN_BYTES
            token_scale_ptr = (
                cache_block_ptr
                + extra_block_size * TOKEN_BYTES
                + pos_in_block * SCALE_BYTES_PER_TOKEN
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ:
                x_fp8 = x_uint8.to(tl.float8e4b15, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope, tl.trans(k_rope)
            )
            scores *= scale
            scores = tl.where(valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            p_bf16 = p.to(tl.bfloat16)
            acc_nope = acc_nope * alpha[:, None] + tl.dot(p_bf16, k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p_bf16, k_rope)
            m_i = m_new
            l_i = l_new

    # ------------------------------------------------------------------
    # Write partial (m, l, acc) for this (query, head_block, split) tile.
    # Layout: [Q * num_head_blocks * SPLIT_K, BLOCK_H, ...]
    # ------------------------------------------------------------------
    base = (query_idx * num_head_blocks + pid_h) * SPLIT_K + pid_split

    tl.store(scratch_m_ptr + base * BLOCK_H + tl.arange(0, BLOCK_H),
             m_i, mask=head_mask)
    tl.store(scratch_l_ptr + base * BLOCK_H + tl.arange(0, BLOCK_H),
             l_i, mask=head_mask)
    tl.store(
        scratch_acc_nope_ptr + base * BLOCK_H * NOPE_BLOCK
        + tl.arange(0, BLOCK_H)[:, None] * NOPE_BLOCK
        + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        scratch_acc_rope_ptr + base * BLOCK_H * ROPE_DIM
        + tl.arange(0, BLOCK_H)[:, None] * ROPE_DIM
        + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Reduce kernel — flash-attention log-sum-exp combine across splits.
# ---------------------------------------------------------------------------
@triton.jit
def _v1_reduce_kernel(
    scratch_m_ptr,
    scratch_l_ptr,
    scratch_acc_nope_ptr,
    scratch_acc_rope_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    num_heads,
    num_head_blocks,
    HAS_ATTN_SINK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    neg_large = -3.4028234663852886e38
    m = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)

    base0 = (query_idx * num_head_blocks + pid_h) * SPLIT_K
    h_arange = tl.arange(0, BLOCK_H)
    for s in tl.static_range(0, SPLIT_K):
        base = base0 + s
        m_s = tl.load(scratch_m_ptr + base * BLOCK_H + h_arange,
                      mask=head_mask, other=neg_large).to(tl.float32)
        l_s = tl.load(scratch_l_ptr + base * BLOCK_H + h_arange,
                      mask=head_mask, other=0.0).to(tl.float32)
        acc_nope_s = tl.load(
            scratch_acc_nope_ptr + base * BLOCK_H * NOPE_BLOCK
            + h_arange[:, None] * NOPE_BLOCK + nope_offsets[None, :],
            mask=head_mask[:, None] & nope_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_rope_s = tl.load(
            scratch_acc_rope_ptr + base * BLOCK_H * ROPE_DIM
            + h_arange[:, None] * ROPE_DIM + rope_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, m_s)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(m_s - m_new)
        l = l * alpha + l_s * beta
        acc_nope = acc_nope * alpha[:, None] + acc_nope_s * beta[:, None]
        acc_rope = acc_rope * alpha[:, None] + acc_rope_s * beta[:, None]
        m = m_new

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets,
                       mask=head_mask, other=neg_large).to(tl.float32)
        m_final = tl.maximum(m, sink)
        alpha = tl.exp(m - m_final)
        l_final = l * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l, 1.0e-30)
        out_nope = tl.where(l[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope.to(tl.bfloat16),
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Host glue.
# ---------------------------------------------------------------------------
def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.int32:
        x = x.to(torch.int32)
    if not x.is_contiguous():
        x = x.contiguous()
    return x.view(-1)


_NUM_CUS = 256  # gfx950 (MI350) CU count; fine to over-estimate on smaller parts
_SPLIT_K_OVERRIDE = os.environ.get("V1_SPLIT_K")


def _pick_split_k(num_queries: int, num_head_blocks: int) -> int:
    """Pick a SPLIT_K (constexpr power of two).

    Reducing SPLIT_K shrinks the per-split scratch buffer (the reduce
    kernel reads ``Q * num_head_blocks * SPLIT_K * BLOCK_H *
    (NOPE_BLOCK + ROPE) * 4`` bytes); profiling showed the reduce
    kernel can dominate when SPLIT_K is unnecessarily large.

    The trade-off is GPU occupancy in the partial kernel — for small
    ``num_queries * num_head_blocks`` we need a larger SPLIT_K to fill
    CUs.  Tuned values below balance both.
    """
    if _SPLIT_K_OVERRIDE is not None:
        return int(_SPLIT_K_OVERRIDE)
    qh = max(1, num_queries * num_head_blocks)
    target = max(1, _NUM_CUS // qh)
    bins = (1, 2, 4, 8, 16, 32, 64)
    best = 1
    for b in bins:
        if b <= target:
            best = b
    # Cap at 16: above this the reduce kernel scratch bandwidth eats
    # the savings from more parallel partial-kernel programs.
    return min(best, 16)


def _decode_v1(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_indptr: torch.Tensor | None,
) -> torch.Tensor:
    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device,
                                   dtype=torch.int32)

    BLOCK_H = 16
    BLOCK_K = 32
    NOPE_BLOCK = triton.next_power_of_2(nope_head_dim)

    num_head_blocks = triton.cdiv(num_heads, BLOCK_H)
    SPLIT_K = _pick_split_k(num_queries, num_head_blocks)

    out = torch.empty_like(q, dtype=torch.bfloat16)

    scratch_size = num_queries * num_head_blocks * SPLIT_K * BLOCK_H
    scratch_m = torch.empty(scratch_size, device=q.device, dtype=torch.float32)
    scratch_l = torch.empty(scratch_size, device=q.device, dtype=torch.float32)
    scratch_acc_nope = torch.empty(
        scratch_size * NOPE_BLOCK, device=q.device, dtype=torch.float32,
    )
    scratch_acc_rope = torch.empty(
        scratch_size * rope_head_dim, device=q.device, dtype=torch.float32,
    )

    grid_partial = (num_queries, num_head_blocks * SPLIT_K)
    _v1_partial_kernel[grid_partial](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        scratch_m,
        scratch_l,
        scratch_acc_nope,
        scratch_acc_rope,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        num_head_blocks,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=NOPE_BLOCK,
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=current_platform.is_fp8_fnuz(),
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
        num_stages=1,
    )

    grid_reduce = (num_queries, num_head_blocks)
    _v1_reduce_kernel[grid_reduce](
        scratch_m,
        scratch_l,
        scratch_acc_nope,
        scratch_acc_rope,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        num_heads,
        num_head_blocks,
        HAS_ATTN_SINK=has_attn_sink,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=NOPE_BLOCK,
        ROPE_DIM=rope_head_dim,
        BLOCK_H=BLOCK_H,
        SPLIT_K=SPLIT_K,
        num_warps=2,
    )
    return out


def rocm_sparse_attn_decode_v1(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """Drop-in replacement for `rocm_sparse_attn_decode`.

    Same signature, same semantics; uses the optimised partial+reduce
    kernels above.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
        _validate_dsv4_sparse_dims,
    )

    assert swa_k_cache.dtype == torch.uint8
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "rocm_sparse_attn_decode_v1",
    )

    # Resolve main (SWA) ragged indices.
    if swa_ragged_indices is None or swa_ragged_indptr is None:
        main_indices_dense = swa_indices.reshape(swa_indices.shape[0], -1)
        lengths = swa_lens if swa_lens is not None else (
            (main_indices_dense >= 0).sum(dim=-1, dtype=torch.int32)
        )
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices_dense,
            lengths,
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
        )
    else:
        main_ragged_indices = swa_ragged_indices
        main_ragged_indptr = swa_ragged_indptr

    # Resolve extra (top-k) ragged indices.
    has_extra = not swa_only
    extra_cache = None
    extra_ragged_indices = None
    extra_ragged_indptr = None
    if has_extra:
        assert kv_cache is not None
        assert kv_cache.dtype == torch.uint8
        extra_cache = kv_cache
        if topk_ragged_indices is None or topk_ragged_indptr is None:
            assert topk_indices is not None
            ex_dense = topk_indices.reshape(topk_indices.shape[0], -1)
            lengths = topk_lens if topk_lens is not None else (
                (ex_dense >= 0).sum(dim=-1, dtype=torch.int32)
            )
            extra_ragged_indices, extra_ragged_indptr = (
                build_ragged_indices_from_dense(
                    ex_dense,
                    lengths,
                    num_rows=kv_cache.shape[0] * kv_cache.shape[1],
                )
            )
        else:
            extra_ragged_indices = topk_ragged_indices
            extra_ragged_indptr = topk_ragged_indptr

    attn_out = _decode_v1(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
    )
    output.copy_(attn_out.to(output.dtype))
