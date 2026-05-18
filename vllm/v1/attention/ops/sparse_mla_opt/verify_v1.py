#!/usr/bin/env python3
"""Correctness check: v1 optimized decode kernel vs. baseline.

Random uint8 scales saturate softmax (exp(255-127)=inf), so the
benchmark inputs aren't suitable for correctness — they happen to
produce all-zero outputs from both kernels.  Here we pre-fill the
cache with bounded fp8 bytes and small bounded scales so the output
is in a well-defined range, then diff the two kernels.
"""
import sys
import torch

import test_sparse_mla as _tsm
_tsm.torch = torch
sys.path.insert(0, ".")
from test_sparse_mla import build_inputs, TEST_CONFIGS  # noqa: E402

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    rocm_sparse_attn_decode,
)
from vllm.v1.attention.ops.sparse_mla_opt.rocm_aiter_mla_sparse_v1 import (  # noqa: E402
    rocm_sparse_attn_decode_v1,
)


def _fill_cache(c: torch.Tensor):
    """Write bounded fp8 data bytes + scale=127 to a strided cache view."""
    n = c.numel()  # logical numel via stride view; underlying buffer is bigger
    storage = c.untyped_storage()
    nbytes = storage.size()
    flat = torch.empty(0, dtype=torch.uint8, device=c.device).set_(storage)
    flat = torch.as_strided(flat, (nbytes,), (1,))
    # Random fp8 bytes around small magnitudes (exp 0110..0111 in e4m3 →
    # values ~0.5–1.5).
    rand = torch.empty(nbytes, dtype=torch.uint8, device=c.device)
    rand.random_(0x30, 0x40)
    flat.copy_(rand)
    # Override per-block scale section (`block_size * 8` bytes at offset
    # `block_size * 576` from each block start).
    n_blocks, block_size, _ = c.shape
    stride0 = c.stride(0)
    scale_offset_in_block = block_size * 576
    scale_len = block_size * 8
    for b in range(n_blocks):
        s = b * stride0 + scale_offset_in_block
        flat[s : s + scale_len] = 127  # exp2(0) = 1.0


def make_inputs(batch, mode, topk_ragged_len):
    inp = build_inputs(batch, mode, topk_ragged_len)
    inp["q"] = 0.05 * torch.randn_like(inp["q"])
    _fill_cache(inp["swa_k_cache"])
    if (inp["kv_cache"] is not None
            and inp["kv_cache"].data_ptr() != inp["swa_k_cache"].data_ptr()):
        _fill_cache(inp["kv_cache"])
    return inp


def run_and_diff(batch, mode, topk_ragged_len):
    inp = make_inputs(batch, mode, topk_ragged_len)
    out_base = torch.empty_like(inp["output"])
    out_v1 = torch.empty_like(inp["output"])
    rocm_sparse_attn_decode(**{**inp, "output": out_base})
    rocm_sparse_attn_decode_v1(**{**inp, "output": out_v1})

    a = out_base.float()
    b = out_v1.float()
    diff = (a - b).abs()
    return diff.max().item(), a.abs().max().item(), b.abs().max().item()


def main():
    print(f"{'label':<22s} {'max_abs':>12s} {'base_max':>10s} {'v1_max':>10s} {'rel':>10s}")
    print("-" * 78)
    bad = 0
    for batch, mode, topk_ragged_len, label in TEST_CONFIGS:
        try:
            max_abs, base_abs, v1_abs = run_and_diff(
                batch, mode, topk_ragged_len
            )
            ref = max(base_abs, 1e-6)
            rel = max_abs / ref
            tol = 0.05  # bf16 + fp8 dequant + reduce-order changes
            status = "OK " if rel < tol else "BAD"
            if status == "BAD":
                bad += 1
            print(f"{label:<22s} {max_abs:>12.4g} {base_abs:>10.4g} "
                  f"{v1_abs:>10.4g} {rel:>10.4g}  {status}")
        except Exception as e:
            print(f"{label:<22s}  FAILED: {e}")
            bad += 1
    print()
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
