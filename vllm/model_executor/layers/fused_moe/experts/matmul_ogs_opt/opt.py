#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Optimized matmul_ogs wrapper for DSV4-Pro MoE on CDNA4 (MI300/MI350).

Strategy:
  * The stock triton_kernels matmul_ogs disables split_k whenever the
    weights carry an MX-scale tensor (`can_use_split_k = ... and not w_has_mx`).
    For very small batch sizes (M <= 16, especially M = 1..4) the launch grid
    is tiny — only `n_routed_experts * grid_n` tiles — and a 256-CU GPU sits
    mostly idle.
  * The underlying `_matmul_ogs` Triton kernel actually does support
    `SPLIT_K > 1` with MX scales (it advances `WMxScalePtrs` by SPLIT_K each
    iter), and the host-side reduce/scratchpad logic in `matmul_ogs.py`
    already handles SPLIT_K > 1 correctly.
  * The `can_use_split_k` flag is only used to decide whether the
    `make_opt_flags` constraint check will reject `split_k > 1`. When we
    *directly* install an `OptFlags` instance via `set_opt_flags()`,
    `make_opt_flags` returns early and never consults that flag.

This module monkey-patches `triton_kernels.matmul_ogs.matmul_ogs` with a
wrapper that:
  1. Computes a tuned `OptFlags` for the call (split_k for tiny grids,
     deeper pipeline, etc).
  2. Installs it via `set_opt_flags()` so the constraint check is bypassed.
  3. Calls the original kernel entry point, then resets the override.

Tunings (CDNA4 / MXFP4 weights, BF16 activations):
  * Small W1 grids (scatter_indx is None) get split_k chosen to bring the
    total tile count up to roughly 2 * n_cu.
  * num_stages bumped to 3 for the BF16xMXFP4 path (the stock value is 2;
    deeper pipeline = more concurrent HBM loads).
  * waves_per_eu = 2 (3 is too aggressive when num_stages = 3 because of
    LDS pressure).
"""

from __future__ import annotations

import functools
import os

import torch
import triton

# Import the original symbols up front so monkey-patching happens once.
import triton_kernels.matmul_ogs as _mo_mod
from triton_kernels.matmul_ogs_details.opt_flags import (
    OptFlags,
    reset_opt_flags,
    set_opt_flags,
)
from triton_kernels.target_info import get_cdna_version
from triton_kernels.tensor import FP4, bitwidth

# Stash original entry point.
_orig_matmul_ogs = _mo_mod.matmul_ogs

# Knobs (overridable from env for sweeping during tuning).
_TARGET_TILES_PER_CU = float(os.environ.get("MATMUL_OGS_TARGET_TILES", "2.0"))
_MAX_SPLIT_K = int(os.environ.get("MATMUL_OGS_MAX_SPLIT_K", "8"))
_NUM_STAGES = int(os.environ.get("MATMUL_OGS_NUM_STAGES", "2"))
_NUM_WARPS = int(os.environ.get("MATMUL_OGS_NUM_WARPS", "4"))
_WAVES_PER_EU = int(os.environ.get("MATMUL_OGS_WAVES_PER_EU", "3"))
_BLOCK_M = int(os.environ.get("MATMUL_OGS_BLOCK_M", "0"))
_BLOCK_N = int(os.environ.get("MATMUL_OGS_BLOCK_N", "0"))
_BLOCK_K = int(os.environ.get("MATMUL_OGS_BLOCK_K", "0"))
_FORCE_SPLIT_K = int(os.environ.get("MATMUL_OGS_FORCE_SPLIT_K", "0"))


@functools.lru_cache(maxsize=1)
def _num_cu() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _is_supported(lhs_dtype, rhs_dtype, precision_config) -> bool:
    """Only intercept the BF16 x MXFP4 CDNA4 case the stock opt_flags
    targets — everything else falls through to the original kernel."""
    if get_cdna_version() != 4:
        return False
    if precision_config is None or precision_config.weight_scale is None:
        return False
    if bitwidth(lhs_dtype) != 16 or bitwidth(rhs_dtype) != 4:
        return False
    return True


def _routed_n_blocks(routing_data, m_kernel: int, block_m: int) -> int:
    """Approximate grid_m the same way matmul_ogs.py does."""
    if routing_data is None or routing_data.expt_data is None:
        return triton.cdiv(m_kernel, block_m)
    return routing_data.n_blocks(m_kernel, block_m)


def _build_opt_flags(
    *,
    x_dtype,
    w_dtype,
    precision_config,
    m_kernel: int,
    n: int,
    routing_data,
    scatter_indx,
) -> OptFlags | None:
    """Return tuned OptFlags, or None to fall back to stock behaviour."""
    if not _is_supported(x_dtype, w_dtype, precision_config):
        return None

    block_m = _BLOCK_M or 32
    block_n = _BLOCK_N or 128
    block_k = _BLOCK_K or 256

    grid_m = _routed_n_blocks(routing_data, m_kernel, block_m)
    grid_n = triton.cdiv(n, block_n)

    # Choose split_k. The reduce step disallows split_k > 1 when
    # scatter_indx is set, so only the W1 (gather-only) path benefits.
    if scatter_indx is not None or routing_data is None:
        split_k = 1
    elif _FORCE_SPLIT_K > 0:
        split_k = _FORCE_SPLIT_K
    else:
        target = int(_num_cu() * _TARGET_TILES_PER_CU)
        tiles = max(1, grid_m * grid_n)
        split_k = max(1, target // tiles)
        # Cap and round to a power-of-two-friendly value.
        split_k = min(split_k, _MAX_SPLIT_K)

    # Fall through to stock opt_flags when split_k=1 — stock already chose
    # the same block sizes / warp counts and applying our flags adds a tiny
    # set_opt_flags()/reset overhead per call.
    if split_k == 1:
        return None

    return OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        group_m=4,
        xcd_swizzle=8,
        w_cache_modifier=".cg" if block_m <= 32 else None,
        split_k=split_k,
        is_persistent=False,
        idle_sms=0,
        epilogue_subtile=1,
        arch=None,
        target_kernel_kwargs={
            "waves_per_eu": _WAVES_PER_EU,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
    )


def _matmul_ogs_opt(
    x,
    w,
    bias,
    routing_data=None,
    gather_indx=None,
    scatter_indx=None,
    precision_config=None,
    betas=None,
    gammas=None,
    out_alpha=None,
    y=None,
    y_acc_in=None,
    fused_activation=None,
    epilogue=None,
    fused_comm=None,
    inner_routing_data=None,
):
    # Compute "M" the way matmul_ogs does, so opt_flags get the right grid.
    if gather_indx is None:
        m_kernel = x.shape[-2]
    else:
        m_kernel = gather_indx.src_indx.shape[0]
    n = w.shape[-1]

    flags = _build_opt_flags(
        x_dtype=x.dtype,
        w_dtype=w.dtype if not hasattr(w, "storage") else w.dtype,
        precision_config=precision_config,
        m_kernel=m_kernel,
        n=n,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
    )

    if flags is None:
        # Fall through to stock kernel for unsupported configurations.
        return _orig_matmul_ogs(
            x, w, bias, routing_data,
            gather_indx=gather_indx, scatter_indx=scatter_indx,
            precision_config=precision_config, betas=betas, gammas=gammas,
            out_alpha=out_alpha, y=y, y_acc_in=y_acc_in,
            fused_activation=fused_activation, epilogue=epilogue,
            fused_comm=fused_comm, inner_routing_data=inner_routing_data,
        )

    reset_opt_flags()
    set_opt_flags(flags)
    try:
        return _orig_matmul_ogs(
            x, w, bias, routing_data,
            gather_indx=gather_indx, scatter_indx=scatter_indx,
            precision_config=precision_config, betas=betas, gammas=gammas,
            out_alpha=out_alpha, y=y, y_acc_in=y_acc_in,
            fused_activation=fused_activation, epilogue=epilogue,
            fused_comm=fused_comm, inner_routing_data=inner_routing_data,
        )
    finally:
        reset_opt_flags()


def install() -> None:
    """Replace triton_kernels.matmul_ogs.matmul_ogs in-process."""
    _mo_mod.matmul_ogs = _matmul_ogs_opt


def uninstall() -> None:
    _mo_mod.matmul_ogs = _orig_matmul_ogs


# Install on import so `python -c "import opt; from triton_kernels.matmul_ogs import matmul_ogs"`
# picks up the patched version. Re-imports of matmul_ogs *after* install pick
# up the patched reference via the module attribute.
install()
