# SPDX-License-Identifier: Apache-2.0
"""
Optimized pack_bitmatrix kernel v2: supports n_rows up to 256.

Key optimizations over the original (BLOCK_SIZE_M=512, BLOCK_SIZE_K=32):
1. BLOCK_SIZE_M reduced from 512 to 256 - halves register footprint for the
   M dimension while still handling the full token range.
2. BLOCK_SIZE_K reduced from 32 to 8 - only 6 topk entries are valid,
   so 26/32 lanes were wasted. Now only 2/8 are wasted (75% -> 25% waste).
3. Eliminated 3D tensor expansion - the original creates a
   [BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_K//32] tensor (16K elements).
   Since BLOCK_SIZE_K//32=1 with the original, that 3rd dim is always 1.
   We use direct 2D comparison + reduce, avoiding the broadcast and the
   extra dimension entirely.
4. Explicit num_warps=4 tuning for the reduced tile size.
"""

import triton
import triton.language as tl


@triton.jit
def pack_bitmatrix_v2(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        belongs_to_col = valid & (div == i)
        bits = tl.where(belongs_to_col, one << rem, tl.cast(0, tl.uint32))
        result = tl.reduce_or(bits, axis=1)
        store_mask = offsets_m < n_rows
        tl.store(bitmatrix + offsets_m * bm_cols + i, result, mask=store_mask)
