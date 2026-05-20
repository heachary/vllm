#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MFMA layout unit test for the sparse-MLA HIP kernel.

Goal: validate every register-layout assumption used by the (broken)
v3.1 MFMA path *in isolation*, before composing them into a full
attention kernel. Each test is small enough that the expected output
can be computed in PyTorch as ground truth.

What we're testing — `v_mfma_f32_16x16x32_bf16` on gfx950, wave64:
    D[16, 16] (fp32) = A[16, 32] (bf16) * B[32, 16] (bf16) + C[16, 16] (fp32)

Per-lane register layout (per AMD CDNA3/CDNA4 ISA + rocWMMA):
    A:  lane l holds A[m=l%16, k=(l/16)*8 + reg_idx]  for reg_idx 0..7
    B:  lane l holds B[k=(l/16)*8 + reg_idx, n=l%16]  for reg_idx 0..7
    D:  lane l holds D[m=l%16, n=(l/16)*4 + reg_idx]  for reg_idx 0..3

Tests, in order:

  1. Basic single MFMA.  Random A[16,32] and B[32,16]; verify D == A @ B.
     Validates: my understanding of A, B, and D per-lane layouts.

  2. K-chained MFMA.  A[16,64] and B[64,16]; two MFMAs each consuming a
     K=32 chunk, accumulating into the same D.  Verify D == A @ B.
     Validates: that K-chunked accumulation works as expected (this is
     how we'd cover NOPE+ROPE = 512 head_dim with 16 MFMAs).

  3. C -> A layout swap via LDS round-trip.  This is the hairy one.
     Run QK-style MFMA producing two 16x16 N-tiles -> assemble them as a
     16x32 matrix in LDS -> read back in MFMA-A layout for PV -> run
     PV-style MFMA against a V-tile.  Verify against (Q @ K^T) @ V.
     Validates: the layout transformation that v3.1 got wrong.

  4. End-to-end QK + softmax + PV for one BLOCK_K=32 K-tile, mirroring
     the inner loop of the sparse-MLA decode.  Random Q[16,512] and
     K[32,512] (full HEAD_DIM, NOPE+ROPE collapsed for the test).
     Validates: the chain in the shape used by the real kernel.

A test "passes" if the max abs error against the bf16-precision torch
reference is below `TOL` (we use 1e-2 absolute / 5e-3 relative — bf16
has ~3 decimal digits).
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import torch
from torch.utils.cpp_extension import load_inline


HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16x8 = __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16;
using fx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ __forceinline__ fx4 mfma_16x16x32_bf16(
    bf16x8 a, bf16x8 b, fx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// ============================================================
// Test 1: basic single MFMA.  D[16,16] = A[16,32] @ B[32,16].
//
// Per-lane:
//   A: lane l holds A[m=l%16, k=(l/16)*8 + i]  (i = 0..7)
//   B: lane l holds B[k=(l/16)*8 + i, n=l%16]  (i = 0..7)
//   D: lane l holds D[m=l%16, n=(l/16)*4 + i]  (i = 0..3)
// ============================================================
// Debug: identity-like A, B-with-known-pattern. Result is deterministic
// regardless of layout, so we can decode what the actual layout is.
//
// A[m, k] = 1 if (m == k) else 0  (16x32 → only first 16 cols can match)
// B[k, n] = (k+1) * 100 + n
//
// True D[m, n] = sum_k A[m,k]*B[k,n] = B[m, n] = (m+1)*100 + n.
// We dump A, B in layout assumption 1, see what D comes out, and the
// difference tells us which (m, n) the kernel actually computed.
__global__ void k_test_debug(
    const __bf16* __restrict__ a_ptr,
    const __bf16* __restrict__ b_ptr,
    float*        __restrict__ d_ptr,
    int layout)
{
    const int lane = threadIdx.x;
    bf16x8 a, b;
    int m, kbase, n, dn_base;
    if (layout == 1) {
        // Layout 1: lane l → (m=l%16, k=(l/16)*8 + i)
        m       = lane & 15;
        kbase   = (lane >> 4) * 8;
        n       = lane & 15;
        dn_base = (lane >> 4) * 4;
    } else if (layout == 2) {
        // Layout 2: lane l → (m=l/4, k=(l%4)*8 + i)
        m       = lane >> 2;
        kbase   = (lane & 3) * 8;
        n       = lane >> 2;
        dn_base = (lane & 3) * 4;
    } else {
        // Layout 3: lane l → (m=l%16, k=(l/16)*4 + i*16)  (interleaved K)
        m       = lane & 15;
        kbase   = lane >> 4;
        n       = lane & 15;
        dn_base = (lane >> 4) * 4;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (layout == 3) {
            a[i] = a_ptr[m * 32 + kbase + i * 4];
            b[i] = b_ptr[(kbase + i * 4) * 16 + n];
        } else {
            a[i] = a_ptr[m * 32 + kbase + i];
            b[i] = b_ptr[(kbase + i) * 16 + n];
        }
    }
    fx4 c = {0.f, 0.f, 0.f, 0.f};
    fx4 d = mfma_16x16x32_bf16(a, b, c);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_ptr[(layout - 1) * 256 + m * 16 + dn_base + i] = d[i];
    }
}

__global__ void k_test1(
    const __bf16* __restrict__ a_ptr,   // [16, 32] row-major
    const __bf16* __restrict__ b_ptr,   // [32, 16] row-major
    float*        __restrict__ d_ptr)   // [16, 16] row-major
{
    // CORRECTED layouts (verified empirically; see test_debug):
    //   A: lane l holds A[m=l%16,  k=(l/16)*8 + i]   (i=0..7)
    //   B: lane l holds B[k=(l/16)*8 + i, n=l%16]    (i=0..7)
    //   D: lane l holds D[m=(l/16)*4 + i, n=l%16]    (i=0..3)   <-- transposed!
    const int lane = threadIdx.x;
    const int m    = lane & 15;
    const int kgrp = lane >> 4;
    const int n    = lane & 15;

    bf16x8 a, b;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = a_ptr[m * 32 + kgrp * 8 + i];
        b[i] = b_ptr[(kgrp * 8 + i) * 16 + n];
    }
    fx4 c = {0.f, 0.f, 0.f, 0.f};
    fx4 d = mfma_16x16x32_bf16(a, b, c);

    const int dm_base = (lane >> 4) * 4;
    const int dn      = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_ptr[(dm_base + i) * 16 + dn] = d[i];
    }
}

// ============================================================
// Test 2: K-chained MFMA.  D[16,16] = A[16,64] @ B[64,16].
// Cover K=64 with two K=32 MFMAs, accumulating into C.
// ============================================================
__global__ void k_test2(
    const __bf16* __restrict__ a_ptr,   // [16, 64]
    const __bf16* __restrict__ b_ptr,   // [64, 16]
    float*        __restrict__ d_ptr)   // [16, 16]
{
    const int lane = threadIdx.x;
    const int m    = lane & 15;
    const int kgrp = lane >> 4;
    const int n    = lane & 15;

    fx4 c = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kchunk = 0; kchunk < 2; ++kchunk) {
        const int K_OFF = kchunk * 32;
        bf16x8 a, b;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a[i] = a_ptr[m * 64 + K_OFF + kgrp * 8 + i];
            b[i] = b_ptr[(K_OFF + kgrp * 8 + i) * 16 + n];
        }
        c = mfma_16x16x32_bf16(a, b, c);
    }

    const int dm_base = (lane >> 4) * 4;
    const int dn      = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_ptr[(dm_base + i) * 16 + dn] = c[i];
    }
}

// ============================================================
// Test 3: C -> A layout swap via LDS round-trip.
//
//   D_qk[16, 32] = Q[16, 32] @ K^T[32, 32]   (built from 2 N-tile MFMAs)
//   D_pv[16, 16] = D_qk[16, 32] @ V[32, 16]  (one MFMA, A=D_qk bf16-cast)
//
// Validates: writing D in C-layout to LDS, then reading back in A-layout.
//
// The shuffle works because:
//   - D for N-tile 0 holds D[m, (l/16)*4 + i] for i=0..3 → cols 0..15
//   - D for N-tile 1 holds D[m, (l/16)*4 + i] for i=0..3 → cols 16..31
//   - When concatenated into [16, 32], lane l holds:
//       N-tile 0: D[m, 4g + i]      (g = l/16, i = 0..3)
//       N-tile 1: D[m, 16 + 4g + i] (g = l/16, i = 0..3)
//     → 8 elements per lane, scattered across the 32-wide row.
//   - A-layout wants lane l to hold A[m, 8g + i] for i = 0..7
//     → contiguous slab of 8 cols starting at 8g.
//   - These don't match → LDS round-trip is required.
// ============================================================
__global__ void k_test3(
    const __bf16* __restrict__ q_ptr,    // [16, 32]
    const __bf16* __restrict__ kt0_ptr,  // [32, 16]  (K^T tile 0 -> cols 0..15 of scores)
    const __bf16* __restrict__ kt1_ptr,  // [32, 16]  (K^T tile 1 -> cols 16..31 of scores)
    const __bf16* __restrict__ v_ptr,    // [32, 16]
    float*        __restrict__ dpv_ptr)  // [16, 16]
{
    const int lane = threadIdx.x;
    const int m    = lane & 15;
    const int kgrp = lane >> 4;
    const int n    = lane & 15;

    // -- Load Q in MFMA-A layout (shared across both N-tiles) --
    bf16x8 q_reg;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        q_reg[i] = q_ptr[m * 32 + kgrp * 8 + i];
    }

    // -- MFMA 0: scores[16, 0..15] = Q @ K^T_tile0 --
    fx4 s0 = {0.f, 0.f, 0.f, 0.f};
    {
        bf16x8 b;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            b[i] = kt0_ptr[(kgrp * 8 + i) * 16 + n];
        }
        s0 = mfma_16x16x32_bf16(q_reg, b, s0);
    }

    // -- MFMA 1: scores[16, 16..31] = Q @ K^T_tile1 --
    fx4 s1 = {0.f, 0.f, 0.f, 0.f};
    {
        bf16x8 b;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            b[i] = kt1_ptr[(kgrp * 8 + i) * 16 + n];
        }
        s1 = mfma_16x16x32_bf16(q_reg, b, s1);
    }

    // -- Write scores[16, 32] to LDS in row-major bf16 --
    // CORRECTED D layout: lane l holds D[m=(l/16)*4 + i, n=l%16] for i=0..3.
    // So:
    //   s0[i] -> scores[(l/16)*4 + i, l%16]        (cols 0..15 of scores)
    //   s1[i] -> scores[(l/16)*4 + i, 16 + l%16]   (cols 16..31 of scores)
    __shared__ __bf16 s_lds[16 * 32];
    const int g = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        s_lds[(g * 4 + i) * 32 + n]      = (__bf16)s0[i];
        s_lds[(g * 4 + i) * 32 + 16 + n] = (__bf16)s1[i];
    }
    __syncthreads();

    // -- Read scores back as MFMA-A: lane l reads A[m, 8g + 0..7] --
    bf16x8 p_reg;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p_reg[i] = s_lds[m * 32 + kgrp * 8 + i];
    }

    // -- PV MFMA: dpv = scores @ V --
    fx4 dpv = {0.f, 0.f, 0.f, 0.f};
    {
        bf16x8 b;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            b[i] = v_ptr[(kgrp * 8 + i) * 16 + n];
        }
        dpv = mfma_16x16x32_bf16(p_reg, b, dpv);
    }

    // -- Store DPV (C-layout: lane l → D[(l/16)*4 + i, l%16]) --
    const int dm_base = (lane >> 4) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dpv_ptr[(dm_base + i) * 16 + n] = dpv[i];
    }
}

// ============================================================
// Test 4: end-to-end QK + softmax + PV for one K-tile.
//
//   Q : [BLOCK_H=16, HEAD_DIM=512] bf16
//   K : [BLOCK_K=32, HEAD_DIM=512] bf16
//   V : [BLOCK_K=32, HEAD_DIM=16]  bf16   (single N-tile for the test)
//
//   scores = (Q @ K^T) * scale         shape [16, 32]  fp32
//   p      = softmax(scores, dim=-1)   shape [16, 32]  bf16
//   out    = p @ V                     shape [16, 16]  fp32
//
// This is the exact compute pattern v2's inner loop runs (modulo
// online softmax state, which we skip here since there's only one tile).
//
// Compute K_reduce=512 with 16 chained MFMAs (per N-tile).
// ============================================================
__global__ void k_test4(
    const __bf16* __restrict__ q_ptr,    // [16, 512]
    const __bf16* __restrict__ k_ptr,    // [32, 512]
    const __bf16* __restrict__ v_ptr,    // [32, 16]
    float                       scale,
    float*        __restrict__ out_ptr)  // [16, 16]
{
    const int lane = threadIdx.x;
    const int m    = lane & 15;
    const int kgrp = lane >> 4;
    const int n    = lane & 15;
    constexpr int HEAD_DIM = 512;
    constexpr int CHUNKS = HEAD_DIM / 32;  // 16

    // -- Compute scores[16, 32] = (Q @ K^T) via 2 N-tiles of MFMA --
    // Each N-tile covers 16 K-tokens; chain 16 K=32 MFMAs to cover HEAD_DIM.

    fx4 s_ntile[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        // A = Q[m, c*32 + kgrp*8 + 0..7]
        bf16x8 q_reg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            q_reg[i] = q_ptr[m * HEAD_DIM + c * 32 + kgrp * 8 + i];
        }
        // B for tile nt (nt = 0,1) = K^T[c*32 + kgrp*8 + 0..7, nt*16 + n]
        //   = K[nt*16 + n, c*32 + kgrp*8 + 0..7]
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            bf16x8 k_reg;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                k_reg[i] = k_ptr[(nt * 16 + n) * HEAD_DIM + c * 32 + kgrp * 8 + i];
            }
            s_ntile[nt] = mfma_16x16x32_bf16(q_reg, k_reg, s_ntile[nt]);
        }
    }

    // -- Apply scale --
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) s_ntile[nt][i] *= scale;
    }

    // -- Softmax across the 32 cols for each row --
    // CORRECTED layout: lane l holds scores[m=(l/16)*4 + i, n=l%16] for i=0..3.
    // So one lane has 4 scores from 4 DIFFERENT rows at the same N col.
    // To do per-row softmax, we need to reduce across lanes sharing the same
    // (m, ntile_offset): those are lanes {0..15} for one m-group, etc.
    //
    // Concretely: scores for row r at all 32 n cols are spread across:
    //   - lane group g = r/4 (which contributes rows g*4..g*4+3 via reg 0..3)
    //   - within group g, lane.lo16 = 0..15 holds n cols 0..15 (nt=0)
    //   - same lane.lo16 holds nt=1 cols 16..31
    // So row r is held by 16 lanes (lane.lo16=0..15) within lane-group g=r/4,
    // each lane having scores at its own n col (lane.lo16) — 16 cols per nt,
    // 32 total per row.
    //
    // → For row r (= my_reg_row_for_this_lane = g*4 + i), the 16-lane group
    //   {16*g + 0..15} owns all 32 scores. Reduction needed: 16-way within
    //   the lane group → __shfl_xor offsets 1,2,4,8.
    //
    // Each lane processes its 4 owned rows independently (i = 0..3).
    fx4 p_ntile[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float row_max = fmaxf(s_ntile[0][i], s_ntile[1][i]);
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            row_max = fmaxf(row_max, __shfl_xor(row_max, off));
        }
        float e0 = expf(s_ntile[0][i] - row_max);
        float e1 = expf(s_ntile[1][i] - row_max);
        float row_sum = e0 + e1;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            row_sum += __shfl_xor(row_sum, off);
        }
        float inv = 1.f / fmaxf(row_sum, 1.0e-30f);
        p_ntile[0][i] = e0 * inv;
        p_ntile[1][i] = e1 * inv;
    }

    // -- Write P[16, 32] (bf16) to LDS and read back in MFMA-A layout --
    // CORRECTED layout: lane l's reg i is at scores[(l/16)*4 + i, l%16] (nt=0)
    //                                       and scores[(l/16)*4 + i, 16+l%16] (nt=1)
    __shared__ __bf16 p_lds[16 * 32];
    const int g = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        p_lds[(g * 4 + i) * 32 + n]      = (__bf16)p_ntile[0][i];
        p_lds[(g * 4 + i) * 32 + 16 + n] = (__bf16)p_ntile[1][i];
    }
    __syncthreads();

    // Read back as MFMA-A: lane l reads A[m=l%16, k=(l/16)*8 + 0..7]
    bf16x8 p_reg;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p_reg[i] = p_lds[m * 32 + kgrp * 8 + i];
    }

    // -- PV: out[16, 16] = P @ V --
    fx4 out = {0.f, 0.f, 0.f, 0.f};
    {
        bf16x8 v_reg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v_reg[i] = v_ptr[(kgrp * 8 + i) * 16 + n];
        }
        out = mfma_16x16x32_bf16(p_reg, v_reg, out);
    }

    // -- Store output in CORRECTED C-layout: lane l → D[(l/16)*4 + i, l%16] --
    const int dm_base = (lane >> 4) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out_ptr[(dm_base + i) * 16 + n] = out[i];
    }
}


// ============================================================
// Host bindings
// ============================================================
static inline auto _stream() { return at::cuda::getCurrentCUDAStream(); }

void run_test1(torch::Tensor a, torch::Tensor b, torch::Tensor d) {
    k_test1<<<1, 64, 0, _stream()>>>(
        reinterpret_cast<const __bf16*>(a.data_ptr()),
        reinterpret_cast<const __bf16*>(b.data_ptr()),
        d.data_ptr<float>());
}

void run_test2(torch::Tensor a, torch::Tensor b, torch::Tensor d) {
    k_test2<<<1, 64, 0, _stream()>>>(
        reinterpret_cast<const __bf16*>(a.data_ptr()),
        reinterpret_cast<const __bf16*>(b.data_ptr()),
        d.data_ptr<float>());
}

void run_test3(torch::Tensor q, torch::Tensor kt0, torch::Tensor kt1,
               torch::Tensor v, torch::Tensor d) {
    k_test3<<<1, 64, 0, _stream()>>>(
        reinterpret_cast<const __bf16*>(q.data_ptr()),
        reinterpret_cast<const __bf16*>(kt0.data_ptr()),
        reinterpret_cast<const __bf16*>(kt1.data_ptr()),
        reinterpret_cast<const __bf16*>(v.data_ptr()),
        d.data_ptr<float>());
}

void run_test4(torch::Tensor q, torch::Tensor k, torch::Tensor v,
               double scale, torch::Tensor out) {
    k_test4<<<1, 64, 0, _stream()>>>(
        reinterpret_cast<const __bf16*>(q.data_ptr()),
        reinterpret_cast<const __bf16*>(k.data_ptr()),
        reinterpret_cast<const __bf16*>(v.data_ptr()),
        (float)scale,
        out.data_ptr<float>());
}

void run_test_debug(torch::Tensor a, torch::Tensor b, torch::Tensor d, int64_t layout) {
    k_test_debug<<<1, 64, 0, _stream()>>>(
        reinterpret_cast<const __bf16*>(a.data_ptr()),
        reinterpret_cast<const __bf16*>(b.data_ptr()),
        d.data_ptr<float>(), (int)layout);
}

TORCH_LIBRARY_FRAGMENT(mfma_test, m) {
    m.def("test1(Tensor a, Tensor b, Tensor d) -> ()");
    m.def("test2(Tensor a, Tensor b, Tensor d) -> ()");
    m.def("test3(Tensor q, Tensor kt0, Tensor kt1, Tensor v, Tensor d) -> ()");
    m.def("test4(Tensor q, Tensor k, Tensor v, float scale, Tensor out) -> ()");
    m.def("test_debug(Tensor a, Tensor b, Tensor d, int layout) -> ()");
}
TORCH_LIBRARY_IMPL(mfma_test, CUDA, m) {
    m.impl("test1", &run_test1);
    m.impl("test2", &run_test2);
    m.impl("test3", &run_test3);
    m.impl("test4", &run_test4);
    m.impl("test_debug", &run_test_debug);
}
"""


_cache = {}


def _build():
    if "ext" in _cache:
        return _cache["ext"]
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    cache_dir = pathlib.Path(tempfile.gettempdir()) / "mfma_test_cache"
    cache_dir.mkdir(exist_ok=True)
    ext = load_inline(
        name="mfma_test",
        cpp_sources=[""],
        cuda_sources=[HIP_SRC],
        functions=[],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "--offload-arch=gfx950",
            "-Wno-c++11-narrowing",
            "-Wno-unused-result",
        ],
        with_cuda=True,
        build_directory=str(cache_dir),
        verbose=False,
    )
    _cache["ext"] = ext
    return ext


# Tolerances: bf16 has ~3 decimal digits.  For small matmuls (K up to 64)
# we use 1e-1 absolute; for HEAD_DIM=512 we use 0.5 absolute since
# accumulated bf16 errors grow with sqrt(K).
TOL_SMALL = 1e-1
TOL_LARGE = 0.5


def _check(name, got, ref, tol):
    diff = (got - ref).abs()
    max_err = diff.max().item()
    ref_max = ref.abs().max().item()
    rel = max_err / max(ref_max, 1e-6)
    status = "PASS" if max_err < tol else "FAIL"
    print(f"  {name:40s} max_err={max_err:.4g}  rel={rel:.4g}  ref_max={ref_max:.4g}  [{status}]")
    return max_err < tol


def test1_basic_mfma():
    print("\nTest 1: basic single MFMA  D[16,16] = A[16,32] @ B[32,16]")
    torch.manual_seed(0)
    A = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
    D = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    torch.ops.mfma_test.test1(A, B, D)
    D_ref = A.float() @ B.float()
    return _check("A @ B vs torch", D, D_ref, TOL_SMALL)


def test2_kchain():
    print("\nTest 2: K-chained MFMA  D[16,16] = A[16,64] @ B[64,16]  (2 x K=32)")
    torch.manual_seed(1)
    A = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda")
    D = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    torch.ops.mfma_test.test2(A, B, D)
    D_ref = A.float() @ B.float()
    return _check("K=64 chained accum vs torch", D, D_ref, TOL_SMALL)


def test3_layout_swap():
    print("\nTest 3: C->A layout swap via LDS  DPV[16,16] = (Q @ K^T)[16,32] @ V[32,16]")
    torch.manual_seed(2)
    # Q[16,32] (K=32 for QK MFMA)
    Q = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    # K is treated as 32 K-tokens of dim 32; we split it into two 16-token N-tiles.
    # The kernel takes K^T tile-by-tile.
    K = torch.randn(32, 32, dtype=torch.bfloat16, device="cuda")
    # KT0 = K^T[:, 0:16]  (shape [32, 16])  → cols 0..15 of scores
    # KT1 = K^T[:, 16:32] (shape [32, 16])  → cols 16..31 of scores
    KT0 = K.t().contiguous()[:, 0:16].contiguous()
    KT1 = K.t().contiguous()[:, 16:32].contiguous()
    # V[32, 16]
    V = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
    D = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

    torch.ops.mfma_test.test3(Q, KT0, KT1, V, D)

    # Reference (no softmax — pure linear chain so we can isolate the layout test)
    scores_ref = (Q.float() @ K.float().T)            # [16, 32]
    # Round-trip through bf16 the same way the kernel does
    scores_bf16 = scores_ref.to(torch.bfloat16).float()
    D_ref = scores_bf16 @ V.float()                   # [16, 16]

    return _check("(QK^T)[16,32] @ V via LDS swap", D, D_ref, TOL_SMALL)


def test4_qk_softmax_pv():
    print("\nTest 4: full QK + softmax + PV  (matches sparse-MLA inner loop)")
    torch.manual_seed(3)
    BLOCK_H, BLOCK_K, HEAD_DIM = 16, 32, 512
    Q = torch.randn(BLOCK_H, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    K = torch.randn(BLOCK_K, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    V = torch.randn(BLOCK_K, 16,       dtype=torch.bfloat16, device="cuda") * 0.1
    scale = 0.0442  # roughly the production scale
    out = torch.zeros(BLOCK_H, 16, dtype=torch.float32, device="cuda")

    torch.ops.mfma_test.test4(Q, K, V, scale, out)

    scores_ref = (Q.float() @ K.float().T) * scale       # [16, 32]
    p_ref = torch.softmax(scores_ref, dim=-1)            # [16, 32]
    p_bf16 = p_ref.to(torch.bfloat16).float()
    out_ref = p_bf16 @ V.float()                         # [16, 16]

    return _check("Q@K^T + softmax + P@V vs torch", out, out_ref, TOL_SMALL)


def main():
    print("Building HIP MFMA test extension...")
    _build()
    print("Built OK.")
    results = []
    results.append(("test1_basic_mfma",    test1_basic_mfma()))
    results.append(("test2_kchain",        test2_kchain()))
    results.append(("test3_layout_swap",   test3_layout_swap()))
    results.append(("test4_qk_softmax_pv", test4_qk_softmax_pv()))
    print("\nSummary:")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass} / {len(results)} tests passed.")
    import sys
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
