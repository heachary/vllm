# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized ROCm sparse-MLA decode kernel (v3) — HIP MFMA implementation.

Iteration history:
  v3.3 (single-kernel MFMA):  HCA 60us, CSA 122-195us. Bottleneck: only
                              16 wgs on 256 CUs (no split-K), no latency
                              hiding within CU.
  v3.5 (this version):        + partial+reduce SPLIT_K to get more wgs.

MFMA layouts (gfx950 wave64, verified by `test_mfma_layouts.py`):
    A: lane l holds A[m=l%16,         k=(l/16)*8 + i]   for i=0..7
    B: lane l holds B[k=(l/16)*8 + i, n=l%16]           for i=0..7
    D: lane l holds D[m=(l/16)*4 + i, n=l%16]           for i=0..3

Online softmax row reduction is a 16-way `shfl_xor` butterfly across
`lane.lo4` — the 16 lanes that share the same `(l/16)` group together
own all N cols of one row.

Split-K + reduce merges across partials with the FlashDecoding+ formula:
  m_combined  = max(m_split for split in 0..SPLIT_K)
  alpha_split = exp(m_split - m_combined)
  l_combined  = sum(l_split * alpha_split)
  acc_combined = sum(acc_split * alpha_split)
then apply attn_sink and divide by l_combined.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import torch
from torch.utils.cpp_extension import load_inline


_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using bf16x8 = __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16;
using fx4    = __attribute__((__vector_size__(4 * sizeof(float))))  float;

static constexpr int NOPE_DIM    = 448;
static constexpr int ROPE_DIM    = 64;
static constexpr int TOKEN_BYTES = 576;
static constexpr int SCALE_BYTES = 8;
static constexpr int HEAD_DIM    = 512;
static constexpr int BLOCK_H     = 16;
static constexpr int BLOCK_K     = 32;
static constexpr int N_TILES     = HEAD_DIM / 16;  // 32
static constexpr int QK_N_TILES  = BLOCK_K / 16;   // 2

__device__ __forceinline__ fx4 mfma_16x16x32_bf16(
    bf16x8 a, bf16x8 b, fx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// =========================================================================
// Helper: cooperative K-tile gather + dequant.
// All 256 threads (4 waves) cooperate to populate
//   k_lds[BLOCK_K, HEAD_DIM] (bf16)
//   kv_lds[BLOCK_K]          (valid flags)
// Each thread handles 1 (token, chunk) unit (32 tokens × 8 chunks = 256).
// =========================================================================
__device__ __forceinline__ void gather_and_dequant_k_tile(
    int k_start, int k_len, const uint8_t* cache_base,
    int64_t cache_stride0, int num_rows, int block_size,
    const int32_t* idx_base,
    __bf16* k_lds, int8_t* kv_lds, int tid)
{
    const int tok_id = tid >> 3;  // 0..31
    const int chunk  = tid & 7;   // 0..7
    const int col0   = chunk * 64;

    int k_pos = k_start + tok_id;
    bool in_range = (k_pos < k_len);
    int slot = in_range ? idx_base[k_pos] : 0;
    bool valid = in_range && (slot >= 0) && (slot < num_rows);
    int safe_slot = valid ? slot : 0;
    int bi = safe_slot / block_size;
    int pib = safe_slot - bi * block_size;
    const uint8_t* block_ptr = cache_base
                               + (int64_t)bi * cache_stride0;
    const uint8_t* token_ptr = block_ptr + pib * TOKEN_BYTES;

    __bf16* dst_row = &k_lds[tok_id * HEAD_DIM + col0];

    if (!valid) {
        int4 z; z.x = z.y = z.z = z.w = 0;
        int4* d4 = reinterpret_cast<int4*>(dst_row);
        #pragma unroll
        for (int j = 0; j < 8; ++j) d4[j] = z;
    } else if (col0 < NOPE_DIM) {
        const uint8_t* scale_ptr = block_ptr
                                   + block_size * TOKEN_BYTES
                                   + pib * SCALE_BYTES;
        uint8_t scl_u = scale_ptr[chunk];
        union { uint32_t u; float fv; } sb;
        sb.u = ((uint32_t)scl_u) << 23;
        float scl_f = sb.fv;

        const uint32_t* src32 = reinterpret_cast<const uint32_t*>(
            token_ptr + col0);
        #pragma unroll
        for (int u32_i = 0; u32_i < 16; ++u32_i) {
            uint32_t word = src32[u32_i];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint8_t kb = (word >> (b * 8)) & 0xFF;
                uint32_t packed = (uint32_t)kb;
                float f = __builtin_amdgcn_cvt_f32_fp8(packed, 0) * scl_f;
                dst_row[u32_i * 4 + b] = (__bf16)f;
            }
        }
    } else {
        const int4* src4 = reinterpret_cast<const int4*>(
            token_ptr + NOPE_DIM);
        int4* d4 = reinterpret_cast<int4*>(dst_row);
        #pragma unroll
        for (int j = 0; j < 8; ++j) d4[j] = src4[j];
    }

    // Validity flags (1 byte per token). 32 tokens, 256 threads: thread 0..31
    // (token 0..3, chunk 0) write the flag. Actually use tid < 32.
    if (tid < BLOCK_K) {
        int kp = k_start + tid;
        int sl = (kp < k_len) ? idx_base[kp] : -1;
        kv_lds[tid] = (kp < k_len) && (sl >= 0) && (sl < num_rows) ? 1 : 0;
    }
}


// =========================================================================
// Helper: per-K-tile compute (QK + softmax + PV update) into acc[N_TILES].
// =========================================================================
// process_k_tile (v3.7): distributed PV across 4 waves.
//
// Layout:
//   Wave 0: QK MFMA → writes scores (fp32) to scores_lds.
//   All waves: read scores from LDS, compute softmax (consistent per-row),
//              write P (bf16, A-layout) to p_lds (only wave 0 writes;
//              others read).
//   All waves: read P in A-layout, do PV for their 8 N-tiles.
//   Each wave's acc holds 8 of the 32 N-tiles:
//      acc[n_tile_local] = global N-tile (wave * 8 + n_tile_local)
//      head_dim cols owned: [wave*128, wave*128+127]
//
// Per-wave acc footprint: 8 N-tiles × 4 fp32 = 32 VGPR/lane (was 128).
constexpr int N_TILES_PER_WAVE = 8;  // 32 N-tiles / 4 waves
__device__ __forceinline__ void process_k_tile(
    const __bf16* q_lds, const __bf16* k_lds, const int8_t* kv_lds,
    __bf16* p_lds, float* scores_lds,
    float* m_state, float* l_state, fx4* acc, float scale,
    int lane, int m_a, int kg, int n_b, int m_d_base, int n_d,
    int wave)
{
    // ----- Wave 0: QK MFMA + write scores -----
    if (wave == 0) {
        fx4 qk[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 32; ++c) {
            bf16x8 q_reg;
            const __bf16* q_src = &q_lds[m_a * HEAD_DIM + c * 32 + kg * 8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) q_reg[i] = q_src[i];

            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                bf16x8 k_reg;
                const __bf16* k_src = &k_lds[(nt * 16 + n_b) * HEAD_DIM
                                              + c * 32 + kg * 8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) k_reg[i] = k_src[i];
                qk[nt] = mfma_16x16x32_bf16(q_reg, k_reg, qk[nt]);
            }
        }
        // Apply scale + validity mask, write to scores_lds (fp32, [16,32]).
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int k_col = nt * 16 + n_d;
                float s = qk[nt][i] * scale;
                if (!kv_lds[k_col]) s = -3.4028234663852886e38f;
                scores_lds[(m_d_base + i) * BLOCK_K + nt * 16 + n_d] = s;
            }
        }
    }

    __syncthreads();

    // ----- All waves: read scores, do softmax independently -----
    // Each lane reads scores for its 4 owned rows (D-layout: row = m_d_base + i).
    fx4 qk_local[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qk_local[0][i] = scores_lds[(m_d_base + i) * BLOCK_K + n_d];
        qk_local[1][i] = scores_lds[(m_d_base + i) * BLOCK_K + 16 + n_d];
    }

    fx4 p[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float row_max = fmaxf(qk_local[0][i], qk_local[1][i]);
        row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 8));

        float m_new = fmaxf(m_state[i], row_max);
        float alpha = __builtin_amdgcn_exp2f(
            (m_state[i] - m_new) * 1.4426950408889634f);

        float e0 = __builtin_amdgcn_exp2f(
            (qk_local[0][i] - m_new) * 1.4426950408889634f);
        float e1 = __builtin_amdgcn_exp2f(
            (qk_local[1][i] - m_new) * 1.4426950408889634f);

        float row_sum = e0 + e1;
        row_sum += __shfl_xor(row_sum, 1);
        row_sum += __shfl_xor(row_sum, 2);
        row_sum += __shfl_xor(row_sum, 4);
        row_sum += __shfl_xor(row_sum, 8);

        float l_new = l_state[i] * alpha + row_sum;
        p[0][i] = e0;
        p[1][i] = e1;

        // Scale THIS wave's acc by alpha
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WAVE; ++nt) acc[nt][i] *= alpha;

        m_state[i] = m_new;
        l_state[i] = l_new;
    }

    // Wave 0 writes P (bf16) to p_lds in D-layout. Other waves no-op the
    // write but still hit the next __syncthreads.
    if (wave == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_lds[(m_d_base + i) * BLOCK_K + n_d]      = (__bf16)p[0][i];
            p_lds[(m_d_base + i) * BLOCK_K + 16 + n_d] = (__bf16)p[1][i];
        }
    }

    __syncthreads();

    // ----- All waves: read P in A-layout, do PV for own 8 N-tiles -----
    bf16x8 p_reg;
    const __bf16* p_src = &p_lds[m_a * BLOCK_K + kg * 8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) p_reg[i] = p_src[i];

    // Wave w handles N-tiles w*8 + 0..7. Global n_tile = wave*8 + nt_local.
    #pragma unroll
    for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
        int n_tile = wave * N_TILES_PER_WAVE + nt_local;
        bf16x8 k_reg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            k_reg[i] = k_lds[(kg * 8 + i) * HEAD_DIM
                             + n_tile * 16 + n_b];
        }
        acc[nt_local] = mfma_16x16x32_bf16(p_reg, k_reg, acc[nt_local]);
    }
}


// =========================================================================
// Helper: load Q[BLOCK_H, HEAD_DIM] into q_lds. All 64 threads cooperate.
// =========================================================================
// 256 threads cooperatively load Q[BLOCK_H=16, HEAD_DIM=512].
// 8192 bf16 / 256 threads = 32 bf16 / thread = 4 int4.
// Thread t -> head (t/16), col_start = (t%16)*32.
__device__ __forceinline__ void load_q(
    const __bf16* q, int64_t q_stride0, int64_t q_stride1,
    int query, int pid_h, int num_heads,
    __bf16* q_lds, int tid)
{
    const int qh  = tid >> 4;          // 0..15
    const int qc0 = (tid & 15) << 5;   // 0,32,...,480
    const int head_global = pid_h * BLOCK_H + qh;
    __bf16* dst = &q_lds[qh * HEAD_DIM + qc0];
    if (head_global < num_heads) {
        const __bf16* src = q + query * q_stride0
                              + head_global * q_stride1 + qc0;
        const int4* s4 = reinterpret_cast<const int4*>(src);
        int4* d4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) d4[i] = s4[i];
    } else {
        int4 z; z.x = z.y = z.z = z.w = 0;
        int4* d4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) d4[i] = z;
    }
}


// =========================================================================
// Single-kernel decode (SPLIT_K = 1). Direct bf16 output.
// =========================================================================
// Single-kernel decode. 256 threads = 4 waves per WG.
// Wave 0 holds and does MFMA; all 4 waves cooperate on gather + Q load.
template <bool HAS_ATTN_SINK, bool HAS_EXTRA>
__global__ __launch_bounds__(256, 2)
void v3_mfma_kernel(
    const __bf16* __restrict__ q,
    const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    const float* __restrict__ attn_sink,
    __bf16* __restrict__ output,
    int64_t q_stride0, int64_t q_stride1,
    int64_t out_stride0, int64_t out_stride1,
    int64_t main_cache_stride0, int64_t extra_cache_stride0,
    int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size,
    float scale, int num_heads)
{
    const int query = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;

    const int m_a       = lane & 15;
    const int kg        = lane >> 4;
    const int n_b       = lane & 15;
    const int m_d_base  = (lane >> 4) * 4;
    const int n_d       = lane & 15;

    __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
    __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
    __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
    __shared__ float  scores_lds[BLOCK_H * BLOCK_K];
    __shared__ int8_t kv_lds[BLOCK_K];
    __shared__ char   force_1wg_per_cu[48 * 1024];  // pads LDS to ~96 KB
    (void)force_1wg_per_cu;

    load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

    float m_state[4], l_state[4];
    fx4   acc[N_TILES_PER_WAVE];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_state[i] = -3.4028234663852886e38f;
        l_state[i] = 0.f;
    }
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
        acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
    }

    __syncthreads();

    {
        int main_start = main_indptr[query];
        int main_end   = main_indptr[query + 1];
        int main_len   = main_end - main_start;
        for (int k_start = 0; k_start < main_len; k_start += BLOCK_K) {
            gather_and_dequant_k_tile(
                k_start, main_len, main_cache, main_cache_stride0,
                main_num_rows, main_block_size,
                main_indices + main_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    if (HAS_EXTRA) {
        int extra_start = extra_indptr[query];
        int extra_end   = extra_indptr[query + 1];
        int extra_len   = extra_end - extra_start;
        for (int k_start = 0; k_start < extra_len; k_start += BLOCK_K) {
            gather_and_dequant_k_tile(
                k_start, extra_len, extra_cache, extra_cache_stride0,
                extra_num_rows, extra_block_size,
                extra_indices + extra_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    // Finalize: each wave writes its own 8 N-tiles slice of output.
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int head_local = m_d_base + i;
            int head_global = pid_h * BLOCK_H + head_local;
            if (head_global >= num_heads) continue;

            float m_final = m_state[i];
            float l_final = l_state[i];
            float alpha_final = 1.f;
            if (HAS_ATTN_SINK) {
                float sink_val = attn_sink[head_global];
                m_final = fmaxf(m_state[i], sink_val);
                alpha_final = __builtin_amdgcn_exp2f(
                    (m_state[i] - m_final) * 1.4426950408889634f);
                l_final = l_state[i] * alpha_final + __builtin_amdgcn_exp2f(
                    (sink_val - m_final) * 1.4426950408889634f);
            }
            float denom = fmaxf(l_final, 1.0e-30f);
            bool live = (l_final > 0.f);

            __bf16* out_row = output + query * out_stride0
                                     + head_global * out_stride1;
            #pragma unroll
            for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
                int n_tile = wave * N_TILES_PER_WAVE + nt_local;
                int col = n_tile * 16 + n_d;
                float v = live ? (acc[nt_local][i] * alpha_final) / denom : 0.f;
                out_row[col] = (__bf16)v;
            }
        }
    }
}


// =========================================================================
// Partial kernel (SPLIT_K > 1). Writes m_state, l_state, acc to scratch.
//
// Grid: (num_queries, num_head_blocks * SPLIT_K)
// scratch_m[Q, HB, SPLIT_K, BLOCK_H]                 fp32
// scratch_l[Q, HB, SPLIT_K, BLOCK_H]                 fp32
// scratch_acc[Q, HB, SPLIT_K, BLOCK_H, HEAD_DIM]     bf16  (saves bw vs fp32)
// =========================================================================
template <bool HAS_EXTRA, int SPLIT_K>
__global__ __launch_bounds__(256, 2)
void v3_mfma_partial_kernel(
    const __bf16* __restrict__ q,
    const uint8_t* __restrict__ main_cache,
    const int32_t* __restrict__ main_indices,
    const int32_t* __restrict__ main_indptr,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_indptr,
    float* __restrict__ scratch_m,
    float* __restrict__ scratch_l,
    __bf16* __restrict__ scratch_acc,
    int64_t q_stride0, int64_t q_stride1,
    int64_t main_cache_stride0, int64_t extra_cache_stride0,
    int main_num_rows, int extra_num_rows,
    int main_block_size, int extra_block_size,
    float scale, int num_heads, int num_head_blocks)
{
    const int query = blockIdx.x;
    const int pid_hs = blockIdx.y;
    const int pid_split = pid_hs / num_head_blocks;
    const int pid_h = pid_hs - pid_split * num_head_blocks;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;

    const int m_a       = lane & 15;
    const int kg        = lane >> 4;
    const int n_b       = lane & 15;
    const int m_d_base  = (lane >> 4) * 4;
    const int n_d       = lane & 15;

    __shared__ __bf16 q_lds[BLOCK_H * HEAD_DIM];
    __shared__ __bf16 k_lds[BLOCK_K * HEAD_DIM];
    __shared__ __bf16 p_lds[BLOCK_H * BLOCK_K];
    __shared__ float  scores_lds[BLOCK_H * BLOCK_K];
    __shared__ int8_t kv_lds[BLOCK_K];
    load_q(q, q_stride0, q_stride1, query, pid_h, num_heads, q_lds, tid);

    float m_state[4], l_state[4];
    fx4   acc[N_TILES_PER_WAVE];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_state[i] = -3.4028234663852886e38f;
        l_state[i] = 0.f;
    }
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WAVE; ++i) {
        acc[i] = (fx4){0.f, 0.f, 0.f, 0.f};
    }

    __syncthreads();

    {
        int main_start = main_indptr[query];
        int main_end   = main_indptr[query + 1];
        int main_len   = main_end - main_start;
        for (int k_start = pid_split * BLOCK_K; k_start < main_len;
             k_start += BLOCK_K * SPLIT_K) {
            gather_and_dequant_k_tile(
                k_start, main_len, main_cache, main_cache_stride0,
                main_num_rows, main_block_size,
                main_indices + main_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    if (HAS_EXTRA) {
        int extra_start = extra_indptr[query];
        int extra_end   = extra_indptr[query + 1];
        int extra_len   = extra_end - extra_start;
        for (int k_start = pid_split * BLOCK_K; k_start < extra_len;
             k_start += BLOCK_K * SPLIT_K) {
            gather_and_dequant_k_tile(
                k_start, extra_len, extra_cache, extra_cache_stride0,
                extra_num_rows, extra_block_size,
                extra_indices + extra_start, k_lds, kv_lds, tid);
            __syncthreads();
            process_k_tile(q_lds, k_lds, kv_lds, p_lds, scores_lds,
                           m_state, l_state, acc, scale,
                           lane, m_a, kg, n_b, m_d_base, n_d, wave);
            __syncthreads();
        }
    }

    // ===== Write scratch =====
    // m_state, l_state are replicated on all 4 waves (all did softmax).
    // Use wave 0 to write m/l (lane.n_d == 0 only).
    const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + pid_split;

    if (wave == 0 && n_d == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = triple * BLOCK_H + m_d_base + i;
            scratch_m[idx] = m_state[i];
            scratch_l[idx] = l_state[i];
        }
    }

    // ACC: each wave has 8 N-tiles. Wave w → cols w*128..w*128+127.
    // Each wave writes its D-layout slice to LDS (k_lds, reused).
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = m_d_base + i;
        #pragma unroll
        for (int nt_local = 0; nt_local < N_TILES_PER_WAVE; ++nt_local) {
            int n_tile = wave * N_TILES_PER_WAVE + nt_local;
            int col = n_tile * 16 + n_d;
            k_lds[row * HEAD_DIM + col] = (__bf16)acc[nt_local][i];
        }
    }
    __syncthreads();

    // All 256 threads cooperate on HBM write (16 KB → 64 bytes/thread = 4 int4).
    {
        // Each thread handles 64 contiguous bf16 cells of [16, 512] flat:
        //   total = 16 * 512 = 8192 bf16, 256 threads * 32 bf16 = 8192. ✓
        //   Use 32 bf16 / thread = 64 bytes = 4 int4.
        // Thread t -> row = t/16, col_start = (t%16)*32.
        int my_row  = tid >> 4;
        int my_col0 = (tid & 15) << 5;
        __bf16* dst = scratch_acc + (int64_t)triple * BLOCK_H * HEAD_DIM
                                  + my_row * HEAD_DIM + my_col0;
        const int4* src4 = reinterpret_cast<const int4*>(
            &k_lds[my_row * HEAD_DIM + my_col0]);
        int4* dst4 = reinterpret_cast<int4*>(dst);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst4[i] = src4[i];
    }
}


// =========================================================================
// Reduce kernel. Combines SPLIT_K partials per (Q, HB) into final output.
//
// Grid: (num_queries, num_head_blocks)
// Block: 64 threads. Each lane handles 4 rows × 8 N-tiles (similar layout).
// =========================================================================
// 256-thread reduce kernel. Each thread handles 32 contiguous cells of
// [16, 512] flat layout (16 rows × 16 col-chunks of 32 = 256 cells).
// Thread t -> row = t/16, col_start = (t%16)*32.
template <bool HAS_ATTN_SINK, int SPLIT_K>
__global__ __launch_bounds__(256, 4)
void v3_mfma_reduce_kernel(
    const float* __restrict__ scratch_m,
    const float* __restrict__ scratch_l,
    const __bf16* __restrict__ scratch_acc,
    const float* __restrict__ attn_sink,
    __bf16* __restrict__ output,
    int64_t out_stride0, int64_t out_stride1,
    int num_heads, int num_head_blocks)
{
    const int query = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int tid   = threadIdx.x;

    const int my_row  = tid >> 4;        // 0..15
    const int my_col0 = (tid & 15) << 5; // 0,32,...,480
    const int head_global = pid_h * BLOCK_H + my_row;

    float m_merged = -3.4028234663852886e38f;
    float l_merged = 0.f;
    float acc_merged[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) acc_merged[i] = 0.f;

    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        const int triple = (query * num_head_blocks + pid_h) * SPLIT_K + s;
        float m_s = scratch_m[triple * BLOCK_H + my_row];
        float l_s = scratch_l[triple * BLOCK_H + my_row];

        float m_new = fmaxf(m_merged, m_s);
        float alpha = __builtin_amdgcn_exp2f(
            (m_merged - m_new) * 1.4426950408889634f);
        float beta  = __builtin_amdgcn_exp2f(
            (m_s      - m_new) * 1.4426950408889634f);
        l_merged = l_merged * alpha + l_s * beta;
        m_merged = m_new;

        // Read 32 bf16 = 64 bytes = 4 int4
        const __bf16* acc_base = scratch_acc
                               + (int64_t)triple * BLOCK_H * HEAD_DIM
                               + my_row * HEAD_DIM + my_col0;
        const int4* src4 = reinterpret_cast<const int4*>(acc_base);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int4 v = src4[i];
            __bf16 vbf[8];
            *reinterpret_cast<int4*>(vbf) = v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a_s = (float)vbf[j];
                acc_merged[i * 8 + j] = acc_merged[i * 8 + j] * alpha
                                      + a_s * beta;
            }
        }
    }

    if (head_global >= num_heads) return;

    float m_final = m_merged;
    float l_final = l_merged;
    float alpha_final = 1.f;
    if (HAS_ATTN_SINK) {
        float sink_val = attn_sink[head_global];
        m_final = fmaxf(m_merged, sink_val);
        alpha_final = __builtin_amdgcn_exp2f(
            (m_merged - m_final) * 1.4426950408889634f);
        l_final = l_merged * alpha_final + __builtin_amdgcn_exp2f(
            (sink_val - m_final) * 1.4426950408889634f);
    }
    float denom = fmaxf(l_final, 1.0e-30f);
    bool live = (l_final > 0.f);
    float inv_denom = live ? (alpha_final / denom) : 0.f;

    __bf16* out_row = output + query * out_stride0
                             + head_global * out_stride1 + my_col0;
    __bf16 out_buf[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) out_buf[i] = (__bf16)(acc_merged[i] * inv_denom);
    int4* dst4 = reinterpret_cast<int4*>(out_row);
    const int4* sb4 = reinterpret_cast<const int4*>(out_buf);
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst4[i] = sb4[i];
}


// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------
void v3_decode_single(
    torch::Tensor q,
    torch::Tensor main_cache,
    torch::Tensor main_indices,
    torch::Tensor main_indptr,
    torch::Tensor extra_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink,
    torch::Tensor output,
    int64_t main_block_size,
    int64_t extra_block_size,
    int64_t main_num_rows,
    int64_t extra_num_rows,
    double scale_d,
    bool has_extra)
{
    const int num_queries = q.size(0);
    const int num_heads = q.size(1);
    const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
    const float scale_f = (float)scale_d;
    const bool has_sink = attn_sink.has_value();

    dim3 grid(num_queries, num_head_blocks);
    dim3 block(256);

    const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
    const uint8_t* mc_ptr = reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
    const uint8_t* ec_ptr = reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
    const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
    const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
    const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
    const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
    __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    const float* sink_ptr = has_sink
        ? attn_sink.value().data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(HAS_S, HAS_E) do { \
        v3_mfma_kernel<HAS_S, HAS_E><<<grid, block, 0, stream>>>( \
            q_ptr, mc_ptr, mi_ptr, mip_ptr, \
            ec_ptr, ei_ptr, eip_ptr, sink_ptr, out_ptr, \
            q.stride(0), q.stride(1), \
            output.stride(0), output.stride(1), \
            main_cache.stride(0), extra_cache.stride(0), \
            main_num_rows, extra_num_rows, \
            main_block_size, extra_block_size, \
            scale_f, num_heads); \
    } while (0)

    if (has_sink && has_extra)  LAUNCH(true, true);
    else if (has_sink)          LAUNCH(true, false);
    else if (has_extra)         LAUNCH(false, true);
    else                        LAUNCH(false, false);

    #undef LAUNCH
}


void v3_decode_split(
    torch::Tensor q,
    torch::Tensor main_cache,
    torch::Tensor main_indices,
    torch::Tensor main_indptr,
    torch::Tensor extra_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_indptr,
    c10::optional<torch::Tensor> attn_sink,
    torch::Tensor output,
    torch::Tensor scratch_m,
    torch::Tensor scratch_l,
    torch::Tensor scratch_acc,
    int64_t main_block_size,
    int64_t extra_block_size,
    int64_t main_num_rows,
    int64_t extra_num_rows,
    double scale_d,
    bool has_extra,
    int64_t split_k)
{
    const int num_queries = q.size(0);
    const int num_heads = q.size(1);
    const int num_head_blocks = (num_heads + BLOCK_H - 1) / BLOCK_H;
    const float scale_f = (float)scale_d;
    const bool has_sink = attn_sink.has_value();

    dim3 grid_p(num_queries, num_head_blocks * (int)split_k);
    dim3 grid_r(num_queries, num_head_blocks);
    dim3 block_p(256);
    dim3 block_r(256);

    const __bf16* q_ptr = reinterpret_cast<const __bf16*>(q.data_ptr());
    const uint8_t* mc_ptr = reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
    const uint8_t* ec_ptr = reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
    const int32_t* mi_ptr = main_indices.data_ptr<int32_t>();
    const int32_t* mip_ptr = main_indptr.data_ptr<int32_t>();
    const int32_t* ei_ptr = extra_indices.data_ptr<int32_t>();
    const int32_t* eip_ptr = extra_indptr.data_ptr<int32_t>();
    __bf16* out_ptr = reinterpret_cast<__bf16*>(output.data_ptr());
    float* sm_ptr = scratch_m.data_ptr<float>();
    float* sl_ptr = scratch_l.data_ptr<float>();
    __bf16* sa_ptr = reinterpret_cast<__bf16*>(scratch_acc.data_ptr());
    const float* sink_ptr = has_sink
        ? attn_sink.value().data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH_P(HAS_E, SK) do { \
        v3_mfma_partial_kernel<HAS_E, SK><<<grid_p, block_p, 0, stream>>>( \
            q_ptr, mc_ptr, mi_ptr, mip_ptr, \
            ec_ptr, ei_ptr, eip_ptr, \
            sm_ptr, sl_ptr, sa_ptr, \
            q.stride(0), q.stride(1), \
            main_cache.stride(0), extra_cache.stride(0), \
            main_num_rows, extra_num_rows, \
            main_block_size, extra_block_size, \
            scale_f, num_heads, num_head_blocks); \
    } while (0)

    #define LAUNCH_R(HAS_S, SK) do { \
        v3_mfma_reduce_kernel<HAS_S, SK><<<grid_r, block_r, 0, stream>>>( \
            sm_ptr, sl_ptr, sa_ptr, sink_ptr, out_ptr, \
            output.stride(0), output.stride(1), \
            num_heads, num_head_blocks); \
    } while (0)

    #define DISPATCH_SK(SK) do { \
        if (has_extra) LAUNCH_P(true, SK); \
        else            LAUNCH_P(false, SK); \
        if (has_sink)  LAUNCH_R(true, SK); \
        else            LAUNCH_R(false, SK); \
    } while (0)

    switch ((int)split_k) {
        case  2: DISPATCH_SK(2);  break;
        case  4: DISPATCH_SK(4);  break;
        case  8: DISPATCH_SK(8);  break;
        case 16: DISPATCH_SK(16); break;
        default: TORCH_CHECK(false, "Unsupported SPLIT_K");
    }
    #undef DISPATCH_SK
    #undef LAUNCH_P
    #undef LAUNCH_R
}


TORCH_LIBRARY_FRAGMENT(vllm_v3_mla, m) {
    m.def("decode_single(Tensor q, Tensor main_cache, Tensor main_indices, "
          "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
          "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
          "int main_block_size, int extra_block_size, int main_num_rows, "
          "int extra_num_rows, float scale, bool has_extra) -> ()");
    m.def("decode_split(Tensor q, Tensor main_cache, Tensor main_indices, "
          "Tensor main_indptr, Tensor extra_cache, Tensor extra_indices, "
          "Tensor extra_indptr, Tensor? attn_sink, Tensor output, "
          "Tensor scratch_m, Tensor scratch_l, Tensor scratch_acc, "
          "int main_block_size, int extra_block_size, int main_num_rows, "
          "int extra_num_rows, float scale, bool has_extra, int split_k) -> ()");
}
TORCH_LIBRARY_IMPL(vllm_v3_mla, CUDA, m) {
    m.impl("decode_single", &v3_decode_single);
    m.impl("decode_split", &v3_decode_split);
}
"""


_module_cache = {}


def _build_ext():
    if "ext" in _module_cache:
        return _module_cache["ext"]
    cache_dir = os.environ.get(
        "VLLM_V3_KERNEL_CACHE_DIR",
        str(pathlib.Path(tempfile.gettempdir()) / "vllm_v3_mla_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    ext = load_inline(
        name="vllm_v3_mla",
        cpp_sources=[""],
        cuda_sources=[_HIP_SRC],
        functions=[],
        extra_cflags=["-O3", "-DNDEBUG", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "--offload-arch=gfx950",
            "-DNDEBUG",
            "-Wno-c++11-narrowing",
            "-Wno-unused-result",
        ],
        with_cuda=True,
        build_directory=cache_dir,
        verbose=False,
    )
    _module_cache["ext"] = ext
    return ext


def _as_int32_contiguous_1d(x):
    if x.dtype != torch.int32:
        x = x.to(torch.int32)
    if not x.is_contiguous():
        x = x.contiguous()
    return x.view(-1)


# Workload-aware SPLIT_K heuristic.
# Goal: enough wgs to saturate ~256 CUs, but each split should still have
# at least MIN_K_PER_SPLIT real tokens or the partial kernel does too
# little useful work.
_NUM_CUS = 256
_MIN_K_PER_SPLIT = int(os.environ.get("V3_MIN_K_PER_SPLIT", "32"))
_SPLIT_K_OVERRIDE = os.environ.get("V3_SPLIT_K")


def _pick_split_k(num_queries, num_head_blocks, max_total_k):
    if _SPLIT_K_OVERRIDE is not None:
        return int(_SPLIT_K_OVERRIDE)
    base_tiles = max(1, num_queries * num_head_blocks)
    cu_target = max(1, _NUM_CUS // base_tiles)
    k_limit = max(1, max_total_k // _MIN_K_PER_SPLIT)
    target = max(1, min(cu_target, k_limit))
    # Cap at 8 — SPLIT_K=16 makes the reduce kernel work scale linearly with
    # SPLIT_K (16 partial reads per row) and slows down more than the partial
    # gains. Empirically SPLIT_K=8 beats SPLIT_K=16 for all configs tested.
    best = 1
    for b in (1, 2, 4, 8):
        if b <= target:
            best = b
    return best


def _decode_v3(
    q, main_cache, main_indices, main_indptr,
    scale, attn_sink, nope_head_dim, rope_head_dim,
    extra_cache, extra_indices, extra_indptr,
    max_main_len, max_extra_len,
):
    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr  = _as_int32_contiguous_1d(main_indptr)
    num_queries, num_heads, _ = q.shape

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr  = _as_int32_contiguous_1d(extra_indptr)
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr  = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    out = torch.empty_like(q, dtype=torch.bfloat16)
    sink = attn_sink.contiguous() if attn_sink is not None else None
    q_in = q.contiguous() if not q.is_contiguous() else q

    BLOCK_H = 16
    num_head_blocks = (num_heads + BLOCK_H - 1) // BLOCK_H
    total_max_k = max_main_len + (max_extra_len if has_extra else 0)
    split_k = _pick_split_k(num_queries, num_head_blocks, total_max_k)

    ext = _build_ext()

    if split_k == 1:
        torch.ops.vllm_v3_mla.decode_single(
            q_in, main_cache, main_indices, main_indptr,
            extra_cache, extra_indices, extra_indptr,
            sink, out,
            int(main_cache.shape[1]),
            int(extra_cache.shape[1]),
            int(main_cache.shape[0] * main_cache.shape[1]),
            int(extra_cache.shape[0] * extra_cache.shape[1]),
            float(scale),
            bool(has_extra),
        )
    else:
        # Allocate scratch.
        scratch_m = torch.empty(
            num_queries * num_head_blocks * split_k * BLOCK_H,
            device=q.device, dtype=torch.float32,
        )
        scratch_l = torch.empty_like(scratch_m)
        scratch_acc = torch.empty(
            num_queries * num_head_blocks * split_k * BLOCK_H * 512,
            device=q.device, dtype=torch.bfloat16,
        )
        torch.ops.vllm_v3_mla.decode_split(
            q_in, main_cache, main_indices, main_indptr,
            extra_cache, extra_indices, extra_indptr,
            sink, out,
            scratch_m, scratch_l, scratch_acc,
            int(main_cache.shape[1]),
            int(extra_cache.shape[1]),
            int(main_cache.shape[0] * main_cache.shape[1]),
            int(extra_cache.shape[0] * extra_cache.shape[1]),
            float(scale),
            bool(has_extra),
            int(split_k),
        )
    return out


def rocm_sparse_attn_decode_v3(
    q, kv_cache, swa_k_cache, swa_only,
    topk_indices, topk_lens,
    swa_indices, swa_lens,
    swa_ragged_indices, swa_ragged_indptr,
    topk_ragged_indices, topk_ragged_indptr,
    attn_sink, scale, head_dim, nope_head_dim, rope_head_dim,
    output,
):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
        _validate_dsv4_sparse_dims,
    )
    assert swa_k_cache.dtype == torch.uint8
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "rocm_sparse_attn_decode_v3",
    )
    assert nope_head_dim == 448
    assert rope_head_dim == 64

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
        main_ragged_indptr  = swa_ragged_indptr

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
            extra_ragged_indptr  = topk_ragged_indptr

    # Determine max K lengths (for SPLIT_K heuristic).
    # .item() syncs, illegal during HIP graph capture, so fall back to
    # SWA window-size upper bound in that case.
    if torch.cuda.is_current_stream_capturing():
        max_main_len = swa_indices.shape[-1]
        max_extra_len = max_main_len if has_extra else 0
    else:
        max_main_len = int(swa_lens.max().item()) if swa_lens is not None else 0
        max_extra_len = 0
        if has_extra and topk_lens is not None:
            max_extra_len = int(topk_lens.max().item())

    attn_out = _decode_v3(
        q=q, main_cache=swa_k_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        max_main_len=max_main_len,
        max_extra_len=max_extra_len,
    )
    output.copy_(attn_out.to(output.dtype))
