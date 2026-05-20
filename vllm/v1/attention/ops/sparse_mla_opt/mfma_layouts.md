# MFMA Register Layouts on gfx950 (MI350, wave64)

Reference for `__builtin_amdgcn_mfma_f32_16x16x32_bf16` — the MFMA used by
the sparse-MLA decode kernel. **Verified empirically** by
`test_mfma_layouts.py` (4/4 PASS, max relative error ~6e-8).

## What the MFMA computes

```
D[16, 16] (fp32)  =  A[16, 32] (bf16)  *  B[32, 16] (bf16)  +  C[16, 16] (fp32)
```

The instruction is data-parallel across all 64 lanes of a wave. The
inputs and outputs are spread across the lanes' VGPRs in a fixed layout
defined by the hardware — you don't choose it. You only choose how to
get your matrix data **into** that layout from LDS/HBM, and back out.

## The layouts

In the table below: `l` is the lane index (0..63), `i` is the per-lane
register slot.

| Operand | Type    | Per-lane storage   | Row index    | Col index    |
|---------|---------|--------------------|--------------|--------------|
| **A**   | bf16    | 8 elements / lane  | `m = l % 16` | `k = (l/16)*8 + i`,  `i = 0..7` |
| **B**   | bf16    | 8 elements / lane  | `k = (l/16)*8 + i`,  `i = 0..7` | `n = l % 16` |
| **C, D**| fp32    | 4 elements / lane  | `m = (l/16)*4 + i`,  `i = 0..3` | `n = l % 16` |

**Key gotcha** — A's lane-to-row mapping (`m = l % 16`) is *not* the same
as D's (`m = (l/16)*4 + i`). D is "transposed" relative to how A is laid
out. Both inputs share the lane.lo4 → col-N (for B) / col-K-group
(for A) pattern, but D's lane.lo4 maps to col-N and its lane.hi2 *plus*
reg slot together cover all 16 M-rows.

If you feed a freshly-computed D back as the A of the next MFMA, you
**must** shuffle through LDS — the layouts do not naturally chain.

### Intrinsic call

```cpp
using bf16x8 = __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16;
using fx4    = __attribute__((__vector_size__(4 * sizeof(float))))  float;

__device__ __forceinline__ fx4 mfma_16x16x32_bf16(
    bf16x8 a, bf16x8 b, fx4 c) {
    // cbsz=0, abid=0, blgp=0 → single-block, no broadcast
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}
```

## How to load each operand from LDS

Assuming a row-major LDS tile:

```cpp
// A[16, 32]: lds_a[m * 32 + k]
const int lane = threadIdx.x;
const int m    = lane & 15;
const int kg   = lane >> 4;          // 0..3
bf16x8 a;
#pragma unroll
for (int i = 0; i < 8; ++i) a[i] = lds_a[m * 32 + kg * 8 + i];

// B[32, 16]: lds_b[k * 16 + n]
const int n  = lane & 15;
bf16x8 b;
#pragma unroll
for (int i = 0; i < 8; ++i) b[i] = lds_b[(kg * 8 + i) * 16 + n];

// Run MFMA
fx4 c = {0.f, 0.f, 0.f, 0.f};
fx4 d = mfma_16x16x32_bf16(a, b, c);

// Store D[16, 16]: lds_d[m * 16 + n]
const int mg = lane >> 4;            // 0..3
#pragma unroll
for (int i = 0; i < 4; ++i) lds_d[(mg * 4 + i) * 16 + n] = d[i];
```

## Per-row reductions in the D layout

For row-wise ops (softmax max/sum) you need to reduce across the lanes
that share the same `m`. In the D layout each lane owns 4 *different*
rows at the same N col, so the reduce is across all 16 lanes that share
a `lane.hi2` group (the 16 lanes that collectively cover the 16 N cols
of the same M row).

```cpp
// Reduce one row of the D-layout matrix.
// Caller: for each i in 0..3, the lane holds D[(l/16)*4 + i, l%16].
// Reduce across lane.lo4 (16 lanes), independently for each i.
float row_max = my_val;
#pragma unroll
for (int off = 8; off > 0; off >>= 1) {
    row_max = fmaxf(row_max, __shfl_xor(row_max, off));
}
```

If your row is wider than 16 cols (e.g. two MFMAs concatenated along N
to make a 32-wide row), reduce across both lane-groups before the
butterfly: `my_val = fmaxf(my_val_ntile0, my_val_ntile1)` first.

## C → A layout swap (the chain that broke v3.1)

To feed an MFMA output into the next MFMA's A operand (FlashAttention's
QK^T → softmax → PV pattern), an LDS round-trip is mandatory. The two
layouts don't naturally line up.

```cpp
__shared__ __bf16 swap_lds[16 * BLOCK_K];  // BLOCK_K = 32 for sparse-MLA

// Step 1: write D-layout output to LDS (using D's row/col mapping).
// Lane l writes its 4 elements to rows (l/16)*4 + i, col l%16.
const int lane = threadIdx.x;
const int mg   = lane >> 4;
const int n    = lane & 15;
// For BLOCK_K=32 (two N-tiles): nt=0 fills cols 0..15, nt=1 fills 16..31.
#pragma unroll
for (int i = 0; i < 4; ++i) {
    swap_lds[(mg * 4 + i) * BLOCK_K + n]              = (__bf16)d_nt0[i];
    swap_lds[(mg * 4 + i) * BLOCK_K + 16 + n]         = (__bf16)d_nt1[i];
}
__syncthreads();

// Step 2: read back in A-layout.
// Lane l reads A[m=l%16, k=(l/16)*8 + i] for i=0..7.
const int m  = lane & 15;
const int kg = lane >> 4;
bf16x8 a_next;
#pragma unroll
for (int i = 0; i < 8; ++i) {
    a_next[i] = swap_lds[m * BLOCK_K + kg * 8 + i];
}
```

The swap costs one workgroup-wide barrier + ~1024 bytes of LDS bandwidth
(for 16×32 bf16), which is trivial compared to MFMA throughput.

## The unit test (`test_mfma_layouts.py`)

Lives at `vllm/v1/attention/ops/sparse_mla_opt/test_mfma_layouts.py`.
Runs four progressively-harder MFMA patterns, each compared against a
PyTorch fp32 reference. Run it as:

```bash
PYTORCH_ROCM_ARCH=gfx950 python test_mfma_layouts.py
```

JIT-compiles on first run (~10 s; cached in `$TMPDIR/mfma_test_cache/`).

### The four tests

| # | What it verifies                                                                                   | Why it's there |
|---|----------------------------------------------------------------------------------------------------|----------------|
| 1 | `D[16,16] = A[16,32] @ B[32,16]` on random bf16 inputs.                                            | Smallest correctness gate. Validates A, B, and D per-lane layouts in isolation. If this fails, nothing else can work. |
| 2 | `D[16,16] = A[16,64] @ B[64,16]` — two MFMAs chained on the K reduction dim.                       | Verifies that `c = mfma(a, b, c)` correctly accumulates into the same fp32 register file. This is the pattern that covers `HEAD_DIM=512` with 16 K=32 MFMAs. |
| 3 | `(Q @ K^T)[16,32] @ V[32,16]` — *no softmax*, just two MFMAs separated by an LDS round-trip.        | Isolates the C→A layout swap. Removes softmax so any error has to be in the layout shuffle, not in the math. |
| 4 | Full `Q[16,512] @ K^T[32,512] * scale → softmax → @ V[32,16]` for one K-tile.                       | The exact compute pattern of the sparse-MLA inner loop, with row-wise softmax in the D layout. Adds the per-row max/sum reduction on top of test 3. |

### How the debug kernel works

A separate `k_test_debug` kernel runs three candidate A/B/D layouts against
a known-result input (`A = I`, `B[k,n] = (k+1)*100/1000 + n/1000`). Since
`I @ B = B`, the actual `D[m, n]` *must* equal `(m+1)*100/1000 + n/1000`.
By comparing what shows up at each `(m, n)` slot for each candidate
layout, you can decode which layout the hardware is actually using.

This is how I caught the bug: my original v3.1 stored MFMA's D as if the
layout were `D[l%16, (l/16)*4 + i]`. The debug kernel showed `D[0, n]`
was producing the *first column* of `B` (so `0.1, 0.2, 0.3, 0.4` for
`n=0..3`) instead of the first row. That pattern uniquely identifies
"reg slot is the M row, not the N col" → fixed layout to
`D[(l/16)*4 + i, l%16]`. All four tests then passed at machine precision.

### Pass/fail tolerance

Tolerance is `1e-1` absolute for the small tests (K up to 64) and
relative error reported against `ref.abs().max()`. The actual measured
errors are ~1e-6 to 1e-9 — well into machine precision, far below what
bf16 accumulation noise would produce. **A layout bug would not produce
small errors; it produces errors on the order of the reference magnitude
itself** (because the kernel is reading/writing the wrong elements
entirely). So a "borderline" pass with err ~tol/10 should be treated as
suspicious — re-run with the debug kernel.

## What this enables for v3

With the layouts verified, the path to an MFMA-based v3 is:

1. **Inner loop:** replace v3.2's per-lane vector-dot QK with the
   chained-MFMA pattern from Test 4. Q stays in LDS; K is pre-dequanted
   to bf16 in LDS (as v3.2 already does); the QK and PV MFMAs use the
   verified layouts.
2. **Online softmax:** uses the 16-way row reduction shown above (one
   reduction per owned row, independent across the 4 reg slots).
3. **Final write:** D-layout (`out[(l/16)*4 + i, l%16]`) writes
   directly to the bf16 output tensor.

The skeleton (cooperative LDS gather, K pre-dequant, single-kernel)
already lives in `rocm_aiter_mla_sparse_v3.py`; swap the QK/PV
arithmetic for the MFMA pattern verified here and the kernel should
become competitive with v2.
