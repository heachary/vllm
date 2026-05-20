# ROCm Sparse-MLA Decode Optimization Report (v3 — HIP MFMA)

Target: hand-written HIP kernel for `rocm_sparse_attn_decode` at B=4,
stretch goal 10× speedup over v2.

## Result

**v3 beats v2 on every config**, by 1.6× to 2.3×. The 10× stretch
target was not met; thread-trace analysis (see below) shows the
remaining gap is gather latency on data-dependent random gathers
(the same wall v2 hit), now combined with the inability to use
more than ~32 CUs on the B=4 grid without paying more in launch
overhead than we save in per-CU compute.

| B | mode | topk_ragged | v2 (µs) | v3 (µs) | v3 vs v2 |
|---|------|-------------|---------|---------|----------|
| 4 | csa  | 1024        | 56.4    | 30.4    | **1.86×** |
| 4 | csa  | 1280        | 71.2    | 35.7    | **2.00×** |
| 4 | csa  | 1536        | 69.8    | 35.8    | **1.95×** |
| 4 | csa  | 1792        | 69.8    | 35.8    | **1.95×** |
| 4 | csa  | 2048        | 70.1    | 35.8    | **1.96×** |
| 4 | hca  | 32          | 48.8    | 21.4    | **2.28×** |
| 4 | hca  | 40          | 48.8    | 21.1    | **2.31×** |
| 4 | hca  | 48          | 48.9    | 21.9    | **2.23×** |
| 4 | hca  | 56          | 49.0    | 21.8    | **2.25×** |
| 4 | hca  | 64          | 49.0    | 21.3    | **2.30×** |

Correctness verified end-to-end against the baseline: max relative
error ~0.7 % on the `verify_v1.py` bounded-input fixture.

## What v3 does

`rocm_aiter_mla_sparse_v3.py` — three HIP kernels behind a heuristic
SPLIT_K dispatcher:

1. **`v3_mfma_kernel`** — single-kernel (SPLIT_K=1) fast path. Used
   only when work is small enough that the partial+reduce overhead
   would outweigh the parallelism gain.

2. **`v3_mfma_partial_kernel<HAS_EXTRA, SPLIT_K>`** — per-(Q, HB, split)
   tile. Writes m_state, l_state, acc to scratch.

3. **`v3_mfma_reduce_kernel<HAS_ATTN_SINK, SPLIT_K>`** — combines
   SPLIT_K partials per (Q, HB) using FlashDecoding+ merge, applies
   attn_sink, normalises, writes bf16 output.

### Key design choices (in order of impact)

1. **MFMA inner loop using verified gfx950 layouts.** Both QK^T and
   PV use `v_mfma_f32_16x16x32_bf16`. Layouts are documented in
   `mfma_layouts.md` and unit-tested by `test_mfma_layouts.py` (4/4
   pass at machine precision). The C → A layout swap between QK and
   PV is mediated through `p_lds` (1 KB LDS round-trip).

2. **K-tile pre-dequant into LDS.** Each K-iter, the 32 raw fp8 K
   tokens are gathered into LDS and dequanted to bf16 *once*
   (cooperative across 256 threads, 8 threads per token chunk). The
   MFMA inner loop then reads pure bf16 from LDS — no per-element
   fp8 conversion or e8m0-scale lookup inside the hot loop. This was
   the single largest stall site in v2's thread trace (30% of all
   stalls). bf16 K-LDS is 32 KB per workgroup.

3. **4 waves per WG with role split: wave 0 does QK + softmax,
   all 4 waves do PV.** Each wave owns 8 of the 32 PV N-tiles (cols
   `wave*128 .. wave*128+127` of the output head_dim).  This (a)
   parallelises PV 4× across waves and (b) cuts per-wave VGPR
   footprint from 128 fp32 → 32 fp32 for acc, letting the scheduler
   keep 2 wgs resident per CU instead of 1.

4. **All waves redo the softmax independently** from `scores_lds`
   (fp32, 2 KB) written by wave 0. This gives every wave consistent
   per-row `m_state`, `l_state`, `alpha` without inter-wave
   communication beyond the scores LDS write.

5. **SPLIT_K heuristic capped at 8.** `_pick_split_k` picks
   `min(NUM_CUS / num_tiles, max_K / MIN_K_PER_SPLIT, 8)`. The
   cap at 8 matters: SPLIT_K=16 makes the reduce kernel work scale
   linearly with SPLIT_K (16 partial reads × 16 KB each per output
   wg), which costs more than the extra partial-kernel parallelism
   buys back. Empirically SPLIT_K=4 wins for HCA and small CSA;
   SPLIT_K=8 wins for the largest CSA topk.

6. **Coalesced HBM scratch I/O via LDS-mediated layout swap.** The
   partial writes acc to `k_lds` (reused as scratch) in flat
   `[BLOCK_H, HEAD_DIM]` layout, then all 256 threads cooperate on
   the HBM write as 4 int4 each — fully coalesced. The reduce reads
   the partial scratch the same way (256 threads × 32 bf16 each).
   Without this, the natural per-lane D-layout gives 8-way strided
   writes/reads and v3 was 6-10× slower (see iteration history below).

## Thread-trace-driven iterations

Each step was driven by `rocprofv3 --kernel-trace`, measuring the
median per-kernel duration over 20 reps.

| version                    | partial (µs) | reduce (µs) | total HCA (µs) | total CSA-2048 (µs) | what changed |
|----------------------------|--------------|-------------|----------------|---------------------|--------------|
| v3.3 single-kernel (SK=1)   | n/a          | n/a         | 60             | 122-195             | first correct MFMA path; baseline |
| v3.4 + SPLIT_K=4 (1-wave)  | 29           | 163         | 134            | 280-361             | reduce kernel **slower** than partial — strided HBM reads from D-layout scratch |
| v3.5 + coalesced reduce    | 46           | 23          | 118-160        | 290-400             | fixed reduce via LDS-mediated layout swap (16-byte int4 loads) |
| v3.6 + 4 waves per WG      | 18           | 6           | 25             | 100-189             | gather 4×-cooperative, MFMA only on wave 0 |
| v3.7 + 256-thread reduce   | 18           | 5.5 / 14    | 25             | 33-46               | reduce wg → 256 threads; CSA reduce 12× faster |
| **v3.8 (final): distributed PV** | 15.7         | 5.5 / 14    | **21**         | **30-36**           | all 4 waves do PV (8 N-tiles each), acc footprint 128→32 fp32/lane |

### Bottleneck at each step

- **v3.4 → v3.5**: reduce was reading scratch in a strided per-lane
  pattern (lane reads 4 rows × 32 N-tiles × 1 col each, all at non-
  consecutive HBM addresses). Restructured both partial *write* and
  reduce *read* to use a flat `[BLOCK_H, HEAD_DIM]` layout via an
  LDS round-trip, so each thread reads/writes 16 contiguous bf16
  elements (1 int4) at a time. Reduce time dropped 163 → 23 µs.

- **v3.5 → v3.6**: partial was bottlenecked by single-wave (64
  threads) cooperative gather; HBM round-trips weren't pipelined
  enough. Bumped to 256 threads (4 waves) per WG. Wave 0 still does
  the MFMA work; other waves participate in gather and the LDS-
  mediated scratch write. Partial 29 → 18 µs, full HCA 134 → 25 µs.

- **v3.6 → v3.7**: reduce was still 1-wave (64 threads), couldn't
  keep enough HBM loads in flight for CSA's 16 partials × 16 KB
  reads. Bumped reduce wg to 256 threads. Reduce time for CSA
  dropped from 169 → 14 µs.

- **v3.7 → v3.8**: only wave 0 was doing PV (32 MFMAs serially) and
  also holding the full `acc[32]` accumulator (128 fp32/lane).
  Distributed PV across all 4 waves: each wave does 8 PV N-tiles
  and holds `acc[8]` (32 fp32/lane). This cut wave 0's PV serial
  time 4× and made the acc footprint small enough to keep 2 wgs
  resident per CU. Partial 18 → 15.7 µs.

## How v3 (HIP) gets past v2 (Triton)'s memory-latency wall

v2's thread-trace analysis identified four concrete stall sites that
together accounted for ~73 % of stall cycles. Triton has limited
direct control over LDS layout, VGPR allocation, or workgroup-level
work distribution, so several of these stalls are structural for any
Triton implementation at our shape. v3's HIP rewrite addresses each
one explicitly. The accounting below maps v2's identified stalls to
the v3 design change that removes it.

### 1. v2 stall: dequant-after-K-load (30 % of all stalls)

> "The first use of `x_uint8` and `encoded_scales` immediately after their
> loads forces an `s_waitcnt vmcnt(0)` on both vmem fetches. The dequant ALU
> is fast — the wait is for HBM data to land." (v2 thread trace)

In v2's inner loop, every fp8 K element triggers a per-element dequant
chain (`load_byte → exp2(scale - 127) → cvt → mul`) right inside the
MFMA's K accumulation. The `vmcnt(0)` wait blocks the entire wave on
HBM data for that one element, *for every MFMA tile*.

**v3 fix — pre-dequant K to bf16 in LDS, once per K-tile.** The
cooperative gather (256 threads, 8 threads per K token) reads the raw
fp8 bytes and e8m0 scales from HBM into LDS, dequants once, and writes
bf16 K back to LDS in `[BLOCK_K, HEAD_DIM]` flat layout. The MFMA
inner loop then reads pure bf16 from LDS — no per-element fp8 conversion
and no per-element scale lookup. The HBM-vmcnt wait happens once per
K-tile (at the post-gather `__syncthreads`), not once per MFMA tile, so
the `vmcnt(0)` stall amortises across 16 K-chunks of QK + 32 N-tiles
of PV per K-iter. Triton can't easily express "dequant once, MFMA
N+M times against the result" — its `tl.dot` schedules the dequant
inside the dot's K-reduction loop.

### 2. v2 wall: 256 VGPRs/wave forces ≤2 wgs/CU (occupancy starvation)

> "The dominating consumer is the `[BLOCK_H, NOPE_BLOCK] = [16, 512]`
> fp32 accumulator (32 VGPRs/thread on its own). … only **one workgroup
> fits per CU**, so when a wave stalls on K-load latency there is no
> other wave on that CU to swap in — classic occupancy-starvation."

Triton holds the full `acc[BLOCK_H, HEAD_DIM]` accumulator in one
wave's VGPR file. With 4 waves × 256 VGPRs each, only 1 wg fits per CU.
At the 32-wg grid for B=4, most CUs are idle and the resident waves
have nobody to swap to on every HBM stall.

**v3 fix — distribute PV across all 4 waves.** Wave 0 still does QK and
softmax (which needs a single coherent set of m/l state), but PV is
split: each wave owns 8 of the 32 PV N-tiles (head_dim cols
`wave*128 .. wave*128+127`). Per-wave VGPR for acc is `8 × 4 = 32 fp32`
instead of `32 × 4 = 128`. With those VGPRs freed, the kernel fits
**2 wgs/CU resident** (8 waves per CU for latency hiding) instead of 1.

Triton's compiler does its own MFMA layout selection and won't make this
decision for you — its `tl.dot` keeps the full output in one wave's
register file.

### 3. v2 wall: per-wave K gather can't pipeline enough HBM loads

In v2 each wave's threads issue their own K-byte loads (`tl.load` with
masked broadcasts). With ≤2 wgs/CU and `num_warps=4`, the per-CU
HBM-load pipeline depth tops out at ~32 outstanding loads. HBM
round-trips of ~500-1000 cycles dominate.

**v3 fix — cooperative LDS gather, 256 threads per WG, 1 thread per
(token, chunk) pair.** 256 threads × ~16 outstanding HBM loads each =
~4 K outstanding HBM transactions per WG. With 2 wgs/CU resident
that's ~8 K outstanding per CU — enough to saturate HBM bandwidth and
mostly hide the round-trip latency. The trace shows per-K-iter gather
time dropped from v2's effective ~5-7 µs/iter to ~1-2 µs/iter in v3.

### 4. v2 wall: reduce kernel uses fp32 scratch + strided HBM reads

v2's reduce reads `acc_nope[Q, HB, SPLIT_K, BLOCK_H, NOPE_DIM]` from
scratch (originally fp32, later bf16 in v2). The Triton-generated
loads were strided — each lane reading 4 non-consecutive rows × 32
non-consecutive cols per partial.

**v3 fix — coalesced HBM I/O via LDS-mediated layout swap.** The
partial kernel keeps acc in MFMA's per-lane "D-layout" during the K
loop (`lane l holds D[(l/16)*4 + i, l%16]`), then at the end:
write acc to `k_lds` (reused) in flat `[BLOCK_H, HEAD_DIM]` layout,
sync, then all 256 threads cooperate on the HBM write as 4 int4
each (fully coalesced). The reduce kernel mirrors this — 256 threads
each read 32 contiguous bf16 cells (= 4 int4) per partial. v2's reduce
was ~19 µs (fp32 scratch) or ~6 µs (bf16); v3's reduce is 5-14 µs but
on a smaller scratch footprint and with better SPLIT_K choices.

### 5. v2 wall: SPLIT_K=16 chosen by Triton's heuristic over-splits

v2's `_pick_split_k` doesn't account for the reduce kernel's per-
SPLIT_K work growth, so for B=4 it picks SPLIT_K=16, leaving each
split with too few real K tokens (HCA: 144/16 ≈ 9 tokens per split,
mostly launch + Q-load overhead).

**v3 fix — explicit cap at SPLIT_K=8.** Measured per-config: SPLIT_K=4
wins for HCA and small CSA; SPLIT_K=8 wins for the largest CSA topk.
Beyond 8, the reduce kernel becomes the bottleneck because per output
row it reads `SPLIT_K × 1 KB` from scratch.

### Summary of HIP-specific levers

| v2 stall site                            | What HIP lets us do                                       | v3 measured impact                  |
|------------------------------------------|-----------------------------------------------------------|-------------------------------------|
| Dequant after K load (30 % of stalls)    | Pre-dequant once per K-tile into a bf16 LDS staging area  | Inner MFMA loop reads pure bf16; vmcnt wait amortises across 48 MFMAs per iter |
| 256 VGPR/wave → 1 wg/CU                  | Distribute PV across 4 waves; per-wave acc 32 fp32/lane   | 2 wgs/CU resident; 8 waves/CU for latency hiding |
| Per-wave gather pipeline depth           | 256-thread cooperative gather (8 threads per K token)     | ~4 K outstanding HBM loads per WG → gather time per K-iter 5-7 µs → 1-2 µs |
| Strided HBM reads in reduce              | LDS-mediated layout swap, then coalesced int4 I/O         | Reduce HBM access pattern matches a contiguous tile load → reduce 7× faster than naive |
| Triton's SPLIT_K=16 heuristic            | Direct, measured cap at SPLIT_K=8                         | Avoids the reduce-kernel SPLIT_K-linear blowup |

### What HIP can't fix (the remaining floor)

The structural ceilings below survive the HIP rewrite because they're
properties of the workload, not the kernel:

- **Random K-token gather is data-dependent.** Each of the 32 K-token
  addresses in a tile is determined by `swa_indices[q_idx, k_pos]`
  and cannot be hoisted ahead of time. Each gather is its own HBM
  round-trip. We can deepen the pipeline (HIP does) but we can't
  reduce the *count* of round-trips.
- **B=4 doesn't generate enough independent work for 256 CUs.** Even
  with SPLIT_K=8 we have 128 wgs; at 2 wgs/CU LDS limit that's
  64 CUs of the 256 available. The remaining 192 CUs are idle no
  matter what we do at this batch size.

## Why the 10× target wasn't met

Two structural ceilings, both visible in the thread trace:

1. **Grid size cap at B=4.** Even with SPLIT_K=8 we have only
   `4 (Q) × 4 (HB) × 8 = 128` partial wgs, and at our LDS budget
   (~51 KB per WG: 16 KB Q + 32 KB K + 2 KB scores + 1 KB P) we get
   at most 2-3 wgs/CU resident. That puts a hard ceiling of 32-43
   CUs of the 256 available being able to do useful work
   concurrently. Bumping SPLIT_K higher (16) hurts because the
   reduce kernel work grows linearly with SPLIT_K and overruns the
   extra partial parallelism.

2. **K-token gather latency.** Each K iteration's 32 random gathers
   are independent HBM transactions (~250-500 ns each), and the
   addresses are data-dependent so cannot be hoisted. The 32 lanes
   doing the gather cooperatively can pipeline ~16 outstanding
   loads each, but the round-trip floor is still ~1 µs per K-iter
   in the steady state — almost exactly what we measure at 2-3
   µs/iter today.

The realistic theoretical floor for this workload at B=4 (per
opt_report_v2's analysis) is around 0.1 µs of bandwidth + a few
hundred ns of latency per CU — but `B=4` simply doesn't generate
enough independent work units to push the whole GPU there.
Achieving 10× would need either (a) a persistent kernel + atomic
reduce that eliminates the partial-write + reduce-read traffic
entirely (saves the ~5 µs scratch round trip but requires inter-wg
sync), (b) a re-architected cache layout that allows coalesced
multi-token gathers, or (c) larger batch sizes that put more wgs
on more CUs.

## Files

- `rocm_aiter_mla_sparse_v3.py` — three HIP kernels + Python wrapper,
  JIT-compiled via `torch.utils.cpp_extension`.
- `mfma_layouts.md` — verified MFMA per-lane register layouts.
- `test_mfma_layouts.py` — 4-test ladder validating the layouts
  against a torch reference (run as a unit test).
- Toggle the implementation with `VLLM_ENABLE_SPARSE_MLA_OPT_V3=1`.
- Override the split-K choice with `V3_SPLIT_K=<n>` for sweeps.
- Bench with `python test_sparse_mla.py --impl v3 --warmup 25 --rep 100`.
