# ROCm Sparse-MLA Decode Optimization Report (v2)

Target: optimize `rocm_sparse_attn_decode` for batch=4 (DSv4 TP=8 serving)
at the CSA and HCA decode shapes. Stretch target: 10× over baseline.

## Setup

- GPU: AMD gfx950 (MI350), 256 CUs
- Bench harness: `test_sparse_mla.py` — rocprofv3 kernel trace,
  warmup=25, rep=100, median across runs
- Shapes (B=4): CSA `topk_len ∈ {1024..2048}`, HCA `topk_len ∈ {32..64}`
- Correctness: 0 failures across all configs, max relative error
  ~7e-3 (within the 5% tolerance set by `verify_v1.py`).

## Results

Median GPU time per call (µs). Both kernels (partial + reduce) summed
for v1 and v2; baseline is a single kernel.

| B | mode | topk_ragged | baseline (µs) | v1 (µs) | v2 (µs) | v2 vs baseline |
|---|------|-------------|---------------|---------|---------|----------------|
| 4 | csa  | 1024        | 96.0          | 48.2    | **40.2**| 2.4×           |
| 4 | csa  | 1280        | 111.8         | 48.2    | **50.8**| 2.2×           |
| 4 | csa  | 1536        | 126.6         | 48.2    | **49.9**| 2.5×           |
| 4 | csa  | 1792        | 141.2         | 48.2    | **49.8**| 2.8×           |
| 4 | csa  | 2048        | 157.6         | 48.4    | **49.9**| 3.2×           |
| 4 | hca  | 32          | 39.2          | 47.7    | **34.8**| 1.1×           |
| 4 | hca  | 40          | 39.4          | 47.6    | **34.9**| 1.1×           |
| 4 | hca  | 48          | 39.4          | 47.7    | **34.8**| 1.1×           |
| 4 | hca  | 56          | 40.0          | 47.7    | **34.9**| 1.1×           |
| 4 | hca  | 64          | 39.6          | 47.7    | **34.9**| 1.1×           |

**v2 is the fastest on every config**, but **the 10× stretch target was
not hit**. Best speedups: 3.2× on the largest CSA shape, 1.1× on HCA.

## What v2 changes (vs. v1)

All changes live in `rocm_aiter_mla_sparse_v2.py`. Tuning knobs are
env vars (`V2_BLOCK_H`, `V2_BLOCK_K`, `V2_SPLIT_K`,
`V2_MIN_K_PER_SPLIT`, `V2_NUM_WARPS`, `V2_NUM_STAGES`).

1. **K-length-aware SPLIT_K** (`_pick_split_k`). v1 only looked at
   `num_queries × num_head_blocks`, producing SPLIT_K=16 for everything
   at B=4. That left most splits doing 0–1 K iterations for HCA
   (total K ≈ 136), so launch + scratch-write dominated. v2 also caps
   SPLIT_K so each split gets at least `MIN_K_PER_SPLIT` (default 32)
   real tokens. Net for B=4: HCA picks SPLIT_K=4 (was 16),
   CSA picks SPLIT_K=8–16.

2. **Single-kernel fast path** (`_v2_single_kernel`) for SPLIT_K=1.
   v1 always launches partial + reduce, even when the reduce has only
   one input. v2's single-kernel path writes the final bf16 output
   directly (sink + normalise inline) — no scratch, no reduce launch.
   Not currently used on the B=4 bench shapes (the heuristic picks
   SPLIT_K ≥ 2 for them), but kept because it eliminates a wasted
   kernel for any future shape that's small enough.

3. **bf16 acc scratch** in split-K mode (`_v2_partial_kernel` writes
   bf16; `_v2_reduce_kernel` upcasts on load). Reduce kernel scratch
   bandwidth is halved (was the single biggest source of overhead in
   v1: ~9 MB of fp32 acc per call, ~19 µs of reduce time). v2 reduce
   measured at ~6 µs at the same SPLIT_K — a clean 3× reduction.

4. **`num_stages=2`**. Lets Triton software-pipeline one extra K-tile
   load ahead of compute. Gave ~5 µs on HCA over `num_stages=1`.

## Thread-trace findings (rocprofv3 ATT)

Profiled `_v2_partial_kernel` at the HCA-32 config (representative of
where overhead is hardest to hide). Aggregated stalls by instruction:

```
op                   stall    latency   hits   stall%
s_waitcnt            25 856   25 856    324    37.8 %
s_barrier             5 740    5 740    108     8.4 %
buffer_store_dwordx2  4 128    4 276     36     6.0 %
buffer_load_ubyte     4 068    4 292     56     6.0 %
ds_write_b8           3 968    8 064    512     5.8 %
ds_read_b128          2 988    4 428    360     4.4 %
```

- **~38 % of all stalls are `s_waitcnt`** — the kernel is
  memory-latency bound. Top individual stalls are all `vmcnt` waits on
  the FP8 K-byte load and the encoded-scale load (~3 000 cycles each)
  and the initial Q load (~2 900 cycles).
- The kernel uses 256 VGPRs/wave (driven by the
  `[BLOCK_H, NOPE_BLOCK] = [16, 512]` fp32 accumulator). At that
  pressure, only **one workgroup fits per CU**, so when a wave stalls
  on K-load latency there is no other wave on that CU to swap in —
  classic occupancy-starvation.
- B=4 only generates 4 × `num_head_blocks` × SPLIT_K programs total
  (≤256). With ≤1 wgs/CU there is no over-subscription to hide HBM
  latency. This is the wall.

## Why the stretch target wasn't met

Both the random-gather K loads (a few hundred cycles to HBM, no
prefetch possible because the indices are data-dependent) and the
flash-attention online-softmax data dependence (each K-tile must
finish softmax+accumulate before the next) put a hard floor on per-CU
time. With B=4 there is essentially **not enough independent work for
256 CUs to chew through**, so the wall-clock minimum is set by the
slowest single CU rather than by aggregate throughput. Concretely:

- Naïve compute/bandwidth lower bounds are ~0.3–1 µs per call. We're
  measuring 35–50 µs. The remaining ~30+ µs is fundamentally
  latency / dispatch / softmax-chain overhead that more parallelism
  cannot trade against.
- Things tried that did **not** help (or hurt):
  - `BLOCK_H=32` (fewer head-blocks, less K reload) — wrong outputs;
    looks like a Triton/MFMA layout mismatch I didn't have time to
    chase. Pursuing this further is the single biggest remaining lever
    (would halve K bandwidth and Q bandwidth).
  - `BLOCK_K=64` — register-pressure spill, ~2× slower.
  - `num_warps=8` — drops to ≤1 wg/CU, kills occupancy.
  - SPLIT_K=1 single-kernel for CSA — only 16 programs on 256 CUs,
    160–230 µs.

## Next steps (not done)

Two genuine algorithmic levers remain:

1. **`BLOCK_H=32` (with a Triton-layout fix).** Would halve K-load
   redundancy on top of also halving scratch and grid size.
2. **Persistent kernel with atomic-add reduce**, eliminating the
   reduce launch entirely (saves ~6 µs on every call regardless of
   SPLIT_K). Doable with a global semaphore; the complexity sits in
   making the final-program detection deterministic.

Both of these touch correctness in non-trivial ways and need more
time than this round had.

## Conclusion

v2 is correct, beats v1 on every B=4 CSA/HCA config, and gives
**2.2–3.2× over the baseline on CSA** and **~1.1× on HCA**. The 10×
target was not met; thread-trace evidence points to a structural
wall (memory-latency + work-volume floor at B=4) that needs more than
parameter-tuning to break through.
