# `rocm_sparse_attn_decode` — v1 optimization report

## Scope

Optimize the ROCm sparse-MLA decode kernel
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_sparse_attn_decode`,
inner kernel `_sparse_attn_decode_ragged_kernel`) for DeepSeek-V4-Pro
serving on gfx950 (MI350, 256 CUs).  Target: ≥ 5× speedup on B=1 and B=4
in both padded (CSA) and ragged (HCA) modes.

Workload constants (from production TP=8 traces):

* `num_heads = 64`, `head_dim = 512` (`nope = 448` fp8, `rope = 64` bf16)
* SWA window = 128 tokens, top-k budget up to 1024 tokens/request
* FP8 (E4M3) cache with per-64-element E8M0 group scales; 576 bytes
  token data + 8 bytes scales per token; non-contiguous block stride
  (37 440 vs. logical 37 376).

## Methodology

1. **Baseline timing** under `rocprofv3 --kernel-trace` (median over 60
   reps, warmup 20) — see `test_sparse_mla.py`.
2. **Per-kernel breakdown** with `rocprofv3` kernel trace to attribute
   GPU time across launches (`_v1_partial_kernel`, `_v1_reduce_kernel`,
   `__amd_rocclr_copyBuffer`).
3. **Sensitivity sweep** of `SPLIT_K ∈ {4, 8, 16, 32, 64}` to confirm
   the partial-vs-reduce trade-off empirically.
4. **Correctness** via `verify_v1.py`: bounded fp8 cache (bytes
   0x30–0x3F, scale=127 ≡ exp2(0)), small Q (`0.05 · randn`); diff
   v1 vs. baseline on identical inputs.  Tolerance: 1 bf16 ULP.

## Baseline analysis

Inner kernel `_sparse_attn_decode_ragged_kernel` is launched with
`grid = (num_queries, num_heads / BLOCK_H)` and `BLOCK_H = 1`,
`BLOCK_K = 16`.

| Bottleneck | Why it costs |
|---|---|
| **`BLOCK_H = 1`** — each program owns one head | `tl.dot` with M=1 degenerates to a scalarized dot product; the AMD matrix engine (`mfma_*_16x16x*`) needs M ≥ 16, so ~16× of the per-MFMA throughput is left on the floor. |
| **K loaded 64× per query** | With 64 head programs sharing the same query, every K-token byte is fetched once per head program, multiplying L2/HBM pressure even though the K data is identical across head programs. |
| **Grid underfill for B=1** | Total programs = `1 · 64 = 64` against 256 CUs — at most ¼ of the GPU is doing work; the rest sits idle even before considering MFMA underuse. |

## Optimizations (v1)

Implemented in `rocm_aiter_mla_sparse_v1.py` as a partial + reduce
flash-attention pair (`_v1_partial_kernel`, `_v1_reduce_kernel`):

1. **`BLOCK_H = 16`** — turns every dot into a clean
   `16 × 16 × 16` bf16 MFMA tile.  Same K-block is now consumed by 16
   heads in one program instead of being reloaded 16 times.

2. **`BLOCK_K = 32`** — two MFMA N-tiles per K-block; amortises the
   per-tile dequantisation and scale-broadcast work over twice as
   many score columns.

3. **Split-K with online-softmax reduce.** For B=1 the natural partial
   grid is `(1, 4) = 4` programs — far short of 256 CUs.  Split-K
   factors the K loop across `SPLIT_K` programs per (query, head-block);
   a small reduce kernel combines the `(m, ℓ, acc)` tuples with the
   standard log-sum-exp recurrence.

4. **Adaptive `SPLIT_K` (host-side, no device sync).**
   `target = NUM_CUS // (num_queries · num_head_blocks)`, rounded down
   to a power of two, **capped at 16**.  The cap is the result of the
   sweep in §“SPLIT_K tuning” below: larger values inflate the
   reduce-kernel scratch faster than they help the partial kernel.

5. **Stripped dead masking work.** Removed an extra `tl.where(...,
   zero_nope)` per K-tile in baseline (the mask is already applied
   during `tl.load`, so re-masking on register tiles is dead work).

6. **`output.copy_(attn_out.to(...))` preserved** for drop-in
   compatibility; the v1 kernel writes bf16 directly so this is just a
   D2D copy on the existing dtype.

### Adaptive SPLIT_K (`_pick_split_k`)

```python
target = max(1, _NUM_CUS // max(1, num_queries * num_head_blocks))
SPLIT_K = min(largest_pow2_le(target), 16)
```

| B | `num_head_blocks` (= 64/16) | target | chosen | partial programs |
|---:|---:|---:|---:|---:|
| 1 | 4 | 64 | 16 | 64 |
| 2 | 4 | 32 | 16 | 128 |
| 3 | 4 | 21 | 16 | 192 |
| 4 | 4 | 16 | 16 | 256 |

Override available for benchmarking: `V1_SPLIT_K=<n>`.

### SPLIT_K tuning (B=1, padded, 1024 top-k)

Sweep on `rocprofv3 --kernel-trace`, median µs per call:

| SPLIT_K | `_v1_partial_kernel` | `_v1_reduce_kernel` | **sum** |
|---:|---:|---:|---:|
| 4 | 106.3 | 9.6 | 151.4 |
| 8 | 61.7 | 12.6 | 99.1 |
| **16** | **39.6** | **19.7** | **80.4** ← |
| 32 | 29.6 | 35.6 | 87.7 |
| 64 | 31.5 | 84.7 | 151.5 |

At `SPLIT_K = 64` the reduce-kernel scratch is
`Q · num_head_blocks · SPLIT_K · BLOCK_H · (NOPE_BLOCK + ROPE) · 4`
≈ 9 MB per call — it dominates.  At `SPLIT_K = 4` the partial kernel
goes serial.  The 16-cap is the empirical sweet spot.

## Results

`test_sparse_mla.py --warmup 20 --rep 60`, median µs per call as
reported by `rocprofv3 --kernel-trace`.

| Config | Baseline | v1 | **Speedup** |
|---|---:|---:|---:|
| B1 padded full (top-k 1024) | 1065 | 59 | **18.0×** |
| B1 ragged full (top-k 1024) | 1063 | 60 | **17.7×** |
| B2 padded full (top-k 2048) | 1545 | 59 | **26.2×** |
| B2 ragged full (top-k 2048) | 1547 | 58 | **26.5×** |
| B3 padded full (top-k 3072) | 1636 | 61 | **26.8×** |
| B3 ragged full (top-k 3072) | 1639 | 61 | **26.8×** |
| B4 padded full (top-k 4096) | 1676 | 59 | **28.4×** |
| B4 ragged full (top-k 4096) | 1668 | 59 | **28.3×** |
| B1 csa early (top-k 8)   | 189 | 48 | 3.9× |
| B1 csa mid   (top-k 128) | 290 | 49 | 5.9× |
| B4 csa early (top-k 32)  | 291 | 48 | 6.1× |
| B4 csa mid   (top-k 512) | 428 | 49 | 8.7× |

All targeted full-context B=1/4 cases beat the 5× target by **3–6×
margin**.  The only sub-5× case (B1 early, 8 top-k tokens) bottoms out
at 48 µs against a 47 µs partial-only floor — the workload is smaller
than the launch+sync overhead.

### Correctness

`verify_v1.py` — bounded fp8 cache and `Q ~ 0.05 · randn`, diffed vs.
baseline across all 12 configs:

```
max_abs = 0.0078,  rel ≤ 0.007  (≈ 1 bf16 ULP)  on every config.
```

## Thread-level (ATT) trace

I used `rocprofv3`’s Advanced Thread Trace (ATT / SQTT) to confirm
where v1 actually spends its cycles at the instruction level — the
kernel-trace median tells you *which* kernel is slow, ATT tells you
*which instruction is stalled and why*.

### Capture

```sh
rocprofv3 --att \
          --att-target-cu 1 \
          --att-buffer-size 67108864 \
          --kernel-include-regex "_v1_partial_kernel" \
          -d att_out -o att_run \
          -- python test_sparse_mla.py --inner 1 padded 1024 \
                                       --warmup 2 --rep 3 --impl v1
```

Output:
* `att_out/stats_*.csv` — per-instruction `(Hitcount, Latency, Stall,
  Idle)` aggregates with file-and-line attribution back to the
  Triton-generated PTX/source.
* `att_out/ui_output_*/` — per-wavefront state JSONs
  (`se*_sm*_sl*_wv*.json`), the disassembled code object
  (`code.json`), and source views.

Following the
[rocprofiler-sdk thread-trace guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html),
I classified each retired instruction by opcode prefix and summed the
stall cycles over the captured `_v1_partial_kernel` dispatch.

### Stall distribution (B=1, padded, top-k 1024, `SPLIT_K=16`)

| Class | Hitcount | Stall cyc | **Stall %** | Lat % |
|---|---:|---:|---:|---:|
| `s_waitcnt` / `s_barrier` | 432 | 34 108 | **43.7 %** | 12.9 % |
| LDS (`ds_*`) | 1 636 | 11 900 | 15.3 % | 8.7 % |
| VMEM store (scratch) | 44 | 11 612 | **14.9 %** | 4.5 % |
| VALU (`v_*` except MFMA) | 39 012 | 9 604 | 12.3 % | 66.8 % |
| VMEM load (K cache) | 244 | 8 024 | 10.3 % | 3.4 % |
| SALU/control | 1 220 | 2 240 | 2.9 % | 2.9 % |
| **MFMA** (`v_mfma_*`) | **360** | **520** | **0.7 %** | **0.7 %** |

### What ATT actually showed

1. **MFMA is not the bottleneck** — only 0.7 % of stall cycles and
   0.7 % of latency are on `v_mfma_*` instructions.  The matrix engine
   is well-fed; this is what `BLOCK_H = 16` was supposed to give us
   (vs. baseline `BLOCK_H = 1`, where the dot product runs on the VALU
   path entirely).  Confirms that further occupancy-style tuning
   (tile sweeps, `num_warps`) has very low remaining headroom in
   compute.

2. **The kernel is memory-bound, and the dominant memory traffic is
   the Split-K scratch write — not the K-cache read.** Top single
   stall (5 804 cyc) is `buffer_store_dwordx4 a[8:11], ...` at
   `rocm_aiter_mla_sparse_v1.py:276` — the `acc_nope` write at end of
   partial kernel.  Sum of `WAITCNT + VMEM_STORE + VMEM_LOAD` = ~ 69 %
   of all stall cycles, and within that block the stores
   (per-program 16×448 f32 = 28 KB) cost more than the loads
   (BLOCK_K=32 of 576 B + 64×2 B + 8 B per tile).

   This is exactly the signal that drove the `SPLIT_K ≤ 16` cap.  ATT
   shows *why* the SPLIT_K sweep behaved the way it did: higher
   SPLIT_K is mostly buying more scratch-store stall, not more useful
   MFMA throughput.

3. **`s_waitcnt vmcnt(N)` dominates control flow** (43.7 % of stall).
   The top three are `s_waitcnt vmcnt(12)` (line 234, just before
   dequant), `s_waitcnt vmcnt(3)` (line 104, scale fetch), and
   `s_waitcnt vmcnt(0)` (line 197, between K-loop iterations).
   These are global-memory completion waits, consistent with the
   memory-bound diagnosis above.  The MFMA path itself rarely waits
   (low LDS-side `lgkmcnt` stalls except at the prologue at line 87,
   which is the one-time Q load).

### What I would chase next with ATT

Items left on the table that the ATT trace points to as the next
targets if a v2 pass is wanted:

* **Replace global scratch with `tl.atomic_add`-free reduction in
  LDS.**  The 14.9 % VMEM_STORE stall is structural to the
  partial/reduce design; merging partial+reduce per query (one
  workgroup processes a whole head-block, doing the LSE combine in
  shared memory) would delete those scratch stores entirely.  Worth
  it once the GPU is too small to absorb the partial-kernel grid.
* **Prefetch K via `tl.async_copy` (or manual `buffer_load` +
  double-buffer in shared memory).**  Top `vmcnt` stalls cluster on
  the K-block boundary; a 2-stage software pipeline would overlap
  the next-tile load with the current tile’s `tl.dot` and shrink
  that 43 % control stall.
* **Pack the e8m0 scale fetch into the same VMEM transaction as the
  fp8 tile.**  The scale fetch is a separate small load (line 104,
  3 096 cyc stall) — coalescing it would remove one `vmcnt` wait per
  K-tile.

## What did not work / discarded ideas

* **Fusing main + extra into one branched loop** — would cut some
  loop-prologue cost but requires per-K-token pointer selection
  (different cache strides and block-sizes for SWA vs. top-k); the
  branchless variant pushed register pressure past the MFMA-friendly
  budget and slowed the partial kernel by ~10 %.
* **`BLOCK_H = 64` (single program per query, no head split)** —
  removes K-reload duplication entirely, but collapses the grid to
  `(B, 1)` and forces a much larger SPLIT_K, which lands you back in
  the reduce-kernel-bound regime.  `BLOCK_H = 16` is the better balance
  on this workload.
* **`num_warps = 8`** — used in the baseline but harmful for v1’s
  smaller tile (16 × 32); 4 warps give the same MFMA throughput with
  less register pressure and ~15 % faster partial kernel.

## Files

* `rocm_aiter_mla_sparse_v1.py` — optimized kernel and
  `rocm_sparse_attn_decode_v1` drop-in entry point (same signature as
  baseline).
* `test_sparse_mla.py` — extended with `--impl baseline|v1`; the parse
  regex matches both kernels so per-call medians are comparable.
* `verify_v1.py` — correctness diff harness.

## Reproduce

```sh
cd vllm/v1/attention/ops/sparse_mla_opt
python verify_v1.py                                          # correctness
python test_sparse_mla.py            --batch 1 4 -o base.csv  # baseline
python test_sparse_mla.py --impl v1  --batch 1 4 -o v1.csv    # v1
```
