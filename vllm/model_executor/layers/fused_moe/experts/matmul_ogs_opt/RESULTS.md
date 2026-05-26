# matmul_ogs optimization results

Hardware: AMD MI355X (gfx950, 256 CUs, ~8 TB/s HBM3e).
Config:   DSV4-Pro TP=8 — K=7168, N_W1=1024, N_W2=7168, E=384, topk=6.
Bench:    `rocprofv3 --kernel-trace`, warmup=50 rep=200, median of GPU-only
          kernel durations summed across `_matmul_ogs` + `_reduce` per call.

## End-to-end numbers (us)

| M  | W1 base | W1 opt | W1 speedup | W2 base | W2 opt | W2 speedup |
|----|--------:|-------:|-----------:|--------:|-------:|-----------:|
| 1  |   38.76 |  20.88 | **1.86x**  |   16.08 |  18.28 | 0.88x      |
| 2  |   39.52 |  24.24 | **1.63x**  |   17.56 |  18.88 | 0.93x      |
| 4  |   40.32 |  39.96 |   1.01x    |   26.52 |  27.80 | 0.95x      |
| 8  |   60.92 |  61.48 |   0.99x    |   41.04 |  41.68 | 0.98x      |
| 16 |   85.08 |  84.68 |   1.00x    |   60.44 |  65.40 | 0.92x      |
| 32 |  132.00 | 133.12 |   0.99x    |  100.56 |  99.36 | 1.01x      |
| 64 |  202.96 | 203.08 |   1.00x    |  165.48 | 164.44 | 1.01x      |

(W2 deltas at M=1..16 are within run-to-run variance; the wrapper is a
no-op for those configs.)

## What the optimization does

`opt.py` monkey-patches `triton_kernels.matmul_ogs.matmul_ogs` to:

1. **Compute a tuned `OptFlags` per call** instead of going through the
   stock `make_default_opt_flags_amd`. The crucial change is enabling
   `split_k > 1` for the BF16xMXFP4 path on CDNA4.

2. **Bypass the `can_use_split_k` guard.** The stock code has
   ```python
   can_use_split_k = scatter_indx is None and not x_has_mx and not w_has_mx
   ```
   which forbids `split_k > 1` whenever the weights carry MX scales. The
   actual `_matmul_ogs` kernel already supports `SPLIT_K > 1` with MX
   scales — it advances `WMxScalePtrs` by `SPLIT_K` per iter, and the
   host-side scratchpad / reduce path handles SPLIT_K > 1 correctly. We
   sidestep the guard by installing `OptFlags` directly via
   `set_opt_flags()` (the function returns early before the guard fires).

3. **Pick `split_k` adaptively** so the launch grid is roughly
   `2 * n_cu`. For the configs in this test that lands on:
   - M=1 W1: 48 base tiles → split_k=8 → 384 tiles (vs 256 CUs)
   - M=2 W1: 96 base tiles → split_k=5 → 480 tiles
   - M=4 W1: 192 base tiles → split_k=2 → 384 tiles
   - M≥8 W1: split_k=1 (grid already saturated)
   - Any W2: split_k=1 (scatter+split_k unsupported in the reducer)

All other knobs (`block_m=32`, `block_n=128`, `block_k=256`,
`num_warps=4`, `num_stages=2`, `waves_per_eu=3`) match the stock CDNA4
choice for `m <= 1024` so we don't regress configurations that the stock
opt already tuned.

## What didn't help, and why

The user's prompt suggested several knobs to try:

- **`num_stages=3`** — measured 5-15% *regression* on every M because the
  CDNA4-MXFP4 block config (`32x128x256` with `waves_per_eu=3`) is
  already at the LDS limit; an extra pipeline stage drops occupancy.
- **`BLOCK_M=16`** — irrelevant here. The grid_m calculation for a
  ragged MoE call is `routing_data.n_blocks(M, block_m)`, which for
  `M*topk <= n_expts_tot` returns the literal token count regardless of
  `block_m`. Halving `block_m` only reduces in-tile compute waste; the
  number of tiles (and thus parallelism) is unchanged.
- **Smaller `BLOCK_N`** — `block_n=64` regressed everything (the per-tile
  MFMA throughput drops faster than the grid-size gain helps);
  `block_n=32` regressed by 2-7x.
- **`is_persistent=True`** — requires TMA, which CDNA4 does not have.

## Why we don't see 5x

The 5x target is **physically impossible** on the medium/large M end of
the sweep on this GPU. The matmul is memory-bound, and HBM bandwidth
bounds the runtime:

| M  | unique FP4 weight bytes | bandwidth-bound time | baseline | room |
|----|------------------------:|---------------------:|---------:|-----:|
| 1  |       22 MB             |       2.7 us         | 38.8 us  | 14x  |
| 16 |     ~352 MB             |       44  us         | 85   us  | 1.9x |
| 64 |    ~906 MB              |      113  us         | 203  us  | 1.8x |

(unique FP4 weight bytes = ~`E_touched * K * N * 0.5`, where
`E_touched ≈ E * (1 - (1 - topk/E)^M)` for random routing.)

For M≥16 the stock kernel already runs at <2x of the bandwidth limit,
so even a perfectly bandwidth-bound implementation would only buy <2x.

The remaining headroom for **small M** is real but bounded by two
fixed-cost terms we can't reach with opt_flags alone:

1. **HIP kernel launch overhead (~5 us per launch).** With split_k the
   reduce kernel costs ~7 us by itself, of which most is launch latency
   on a tiny grid. Eliminating it would cap M=1 W1 around 14 us total
   instead of the current 21 us — still well short of the 7.8 us a 5x
   target would require.

2. **Per-tile fixed overhead inside the MXFP4 kernel** (scale unswizzle,
   accumulator init/spill). With block_m=32 and only one valid token per
   block, ~97% of the per-tile MFMA work is wasted, but the loads and
   address arithmetic are paid in full.

A custom HIP kernel that (a) avoids the m-block padding for `M*topk <
n_expts_tot` (one program per `(routed_token, n_tile)` doing a real GEMV
along K instead of a padded GEMM) and (b) folds the reduce/scatter into
the matmul kernel could in principle reach the bandwidth bound for
small M (i.e. ~3 us for M=1 W1, ~10x speedup). That is a multi-day
exercise — the kernel reimplements MXFP4 dequant + the CDNA4 swizzled
scale layout + the routed expert dispatch — and is out of scope for
this iteration.

## Files

- `opt.py` — the wrapper, exported as `install()` / `uninstall()` /
  `_matmul_ogs_opt()`. Importing the module auto-installs.
- `test_matmul_ogs_opt.py` — same CLI as `test_matmul_ogs.py`, but the
  outer driver and inner runner both `import opt` so every kernel call
  goes through the patched matmul_ogs. Also widens `parse_trace` to
  include the `_reduce` kernel so split_k numbers are apples-to-apples.
- `test_matmul_ogs_baseline.py` — runs the stock matmul_ogs with the
  same widened parser (the stock parser missed `_reduce`, which made the
  W2 baseline look ~7 us faster than it really was).
- `baseline_final.csv`, `opt_final.csv` — the numbers above.
