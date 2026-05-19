# Thread-level trace analysis — `_v2_partial_kernel` (B=4 HCA)

GPU: AMD gfx950 (MI350), 256 CUs, 4 SIMDs/CU.
Workload: `rocm_sparse_attn_decode_v2`, batch=4, mode=hca,
`topk_ragged_len=32` (per-query main_len=128, extra_len=8).
Tool: `rocprofv3 --advanced-thread-trace --att-target-cu 0
--att-consecutive-kernels 1`.

Kernel resource footprint (from kernel dispatch info):
- VGPRs/wave: **256** (the binding constraint, see below)
- SGPRs/wave: 112
- LDS:        0 bytes (no static SLM use)
- Scratch:    0 bytes (no spills)
- Workgroup:  256 threads = 4 waves of 64
- Grid:       4 (Q) × 4 head-blocks × 2 splits = 32 workgroups

## Headline numbers

```
Total sampled cycles (one wave):  latency = 257,100   stall = 68,312  (26.6%)
```

So roughly **a quarter of the kernel's wall time is spent stalled**,
and the remaining three-quarters is "executing", but that execution is
overwhelmingly bookkeeping VALU + transcendentals + LDS shuffles —
**not MFMA**. (See "Compute vs. memory bound" below.)

## Stall breakdown by instruction class

Aggregated across all sampled instructions in the dispatch:

```
class                stall    lat        hits   count  stall%   lat%
wait_vmem           18,436   18,436       164     41    27.0%    7.2%
valu                 8,140  134,324    31,496  8,075    11.9%   52.2%
vmem_load            7,884    8,780       220     55    11.5%    3.4%
wait_lds_scalar      7,420    7,420       160     42    10.9%    2.9%
lds_read             5,992    9,800       952    238     8.8%    3.8%
lds_write            5,656   12,760       684    171     8.3%    5.0%
barrier              5,740    5,740       108     27     8.4%    2.2%
vmem_store           4,136    4,316        44     11     6.1%    1.7%
nop                  2,624    2,624       576    188     3.8%    1.0%
convert              1,588   22,552     5,240  1,312     2.3%    8.8%
transcendental         340   21,396     2,632    658     0.5%    8.3%
mfma                   212    1,652       360     90     0.3%    0.6%
salu                     0    7,060       776    226     0.0%    2.7%
```

(`wait_*` rows count cycles spent at `s_waitcnt` / `s_barrier`;
`vmem_load/store` rows are stalls *during issue* of those ops.)

What this says:
- **`s_waitcnt vmcnt(...)` alone burns 27 % of the kernel's stall
  budget.** Those are explicit blocks on outstanding HBM loads — the
  classic latency-bound signature.
- Combined "memory-related" stall (`wait_vmem` + `vmem_load` +
  `wait_lds_scalar` + `lds_read` + `lds_write` + `vmem_store`) is
  **~73 % of all stall cycles**. Everything else (VALU back-pressure,
  barriers, nops) is small individually.
- MFMA — the work the matrix engines do — accounts for **0.6 % of
  total latency cycles and 0.3 % of stalls**. We are not running out
  of matrix throughput.

## Where the waits land (per source-line attribution)

Top source lines in `rocm_aiter_mla_sparse_v2.py`, ranked by stall
cycles attributed to instructions emitted for that line:

```
line  stall    lat      stall%  lat%   description
398   20,648   70,268   30.2%   27.3%  k_nope = x_fp8.to(bf16) * scales.to(bf16)    <-- dequant after K load
393    8,684    9,612   12.7%    3.7%  encoded_scales = tl.load(token_scale, …)    <-- issuing scale load
407    6,780   56,188    9.9%   21.9%  scores = tl.dot(q_nope, k_nope.T) + …       <-- QK^T
342    5,252    6,588    7.7%    2.6%  q_nope = tl.load(…)                          <-- initial Q load
430    3,668    3,684    5.4%    1.4%  acc_nope = … + tl.dot(p, k_nope)            <-- PV dot
507    3,292    3,940    4.8%    1.5%  (extra loop) acc_nope PV
366    2,672    4,080    3.9%    1.6%  slot = tl.load(main_indices_ptr, …)         <-- gather index
330    2,640    3,136    3.9%    1.2%  pid setup / scalar indptr loads
361    2,204    2,220    3.2%    0.9%  main_len = main_end - main_start             <-- waits on indptr
408    2,076    2,876    3.0%    1.1%  + tl.dot(q_rope, k_rope.T)
397    1,096   43,912    1.6%   17.1%  scales = tl.exp2(encoded_scales - 127)       <-- transcendentals run
```

The picture this paints, in plain English:

1. **The biggest single stall (line 398, 30 % of all stalls) is the
   dequant of K.** It's not the math; it's that the *first use* of
   `x_uint8` and `encoded_scales` immediately after their loads forces
   an `s_waitcnt vmcnt(0)` on both vmem fetches. The dequant ALU is
   fast — the wait is for HBM data to land.
2. **Line 393 (12.7 %) is the issue-side stall on the scale load**
   itself — back-pressure on the vmem pipe while the K-byte load
   queue is full.
3. **Line 407 (9.9 % stall, 22 % latency) is the QK^T dot.** Its
   stall is small relative to its total latency: most of those 56k
   latency cycles are real VALU/MFMA work. This is the only
   "honest compute" line in the top of the list.
4. **Line 342 (7.7 %) — the initial Q-nope load** — costs us
   ~2 900 cycles per wave once per kernel invocation. That's a
   prologue tax that does not shrink with more work.
5. **Lines 330/361 (~7 % combined)** are scalar `lgkmcnt(0)` waits on
   the small indptr loads. Tiny payloads, full HBM round-trip.

## Compute vs. memory bound: **memory-latency bound**

Three independent lines of evidence say the same thing:

1. **Stall composition.** 27 % of stall cycles are `s_waitcnt vmcnt`
   and another ~28 % are stalls during the issue of vmem/LDS ops.
   Only ~12 % of stalls are VALU back-pressure, and ~0.3 % are
   MFMA-related.
2. **Latency composition.** 52 % of executed cycles are VALU,
   ~17 % LDS, 8 % converts, 8 % transcendentals, and only **0.6 %
   MFMA**. With a 256-thread workgroup on gfx950's matrix engines,
   if compute were the bottleneck MFMA should dominate. It doesn't.
3. **Theoretical bounds.** Compute work at this shape is ~47 MFLOPs
   (4 × 64 × 136 × 576 × 2); at 1.3 PFLOPs bf16 the compute lower
   bound is ~0.04 µs. K bandwidth is ~310 KB; at 5 TB/s the bandwidth
   lower bound is ~0.06 µs. **Measured kernel time is ~35 µs** — ~800×
   off both compute and bandwidth ceilings. That gap is latency, not
   throughput.

Important nuance: we are **memory-*latency* bound, not memory-
*bandwidth* bound.** HBM bandwidth is barely touched; what hurts is
the round-trip time of each individual gather load combined with too
few in-flight waves to hide that round-trip.

### Why latency cannot be hidden here

The standard cure for memory latency on GPUs is wave-level parallelism
— give each CU enough resident waves that while one is stalled on
HBM, another is ready to issue. Two things break that here:

- **VGPRs/wave = 256.** The dominating consumer is the
  `[BLOCK_H, NOPE_BLOCK] = [16, 512]` fp32 accumulator (32 VGPRs/thread
  on its own). gfx950 has 512 VGPRs per SIMD (~2048 per CU), so
  `2048 / 256 = 8 waves/CU max`. With `num_warps=4` (4 waves per
  workgroup), at best **2 workgroups can be resident per CU**.
- **Workgroup count is small.** With B=4, BLOCK_H=16, SPLIT_K=2, the
  grid is **32 workgroups for 256 CUs** — we cannot even put 1
  workgroup on each CU, let alone over-subscribe to hide latency.
  Most CUs sit idle; the busy CUs each run one workgroup with no
  partner to swap to on every K-load stall.

That is the wall. It is structural at this batch size: the
sparse-MLA work for 4 queries is just not big enough to feed 256 CUs
with enough independent waves.

## Why each stall happens — and what could remove it

| Stall site                                  | Root cause                                                                                                  | Realistic mitigation                                                                                                                                                                                                                                                                          | What still won't yield                                                                |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **line 398 / 393** — dequant after K load   | Data-dependent gather of FP8 K bytes + 1-byte e8m0 scale → the very next instr needs the data → `vmcnt` wait | (a) **Software-pipeline manually** so iter N+1's loads are issued before iter N's compute consumes its loads (Triton's `num_stages=2` only buys us one stage — making more would explode VGPR pressure since each in-flight K tile costs another 16 VGPRs/thread). (b) Pull the scale byte into the same physical load as the data (re-pack the cache so each token = 448 fp8 + 8 scale + 128 rope in one contiguous 584-byte line). (c) Cooperative LDS K-tile: have warps share the gather, then everyone reads from LDS — turns 32 random HBM ops/iter into 32 LDS broadcasts after one cooperative fetch. | The HBM round-trip itself stays — gathers cannot be coalesced when the indices are independent and unsorted. |
| **line 407 / 408 / 430 / 507** — `tl.dot`   | Real compute, but every dot waits on the K it depends on                                                    | Same as above — once K is in registers/LDS, the dot itself is fast (only 9.9 % stall on 22 % latency = high IPC).                                                                                                                                                                            | Same — bound by upstream K availability.                                              |
| **line 342** — initial Q-nope load          | Cold HBM read of 16 × 576 bytes per program at kernel start                                                  | (a) **Persistent kernel**: amortise this 3 µs prologue across multiple (Q, head-block, split) tiles per CU instead of paying it once per tile launch. (b) Pre-load Q into shared / register file via the dispatch's first wavefront and keep it resident.                                  | At least one Q load per query — fundamental.                                          |
| **lines 330 / 361 / 366** — indptr/index loads | Tiny scalar/SGPR loads serialised by `lgkmcnt(0)`                                                            | (a) Pack indptr and indices into a single host-built struct so they share a cache line. (b) Pre-fetch all indptr values into LDS once at kernel start (one warp does the gather), reuse from LDS in every iter.                                                                                | The first cold cache fill is unavoidable.                                              |
| **`s_barrier` (8.4 %)**                     | `tl.dot` and `tl.max`/`tl.sum` emit workgroup-wide barriers between the matmul tiles and the LDS reductions | (a) Reduce # of barriers per iter by fusing main + extra loops into one virtual K stream (saves a full softmax/dot epilogue). (b) Smaller `BLOCK_H` so the reductions fit in a single wave (would eliminate the cross-warp `s_barrier`), but at the cost of more programs and more Q reloads. | Cross-warp softmax reductions need *some* barriers.                                   |
| **High `valu` stall (11.9 %)**              | VALU pipe back-pressure from `v_cndmask`, `v_lshrrev`, `v_and` — the mask/index-arithmetic surrounding every gather (line 397's `tl.exp2` and surrounding ops show up here) | (a) Move the `// 64`, `>= 0`, `< main_num_rows` checks out of the hot loop. (b) Pre-bake `safe_slot`, `pos_in_block`, `block_idx` for the whole K range in a fused pre-pass kernel, so the inner loop is straight-line "load → dequant → dot". | Not a huge win on its own (~5 µs ceiling).                                            |
| **256-VGPR occupancy starvation**           | Accumulator is 16 × 512 × fp32 = 32 VGPRs/thread; plus q_nope, q_rope, transient K tile — 256 total          | (a) **Spill `acc_nope` to LDS** between iters: per-CU LDS is 160 KB, and acc_nope at our shape is 32 KB — fits easily; halves VGPR demand and lets us go from 2 to 4 wgs/CU. (b) **Tile the NOPE dim inside the inner loop** so only a slice of the accumulator lives in VGPRs at a time — adds complexity but cuts pressure substantially. | LDS-resident accumulator adds 2-3 cycles per access; only worth it if extra wave-parallelism actually fills the latency holes (it does, for this shape). |

## Things we tried in the trace's light, with outcomes

- `V2_NUM_STAGES=2` (added one stage of K-tile prefetch): HCA 45 → 35 µs.
  Confirmed by trace — `wait_vmem` stalls dropped vs. `num_stages=1`.
- `V2_NUM_WARPS=8` (more waves per wg, theoretically better latency
  hiding): regresses to 65–93 µs. The issue: 8 warps × 256 VGPRs is
  too much for the SIMD; we actually drop to 1 wg/CU and *fewer*
  waves resident in total.
- `V2_BLOCK_K=64` (fewer iters, more compute per iter, bigger
  amortisation): doubles VGPR demand on the K tile and spills to
  scratch → 75–110 µs.
- `V2_BLOCK_H=32` (halve K-load redundancy across head-blocks): hits
  a numerical bug we did not finish debugging (likely a Triton
  MFMA-tiling mismatch at `M=N=32, K=512`). This is the largest
  untaken structural win on the table — would cut K-bandwidth and
  Q-bandwidth in half *and* shrink the grid.

## Bottom line

The kernel is **memory-latency bound, with severe occupancy
starvation**. ~73 % of stall cycles trace to memory waits and another
~12 % to mask/index VALU bookkeeping around those memory ops. MFMA
time is well under 1 % — there is **no compute deficit to be made up**.

The remaining headroom lives in three places, in roughly descending
expected payoff:

1. **Wave-level parallelism**: push VGPR usage down (LDS-resident
   accumulator and/or NOPE-tiling) so that 4+ workgroups can be
   resident per CU and there is always a non-stalled wave to issue.
   This is the lever most directly addressing the 27 % `wait_vmem`
   stall.
2. **Per-tile launch amortisation**: persistent kernel + atomic-add
   reduce. Removes both the recurring Q-load prologue (~3 µs) and
   the entire reduce-kernel launch.
3. **Cache-layout / pipeline changes**: cooperative LDS gather of
   each K tile (turns 32 random HBM ops into LDS broadcasts after a
   single coalesced fetch); fused main+extra K stream (kills one
   set of softmax-epilogue barriers per call).

None of those are tile-size knob tweaks; all three are structural
changes that require rewriting the kernel body, not just retuning
constants.
