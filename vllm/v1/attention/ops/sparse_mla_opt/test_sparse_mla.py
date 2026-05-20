#!/usr/bin/env python3
"""
Benchmark rocm_sparse_attn_decode using rocprofv3 kernel tracing for
accurate GPU-only timing (no Python dispatch overhead).

Shapes are derived from DeepSeek-V4-Pro serving with TP=8,
isl/osl=1024/1024, max_num_seqs=128, concurrency=4.

Usage:
    python test_sparse_mla.py
    python test_sparse_mla.py --warmup 50 --rep 200 -o results.csv
    python test_sparse_mla.py --batch 1 4 --mode hca csa

    # Generate thread-level traces for rocprof viewer:
    python test_sparse_mla.py --trace --trace-output-dir /tmp/att_traces \
        --impl v2 --mode hca --warmup 5 --rep 3

Modes:
    (default)    Outer driver: spawns rocprofv3 per config, parses traces,
                 writes CSV.
    --trace      Thread-level trace mode: spawns rocprofv3 with
                 --advanced-thread-trace (ATT) to generate per-instruction
                 stall/latency traces viewable in rocprof viewer (ui.perfetto.dev
                 or rocprofv3 --ui). Output is NOT parsed — files are left in
                 --trace-output-dir for manual inspection.
    --inner      Called by the driver under rocprofv3. Runs the kernel
                 warmup+rep times, then exits.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants from DSv4 TP=8 serving logs
# ---------------------------------------------------------------------------
NUM_BLOCKS = 44553
NUM_HEADS = 64
HEAD_DIM = 512
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
CACHE_LINE = 584
SCALE = 0.04419417382415922

# Strides observed from production (non-contiguous cache views)
SWA_CACHE_STRIDE0 = 37440      # swa_k_cache: [NUM_BLOCKS, 64, 584]
LATENT_CACHE_STRIDE0 = 1728    # kv_cache:    [NUM_BLOCKS,  2, 584]
LATENT_KV_HEADS = 2
SWA_WINDOW = 128
TOPK_BUDGET = 8192

# Representative test configurations: (batch, mode, topk_ragged_len)
# "hca" = HCA layers (compress_ratio=128, latent cache [N,2,584], topk_indices present)
#          Attends to ALL compressed entries (few entries due to 128:1 compression).
# "csa" = CSA layers (compress_ratio=4, full-head cache [N,64,584], topk_indices=None)
#          Uses sparse top-K selection (many entries due to 4:1 compression).
#
# Batch=4, per-request topk ranges: CSA [256..512], HCA [8..16]
TEST_CONFIGS = [
    # CSA layers (per-request topk: 256 → 512, 5 equally spaced)
    (4, "csa", 256 * 4),
    (4, "csa", 320 * 4),
    (4, "csa", 384 * 4),
    (4, "csa", 448 * 4),
    (4, "csa", 512 * 4),
    # HCA layers (per-request topk: 8 → 16, 5 equally spaced)
    (4, "hca", 8 * 4),
    (4, "hca", 10 * 4),
    (4, "hca", 12 * 4),
    (4, "hca", 14 * 4),
    (4, "hca", 16 * 4),
]


def _make_noncontig_cache(num_blocks, num_heads, cache_line, stride0, device):
    """Allocate a flat buffer and return an as_strided view that reproduces
    the non-contiguous cache layout observed in production."""
    flat = torch.empty(num_blocks * stride0, dtype=torch.uint8, device=device)
    return torch.as_strided(
        flat,
        size=(num_blocks, num_heads, cache_line),
        stride=(stride0, cache_line, 1),
    )


def build_inputs(batch, mode, topk_ragged_len, device="cuda:0"):
    """Construct all inputs for rocm_sparse_attn_decode with garbage data."""
    q = torch.randn(batch, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)

    swa_k_cache = _make_noncontig_cache(
        NUM_BLOCKS, NUM_HEADS, CACHE_LINE, SWA_CACHE_STRIDE0, device,
    )

    if mode == "hca":
        kv_cache = _make_noncontig_cache(
            NUM_BLOCKS, LATENT_KV_HEADS, CACHE_LINE, LATENT_CACHE_STRIDE0, device,
        )
        topk_indices = torch.randint(
            0, NUM_BLOCKS, (batch, 1, TOPK_BUDGET), dtype=torch.int32, device=device,
        )
    else:
        kv_cache = _make_noncontig_cache(
            NUM_BLOCKS, NUM_HEADS, CACHE_LINE, SWA_CACHE_STRIDE0, device,
        )
        topk_indices = None

    topk_lens = torch.full((batch,), topk_ragged_len // max(batch, 1),
                           dtype=torch.int32, device=device)
    swa_lens = torch.full((batch,), SWA_WINDOW, dtype=torch.int32, device=device)

    swa_indices = torch.randint(
        0, NUM_BLOCKS, (batch, 1, SWA_WINDOW), dtype=torch.int32, device=device,
    )

    swa_ragged_indices = torch.randint(
        0, NUM_BLOCKS, (batch * SWA_WINDOW,), dtype=torch.int32, device=device,
    )
    swa_ragged_indptr = torch.arange(
        0, batch * SWA_WINDOW + 1, SWA_WINDOW, dtype=torch.int32, device=device,
    )

    topk_ragged_indices = torch.randint(
        0, NUM_BLOCKS, (topk_ragged_len,), dtype=torch.int32, device=device,
    )
    per_req = topk_ragged_len // max(batch, 1)
    topk_ragged_indptr = torch.arange(
        0, topk_ragged_len + 1, per_req if per_req > 0 else 1,
        dtype=torch.int32, device=device,
    )
    if topk_ragged_indptr.shape[0] != batch + 1:
        topk_ragged_indptr = torch.linspace(
            0, topk_ragged_len, batch + 1, dtype=torch.int32, device=device,
        ).to(torch.int32)

    attn_sink = torch.randn(NUM_HEADS, dtype=torch.float32, device=device)

    return dict(
        q=q,
        kv_cache=kv_cache,
        swa_k_cache=swa_k_cache,
        swa_only=False,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_ragged_indices=swa_ragged_indices,
        swa_ragged_indptr=swa_ragged_indptr,
        topk_ragged_indices=topk_ragged_indices,
        topk_ragged_indptr=topk_ragged_indptr,
        attn_sink=attn_sink,
        scale=SCALE,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        output=output,
    )


def run_inner(batch, mode, topk_ragged_len, warmup, rep, impl="baseline"):
    """Execute the kernel warmup+rep times (called under rocprofv3)."""
    if impl == "v1":
        from vllm.v1.attention.ops.sparse_mla_opt.rocm_aiter_mla_sparse_v1 import (  # noqa: E501
            rocm_sparse_attn_decode_v1 as kernel_fn,
        )
    elif impl == "v2":
        from vllm.v1.attention.ops.sparse_mla_opt.rocm_aiter_mla_sparse_v2 import (  # noqa: E501
            rocm_sparse_attn_decode_v2 as kernel_fn,
        )
    elif impl == "v3":
        from vllm.v1.attention.ops.sparse_mla_opt.rocm_aiter_mla_sparse_v3 import (  # noqa: E501
            rocm_sparse_attn_decode_v3 as kernel_fn,
        )
    else:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            rocm_sparse_attn_decode as kernel_fn,
        )

    inputs = build_inputs(batch, mode, topk_ragged_len)

    for _ in range(warmup):
        kernel_fn(**inputs)
    torch.cuda.synchronize()

    for _ in range(rep):
        kernel_fn(**inputs)
    torch.cuda.synchronize()


def parse_trace(csv_path, rep):
    """Parse rocprofv3 kernel trace CSV → median_us over the last `rep`
    invocations of each kernel, summed across all kernels per invocation."""
    df = pd.read_csv(csv_path)
    if "Kernel_Name" not in df.columns:
        print(f"[WARN] No Kernel_Name column in {csv_path}", file=sys.stderr)
        return [], 0.0

    kernel_rows = df[df["Kernel_Name"].str.contains(
        "sparse_attn|_paged_attn|triton_|_v1_partial_kernel|_v1_reduce_kernel"
        "|_v2_partial_kernel|_v2_reduce_kernel|_v2_single_kernel"
        "|v3_single_kernel|v3_partial_kernel|v3_reduce_kernel|v3_mfma_kernel"
        "|v3_mfma_partial_kernel|v3_mfma_reduce_kernel",
        case=False, na=False,
    )]
    if kernel_rows.empty:
        kernel_rows = df[~df["Kernel_Name"].str.contains(
            "memcpy|memset|fill", case=False, na=False,
        )]

    kernel_names = sorted(kernel_rows["Kernel_Name"].unique().tolist())

    total_duration_ns = None
    for _, grp in kernel_rows.groupby("Kernel_Name"):
        dur = (grp["End_Timestamp"] - grp["Start_Timestamp"]).to_numpy()
        dur = dur[-rep:]
        if total_duration_ns is None:
            total_duration_ns = dur.copy()
        else:
            n = min(len(total_duration_ns), len(dur))
            total_duration_ns = total_duration_ns[-n:] + dur[-n:]

    if total_duration_ns is None:
        return kernel_names, 0.0

    duration_us = total_duration_ns / 1e3
    sort_idx = np.argsort(duration_us)
    median_us = float(duration_us[sort_idx[len(sort_idx) // 2]])
    return kernel_names, median_us


def profile_one(batch, mode, topk_ragged_len, warmup, rep, tmpdir,
                impl="baseline"):
    """Run one config under rocprofv3 and return parsed results."""
    label = f"B{batch}_{mode}_{topk_ragged_len}"
    tag = f"bench_{label}_{impl}"
    trace_prefix = os.path.join(tmpdir, tag)
    trace_csv = f"{trace_prefix}_kernel_trace.csv"

    if os.path.exists(trace_csv):
        os.remove(trace_csv)

    cmd = [
        "rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace_prefix,
        "--", sys.executable, os.path.abspath(__file__),
        "--inner", str(batch), mode, str(topk_ragged_len),
        "--warmup", str(warmup), "--rep", str(rep),
        "--impl", impl,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"[ERROR] rocprofv3 failed for {label}:", file=sys.stderr)
        print(stderr, file=sys.stderr)
        print(stdout, file=sys.stderr)
        return None

    if not os.path.isfile(trace_csv):
        print(f"[ERROR] trace CSV not found: {trace_csv}", file=sys.stderr)
        print(f"  stdout: {stdout[:500]}", file=sys.stderr)
        return None

    kernel_names, median_us = parse_trace(trace_csv, rep)

    return {
        "label": label,
        "batch": batch,
        "mode": mode,
        "topk_ragged_len": topk_ragged_len,
        "time_us": median_us,
        "kernels": "; ".join(kernel_names[:5]),
    }


def trace_one(batch, mode, topk_ragged_len, warmup, rep, outdir,
              impl="baseline", att_target_cu=0, kernel_include_regex=None):
    """Run one config under rocprofv3 ATT and leave traces in outdir.

    Uses -d (output directory) so the trace decoder generates
    ui_output_agent_*_dispatch_* subdirs loadable in rocprof compute viewer.
    """
    label = f"B{batch}_{mode}_{topk_ragged_len}"
    tag = f"att_{label}_{impl}"
    config_dir = os.path.join(outdir, tag)
    os.makedirs(config_dir, exist_ok=True)

    att_consecutive = str(rep) if kernel_include_regex else "1"
    cmd = [
        "rocprofv3",
        "--advanced-thread-trace",
        "--att-target-cu", str(att_target_cu),
        "--att-consecutive-kernels", att_consecutive,
    ]
    if kernel_include_regex:
        cmd += ["--kernel-include-regex", kernel_include_regex]
    cmd += [
        "-d", config_dir,
        "--",
        sys.executable, os.path.abspath(__file__),
        "--inner", str(batch), mode, str(topk_ragged_len),
        "--warmup", str(warmup), "--rep", str(rep),
        "--impl", impl,
    ]

    print(f"  [{label}] running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"  [{label}] FAILED (rc={proc.returncode})", file=sys.stderr)
        print(stderr, file=sys.stderr)
        print(stdout, file=sys.stderr)
        return False

    entries = os.listdir(config_dir)
    ui_dirs = [e for e in entries if e.startswith("ui_output_")]
    other = [e for e in entries if not e.startswith("ui_output_")]
    print(f"  [{label}] done — {len(entries)} item(s) in {config_dir}/")
    for d in sorted(ui_dirs):
        print(f"    {d}/  (viewer dir)")
    for fn in sorted(other):
        fpath = os.path.join(config_dir, fn)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {fn}  ({size_kb:.1f} KB)")
        else:
            print(f"    {fn}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Bench rocm_sparse_attn_decode")
    parser.add_argument("--inner", nargs=3, metavar=("BATCH", "MODE", "TOPK_RAGGED"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("-o", "--output", type=str, default="sparse_mla_results.csv")
    parser.add_argument("--batch", type=int, nargs="+", default=[4],
                        help="Filter to specific batch sizes (default: 4)")
    parser.add_argument("--mode", type=str, nargs="+", default=["hca", "csa"],
                        choices=["hca", "csa"],
                        help="Filter to specific modes (default: both)")
    parser.add_argument("--impl", type=str, default="baseline",
                        choices=["baseline", "v1", "v2", "v3"],
                        help="Kernel implementation to benchmark")
    parser.add_argument("--trace", action="store_true",
                        help="Generate thread-level traces (ATT) instead of "
                             "kernel traces. Output is left in --trace-output-dir "
                             "for viewing with rocprof viewer.")
    parser.add_argument("--trace-output-dir", type=str,
                        default=None,
                        help="Directory to store ATT trace output. "
                             "Created if it does not exist. "
                             "(default: auto-generated temp dir)")
    parser.add_argument("--att-target-cu", type=int, default=0,
                        help="CU to target for ATT collection (default: 0)")
    parser.add_argument("--topk-ragged-len", type=int, nargs="+", default=None,
                        help="Filter to specific topk_ragged_len values "
                             "(default: all from TEST_CONFIGS)")
    parser.add_argument("--kernel-include-regex", type=str, default=None,
                        help="Regex passed to rocprofv3 --kernel-include-regex "
                             "to filter ATT collection. When --impl=v2 and this "
                             "is not set, defaults to matching v2 Triton kernels.")
    args = parser.parse_args()

    if args.inner:
        batch = int(args.inner[0])
        mode = args.inner[1]
        topk_ragged_len = int(args.inner[2])
        run_inner(batch, mode, topk_ragged_len, args.warmup, args.rep,
                  impl=args.impl)
        return

    configs = TEST_CONFIGS
    if args.batch:
        configs = [c for c in configs if c[0] in args.batch]
    if args.mode:
        configs = [c for c in configs if c[1] in args.mode]
    if args.topk_ragged_len:
        configs = [c for c in configs if c[2] in args.topk_ragged_len]

    # ── Thread-level trace (ATT) mode ──────────────────────────────────
    if args.trace:
        if args.trace_output_dir:
            outdir = args.trace_output_dir
        else:
            outdir = tempfile.mkdtemp(prefix="att_sparse_mla_")
        os.makedirs(outdir, exist_ok=True)

        kernel_include_regex = args.kernel_include_regex
        if kernel_include_regex is None:
            if args.impl == "v2":
                kernel_include_regex = (
                    "_v2_partial_kernel|_v2_reduce_kernel|_v2_single_kernel"
                )
            elif args.impl == "v1":
                kernel_include_regex = (
                    "_v1_partial_kernel|_v1_reduce_kernel"
                )

        print(f"\n{'='*90}")
        print(f"  ATT thread-level traces  (impl={args.impl}, warmup={args.warmup}, "
              f"rep={args.rep}, target_cu={args.att_target_cu})")
        if kernel_include_regex:
            print(f"  kernel filter: {kernel_include_regex}")
        print(f"  output dir: {outdir}")
        print(f"{'='*90}\n")

        ok = 0
        for batch, mode, topk_ragged_len in configs:
            success = trace_one(
                batch, mode, topk_ragged_len,
                args.warmup, args.rep, outdir,
                impl=args.impl, att_target_cu=args.att_target_cu,
                kernel_include_regex=kernel_include_regex,
            )
            if success:
                ok += 1

        print(f"\n{ok}/{len(configs)} configs traced.")
        print(f"Trace files in: {outdir}")
        print("Open with:  rocprofv3 --ui  or load .json/.pftrace in "
              "ui.perfetto.dev")
        return

    # ── Normal kernel-trace benchmark mode ─────────────────────────────
    csv_header = ["batch", "mode", "topk_ragged_len", "time_us", "kernels"]
    rows = []

    print(f"\n{'='*90}")
    print(f"  rocm_sparse_attn_decode benchmark  (warmup={args.warmup}, rep={args.rep})")
    print(f"{'='*90}")
    print(f"{'B':>3s} {'Mode':<5s} {'topk_rag':>10s} {'time_us':>10s}")
    print(f"{'-'*40}")

    with tempfile.TemporaryDirectory(prefix="bench_sparse_mla_") as tmpdir:
        for batch, mode, topk_ragged_len in configs:
            result = profile_one(
                batch, mode, topk_ragged_len,
                args.warmup, args.rep, tmpdir, impl=args.impl,
            )
            if result is None:
                print(f"{batch:>3d} {mode:<5s} {topk_ragged_len:>10d}    FAILED")
                continue

            print(f"{batch:>3d} {mode:<5s} "
                  f"{topk_ragged_len:>10d} {result['time_us']:>10.3f}")
            if result["kernels"]:
                print(f"  kernels: {result['kernels']}")

            rows.append([
                result["batch"], result["mode"],
                result["topk_ragged_len"],
                f"{result['time_us']:.3f}",
                result["kernels"],
            ])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(rows)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    import torch
    main()
