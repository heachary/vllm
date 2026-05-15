#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark pack_bitmatrix kernel using rocprofv3 kernel tracing for accurate
GPU-only timing (no Python dispatch overhead).

Follows the same pattern as triton_bench_rocprof_v3.py:
  - Outer driver: spawns rocprofv3 per (variant, n_rows), parses kernel traces
  - --inner: called by driver under rocprofv3, runs warmup+rep dispatches

Usage:
    python bench_rocprof.py
    python bench_rocprof.py --n_rows 1 2 3 4 128 256 --rep 200
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

NUM_LOCAL_EXPERTS = 48
NUM_TOPK = 6

ORIG_BLOCK_SIZE_M = 512
ORIG_BLOCK_SIZE_K = 32
V2_BLOCK_SIZE_M = 256
V2_BLOCK_SIZE_K = 8


def run_inner(variant, n_rows, warmup, rep):
    """Execute the kernel warmup+rep times (called under rocprofv3)."""
    import torch
    import triton

    torch.set_default_device("cuda")

    bm_cols = triton.cdiv(NUM_LOCAL_EXPERTS, 32)
    topk_ids = torch.randint(
        0, NUM_LOCAL_EXPERTS, (n_rows, NUM_TOPK), dtype=torch.int16
    )
    bitmatrix = torch.zeros((n_rows, bm_cols), dtype=torch.uint32)

    if variant == "original":
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
            pack_bitmatrix,
        )
        grid = (triton.cdiv(n_rows, ORIG_BLOCK_SIZE_M),)

        for _ in range(warmup):
            pack_bitmatrix[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=ORIG_BLOCK_SIZE_M, BLOCK_SIZE_K=ORIG_BLOCK_SIZE_K,
            )
        torch.cuda.synchronize()

        for _ in range(rep):
            pack_bitmatrix[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=ORIG_BLOCK_SIZE_M, BLOCK_SIZE_K=ORIG_BLOCK_SIZE_K,
            )
        torch.cuda.synchronize()

        print("KERNEL_NAME=pack_bitmatrix")
    else:
        from vllm.model_executor.layers.fused_moe.experts.pack_bitmatrix_optimization.pack_bitmatrix_v2 import (
            pack_bitmatrix_v2,
        )
        grid = (triton.cdiv(n_rows, V2_BLOCK_SIZE_M),)

        for _ in range(warmup):
            pack_bitmatrix_v2[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=V2_BLOCK_SIZE_M, BLOCK_SIZE_K=V2_BLOCK_SIZE_K,
            )
        torch.cuda.synchronize()

        for _ in range(rep):
            pack_bitmatrix_v2[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=V2_BLOCK_SIZE_M, BLOCK_SIZE_K=V2_BLOCK_SIZE_K,
            )
        torch.cuda.synchronize()

        print("KERNEL_NAME=pack_bitmatrix_v2")


def parse_trace(csv_path, kernel_name, rep):
    """Parse rocprofv3 kernel trace CSV -> median duration in us.

    Only considers the last `rep` invocations (skips warmup).
    """
    df = pd.read_csv(csv_path)
    kernel_rows = df[df["Kernel_Name"].str.contains(kernel_name, case=False)]

    if kernel_rows.empty:
        print(f"  WARNING: No dispatches found for kernel '{kernel_name}'")
        print(f"  Available kernels: {df['Kernel_Name'].unique()[:5]}")
        return None

    dur_ns = (kernel_rows["End_Timestamp"] - kernel_rows["Start_Timestamp"]).to_numpy()
    dur_ns = dur_ns[-rep:]
    dur_us = dur_ns / 1e3

    return {
        "median_us": float(np.median(dur_us)),
        "mean_us": float(np.mean(dur_us)),
        "min_us": float(np.min(dur_us)),
        "max_us": float(np.max(dur_us)),
        "count": len(dur_us),
    }


def profile_one(variant, n_rows, warmup, rep, tmpdir):
    """Run one (variant, n_rows) case under rocprofv3 and return parsed results."""
    tag = f"{variant}_n{n_rows}"
    trace_prefix = os.path.join(tmpdir, tag)
    trace_csv = f"{trace_prefix}_kernel_trace.csv"

    cmd = [
        "rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace_prefix,
        "--", "python3", os.path.abspath(__file__),
        "--inner", variant, str(n_rows),
        "--warmup", str(warmup), "--rep", str(rep),
    ]

    env = os.environ.copy()
    env["HSA_TOOLS_DISABLE_REGISTER"] = "0"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"[ERROR] rocprofv3 failed for {variant} n_rows={n_rows}:")
        print(stderr)
        return None

    kernel_name = "unknown"
    for line in stdout.splitlines():
        if line.startswith("KERNEL_NAME="):
            kernel_name = line.split("=", 1)[1]

    if not os.path.isfile(trace_csv):
        print(f"[ERROR] trace CSV not found: {trace_csv}")
        print(f"  Files in tmpdir: {os.listdir(tmpdir)}")
        return None

    stats = parse_trace(trace_csv, kernel_name, rep)
    if stats is None:
        return None

    stats["kernel_name"] = kernel_name
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pack_bitmatrix via rocprofv3 kernel tracing"
    )
    parser.add_argument("--inner", nargs=2, metavar=("VARIANT", "N_ROWS"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--n_rows", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if args.inner:
        variant, n_rows = args.inner[0], int(args.inner[1])
        run_inner(variant, n_rows, args.warmup, args.rep)
        return

    # Outer driver mode
    print("=" * 70)
    print("  pack_bitmatrix rocprofv3 Kernel Trace Benchmark")
    print(f"  warmup={args.warmup}, rep={args.rep}")
    print("=" * 70)

    all_results = {}
    with tempfile.TemporaryDirectory(prefix="bench_pack_bm_") as tmpdir:
        for variant in ["original", "optimized"]:
            print(f"\n--- {variant.upper()} ---")
            variant_results = {}
            for n_rows in args.n_rows:
                result = profile_one(variant, n_rows, args.warmup, args.rep, tmpdir)
                if result is None:
                    print(f"  n_rows={n_rows}: FAILED")
                    continue
                print(f"  n_rows={n_rows}: median={result['median_us']:.2f} us, "
                      f"mean={result['mean_us']:.2f} us, "
                      f"min={result['min_us']:.2f} us "
                      f"({result['count']} samples, kernel={result['kernel_name']})")
                variant_results[n_rows] = result
            all_results[variant] = variant_results

    # Comparison table
    if "original" in all_results and "optimized" in all_results:
        print("\n" + "=" * 70)
        print("=== rocprofv3 Kernel Trace: GPU Execution Time Comparison ===")
        print("=" * 70)
        rows = []
        for n_rows in args.n_rows:
            orig = all_results["original"].get(n_rows, {})
            opt = all_results["optimized"].get(n_rows, {})
            orig_us = orig.get("median_us", float("nan"))
            opt_us = opt.get("median_us", float("nan"))
            speedup = orig_us / opt_us if opt_us > 0 else float("nan")
            rows.append({
                "n_rows": n_rows,
                "original_median_us": f"{orig_us:.2f}",
                "optimized_median_us": f"{opt_us:.2f}",
                "speedup": f"{speedup:.2f}x",
            })
        df = pd.DataFrame(rows)
        print("\n" + df.to_markdown(index=False))
        print()


if __name__ == "__main__":
    main()
