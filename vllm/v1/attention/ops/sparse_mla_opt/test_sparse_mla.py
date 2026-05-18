#!/usr/bin/env python3
"""
Benchmark rocm_sparse_attn_decode using rocprofv3 kernel tracing for
accurate GPU-only timing (no Python dispatch overhead).

Shapes are derived from DeepSeek-V4-Pro serving with TP=8,
isl/osl=1024/1024, max_num_seqs=128, concurrency=4.

Usage:
    python test_sparse_mla.py
    python test_sparse_mla.py --warmup 50 --rep 200 -o results.csv
    python test_sparse_mla.py --batch 1 4 --mode padded ragged

Modes:
    (default)    Outer driver: spawns rocprofv3 per config, parses traces,
                 writes CSV.
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

# Representative test configurations: (batch, mode, topk_ragged_len, label)
# "padded" / CSA = topk_indices tensor present (latent cache [N,2,584])
# "ragged" / HCA = topk_indices=None, kv_cache is full-head [N,64,584]
# HCA always attends to all pages (1024*B). CSA sparse budget grows
# incrementally per decode step, reaching 1024*B at steady state.
TEST_CONFIGS = [
    # Steady-state (full context ~1024 sparse pages per request)
    (1, "padded", 1024, "B1_csa_full"),
    (1, "ragged", 1024, "B1_hca_full"),
    (4, "padded", 4096, "B4_csa_full"),
    (4, "ragged", 4096, "B4_hca_full"),
    # Early decode (few pages filled, CSA only — HCA always full)
    (1, "padded", 8,    "B1_csa_early"),
    (4, "padded", 32,   "B4_csa_early"),
    # Mid decode
    (1, "padded", 128,  "B1_csa_mid"),
    (4, "padded", 512,  "B4_csa_mid"),
    # Batch=2 steady
    (2, "padded", 2048, "B2_csa_full"),
    (2, "ragged", 2048, "B2_hca_full"),
    # Batch=3 steady
    (3, "padded", 3072, "B3_csa_full"),
    (3, "ragged", 3072, "B3_hca_full"),
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

    if mode == "padded":
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
        "sparse_attn|_paged_attn|triton_|_v1_partial_kernel|_v1_reduce_kernel",
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


def profile_one(batch, mode, topk_ragged_len, label, warmup, rep, tmpdir,
                impl="baseline"):
    """Run one config under rocprofv3 and return parsed results."""
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


def main():
    parser = argparse.ArgumentParser(description="Bench rocm_sparse_attn_decode")
    parser.add_argument("--inner", nargs=3, metavar=("BATCH", "MODE", "TOPK_RAGGED"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("-o", "--output", type=str, default="sparse_mla_results.csv")
    parser.add_argument("--batch", type=int, nargs="+", default=None,
                        help="Filter to specific batch sizes")
    parser.add_argument("--mode", type=str, nargs="+", default=None,
                        choices=["padded", "ragged"],
                        help="Filter to specific modes")
    parser.add_argument("--impl", type=str, default="baseline",
                        choices=["baseline", "v1"],
                        help="Kernel implementation to benchmark")
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

    csv_header = ["label", "batch", "mode", "topk_ragged_len", "time_us", "kernels"]
    rows = []

    print(f"\n{'='*90}")
    print(f"  rocm_sparse_attn_decode benchmark  (warmup={args.warmup}, rep={args.rep})")
    print(f"{'='*90}")
    print(f"{'Label':<22s} {'B':>3s} {'Mode':<8s} {'topk_rag':>10s} {'time_us':>10s}")
    print(f"{'-'*70}")

    with tempfile.TemporaryDirectory(prefix="bench_sparse_mla_") as tmpdir:
        for batch, mode, topk_ragged_len, label in configs:
            result = profile_one(
                batch, mode, topk_ragged_len, label,
                args.warmup, args.rep, tmpdir, impl=args.impl,
            )
            if result is None:
                print(f"{label:<22s} {batch:>3d} {mode:<8s} {topk_ragged_len:>10d}    FAILED")
                continue

            print(f"{label:<22s} {batch:>3d} {mode:<8s} "
                  f"{topk_ragged_len:>10d} {result['time_us']:>10.3f}")
            if result["kernels"]:
                print(f"  kernels: {result['kernels']}")

            rows.append([
                result["label"], result["batch"], result["mode"],
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
