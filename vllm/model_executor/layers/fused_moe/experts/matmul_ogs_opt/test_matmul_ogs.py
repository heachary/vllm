#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark matmul_ogs (W1 gate+up and W2 down projections) using rocprofv3
kernel tracing for accurate GPU-only timing.

Shapes are derived from DeepSeek-V4-Pro serving with TP=8,
--moe-backend triton_unfused.

Usage:
    python test_matmul_ogs.py
    python test_matmul_ogs.py --warmup 50 --rep 200 -o results.csv
    python test_matmul_ogs.py --num-tokens 1 4 8

    # Generate thread-level traces for rocprof viewer:
    python test_matmul_ogs.py --trace --trace-output-dir /tmp/moe_traces \
        --warmup 5 --rep 3

Modes:
    (default)    Outer driver: spawns rocprofv3 per config, parses traces,
                 writes CSV.
    --trace      Thread-level trace mode: spawns rocprofv3 with
                 --advanced-thread-trace (ATT) for per-instruction traces.
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
# DSV4-Pro TP=8 constants (from model config + ROCm MXFP4 rounding)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 7168  # K
INTERMEDIATE_PER_PARTITION = 512  # round_up(3072/8, 256)
N_W1 = 2 * INTERMEDIATE_PER_PARTITION  # 1024 (fused gate+up)
N_W2 = INTERMEDIATE_PER_PARTITION  # 512  (post-activation input to W2)
NUM_EXPERTS = 384  # all experts per rank (TP shards intermediate)
TOPK = 6
NUM_WARPS = 8

NUM_TOKENS_SWEEP = [1, 2, 4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Weight construction helpers
# ---------------------------------------------------------------------------


def make_mxfp4_weights(E, K, N, device="cuda:0"):
    """Create MXFP4-quantized weight tensor + scale in the layout expected
    by matmul_ogs on ROCm (column-major FP4 with swizzled scales)."""
    import torch
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout as tk_layout

    from vllm.utils.math_utils import round_up

    k_align, n_align = 256, 512
    K_pad = round_up(K, k_align)
    N_pad = round_up(N, n_align)

    w_bf16 = torch.randn((E, K_pad, N_pad), dtype=torch.bfloat16, device=device)

    w_raw, w_scale_raw = downcast_to_mxfp(w_bf16, torch.uint8, axis=1)

    w_layout, w_layout_opts = tk_layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w_scale_layout, w_scale_layout_opts = (
        tk_layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=NUM_WARPS
        )
    )

    w = convert_layout(wrap_torch_tensor(w_raw, FP4), w_layout, **w_layout_opts)
    w_scale = convert_layout(
        wrap_torch_tensor(w_scale_raw), w_scale_layout, **w_scale_layout_opts
    )
    return w, w_scale


def build_inputs(M, call_site, device="cuda:0"):
    """Construct all inputs for a single matmul_ogs call with garbage data."""
    import torch
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        PrecisionConfig,
    )
    from triton_kernels.numerics import InFlexData

    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        make_routing_data,
    )

    E = NUM_EXPERTS
    K = HIDDEN_SIZE
    topk = TOPK

    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:topk] for _ in range(M)]
    ).to(torch.int32)
    topk_weights = torch.rand((M, topk), dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    routing_data, gather_indx, scatter_indx = make_routing_data(
        topk_ids, topk_weights, E
    )

    if call_site == "W1":
        N = N_W1
        x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        w, w_scale = make_mxfp4_weights(E, K, N, device)
        y = torch.empty((1, M * topk, N), dtype=torch.bfloat16, device=device)
        pc = PrecisionConfig(
            weight_scale=w_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
        )
        return dict(
            x=x,
            w=w,
            bias=None,
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=None,
            precision_config=pc,
            gammas=None,
            y=y,
        )
    else:
        N = N_W2
        x = torch.randn((M * topk, N), dtype=torch.bfloat16, device=device)
        w, w_scale = make_mxfp4_weights(E, N, K, device)
        y = torch.empty((1, M * topk, K), dtype=torch.bfloat16, device=device)
        routing_data.n_expts_act = 1
        pc = PrecisionConfig(
            weight_scale=w_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
        )
        gammas = routing_data.gate_scal
        return dict(
            x=x,
            w=w,
            bias=None,
            routing_data=routing_data,
            gather_indx=None,
            scatter_indx=scatter_indx,
            precision_config=pc,
            gammas=gammas,
            y=y,
        )


# ---------------------------------------------------------------------------
# Kernel runner (called under rocprofv3)
# ---------------------------------------------------------------------------


def run_inner(M, call_site, warmup, rep):
    """Execute the kernel warmup+rep times (called under rocprofv3)."""
    import torch
    from triton_kernels.matmul_ogs import matmul_ogs

    inputs = build_inputs(M, call_site)

    for _ in range(warmup):
        matmul_ogs(
            inputs["x"],
            inputs["w"],
            inputs["bias"],
            inputs["routing_data"],
            gather_indx=inputs["gather_indx"],
            scatter_indx=inputs["scatter_indx"],
            precision_config=inputs["precision_config"],
            gammas=inputs["gammas"],
            y=inputs["y"],
        )
    torch.cuda.synchronize()

    for _ in range(rep):
        matmul_ogs(
            inputs["x"],
            inputs["w"],
            inputs["bias"],
            inputs["routing_data"],
            gather_indx=inputs["gather_indx"],
            scatter_indx=inputs["scatter_indx"],
            precision_config=inputs["precision_config"],
            gammas=inputs["gammas"],
            y=inputs["y"],
        )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------


def parse_trace(csv_path, rep):
    """Parse rocprofv3 kernel trace CSV -> median_us over the last `rep`
    invocations of each kernel, summed across all kernels per invocation."""
    df = pd.read_csv(csv_path)
    if "Kernel_Name" not in df.columns:
        print(f"[WARN] No Kernel_Name column in {csv_path}", file=sys.stderr)
        return [], 0.0

    kernel_rows = df[
        df["Kernel_Name"].str.contains(
            "_matmul_ogs|_p_matmul_ogs",
            case=False,
            na=False,
        )
    ]
    if kernel_rows.empty:
        kernel_rows = df[
            ~df["Kernel_Name"].str.contains(
                "memcpy|memset|fill|bitmatrix|ragged|_sum_|_stage",
                case=False,
                na=False,
            )
        ]

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


# ---------------------------------------------------------------------------
# Profiling driver
# ---------------------------------------------------------------------------


def profile_one(M, call_site, warmup, rep, tmpdir):
    """Run one config under rocprofv3 and return parsed results."""
    label = f"M{M}_{call_site}"
    tag = f"bench_{label}"
    trace_prefix = os.path.join(tmpdir, tag)
    trace_csv = f"{trace_prefix}_kernel_trace.csv"

    if os.path.exists(trace_csv):
        os.remove(trace_csv)

    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "-o",
        trace_prefix,
        "--",
        sys.executable,
        os.path.abspath(__file__),
        "--inner",
        str(M),
        call_site,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"[ERROR] rocprofv3 failed for {label}:", file=sys.stderr)
        print(stderr[:2000], file=sys.stderr)
        return None

    if not os.path.isfile(trace_csv):
        print(f"[ERROR] trace CSV not found: {trace_csv}", file=sys.stderr)
        return None

    kernel_names, median_us = parse_trace(trace_csv, rep)

    return {
        "label": label,
        "M": M,
        "call_site": call_site,
        "time_us": median_us,
        "kernels": "; ".join(kernel_names[:5]),
    }


def trace_one(
    M, call_site, warmup, rep, outdir, att_target_cu=0, kernel_include_regex=None
):
    """Run one config under rocprofv3 ATT and leave traces in outdir."""
    label = f"M{M}_{call_site}"
    tag = f"att_{label}"
    config_dir = os.path.join(outdir, tag)
    os.makedirs(config_dir, exist_ok=True)

    att_consecutive = str(rep) if kernel_include_regex else "1"
    cmd = [
        "rocprofv3",
        "--advanced-thread-trace",
        "--att-target-cu",
        str(att_target_cu),
        "--att-consecutive-kernels",
        att_consecutive,
    ]
    if kernel_include_regex:
        cmd += ["--kernel-include-regex", kernel_include_regex]
    cmd += [
        "-d",
        config_dir,
        "--",
        sys.executable,
        os.path.abspath(__file__),
        "--inner",
        str(M),
        call_site,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
    ]

    print(f"  [{label}] running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"  [{label}] FAILED (rc={proc.returncode})", file=sys.stderr)
        print(stderr[:2000], file=sys.stderr)
        return False

    entries = os.listdir(config_dir)
    ui_dirs = [e for e in entries if e.startswith("ui_output_")]
    print(f"  [{label}] done - {len(entries)} item(s) in {config_dir}/")
    for d in sorted(ui_dirs):
        print(f"    {d}/  (viewer dir)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bench matmul_ogs for DSV4-Pro MoE experts"
    )
    parser.add_argument(
        "--inner",
        nargs=2,
        metavar=("M", "CALL_SITE"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="matmul_ogs_results.csv",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help=f"Token counts to benchmark (default: {NUM_TOKENS_SWEEP})",
    )
    parser.add_argument(
        "--call-site",
        type=str,
        nargs="+",
        default=["W1", "W2"],
        choices=["W1", "W2"],
        help="Which matmul_ogs call to benchmark (default: both)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Generate thread-level traces (ATT) instead of kernel traces.",
    )
    parser.add_argument("--trace-output-dir", type=str, default=None)
    parser.add_argument("--att-target-cu", type=int, default=0)
    parser.add_argument("--kernel-include-regex", type=str, default=None)
    args = parser.parse_args()

    # ── Inner mode (called under rocprofv3) ────────────────────────────
    if args.inner:
        M = int(args.inner[0])
        call_site = args.inner[1]
        run_inner(M, call_site, args.warmup, args.rep)
        return

    token_counts = args.num_tokens or NUM_TOKENS_SWEEP
    configs = [(M, cs) for M in token_counts for cs in args.call_site]

    # ── ATT trace mode ─────────────────────────────────────────────────
    if args.trace:
        outdir = args.trace_output_dir or tempfile.mkdtemp(prefix="att_matmul_ogs_")
        os.makedirs(outdir, exist_ok=True)

        kernel_regex = args.kernel_include_regex
        if kernel_regex is None:
            kernel_regex = "_matmul_ogs|_p_matmul_ogs"

        print(f"\n{'=' * 80}")
        print(
            f"  ATT traces  (warmup={args.warmup}, rep={args.rep}, "
            f"target_cu={args.att_target_cu})"
        )
        print(f"  kernel filter: {kernel_regex}")
        print(f"  output dir: {outdir}")
        print(f"{'=' * 80}\n")

        ok = 0
        for M, call_site in configs:
            if trace_one(
                M,
                call_site,
                args.warmup,
                args.rep,
                outdir,
                args.att_target_cu,
                kernel_regex,
            ):
                ok += 1
        print(f"\n{ok}/{len(configs)} configs traced.")
        print(f"Trace files in: {outdir}")
        return

    # ── Normal kernel-trace benchmark mode ─────────────────────────────
    csv_header = [
        "M",
        "call_site",
        "time_us",
        "K",
        "N",
        "E",
        "topk",
        "kernels",
    ]
    rows = []

    print(f"\n{'=' * 80}")
    print(f"  matmul_ogs benchmark  (warmup={args.warmup}, rep={args.rep})")
    print(
        f"  DSV4-Pro TP=8: K={HIDDEN_SIZE}, I_part={INTERMEDIATE_PER_PARTITION}, "
        f"E={NUM_EXPERTS}, topk={TOPK}"
    )
    print(f"{'=' * 80}")
    print(f"{'M':>5s} {'site':<4s} {'time_us':>10s}  {'N':>5s}  kernels")
    print(f"{'-' * 60}")

    with tempfile.TemporaryDirectory(prefix="bench_matmul_ogs_") as tmpdir:
        for M, call_site in configs:
            result = profile_one(M, call_site, args.warmup, args.rep, tmpdir)
            if result is None:
                print(f"{M:>5d} {call_site:<4s}     FAILED")
                continue

            N_eff = N_W1 if call_site == "W1" else HIDDEN_SIZE
            print(
                f"{M:>5d} {call_site:<4s} {result['time_us']:>10.3f}  "
                f"{N_eff:>5d}  {result['kernels']}"
            )

            rows.append(
                [
                    M,
                    call_site,
                    f"{result['time_us']:.3f}",
                    HIDDEN_SIZE,
                    N_W1 if call_site == "W1" else N_W2,
                    NUM_EXPERTS,
                    TOPK,
                    result["kernels"],
                ]
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(rows)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
