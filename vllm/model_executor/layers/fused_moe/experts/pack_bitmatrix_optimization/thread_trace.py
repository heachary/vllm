#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Thread-level trace analysis for pack_bitmatrix kernels using rocprofv3 --att.

Captures instruction-level wavefront traces to analyze bottlenecks:
  - LDS stalls, barrier waits, VMEM latency
  - Instruction mix (VALU, SALU, LDS, VMEM)
  - Idle/stall cycle breakdown

Requirements:
  - rocprofv3 with ATT support
  - HSA_CU_MASK=0x1 to pin work to traced CU
  - rocprof-trace-decoder for CSV output

Usage:
    python thread_trace.py --variant original --n_rows 4
    python thread_trace.py --variant optimized --n_rows 4
"""

import argparse
import os
import subprocess
import sys
import tempfile


NUM_LOCAL_EXPERTS = 48
NUM_TOPK = 6

ORIG_BLOCK_SIZE_M = 512
ORIG_BLOCK_SIZE_K = 32
V2_BLOCK_SIZE_M = 256
V2_BLOCK_SIZE_K = 8


def run_kernel(variant, n_rows, reps):
    """Run the kernel under rocprofv3 --att for thread trace capture."""
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
        for _ in range(reps):
            pack_bitmatrix[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=ORIG_BLOCK_SIZE_M, BLOCK_SIZE_K=ORIG_BLOCK_SIZE_K,
            )
    else:
        from vllm.model_executor.layers.fused_moe.experts.pack_bitmatrix_optimization.pack_bitmatrix_v2 import (
            pack_bitmatrix_v2,
        )
        grid = (triton.cdiv(n_rows, V2_BLOCK_SIZE_M),)
        for _ in range(reps):
            pack_bitmatrix_v2[grid](
                bitmatrix, topk_ids, n_rows, bm_cols, NUM_TOPK,
                BLOCK_SIZE_M=V2_BLOCK_SIZE_M, BLOCK_SIZE_K=V2_BLOCK_SIZE_K,
            )

    torch.cuda.synchronize()
    print(f"DONE: {variant} n_rows={n_rows} reps={reps}")


def main():
    parser = argparse.ArgumentParser(
        description="Thread trace analysis for pack_bitmatrix kernels"
    )
    parser.add_argument("--variant", choices=["original", "optimized"],
                        default="original")
    parser.add_argument("--n_rows", type=int, default=4)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.inner:
        run_kernel(args.variant, args.n_rows, args.reps)
        return

    with tempfile.TemporaryDirectory(prefix="att_trace_") as tmpdir:
        out_prefix = os.path.join(tmpdir, f"{args.variant}_n{args.n_rows}")

        cmd = [
            "rocprofv3",
            "--att",
            "--att-target-cu", "0",
            "--att-se-mask", "0xFFFFFFFF",
            "--att-buffer-size", "0x4000000",
            "-o", out_prefix,
            "--", "python3", os.path.abspath(__file__),
            "--inner",
            "--variant", args.variant,
            "--n_rows", str(args.n_rows),
            "--reps", str(args.reps),
        ]

        env = os.environ.copy()
        env["HSA_TOOLS_DISABLE_REGISTER"] = "0"
        env["HSA_CU_MASK"] = "0x1"

        print(f"Running: {' '.join(cmd)}")
        print(f"Output prefix: {out_prefix}")

        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
        )

        print(f"Return code: {proc.returncode}")
        if proc.stdout:
            print(f"STDOUT:\n{proc.stdout}")
        if proc.stderr:
            print(f"STDERR (last 30 lines):")
            for line in proc.stderr.splitlines()[-30:]:
                print(f"  {line}")

        print(f"\nOutput files:")
        for f in sorted(os.listdir(tmpdir)):
            fpath = os.path.join(tmpdir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size} bytes)")


if __name__ == "__main__":
    main()
