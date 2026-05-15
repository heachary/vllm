# SPDX-License-Identifier: Apache-2.0
"""
Performance test for the pack_bitmatrix Triton kernel used in MoE routing.

Shapes confirmed from profiling DeepSeek-V4-Pro (TP=8) serving traces
(record_shapes=true):
  - 384 global routed experts, 48 local experts per rank (384/8)
  - top-k = 6 activated experts per token
  - bm_cols = cdiv(48, 32) = 2
  - Decode:  n_rows = 1   (single token per step observed in trace)
             topk_ids (1, 6) int16, bitmatrix (1, 2) uint32
  - Prefill: n_rows = 256 (chunked) or 1024 (full context)
             topk_ids (256, 6) int16, bitmatrix (256, 2) uint32

Usage:
    python test_pack_bitmatrix_perf.py
    python test_pack_bitmatrix_perf.py --n_rows 1 2 4 8 16 128 1024
"""

import argparse
import itertools

import pandas as pd
import torch
import triton

from aiter.test_common import run_perftest

torch.set_default_device("cuda")

NUM_LOCAL_EXPERTS = 48
NUM_TOPK = 6
BLOCK_SIZE_M = 512
BLOCK_SIZE_K = 32


def make_inputs(n_rows: int, num_local_experts: int, num_topk: int):
    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    topk_ids = torch.randint(
        0, num_local_experts, (n_rows, num_topk), dtype=torch.int16
    )
    bitmatrix = torch.zeros((n_rows, bm_cols), dtype=torch.uint32)
    return bitmatrix, topk_ids, bm_cols


def run_pack_bitmatrix(bitmatrix, topk_ids, n_rows, bm_cols, num_topk):
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
        pack_bitmatrix,
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


def bench_pack_bitmatrix(n_rows, num_local_experts, num_topk, num_iters):
    bitmatrix, topk_ids, bm_cols = make_inputs(
        n_rows, num_local_experts, num_topk
    )
    _, us = run_perftest(
        run_pack_bitmatrix,
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        num_iters=num_iters,
    )
    return us


def main():
    parser = argparse.ArgumentParser(
        description="Perf test for the pack_bitmatrix Triton kernel",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4],
        help="Batch sizes (n_rows) to benchmark",
    )
    parser.add_argument(
        "--num_local_experts",
        type=int,
        nargs="*",
        default=[NUM_LOCAL_EXPERTS],
        help="Number of local experts (default: 48 for DSV4-Pro TP=8)",
    )
    parser.add_argument(
        "--num_topk",
        type=int,
        nargs="*",
        default=[NUM_TOPK],
        help="Top-k experts per token (default: 6 for DSV4-Pro)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Number of iterations for timing",
    )
    args = parser.parse_args()

    rows = []
    for n_rows, n_experts, topk in itertools.product(
        args.n_rows, args.num_local_experts, args.num_topk
    ):
        bm_cols = triton.cdiv(n_experts, BLOCK_SIZE_K)
        grid = triton.cdiv(n_rows, BLOCK_SIZE_M)
        row = {
            "n_rows": n_rows,
            "num_local_experts": n_experts,
            "topk": topk,
            "bm_cols": bm_cols,
            "grid": grid,
            "bitmatrix": f"({n_rows}, {bm_cols}) uint32",
            "topk_ids": f"({n_rows}, {topk}) int16",
        }

        us_kernel = bench_pack_bitmatrix(
            n_rows, n_experts, topk, args.num_iters
        )
        row["pack_bitmatrix_us"] = f"{us_kernel:.2f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
