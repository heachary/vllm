#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run test_matmul_ogs.py with the fixed parser (counts _reduce kernel too)
so baseline and optimized numbers are directly comparable."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_matmul_ogs


def _parse_trace_all_relevant(csv_path, rep):
    """Counts _matmul_ogs + _reduce kernels per invocation, sums them, returns median."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "Kernel_Name" not in df.columns:
        return [], 0.0
    kernel_rows = df[
        df["Kernel_Name"].str.contains(
            r"_matmul_ogs|_p_matmul_ogs|^_reduce$",
            case=False, na=False, regex=True,
        )
    ]
    if kernel_rows.empty:
        kernel_rows = df[~df["Kernel_Name"].str.contains(
            "memcpy|memset|fill|bitmatrix|ragged|_sum_|_stage",
            case=False, na=False,
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

_self_path = os.path.abspath(__file__)


def profile_one(M, call_site, warmup, rep, tmpdir):
    """Same as opt's profile_one but relaunches the *unpatched* runner."""
    import subprocess
    label = f"M{M}_{call_site}"
    tag = f"bench_{label}"
    trace_prefix = os.path.join(tmpdir, tag)
    trace_csv = f"{trace_prefix}_kernel_trace.csv"
    if os.path.exists(trace_csv):
        os.remove(trace_csv)
    cmd = [
        "rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace_prefix,
        "--", sys.executable, _self_path,
        "--inner", str(M), call_site,
        "--warmup", str(warmup), "--rep", str(rep),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.communicate()
    if proc.returncode != 0 or not os.path.isfile(trace_csv):
        return None
    kernel_names, median_us = _parse_trace_all_relevant(trace_csv, rep)
    return {
        "label": label, "M": M, "call_site": call_site,
        "time_us": median_us,
        "kernels": "; ".join(kernel_names[:5]),
    }


test_matmul_ogs.profile_one = profile_one


if __name__ == "__main__":
    # When relaunched as inner, just run the original (unpatched) inner.
    test_matmul_ogs.main()
