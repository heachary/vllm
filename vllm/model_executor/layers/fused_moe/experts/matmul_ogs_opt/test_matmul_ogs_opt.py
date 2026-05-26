#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper around test_matmul_ogs.py that installs `opt.py`'s monkey-patched
matmul_ogs before invoking the inner runner.

Usage is identical to test_matmul_ogs.py. Just replace `test_matmul_ogs.py`
with `test_matmul_ogs_opt.py`:

    python test_matmul_ogs_opt.py --warmup 25 --rep 100 -o opt_results.csv

The outer driver (rocprofv3 launcher) and the inner kernel runner both go
through opt.install() so the wrapper applies to every invocation.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Install the patch before any triton_kernels import.
sys.path.insert(0, HERE)
import opt  # noqa: F401  (side effect: monkey-patches matmul_ogs)

# Now run the original test as if it were `python test_matmul_ogs.py ...`.
# We need the inner subprocess to also load `opt.py`, so we pass our own
# script path in sys.argv[0] and rely on test_matmul_ogs.py using
# `os.path.abspath(__file__)` to relaunch -- which would re-launch the
# original. To work around that, we monkey-patch the path it relaunches.

import test_matmul_ogs

_orig_profile_one = test_matmul_ogs.profile_one
_orig_trace_one = test_matmul_ogs.trace_one
_self_path = os.path.abspath(__file__)
_inner_path = os.path.join(HERE, "test_matmul_ogs.py")


def _patch_cmd(cmd):
    """Rewrite the relaunch path so the inner subprocess imports opt.py."""
    new_cmd = []
    for token in cmd:
        if token == _inner_path:
            new_cmd.append(_self_path)
        else:
            new_cmd.append(token)
    return new_cmd


def _parse_trace_all_relevant(csv_path, rep):
    """Like test_matmul_ogs.parse_trace, but also includes the `_reduce`
    kernel that split_k emits, so that opt vs baseline comparisons are
    apples-to-apples."""
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


def profile_one(M, call_site, warmup, rep, tmpdir):
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
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"[ERROR] rocprofv3 failed for {label}:", file=sys.stderr)
        print(stderr[:2000], file=sys.stderr)
        return None
    if not os.path.isfile(trace_csv):
        print(f"[ERROR] trace CSV not found: {trace_csv}", file=sys.stderr)
        return None
    kernel_names, median_us = _parse_trace_all_relevant(trace_csv, rep)
    return {
        "label": label, "M": M, "call_site": call_site,
        "time_us": median_us,
        "kernels": "; ".join(kernel_names[:5]),
    }


def trace_one(M, call_site, warmup, rep, outdir, att_target_cu=0, kernel_include_regex=None):
    import subprocess
    label = f"M{M}_{call_site}"
    tag = f"att_{label}"
    config_dir = os.path.join(outdir, tag)
    os.makedirs(config_dir, exist_ok=True)
    att_consecutive = str(rep) if kernel_include_regex else "1"
    cmd = [
        "rocprofv3", "--advanced-thread-trace",
        "--att-target-cu", str(att_target_cu),
        "--att-consecutive-kernels", att_consecutive,
    ]
    if kernel_include_regex:
        cmd += ["--kernel-include-regex", kernel_include_regex]
    cmd += [
        "-d", config_dir, "--", sys.executable, _self_path,
        "--inner", str(M), call_site,
        "--warmup", str(warmup), "--rep", str(rep),
    ]
    print(f"  [{label}] running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


test_matmul_ogs.profile_one = profile_one
test_matmul_ogs.trace_one = trace_one


if __name__ == "__main__":
    test_matmul_ogs.main()
