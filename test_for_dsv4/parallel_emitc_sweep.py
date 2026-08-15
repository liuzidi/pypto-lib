#!/usr/bin/env python3
"""Parallel EmitC sweep: distribute kernels across multiple NPU devices.

EmitC route doesn't use the tilelib daemon, so parallel execution across
devices is safe (unlike VMI route which shares /tmp/tilelib_daemon_*.sock).

Uses ThreadPoolExecutor — subprocess calls release the GIL, so real
parallelism is achieved even under Python's GIL.

Usage:
    source ../scripts/vpto_env.sh
    python3 parallel_emitc_sweep.py --devices 2,5,6,7
    python3 parallel_emitc_sweep.py --devices 7 --kernel kv_proj_seed
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
VPTO_ENV_SH = REPO_ROOT / "scripts" / "vpto_env.sh"

sys.path.insert(0, str(ROOT))
from run_all import find_kernels, run_one_kernel, RESULT_FIELDS


def main():
    ap = argparse.ArgumentParser(description="Parallel EmitC sweep across NPU devices.")
    ap.add_argument("-d", "--devices", default="2,5,6,7",
                    help="Comma-separated NPU device ids")
    ap.add_argument("--kernel", default=None, help="Run only this kernel")
    ap.add_argument("--timeout", type=int, default=300, help="Per-kernel timeout (s)")
    args = ap.parse_args()

    devices = [int(d.strip()) for d in args.devices.split(",")]
    kernels = find_kernels()
    if args.kernel:
        kernels = [k for k in kernels if k == args.kernel]
    if not kernels:
        print("No kernels found.", file=sys.stderr)
        return 2

    # Source env once (run_one_kernel uses os.environ)
    r = subprocess.run(
        ["bash", "-c", f"source {VPTO_ENV_SH} 2>/dev/null && env"],
        capture_output=True, text=True, timeout=30)
    for line in r.stdout.split("\n"):
        if "=" in line and not line.startswith("[vpto_env"):
            k, v = line.split("=", 1)
            os.environ[k] = v

    # Round-robin distribute kernels across devices
    task_list = []
    for i, k in enumerate(kernels):
        d = devices[i % len(devices)]
        task_list.append((k, d))

    print(f"EmitC parallel sweep: {len(kernels)} kernels across devices {devices}")
    for d in devices:
        n = sum(1 for _, dd in task_list if dd == d)
        print(f"  device {d}: {n} kernels")
    print()

    results = {}
    lock = threading.Lock()
    done_count = [0]
    t0 = time.time()

    def run_task(kernel, device):
        res = run_one_kernel(kernel, device, "emitc", args.timeout)
        with lock:
            results[kernel] = res
            done_count[0] += 1
            dc = done_count[0]
            n_pass = sum(1 for r in results.values() if r.get("pass"))
            n_fail = dc - n_pass
            elapsed = time.time() - t0
            status = "PASS" if res.get("pass") else "FAIL"
            md = res.get("max_diff", "")
            err = (res.get("error") or "")[:60]
            print(f"  [{dc}/{len(kernels)}] {elapsed:.0f}s dev{device} "
                  f"{kernel:<40} {status}"
                  + (f" max_diff={md}" if md not in ("", None) else "")
                  + (f" {err}" if err else ""),
                  flush=True)
        return res

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futures = [ex.submit(run_task, k, d) for k, d in task_list]
        for f in as_completed(futures):
            f.result()

    # Write merged CSV in kernel order
    merged_csv = ROOT / "emitc_sweep_results.csv"
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS + ["elapsed_s"])
        w.writeheader()
        for k in kernels:
            r = results.get(k, {"kernel": k, "pass": False,
                                "compare_status": "not_run",
                                "exit_code": -1, "error": "not executed"})
            w.writerow({fn: r.get(fn, "") for fn in RESULT_FIELDS + ["elapsed_s"]})

    # Summary
    n_pass = sum(1 for r in results.values() if r.get("pass"))
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"EmitC sweep complete: {n_pass}/{len(results)} PASS, {n_fail} FAIL")
    print(f"Results: {merged_csv}")

    cats = Counter(r.get("compare_status", "?") for r in results.values())
    print("\nCategory breakdown:")
    for cat, cnt in cats.most_common():
        print(f"  {cat:<25} {cnt}")

    print("\nPer-kernel results:")
    for k in kernels:
        r = results.get(k, {})
        status = "PASS" if r.get("pass") else "FAIL"
        line = f"  {k:<40} {status}"
        if r.get("max_diff") and r["max_diff"] != "":
            line += f" (max_diff={r['max_diff']})"
        if r.get("error"):
            line += f" — {r['error'][:50]}"
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
