#!/usr/bin/env python3
"""Run all DSV4 VPTO kernels through the VMI+membar+vf-off route.

Uses vpto_run.py --golden-lib mode: vpto_run.py internally generates
main.cpp + launch.cpp + runs ptoas → bisheng → NPU → compare.
The golden data comes from dsv4_golden_lib.py (CPU torch, no GM dump).

Usage:
    source scripts/vpto_env.sh
    python3 run_all.py --device 7
    python3 run_all.py --device 7 --kernel rms_norm   # single kernel
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
VPTO_ENV_SH = REPO_ROOT / "scripts" / "vpto_env.sh"
VPTO_RUN = REPO_ROOT / ".claude" / "skills" / "vpto-board-validate" / "vpto_run.py"
GOLDEN_LIB = ROOT / "dsv4_golden_lib.py"

RESULT_FIELDS = [
    "kernel", "pass", "compare_status", "exit_code", "max_diff",
    "n_over", "n_total", "threshold", "timing_ms", "fatobj_bytes",
    "error",
]


def find_kernels() -> list:
    """Find all .pto files, expanding split kernels (_aic/_aiv) into two entries."""
    pto_dir = ROOT / ".pto"
    kernels = []
    for p in sorted(pto_dir.glob("*.pto")):
        text = p.read_text()
        func_names = re.findall(r'func\.func\s+@(\w+)', text)
        if len(func_names) > 1 and any(fn.endswith("_aic") for fn in func_names):
            # Split kernel (_aic/_aiv): unsupported, skip both variants
            print(f"  [skip] {p.stem}: split kernel (_aic/_aiv) — unsupported")
            continue
        else:
            kernels.append(p.stem)
    return kernels


def run_one_kernel(kernel: str, device: int, route: str, timeout: int = 300) -> dict:
    """Run one kernel via vpto_run.py --golden-lib (VPTO) or emitc_run.py (EmitC)."""
    if route == "emitc":
        # EmitC route: use standalone emitc_run.py
        import sys
        sys.path.insert(0, str(ROOT))
        import emitc_run
        # Source env first
        env_cmd = f"source {VPTO_ENV_SH} 2>/dev/null"
        r = subprocess.run(["bash", "-c", env_cmd + " && env"],
                          capture_output=True, text=True, timeout=30)
        for line in r.stdout.split("\n"):
            if "=" in line and not line.startswith("[vpto_env"):
                k, v = line.split("=", 1)
                os.environ[k] = v
        return emitc_run.run_emitc_kernel(kernel, device, timeout)
    # Resolve .pto path: for split kernels (qk_pv_aic), the .pto file is qk_pv.pto
    pto_path = ROOT / ".pto" / f"{kernel}.pto"
    if not pto_path.exists():
        # Try stripping _aic/_aiv suffix
        base = re.sub(r'_(aic|aiv)$', '', kernel)
        pto_path = ROOT / ".pto" / f"{base}.pto"
    build_dir = REPO_ROOT / "build_output" / f"vpto_{kernel}"

    cmd = [
        "bash", "-c",
        f"source {VPTO_ENV_SH} 2>/dev/null; "
        f"exec python3 {VPTO_RUN} "
        f"--pto {pto_path} --golden-lib {GOLDEN_LIB} "
        f"--kernel {kernel} --mode decode --device {device} "
        f"--route {route}"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"kernel": kernel, "pass": False, "compare_status": "timeout",
                "exit_code": -1, "error": "timed out (>300s)"}
    except Exception as e:
        return {"kernel": kernel, "pass": False, "compare_status": "crash",
                "exit_code": -1, "error": f"launch failed: {e}"}

    # Print tail
    out = r.stdout or ""
    err = r.stderr or ""
    combined = out + "\n" + err
    if out.strip():
        for line in out.strip().split("\n")[-8:]:
            print(f"  {line}")

    # Try to read result.json
    result_json = build_dir / "run" / "result.json"
    if result_json.exists():
        try:
            res = json.loads(result_json.read_text())
            res["kernel"] = kernel
            return {f: res.get(f, "") for f in RESULT_FIELDS}
        except Exception:
            pass

    # Fallback: parse from stdout/stderr
    if "compare passed" in combined:
        m = re.search(r"max_diff=([\d.eE+-]+)", combined)
        max_diff = float(m.group(1)) if m else 0.0
        m2 = re.search(r"\[TIMING\] kernel=\S+ time=([\d.]+)\s*ms", combined)
        timing = float(m2.group(1)) if m2 else None
        return {"kernel": kernel, "pass": True, "compare_status": "pass",
                "exit_code": 0, "max_diff": max_diff, "timing_ms": timing, "error": ""}
    elif "compare failed" in combined:
        m = re.search(r"max_diff=([\d.eE+-]+)", combined)
        return {"kernel": kernel, "pass": False, "compare_status": "fail",
                "exit_code": r.returncode, "max_diff": float(m.group(1)) if m else None,
                "error": "golden compare failed"}

    # Crash: extract error
    err_line = ""
    for pattern in ["NoMatchingTemplate", "error:", "AICore", "exception",
                    "not aligned", "FAILED step", "Error:", "exited"]:
        if pattern in combined:
            for line in combined.split("\n"):
                if pattern in line:
                    err_line = line.strip()[:120]
                    break
            break
    if not err_line:
        err_line = combined.strip().split("\n")[-1][:120] if combined.strip() else f"exit {r.returncode}"
    return {"kernel": kernel, "pass": False, "compare_status": "crash",
            "exit_code": r.returncode, "error": err_line}


def main():
    ap = argparse.ArgumentParser(description="Run all DSV4 VPTO kernels.")
    ap.add_argument("-d", "--device", type=int, default=0, help="NPU device id")
    ap.add_argument("--kernel", default=None, help="Run only this kernel")
    ap.add_argument("--route", default="baseline",
                    choices=["baseline", "vmi-membar-vfoff", "emitc"],
                    help="Codegen route: baseline (VPTO VF-on), vmi-membar-vfoff (VMI+membar+VF-off), emitc (simpler/g++)")
    ap.add_argument("--timeout", type=int, default=300, help="Per-kernel timeout (s)")
    args = ap.parse_args()

    if not VPTO_ENV_SH.exists():
        print(f"ERROR: {VPTO_ENV_SH} not found", file=sys.stderr)
        return 2
    if not GOLDEN_LIB.exists():
        print(f"ERROR: {GOLDEN_LIB} not found. Write it first.", file=sys.stderr)
        return 2

    kernels = find_kernels()
    if args.kernel:
        kernels = [k for k in kernels if k == args.kernel]
    if not kernels:
        print("No kernels found.", file=sys.stderr)
        return 2

    print(f"Found {len(kernels)} kernels to run on device {args.device}, route={args.route}")
    results = []
    for i, kname in enumerate(kernels, 1):
        print(f"\n[{i}/{len(kernels)}] === {kname} ===")
        t0 = time.time()
        res = run_one_kernel(kname, args.device, args.route, args.timeout)
        res["elapsed_s"] = round(time.time() - t0, 1)
        results.append(res)
        status = "PASS" if res.get("pass") else "FAIL"
        md = res.get("max_diff", "")
        err = (res.get("error") or "")[:60]
        print(f"  -> {status}", f"max_diff={md}" if md not in ("", None) else "", err)

    # Write CSV
    out_csv = ROOT / "sweep_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS + ["elapsed_s"])
        w.writeheader()
        w.writerows(results)

    # Summary
    n_pass = sum(1 for r in results if r.get("pass"))
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"Sweep complete: {n_pass}/{len(results)} PASS, {n_fail} FAIL")
    print(f"Results: {out_csv}")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        line = f"  {r['kernel']:<35} {status}"
        if r.get("max_diff") is not None:
            line += f" (max_diff={r['max_diff']})"
        if r.get("timing_ms") is not None:
            line += f" [{r['timing_ms']}ms]"
        if r.get("error"):
            line += f" — {r['error'][:50]}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
