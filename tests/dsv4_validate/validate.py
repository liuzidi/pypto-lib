# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSV4 VPTO sweep framework: run all runnable DSV4 VPTO kernels on a real A5
NPU with one command and report precision/performance per kernel.

Reads baselines/vpto_dsv4_vector/classification.csv, filters rows where
is_leaf_strict is True (these are the kernels whose .pto ptr-args map 1:1 to
the owning module's TensorSpecs, so the run_jit golden_fn can produce their
inputs/outputs directly), and invokes the vpto-board-validate skill's
vpto_run.py on each — serially on a single card.

Per kernel, the skill writes run_dir/result.json (a structured sidecar); the
framework reads it and aggregates into sweep_results.csv. Crashes, ptoas
failures, or bisheng failures are recorded as findings and the sweep
continues — they are never fatal.

Non-leaf kernels (the ~100 inner kernels of multi-kernel modules whose
ptr-args are intermediates) are skipped automatically; they need Phase 5
intermediate capture, which is out of scope here.

Usage:
    python tests/dsv4_validate/validate.py -p a5 -d 0 [--mode decode]
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_RUN = REPO_ROOT / ".claude" / "skills" / "vpto-board-validate" / "vpto_run.py"
CLASSIFICATION_CSV = (
    REPO_ROOT / "baselines" / "vpto_dsv4_vector" / "classification.csv")
SWEEP_RESULTS_CSV = (
    REPO_ROOT / "baselines" / "vpto_dsv4_vector" / "sweep_results.csv")
VPTO_ENV_SH = REPO_ROOT / "scripts" / "vpto_env.sh"

RESULT_FIELDS = [
    "kernel", "module", "route", "mode", "device", "pass", "compare_status",
    "exit_code", "max_diff", "n_over", "n_total", "threshold", "timing_ms",
    "rtol", "atol", "fatobj_bytes", "has_kernel_sym", "has_ctor",
    "nested_elf_offsets", "error",
]


def load_strict_leaves(csv_path: Path, mode: str) -> list[dict]:
    """Return classification rows where is_leaf_strict is True.

    Prefill-only modules (prefill_*) are skipped when mode != prefill, and
    vice versa, so --mode decode does not attempt prefill leaves and vice
    versa. This keeps the sweep to a single batch-shape contract per run.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("is_leaf_strict") != "True":
                continue
            module = r.get("module", "")
            is_prefill = module.startswith("prefill_")
            if mode == "decode" and is_prefill:
                continue
            if mode == "prefill" and not is_prefill:
                continue
            rows.append(r)
    return rows


def run_one_leaf(row: dict, mode: str, device: int) -> dict:
    """Invoke vpto_run.py for one leaf with vpto_env.sh sourced.

    Returns a result dict (schema = RESULT_FIELDS). On any subprocess error,
    returns a row with pass=False and the error filled in — never raises.
    """
    kernel = row["kernel"]
    pto_path = REPO_ROOT / row["pto_path"]
    model_py = REPO_ROOT / row["model_py"]
    build_root = REPO_ROOT / "build_output" / f"vpto_{kernel}"
    result_json = build_root / "run" / "result.json"

    cmd = [
        "bash", "-c",
        f"source {VPTO_ENV_SH} && exec python3 {SKILL_RUN} "
        f"--pto {pto_path} --model-py {model_py} "
        f"--mode {mode} --device {device}",
    ]
    print(f"\n[validate] {kernel}: running ...", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    except Exception as e:  # noqa: BLE001 — sweep must not abort
        return _error_row(row, mode, device, f"subprocess launch failed: {e}")
    # print the skill's tail so the user sees ptoas/bisheng/NPU/compare lines
    if r.stdout:
        print(r.stdout.rstrip()[-1200:])
    if r.returncode != 0 and r.stderr:
        print(r.stderr.rstrip()[-600:], file=sys.stderr)

    if not result_json.exists():
        # vpto_run.py exited before writing the sidecar (e.g. ptoas/bisheng
        # crash under `check=True` → sys.exit(1)). The exit code still tells
        # us it failed.
        return _error_row(
            row, mode, device,
            f"vpto_run.py exited {r.returncode} without writing result.json "
            f"(likely ptoas/bisheng/parse failure; see stderr above)")

    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return _error_row(row, mode, device, f"failed to parse result.json: {e}")

    out = {f: result.get(f, "") if not isinstance(result.get(f), bool)
           else result.get(f) for f in RESULT_FIELDS}
    out["kernel"] = kernel
    out["module"] = row.get("module", "")
    out["route"] = "vpto"
    out["mode"] = mode
    out["device"] = device
    return out


def _error_row(row: dict, mode: str, device: int, error: str) -> dict:
    out = {f: "" for f in RESULT_FIELDS}
    out.update({
        "kernel": row.get("kernel", ""),
        "module": row.get("module", ""),
        "route": "vpto",
        "mode": mode,
        "device": device,
        "pass": False,
        "compare_status": "crash",
        "exit_code": -1,
        "error": error,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DSV4 VPTO sweep: run all runnable VPTO kernels on A5 "
                    "and report precision/performance per kernel.")
    ap.add_argument("-p", "--platform", default="a5", choices=["a5"],
                    help="target platform (only a5 supported; Route 2 is "
                         "A5-only). Default: a5.")
    ap.add_argument("-d", "--device", type=int, default=0,
                    help="NPU device id. Single card, serial. Default: 0.")
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill"],
                    help="batch-shape contract to use. Default: decode.")
    args = ap.parse_args()

    if not CLASSIFICATION_CSV.exists():
        print(f"[validate] ERROR: {CLASSIFICATION_CSV} not found. "
              f"Run the classification generator first.", file=sys.stderr)
        return 2
    if not VPTO_ENV_SH.exists():
        print(f"[validate] ERROR: {VPTO_ENV_SH} not found.", file=sys.stderr)
        return 2

    leaves = load_strict_leaves(CLASSIFICATION_CSV, args.mode)
    print(f"[validate] platform={args.platform} device={args.device} "
          f"mode={args.mode}")
    print(f"[validate] {len(leaves)} strict-leaf kernel(s) to sweep "
          f"(non-leaf kernels skipped — Phase 5 deferred)")

    results = []
    for i, row in enumerate(leaves, 1):
        print(f"\n[validate] === [{i}/{len(leaves)}] {row['kernel']} "
              f"(module={row.get('module','')}) ===", flush=True)
        res = run_one_leaf(row, args.mode, args.device)
        results.append(res)
        status = "PASS" if res.get("pass") else "FAIL"
        md = res.get("max_diff", "")
        t = res.get("timing_ms", "")
        parts = [f"[validate] {res['kernel']}: {status}"]
        if md not in ("", None):
            parts.append(f"max_diff={md}")
        if t not in ("", None):
            parts.append(f"timing={t}ms")
        if res.get("error"):
            parts.append(f"err={res['error'][:80]}")
        print(" ".join(parts), flush=True)

    # write aggregated CSV
    SWEEP_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)

    # stdout summary
    n_pass = sum(1 for r in results if r.get("pass"))
    n_total = len(results)
    print(f"\n[validate] ===== sweep summary =====")
    print(f"[validate] route: VPTO (Route 2)")
    print(f"[validate] {n_pass}/{n_total} leaves PASS; "
          f"{n_total - n_pass} FAIL")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        line = f"[validate]   {r['kernel']:<20} {status}"
        if r.get("max_diff") not in ("", None):
            line += f" (max_diff={r['max_diff']})"
        if r.get("timing_ms") not in ("", None):
            line += f" [{r['timing_ms']}ms]"
        if r.get("error"):
            line += f" — {r['error'][:70]}"
        print(line)
    print(f"[validate] non-leaf kernels skipped (deferred Phase 5)")
    print(f"[validate] results: {SWEEP_RESULTS_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
