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
import os
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


def run_phase5_module(module: str, model_py: Path, mode: str, device: int) -> list:
    """Phase 5: capture one module's intermediate GM buffers (Route 1 run with
    enable_dump_args=2), then replay each inner kernel on Route 2 (VPTO) using
    the captured buffers and compare.

    Returns a list of result rows (schema = RESULT_FIELDS). Crashes (ptoas
    lowering, bisheng link, NPU run) are recorded as findings and the sweep
    continues — never aborts.
    """
    import json
    # locate capture.py + the skill
    capture_py = REPO_ROOT / "tests" / "dsv4_validate" / "capture.py"
    vpto_run = (REPO_ROOT / ".claude" / "skills" /
                "vpto-board-validate" / "vpto_run.py")
    venv_py = REPO_ROOT / ".venv" / "bin" / "python3"

    # Step 1: run the module once with full args dump (Route 1 capture).
    # The dump work_dir carries ptoas/*.pto + dfx_outputs/{args_dump,name_map}.
    dump_wd = (REPO_ROOT / "build_output" / f"phase5_dump_{model_py.stem}")
    sys.path.insert(0, str(REPO_ROOT / "tests" / "dsv4_validate"))
    import capture as _capture
    if not (dump_wd / "dfx_outputs" / "args_dump" / "args_dump.json").exists():
        print(f"[validate] [phase5] running module {module} with dump...",
              flush=True)
        mod = _capture._load_module(model_py)
        # set the PTOAS_ROOT env so pypto can find ptoas for compilation
        env = dict(os.environ)
        env["PTOAS_ROOT"] = "/data/liuzidi/PTOAS/build311/tools/ptoas"
        old_environ = os.environ
        os.environ.clear(); os.environ.update(env)
        try:
            entry = _DSV4_MODULE_TO_JIT_ENTRY.get(module)
            dump_wd = _capture._run_module_with_dump(
                mod, model_py, mode, device, dump_wd, jit_entry=entry)
        except Exception as e:  # noqa: BLE001
            os.environ.clear(); os.environ.update(old_environ)
            return [_phase5_error_row(module, model_py, mode, device,
                                      f"capture (run_jit) failed: {e}")]
        os.environ.clear(); os.environ.update(old_environ)
    print(f"[validate] [phase5] dump at {dump_wd}", flush=True)

    # Step 2: read name_map to get the kernel list (callable_id -> name).
    nm_glob = list((dump_wd / "dfx_outputs").glob("name_map_*.json"))
    if not nm_glob:
        return [_phase5_error_row(module, model_py, mode, device,
                                  "no name_map_*.json in dump")]
    name_map = json.loads(nm_glob[0].read_text(encoding="utf-8"))
    cid2name = name_map["callable_id_to_name"]
    # dedupe kernel names (a kernel may have multiple task instances)
    kernel_names = sorted(set(cid2name.values()))
    # map kernel name -> .pto file (basename usually matches, but not always;
    # e.g. qk_pv_aic/qk_pv_aiv share qk_pv.pto). Try exact, then prefix.
    ptoas_dir = dump_wd / "ptoas"
    if not ptoas_dir.is_dir():
        ptoas_dir = dump_wd / "kernels"  # fallback (older layout)
    pto_files = {p.stem: p for p in ptoas_dir.glob("*.pto")}
    print(f"[validate] [phase5] {len(kernel_names)} distinct kernels in "
          f"name_map; {len(pto_files)} .pto files", flush=True)

    results = []
    for kname in kernel_names:
        pto = pto_files.get(kname)
        if pto is None:
            # prefix match (qk_pv_aic -> qk_pv.pto); take the longest stem
            # that is a prefix of kname to avoid false short-prefix matches.
            cands = [(s, p) for s, p in pto_files.items() if kname.startswith(s)]
            if not cands:
                results.append(_phase5_error_row(
                    module, model_py, mode, device, kname,
                    f"no .pto for kernel {kname!r} in {ptoas_dir}"))
                continue
            cands.sort(key=lambda sp: -len(sp[0]))
            pto = cands[0][1]
        print(f"\n[validate] [phase5] === {kname} ({pto.name}) ===", flush=True)
        # harvest this kernel's buffers into a fresh run_dir
        run_dir = REPO_ROOT / "build_output" / f"vpto_{kname}" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        # clear stale bins
        for old in run_dir.glob("*.bin"):
            old.unlink()
        if (run_dir / "capture_meta.json").exists():
            (run_dir / "capture_meta.json").unlink()
        try:
            meta = _capture.harvest_kernel(dump_wd, dump_wd, kname, run_dir, pto)
            (run_dir / "capture_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            results.append(_phase5_error_row(
                module, model_py, mode, device, kname,
                f"harvest failed: {e}"))
            continue
        # replay via vpto_run.py --captured-dump
        cmd = [
            "bash", "-c",
            f"source {VPTO_ENV_SH} && exec {venv_py} {vpto_run} "
            f"--pto {pto} --model-py {model_py} --mode {mode} "
            f"--device {device} --kernel {kname} "
            f"--captured-dump {run_dir}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(REPO_ROOT))
        except Exception as e:  # noqa: BLE001
            results.append(_phase5_error_row(
                module, model_py, mode, device, kname,
                f"subprocess failed: {e}"))
            continue
        if r.stdout:
            print(r.stdout.rstrip()[-1000:])
        result_json = run_dir / "result.json"
        if not result_json.exists():
            results.append(_phase5_error_row(
                module, model_py, mode, device, kname,
                f"vpto_run exited {r.returncode} without result.json "
                f"(likely ptoas/bisheng lowering failure)"))
            continue
        try:
            res = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            results.append(_phase5_error_row(
                module, model_py, mode, device, kname,
                f"result.json parse failed: {e}"))
            continue
        out = {f: res.get(f, "") if not isinstance(res.get(f), bool)
               else res.get(f) for f in RESULT_FIELDS}
        out["kernel"] = kname
        out["module"] = module
        out["route"] = "vpto(phase5)"
        out["mode"] = mode
        out["device"] = device
        results.append(out)
        status = "PASS" if out.get("pass") else "FAIL"
        md = out.get("max_diff", "")
        print(f"[validate] [phase5] {kname}: {status}"
              f"{f' max_diff={md}' if md not in ('', None) else ''}",
              flush=True)
    return results


def _phase5_error_row(module: str, model_py: Path, mode: str, device: int,
                      kernel: str, error: str) -> dict:
    out = {f: "" for f in RESULT_FIELDS}
    out.update({
        "kernel": kernel, "module": module, "route": "vpto(phase5)",
        "mode": mode, "device": device, "pass": False,
        "compare_status": "crash", "exit_code": -1, "error": error,
    })
    return out


# DSV4 module -> model .py mapping (the _jit_<mod>_test_* dir name).
_DSV4_MODULE_TO_PY = {
    "attention_csa": "decode_attention_csa.py",
    "attention_hca": "decode_attention_hca.py",
    "attention_swa": "decode_attention_swa.py",
    "compressor": "decode_compressor_ratio4.py",
    "indexer": "decode_indexer.py",
    "indexer_compressor": "decode_indexer_compressor.py",
    "sparse_attn": "decode_sparse_attn.py",
    "gate": "gate.py", "hc_head": "hc_head.py", "hc_post": "hc_post.py",
    "hc_pre": "hc_pre.py", "mtp_projection": "mtp_projection.py",
    "qkv_proj_rope": "qkv_proj_rope.py", "rms_norm": "rmsnorm.py",
    "expert_routed": "expert_routed.py", "expert_shared": "expert_shared.py",
}

# DSV4 module -> the @pl.jit entry fn name (when it differs from <stem>_test).
_DSV4_MODULE_TO_JIT_ENTRY = {
    "attention_csa": "attention_csa_test",
    "attention_hca": "attention_hca_test",
    "attention_swa": "attention_swa_test",
    "compressor": "compressor_test",
    "indexer": "indexer_test",
    "indexer_compressor": "indexer_compressor_test",
    "sparse_attn": "sparse_attn_test",
    "gate": "gate_test", "hc_head": "hc_head_test", "hc_post": "hc_post_test",
    "hc_pre": "hc_pre_test", "mtp_projection": "mtp_projection_test",
    "qkv_proj_rope": "qkv_proj_rope_test", "rms_norm": "rms_norm_test",
    "expert_routed": "expert_routed_test", "expert_shared": "expert_shared_test",
}


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
    ap.add_argument("--module", default=None,
                    help="Phase 5: sweep ALL inner kernels of one module "
                         "(capture once via Route 1, replay each on Route 2). "
                         "e.g. --module attention_csa. When set, the leaf "
                         "sweep is skipped.")
    args = ap.parse_args()

    if not VPTO_ENV_SH.exists():
        print(f"[validate] ERROR: {VPTO_ENV_SH} not found.", file=sys.stderr)
        return 2

    # --- Phase 5: module inner-kernel sweep (capture + replay) ---
    if args.module is not None:
        mod = args.module
        mod_py_name = _DSV4_MODULE_TO_PY.get(mod, f"{mod}.py")
        model_py = REPO_ROOT / "models" / "deepseek_v4_pro" / mod_py_name
        if not model_py.exists():
            # try prefill_/decode_ prefixes
            for pfx in ("decode_", "prefill_"):
                cand = REPO_ROOT / "models" / "deepseek_v4_pro" / f"{pfx}{mod_py_name}"
                if cand.exists():
                    model_py = cand
                    break
        if not model_py.exists():
            print(f"[validate] ERROR: no model .py for module {mod!r} "
                  f"(tried {mod_py_name})", file=sys.stderr)
            return 2
        print(f"[validate] [phase5] module={mod} model={model_py.name} "
              f"device={args.device} mode={args.mode}")
        results = run_phase5_module(mod, model_py, args.mode, args.device)
    else:
        # --- leaf sweep (Route 2 Mode B, no capture needed) ---
        if not CLASSIFICATION_CSV.exists():
            print(f"[validate] ERROR: {CLASSIFICATION_CSV} not found. "
                  f"Run the classification generator first.", file=sys.stderr)
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
