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


def _route_label(route: str, phase5: bool = False) -> str:
    """Build the CSV 'route' column value. Keeps the 'vpto' / 'vpto(phase5)'
    base label so the baseline-vs-vmi distinction is visible alongside the
    phase-vs-leaf distinction."""
    suffix = f"[{route}]" if route and route != "baseline" else ""
    return f"vpto(phase5){suffix}" if phase5 else f"vpto{suffix}"


def run_one_leaf(row: dict, mode: str, device: int, route: str = "baseline") -> dict:
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
        f"--mode {mode} --device {device} --route {route}",
    ]
    print(f"\n[validate] {kernel}: running ...", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(REPO_ROOT), timeout=300)
    except subprocess.TimeoutExpired:
        return _error_row(row, mode, device,
                          f"replay timed out (>300s) — likely ptoas/bisheng/NPU hang")
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
    out["route"] = _route_label(route)
    out["mode"] = mode
    out["device"] = device
    return out


def _error_row(row: dict, mode: str, device: int, error: str,
               route: str = "baseline") -> dict:
    out = {f: "" for f in RESULT_FIELDS}
    out.update({
        "kernel": row.get("kernel", ""),
        "module": row.get("module", ""),
        "route": _route_label(route),
        "mode": mode,
        "device": device,
        "pass": False,
        "compare_status": "crash",
        "exit_code": -1,
        "error": error,
    })
    return out


def run_phase5_module(module: str, model_py: Path, mode: str, device: int,
                      route: str = "baseline") -> list:
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
        # Run the module dump in a subprocess with vpto_env.sh sourced + the
        # CANN env inherited. Doing this in-process fails because pypto's
        # worker.init dlopens libruntime_common.so, which depends on the full
        # CANN LD_LIBRARY_PATH (set by the nested set_env.sh source); the env
        # dict captured by _source_vpto_env is incomplete for that dlopen
        # chain. A bash -c subprocess inherits the bash shell's complete
        # post-source env correctly.
        entry = _DSV4_MODULE_TO_JIT_ENTRY.get(module)
        cap_cmd = [
            "bash", "-c",
            f"source {VPTO_ENV_SH} && export PTOAS_ROOT={str(Path(_resolve_ptoas_bin()).parent)} "
            f"&& exec {venv_py} {capture_py} --capture-only "
            f"--model-py {model_py} --mode {mode} --device {device} "
            f"--jit-entry {entry}"
        ]
        print(f"[validate] [phase5] capturing {module} (one-time Route-1 dump)...",
              flush=True)
        cap_r = subprocess.run(cap_cmd, capture_output=True, text=True,
                                cwd=str(REPO_ROOT), timeout=300)
        if cap_r.stdout:
            print(cap_r.stdout.rstrip()[-1000:])
        if cap_r.returncode != 0:
            print(cap_r.stderr.rstrip()[-600:], file=sys.stderr)
            return [_phase5_error_row(
                module, model_py, mode, device, module,
                f"capture failed (exit {cap_r.returncode}): "
                f"{(cap_r.stderr or cap_r.stdout).strip()[-200:]}")]
    print(f"[validate] [phase5] dump at {dump_wd}", flush=True)

    # Step 2: read name_map to get the kernel list (callable_id -> name).
    nm_glob = list((dump_wd / "dfx_outputs").glob("name_map_*.json"))
    if not nm_glob:
        return [_phase5_error_row(module, model_py, mode, device, module,
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
            f"--device {device} --kernel {kname} --route {route} "
            f"--captured-dump {run_dir}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(REPO_ROOT), timeout=300)
        except subprocess.TimeoutExpired:
            results.append(_phase5_error_row(
                module, model_py, mode, device, kname,
                "replay timed out (>300s) — likely ptoas/bisheng/NPU hang"))
            continue
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
        out["route"] = _route_label(route, phase5=True)
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
                      kernel: str, error: str, route: str = "baseline") -> dict:
    out = {f: "" for f in RESULT_FIELDS}
    out.update({
        "kernel": kernel, "module": module,
        "route": _route_label(route, phase5=True),
        "mode": mode, "device": device, "pass": False,
        "compare_status": "crash", "exit_code": -1, "error": error,
    })
    return out


def _resolve_ptoas_bin() -> str:
    """Parse PTOAS_BIN out of scripts/vpto_env.sh (rather than hardcoding a
    private absolute path). Returns '' if not found."""
    import re
    try:
        text = VPTO_ENV_SH.read_text(encoding="utf-8")
        m = re.search(r'^export\s+PTOAS_BIN=(\S+)', text, re.M)
        if m:
            return m.group(1).strip('"').strip("'")
    except OSError:
        pass
    return ""


def _source_vpto_env() -> dict:
    """Return the environment dict that `source scripts/vpto_env.sh` produces.
    Capture runs run_jit in-process, so the CANN/PTOAS env vars must be set on
    os.environ first. We source the script in bash then print env from python
    (NOT `env` command — which misses some vars that vpto_env.sh's nested
    set_env.sh source sets without exporting to the `env` command's view).
    """
    import subprocess
    venv_py = str(REPO_ROOT / ".venv" / "bin" / "python3")
    cmd = (
        f"source {VPTO_ENV_SH} && {venv_py} -c "
        "'import os; [print(k+chr(0)+v) for k,v in os.environ.items()]'"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=120)
    if r.returncode != 0:
        return dict(os.environ)
    env = {}
    # use chr(0) as separator to handle multi-line values; split on first \0
    for line in r.stdout.split("\n"):
        if "\x00" in line:
            k, v = line.split("\x00", 1)
            env[k] = v
        elif "=" in line and not line.startswith("[vpto_env]"):
            # fallback for lines without separator
            k, v = line.split("=", 1)
            env[k] = v
    merged = dict(os.environ)
    merged.update(env)
    return merged


# DSV4 module -> model .py mapping (the _jit_<mod>_test_* dir name).
_DSV4_MODULE_TO_PY = {
    # decode (16)
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
    # prefill (8) — each emits inner kernels like its decode counterpart
    "prefill_attention_csa": "prefill_attention_csa.py",
    "prefill_attention_hca": "prefill_attention_hca.py",
    "prefill_attention_swa": "prefill_attention_swa.py",
    "prefill_compressor_ratio128": "prefill_compressor_ratio128.py",
    "prefill_compressor_ratio4": "prefill_compressor_ratio4.py",
    "prefill_indexer": "prefill_indexer.py",
    "prefill_indexer_compressor": "prefill_indexer_compressor.py",
    "prefill_sparse_attn": "prefill_sparse_attn.py",
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
    "prefill_attention_csa": "prefill_attention_csa_test",
    "prefill_attention_hca": "prefill_attention_hca_test",
    "prefill_attention_swa": "prefill_attention_swa_test",
    "prefill_compressor_ratio128": "prefill_compressor_ratio128_test",
    "prefill_compressor_ratio4": "prefill_compressor_ratio4_test",
    "prefill_indexer": "prefill_indexer_test",
    "prefill_indexer_compressor": "prefill_indexer_compressor_test",
    "prefill_sparse_attn": "prefill_sparse_attn_test",
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
    ap.add_argument("--all-modules", action="store_true",
                    help="Phase 5: sweep ALL inner kernels of ALL DSV4 "
                         "modules (24 modules: 16 decode + 8 prefill). "
                         "Capture once per module, replay each kernel. "
                         "Prefill modules use --mode prefill automatically.")
    ap.add_argument("--route", default="baseline",
                    choices=["baseline", "vmi-membar-vfoff"],
                    help="VPTO codegen route. 'baseline' = op-fusion on, "
                         "no membar, bisheng VF-fusion on. 'vmi-membar-vfoff' "
                         "= VMI fusion + vecscope membar + bisheng VF-fusion "
                         "off (7 -mllvm options). See ptoas "
                         "README-vmi-membar-bishengvfoff.md. Default: baseline.")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: "
                         "baselines/vpto_dsv4_vector/sweep_results.csv for "
                         "baseline route, sweep_results_<route>.csv for others).")
    args = ap.parse_args()

    if not VPTO_ENV_SH.exists():
        print(f"[validate] ERROR: {VPTO_ENV_SH} not found.", file=sys.stderr)
        return 2

    # --- Phase 5: module inner-kernel sweep (capture + replay) ---
    if args.all_modules:
        # sweep every module in _DSV4_MODULE_TO_PY. Prefill modules (key
        # starts with prefill_) use mode=prefill; decode modules use args.mode
        # (default decode). Results aggregate into one sweep_results.csv.
        all_results = []
        n_modules = len(_DSV4_MODULE_TO_PY)
        for mi, mod in enumerate(_DSV4_MODULE_TO_PY, 1):
            mod_mode = "prefill" if mod.startswith("prefill_") else args.mode
            model_py = REPO_ROOT / "models" / "deepseek_v4_pro" / _DSV4_MODULE_TO_PY[mod]
            if not model_py.exists():
                print(f"\n[validate] [{mi}/{n_modules}] SKIP {mod}: "
                      f"{model_py.name} not found", flush=True)
                all_results.append(_phase5_error_row(
                    mod, model_py, mod_mode, args.device, mod,
                    f"model .py not found: {model_py}", route=args.route))
                continue
            print(f"\n[validate] [{mi}/{n_modules}] === module {mod} "
                  f"({model_py.name}, mode={mod_mode}, route={args.route}) ===",
                  flush=True)
            try:
                mod_results = run_phase5_module(mod, model_py, mod_mode, args.device,
                                                route=args.route)
            except Exception as e:  # noqa: BLE001 — never abort the full sweep
                mod_results = [_phase5_error_row(
                    mod, model_py, mod_mode, args.device, mod,
                    f"module sweep crashed: {e}", route=args.route)]
            all_results.extend(mod_results)
        results = all_results
    elif args.module is not None:
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
        mod_mode = "prefill" if mod.startswith("prefill_") else args.mode
        print(f"[validate] [phase5] module={mod} model={model_py.name} "
              f"device={args.device} mode={mod_mode}")
        results = run_phase5_module(mod, model_py, mod_mode, args.device,
                                    route=args.route)
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
            res = run_one_leaf(row, args.mode, args.device, route=args.route)
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

    # write aggregated CSV (route-specific filename so baseline sweep is not
    # overwritten by a vmi-membar-vfoff sweep)
    out_csv = Path(args.out) if args.out else (
        SWEEP_RESULTS_CSV if args.route == "baseline"
        else SWEEP_RESULTS_CSV.parent / f"sweep_results_{args.route}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)

    # stdout summary
    n_pass = sum(1 for r in results if r.get("pass"))
    n_total = len(results)
    print(f"\n[validate] ===== sweep summary =====")
    print(f"[validate] route: {args.route}")
    print(f"[validate] results CSV: {out_csv}")
    print(f"[validate] {n_pass}/{n_total} kernels PASS; "
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
