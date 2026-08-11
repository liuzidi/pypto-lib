#!/usr/bin/env python3
"""vpto_run.py — one-command VPTO board-validation executor.

Runs the full VPTO route (Route 2: pto -> ptoas LLVM -> bisheng fatobj .o) for
a single .pto on a real Ascend A5 NPU, and compares the output against a
torch/numpy golden reference. This is the executor for the
`vpto-board-validate` skill.

Usage:
    # Mode A — test_for_ptoas golden_lib contract (BUILDERS + run_case):
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto <path.pto> --golden-lib <golden_lib.py> \\
        --device 0 [--kernel <name>] [--build-dir <dir>] [--keep]

    # Mode B — DSV4 run_jit-style golden (model .py + MODES + golden_fn):
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto <path.pto> --model-py models/deepseek_v4_pro/<mod>.py \\
        --mode decode --device 0

Inputs:
  --pto          a pypto-emitted .pto (EmitC-era tile dialect: tile_buf/tload/...).
  --golden-lib   (Mode A) a *_golden_lib.py exposing BUILDERS = {"<kernel>": ...}
                 and run_case(name). Mirrors the test_for_ptoas contract.
  --model-py     (Mode B) a DSV4 model .py exposing <name>_test (the @pl.jit fn),
  --mode         (Mode B) "decode" | "prefill" — selects MODES entry (B,S).
                 build_tensor_specs(B,S) -> [TensorSpec...], golden_<name>_test.
                 Only leaf modules supported (ptr-arg count == spec count).

The skill sources `scripts/vpto_env.sh` (if not already sourced) so the ptodsl
daemon's mlir_core_vmi is built and on PYTHONPATH — otherwise ptoas emits an
empty ctor-only fatobj (the failure mode misdiagnosed as a dialect mismatch).

Prereqs (the skill checks and reports if missing):
  PTOAS_BIN, ASCEND_HOME_PATH, BISHENG_BIN, PTO_ISA_PATH, TILELANG_PATH,
  TILELANG_PKG  — set by scripts/vpto_env.sh + this host's CANN install.

Examples:
    # Mode A (test_for_ptoas reference set):
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto test_for_ptoas_extracted/test_for_ptoas/rmsnorm.pto \\
        --golden-lib test_for_ptoas_extracted/test_for_ptoas/qwen3_decode_golden_lib.py \\
        --device 0

    # Mode B (DSV4 leaf module):
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto build_output/_jit_rms_norm_test_*/ptoas/rms_norm.pto \\
        --model-py models/deepseek_v4_pro/rmsnorm.py --mode decode --device 0
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make the vendored lib importable when run as a script.
_SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_DIR))
from lib.pto_parse import parse_pto, get_outputs_from_golden_lib, get_golden_constants, get_scalar_semantic_names  # noqa: E402
from lib import setup_vpto, setup_main  # noqa: E402
from lib import run_jit_golden  # noqa: E402

# --- env: no host paths hardcoded here. Source scripts/vpto_env.sh (which
# also sources CANN + builds /tmp/mlir_core_vmi) to set all of these. The
# skill refuses to run if any are missing rather than guessing a host path.
_DEFAULTS: dict[str, str] = {}


def _env(key: str) -> str:
    return os.environ.get(key) or _DEFAULTS.get(key, "")


def _check_prereqs() -> list[str]:
    """Return a list of missing-prerequisite messages (empty = OK)."""
    missing = []
    for k in ("PTOAS_BIN", "ASCEND_HOME_PATH", "BISHENG_BIN", "PTO_ISA_PATH",
              "TILELANG_PATH", "TILELANG_PKG"):
        v = _env(k)
        if not v or not Path(v).exists():
            missing.append(f"  {k} = {v!r} (missing)")
    if missing:
        missing.append("  -> source scripts/vpto_env.sh and export the above, then retry.")
    return missing


def _ensure_vpto_env() -> None:
    """Ensure /tmp/mlir_core_vmi is built (needed for the ptodsl daemon).

    Idempotent: if vpto_env.sh was already sourced, this is a fast no-op.
    We re-source it in a subprocess to build the vmi if missing.
    """
    # vpto_env.sh lives at <repo-root>/scripts/vpto_env.sh — three levels up
    # from this skill dir (.claude/skills/vpto-board-validate/).
    vpto_env_sh = _SKILL_DIR.parents[2] / "scripts" / "vpto_env.sh"
    if not vpto_env_sh.exists():
        return  # caller must have sourced it; we'll fail later if env is unset
    mlir_core_vmi = Path("/tmp/mlir_core_vmi")
    if mlir_core_vmi.is_dir() and (mlir_core_vmi / "mlir" / "ir.py").exists():
        return  # already built
    print(f"[vpto_run] building {mlir_core_vmi} via {vpto_env_sh} ...")
    r = subprocess.run(["bash", "-c", f"set -e; source {vpto_env_sh} >/dev/null 2>&1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[vpto_run] WARN: vpto_env.sh returned {r.returncode}", file=sys.stderr)
        print(r.stderr[-500:], file=sys.stderr)


def _run(cmd: list[str], env: dict | None = None, cwd: str | Path | None = None,
         check: bool = True, label: str = "") -> subprocess.CompletedProcess:
    print(f"[vpto_run] {' '.join(str(c) for c in cmd)[:160]}")
    r = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.rstrip()[-1200:])
    if r.returncode != 0:
        print(r.stderr.rstrip()[-1200:], file=sys.stderr)
        if check:
            print(f"[vpto_run] FAILED step: {label or cmd[0]}", file=sys.stderr)
            sys.exit(1)
    return r


def _nm_fatobj(fatobj: Path, kernel: str) -> dict:
    """Inspect the fatobj: does it have the kernel symbol + nested ELF?"""
    info = {"has_kernel_sym": False, "has_ctor": False, "nested_elf_offsets": [],
            "size": fatobj.stat().st_size if fatobj.exists() else 0}
    if not fatobj.exists():
        return info
    r = subprocess.run(["nm", str(fatobj)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if f" T {kernel}" in line or f" t {kernel}" in line:
            info["has_kernel_sym"] = True
        if "cceModuleCtor" in line:
            info["has_ctor"] = True
    try:
        data = fatobj.read_bytes()
        import re as _re
        info["nested_elf_offsets"] = [m.start() for m in _re.finditer(b"\x7fELF", data)][:5]
    except Exception:
        pass
    return info


def _write_result_sidecar(path: Path, *, kernel: str, mode: str, device: int,
                          exit_code: int, fatobj_info: dict,
                          npu_stdout: str, compare_stdout: str,
                          rtol: float | None = None, atol: float | None = None,
                          error: str | None = None) -> None:
    """Write run_dir/result.json with a stable schema for the sweep framework.

    Parses the [TIMING] line from the NPU binary's stdout and the
    max_diff/threshold/n_over fields from compare.py's stdout (both the
    Mode B tolerance format and Mode A exact/ULP format). Missing fields
    are null so the consumer never needs to guess.
    """
    import json, re

    timing_ms = None
    m = re.search(r"\[TIMING\] kernel=\S+ time=([\d.]+)\s*ms", npu_stdout or "")
    if m:
        timing_ms = float(m.group(1))

    max_diff = None
    n_over = None
    n_total = None
    threshold = None
    compare_status = "unknown"  # "pass" | "fail" | "unknown"
    # Mode B tolerance format: "[INFO] X compare passed: max_diff=Y threshold=Z"
    # or "[ERROR] X compare failed: max_diff=Y threshold=Z n_over=A/B"
    cm = re.search(
        r"(passed|failed):\s*max_diff=([\d.eE+-]+)"
        r"(?:\s+threshold=([\d.eE+-]+))?"
        r"(?:\s+n_over=(\d+)/(\d+))?",
        compare_stdout or "")
    if cm:
        compare_status = "pass" if cm.group(1) == "passed" else "fail"
        max_diff = float(cm.group(2))
        if cm.group(3) is not None:
            threshold = float(cm.group(3))
        if cm.group(4) is not None:
            n_over = int(cm.group(4))
            n_total = int(cm.group(5))
    else:
        # Mode A exact/ULP: "compare passed (exact match)" or "compare passed
        # (max_ulp=N ...)" or "compare failed (name): max_ulp=..."
        if "compare passed" in (compare_stdout or ""):
            compare_status = "pass"
        elif "compare failed" in (compare_stdout or ""):
            compare_status = "fail"

    if exit_code == 0 and compare_status == "unknown":
        compare_status = "pass"
    elif exit_code != 0 and compare_status == "unknown":
        compare_status = "fail"

    result = {
        "kernel": kernel,
        "mode": mode,
        "device": device,
        "pass": compare_status == "pass",
        "compare_status": compare_status,
        "exit_code": exit_code,
        "max_diff": max_diff,
        "n_over": n_over,
        "n_total": n_total,
        "threshold": threshold,
        "timing_ms": timing_ms,
        "rtol": rtol,
        "atol": atol,
        "fatobj_bytes": fatobj_info.get("size"),
        "has_kernel_sym": fatobj_info.get("has_kernel_sym"),
        "has_ctor": fatobj_info.get("has_ctor"),
        "nested_elf_offsets": fatobj_info.get("nested_elf_offsets"),
        "error": error,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _ctx_len_from_config(model_py: Path, mode: str) -> int:
    """Read B*S from the model's config.py (ctx_len for the trailing index
    scalar). Used in --captured-dump mode where resolve_meta is skipped."""
    import subprocess
    b = "DECODE_BATCH" if mode == "decode" else "PREFILL_BATCH"
    s = "DECODE_SEQ" if mode == "decode" else "PREFILL_SEQ"
    cfg = model_py.parent / "config.py"
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('cfg', {str(cfg.absolute())!r})\n"
        "cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)\n"
        f"print(getattr(cfg, {b!r}) * getattr(cfg, {s!r}))\n"
    )
    r = subprocess.run(
        [str(run_jit_golden._venv_python()), "-c", code],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"[vpto_run] WARN: ctx_len from config failed: {r.stderr[-200:]}",
              file=sys.stderr)
        return 8  # fallback
    return int(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description="VPTO board-validation executor")
    ap.add_argument("--pto", required=True, type=Path)
    gsrc = ap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--golden-lib", type=Path,
                      help="Mode A: a *_golden_lib.py (BUILDERS + run_case).")
    gsrc.add_argument("--model-py", type=Path,
                      help="Mode B: a DSV4 model .py (run_jit-style golden).")
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill"],
                    help="Mode B: which MODES entry to use (default: decode).")
    ap.add_argument("--rtol", type=float, default=None,
                    help="override golden compare rtol (default: model's 5e-3).")
    ap.add_argument("--atol", type=float, default=None,
                    help="override golden compare atol (default: model's 5e-3).")
    ap.add_argument("--kernel", default=None, help="kernel name (default: .pto stem)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--captured-dump", type=Path, default=None,
                    help="Phase 5: a run_dir already containing vN.bin / "
                         "golden_vN.bin + capture_meta.json (from capture.py). "
                         "Skips resolve_meta + dump_bins; uses the captured "
                         "buffers directly. For non-leaf inner kernels whose "
                         "ptrs are intermediates from a preceding kernel.")
    ap.add_argument("--keep", action="store_true", help="keep the run dir (default: build_output)")
    args = ap.parse_args()

    if not args.pto.exists():
        print(f"[vpto_run] ERROR: .pto not found: {args.pto}", file=sys.stderr)
        return 2
    if args.golden_lib and not args.golden_lib.exists():
        print(f"[vpto_run] ERROR: golden_lib not found: {args.golden_lib}", file=sys.stderr)
        return 2
    if args.model_py and not args.model_py.exists():
        print(f"[vpto_run] ERROR: model_py not found: {args.model_py}", file=sys.stderr)
        return 2

    kernel = args.kernel or args.pto.stem
    missing = _check_prereqs()
    if missing:
        print("[vpto_run] missing prerequisites:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 2
    _ensure_vpto_env()

    # --- resolve env + build dir ---
    ascend = _env("ASCEND_HOME_PATH")
    pto_isa = _env("PTO_ISA_PATH")
    bisheng = _env("BISHENG_BIN")
    ptoas_bin = _env("PTOAS_BIN")
    tilelang = _env("TILELANG_PATH")
    tilelang_pkg = _env("TILELANG_PKG")
    llvm_build = _env("LLVM_BUILD")
    ptoas_source = _env("PTOAS_SOURCE")

    build_root = args.build_dir or (Path("build_output") / f"vpto_{kernel}")
    run_dir = build_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # CANN set_env (for the bisheng + NPU run)
    cann_env = dict(os.environ)
    cann_env["ASCEND_HOME_PATH"] = ascend
    cann_env["PTO_ISA_PATH"] = pto_isa
    cann_env["TILELANG_PATH"] = tilelang
    cann_env["TILELANG_PKG"] = tilelang_pkg
    cann_env["PTOAS_BIN"] = ptoas_bin
    cann_env["DEVICE_ID"] = str(args.device)
    cann_env["BISHENG_BIN"] = bisheng
    cann_env["LD_LIBRARY_PATH"] = (
        f"{llvm_build}/lib:{ascend}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}")
    set_env_sh = Path(ascend) / "set_env.sh"
    if set_env_sh.exists():
        subprocess.run(["bash", "-c", f"set +u; source {set_env_sh} >/dev/null 2>&1; env"],
                       capture_output=True, text=True)
        # carry CANN env forward
        r = subprocess.run(["bash", "-c", f"set +u; source {set_env_sh} >/dev/null 2>&1; env"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                if k in ("LD_LIBRARY_PATH", "ASCEND_AICPU_PATH", "ASCEND_OPP_PATH",
                         "TOOLCHAIN_HOME", "ASCEND_SLOG_PRINT_TO_STDOUT"):
                    cann_env.setdefault(k, v)

    # --- parse .pto + golden metadata (mode-specific) ---
    info = parse_pto(args.pto)
    if not info["func_name"]:
        print(f"[vpto_run] ERROR: cannot parse .pto: {args.pto}", file=sys.stderr)
        return 2
    pto_text = args.pto.read_text(encoding="utf-8")
    kind = setup_vpto.detect_kernel_kind(pto_text)

    use_model_py = args.model_py is not None
    use_captured = args.captured_dump is not None
    if use_captured:
        # Phase 5: non-leaf inner kernel. The golden buffers were captured
        # from a Route-1 run of the owning module (capture.py) and written
        # into args.captured_dump as vN.bin / golden_vN.bin + capture_meta.json.
        # Skip resolve_meta (which would fail: no ptr->spec map for non-leaf).
        import json as _json
        meta_path = args.captured_dump / "capture_meta.json"
        if not meta_path.exists():
            print(f"[vpto_run] ERROR: {meta_path} not found "
                  f"(--captured-dump needs a capture.py run_dir)", file=sys.stderr)
            return 2
        cmeta = _json.loads(meta_path.read_text(encoding="utf-8"))
        outputs = cmeta["outputs"]
        consts = {}
        scalar_sem = []
        ctx_marked = False
        # ctx_len still needed for the trailing index scalar; read from config.
        ctx_len = _ctx_len_from_config(args.model_py, args.mode) if args.model_py else 8
        for p in info["params"]:
            if p["pto_type"] in ("i32", "index"):
                sig = p.get("sig_name", "") or p["name"]
                if not ctx_marked and "spmd" not in sig:
                    scalar_sem.append("ctx_len")
                    ctx_marked = True
                else:
                    scalar_sem.append(None)
        load_vals = {"ctx_len": ctx_len, "ctx_blocks": None}
        golden_np_types = cmeta["np_types"]
        # elem_counts_override: captured numel per vN (from the dump shapes)
        elem_counts_override = cmeta["elem_counts"]
    elif use_model_py:
        # Mode B: DSV4 run_jit-style. The golden metadata (which specs are
        # outputs, ctx_len = B*S, np_types, elem_counts) comes from importing
        # the model. Resolve BEFORE main.cpp (needs outputs + elem_counts).
        model_meta = run_jit_golden.resolve_meta(args.model_py, args.mode, info)
        outputs = model_meta["outputs"]
        consts = {}
        # DSV4 .pto kernels take a trailing `index` arg = the dynamic T dim
        # (T_DYN = B*S). setup_main fills it from load_vals["ctx_len"] when
        # the scalar's semantic is "ctx_len" — so build scalar_sem marking the
        # first non-spmd index/i32 scalar as ctx_len. (spmd args keep the
        # setup_main SPMD special-case: block_num=1, block_idx=0.)
        scalar_sem = []
        ctx_marked = False
        for p in info["params"]:
            if p["pto_type"] in ("i32", "index"):
                sig = p.get("sig_name", "") or p["name"]
                if not ctx_marked and "spmd" not in sig:
                    scalar_sem.append("ctx_len")
                    ctx_marked = True
                else:
                    scalar_sem.append(None)
        load_vals = {"ctx_len": model_meta["ctx_len"],
                     "ctx_blocks": model_meta.get("ctx_blocks")}
        golden_np_types = model_meta["np_types"]
        elem_counts_override = model_meta.get("elem_counts")
    else:
        outputs = get_outputs_from_golden_lib(args.golden_lib.parent, kernel) or []
        consts = get_golden_constants(args.golden_lib.parent) or {}
        scalar_sem = get_scalar_semantic_names(args.golden_lib.parent, kernel) or []
        load_vals = {"ctx_len": consts.get("MAX_SEQ"),
                     "ctx_blocks": consts.get("MAX_CTX_BLOCKS")}
        golden_np_types = None
        elem_counts_override = None
    print(f"[vpto_run] kernel={kernel} kind={kind} params="
          f"{[(p['name'], p['pto_type']) for p in info['params']]} "
          f"outputs={outputs} elem_counts={info.get('elem_counts', {})} "
          f"golden_mode={'captured' if use_captured else 'model-py' if use_model_py else 'golden-lib'}")

    # --- 1. write golden.py stub + compare.py + validation_runtime.py into run_dir ---
    if use_captured:
        # Phase 5: vN.bin / golden_vN.bin already in run_dir (from capture.py).
        # Copy them in if captured_dump != run_dir, else they're already here.
        if args.captured_dump.resolve() != run_dir.resolve():
            for src in args.captured_dump.glob("*.bin"):
                shutil.copy(src, run_dir / src.name)
        # no gen_golden.py stub — bins are pre-existing.
    elif use_model_py:
        # Mode B: stub calls run_jit_golden.dump_bins, which imports the model
        # under the repo venv + golden package, runs golden_fn, dumps *.bin.
        # Metadata was already resolved above; this step just materializes bins.
        # NB: stub filename must NOT be golden.py — it would shadow the repo
        # golden package (circular import via `from golden import TensorSpec`).
        (run_dir / "gen_golden.py").write_text(
            f"""#!/usr/bin/env python3
import sys
from pathlib import Path
SKILL = Path({str(_SKILL_DIR)!r})
sys.path.insert(0, str(SKILL)); sys.path.insert(0, str(SKILL / "lib"))
from run_jit_golden import dump_bins
dump_bins(
    model_py=Path({str(args.model_py.absolute())!r}),
    mode={args.mode!r},
    run_dir=Path("."),
    pto_path=Path({str(args.pto.absolute())!r}),
)
""", encoding="utf-8")
    else:
        golden_lib_name = args.golden_lib.stem
        (run_dir / "golden.py").write_text(
            f"""#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, {str(args.golden_lib.parent)!r})
from {golden_lib_name} import run_case

if __name__ == "__main__":
    run_case({kernel!r})
""", encoding="utf-8")
    # compare.py + validation_runtime.py at the build_root (parents[1] of golden.py)
    shutil.copy(_SKILL_DIR / "runtime" / "validation_runtime.py", build_root / "validation_runtime.py")
    shutil.copy(_SKILL_DIR / "runtime" / "compare.py", run_dir / "compare.py")
    (run_dir / "outputs.txt").write_text("\n".join(outputs) + "\n", encoding="utf-8")

    # --- 2. write main.cpp + launch.cpp ---
    main_cpp = setup_main.gen_main_cpp(
        info, outputs,
        load_vals=load_vals,
        consts=consts, scalar_sem_names=scalar_sem,
        elem_counts_override=elem_counts_override)
    (run_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")
    launch_cpp = setup_vpto.generate_launch_cpp(info)
    (run_dir / "launch.cpp").write_text(launch_cpp, encoding="utf-8")

    # --- 3. sed-preprocess .pto -> build/<kernel>.pto ---
    pto_file = build_root / f"{kernel}.pto"
    setup_vpto.preprocess_pto(args.pto, pto_file, kind)

    # --- 4. ptoas VPTO -> fatobj ---
    fatobj = build_root / f"{kernel}.o"
    daemon_env = dict(cann_env)
    daemon_env["PYTHONPATH"] = f"/tmp/mlir_core_vmi:{ptoas_source}/ptodsl:{tilelang_pkg}"
    # clear stale daemon state
    for sock in Path("/tmp").glob("tilelib_daemon_*.sock"):
        try: sock.unlink()
        except OSError: pass
    tileops_pycache = Path(tilelang) / "__pycache__"
    if tileops_pycache.exists():
        shutil.rmtree(tileops_pycache, ignore_errors=True)
    _run([ptoas_bin] + setup_vpto.PTO_COMPILE_OPT
         + ["--tilelang-path", tilelang, "--tilelang-pkg-path", tilelang_pkg,
            str(pto_file), "-o", str(fatobj)],
         env=daemon_env, label="ptoas VPTO")

    fatobj_info = _nm_fatobj(fatobj, kernel)
    print(f"[vpto_run] fatobj: {fatobj_info['size']} bytes, "
          f"has_kernel_sym={fatobj_info['has_kernel_sym']}, "
          f"has_ctor={fatobj_info['has_ctor']}, "
          f"nested_elf_offsets={fatobj_info['nested_elf_offsets']}")
    if not fatobj_info["has_kernel_sym"]:
        print("[vpto_run] FAIL: fatobj has no kernel symbol — ptoas skipped codegen. "
              "Check: (1) sed preprocessing added pto.kernel to the func attr, "
              "(2) --enable-op-fusion is ON (not false).", file=sys.stderr)
        return 1

    # --- 5. bisheng: launch.o, .so, host bin ---
    _run([bisheng, "-c", "-fPIC", "-xcce", "-fenable-matrix", "--cce-aicore-enable-tl",
          "-fPIC", "-Xhost-start", "-Xhost-end",
          "-mllvm", "-cce-aicore-stack-size=0x8000",
          "-mllvm", "-cce-aicore-function-stack-size=0x8000",
          "-mllvm", "-cce-aicore-record-overflow=true",
          "-mllvm", "-cce-aicore-addr-transform",
          "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
          "--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE", "-std=c++17",
          "-Wno-macro-redefined", "-Wno-ignored-attributes",
          "-I", f"{ascend}/include", "-I", f"{ascend}/pkg_inc",
          "-I", f"{ascend}/pkg_inc/profiling", "-I", f"{ascend}/pkg_inc/runtime/runtime",
          "-I", f"{pto_isa}/include", "-I", f"{pto_isa}/tests/common",
          str(run_dir / "launch.cpp"), "-o", str(build_root / "launch.o")],
         env=cann_env, label="bisheng launch.o")
    _run([bisheng, "-fPIC", "-s", "-Wl,-z,relro", "-Wl,-z,now", "--cce-fatobj-link",
          "-shared", f"-Wl,-soname,lib{kernel}_kernel.so",
          "-L", f"{ascend}/lib64", "-Wl,-rpath," + f"{ascend}/lib64",
          "-o", str(build_root / f"lib{kernel}_kernel.so"),
          str(fatobj), str(build_root / "launch.o"),
          "-Wl,--no-as-needed", "-lruntime"],
         env=cann_env, label="bisheng link .so")
    _run([bisheng, "-xc++", "-include", "stdint.h", "-include", "stddef.h", "-std=c++17",
          str(run_dir / "main.cpp"),
          "-I", f"{ascend}/include", "-I", f"{pto_isa}/include", "-I", f"{pto_isa}/tests/common",
          "-L", str(build_root), "-L", f"{ascend}/lib64",
          "-Wl,-rpath," + str(build_root), "-Wl,-rpath," + f"{ascend}/lib64",
          "-o", str(build_root / kernel), f"-l{kernel}_kernel",
          "-Wl,--allow-shlib-undefined", "-Wl,--no-as-needed", "-lruntime",
          "-lstdc++", "-lascendcl", "-lm", "-ltiling_api", "-lplatform", "-lc_sec", "-ldl", "-lnnopbase"],
         env=cann_env, label="bisheng main.cpp")

    # --- 6. golden ---
    golden_env = dict(cann_env)
    if use_captured:
        # Phase 5: bins already in run_dir (copied in step 1). No gen_golden.py.
        print("[vpto_run] golden: using captured bins (skipping gen_golden.py)")
    elif use_model_py:
        # Mode B: needs torch + golden package → run under the repo venv.
        venv_py = run_jit_golden._venv_python()
        golden_env["PYTHONPATH"] = (
            f"{_SKILL_DIR}:{_SKILL_DIR / 'lib'}:"
            f"{args.model_py.parent.parents[1]}:{args.model_py.parent}")
        gp = _run([venv_py, str((run_dir / "gen_golden.py").absolute())],
                  env=golden_env, cwd=str(run_dir.absolute()), label="golden (model-py)")
    else:
        golden_env["PYTHONPATH"] = (
            f"/tmp/mlir_core_vmi:{ptoas_source}/ptodsl:"
            f"{str(build_root.absolute())}:{args.golden_lib.parent.absolute()}")
        gp = _run(["python3", str((run_dir / "golden.py").absolute())],
                  env=golden_env, cwd=str(run_dir.absolute()), label="golden (golden-lib)")
    # golden.py writes *.bin into its cwd (run_dir); the host binary also reads
    # ./vN.bin from its cwd (run_dir), so no extra linking is needed.

    # --- 7. NPU run ---
    run_env = dict(cann_env)
    run_env["ACL_DEVICE_ID"] = str(args.device)
    run_env["LD_LIBRARY_PATH"] = f"{build_root.absolute()}:{ascend}/lib64:{cann_env.get('LD_LIBRARY_PATH', '')}"
    r = _run([str((build_root / kernel).absolute())], env=run_env, cwd=str(run_dir.absolute()),
             check=False, label="NPU run")
    if r.returncode != 0:
        print(f"[vpto_run] NPU run FAILED (exit {r.returncode})", file=sys.stderr)
        _write_result_sidecar(
            run_dir / "result.json", kernel=kernel, mode=args.mode,
            device=args.device, exit_code=1, fatobj_info=fatobj_info,
            npu_stdout=r.stdout, compare_stdout="",
            rtol=(args.rtol if args.rtol is not None else 5e-3) if use_model_py else None,
            atol=(args.atol if args.atol is not None else 5e-3) if use_model_py else None,
            error=f"NPU run failed (exit {r.returncode})")
        return 1

    # --- 8. compare ---
    cmp_env = dict(cann_env)
    if use_model_py:
        venv_py = run_jit_golden._venv_python()
        cmp_env["PYTHONPATH"] = f"{build_root.absolute()}:{_SKILL_DIR}"
        # pass tolerances via env so compare.py can pick them up
        rtol = args.rtol if args.rtol is not None else 5e-3
        atol = args.atol if args.atol is not None else 5e-3
        cmp_env["VPTO_COMPARE_RTOL"] = str(rtol)
        cmp_env["VPTO_COMPARE_ATOL"] = str(atol)
        cr = _run([venv_py, str((run_dir / "compare.py").absolute())], env=cmp_env,
                  cwd=str(run_dir.absolute()), check=False, label="compare (model-py)")
    else:
        cmp_env["PYTHONPATH"] = f"{build_root.absolute()}:{args.golden_lib.parent.absolute()}"
        cr = _run(["python3", str((run_dir / "compare.py").absolute())], env=cmp_env,
                  cwd=str(run_dir.absolute()), check=False, label="compare (golden-lib)")
    print(f"[vpto_run] compare exit: {cr.returncode}")
    print(f"[vpto_run] DONE. artifacts in {build_root}")

    # Write a structured result sidecar so the sweep framework (and any
    # caller) can read per-run results without scraping stdout. Schema is
    # stable; missing fields are null. Timing comes from the NPU binary's
    # own [TIMING] stdout line; compare stats from compare.py's stdout
    # (both Mode A exact/ULP and Mode B tolerance variants are handled).
    _write_result_sidecar(
        run_dir / "result.json", kernel=kernel, mode=args.mode,
        device=args.device, exit_code=cr.returncode,
        fatobj_info=fatobj_info, npu_stdout=r.stdout, compare_stdout=cr.stdout,
        rtol=(args.rtol if args.rtol is not None else 5e-3) if use_model_py else None,
        atol=(args.atol if args.atol is not None else 5e-3) if use_model_py else None,
    )

    if not args.keep and build_root.is_relative_to(Path("build_output")):
        # keep build_output artifacts (they're gitignored) — do not auto-delete
        pass
    return cr.returncode


if __name__ == "__main__":
    sys.exit(main())
