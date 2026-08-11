#!/usr/bin/env python3
"""vpto_run.py — one-command VPTO board-validation executor.

Runs the full VPTO route (Route 2: pto -> ptoas LLVM -> bisheng fatobj .o) for
a single .pto on a real Ascend A5 NPU, and compares the output against a
torch/numpy golden reference. This is the executor for the
`vpto-board-validate` skill.

Usage:
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto <path.pto> --golden-lib <golden_lib.py> \\
        --device 0 [--kernel <name>] [--build-dir <dir>] [--keep]

Inputs:
  --pto          a pypto-emitted .pto (EmitC-era tile dialect: tile_buf/tload/...).
  --golden-lib   a *_golden_lib.py exposing BUILDERS = {"<kernel>": build_fn, ...}
                 and run_case(name). Mirrors the test_for_ptoas contract.

The skill sources `scripts/vpto_env.sh` (if not already sourced) so the ptodsl
daemon's mlir_core_vmi is built and on PYTHONPATH — otherwise ptoas emits an
empty ctor-only fatobj (the failure mode misdiagnosed as a dialect mismatch).

Prereqs (the skill checks and reports if missing):
  PTOAS_BIN, ASCEND_HOME_PATH, BISHENG_BIN, PTO_ISA_PATH, TILELANG_PATH,
  TILELANG_PKG  — set by scripts/vpto_env.sh + this host's CANN install.

Example:
    python .claude/skills/vpto-board-validate/vpto_run.py \\
        --pto test_for_ptoas_extracted/test_for_ptoas/rmsnorm.pto \\
        --golden-lib test_for_ptoas_extracted/test_for_ptoas/qwen3_decode_golden_lib.py \\
        --device 0
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


def main() -> int:
    ap = argparse.ArgumentParser(description="VPTO board-validation executor")
    ap.add_argument("--pto", required=True, type=Path)
    ap.add_argument("--golden-lib", required=True, type=Path)
    ap.add_argument("--kernel", default=None, help="kernel name (default: .pto stem)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the run dir (default: build_output)")
    args = ap.parse_args()

    if not args.pto.exists():
        print(f"[vpto_run] ERROR: .pto not found: {args.pto}", file=sys.stderr)
        return 2
    if not args.golden_lib.exists():
        print(f"[vpto_run] ERROR: golden_lib not found: {args.golden_lib}", file=sys.stderr)
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

    # --- parse .pto + golden_lib metadata ---
    info = parse_pto(args.pto)
    if not info["func_name"]:
        print(f"[vpto_run] ERROR: cannot parse .pto: {args.pto}", file=sys.stderr)
        return 2
    outputs = get_outputs_from_golden_lib(args.golden_lib.parent, kernel) or []
    consts = get_golden_constants(args.golden_lib.parent) or {}
    scalar_sem = get_scalar_semantic_names(args.golden_lib.parent, kernel) or []
    pto_text = args.pto.read_text(encoding="utf-8")
    kind = setup_vpto.detect_kernel_kind(pto_text)
    print(f"[vpto_run] kernel={kernel} kind={kind} params="
          f"{[(p['name'], p['pto_type']) for p in info['params']]} "
          f"outputs={outputs} elem_counts={info.get('elem_counts', {})}")

    # --- 1. write golden.py stub + compare.py + validation_runtime.py into run_dir ---
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
        load_vals={"ctx_len": consts.get("MAX_SEQ"), "ctx_blocks": consts.get("MAX_CTX_BLOCKS")},
        consts=consts, scalar_sem_names=scalar_sem)
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
    golden_env["PYTHONPATH"] = f"/tmp/mlir_core_vmi:{ptoas_source}/ptodsl:{str(build_root.absolute())}:{args.golden_lib.parent.absolute()}"
    _run(["python3", str((run_dir / "golden.py").absolute())], env=golden_env,
         cwd=str(run_dir.absolute()), label="golden")
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
        return 1

    # --- 8. compare ---
    cmp_env = dict(cann_env)
    cmp_env["PYTHONPATH"] = f"{build_root.absolute()}:{args.golden_lib.parent.absolute()}"
    cr = _run(["python3", str((run_dir / "compare.py").absolute())], env=cmp_env,
              cwd=str(run_dir.absolute()), check=False, label="compare")
    print(f"[vpto_run] compare exit: {cr.returncode}")
    print(f"[vpto_run] DONE. artifacts in {build_root}")
    if not args.keep and build_root.is_relative_to(Path("build_output")):
        # keep build_output artifacts (they're gitignored) — do not auto-delete
        pass
    return cr.returncode


if __name__ == "__main__":
    sys.exit(main())
