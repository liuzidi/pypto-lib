#!/usr/bin/env python3
"""EmitC route runner: compile .pto via ptoas --backend=emitc → bisheng → NPU.

EmitC generates a .cpp file from .pto, then bisheng compiles it to .o → .so.
This route uses simpler runtime (not CANN module-launch), and op-fusion is OFF
by default (matching the known-good EmitC configuration).

Used by run_all.py when --route emitc is specified.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
VPTO_ENV_SH = REPO_ROOT / "scripts" / "vpto_env.sh"
GOLDEN_LIB = ROOT / "dsv4_golden_lib.py"

# pto_parse + setup_main are in the skill's lib (symlinked as lib/ package)
sys.path.insert(0, str(ROOT))
from lib.pto_parse import parse_pto, derive_scalar_values, pto_type_to_c, get_outputs_from_golden_lib
from lib.setup_main import gen_main_cpp
from lib import setup_vpto


def _source_env():
    """Source vpto_env.sh and return the env dict."""
    import subprocess
    r = subprocess.run(
        ["bash", "-c", f"source {VPTO_ENV_SH} 2>/dev/null && env"],
        capture_output=True, text=True, timeout=30)
    env = {}
    for line in r.stdout.split("\n"):
        if "=" in line and not line.startswith("[vpto_env"):
            k, v = line.split("=", 1)
            env[k] = v
            os.environ[k] = v
    return env

_env_cache = None

def _env(key):
    global _env_cache
    if _env_cache is None:
        _env_cache = _source_env()
    return _env_cache.get(key, os.environ.get(key, ""))


def _get_scalar_overrides(kernel: str) -> dict:
    """Load per-kernel scalar overrides from dsv4_golden_lib.SCALAR_VALUES.

    Returns {vN: int} for scalars whose value the harness cannot derive from
    .pto tensor-view shapes (partition_view offsets, base indices). The golden
    lib is the source of truth for these runtime values. Returns {} if the
    golden lib or the kernel's entry is absent (no override).
    """
    import importlib
    try:
        mod = importlib.import_module("dsv4_golden_lib")
    except Exception:
        # dsv4_golden_lib is in test_for_dsv4/ (ROOT); ensure importable.
        sys.path.insert(0, str(ROOT))
        try:
            mod = importlib.import_module("dsv4_golden_lib")
        except Exception:
            return {}
    return dict(getattr(mod, "SCALAR_VALUES", {}).get(kernel, {}))


def run_emitc_kernel(kernel: str, device: int, timeout: int = 300) -> dict:
    """Run one kernel through the EmitC route."""
    pto_path = ROOT / ".pto" / f"{kernel}.pto"
    # Split kernels (gate_aic, gate_aiv) live in a shared .pto (gate.pto)
    if not pto_path.exists():
        base = re.sub(r'_(aic|aiv)$', '', kernel)
        pto_path = ROOT / ".pto" / f"{base}.pto"
    build_dir = REPO_ROOT / "build_output" / f"emitc_{kernel}"
    run_dir = build_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Source env
    env_cmd = f"source {VPTO_ENV_SH} 2>/dev/null"
    ptoas_bin = _env("PTOAS_BIN") or "/data/liuzidi/PTOAS/build/tools/ptoas/ptoas"
    bisheng = _env("BISHENG_BIN") or f"{_env('ASCEND_HOME_PATH')}/bin/bisheng"
    ascend = _env("ASCEND_HOME_PATH") or "/usr/local/Ascend/cann-9.1.0-beta.3"
    pto_isa = _env("PTO_ISA_PATH") or "/data/liuzidi/pto-isa"
    tilelang = _env("TILELANG_PATH") or "/data/liuzidi/PTOAS/lib/TileOps"
    tilelang_pkg = _env("TILELANG_PKG") or "/data/liuzidi/PTOAS/tilelang-dsl/python"

    # Parse .pto
    try:
        info = parse_pto(pto_path, kernel=kernel)
    except Exception as e:
        return {"kernel": kernel, "pass": False, "compare_status": "parse_fail",
                "exit_code": -1, "error": f"cannot parse .pto: {e}"}

    func_name = info["func_name"]
    kind = "cube" if "cube" in pto_path.read_text()[:2000] else "vector"

    # Get outputs from golden_lib
    outputs = get_outputs_from_golden_lib(ROOT, kernel) or []
    # Dedupe
    seen = set()
    outputs_unique = []
    for o in outputs:
        if o not in seen:
            outputs_unique.append(o)
            seen.add(o)

    # Derive scalar values
    derived = derive_scalar_values(info)

    # Per-kernel scalar overrides from the golden lib (SCALAR_VALUES dict).
    # Some index scalars are NOT derivable from tensor-view shapes — e.g.
    # mtp_projection_norm %arg12 (v13) is a partition_view row OFFSET, not a
    # tensor dim, so derive_scalar_values() cannot recover it. The harness
    # would otherwise default it to ctx_len=8, making the kernel read rows
    # [8, 16) of an 8-row buffer → "DDR address of the MTE instruction is out
    # of range". The golden lib knows the correct value (0 for a single-block
    # launch) and exports it as SCALAR_VALUES[kernel][vN].
    scalar_overrides = _get_scalar_overrides(kernel)

    # Generate golden.py stub
    golden_py = f'''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from dsv4_golden_lib import run_case
if __name__ == "__main__":
    run_case({kernel!r})
'''
    (run_dir / "golden.py").write_text(golden_py)

    # Copy compare.py + validation_runtime.py (from golden/ subdir, which has
    # the bf16 ULP tolerance fix)
    golden_dir = ROOT / "golden"
    for fname in ["compare.py", "validation_runtime.py"]:
        src = golden_dir / fname
        dst = run_dir / fname
        try:
            shutil.copy(src, dst)
        except Exception:
            pass

    # Write outputs.txt
    (run_dir / "outputs.txt").write_text("\n".join(outputs_unique) + "\n")

    # Generate a PRELIMINARY main.cpp using parsed elem_counts (some may be 0
    # for dynamic shapes). The golden function needs main.cpp to exist so
    # load_case_meta() can parse buffer types + read_order. We'll regenerate
    # it with correct elem_counts after measuring the golden .bin file sizes.
    scalar_sem = []
    dumped_vals = {}  # {vN: int} for scalars overridden via SCALAR_VALUES
    for p in info["params"]:
        if p.get("is_ptr", True):
            continue  # only build sem for scalars (non-ptr params)
        if p["pto_type"] in ("i32", "index", "f32"):
            sig = p.get("sig_name", "") or p["name"]
            if "spmd" in sig:
                scalar_sem.append(None)
            elif p["name"] in scalar_overrides:
                # Golden lib supplied the runtime value directly (e.g. a
                # partition_view offset the harness can't derive).
                scalar_sem.append("dumped")
                dumped_vals[p["name"]] = scalar_overrides[p["name"]]
            elif derived.get(p["arg"]):
                scalar_sem.append("derived")
            else:
                scalar_sem.append("ctx_len")

    # For the preliminary main.cpp, set a placeholder count of 1 for any
    # 0-elem ptr so gen_main_cpp emits ReadFile3/WriteFile3 for it. This
    # ensures the golden function (which iterates meta.read_order from the
    # ReadFile3 calls) writes ALL buffers, including dynamic-shaped ones.
    # The placeholder is only used for the preliminary; the real main.cpp
    # gets the actual elem_counts from golden .bin sizes.
    elem_counts_initial = {}
    for p in info["params"]:
        if not p.get("is_ptr", True):
            continue
        n = p["name"]
        ec_val = info.get("elem_counts", {}).get(n, 0)
        if ec_val == 0:
            elem_counts_initial[n] = 1  # placeholder for golden detection
        else:
            elem_counts_initial[n] = ec_val

    try:
        main_cpp_prelim = gen_main_cpp(
            info, outputs_unique,
            load_vals={"ctx_len": 8, "derived": derived, "dumped": dumped_vals},
            consts={},
            scalar_sem_names=scalar_sem,
            elem_counts_override=elem_counts_initial,
        )
    except Exception as e:
        return {"kernel": kernel, "pass": False, "compare_status": "gen_fail",
                "exit_code": -1, "error": f"gen_main_cpp failed: {e}"}
    (run_dir / "main.cpp").write_text(main_cpp_prelim)

    # Step 1: Generate golden. The golden function calls load_case_meta()
    # which reads main.cpp to determine buffer types — so main.cpp must
    # exist before this step (even with 0-elem placeholders).
    golden_env = dict(os.environ)
    golden_env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'golden_parts'}:{golden_env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        ["bash", "-c", f"{env_cmd} && cd {run_dir} && python3 golden.py"],
        capture_output=True, text=True, timeout=timeout,
        env=golden_env)
    if r.returncode != 0:
        return {"kernel": kernel, "pass": False, "compare_status": "golden_fail",
                "exit_code": r.returncode, "error": f"golden.py failed: {r.stderr[-200:]}"}

    # Detect actual elem_counts from the generated .bin files.
    # Each buffer vN has a known pto_type → byte size → elem count.
    # This fixes dynamic-shaped buffers that were 0-elem in the .pto parse.
    elem_counts_from_golden = {}
    for p in info["params"]:
        if not p.get("is_ptr", True):
            continue
        n = p["name"]
        bin_path = run_dir / f"{n}.bin"
        if bin_path.exists():
            byte_size = bin_path.stat().st_size
            pt = p["pto_type"]
            elem_size = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i64": 8,
                          "i16": 2, "i8": 1, "u8": 1, "index": 8}.get(pt, 4)
            if byte_size > 0 and byte_size % elem_size == 0:
                elem_counts_from_golden[n] = byte_size // elem_size

    # Regenerate main.cpp with correct elem_counts from golden .bin sizes.
    # This fixes the null-GM-ptr crash for dynamic-shaped buffers.
    if elem_counts_from_golden:
        elem_counts_override = dict(elem_counts_initial)
        elem_counts_override.update(elem_counts_from_golden)
        try:
            main_cpp = gen_main_cpp(
                info, outputs_unique,
                load_vals={"ctx_len": 8, "derived": derived, "dumped": dumped_vals},
                consts={},
                scalar_sem_names=scalar_sem,
                elem_counts_override=elem_counts_override,
            )
        except Exception as e:
            return {"kernel": kernel, "pass": False, "compare_status": "gen_fail",
                    "exit_code": -1, "error": f"gen_main_cpp regen failed: {e}"}
        (run_dir / "main.cpp").write_text(main_cpp)

    # Step 2: ptoas EmitC → .cpp
    kernel_cpp = build_dir / f"{kernel}_kernel.cpp"
    pto_preprocessed = build_dir / f"{kernel}.pto"

    # Preprocess .pto (add pto.kernel_kind + pto.kernel attrs)
    pto_text = pto_path.read_text()
    pto_text = re.sub(
        r'module attributes \{pto\.target_arch = "a5"\}',
        f'module attributes {{pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<{kind}>}}',
        pto_text)
    pto_text = pto_text.replace(
        "attributes {pto.kernel_kind",
        "attributes {pto.kernel, pto.kernel_kind")
    pto_preprocessed.write_text(pto_text)

    ptoas_cmd = [
        ptoas_bin,
        "--pto-arch=a5", "--pto-level=level3", "--pto-backend=emitc",
        "--enable-insert-sync",
        "--enable-op-fusion=false",
        str(pto_preprocessed), "-o", str(kernel_cpp),
    ]
    daemon_env = dict(os.environ)
    # ptoas 0.59+ bundles mlir + ptodsl internally; just need build/python on PYTHONPATH
    ptoas_source = _env("PTOAS_SOURCE") or "/data/liuzidi/PTOAS"
    daemon_env["PYTHONPATH"] = f"{ptoas_source}/build/python:{ptoas_source}/ptodsl"
    daemon_env["ASCEND_HOME_PATH"] = ascend

    r = subprocess.run(ptoas_cmd, capture_output=True, text=True, timeout=timeout,
                       env=daemon_env, cwd=str(build_dir))
    if r.returncode != 0:
        err = r.stderr[-200:] if r.stderr else "unknown"
        return {"kernel": kernel, "pass": False, "compare_status": "lowering_fail",
                "exit_code": r.returncode, "error": f"ptoas emitc failed: {err}"}

    if not kernel_cpp.exists():
        return {"kernel": kernel, "pass": False, "compare_status": "lowering_fail",
                "exit_code": -1, "error": "ptoas produced no .cpp"}

    # Step 2b: Generate launch.cpp from the EmitC kernel.cpp's actual signature.
    # We CANNOT use setup_vpto.generate_launch_cpp() because parse_pto drops
    # f32 scalars (only handles i32/index), causing signature mismatch → NPU crash.
    # Instead, regex the `extern "C" __global__ AICORE void <name>(...)` line
    # from the EmitC-generated kernel.cpp for the TRUE signature.
    launch_cpp = build_dir / "launch.cpp"
    try:
        kernel_text = kernel_cpp.read_text()
        # Extract the extern declaration for THIS specific kernel. For split
        # .pto files (gate.pto has both gate_aic and gate_aiv), the generated
        # kernel.cpp contains both functions — we must match the one whose
        # name equals the requested kernel, not the first one.
        ext_match = re.search(
            rf'extern "C" __global__ AICORE void {re.escape(kernel)}\(([^)]+)\)',
            kernel_text)
        if ext_match:
            func_name = kernel
            raw_params = ext_match.group(1)
        else:
            # Fallback: first function (non-split kernels)
            ext_match = re.search(
                r'extern "C" __global__ AICORE void (\w+)\(([^)]+)\)', kernel_text)
            if not ext_match:
                raise ValueError(f"cannot find extern AICORE in {kernel_cpp.name}")
            func_name = ext_match.group(1)
            raw_params = ext_match.group(2)

        # Parse params: "__gm__ bfloat16_t* v1, __gm__ float* v2, float v5, int64_t v6"
        params = []
        for p in raw_params.split(","):
            p = p.strip()
            if not p:
                continue
            parts = p.rsplit(None, 1)
            if len(parts) == 2:
                full_type, name = parts
                is_ptr = "*" in full_type or "&" in full_type
                # Strip __gm__ prefix and * for the base type name
                dev_type = full_type.replace("__gm__ ", "").replace("__gm__", "")
                base_type = dev_type.replace("*", "").replace("&", "").strip()
                # Map device types to host-equivalent types so the host-facing
                # LaunchXxx() symbol has the SAME mangling as main.cpp's declaration.
                # main.cpp uses its own `typedef uint16_t bfloat16_t`, but when
                # launch.cpp is compiled with bisheng -xcce, bfloat16_t resolves to
                # the built-in __bf16, producing a different mangled name (__bf16 vs
                # unsigned short) → link error. Using uint16_t for the host-side
                # wrapper avoids the device type entirely.
                host_base = base_type
                if base_type in ("bfloat16_t", "float16_t"):
                    host_base = "uint16_t"
                params.append((base_type, host_base, name, is_ptr))
            else:
                params.append((p, p, f"v{len(params)+1}", False))

        cap_name = func_name[0].upper() + func_name[1:]

        # For _aiv split kernels: the _aiv (vector epilogue) reads from the
        # c2v pipe that _aic (cube matmul) writes to. Running _aiv alone
        # crashes because no _aic sends data through the pipe. Fix: declare
        # both functions and launch _aic first, then _aiv, on the same stream.
        is_aiv = kernel.endswith("_aiv")
        aic_name = kernel[:-4] + "_aic" if is_aiv else None
        aic_extern = ""
        aic_launch = ""
        if is_aiv:
            # Check if _aic function exists in the kernel.cpp
            aic_match = re.search(
                rf'extern "C" __global__ AICORE void {re.escape(aic_name)}\(([^)]+)\)',
                kernel_text)
            if aic_match:
                aic_raw_params = aic_match.group(1)
                aic_extern = f'extern "C" __global__ AICORE void {aic_name}({aic_raw_params});'
                # _aic has the same params, so call with the same args
                aic_call_args = []
                for dev_base, host_base, name, is_ptr in params:
                    if is_ptr:
                        aic_call_args.append(f"(__gm__ {dev_base}*){name}")
                    else:
                        aic_call_args.append(name)
                aic_launch = f"    {aic_name}<<<1, nullptr, stream>>>({', '.join(aic_call_args)});\n    aclrtSynchronizeStream(stream);\n"

        # extern decl matches kernel.cpp exactly (device types + __gm__)
        extern_decl = f'extern "C" __global__ AICORE void {func_name}({raw_params});'

        # Host-facing Launch signature uses HOST types (uint16_t not bfloat16_t)
        # to match main.cpp's own typedef, ensuring link symbol mangling agrees.
        host_decl_args = []
        for dev_base, host_base, name, is_ptr in params:
            ptr_suffix = "*" if is_ptr else ""
            host_decl_args.append(f"{host_base}{ptr_suffix} {name}")
        host_decl_args.append("void *stream")

        # Kernel call args: cast host ptrs to device __gm__ types.
        # C-style cast is needed because reinterpret_cast between unrelated
        # pointer types (uint16_t* → __gm__ bfloat16_t*) is rejected by bisheng.
        call_args = []
        for dev_base, host_base, name, is_ptr in params:
            if is_ptr:
                call_args.append(f"(__gm__ {dev_base}*){name}")
            else:
                call_args.append(name)

        # Build launch text: if _aiv, include _aic extern + launch _aic first
        aiv_extra = ""
        if is_aiv and aic_extern:
            aiv_extra = f"\n{aic_extern}\n"

        launch_text = f'''// Auto-generated launch.cpp for EmitC route
#include "pto/pto-inst.hpp"
#include "acl/acl.h"

{aiv_extra}{extern_decl}

void Launch{cap_name}({", ".join(host_decl_args)}) {{
{aic_launch}    {func_name}<<<1, nullptr, stream>>>({", ".join(call_args)});
}}
'''
        launch_cpp.write_text(launch_text)
    except Exception as e:
        return {"kernel": kernel, "pass": False, "compare_status": "gen_fail",
                "exit_code": -1, "error": f"launch.cpp generation failed: {e}"}

    # Step 3: bisheng compile kernel.cpp + launch.cpp → .o → .so
    kernel_o = build_dir / f"{kernel}_kernel.o"
    launch_o = build_dir / "launch.o"
    kernel_so = build_dir / f"lib{kernel}_kernel.so"

    bisheng_compile = [
        bisheng, "-c", "-fPIC", "-xcce", "-fenable-matrix", "--cce-aicore-enable-tl",
        "-fPIC", "-Xhost-start", "-Xhost-end",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE", "-std=c++17",
        "-Wno-macro-redefined", "-Wno-ignored-attributes",
        "-I", f"{ascend}/include", "-I", f"{ascend}/pkg_inc",
        "-I", f"{ascend}/pkg_inc/profiling",
        "-I", f"{ascend}/pkg_inc/runtime/runtime",
        "-I", f"{pto_isa}/include", "-I", f"{pto_isa}/tests/common",
        str(kernel_cpp), "-o", str(kernel_o),
    ]
    r = subprocess.run(bisheng_compile, capture_output=True, text=True, timeout=timeout,
                       env=daemon_env, cwd=str(build_dir))
    if r.returncode != 0:
        err = r.stderr[-300:] if r.stderr else "unknown"
        return {"kernel": kernel, "pass": False, "compare_status": "compile_fail",
                "exit_code": r.returncode, "error": f"bisheng compile: {err[-150:]}"}

    # Compile launch.cpp → launch.o (same flags as kernel)
    bisheng_launch = [bisheng] + bisheng_compile[1:-2] + [str(launch_cpp), "-o", str(launch_o)]
    # Replace kernel_cpp with launch_cpp in the arg list
    bisheng_launch = [bisheng, "-c", "-fPIC", "-xcce", "-fenable-matrix", "--cce-aicore-enable-tl",
                      "-fPIC", "-Xhost-start", "-Xhost-end",
                      "-mllvm", "-cce-aicore-stack-size=0x8000",
                      "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                      "--cce-aicore-arch=dav-c310-vec",
                      "-DREGISTER_BASE", "-std=c++17",
                      "-Wno-macro-redefined", "-Wno-ignored-attributes",
                      "-I", f"{ascend}/include", "-I", f"{ascend}/pkg_inc",
                      "-I", f"{ascend}/pkg_inc/profiling",
                      "-I", f"{ascend}/pkg_inc/runtime/runtime",
                      "-I", f"{pto_isa}/include", "-I", f"{pto_isa}/tests/common",
                      str(launch_cpp), "-o", str(launch_o)]
    r = subprocess.run(bisheng_launch, capture_output=True, text=True, timeout=timeout,
                       env=daemon_env, cwd=str(build_dir))
    if r.returncode != 0:
        err = r.stderr[-300:] if r.stderr else "unknown"
        return {"kernel": kernel, "pass": False, "compare_status": "compile_fail",
                "exit_code": r.returncode, "error": f"bisheng launch: {err[-150:]}"}

    bisheng_link = [
        bisheng, "-fPIC", "-s", "-Wl,-z,relro", "-Wl,-z,now",
        "--cce-fatobj-link", "-shared",
        f"-Wl,-soname,lib{kernel}_kernel.so",
        "-L", f"{ascend}/lib64", "-Wl,-rpath", f"{ascend}/lib64",
        "-o", str(kernel_so), str(kernel_o), str(launch_o),
        "-Wl,--no-as-needed", "-lruntime",
    ]
    r = subprocess.run(bisheng_link, capture_output=True, text=True, timeout=timeout,
                       env=daemon_env, cwd=str(build_dir))
    if r.returncode != 0:
        err = r.stderr[-200:] if r.stderr else "unknown"
        return {"kernel": kernel, "pass": False, "compare_status": "link_fail",
                "exit_code": r.returncode, "error": f"bisheng link: {err[-150:]}"}

    # Step 4: Compile main.cpp → host binary (use bash -c with proper -Wl,-rpath,path)
    host_bin = build_dir / kernel
    bisheng_main_args = " ".join([
        bisheng, "-xc++", "-include", "stdint.h", "-include", "stddef.h",
        "-std=c++17", str(run_dir / "main.cpp"),
        "-I", f"{ascend}/include", "-I", f"{pto_isa}/include",
        "-I", f"{pto_isa}/tests/common",
        "-L", str(build_dir), "-L", f"{ascend}/lib64",
        f"-Wl,-rpath,{build_dir}", f"-Wl,-rpath,{ascend}/lib64",
        "-o", str(host_bin),
        f"-l{kernel}_kernel",
        "-Wl,--allow-shlib-undefined", "-Wl,--no-as-needed", "-lruntime",
        "-lstdc++", "-lascendcl", "-lm", "-ltiling_api", "-lplatform",
        "-lc_sec", "-ldl", "-lnnopbase",
    ])
    bash_cmd = f'source {VPTO_ENV_SH} 2>/dev/null && cd {build_dir} && {bisheng_main_args}'
    r = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr[-200:] if r.stderr else "unknown"
        return {"kernel": kernel, "pass": False, "compare_status": "link_fail",
                "exit_code": r.returncode, "error": f"bisheng main: {err[-150:]}"}

    # Step 5: NPU run
    run_env = dict(os.environ)
    run_env["ACL_DEVICE_ID"] = str(device)
    run_env["LD_LIBRARY_PATH"] = f"{build_dir}:{ascend}/lib64:{run_env.get('LD_LIBRARY_PATH', '')}"
    r = subprocess.run([str(host_bin)], capture_output=True, text=True, timeout=timeout,
                       env=run_env, cwd=str(run_dir))
    if r.returncode != 0:
        err = r.stderr[-200:] if r.stderr else f"exit {r.returncode}"
        return {"kernel": kernel, "pass": False, "compare_status": "npu_run_fail",
                "exit_code": r.returncode, "error": err[:150]}

    # Step 6: Compare
    compare_env = dict(os.environ)
    compare_env["PYTHONPATH"] = f"{build_dir}:{ROOT}:{ROOT / 'golden'}"
    compare_py = run_dir / "compare.py"
    r = subprocess.run(["python3", str(compare_py)], capture_output=True, text=True,
                       timeout=timeout, env=compare_env, cwd=str(run_dir))
    combined = r.stdout + "\n" + r.stderr
    if "compare passed" in combined:
        # Try max_diff (fp32) first, then max_ulp (bf16)
        m = re.search(r"max_diff=([\d.eE+-]+)", combined)
        if not m:
            m = re.search(r"max_ulp=(\d+)", combined)
        m2 = re.search(r"\[TIMING\] kernel=\S+ time=([\d.]+)\s*ms", r.stdout)
        return {"kernel": kernel, "pass": True, "compare_status": "pass",
                "exit_code": 0, "max_diff": float(m.group(1)) if m else 0.0,
                "timing_ms": float(m2.group(1)) if m2 else None, "error": ""}
    elif "compare failed" in combined:
        m = re.search(r"max_diff=([\d.eE+-]+)", combined)
        if not m:
            m = re.search(r"max_ulp=(\d+)", combined)
        err = "golden compare failed"
        # Include max_ulp info in error if available
        m2 = re.search(r"max_ulp=(\d+).*idx=(\d+)", combined)
        if m2:
            err = f"max_ulp={m2.group(1)} idx={m2.group(2)}"
        return {"kernel": kernel, "pass": False, "compare_status": "fail",
                "exit_code": r.returncode, "max_diff": float(m.group(1)) if m else None,
                "error": err}
    else:
        return {"kernel": kernel, "pass": False, "compare_status": "compare_error",
                "exit_code": r.returncode, "error": combined[-150:]}
