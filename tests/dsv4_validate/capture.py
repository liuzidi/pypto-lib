# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Phase 5 intermediate-GM capture driver for DSV4 multi-kernel modules.

Runs a DSV4 module once via run_jit (Route 1/simpler) with
enable_dump_args=2, then harvests the per-task tensor payloads for ONE
target inner kernel from the dump and writes them as vN.bin / golden_vN.bin
into a run_dir ready for vpto_run.py --captured-dump replay (Route 2).

Why Route 1 for capture: the DFX args-dump collector lives in the
simpler/pypto runtime C++ (args_dump_aicpu.cpp); VPTO Route 2 bypasses
simpler's runtime entirely, so capture must use Route 1. The kernel under
test still runs on Route 2 in the replay step.

Usage (typically called by validate.py, not directly):
    python tests/dsv4_validate/capture.py \
        --model-py models/deepseek_v4_pro/decode_attention_csa.py \
        --kernel hc_post --mode decode --device 0 \
        --run-dir build_output/vpto_hc_post/run
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VPTO_ENV_SH = REPO_ROOT / "scripts" / "vpto_env.sh"
PTOAS_ROOT = Path("/data/liuzidi/PTOAS/build311/tools/ptoas")


def _load_module(model_py: Path):
    """Import a DSV4 model .py (needs config + golden on sys.path)."""
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(model_py.parent))
    spec = importlib.util.spec_from_file_location("_dsv4_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_module_with_dump(mod, model_py: Path, mode: str, device: int,
                          work_dir: Path, jit_entry: str | None = None) -> Path:
    """Run the module via run_jit with full args dump; return the work_dir.

    jit_entry: the @pl.jit entry fn name (e.g. attention_csa_test). If None,
    falls back to <model_py.stem>_test then any *_test callable.
    """
    from golden import run_jit
    work_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT / ".claude" / "skills" /
                           "vpto-board-validate" / "lib"))
    import run_jit_golden
    B, S = _bs_from_config(model_py.parent / "config.py", mode)
    specs = run_jit_golden._call_build_tensor_specs(mod, B, S)
    golden_fn = run_jit_golden._find_golden_fn(mod, model_py.stem)
    if golden_fn is None:
        raise RuntimeError(f"no golden_fn found in {model_py}")
    r = run_jit(
        fn=_jit_entry(mod, model_py.stem, jit_entry),
        specs=specs, golden_fn=golden_fn, save_data=True,
        runtime_cfg=dict(platform="a5", device_id=device,
                         enable_dump_args=2, enable_dep_gen=True),
        rtol=1e-2, atol=1e-2,
    )
    wd = getattr(r, "work_dir", None)
    if wd is None:
        raise RuntimeError("run_jit returned no work_dir")
    return Path(wd)


def _jit_entry(mod, stem: str, override: str | None = None):
    """Find the @pl.jit entry fn. Priority: override -> <stem>_test ->
    first *_test callable in dir(mod)."""
    if override and getattr(mod, override, None) is not None:
        return getattr(mod, override)
    cand = getattr(mod, f"{stem}_test", None)
    if cand is not None:
        return cand
    for n in dir(mod):
        if n.endswith("_test") and callable(getattr(mod, n)):
            return getattr(mod, n)
    raise RuntimeError(f"no entry fn found (tried {override}, {stem}_test)")


def _bs_from_config(config_py: Path, mode: str):
    model_dir = str(config_py.parent.absolute())
    b_attr = "DECODE_BATCH" if mode == "decode" else "PREFILL_BATCH"
    s_attr = "DECODE_SEQ" if mode == "decode" else "PREFILL_SEQ"
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {model_dir!r})\n"
        f"spec = importlib.util.spec_from_file_location('cfg', {str(config_py.absolute())!r})\n"
        "cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)\n"
        f"print(getattr(cfg, {b_attr!r}), getattr(cfg, {s_attr!r}))\n"
    )
    r = subprocess.run([str(REPO_ROOT / ".venv" / "bin" / "python3"), "-c", code],
                      capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"config read failed: {r.stderr[-300:]}")
    parts = r.stdout.strip().split()
    return int(parts[0]), int(parts[1])


_DTYPE_TO_NP = {
    "FLOAT32": "float32", "BFLOAT16": "uint16", "FLOAT16": "uint16",
    "INT8": "int8", "INT32": "int32", "INT64": "int64",
}


def harvest_kernel(dump_dir: Path, work_dir: Path, kernel: str,
                   run_dir: Path, pto_path: Path) -> dict:
    """Read args_dump for one kernel; write vN.bin / golden_vN.bin into run_dir.

    vN is numbered by the kernel's ptr-arg order (arg_index 0 -> v1, ...).
    Inputs (role=input, before_dispatch) become vN.bin; outputs
    (role=output, after_completion) become golden_vN.bin. When a ptr is
    both (inout) or is an output-only ptr (no input record), vN.bin is
    zero-filled so main.cpp can read it; golden_vN.bin carries the reference.
    Returns a meta dict {outputs, np_types, elem_counts} for main.cpp.
    """
    args_dump_dir = dump_dir / "dfx_outputs" / "args_dump"
    manifest = json.loads((args_dump_dir / "args_dump.json").read_text())
    # name_map: callable_id -> kernel name
    name_map_path = next((dump_dir / "dfx_outputs").glob("name_map_*.json"), None)
    if name_map_path is None:
        raise RuntimeError(f"no name_map_*.json in {dump_dir}/dfx_outputs")
    name_map = json.loads(name_map_path.read_text())
    cid2name = name_map["callable_id_to_name"]
    target_fid = None
    for k, v in cid2name.items():
        if v == kernel:
            target_fid = int(k)
            break
    if target_fid is None:
        raise RuntimeError(f"kernel {kernel!r} not in name_map {cid2name}")

    args = manifest["args"]
    # group by arg_index, take first record of each (copies are identical)
    inputs: dict[int, dict] = {}
    outputs: dict[int, dict] = {}
    for a in args:
        if target_fid not in a["func_id"]:
            continue
        ai = a["arg_index"]
        if a["role"] == "input" and a["stage"] == "before_dispatch":
            if ai not in inputs:
                inputs[ai] = a
        elif a["role"] == "output" and a["stage"] == "after_completion":
            if ai not in outputs:
                outputs[ai] = a

    all_indices = sorted(set(inputs) | set(outputs))
    if not all_indices:
        raise RuntimeError(f"no dump records for kernel {kernel} (fid {target_fid})")

    bin_path = args_dump_dir / manifest.get("bin_file", "args.bin")
    run_dir.mkdir(parents=True, exist_ok=True)
    np_types = {}
    elem_counts = {}
    out_names = []
    import numpy as np
    with open(bin_path, "rb") as bf:
        for ai in all_indices:
            vname = f"v{ai + 1}"
            if ai in inputs:
                a = inputs[ai]
                bf.seek(a["bin_offset"])
                data = bf.read(a["bin_size"])
                (run_dir / f"{vname}.bin").write_bytes(data)
                np_types[vname] = _DTYPE_TO_NP.get(a["dtype"], "float32")
                elem_counts[vname] = int(a["numel"])
            else:
                # output-only ptr: zero-fill vN.bin (main.cpp reads it as input)
                a = outputs[ai]
                np_dt = _DTYPE_TO_NP.get(a["dtype"], "float32")
                np_types[vname] = np_dt
                elem_counts[vname] = int(a["numel"])
                zero = np.zeros(a["numel"], dtype=np_dt)
                (run_dir / f"{vname}.bin").write_bytes(zero.tobytes())
            if ai in outputs:
                a = outputs[ai]
                bf.seek(a["bin_offset"])
                gdata = bf.read(a["bin_size"])
                (run_dir / f"golden_{vname}.bin").write_bytes(gdata)
                out_names.append(vname)
    return {
        "outputs": out_names,
        "np_types": np_types,
        "elem_counts": elem_counts,
        "func_id": target_fid,
        "n_inputs": len(inputs),
        "n_outputs": len(outputs),
    }


_CAPTURE_META_NAME = "capture_meta.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 intermediate-GM capture.")
    ap.add_argument("--model-py", required=True, type=Path)
    ap.add_argument("--kernel", default=None,
                    help="target inner kernel name (must be in name_map). "
                         "Required unless --capture-only or --dump-dir is set.")
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="where to write vN.bin / golden_vN.bin for replay")
    ap.add_argument("--pto", type=Path, default=None,
                    help="the kernel's .pto (for replay meta)")
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="reuse an existing dump work_dir instead of re-running")
    ap.add_argument("--capture-only", action="store_true",
                    help="only run the module dump (no harvest); writes the "
                         "dump work_dir. Used by validate.py to run the dump "
                         "in a subprocess so CANN env is inherited correctly.")
    ap.add_argument("--jit-entry", default=None,
                    help="override the @pl.jit entry fn name "
                         "(e.g. attention_csa_test)")
    args = ap.parse_args()

    if args.capture_only:
        mod = _load_module(args.model_py)
        # default dump work_dir: phase5_dump_<model stem>
        dump_wd = (REPO_ROOT / "build_output" /
                   f"phase5_dump_{args.model_py.stem}")
        real_wd = _run_module_with_dump(
            mod, args.model_py, args.mode, args.device, dump_wd,
            jit_entry=args.jit_entry)
        # run_jit picks its own work_dir (compiled.output_dir), which differs
        # from our requested dump_wd. Symlink so the caller (validate.py) can
        # find the dump at the predictable phase5_dump_<stem> path.
        if real_wd.resolve() != dump_wd.resolve():
            if dump_wd.is_symlink() or dump_wd.exists():
                if dump_wd.is_dir() and not dump_wd.is_symlink():
                    shutil.rmtree(dump_wd, ignore_errors=True)
                else:
                    dump_wd.unlink(missing_ok=True)
            dump_wd.parent.mkdir(parents=True, exist_ok=True)
            dump_wd.symlink_to(real_wd.resolve())
        print(f"[capture] capture-only: dump at {dump_wd} -> {real_wd}")
        return 0

    if args.dump_dir is not None:
        dump_wd = args.dump_dir
    else:
        mod = _load_module(args.model_py)
        dump_wd = _run_module_with_dump(mod, args.model_py, args.mode, args.device,
                                        REPO_ROOT / "build_output" /
                                        f"phase5_dump_{args.model_py.stem}",
                                        jit_entry=args.jit_entry)
        print(f"[capture] module dump at {dump_wd}")

    if not args.kernel or not args.run_dir or not args.pto:
        print("[capture] ERROR: --kernel, --run-dir, --pto required "
              "for harvest (or use --capture-only)", file=sys.stderr)
        return 2
    meta = harvest_kernel(dump_wd, dump_wd, args.kernel, args.run_dir, args.pto)
    # write capture_meta.json so vpto_run.py --captured-dump can read
    # outputs/np_types/elem_counts without calling resolve_meta (which would
    # fail: non-leaf kernels have no ptr->spec map).
    (args.run_dir / _CAPTURE_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[capture] {args.kernel}: {meta['n_inputs']} inputs, "
          f"{meta['n_outputs']} outputs, func_id={meta['func_id']}")
    print(f"[capture] outputs (ptr-order vN): {meta['outputs']}")
    print(f"[capture] wrote bins + {_CAPTURE_META_NAME} into {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
