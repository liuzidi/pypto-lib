"""run_jit_golden.py — generate golden inputs/outputs from a DSV4 model .py.

The DSV4 models do NOT use the test_for_ptoas `*_golden_lib.py` contract
(BUILDERS + run_case). Instead each model .py exposes the run_jit-style
contract:
  - `<jit_fn>`              — the @pl.jit function; its param order matches the
                              .pto kernel's ptr-arg order (trailing index +
                              __pypto_spmd_* scalars are dropped).
  - `build_tensor_specs(B, S)` -> [TensorSpec...] in the same order.
  - `golden_fn(tensors)`    — fills outputs in-place.
  - `MODES = {"decode": (B,S), "prefill": (B,S)}`.

Two entry points:
  - resolve_meta(model_py, mode, pto_info) -> dict: lightweight, no torch.
    Resolves outputs (is_output specs), ctx_len = B*S, np_types. Called by
    the skill BEFORE main.cpp generation (main.cpp needs to know which ptrs
    are outputs to allocate+write them back).
  - dump_bins(model_py, mode, run_dir) -> None: heavy, needs torch + golden
    package. Runs golden_fn on torch tensors, dumps vN.bin / golden_vN.bin
    into run_dir. Called as a subprocess (step 6 of the skill).

Only leaf modules are supported: the .pto kernel's ptr-arg count must equal
the model's TensorSpec count (ptr-args == module specs, 1:1). For multi-
kernel modules (ptr-args are intermediates, not module inputs), this emits a
clear error — those need intermediate capture (Phase 5).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _venv_python() -> str:
    """Find the repo's .venv python (needed for torch + golden package).

    The skill runs under whatever python3 is on PATH (often /usr/bin/python3,
    which has no torch). DSV4 golden generation needs torch + the repo golden
    package, both installed in pypto-lib/.venv. Resolve it relative to the
    skill dir (.claude/skills/vpto-board-validate/ -> repo root -> .venv).
    """
    # skill dir is <repo>/.claude/skills/vpto-board-validate/
    repo_root = Path(__file__).resolve().parents[4]
    venv_py = repo_root / ".venv" / "bin" / "python3"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable  # fallback (may lack torch; skill warns if so)


def _import_model(model_py: Path):
    """Import a model .py by path. Caller sets sys.path for config/golden."""
    spec = importlib.util.spec_from_file_location("_vpto_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# torch dtype -> numpy dtype for .bin dump (bfloat16 has no numpy dtype;
# store as uint16 bit pattern, matching validation_runtime's bfloat16_t mapping)
_TORCH_TO_NP = {
    "BFloat16": "uint16",
    "Float32": "float32",
    "Float16": "uint16",
    "Int8": "int8",
    "Int32": "int32",
    "Int64": "int64",
}

# pto ptr dtype -> numpy dtype (mirrors validation_runtime _HOST_TYPE_TO_NP)
_PTO_TO_NP = {
    "bf16": "uint16", "f16": "uint16", "f32": "float32",
    "i8": "int8", "i32": "int32", "i64": "int64",
}


def _np_for_pto(pto_type: str) -> str:
    return _PTO_TO_NP.get(pto_type, "float32")


def resolve_meta(model_py: Path, mode: str, pto_info: dict) -> dict:
    """Lightweight metadata resolution (no torch import).

    Returns {outputs, ctx_len, ctx_blocks, np_types, B, S, T, spec_names}.
    Raises ValueError if not a leaf module or missing expected attrs.

    DSV4 models define MODES = {"decode": (DECODE_BATCH, DECODE_SEQ), ...}
    inside `if __name__ == "__main__"`, so importing the module as a library
    does not expose MODES. We build B/S directly from config.py constants
    (DECODE_BATCH/DECODE_SEQ, PREFILL_BATCH/PREFILL_SEQ) via a cheap
    subprocess that imports the model's config.py.
    """
    import subprocess, json
    model_dir = model_py.parent.resolve()
    repo_root = model_dir.parents[1]  # pypto-lib/
    venv_py = _venv_python()

    # B, S from config.py (the model's own config, in model_dir).
    mode_attrs = {
        "decode": ("DECODE_BATCH", "DECODE_SEQ"),
        "prefill": ("PREFILL_BATCH", "PREFILL_SEQ"),
    }
    if mode not in mode_attrs:
        raise ValueError(f"mode {mode!r} not in {list(mode_attrs)}")
    b_attr, s_attr = mode_attrs[mode]
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(model_dir)!r})\n"
        "spec = importlib.util.spec_from_file_location('cfg', "
        f"{str((model_dir / 'config.py').absolute())!r})\n"
        "cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)\n"
        f"print(getattr(cfg, {b_attr!r}), getattr(cfg, {s_attr!r}))\n"
    )
    r = subprocess.run([venv_py, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"failed to read {b_attr}/{s_attr} from {model_dir}/config.py:\n{r.stderr[-600:]}")
    parts = r.stdout.strip().split()
    B, S = int(parts[0]), int(parts[1])
    T = B * S

    # Spec list: build_tensor_specs needs the golden package (TensorSpec).
    code2 = (
        "import importlib.util, sys, json\n"
        f"sys.path.insert(0, {str(model_dir)!r})\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(model_py.absolute())!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"specs = m.build_tensor_specs({B}, {S})\n"
        "print(json.dumps([{'name': s.name, 'is_output': getattr(s,'is_output',False), "
        "'dtype': str(s.dtype), 'shape': list(s.shape)} for s in specs]))\n"
    )
    r2 = subprocess.run([venv_py, "-c", code2], capture_output=True, text=True)
    if r2.returncode != 0:
        raise ValueError(f"failed to get specs from {model_py}:\n{r2.stderr[-800:]}")
    specs = json.loads(r2.stdout.strip().splitlines()[-1])

    # leaf-module contract: ptr-arg count == spec count
    ptr_params = [p for p in pto_info["params"] if p["pto_type"] not in ("i32", "index")]
    if len(ptr_params) != len(specs):
        raise ValueError(
            f"not a leaf module: .pto has {len(ptr_params)} ptr args but "
            f"{model_py.name} exposes {len(specs)} TensorSpecs. This kernel's "
            f"inputs are intermediates from a preceding kernel; the module "
            f"golden_fn cannot produce them. Needs intermediate capture "
            f"(Phase 5, not yet supported)."
        )

    # outputs: v<i+1> for each is_output spec (specs are in ptr-arg order)
    outputs = [f"v{i+1}" for i, s in enumerate(specs) if s["is_output"]]
    # np_types per v-name, from the .pto ptr dtype (ptr order == spec order)
    np_types = {f"v{i+1}": _np_for_pto(ptr_params[i]["pto_type"])
                for i in range(len(ptr_params))}
    # elem_counts: product of spec shape dims (the full GM allocation size).
    # DSV4 .pto have dynamic shapes ([%arg3, D] where %arg3 = T) that
    # parse_pto can't reduce to a static count; the spec shape carries the
    # resolved dims (T*D etc.), so main.cpp allocates the right size.
    def _prod(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n
    elem_counts = {f"v{i+1}": _prod(specs[i]["shape"])
                   for i in range(len(specs))}

    return {
        "outputs": outputs,
        "ctx_len": T,
        "ctx_blocks": None,  # main.cpp only needs ctx_len; tiling is internal
        "np_types": np_types,
        "elem_counts": elem_counts,
        "B": B, "S": S, "T": T,
        "spec_names": [s["name"] for s in specs],
        "is_output": {s["name"]: s["is_output"] for s in specs},
    }


def dump_bins(model_py: Path, mode: str, run_dir: Path) -> None:
    """Heavy golden generation: import the model under torch + golden package,
    run golden_fn, write vN.bin (inputs) + golden_vN.bin (outputs) into run_dir.

    run_dir is CWD when called as a subprocess (the skill sets cwd=run_dir).
    DSV4 MODES is __main__-only, so B/S come from config.py directly.
    """
    model_dir = model_py.parent.resolve()
    repo_root = model_dir.parents[1]  # pypto-lib/
    # Insert repo root + model dir BEFORE importing golden (the repo golden
    # package lives at <repo>/golden/, not on the subprocess's default path).
    for p in (str(model_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import numpy as np
    from golden import TensorSpec  # repo golden package

    mod = _import_model(model_py)

    # B/S from config (MODES is __main__-only; build directly from config attrs)
    import importlib.util as _ilu
    _cspec = _ilu.spec_from_file_location("_cfg", model_dir / "config.py")
    cfg = _ilu.module_from_spec(_cspec); _cspec.loader.exec_module(cfg)  # type: ignore[union-attr]
    mode_attrs = {"decode": ("DECODE_BATCH", "DECODE_SEQ"),
                  "prefill": ("PREFILL_BATCH", "PREFILL_SEQ")}
    b_attr, s_attr = mode_attrs[mode]
    B, S = getattr(cfg, b_attr), getattr(cfg, s_attr)

    specs = mod.build_tensor_specs(B, S)

    golden_fn = _find_golden_fn(mod, model_py.stem)
    if golden_fn is None:
        raise ValueError(f"{model_py}: no golden_<name>_test fn found")

    tensors: dict[str, torch.Tensor] = {}
    for spec in specs:
        iv = spec.init_value
        if callable(iv):
            val = iv()
        elif iv is None:
            # output specs (is_output) have no init value — zero-fill; the
            # golden_fn fills them in-place.
            val = torch.zeros(spec.shape, dtype=spec.dtype)
        else:
            val = iv
        tensors[spec.name] = torch.as_tensor(val, dtype=spec.dtype).clone()

    golden_fn(tensors)

    for i, spec in enumerate(specs):
        vname = f"v{i+1}"
        arr = _torch_to_np(tensors[spec.name])
        arr.tofile(run_dir / f"{vname}.bin")
        if getattr(spec, "is_output", False):
            arr.tofile(run_dir / f"golden_{vname}.bin")


def _find_golden_fn(mod, model_stem: str):
    """DSV4 convention: golden_<name>_test where <name> matches the model.
    E.g. rmsnorm.py -> golden_rms_norm_test; hc_pre.py -> golden_hc_pre_test."""
    candidates = [
        f"golden_{model_stem}_test",
        f"golden_{model_stem}_test",
    ]
    # also try the jit fn stem (rmsnorm.py -> rms_norm_test -> golden_rms_norm_test)
    for name in dir(mod):
        if name.startswith("golden_") and name.endswith("_test"):
            candidates.append(name)
    for c in candidates:
        fn = getattr(mod, c, None)
        if fn is not None:
            return fn
    return None


def _torch_to_np(torch_tensor):
    """torch tensor -> numpy array. bfloat16 -> uint16 bit pattern (bitcast,
    NOT numeric cast — torch.to(uint16) saturates bf16 values to 0/65535)."""
    import torch
    if torch_tensor.dtype == torch.bfloat16:
        # view-as-uint16 reinterprets the raw bf16 bit pattern, matching what
        # validation_runtime.write_buffers expects (bfloat16_t -> uint16).
        return torch_tensor.view(torch.uint16).detach().cpu().numpy()
    if torch_tensor.dtype == torch.float16:
        return torch_tensor.view(torch.uint16).detach().cpu().numpy()
    return torch_tensor.detach().cpu().numpy()
