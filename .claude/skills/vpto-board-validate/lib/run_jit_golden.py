# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""run_jit_golden.py — generate golden inputs/outputs from a DSV4 model .py.

The DSV4 models do NOT use the test_for_ptoas `*_golden_lib.py` contract
(BUILDERS + run_case). Instead each model .py exposes the run_jit-style
contract:
  - `<jit_fn>`              — the @pl.jit function; its param order matches the
                              .pto kernel's ptr-arg order (trailing index +
                              __pypto_spmd_* scalars are dropped).
  - `build_tensor_specs(...)` -> [TensorSpec...] in the same order. The
                              signature varies across models (8 variants:
                              (B,S); (); (start_pos=None);
                              (layer_id=0, num_tokens=T); (compress_ratio=4);
                              (batch=, seq=); etc.). _call_build_tensor_specs
                              inspects the signature and fills kwargs from
                              config-derived B/S + mode-agnostic defaults.
  - `golden_<name>(tensors)` — fills outputs in-place. The `_test` suffix is
                              optional; _find_golden_fn tries both
                              `golden_<stem>_test` and `golden_<stem>`, then
                              falls back to a unique `golden_*` callable.

Two entry points:
  - resolve_meta(model_py, mode, pto_info) -> dict: lightweight, no torch.
    Resolves outputs (is_output specs), ctx_len = B*S, np_types. Called by
    the skill BEFORE main.cpp generation (main.cpp needs to know which ptrs
    are outputs to allocate+write them back). B/S come from config.py
    constants (DECODE_BATCH/DECODE_SEQ, PREFILL_BATCH/PREFILL_SEQ), NOT from
    MODES (which is __main__-only and unused here).
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


# A snippet (run inside the .venv subprocess) that calls build_tensor_specs
# with whatever signature the model declares. DSV4 models use 8 different
# signatures (B,S; () ; start_pos=None; layer_id=0,num_tokens=T;
# compress_ratio=4; batch=,seq=; etc.), so we cannot pass (B,S) positionally.
# This mirrors _call_build_tensor_specs but is emitted as source for resolve_meta
# (which runs torch-free and must serialize the call into a subprocess).
_BTS_DISPATCH_SRC = """
import inspect
def _bts(mod, B, S):
    sig = inspect.signature(mod.build_tensor_specs)
    kw = {}
    for n, p in sig.parameters.items():
        if p.default is not inspect.Parameter.empty:
            kw[n] = p.default
        elif n in ("B", "batch"):
            kw[n] = B
        elif n in ("S", "seq"):
            kw[n] = S
        elif n in ("num_tokens", "T"):
            kw[n] = B * S
        elif n == "layer_id":
            kw[n] = 0
        elif n == "start_pos":
            kw[n] = 0
        elif n == "compress_ratio":
            kw[n] = 4
        else:
            raise ValueError(f"build_tensor_specs required param {n!r} no default")
    return mod.build_tensor_specs(**kw)
"""


def _call_build_tensor_specs(mod, B: int, S: int):
    """Call mod.build_tensor_specs with whatever signature it declares.

    DSV4 models use 8 different signatures (only 3/23 modules use (B,S)).
    We inspect the params and fill defaults from config-derived B/S plus
    mode-agnostic defaults (layer_id=0, start_pos=0, compress_ratio=4).
    Raises ValueError if a required param has no default and isn't one we
    know how to fill.
    """
    import inspect
    sig = inspect.signature(mod.build_tensor_specs)
    kwargs: dict = {}
    for name, p in sig.parameters.items():
        if p.default is not inspect.Parameter.empty:
            kwargs[name] = p.default            # respect module's own default
        elif name in ("B", "batch"):
            kwargs[name] = B
        elif name in ("S", "seq"):
            kwargs[name] = S
        elif name in ("num_tokens", "T"):
            kwargs[name] = B * S
        elif name == "layer_id":
            kwargs[name] = 0
        elif name == "start_pos":
            kwargs[name] = 0
        elif name == "compress_ratio":
            kwargs[name] = 4
        else:
            raise ValueError(
                f"build_tensor_specs has required param {name!r} with no "
                f"known default; cannot call generically.")
    return mod.build_tensor_specs(**kwargs)


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
    # DSV4 build_tensor_specs has 8 different signatures across models; we
    # cannot pass (B,S) positionally. Emit the dispatch helper into the
    # subprocess so it fills kwargs from signature defaults + B/S/T.
    code2 = (
        "import importlib.util, sys, json, inspect\n"
        f"sys.path.insert(0, {str(model_dir)!r})\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(model_py.absolute())!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        + _BTS_DISPATCH_SRC
        + f"specs = _bts(m, {B}, {S})\n"
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

    # Map each ptr to its spec. DSV4 .pto ptr order does NOT always match the
    # model's build_tensor_specs order (e.g. hc_post: ptrs are [y, post, x,
    # comb, residual] but specs are [x, residual, post, comb, y]). main.cpp
    # numbers buffers by ptr order (v1=%arg0, v2=%arg1, ...) and dump_bins must
    # write the matching spec's tensor to each vN. See _map_ptrs_to_specs.
    ptr_to_spec = _map_ptrs_to_specs(ptr_params, specs)
    spec_names = [s["name"] for s in specs]

    # vN is in ptr order (v1=%arg0...); map to the spec's dtype/shape/output.
    def _prod(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n
    outputs = [f"v{i+1}" for i, si in enumerate(ptr_to_spec)
               if specs[si]["is_output"]]
    np_types = {f"v{i+1}": _np_for_pto(_pto_of_torch(specs[si]["dtype"]))
                for i, si in enumerate(ptr_to_spec)}
    elem_counts = {f"v{i+1}": _prod(specs[ptr_to_spec[i]]["shape"])
                   for i in range(len(ptr_params))}

    return {
        "outputs": outputs,
        "ctx_len": T,
        "ctx_blocks": None,  # main.cpp only needs ctx_len; tiling is internal
        "np_types": np_types,
        "elem_counts": elem_counts,
        "B": B, "S": S, "T": T,
        "spec_names": spec_names,
        "is_output": {s["name"]: s["is_output"] for s in specs},
        "ptr_to_spec": ptr_to_spec,
        "ptr_spec_names": [spec_names[si] for si in ptr_to_spec],
    }


def _pto_of_torch(td: str) -> str:
    """torch dtype string (e.g. 'torch.bfloat16') -> pto ptr dtype ('bf16')."""
    return {"torch.bfloat16": "bf16", "torch.float16": "f16",
            "torch.float32": "f32", "torch.int8": "i8",
            "torch.int32": "i32", "torch.int64": "i64"}.get(td, td)


def _map_ptrs_to_specs(ptr_params: list, specs: list) -> list:
    """Map each ptr (in .pto signature order) to its spec (in build_tensor_specs
    order), returning a list of spec indices parallel to ptr_params.

    DSV4 ptr order does NOT always equal spec order (hc_post: ptrs are
    [y, post, x, comb, residual] vs specs [x, residual, post, comb, y]). We
    match by longest-prefix of the ptr's tensor-view stem against spec names:
    the view stem (e.g. "y_flat", "x", "residual_flat") comes from the jit-fn
    param name, which equals the TensorSpec name. "y_flat" starts with spec
    "y"; "residual_flat" starts with "residual"; longest spec-name match
    disambiguates "x" vs "x_normed". Falls back to positional order when no
    view_name matches (only valid when ptr order == spec order).

    Raises ValueError if a ptr can't be mapped or dtypes disagree — that means
    the kernel isn't a clean leaf module (ptrs are intermediates, Phase 5).
    """
    spec_names = [s["name"] for s in specs]
    used: set[int] = set()
    result: list[int] = []
    for pi, ptr in enumerate(ptr_params):
        vw = ptr.get("view_name", "")
        si = None
        if vw:
            # view stem starts with a spec name -> the spec is the origin.
            # Longest spec name wins (so "x_normed" beats "x" for that view).
            # The reverse (sn.startswith(vw)) is intentionally NOT used: a view
            # "x" being a prefix of spec "x_normed" does NOT mean ptr "x" feeds
            # spec "x_normed" — it feeds spec "x".
            candidates = [i for i, sn in enumerate(spec_names)
                         if vw.startswith(sn) and i not in used]
            if candidates:
                candidates.sort(key=lambda i: -len(spec_names[i]))
                si = candidates[0]
        if si is None:
            # positional fallback: ptr i -> spec i (only valid when orders match)
            if pi < len(specs) and pi not in used:
                si = pi
        if si is None:
            raise ValueError(
                f"could not map ptr {ptr['arg']} (view={vw!r}) to any "
                f"spec; not a clean leaf module.")
        used.add(si)
        result.append(si)

    if len(used) != len(specs):
        raise ValueError(
            f"ptr->spec mapping left some specs unmapped "
            f"({len(used)}/{len(specs)}); not a clean leaf module.")

    # dtype consistency: ptr pto dtype must match the mapped spec's torch dtype.
    # A mismatch (ptr f32 but spec bf16) means main.cpp would allocate the
    # wrong buffer size and mis-read the golden bin — not a clean leaf.
    for i, ptr in enumerate(ptr_params):
        si = result[i]
        ptr_dt = ptr["pto_type"]
        spec_dt = _pto_of_torch(specs[si]["dtype"])
        if ptr_dt != spec_dt:
            raise ValueError(
                f"dtype mismatch at ptr {ptr['arg']} (view="
                f"{ptr.get('view_name','')!r}, pto={ptr_dt}) vs spec "
                f"{spec_names[si]!r} (torch={specs[si]['dtype']} -> pto={spec_dt}); "
                f"not a clean leaf module.")
    return result


def dump_bins(model_py: Path, mode: str, run_dir: Path,
              pto_path: Path | None = None) -> None:
    """Heavy golden generation: import the model under torch + golden package,
    run golden_fn, write vN.bin (inputs) + golden_vN.bin (outputs) into run_dir.

    run_dir is CWD when called as a subprocess (the skill sets cwd=run_dir).
    DSV4 MODES is __main__-only, so B/S come from config.py directly.

    pto_path (Mode B): the .pto file, needed to map ptr-arg order to spec
    order. vN is numbered by ptr order (v1=%arg0...) so main.cpp reads the
    right buffer for each kernel arg; dump_bins writes each spec's tensor to
    the vN of its mapped ptr. Without this, ptr order != spec order (hc_post)
    would mis-feed inputs to outputs and vice versa.
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

    specs = _call_build_tensor_specs(mod, B, S)

    golden_fn = _find_golden_fn(mod, model_py.stem)
    if golden_fn is None:
        raise ValueError(f"{model_py}: no golden fn found (tried "
                         f"golden_{model_py.stem}_test, golden_{model_py.stem}, "
                         f"and unique golden_* fallback)")

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

    # Map ptrs (pto signature order) to specs (build_tensor_specs order).
    # vN is numbered by ptr order (v1=%arg0...) so that main.cpp feeds each
    # kernel arg the right buffer. Without this, ptr order != spec order
    # (hc_post: [y,post,x,comb,residual] vs [x,residual,post,comb,y]) would
    # write the output tensor into v1 (kernel arg0=input y_flat) etc.
    if pto_path is not None:
        # parse_pto is in the same lib dir; import it absolutely (dump_bins
        # runs as a subprocess where sys.path has SKILL/lib prepended, so
        # there is no parent package for a relative import).
        import pto_parse
        pto_info = pto_parse.parse_pto(Path(pto_path))
        ptr_params = [p for p in pto_info["params"]
                     if p["pto_type"] not in ("i32", "index")]
        # normalize TensorSpec objects to dicts (same shape as resolve_meta's
        # subprocess JSON) so _map_ptrs_to_specs sees a uniform type.
        spec_dicts = [{"name": s.name, "is_output": getattr(s, "is_output", False),
                       "dtype": str(s.dtype), "shape": list(s.shape)}
                      for s in specs]
        ptr_to_spec = _map_ptrs_to_specs(ptr_params, spec_dicts)
    else:
        ptr_to_spec = list(range(len(specs)))  # positional (Mode A fallback)

    for ptr_i, spec_i in enumerate(ptr_to_spec):
        spec = specs[spec_i]
        vname = f"v{ptr_i+1}"
        arr = _torch_to_np(tensors[spec.name])
        arr.tofile(run_dir / f"{vname}.bin")
        if getattr(spec, "is_output", False):
            arr.tofile(run_dir / f"golden_{vname}.bin")


def _find_golden_fn(mod, model_stem: str):
    """Find the golden reference fn. DSV4 naming is inconsistent:
    rmsnorm.py -> golden_rms_norm_test; hc_post.py -> golden_hc_post;
    hc_head.py -> golden_hc_head. Try strict-then-loose candidates."""
    # Try the two DSV4 naming conventions first (with and without _test suffix).
    candidates = [
        f"golden_{model_stem}_test",
        f"golden_{model_stem}",
    ]
    for c in candidates:
        fn = getattr(mod, c, None)
        if fn is not None:
            return fn
    # Fallback: scan all non-prefill golden_* callables. When a module defines
    # both golden_<x> and golden_<x>_prefill, the non-prefill one is decode's.
    # When multiple non-prefill fns exist (common: golden_<x> helper + the
    # golden_<x>_test entry point), prefer the _test-suffixed one — it is the
    # DSV4 convention for the callable that fills tensors in-place. This also
    # bridges the filename-stem vs fn-name gap (rmsnorm.py exposes
    # golden_rms_norm_test, not golden_rmsnorm_test).
    gfs = [n for n in dir(mod)
           if n.startswith("golden_") and callable(getattr(mod, n))
           and not n.endswith("_prefill")]
    if len(gfs) == 1:
        return getattr(mod, gfs[0])
    if len(gfs) > 1:
        test_suffixed = [n for n in gfs if n.endswith("_test")]
        if len(test_suffixed) == 1:
            return getattr(mod, test_suffixed[0])
        # multiple _test fns — prefer the one whose name (minus golden_/_test)
        # normalized (drop underscores) matches the stem normalized
        def _norm(s):
            return s.replace("golden_", "").replace("_test", "").replace("_", "")
        stem_n = _norm("golden_" + model_stem + "_test")
        stem_matches = [n for n in test_suffixed if _norm(n) == stem_n]
        if len(stem_matches) == 1:
            return getattr(mod, stem_matches[0])
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
