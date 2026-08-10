#!/usr/bin/env python
"""Run any deepseek_v4_pro single-card operator with PMU collection on.

Why this exists: the operators' own CLIs do not expose ``--enable-pmu``.
Rather than edit every operator's ``__main__``, this wrapper patches
``golden.runner.run_jit`` so that an ``enable_pmu`` flag is injected into
the operator's ``runtime_cfg`` right before execution. The operator's own
``fn`` / ``build_tensor_specs`` / ``golden_fn`` / ``compare_fn`` run
unchanged — only the PMU toggle is added.

Usage:
    source .env_a5.sh
    python scripts/run_op_pmu.py <op> -p a5 -d 6                  # PIPE_UTILIZATION(2)
    python scripts/run_op_pmu.py gate -p a5 -d 6 --pmu 4          # MEMORY
    python scripts/run_op_pmu.py rmsnorm -p a5 -d 6 --pmu 8       # L2_CACHE
    python scripts/run_op_pmu.py gate -p a5 -d 6 --pmu 0          # PMU off (sanity)
    python scripts/run_op_pmu.py gate -p a5 -d 6 --no-fusion      # disable op-fusion
    python scripts/run_op_pmu.py gate -p a5 -d 6 --pmu 2 --no-fusion  # both at once
    python scripts/run_op_pmu.py gate -p a5 -d 6 --pmu 2 --keep-csv  # also write <op>_pmu.csv

<op> is the operator module name under models/deepseek_v4_pro/, with or
without the ``.py`` suffix. Extra args after ``--`` are forwarded to the
operator's own argparse (e.g. ``-- --layer-id 5 --num-tokens 4``).

Output: prints the run result and the path to ``pmu.csv``:
    <work_dir>/dfx_outputs/pmu.csv   (only when --pmu > 0)

By default the pmu.csv is NOT copied next to the operator (to avoid
scattering transient files across a 27-op sweep); add --keep-csv to also
write <op>_pmu.csv there. The canonical archived baselines live under
baselines/ (see scripts/collect_pmu_baseline.py).
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
PYPTO_LIB_ROOT = HERE.parent.parent
MODEL_DIR = PYPTO_LIB_ROOT / "models" / "deepseek_v4_pro"

# --- patch run_jit to inject enable_pmu before the operator runs ----------
import golden.runner as _gr  # noqa: E402

_ORIG_RUN_JIT = _gr.run_jit
_INJECTED_PMU = {"value": 0}  # mutable holders so closures can read CLI flags
_NO_FUSION = {"value": False}


def _run_jit_with_pmu(*args, **kwargs):
    runtime_cfg = kwargs.get("runtime_cfg")
    if runtime_cfg is None:
        runtime_cfg = {}
        kwargs["runtime_cfg"] = runtime_cfg
    pmu = _INJECTED_PMU["value"]
    if pmu and "enable_pmu" not in runtime_cfg:
        runtime_cfg["enable_pmu"] = pmu
        print(f"[op_pmu] injected enable_pmu={pmu} into runtime_cfg",
              flush=True)
    return _ORIG_RUN_JIT(*args, **kwargs)


_gr.run_jit = _run_jit_with_pmu
# keep the package-level re-export in sync (operators do `from golden import run_jit`)
import golden as _g  # noqa: E402
_g.run_jit = _run_jit_with_pmu


# --- optionally patch ptoas flags to disable A5 tile op-fusion ------------
# When --no-fusion is set, append `--enable-op-fusion=false` to every ptoas
# invocation. pypto.backend.pto_backend._get_ptoas_flags builds the default
# flag list; patching it here (before the operator compiles) disables A5
# level2/level3 tile fusion for the whole run. Default ptoas enables fusion
# on A5; this overrides it to diagnose fusion-induced numeric divergence.
def _install_nofusion_patch() -> None:
    import pypto.backend.pto_backend as _pb

    _orig = _pb._get_ptoas_flags

    def _get_ptoas_flags_nofusion(memory_planner=None):
        flags = _orig(memory_planner) if memory_planner is not None else _orig()
        flag = "--enable-op-fusion=false"
        if flag not in flags:
            flags.append(flag)
        return flags

    _pb._get_ptoas_flags = _get_ptoas_flags_nofusion


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a deepseek_v4_pro single-card operator with PMU and/or "
                    "op-fusion disabled.",
        usage="python scripts/run_op_pmu.py <op> [-p a5] [-d N] [--pmu N] "
              "[--no-fusion] [-- <op-args>]",
    )
    ap.add_argument("op", help="operator module name, e.g. gate / rmsnorm")
    ap.add_argument("-p", "--platform", default="a5",
                    choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--pmu", type=int, default=2,
                    help="PMU event type: 0=off, 1 ARITH, 2 PIPE_UTIL(default), "
                         "4 MEMORY, 5 MEM_L0, 6 RESRC_CONFL, 7 MEM_UB, 8 L2_CACHE")
    ap.add_argument("--no-fusion", action="store_true", default=False,
                    help="disable ptoas A5 tile op-fusion "
                         "(appends --enable-op-fusion=false to ptoas)")
    ap.add_argument("--keep-csv", action="store_true", default=False,
                    help="also copy pmu.csv next to the operator as "
                         "<op>_pmu.csv (default OFF to avoid polluting the "
                         "operator directory; build_output keeps the original)")
    known, extra = ap.parse_known_args()

    if known.no_fusion:
        _install_nofusion_patch()
        print("[op_pmu] op-fusion DISABLED (--enable-op-fusion=false)",
              flush=True)

    op_name = known.op.removesuffix(".py")
    mod_path = MODEL_DIR / f"{op_name}.py"
    if not mod_path.is_file():
        ap.error(f"operator not found: {mod_path}")

    _INJECTED_PMU["value"] = known.pmu
    print(f"[op_pmu] op={op_name} platform={known.platform} "
          f"device={known.device} pmu={known.pmu}", flush=True)

    # Put the operator's directory on sys.path FIRST so its sibling-module
    # imports resolve (operators do `from config import ...`, `import moe`,
    # etc., all relative to models/deepseek_v4_pro/).
    model_dir_str = str(MODEL_DIR)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    # Build argv for the operator's own __main__: -p / -d come from us,
    # plus anything after `--` the user passed.
    op_argv = [str(mod_path), "-p", known.platform, "-d", str(known.device)]
    op_argv += extra
    sys.argv = op_argv

    # Run the operator's __main__ in-process. runpy avoids polluting our
    # module namespace and matches how `python <op>.py` behaves.
    import runpy
    try:
        runpy.run_path(str(mod_path), run_name="__main__")
    except SystemExit as e:
        # operators raise SystemExit(1) on FAIL; PMU csv still produced.
        code = int(e.code) if isinstance(e.code, int) else (0 if not e.code else 1)
        return _report_pmu(mod_path, known.pmu, failed=code != 0,
                            keep_csv=known.keep_csv)
    return _report_pmu(mod_path, known.pmu, failed=False,
                        keep_csv=known.keep_csv)


def _report_pmu(mod_path: pathlib.Path, pmu: int, *, failed: bool,
                 keep_csv: bool = False) -> int:
    """Locate the most recent pmu.csv under build_output and print it.

    The original pmu.csv always lives under build_output/ (gitignored). Only
    when --keep-csv is set do we also copy it next to the operator as
    <op>_pmu.csv; otherwise we leave the operator directory untouched (a
    27-op sweep would otherwise scatter 27 transient copies there).
    """
    bo = PYPTO_LIB_ROOT / "build_output"
    pmus = sorted(bo.glob("**/dfx_outputs/pmu.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True) if bo.is_dir() else []
    if pmu > 0 and pmus:
        print(f"[op_pmu] pmu.csv: {pmus[0]}", flush=True)
        if keep_csv:
            dst = mod_path.with_name(f"{mod_path.stem}_pmu.csv")
            import shutil
            shutil.copy2(pmus[0], dst)
            print(f"[op_pmu] copied to: {dst}", flush=True)
    elif pmu > 0:
        print("[op_pmu] WARNING: no pmu.csv found under build_output/", flush=True)
    print(f"[op_pmu] {'operator FAILED (see output above)' if failed else 'operator passed'}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
