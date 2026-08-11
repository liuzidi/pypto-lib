---
name: vpto-board-validate
description: Run a pypto-emitted .pto end-to-end through the VPTO backend (pto -> ptoas LLVM -> bisheng fatobj .o) on a real Ascend A5 NPU and compare against a torch/numpy golden reference. Use when validating Route 2 (VPTO) board runs, comparing Route 1 (EmitC) vs Route 2, or reproducing VPTO codegen behavior. NOT for EmitC (Route 1) runs — use the standard golden harness for those.
---

# VPTO Board Validation (Route 2)

This skill runs the **VPTO** route — `.pto` → `ptoas --pto-backend=vpto` →
bisheng fat-object `.o` → real A5 NPU → golden compare — for a single kernel.
It is the canonical executor for Route 2; the standard `golden/` harness covers
Route 1 (EmitC).

## Proven-working configuration (do not deviate without reason)

The ptoas invocation and preprocessing below are the **only** combination
verified to produce a real kernel (not an empty ctor-only fatobj) from
pypto EmitC-era `.pto` files:

- `ptoas --pto-arch=a5 --pto-level=level3 --pto-backend=vpto
  --enable-tile-op-expand --enable-insert-sync --enable-op-fusion`
  (all three fusion/expansion flags **ON** — turning `--enable-op-fusion`
  off silently skips kernel codegen and emits a fatobj with only
  `cceModuleCtor` + `__cce_ptc_wrapper`, no kernel symbol).
- Two `sed` edits on the `.pto` before feeding ptoas:
  1. module attrs: add `pto.kernel_kind = #pto.kernel_kind<cube|vector>`
     (auto-detected by grepping the source `.pto`).
  2. func attrs: `attributes {pto.kernel_kind` →
     `attributes {pto.kernel, pto.kernel_kind` — the `pto.kernel` attr is
     what triggers kernel body codegen.
- bisheng `--cce-fatobj-link -shared` links the fatobj + a generated
  `launch.cpp` (`func<<<1, nullptr, stream>>>`) into a `.so`, loaded via
  CANN module-load (`rtRegisterGlobals` /
  `__cce_rtKernelLaunchWithFlagV2`). **No simpler-runtime change is needed** —
  the skill bypasses simpler's InCore path entirely with its own `main.cpp`.

A previous (wrong) conclusion claimed "VPTO cannot lower pypto's EmitC tile
dialect". That was a misdiagnosis of the empty-fatobj failure mode caused by
omitting the two steps above. Verified on `rmsnorm` (vector) and
`qwen3_decode_incore_1` (cube, q_proj): both compile to real fatobjs with
`T <kernel>` + a nested `__aicore_rel_binary` ELF, and both execute on npu:0.

## Bundled scripts

- `vpto_run.py` — the one-command executor. Parses the `.pto` + golden lib,
  sed-preprocesses, runs ptoas, bisheng-compiles+links, generates golden,
  runs on the device, compares. Reports fatobj size + symbol presence as an
  early fail signal.
- `lib/pto_parse.py`, `lib/setup_vpto.py`, `lib/setup_main.py` — vendored +
  path-neutralized generators for the `.pto` signature, `launch.cpp`, and
  `main.cpp` (onboard only; camodel/sim modes are not included).
- `runtime/compare.py`, `runtime/validation_runtime.py` — vendored golden
  compare + runtime (copied per-run).

## Prerequisites

1. A free Ascend A5 device allocated to the user (check `npu-smi info`).
2. `scripts/vpto_env.sh` sourced — it builds `/tmp/mlir_core_vmi` (the
   ptodsl daemon needs an mlir_core with `ir.py`; build311's bundled mlir
   lacks it). `vpto_run.py` auto-sources it if not already done, but
   sourcing it once in your shell is faster.
3. Env vars (set by `vpto_env.sh` + this host's CANN install):
   `PTOAS_BIN`, `ASCEND_HOME_PATH`, `BISHENG_BIN`, `PTO_ISA_PATH`,
   `TILELANG_PATH`, `TILELANG_PKG`. `vpto_run.py` checks these and reports
   if missing.

## Inputs

- `--pto`: a pypto-emitted `.pto` (EmitC-era tile dialect:
  `tile_buf`/`tload`/`tstore`/`make_tensor_view`/`partition_view`/...).
- `--golden-lib`: a `*_golden_lib.py` exposing `BUILDERS = {"<kernel>": build_fn}`
  and `run_case(name)`. The build fn returns `buffers, {"vN": golden_output}`.
  This is the same contract as the `test_for_ptoas` harness.

## Workflow

```bash
source scripts/vpto_env.sh
python .claude/skills/vpto-board-validate/vpto_run.py \
    --pto <path.pto> --golden-lib <golden_lib.py> --device <id>
```

The executor runs: parse → sed-preprocess → ptoas VPTO → bisheng compile+link →
golden → NPU run → compare. Output lands in `build_output/vpto_<kernel>/`.

## Known env quirks (encoded in vpto_run.py, documented here)

- **PYTHONPATH overlay for the ptodsl daemon**: ptoas spawns a ptodsl daemon
  that imports `mlir.ir`; build311's bundled mlir has no `ir.py` → daemon
  fails → ptoas emits "InsertTemplateAttributes requires a PTODSL daemon
  socket" → empty fatobj. Fix: ptoas is run with
  `PYTHONPATH=/tmp/mlir_core_vmi:$PTOAS_SOURCE/ptodsl:$TILELANG_PKG`.
- **`KERNEL` over-replace (historical)**: the original test_for_ptoas
  `setup_vpto.py` did `.replace("KERNEL", kernel)` which turned
  `KERNEL_NAME="KERNEL"` into `<kernel>_NAME="<kernel>"`. Harmless in bash
  but the skill's `setup_vpto.generate_run_sh` parameterizes directly to
  avoid it.
- **`main.cpp` location**: `main.cpp` is generated next to `launch.cpp` in
  the run dir; the host binary reads `./vN.bin` from its cwd (run dir).

## Interpreting results

- **Fatobj must have `T <kernel>` + a nested ELF** (offsets reported by
  `vpto_run.py`). If `has_kernel_sym=False` → ptoas skipped codegen; check
  the sed steps + that `--enable-op-fusion` is ON.
- **NPU timing** ~0.4–1.6 ms for the Qwen3 reference kernels. A crash here
  is a real codegen bug (e.g. incore_2's L0C-out-of-range).
- **Precision**: for SPMD kernels (with `__pypto_spmd_block_idx` /
  `__pypto_spmd_block_num` params), the host `main.cpp` fills
  `__pypto_spmd_block_num = 1` (single-block launch; `0` would mean "no
  blocks" → kernel body never runs → all-zero output) and
  `__pypto_spmd_block_idx = 0`. Other non-`ctx_len`/`ctx_blocks` scalars
  default to `0` and may still need per-op tuning via the golden-lib's
  `consts`/`load_vals`. A precision mismatch here does **not** indicate a
  route failure — verify the route is sound first (fatobj symbol + NPU
  execution + output non-zero + range plausible) before chasing precision.

## Reference corpus

The Qwen3 `.pto` + `qwen3_decode_golden_lib.py` in
`test_for_ptoas_extracted/test_for_ptoas/` are a known-good reference set
(unzipped from `test_for_ptoas.zip`). Use them to verify the skill still
works after any change:
- `rmsnorm.pto` (vector) → fatobj 7336 B, `T rmsnorm`, NPU executes.
- `qwen3_decode_incore_1.pto` (cube, q_proj) → fatobj 5392 B,
  `T qwen3_decode_incore_1`, NPU executes, compare runs (precision depends
  on the spmd scalar fill).
- `qwen3_decode_incore_2.pto` (cube) → fatobj produces but NPU run crashes
  with `errcode 168 / L0C out of range` — a real VPTO codegen bug, not a
  skill failure.

## Safety

- **No hardcoded host paths in skill scripts.** All paths come from env
  vars (satisfies the repo's no-private-paths rule). The `_DEFAULTS` table
  in `vpto_run.py` is host-specific but every entry is env-overridable.
- Do not use a device not allocated to the user; check `npu-smi info` first.
- `build_output/` is gitignored — runs leave artifacts there; do not commit.
- CAModel (sim) modes from the original harness are **not** included (this
  host has no `dav_3510` sim lib). For sim runs, use the PTOAS lab's
  `dsv4-vmi-lowering-lab` instead.
- Do not modify the `.pto` source; the skill writes the preprocessed copy
  into the build dir.

## Reporting

Report: kernel kind (cube/vector), fatobj size + kernel-symbol presence +
nested-ELF offsets, bisheng compile/link success, NPU run exit + timing,
precision result (PASS/FAIL + max diff + which buffers). Call out the spmd
scalar fill caveat explicitly when precision fails on an SPMD kernel.
