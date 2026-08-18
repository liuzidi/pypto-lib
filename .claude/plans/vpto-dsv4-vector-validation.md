# VPTO board validation of DSV4 pure-vector kernels

## Goal
Run DSV4's pure-vector inner kernels through the VPTO route on real A5,
validated against torch golden. Compare against the Route-1 baseline
(`baselines/nofusion_a5_pmu2/`) later.

## Current state (verified)
- **97 pure-vector `.pto` inner kernels** exist under
  `build_output/_jit_*/ptoas/` (gitignored but present). Spread across the
  27 top-level modules. Deduped by basename.
- DSV4 does **not** use `*_golden_lib.py` (the skill's current contract).
  It uses `golden.run_jit(fn=, specs=[TensorSpec...], golden_fn=)` — the
  golden fn fills module-level outputs in-place; spec list carries shapes +
  init values. Each model `.py` is self-contained (e.g. `rmsnorm.py` has
  `rms_norm_test`, `golden_rms_norm_test`, `build_tensor_specs`).
- **Leaf modules** (1 module → 1 vector kernel, kernel ptrs == module
  TensorSpecs): `rms_norm` (3 ptrs == 3 specs, verified), `hc_post` (1/1).
  These are the directly-validatable set.
- **Multi-kernel modules** (e.g. `attention_csa`: 33 vector kernels but
  ~20 module-level TensorSpecs): the inner kernels' ptr args are
  *intermediates* from preceding kernels, NOT module inputs. The module
  golden_fn does not produce these intermediates. These need a separate
  intermediate-capture workstream (deferred).
- All DSV4 `.pto` have `__pypto_spmd_block_idx` / `__pypto_spmd_block_num`
  params; the skill already fills `block_num=1` (fixed last session).
- `vpto_env.sh` + skill run end-to-end on the Qwen3 reference set (rmsnorm
  + incore_1) — confirmed working last session.

## Decision (from user)
- Golden source: **extend the skill to accept `run_jit`-style golden**
  (model `.py` + `fn` + `specs` + `golden_fn`), not per-kernel `*_golden_lib.py`.
- Board runs: **serial, single card** (`--device 0`). No multi-card fan-out.

## Plan

### Phase 1 — Extend skill to accept `run_jit`-style golden (by me, not parallel)
Add a second input mode to `vpto_run.py` alongside the existing
`--golden-lib`:

```
python3 vpto_run.py \
  --pto <kernel>.pto \
  --model-py models/deepseek_v4_pro/<mod>.py \
  --mode decode|prefill \
  --device 0
```

The model `.py` must expose (DSV4 convention, all do):
- `<jit_fn>` (e.g. `rms_norm_test`) — the `@pl.jit` fn; its param order
  == the `.pto` kernel ptr-arg order (drop the trailing `index` +
  `__pypto_spmd_*` scalars).
- `build_tensor_specs(B, S)` → `[TensorSpec...]` in the same order.
- `golden_fn(tensors)` filling outputs in-place.
- `MODES = {"decode": (B,S), "prefill": (B,S)}`.

Flow inside the skill (run_jit mode):
1. Import model `.py`, pick `B,S` from `MODES[--mode]`.
2. Build specs, init torch tensors (CPU), run `golden_fn` → expected
   outputs (keep on CPU).
3. Dump each input tensor to `<run>/in/<name>.bin` (raw, row-major, the
   `.pto` kernel's GM layout). The existing `main.cpp` reads these.
4. The existing pipeline (sed → ptoas → bisheng → main.cpp → NPU) runs
   unchanged — it already reads `*.bin`, launches, writes `*.bin`.
5. Compare NPU output `.bin` vs the torch golden (CPU) using the existing
   `compare.py`, extended to accept the `TensorSpec` dtype + the model's
   `compare_fn` / `rtol`/`atol`.

Key implementation points:
- The model `.py` imports `pypto.language` + `config` + `golden` — must
  run under the repo's `.venv` (`/data/liuzidi/pypto-lib/.venv`, which has
  pypto) with `PYTHONPATH` including `models/deepseek_v4_pro` (for
  `config.py`). The skill's golden-gen subprocess must set this up.
- `TensorSpec` dtypes map to the `.pto` ptr dtypes directly (bf16→bf16,
  f32→f32, int8→int8). Element counts from `spec.shape` prod.
- For leaf modules, the `.pto` kernel's ptr args are **exactly** the
  module's specs in order (verified for `rms_norm`). The skill resolves
  which spec is output via `is_output=True` (or the model's `compare_fn`
  keys). No BUILDERS map needed.
- Guard: if the `.pto` ptr-arg count != spec count → not a leaf module;
  emit a clear "multi-kernel module, needs intermediate capture (not yet
  supported)" error and exit gracefully.

### Phase 2 — PoC: prove rms_norm (leaf) runs DSV4→VPTO (by me)
Run the extended skill on `rms_norm.pto` + `rmsnorm.py --mode decode`.
Must reproduce: fatobj with `T rms_norm`, NPU executes, golden compare
against `golden_rms_norm_test` with `rtol=5e-3 atol=5e-3` (the model's
own tolerances). This proves the DSV4 path end-to-end. Do not proceed to
Phase 3-4 until this passes (or the failure is understood + documented).

### Phase 3 — Classify the 97 vector kernels by golden-availability (parallelizable)
Build a table: for each pure-vector `.pto`, record:
- module, kernel name, ptr-arg count, whether ptr-args == module specs
  (leaf) or not (needs-intermediate), spmd block count, .pto path.
Output: `baselines/vpto_dsv4_vector/classification.csv`.

This is pure static analysis (read `.pto` signatures + model `.py`
TensorSpecs) — **fully parallelizable per module group**. Each subagent
takes ~7 modules, parses, appends to the CSV. No NPU, no board.

### Phase 4 — Run all leaf vector kernels through the skill (serial, single card)
For each row in `classification.csv` where `is_leaf=True`, run the skill.
Collect: fatobj size, has_kernel_sym, timing, pass/fail, max_diff.
Output: `baselines/vpto_dsv4_vector/leaf_results.csv`.
Serial because one card (`--device 0`); ~N leaf kernels × ~1-10s each.

### Phase 5 — Multi-kernel modules: intermediate capture (future, out of scope now)
For non-leaf vector kernels (inputs are intermediates), design a capture:
run the module via `run_jit` (Route 1/simpler) with GM-buffer dumping,
replay captured intermediates through the VPTO kernel. Separate
workstream — do not block Phases 1-4.

## Parallelization answer (for the user's question)

**Yes, this can be parallelized, but only after Phase 1-2.** The dependency:

```
Phase 1 (skill ext) ──► Phase 2 (PoC rms_norm) ──┬─► Phase 3 (classify, PARALLEL)
                                                  └─► Phase 4 (run leaves, SERIAL)
```

- Phase 1-2 must be done first by me (the skill extension + PoC). Cannot
  parallelize — the skill is a single shared artifact; parallel agents
  would edit the same files.
- Phase 3 (classification) is the parallelizable CPU work: ~7 module
  groups × 1 agent each, static analysis only, no NPU. Ideal for a
  subagent fan-out.
- Phase 4 (board runs) is serial (single card). Not parallelizable unless
  we later opt into multi-card (user chose serial).
- Phase 5 (intermediate capture) is future parallel work per module.

So the parallel subagent question: **yes for Phase 3**, after I finish
Phase 1-2. The handoff prompt for a fresh AI (or subagents) to continue
from Phase 3 is included below.

## Handoff prompt (for a fresh AI / subagents to run Phase 3-4)

> Task: classify and run DSV4 pure-vector kernels through the
> `vpto-board-validate` skill (Phases 3-4 of
> `.claude/plans/vpto-dsv4-vector-validation.md`).
>
> Prerequisites (already done):
> - The skill at `.claude/skills/vpto-board-validate/` accepts a
>   `--model-py` (run_jit-style golden) mode. Verified on `rms_norm`.
> - `source scripts/vpto_env.sh` sets up ptoas + bisheng + CANN beta.3.
>
> Phase 3 (parallelizable per module group):
> For each pure-vector `.pto` under `build_output/_jit_*/ptoas/`, parse
> its `func.func` signature (ptr-arg count + dtypes, drop trailing
> `index` + `__pypto_spmd_*`). Compare to the owning module `.py`'s
> `build_tensor_specs(B,S)` TensorSpec list (same order). Mark `is_leaf`
> if ptr-arg count == spec count. Write
> `baselines/vpto_dsv4_vector/classification.csv`.
>
> Phase 4 (serial, `--device 0`):
> For each `is_leaf=True` row, run:
> `source scripts/vpto_env.sh && python3 .claude/skills/vpto-board-validate/vpto_run.py --pto <path> --model-py models/deepseek_v4_pro/<mod>.py --mode decode --device 0`
> Collect fatobj size, has_kernel_sym, timing, pass/fail, max_diff into
> `baselines/vpto_dsv4_vector/leaf_results.csv`.
>
> Rules: stay on VPTO (never EmitC). No private paths in outputs.
> English for all intermediate work. If a kernel crashes, record it and
> continue (do not block the sweep).

## Files to touch (Phase 1-2 only)
- `.claude/skills/vpto-board-validate/vpto_run.py` — add `--model-py` /
  `--mode` args; new `_golden_from_run_jit()` path that imports the model,
  builds specs, runs golden_fn, dumps `*.bin`.
- `.claude/skills/vpto-board-validate/runtime/compare.py` — accept a
  `--dtype` / `--rtol` / `--atol` / `--compare-spec` so it can compare
  against torch golden with the model's tolerances (current compare is
  exact-match-only for f32, too strict for DSV4 bf16).
- `.claude/skills/vpto-board-validate/SKILL.md` — document the new
  `--model-py` mode + the leaf-module requirement + the DSV4 golden
  convention.

No changes to: simpler, pypto, ptoas, the model `.py` files, or the
baseline (Phase 4 only *reads* `baselines/nofusion_a5_pmu2/`).
