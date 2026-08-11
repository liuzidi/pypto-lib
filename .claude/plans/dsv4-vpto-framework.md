# Task: Build a VPTO test framework for DeepSeek-V4 Pro on A5

You are working in the pypto-lib repo (cwd: /data/liuzidi/pypto-lib). Communicate
with the user in Chinese; ALL artifacts (code comments, CSV, commit messages,
docs) must be in English.

## Goal

Build a test framework that runs DeepSeek-V4 Pro (DSV4) VPTO-route kernels on a
real A5 NPU with ONE command, and reports whether each has a precision problem.

- Route 2 (VPTO) ONLY: pto→LLVM→bisheng fatobj→CANN module-load. This is the
  route the existing `vpto-board-validate` skill implements.
- The EmitC / simpler (Route 1) path is DEFERRED — do not build it now. The
  user may add it later.

Command shape: `python models/deepseek_v4_pro/validate.py -p a5 -d 0
[--mode decode]` → sweeps all runnable VPTO kernels (currently the 2 strict
leaves), runs each on device 0 (serial, single card), and writes an aggregated
CSV + summary.

## Scope discipline (CRITICAL — read twice)

- DO build: the VPTO sweep framework (this is the entire deliverable).
- DO NOT fix precision. Precision FAIL is the framework's EXPECTED, VALUABLE
  output — it means the framework is working. Record max_diff/n_over and move
  on. Never "fix" a per-kernel numerical mismatch.
- DO NOT do the EmitC/simpler (Route 1) path. Deferred.
- DO NOT do non-leaf intermediate capture (Phase 5) — out of scope. Non-leaf
  kernels are skipped, not attempted.
- DO NOT do performance work (timing column can exist as a placeholder; no
  perf analysis).

## What already exists (committed — read it, reuse it, don't reinvent)

1. `.claude/skills/vpto-board-validate/` — the Route 2 executor. Already
   supports DSV4 `run_jit`-style golden via `--model-py` (Mode B):
   ```
   python .claude/skills/vpto-board-validate/vpto_run.py \
       --pto <X.pto> --model-py models/deepseek_v4_pro/<Y>.py \
       --mode decode --device 0
   ```
   Key files: `vpto_run.py`, `lib/run_jit_golden.py` (resolve_meta +
   dump_bins, already has `_call_build_tensor_specs` signature dispatch +
   `_find_golden_fn`), `lib/setup_main.py`, `lib/setup_vpto.py`,
   `lib/pto_parse.py`, `runtime/compare.py`. Read SKILL.md first.

2. `baselines/vpto_dsv4_vector/classification.csv` — 277 rows classifying all
   DSV4 pure-vector .pto kernels. **Only 2 are strict leaves**: `rms_norm`
   (module rmsnorm.py) and `hc_post` (module hc_post.py). A strict leaf =
   ptr-arg count == spec count AND ptr→spec view-name map succeeds AND dtype
   matches. `hc_head_pre_fused` is the documented false positive (3 of 5 ptrs
   are intermediates: pre_t/mixes_raw/inv_rms from `pl.create_tensor`).
   The framework should READ this CSV and sweep rows where the strict-leaf
   column is True. When future Phase 5 work adds more runnable kernels, the
   framework picks them up automatically — do NOT hardcode the 2-leaf list.

3. `baselines/vpto_dsv4_vector/leaf_results.csv` — existing Route 2 results
   for the 2 leaves: rms_norm max_diff=30464, hc_post max_diff=0.306, both
   FAIL. These are CORRECT findings (VPTO lowering issues), NOT framework
   bugs. The pipeline is verified: golden is sane (±4, no NaN), fatobj links,
   NPU executes, real numerical compare. The framework must reproduce these.

4. `scripts/vpto_env.sh` — MUST be sourced before any board run. It fixes
   CANN-9.2 pollution (sources global set_env.sh FIRST, then overrides all
   CANN vars to beta.3) and sets bisheng. Without it, rmsnorm takes 27s
   instead of 10ms. The framework driver should auto-source it (e.g. spawn
   vpto_run.py via `bash -c "source scripts/vpto_env.sh && exec python3 ..."`
   ) so the user gets a true one-command experience — confirm approach with
   user if unsure.

5. Route 2 proven ptoas flags (all ON):
   `--pto-arch=a5 --pto-level=level3 --pto-backend=vpto --enable-tile-op-expand
   --enable-insert-sync --enable-op-fusion`. Plus two mandatory sed steps on
   the .pto IR: module attrs add `pto.kernel_kind`; func attrs add `pto.kernel`.
   The skill already does all of this — you should not need to re-implement.

## Phased plan

### Phase 0 — Read-only: understand the current invocation (quick)
Read SKILL.md, `vpto_run.py --help`, `classification.csv` (the 2 leaf rows),
and `leaf_results.csv`. Confirm:
- The exact `vpto_run.py` CLI for a leaf run, and how its stdout/exit code
  signals pass/fail/max_diff/timing (so the framework can parse it).
- The classification.csv column names (which column marks strict-leaf).
- Whether vpto_run.py already writes a per-run result file the framework can
  read, or whether the framework must parse stdout.

Report findings to the user before building Phase 1.

### Phase 1 — Build the sweep harness (validate.py)
- A driver that reads `classification.csv`, filters strict-leaf rows, and for
  each leaf invokes `vpto_run.py --pto <pto_path from csv> --model-py
  <model_py from csv> --mode <mode> --device <device>` as a subprocess with
  vpto_env.sh sourced.
- Serial on device 0. Single card.
- Per leaf, collect: `{kernel, module, route=vpto, pass, max_diff, n_over,
  timing_ms, fatobj_bytes, rtol, atol, error}`. crash/ptoas-fail/bisheng-fail
  = finding → record + continue, NEVER abort the sweep.
- Parse the per-run result from vpto_run.py's output (or a result file it
  writes — Phase 0 determines which). If parsing is fragile, prefer asking
  the skill to emit a small JSON sidecar; do NOT regex-scrape human text if
  avoidable.

### Phase 2 — Aggregated report
- Write `baselines/vpto_dsv4_vector/sweep_results.csv` (or append to
  leaf_results.csv — confirm with user) with the schema above.
- Stdout summary: "VPTO route: X/Y leaves pass; per-kernel: rms_norm FAIL
  (max_diff=30464), hc_post FAIL (max_diff=0.306). Non-leaves skipped
  (deferred Phase 5)."
- Single command: `python models/deepseek_v4_pro/validate.py -p a5 -d 0
  [--mode decode]`. Confirm the driver location with user (this path mirrors
  Qwen's per-model layout; alternative is `tests/dsv4_validate/`).

## Background: DSV4 golden convention (for debugging the skill, not re-implementing)

The skill already handles all of this internally. This section is context in
case you need to debug:
- NOT `*_golden_lib.py`. DSV4 uses module-level: `build_tensor_specs(...)`
  → [TensorSpec...], `golden_<name>(tensors)` fills outputs in-place.
- `build_tensor_specs` has 11+ signatures; the skill's
  `_call_build_tensor_specs` dispatches by `inspect.signature`.
- `golden_fn` naming: `golden_<name>` (NO `_test` suffix) for 34/35 models;
  only rmsnorm.py has `golden_rms_norm_test`. `_find_golden_fn` handles both.
- B/S from config.py (DECODE_BATCH=4, DECODE_SEQ=2), NOT from MODES.
- bf16 bin-dump: `t.view(torch.uint16)` (bitcast), never `t.to(uint16)`.
- Repo `.venv/bin/python3` has torch + golden; system python3 lacks both.
  Skill resolves `.venv` relative to repo root — no hardcoded paths.

## Hard rules (non-negotiable)

1. Route 2 stays on VPTO, NEVER EmitC. Do not build the simpler/Route 1 path.
2. No private info (usernames, absolute paths with usernames) in code/docs.
3. English for all artifacts (comments, CSV, commits, docs). Chinese only in
   conversation with the user.
4. `source scripts/vpto_env.sh` before EVERY board run (framework auto-sources
   it so the user doesn't have to).
5. Route 2 Mode B uses repo `.venv/bin/python3` (skill auto-resolves).
6. Do NOT fix per-kernel precision — record and continue. FAIL is expected.
7. Do NOT modify simpler/pypto/ptoas/pto-isa or any model `.py` file, nor the
   vpto-board-validate skill files unless a genuine skill bug blocks the
   sweep. Read-only on all of those. New code goes in the framework driver.
8. Do NOT run non-leaf kernels through Mode B — they fail by design. Skip them.
9. Single card, serial: `--device 0`, no multi-card fan-out.
10. Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
11. Use the git-commit skill workflow (`.claude/skills/git-commit/SKILL.md`):
    pre-commit lint (`python tests/lint/check_headers.py`,
    `python tests/lint/check_english_only.py`, `ruff check .`), no
    `build_output/` artifacts in commits.

## Repo hygiene note

These untracked items exist at the repo root and should NOT be committed:
`BASELINE_COMPARISON.md`, `PROBLEM_REPRODUCTION.md` (these belong to a
different, earlier lab/ device-2 workstream — leave them alone), `.zcode/`
(tool cache), `.claude/settings.json` (contains private absolute paths —
gitignore it). `.claude/plans/*.md` (handoff docs) CAN be committed.

## Expected pitfalls (already hit by prior AIs — don't repeat)

- Forgetting to `source scripts/vpto_env.sh` → CANN-9.2 pollution, 27× slowdown.
- Output TensorSpec `init_value=None` → must zero-fill (`torch.zeros`).
- Trailing `index` scalar = `ctx_len = B*S`; spmd_block_num must be 1, not 0.
- `golden.py` stub name collides with the repo `golden` package → name it
  `gen_golden.py`.
- sys.path: insert repo root BEFORE `from golden import TensorSpec`.
- Don't trust `MODES` — read config.py directly.
- Don't regex-scrape human-readable stdout for results if a structured sidecar
  is available or can be added cheaply.

## First action

Start with Phase 0 (read-only). Report findings to the user before building
Phase 1. Do not write framework code until Phase 0 is done and the user has
confirmed the Phase 1 approach (especially: how to parse per-run results, and
where to place the driver).
