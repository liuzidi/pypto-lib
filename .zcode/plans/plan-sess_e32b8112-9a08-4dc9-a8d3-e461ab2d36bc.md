# Plan: Commit test_for_dsv4 as a shareable DSV4 precision test suite

## Goal
Make `test_for_dsv4/` a one-command, git-tracked test suite that team members can clone and run with minimal setup — just "set your env vars, then `python3 run_all.py --device N`".

## What to commit (3 groups)

### Group 1: test_for_dsv4/ source files (currently entirely untracked)

**Kernel inputs (the test data):**
- `.pto/` — all 110 .pto files (2.6MB total)

**Golden reference (pure numpy, no torch, no machine paths):**
- `dsv4_golden_lib.py` — BUILDERS loader + run_case()
- `golden_parts/attention_kernels.py` (19 builders)
- `golden_parts/hc_gate_moe_kernels.py` (28 builders)
- `golden_parts/prefill_misc_kernels.py` (43 builders)
- `golden_parts/qkv_mtp_compress_kernels.py` (25 builders)
- `golden/validation_runtime.py` — CaseMeta, rng, write_buffers
- `golden/compare.py` — bf16 ULP + fp32 compare

**Run scripts:**
- `run_all.py` — main sweep driver
- `setup_all.sh` — orchestrator
- `setup_test_suite.py` — run dir generator
- `emitc_run.py` — EmitC route runner (has env-overridable fallback defaults)
- `parallel_emitc_sweep.py` — multi-device parallel sweep
- `diagnose_crashes.py` — crash classifier

**Symlinks (all relative, git-friendly):**
- `lib/__init__.py` (empty)
- `lib/pto_parse.py → ../../.claude/skills/vpto-board-validate/lib/pto_parse.py`
- `lib/setup_main.py → ../../.claude/skills/vpto-board-validate/lib/setup_main.py`
- `lib/setup_vpto.py → ../../.claude/skills/vpto-board-validate/lib/setup_vpto.py`
- `compare.py → golden/compare.py`
- `validation_runtime.py → golden/validation_runtime.py`
- `attention_kernels.py → golden_parts/attention_kernels.py`
- `hc_gate_moe_kernels.py → golden_parts/hc_gate_moe_kernels.py`
- `prefill_misc_kernels.py → golden_parts/prefill_misc_kernels.py`
- `qkv_mtp_compress_kernels.py → golden_parts/qkv_mtp_compress_kernels.py`

**Reference data (not run outputs, but useful snapshots):**
- `kernel_signatures.json` — parsed .pto signatures (reference, 60KB)
- `kernel_outputs.json` — output buffer name lists (reference, 5KB)

**Rewritten README.md** (see below)

### Group 2: Skill lib modifications (tracked files with local changes)

These 4 files are already tracked but have uncommitted local modifications (adapting for ptoas 0.59 + 312-native + DSV4). Commit them on the current branch:

- `.claude/skills/vpto-board-validate/vpto_run.py` — daemon_env PYTHONPATH for 0.59 312-native, route flag version-detection
- `.claude/skills/vpto-board-validate/lib/pto_parse.py` — DSV4 parsing fixes
- `.claude/skills/vpto-board-validate/lib/setup_main.py` — 0-elem alloc skip, split kernel handling
- `.claude/skills/vpto-board-validate/lib/setup_vpto.py` — DSV4 setup fixes

### Group 3: scripts/vpto_env.sh (tracked, modified this session)

- `scripts/vpto_env.sh` — updated for 0.59 312-native build path + version-detect logic. This stays as YOUR local env (with your paths). The README will document what env vars users need to set.

## What NOT to commit

- `build_output/` — already in .gitignore, runtime generated
- `run_all_build/` — empty placeholder dir
- `__pycache__/` — already in .gitignore
- All `*.csv` sweep result files (4 files) — run outputs
- All `*.json` sweep/diagnosis result files (3 files: crash_diagnosis.json, vmi_sweep_*.json) — run outputs
- `golden/compare_template.py` — upstream skill template, not used by this suite

## README.md rewrite

The current README has inaccuracies (claims "torch golden" but it's pure numpy; references `--enable-vmi` flags that don't exist on main). Rewrite to:

1. **Fix the golden description**: "pure numpy CPU golden reference, no torch dependency"
2. **Fix route documentation**: only `baseline` route works on main (VPTO + insert-sync + op-fusion + bisheng VF ON); `vmi-membar-vfoff` requires the feature-vmi-vf branch
3. **Add a Prerequisites section** listing exactly what env vars the user must set:
   ```
   Prerequisites (set before running):
     PTOAS_BIN         — path to ptoas binary (e.g. /path/to/PTOAS/build/tools/ptoas/ptoas)
     ASCEND_HOME_PATH  — CANN install root (e.g. /usr/local/Ascend/cann-9.1.0-beta.3)
     PTO_ISA_PATH      — path to pto-isa repo
     BISHENG_BIN        — path to bisheng compiler (usually $ASCEND_HOME_PATH/bin/bisheng)
     PTOAS_SOURCE      — path to PTOAS source tree (for ptodsl)
     LLVM_BUILD        — path to LLVM build (for LD_LIBRARY_PATH)
   Optional:
     TILELANG_PATH     — path to TileOps (only for ptoas < 0.54)
     TILELANG_PKG      — path to tilelang-dsl (only for ptoas < 0.54)
   ```
4. **Add a Quick Start** section:
   ```bash
   # 1. Set your environment (see Prerequisites above)
   export PTOAS_BIN=/path/to/ptoas
   export ASCEND_HOME_PATH=/path/to/cann
   export PTO_ISA_PATH=/path/to/pto-isa
   export PTOAS_SOURCE=/path/to/PTOAS
   export LLVM_BUILD=/path/to/llvm-build
   source $ASCEND_HOME_PATH/set_env.sh
   export BISHENG_BIN=$ASCEND_HOME_PATH/bin/bisheng

   # 2. Run all 110 kernels
   python3 test_for_dsv4/run_all.py --device 0 --route baseline

   # 3. Run a single kernel
   python3 test_for_dsv4/run_all.py --device 0 --kernel rms_norm

   # 4. Results are written to test_for_dsv4/sweep_results.csv
   ```
5. **Add a "Latest results" section** with the 16/117 PASS summary table

## emitc_run.py / diagnose_crashes.py hardcoded path cleanup

- `emitc_run.py`: the fallback defaults `/data/liuzidi/...` are already env-overridable (`_env("PTOAS_BIN") or "/data/..."`). Add a comment noting these are dev fallbacks and should be set via env vars. No code change needed — just README documentation.
- `diagnose_crashes.py`: hardcodes `/tmp/ptoas_py311` + `python3.11` (the old shim). Update to use the same env-overridable pattern as run_all.py (use `$PTOAS_BIN` from env, use `python3` not `python3.11`).

## Execution steps

1. **Fix diagnose_crashes.py** — replace hardcoded `/tmp/ptoas_py311` + `python3.11` with env-based approach matching run_all.py
2. **Rewrite README.md** — fix inaccuracies, add Prerequisites + Quick Start
3. **Add .gitignore entries** inside test_for_dsv4/ for `*.csv`, `crash_diagnosis.json`, `vmi_sweep_*.json`, `run_all_build/` (keep `kernel_signatures.json` + `kernel_outputs.json`)
4. **Stage and commit Group 2** (skill lib modifications) — separate commit: "Fix: adapt vpto-board-validate skill for ptoas 0.59 + 312-native + DSV4"
5. **Stage and commit Group 1 + Group 3** (test_for_dsv4 + vpto_env.sh) — main commit: "Add test_for_dsv4: DSV4 VPTO precision test suite (110 kernels, numpy golden)"

Two commits keeps the skill adaptation separate from the test suite addition, making review easier.