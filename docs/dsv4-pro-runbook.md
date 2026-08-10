# DeepSeek-V4 Pro Operator Runbook (A5)

Quick-reference runbook for running the `deepseek_v4_pro` single-card
operators on A5. Condensed from the auto-session notes; the long-form
guide with full explanations is
[`run-deepseek-v4-pro-single-ops.md`](run-deepseek-v4-pro-single-ops.md)
(§-references below point into it).

This is the "what to actually type and what to watch for" companion to
the guide. Environment-specific gotchas live here because they are not
in the canonical [`get-started/installation.md`](get-started/installation.md).

---

## 1. The one-line run

```bash
source .env_a5.sh
python scripts/run_op_pmu.py <op> -p a5 -d <free-card>          # default
python scripts/run_op_pmu.py <op> -p a5 -d <free-card> --no-fusion   # precision-safe
python models/deepseek_v4_pro/<op>.py -p a5 -d <free-card>      # without wrapper
```

`<op>` is the module name under `models/deepseek_v4_pro/` without
`.py`. See guide §5 for the 27-op list.

## 2. The headline result

With ptoas A5 op-fusion **disabled** (`--no-fusion`), all 27 single-card
operators **PASS** (27/27). With default fusion on, only 9 pass; 5
precision-FAIL and 13 runtime-error. **The single root cause of all 18
failures is ptoas op-fusion** — fix belongs in ptoas, `--no-fusion` is
the workaround. Full sweep + per-op table: guide §10 and Appendix A.

## 3. Environment gotchas (host-specific)

These walls were hit when building pypto-lib from scratch on an
8× Ascend950PR Ubuntu host. Not in the canonical install guide.

1. **GitHub HTTPS blocked / times out.** Clone over SSH:
   `git@github.com:hw-native-sys/pypto.git`. Avoid the `gitcode.com/cann/pypto`
   mirror — wrong layout (no `runtime/` submodule, no `pypto.language`).

2. **PyPI `pypi.org/simple/` flaky.** Use the Aliyun mirror:
   `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`,
   `PIP_TRUSTED_HOST=mirrors.aliyun.com`.

3. **Source CANN before installing simpler**, or A5 onboard runtime
   binaries are not built. `export CANN_ROOT=/usr/local/Ascend/cann-9.1.0-beta.3;
   source "$CANN_ROOT/set_env.sh"; which ccec` must resolve, THEN
   `pip install --no-build-isolation -e "$PYPTO_ROOT/runtime"`.

4. **PTOAS release download throttled.** `gh release download` (CDN
   route) drops mid-stream ~52 MB. Use the `gh api` octet-stream route:
   `gh api -H "Accept: application/octet-stream" \
   repos/hw-native-sys/PTOAS/releases/assets/<id>`. Verify SHA256
   against `$PYPTO_ROOT/toolchain/versions.env`.

5. **A locally-built PTOAS may be broken** if it links against another
   user's external LLVM (`libMLIRTargetCpp.so.*: cannot open shared
   object file`). Use the pinned release binary instead.

6. **pto-isa local-mod conflict.** A working-tree change to
   `include/pto/common/type.hpp` can block `checkout` of the pin. `git
   stash` it, check out `runtime/pto_isa.pin`, `git stash pop` after.

7. **Pins come from the selected PyPTO checkout** (source of truth):
   PTOAS version from `toolchain/versions.env`, PTO ISA commit from
   `runtime/pto_isa.pin`. Never hard-code from an old log.

8. **Full network OOMs on a ~755 GB host.** `decode_fwd.py`'s
   `_make_layer_stacked_spec` does `torch.cat([base_init() for _ in
   range(61)], dim=1)`, peaking ~774 GB. Not fixable here; CI uses a
   large-memory node. Don't run the full network for precision — run
   the single ops (§1).

A one-shot activation script lives at `.env_a5.sh` (repo root). It
sources CANN, activates the venv, sets `PIP_INDEX_URL`, and exports
`PYPTO_LIB_ROOT` / `PYPTO_ROOT` / `PTO_ISA_ROOT` / `PTOAS_ROOT` /
`PYTHONPATH`. See guide §2.5 for its contents.

## 4. Runtime error vs precision FAIL

Two failure kinds look similar; they mean very different things.

**Precision FAIL** — kernel ran to completion, output mismatched golden:
- `'y' FAIL ... (ratio_allclose(...))` with `error_count=N/M (ratio=X%, allowed<=Y%)`
- ends with `Output(s) does not match golden: [...]`
- investigate operator numerics (`docs/debug-and-tune/precision-tuning.md`)

**Runtime / DFX error** — kernel did NOT complete, no comparison:
- `RuntimeError: run failed with code 507901` (AICore bounded-drain
  timeout; card force-reset) or `code -100` (simpler scheduling stall)
- traceback ends in `_execute_dfx_passes` → `execute_on_device` → `simpler.worker.run`
- NO `'...' FAIL` line, NO `Output(s) does not match golden`
- investigate runtime/DFX layer, not numerics

**Correction (important):** An earlier read attributed the 507901/-100
errors on this op set to the `dep_gen` DFX pass hitting pypto#1931.
The `--no-fusion` full sweep **disproved that** — all 13 cleared when
fusion was disabled. The real cause on `deepseek_v4_pro` is op-fusion
generating hanging/stalling AICore code. The dep_gen/#1931 path is still
*possible* in general (look for code **507018** specifically), but was
not the cause here. Full story: guide §6.

DFX = simpler's diagnostic passes around kernel execution: full-occupancy
`pl.system.syncall`, dependency-gen (`dep_gen`), PMU, L2 swimlane, scope
stats. Five toggles (`golden/runner.py` `_DFX_FLAG_KEYS`):
`enable_l2_swimlane, enable_dump_args, enable_pmu, enable_dep_gen, enable_scope_stats`.

## 5. Isolation checklist (when an op fails)

1. **Re-run with `--no-fusion` first.** If it passes → fusion root cause
   (the common case on this op set). Done.
2. `--compile-only` to confirm clean compile (fusion bugs surface at runtime).
3. If still failing under `--no-fusion`, look at DFX: check `# ci:`
   markers (`rg -n '^#\s*ci:' <op>.py`) and look for code **507018** (the
   #1931 signature).
4. Compare the standalone leaf kernel the op calls — if it passes in
   isolation, the failure is orchestration, not numerics.

## 6. PMU and performance

- **PMU:** `python scripts/run_op_pmu.py <op> -p a5 -d <card> --pmu N`
  (N: 0=off, 1 ARITH, 2 PIPE_UTIL default, 4 MEMORY, 5 MEM_L0, 6
  RESRC_CONFL, 7 MEM_UB, 8 L2_CACHE). Output:
  `build_output/_jit_<fn>_<ts>/dfx_outputs/pmu.csv` (one row per task).
  `--no-fusion` composes with `--pmu`. Full PMU reference: guide §9.
- **Benchmark:** `PYPTO_BENCH=1` prints `eff_us min/median/mean/max` per
  dispatch. `PYPTO_BENCH_RAW=1` for per-round spans. Rounds/warmup via
  `PYPTO_BENCH_ROUNDS` / `PYPTO_BENCH_WARMUP`.

## 7. Picking a free card

Shared host — don't race for an "idle" card by probing. Check
`npu-smi info`, read the process table at the bottom, pick a card with
no running processes. For single-card ops you need one free 128 GB card.
Re-confirm before each run (occupancy changes minute to minute).
