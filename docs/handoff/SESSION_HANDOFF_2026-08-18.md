# DSV4 VPTO Validation — Session Handoff (2026-08-18)

> **Context:** This document is a handoff for continuing the DSV4 VPTO precision
> validation work on a different machine. The original dev box was being reclaimed,
> so the last batch of changes was rushed into a single commit (`ecd9648`,
> message "tilelib commit") and pushed to `fork/test-ds-a5`. This file explains
> the whole conversation history, the current state, and what the rushed commit
> actually contains so a new agent can pick up cleanly.

---

## 1. Project background (one-paragraph TL;DR)

We are validating the **DSV4 (DeepSeek v4) kernel suite** through the **VPTO
route** on an Ascend A5 NPU (Ascend950PR). The VPTO route is:
`ptoas --pto-backend=vpto` → fatobj → `bisheng` → NPU, then compare NPU output
against a pure-numpy golden reference (no torch dependency). The test suite lives
in `test_for_dsv4/` (110 `.pto` kernels + golden builders + a sweep driver).
The goal of this session line was: get the suite passing against the latest
`ptoas` build (0.59, cpython-3.12 native, branch `main-llvm19-build`).

## 2. What works now / current PASS state

- **ptoas binary:** use the **pip-installed wheel entry** (`~/.local/bin/ptoas`),
  NOT the build-tree wrapper (`$PTOAS_SOURCE/build/tools/ptoas/ptoas`). The
  wrapper is broken — see §4.
- **Expected PASS count: 18/103** (17 from the si8 sweep + `qproj_matmul`
  restored after its revert). This was NOT re-verified end-to-end after the
  final revert; the natural first task on the new box is to run a confirmation
  sweep:
  ```bash
  python3 test_for_dsv4/run_all.py --device 0 --route baseline
  ```
  and check `test_for_dsv4/sweep_results.csv`.
- The remaining ~85 kernels compile but crash at runtime with
  `retCode=0x26`/`0x31`. These are ptoas VPTO codegen or bisheng issues, not
  test-suite bugs — see §5.

## 3. Conversation history (this session line, chronologically)

1. **Run full VPTO suite with latest ptoas.** User asked to run all tests with
   the newest ptoas artifact (now on `main-llvm19-build`).
2. **Review PTOAS issue #1262 reply.** The issue was about `pto.tcvt` rejecting
   signless `i8`. PTOAS responded that VPTO TileLib templates require explicit
   signedness; only `f16->si8` and `f16->ui8` are registered, not `f16->i8`.
3. **Change i8 → si8 in .pto test cases.** Since the suite is `.pto`-file-based
   and pypto-lib hadn't adapted, user directed changing all `i8` types in the
   test `.pto` files to the standard `si8` form. 23 files changed (788
   `dtype=i8`→`si8`, 308 `xi8>`→`xsi8>`, 49 `ptr<i8>`→`ptr<si8>`). Commit
   `d305298`.
4. **"所以没啥变化？"** — after the si8 fix the net PASS gain was small
   (15→17). Explanation: si8 fixed the `pto.tcvt` PTODSL metadata failures, but
   many of those kernels then proceeded to crash at runtime instead, so the PASS
   count only nudged up by 2.
5. **Revert qproj_matmul.pto to i8.** `pto.tload gm_to_mat` templates do NOT
   support `('si8','si8')` (`NoMatchingTemplate`), so the si8 change broke
   `qproj_matmul`. User directed reverting just that one file to i8 (restores
   its PASS) while keeping the other 22 si8 fixes. Commit `e888233`.
6. **Investigate root-level `compare.py`.** User asked what
   `/data/liuzidi/pypto-lib/compare.py` was for and why it wasn't pushed. It was
   an orphan untracked file — an old version of the compare logic (missing fp32
   atol, missing integer ULP tolerance), superseded by the tracked
   `test_for_dsv4/golden/compare.py`. It was never `git add`ed; `rm`'d as
   garbage.
7. **Machine being reclaimed → rush commit.** User pushed everything outstanding
   as commit `ecd9648` ("tilelib commit") to `fork/test-ds-a5` and asked for
   this handoff.

## 4. Key technical facts a new agent MUST know

### 4.1 ptoas binary selection (critical, do not regress)

- **Use the wheel:** `~/.local/bin/ptoas` (self-contained; `_core.so` and
  `_mlir.so` md5-identical to the build).
- **Do NOT use the build-tree wrapper**
  `$PTOAS_SOURCE/build/tools/ptoas/ptoas`. It calls
  `_disable_editable_import_redirects()` and its `build/python/ptoas/mlir/` is
  partial (has PTO dialect + `_core.so` but NOT `ir.py` or
  `_mlirRegisterEverything.so`), so it throws
  `ModuleNotFoundError: No module named 'ptoas.mlir.ir'`.
- `vpto_run.py` (in `.claude/skills/vpto-board-validate/`) detects wheel vs
  build-tree wrapper and only overrides PYTHONPATH for the wrapper case. Don't
  break this detection.

### 4.2 PYTHONPATH / LLVM_BUILD

- `LLVM_BUILD=/data/liuzidi/llvm-workspace/llvm-project/build-shared` (cpython-3.12).
  The old `/data/c00862531/...` path is cpython-3.11 and fails under py3.12 —
  do not revert to it.
- For the wheel entry: do NOT overlay `build/python` or `/tmp/mlir_core_vmi`
  on PYTHONPATH. Both build/python and mlir_core ship `_mlir.so`; having both
  on sys.path causes namespace-merge ABI conflicts:
  `"PyCapsule_GetPointer called with incorrect name"` /
  `"Unable to cast Python instance... to C++ type 'MlirContext'`.
- `scripts/vpto_env.sh` encodes all of this; it is committed (with dev paths —
  README documents what env vars a new user must set).

### 4.3 pypto type system vs ptoas TileLib

- pypto has NO `si8` token — `INT8`→`"i8"` (MLIR convention, signed), `UINT8`→
  `"ui8"`. So the si8 fix is **test-case-side**, not pypto-side.
- ptoas VPTO TileLib templates require explicit signedness: `f16->si8` and
  `f16->ui8` registered, NOT `f16->i8`.
- `pto.tload gm_to_mat` does NOT support `('si8','si8')` yet — that's why
  `qproj_matmul.pto` stays on `i8`. This is a ptoas-side template gap; I offered
  to file a PTOAS issue for it but the user redirected to reverting. **Open
  question:** file the PTOAS issue for the `tload gm_to_mat` si8 gap? (Not yet
  done.)

### 4.4 Golden reference

- Pure numpy CPU golden, NO torch dependency. bf16 ULP comparison with
  `VPTO_COMPARE_MAX_ULP=3` (default). `test_for_dsv4/golden/compare.py` is
  canonical; `test_for_dsv4/compare.py` is a relative symlink to it.

## 5. The remaining ~85 runtime crashes

After the si8 fix, most non-passing kernels compile cleanly through ptoas but
crash on the NPU with `retCode=0x26`/`0x31`. These are ptoas VPTO codegen or
bisheng issues, NOT test-suite bugs. `test_for_dsv4/diagnose_crashes.py`
classifies them. This is the main body of remaining work — but it's ptoas/
bisheng-side, so it likely needs upstream fixes or issue filings, not
test-suite changes.

## 6. What the rushed commit `ecd9648` ("tilelib commit") actually contains

This is the important part — the commit message is vague, so here is the
itemized contents and a verdict on each:

| File | What it is | Verdict |
|---|---|---|
| `.claude/plans/route2-simpler-findings.md` (1723 lines) | Large findings doc on Route 2 / simpler fatobj loading | keep (real content) |
| `.claude/plans/simpler-route2-fatobj-loading.md` (916 lines) | Fatobj loading investigation | keep (real content) |
| `.claude/plans/vpto-dsv4-vector-validation.md` (178 lines) | DSV4 vector validation plan | keep (real content) |
| `.zcode/plans/plan-sess_*.md` (5 files) | ZCode session plan files — actual engineering plans (commit-test-suite, C6/C7b/C8b fixes, C1b shape scalars, C3/C4 fix) | keep (substantive), but note these are tool-session artifacts |
| `BASELINE_COMPARISON.md` (root) | Baseline comparison writeup | keep |
| `baselines/vpto_dsv4_vector/PRECISION_REPORT.md` (728 lines) | Precision report | keep |
| `baselines/vpto_dsv4_vector/PTOAS_0.59_REGRESSION_REPORT.md` (269 lines) | ptoas 0.59 regression report | keep |
| `baselines/vpto_dsv4_vector/SUMMARY_TABLE.md` (50 lines) | Summary table | keep |
| `baselines/vpto_dsv4_vector/sweep_results_vmi-membar-vfoff.csv` (modified) | Sweep results CSV | arguably a run output, but already partially tracked before — keeping is fine |
| `baselines/.../sweep_results_vmi-membar-vfoff.csv.prev_partial` | Partial previous sweep | **should be removed** (run-output leftover) |
| `docs/debug-and-tune/vpto-msprof-pmu-collection.md` (modified) | PMU collection procedure | keep |
| `test_for_dsv4/sweep_results.csv.tmp` (2 lines) | Stray partial CSV header+1 row | **should be removed** (garbage) |

**Cleanup recommended on the new box (do this first):**
```bash
git rm --cached test_for_dsv4/sweep_results.csv.tmp
git rm --cached baselines/vpto_dsv4_vector/sweep_results_vmi-membar-vfoff.csv.prev_partial
# add to .gitignore:
echo "sweep_results.csv.tmp" >> test_for_dsv4/.gitignore
echo "*.csv.prev_partial" >> baselines/vpto_dsv4_vector/.gitignore  # or root .gitignore
git commit -m "Cleanup: remove stray .tmp and .prev_partial run-output files from rushed commit"
```

**Also note:** `.claude/settings.json` and `.claude/plans/vpto-dsv4-phase3-handoff.md`
exist locally but were NOT committed (still untracked). If you want them shared,
`git add` them explicitly.

## 7. Git state

- **Branch:** `test-ds-a5`
- **Upstream:** `fork/test-ds-a5` (git@github.com:liuzidi/pypto-lib.git) —
  this is the user's personal fork; `origin` is
  `https://github.com/hw-native-sys/pypto-lib.git` (the shared repo).
- **All session work is pushed to `fork/test-ds-a5`.** Nothing is stranded
  locally (working tree is clean as of the handoff).
- **Commit chain on this branch (since `origin/main`):** see `git log --oneline
  origin/main..HEAD`. The most recent 3 are the si8 work:
  - `ecd9648` tilelib commit (the rushed one — see §6)
  - `e888233` Revert: keep qproj_matmul.pto as signless i8
  - `d305298` Fix: replace signless i8 with si8 in DSV4 .pto test cases

## 8. Suggested next steps on the new box (in order)

1. **Clean up the rushed commit** — remove the `.tmp` and `.prev_partial`
   files (§6). Small, safe, do first.
2. **Set up the environment** following `scripts/vpto_env.sh` and the
   Prerequisites in `test_for_dsv4/README.md`. Key: point `PTOAS_BIN` at the
   wheel, set `LLVM_BUILD` to a cpython-3.12 build.
3. **Run a confirmation sweep:**
   `python3 test_for_dsv4/run_all.py --device 0 --route baseline` and verify
   the PASS count is ~18/103. If it's 0, the env is wrong (re-check
   `PTOAS_BIN`/PYTHONPATH per §4).
4. **(Open question for user)** File a PTOAS issue for the
   `pto.tload gm_to_mat` si8 template gap (so `qproj_matmul` can eventually
   move to si8 too). Not done yet.
5. **Main body of work:** triage the ~85 runtime crashes
   (`retCode=0x26`/`0x31`) via `diagnose_crashes.py`; these need ptoas/bisheng
   upstream fixes or issue filings.

## 9. Pointers into the codebase

- Test suite: `test_for_dsv4/` (driver `run_all.py`, golden in `golden/` +
  `golden_parts/`, `.pto` inputs in `.pto/`).
- Skill (VPTO execution): `.claude/skills/vpto-board-validate/` —
  `vpto_run.py` (route driver), `lib/setup_main.py` + `lib/setup_vpto.py`
  (buffer setup), `lib/pto_parse.py` (.pto parser), `runtime/compare.py`.
- Env script: `scripts/vpto_env.sh`.
- Prior plans/findings: `.claude/plans/` (route2-simpler-findings,
  simpler-route2-fatobj-loading, vpto-dsv4-vector-validation).
- Precision reports: `baselines/vpto_dsv4_vector/PRECISION_REPORT.md`,
  `PTOAS_0.59_REGRESSION_REPORT.md`, `SUMMARY_TABLE.md`.
