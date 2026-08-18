# PTOAS 0.59 VMI Regression Report — DSV4 Board Precision Sweep

Date: 2026-08-14
Author: liuzidi (via ZCode agent)
Branch: `feature-vmi-vf-rebase`
PTOAS repo: `/data/liuzidi/PTOAS` (build311)
Device: Ascend A5 NPU (Ascend950PR), device 6

## Executive summary

A full DSV4 VMI sweep (117 kernel rows) was run twice against ptoas 0.59:

| Run | PTOAS HEAD | Uncommitted changes | PASS | NPU crash | Precision fail |
|---|---|---|---:|---:|---:|
| Run A (2026-08-13) | `de6eb783` | `VMIToVPTO.cpp` (+3 lines) | 23 | 86 | 8 |
| Run B (2026-08-14) | `bb82a670` | `PTOOpScheduling.cpp`, `_vmi_common.py` (+ more) | 15 | 97 | 5 |

Run B introduces a **net regression of 8 PASS kernels** (9 regressed, 2
improved). The regression is reproducible per-kernel (not intermittent) for
7 of 9 kernels; the other 2 are intermittent daemon-socket races.

## Environment

### PTOAS build

```
repo:     /data/liuzidi/PTOAS
branch:   feature-vmi-vf-rebase
HEAD:     bb82a670fb457d8faf74336e136512c6f15a811b  (Run B)
          de6eb783 (Run A — the "23 PASS" baseline)
version:  ptoas 0.59
build:    build311 (Ninja, Release, Python 3.11, LTO)
python:   .venv311/bin/python (CPython 3.11.15) — REQUIRED, system python3
          is 3.12 and the mlir_core C extensions are cpython-311.
```

### Commits between Run A and Run B (the regression range)

```
bb82a670 fix(ptoas): use explicit bool for op-fusion gate and clean up stale pass refs
212058b7 docs(vmi): update fa-softmax-dn-init-rowplusone README for new defaults
8ff598a5 fix(ptoas): always disable bisheng VF fusion on VPTO path
6162fe60 fix(vmi): restore load/store elision after pointer_cast→castptr rebase
bd675f90 fix(vmi): keep legacy loop fusion separate
```

### Uncommitted changes present in Run B (not in Run A)

| File | Change summary |
|---|---|
| `lib/PTO/Transforms/TileFusion/PTOOpScheduling.cpp` | `normalizeBlockFusionMetadata`: keep singleton spans (was: drop spans < 2 members). **Suspect for ExpandTileOp regressions.** |
| `lib/TileOps/a5/_vmi_common.py` | Add `si8`/`si16`/`si32` signed ScalarTypes; add `rows * cols in _VMI_LANE_COUNTS` guard to `row_reduce_vmi_constraint`. **Suspect for rms_norm precision regression** (row_reduce constraint changed). |
| `test/lit/...` (several .pto) | lit test updates (not codegen-affecting) |
| `.gitignore` | unrelated |

## VMI route flags (exact)

```bash
# ptoas (the shim wraps python3.11 + the real wrapper):
/tmp/ptoas_py311 \
  --pto-arch=a5 --pto-level=level3 --pto-backend=vpto \
  --enable-tile-op-expand --enable-insert-sync \
  --enable-vmi --enable-op-fusion=true --enable-vecscope-mem-bar \
  <preprocessed.pto> -o <out.o>

# bisheng launch.o compile (7 VF-off -mllvm + stack bump):
bisheng -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
  -fPIC -Xhost-start -Xhost-end \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch=dav-c310-vec \
  -DREGISTER_BASE -std=c++17 \
  -Wno-macro-redefined -Wno-ignored-attributes \
  -I <CANN>/include -I <CANN>/pkg_inc -I <pto-isa>/include -I <pto-isa>/tests/common \
  -mllvm -cce-vf-enable-vf-fusion=false \
  -mllvm -cce-vf-enable-vf-loop-extender=false \
  -mllvm -cce-vf-enable-loop-fusion=false \
  -mllvm -cce-vf-enable-vf-ldst-elimination=false \
  -mllvm -cce-vf-enable-ub-dead-st-elimination=false \
  -mllvm -cce-vf-auto-sync=off \
  -mllvm -cce-vf-enable-vf-ifelse-extender=false \
  -mllvm -cce-vf-stack-size=0x10000 \
  launch.cpp -o launch.o
```

## Regression detail — 4 distinct failure classes

### Class 1: `pto.tdivs` i32 NoMatchingTemplate (4 kernels, STABLE)

**Affected:** `rope`, `qr_rope`, `prefill_idx_qr_rope`, `qproj_dequant_rms_nope_rope`

**Full error:**
```
NoMatchingTemplate: no legal template for op='pto.tdivs' target='a5';
  template_tdivs_tile_scalar: dtype signature ('i32','i32','i32') is not supported;
  template_tdivs_scalar_tile: dtype signature ('i32','i32','i32') is not supported;
  template_tdivs_tile_scalar_1d: dtype signature ('i32','i32','i32') is not supported;
  template_tdivs_scalar_tile_1d: dtype signature ('i32','i32','i32') is not supported
loc("build_output/vpto_rope/rope.pto":105:7): error: in-process PTODSL metadata query failed
Error: Pass execution failed.
```

**Was:** PASS (all 4) in Run A. In Run A the VMI fusion pipeline fused
away the `pto.tdivs` op for these RoPE kernels, so the i32 tdivs template
was never needed. Run B's `6162fe60 fix(vmi): restore load/store elision`
or `bd675f90 fix(vmi): keep legacy loop fusion separate` likely changed
the fusion boundary so tdivs is no longer fused → falls back to the
unsupported i32 template.

**Repro:**
```bash
cd /data/liuzidi/pypto-lib/test_for_dsv4
source /data/liuzidi/pypto-lib/scripts/vpto_env.sh
export PTOAS_BIN=/tmp/ptoas_py311 ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
rm -f /tmp/tilelib_daemon_*.sock
python3.11 /data/liuzidi/pypto-lib/.claude/skills/vpto-board-validate/vpto_run.py \
  --pto .pto/rope.pto --golden-lib dsv4_golden_lib.py \
  --kernel rope --mode decode --device 6 --route vmi-membar-vfoff
```

**Key file:** `/data/liuzidi/pypto-lib/test_for_dsv4/.pto/rope.pto` line 105
has the `pto.tdivs` op that triggers the error.

### Class 2: `ExpandTileOp` template instantiation failure (3 kernels, STABLE)

**Affected:** `hc_pre_seed`, `prefill_idx_score_init`, `exp_h_q`

**Full error (hc_pre_seed representative):**
```
/data/liuzidi/PTOAS/build311/python/ptodsl/_tile_template_tracing.py(570): trace_entry
/data/liuzidi/PTOAS/build311/python/ptodsl/_tile_template_tracing.py(850): build_module_in_context
ExpandTileOp: in-process PTODSL materialization failed
loc("build_output/vpto_hc_pre_seed/hc_pre_seed.pto":12:5): error: ExpandTileOp:
  failed to instantiate TileLib template for texpands
```

**Was:** PASS (all 3) in Run A. The uncommitted
`PTOOpScheduling.cpp` change (keep singleton spans instead of dropping
spans < 2 members) is the prime suspect — it changes which fusion
regions survive, which in turn changes which `texpands` ops the VMI
pipeline emits and which TileLib template they match.

**Note:** `exp_h_q` is **intermittent** — single-run PASS, sweep FAIL.
`hc_pre_seed` and `prefill_idx_score_init` are **stable** failures.

**Repro:**
```bash
# (same env as Class 1)
python3.11 .../vpto_run.py --pto .pto/hc_pre_seed.pto --golden-lib dsv4_golden_lib.py \
  --kernel hc_pre_seed --mode decode --device 6 --route vmi-membar-vfoff
```

### Class 3: PTODSL metadata query failure (1 kernel, STABLE)

**Affected:** `kv_proj_matmul`

**Full error:**
```
TileLib: PTODSL metadata query raised Python exception:
  /data/liuzidi/PTOAS/build311/python/ptodsl/tilelib/_selection.py(238): _require_unambiguous_top_candidate
  /data/liuzidi/PTOAS/build311/python/ptodsl/tilelib/_selection.py(320): metadata_request
  /data/liuzidi/PTOAS/build311/python/ptodsl/tilelib/_compiler_runtime.py(32): metadata
loc("build_output/vpto_kv_proj_matmul/kv_proj_matmul.pto":50:7): error: in-process PTODSL metadata query failed
Error: Pass execution failed.
```

**Was:** PASS in Run A. The `_require_unambiguous_top_candidate` exception
means multiple TileLib templates match equally well for a `trowm` op and
the disambiguation logic can't pick one. This is likely a side effect of
the `_vmi_common.py` uncommitted changes (new `si8`/`si16`/`si32` types
may have added overlapping template candidates).

**Repro:**
```bash
python3.11 .../vpto_run.py --pto .pto/kv_proj_matmul.pto --golden-lib dsv4_golden_lib.py \
  --kernel kv_proj_matmul --mode decode --device 6 --route vmi-membar-vfoff
```

### Class 4: rms_norm precision regression (1 kernel, STABLE)

**Affected:** `rms_norm`

**Error:**
```
[ERROR] bf16 compare failed (v2): max_ulp=829 idx=181
  golden_bits=15999 output_bits=15170
  golden=0.2490234375 output=0.002960205078125
[ERROR] compare failed
```

**Was:** PASS (max_ulp=1) in Run A. Now max_ulp=829 — a massive precision
deviation (golden 0.249 → output 0.003, 829 ULP off). The NPU ran to
completion (no crash) but produced wrong output.

**Suspect:** The uncommitted `_vmi_common.py` change added a
`rows * cols in _VMI_LANE_COUNTS` guard to `row_reduce_vmi_constraint`.
rms_norm's reduction is a `row_reduce`; if this new guard rejects the
previous (correct) reduction path and forces a different (incorrect)
fallback, the reduction accumulator would produce garbage → 829 ULP.

**Repro:**
```bash
python3.11 .../vpto_run.py --pto .pto/rms_norm.pto --golden-lib dsv4_golden_lib.py \
  --kernel rms_norm --mode decode --device 6 --route vmi-membar-vfoff
```

### Intermittent (daemon-socket race, NOT a real regression)

**Affected:** `exp_h_q`, `hc_post_prefill`, `x_norm_quant`

These 3 kernels FAIL in the serial sweep but **PASS** when run individually
with `rm -f /tmp/tilelib_daemon_*.sock` before each run. The sweep's
per-kernel socket cleanup races with residual daemon processes from the
previous kernel. These are NOT real regressions — they are a framework
issue (vpto_run.py's `glob("tilelib_daemon_*.sock").unlink()` can delete a
socket that another user's daemon is using, or a stale daemon lingers).

## Improved kernels (2, FAIL→PASS)

| Kernel | Run A | Run B |
|---|---|---|
| `mtp_projection_output` | NPU crash | PASS |
| `qproj_matmul` | NPU crash | PASS |

These improvements are real (stable across individual re-runs).

## Full result comparison

| metric | Run A (de6eb783) | Run B (bb82a670) | delta |
|---|---:|---:|---:|
| total rows | 117 | 117 | — |
| PASS | 23 | 15 | **-8** |
| NPU crash (exit 1) | 86 | 97 | +11 |
| precision fail (has max_diff) | 8 | 5 | -3 |
| sweep elapsed | 12.5 min | 14.2 min | — |

## Suggested investigation order

1. **Stash uncommitted changes** (`PTOOpScheduling.cpp` + `_vmi_common.py`)
   and re-run the sweep. If 23 PASS returns, the regression is in the
   uncommitted work, not in the 5 new commits.

2. If stash doesn't fix it, **bisect the 5 commits** (`bd675f90` → `bb82a670`):
   - `6162fe60 fix(vmi): restore load/store elision` — prime suspect for
     Class 1 (tdivs fusion boundary change) and Class 2 (ExpandTileOp).
   - `bd675f90 fix(vmi): keep legacy loop fusion separate` — second suspect
     for Class 1/2.

3. **Class 4 (rms_norm max_ulp=829)**: the `_vmi_common.py` uncommitted
   `rows * cols in _VMI_LANE_COUNTS` guard is the only change to
   `row_reduce_vmi_constraint`. Revert just that one guard and re-test
   rms_norm — if max_ulp returns to 1, the guard is too strict.

4. **Class 3 (kv_proj_matmul)**: the `_vmi_common.py` new `si8`/`si16`/`si32`
   types may have introduced ambiguous template candidates. Check
   `_require_unambiguous_top_candidate` in
   `ptodsl/tilelib/_selection.py:238`.

## Artifact locations

- Sweep CSV: `/data/liuzidi/pypto-lib/test_for_dsv4/sweep_results.csv`
- Sweep log: `/tmp/dsv4_vmi_sweep_latest.log`
- Build output per kernel: `/data/liuzidi/pypto-lib/test_for_dsv4/build_output/vpto_<kernel>/`
- .pto files: `/data/liuzidi/pypto-lib/test_for_dsv4/.pto/<kernel>.pto`
- vpto_run.py: `/data/liuzidi/pypto-lib/.claude/skills/vpto-board-validate/vpto_run.py`
- ptoas shim: `/tmp/ptoas_py311` (wraps `python3.11 /data/liuzidi/PTOAS/build311/tools/ptoas/ptoas`)
