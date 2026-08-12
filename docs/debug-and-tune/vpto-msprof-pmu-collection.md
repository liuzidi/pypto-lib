# VPTO Board PMU Collection — `msprof op` on NPU-Built Kernels

> How to collect A5 **VEC-only cycle** (and full pipe utilization) from a
> kernel built and run via the `vpto-board-validate` skill, using CANN's
> `msprof op` operator profiler. This gives the same hardware PMU counters
> as the `incore-profiling` skill's simulator path, but on **real silicon**
> after the full ptoas→bisheng→CANN route.

## 1. When to use this

- You want **`aiv_vec_time`** (vector compute cycle, excluding MTE
 搬运/sync) for a VPTO kernel — e.g. to measure VMI VF-fusion benefit.
- The `vpto_run.py` `[TIMING]` line is host-side wall-clock (includes MTE +
  CANN runtime overhead); it cannot separate VEC from MTE.
- You already have a built `vpto_run.py` binary (the skill's `--build-dir`
  output) that runs correctly on the device.

## 2. Prerequisites

1. `scripts/vpto_env.sh` sourced (sets `PTOAS_BIN`, `ASCEND_HOME_PATH`,
   builds `/tmp/mlir_core_vmi`).
2. CANN `setenv.bash` sourced (sets `LD_LIBRARY_PATH` for msprof's shared
   libs):
   ```bash
   source "$ASCEND_HOME_PATH/bin/setenv.bash"
   ```
3. `PTOAS_ROOT` exported:
   ```bash
   export PTOAS_ROOT="$(dirname "$PTOAS_BIN")"
   ```
4. The kernel binary built by `vpto_run.py` (see step 3 below).

## 3. Step-by-step procedure

### 3.1 Build the kernel binary (if not already built)

```bash
# Baseline route:
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_<module>_test_*/ptoas/<kernel>.pto \
  --model-py models/deepseek_v4_pro/<module>.py \
  --mode decode --device 0 --kernel <kernel> --route baseline \
  --captured-dump <run_dir_with_vN_bin> \
  --build-dir build_output/vpto_<kernel>_baseline

# VMI+membar+vf-off route (for comparison):
#   same but --route vmi-membar-vfoff
#   and --build-dir build_output/vpto_<kernel>_vmi
```

Verify it runs:
```bash
export LD_LIBRARY_PATH="build_output/vpto_<kernel>_<route>:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH}"
cd build_output/vpto_<kernel>_<route>/run
../<kernel>   # should print "[TIMING] kernel=<name> time=X ms"
```

### 3.2 Create a wrapper script (msprof needs correct cwd)

`msprof op` runs the binary from its own cwd, but the binary reads
`./vN.bin` from its working directory. Create a wrapper:

```bash
REPO=/data/liuzidi/pypto-lib    # adjust if different
KERNEL=<kernel>                 # e.g. qr_proj_seed
ROUTE=<route>                   # baseline or vmi-membar-vfoff

cat > build_output/vpto_${KERNEL}_${ROUTE}/run_wrapper.sh <<EOF
#!/bin/bash
cd ${REPO}/build_output/vpto_${KERNEL}_${ROUTE}/run
exec ${REPO}/build_output/vpto_${KERNEL}_${ROUTE}/${KERNEL}
EOF
chmod +x build_output/vpto_${KERNEL}_${ROUTE}/run_wrapper.sh
```

### 3.3 Run `msprof op`

```bash
MSPROF="$ASCEND_HOME_PATH/tools/profiler/bin/msprof"
PROF_OUT="build_output/msprof_${KERNEL}_${ROUTE}"
rm -rf "$PROF_OUT"; mkdir -p "$PROF_OUT"; chmod 700 "$PROF_OUT"

"$MSPROF" op \
  --aic-metrics=PipeUtilization \
  --kernel-name="$KERNEL" \
  --launch-count=1 \
  --warm-up=0 \
  --output="$PROF_OUT" \
  "build_output/vpto_${KERNEL}_${ROUTE}/run_wrapper.sh"
```

### 3.4 Read the results

```bash
# OpBasicInfo — task duration, block dim, frequency
cat build_output/msprof_${KERNEL}_${ROUTE}/OPPROF_*/OpBasicInfo.csv

# PipeUtilization — the VEC/MTE/scalar breakdown
cat build_output/msprof_${KERNEL}_${ROUTE}/OPPROF_*/PipeUtilization.csv
```

## 4. Key metrics in `PipeUtilization.csv`

| column | what it measures | unit |
|---|---|---|
| `aiv_vec_time(us)` | **VEC compute time (no MTE)** — the metric you want | µs |
| `aiv_total_cycles` | total AIV cycles | cycles |
| `aiv_vec_ratio` | VEC fraction of total cycles | 0–1 |
| `aiv_scalar_time(us)` | scalar compute time | µs |
| `aiv_mte2_time(us)` | MTE2 load (GM→UB) time | µs |
| `aiv_mte3_time(us)` | MTE3 store (UB→GM) time | µs |
| `aic_*` columns | AIC (cube) pipe metrics — `NA` for vector kernels | — |

**Derived metric:**
- VEC cycles = `aiv_total_cycles × aiv_vec_ratio`
- VEC time at 1650 MHz = `aiv_vec_time(us)` (directly reported)

## 5. Comparing two routes

Run steps 3.1–3.3 for each `--route` (baseline + vmi-membar-vfoff), then:

```bash
.venv/bin/python3 - <<'PY'
import csv, glob

def read_pmu(route):
    base = f"build_output/msprof_qr_proj_seed_{route}"
    f = glob.glob(f"{base}/OPPROF_*/PipeUtilization.csv")[0]
    return list(csv.DictReader(open(f)))[0]

base = read_pmu("baseline")
vmi  = read_pmu("vmi-membar-vfoff")

print(f"{'metric':30s} {'baseline':>12s} {'vmi':>12s} {'delta':>12s}")
for k in ("aiv_vec_time(us)", "aiv_total_cycles", "aiv_scalar_time(us)",
          "aiv_mte3_time(us)"):
    b, v = float(base[k]), float(vmi[k])
    d = v - b
    print(f"{k:30s} {b:12.3f} {v:12.3f} {d:+12.3f} ({d/b*100:+.1f}%)")
PY
```

## 6. Verified example: `qr_proj_seed` (2026-08-12)

| metric | baseline | vmi-membar-vfoff | delta |
|---|---:|---:|---|
| Task Duration | 27.69 µs | 1.35 µs | -95.1% |
| **aiv_vec_time (VEC only)** | **4.07 µs** | **0.00 µs** | **-100%** |
| aiv_total_cycles | 43,664 | 424 | -99.0% |
| aiv_scalar_time | 0.96 µs | 0.23 µs | -75.7% |
| aiv_mte3_time (store) | 20.96 µs | 0.00 µs | -100% |
| aiv_vec_ratio | 15.4% | 0.0% | VEC eliminated |

VMI+membar route eliminated VEC compute entirely for this kernel — the work
moved to scalar pipe (`aiv_scalar_ratio` 3.6%→90.6%) and MTE3 stores were
fused away. This is the VF-fusion benefit visible only via PMU, invisible in
host-side wall-clock timing.

## 7. Limitations

- **Single-kernel**: `msprof op` profiles one kernel per run. For a sweep,
  loop over kernels manually, or integrate `--msprof` into `vpto_run.py`
  (not yet done — the wrapper-script approach above is the manual path).
- **Output dir permissions**: `msprof op` refuses world-writable dirs
  (e.g. `/tmp`). Use a repo-relative dir with `chmod 700`.
- **`msprof op` vs `msprof` (app mode)**: use `msprof op` (operator mode)
  for single-kernel PMU; `msprof` (app mode) wraps the whole binary and
  collects system-level data, less precise per-kernel.
- **`--aic-mode` not available in `msprof op`**: task-based is the default;
  do not pass `--aic-mode=task-based` (it errors).

## 8. Related

- `PRECISION_REPORT.md` §10 — VMI+membar route comparison (precision +
  stability); this doc adds the **performance** dimension.
- `incore-profiling` skill — the **simulator** path (`msprof op simulator`),
  no NPU needed but approximate. This doc is the **real-silicon** path.
- `BASELINE_COMPARISON.md` — earlier manual msprof runs (app mode, less
  precise); this doc supersedes it for per-kernel PMU.
- `ptoas/README-vmi-membar-bishengvfoff.md` — the ptoas-side route spec
  (VMI + membar + bisheng VF-off flags).
