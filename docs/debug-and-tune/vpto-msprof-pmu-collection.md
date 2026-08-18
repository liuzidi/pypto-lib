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

## 6. `qr_proj_seed` case study — and why you must verify (2026-08-12)

> **This section documents a false positive.** The initial msprof run
> appeared to show VMI eliminating VEC compute (`aiv_vec_time` 4.07 µs →
> 0.00 µs). Binary-level verification proved the kernel machine code is
> identical between routes — the msprof numbers were a sampling artifact.

### 6.1 The false positive

First msprof run (both with `--warm-up=0`):

| metric | baseline | vmi-membar-vfoff | appeared to show |
|---|---:|---:|---|
| Task Duration | 27.69 µs | 1.35 µs | -95% |
| aiv_vec_time | 4.07 µs | 0.00 µs | VEC eliminated |
| aiv_total_cycles | 43,664 | 424 | -99% |

### 6.2 The verification that caught the error

**Step 1 — compare device kernel binary across routes:**

```bash
# Extract nested AICore ELF from each fatobj, compare .text section
.venv/bin/python3 - <<'PY'
import re, hashlib
for route, path in [("baseline", "build_output/vpto_qr_proj_seed/qr_proj_seed.o"),
                    ("vmi", "build_output/vpto_qr_proj_seed_vmi/qr_proj_seed.o")]:
    data = open(path, "rb").read()
    offsets = [m.start() for m in re.finditer(b"\x7fELF", data)]
    elf = data[offsets[1]:]  # ELF[1] = AICore kernel
    # .text is at offset 0x100, size from section header
    text = elf[0x100:0x100+552]
    print(f"{route}: .text md5={hashlib.md5(text).hexdigest()}")
PY
# Result: both md5 = d4b905a3f73e  ← IDENTICAL
```

Also confirmed: bisheng-linked `.so` md5 identical; msprof-dumped
`aicore_binary.o` (device-executed binary) md5 identical across routes.

**Step 2 — run msprof multiple times with warm-up:**

| run | warm-up | freq (MHz) | aiv_vec_time | aiv_total_cycles |
|---|---:|---:|---:|---:|
| baseline r1 | 3 | 1650 | 4.07 µs | 43714 |
| baseline r2 | 3 | **875** | 7.68 µs | 38852 |
| baseline orig | 0 | 1650 | 4.07 µs | 43664 |
| vmi orig | 0 | 1650 | **0.00 µs** | **424** |
| vmi r1 | 0 | **875** | **0.00 µs** | **661** |

### 6.3 Root cause of the false positive

Two problems compounded:

1. **`--warm-up=0` caused msprof to sample a launch placeholder task**
   instead of the real kernel execution. The VMI runs reported
   `aiv_total_cycles = 424/661` — impossible for a 552-byte `.text`
   (baseline shows ~43700 cycles). The sampling window truncated.

2. **NPU frequency instability**: some runs hit 875 MHz instead of 1650
   (nearly 2× slower). `Current Freq` in `OpBasicInfo.csv` must be
   checked before comparing absolute cycle/time numbers.

### 6.4 Lessons (apply to every msprof run)

- **Always set `--warm-up≥3`** so the device reaches steady state.
- **Run at least 3 times** and take the median (or check variance).
- **Check `Current Freq`** in `OpBasicInfo.csv` — if it's below `Rated
  Freq`, the run hit thermal throttling or power capping. Discard or
  normalize.
- **Sanity-check `aiv_total_cycles`** against the `.text` section size
  — a 552-byte kernel reporting <1000 cycles almost certainly wasn't
  sampled correctly.
- **Verify codegen difference before trusting PMU difference**: if
  the device binary (msprof dump `aicore_binary.o`) is byte-identical
  across routes, any PMU difference is sampling noise, not codegen.

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
- **Sampling artifacts on short kernels**: `--warm-up=0` can produce
  `aiv_vec_time=0` and `aiv_total_cycles` orders of magnitude too low for
  sub-10 µs kernels. See §6 for a documented case. Always use
  `--warm-up≥3`, run multiple times, check `Current Freq`, and sanity-check
  cycle counts.
- **Frequency instability**: the A5 can drop from 1650 MHz to 875 MHz
  between runs (thermal/power). Compare `Current Freq` in
  `OpBasicInfo.csv`; normalize or discard throttled runs.

## 8. Related

- `PRECISION_REPORT.md` §10.7 — the full verification of the
  `qr_proj_seed` false positive (binary comparison + multi-run PMU data).
- `incore-profiling` skill — the **simulator** path (`msprof op simulator`),
  no NPU needed but approximate. This doc is the **real-silicon** path.
- `BASELINE_COMPARISON.md` — earlier manual msprof runs (app mode, less
  precise); this doc supersedes it for per-kernel PMU.
- `ptoas/README-vmi-membar-bishengvfoff.md` — the ptoas-side route spec
  (VMI + membar + bisheng VF-off flags).
