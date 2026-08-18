# VPTO Route (Route 2) vs Baseline — vec_busy Comparison

Date: 2026-08-11
Baseline: `baselines/nofusion_a5_pmu2/` (op-fusion OFF, real-PMU
`pmu_idc_aic_vec_busy_o`, event 0x501, A5 @ 1650 MHz).
VPTO route: lab `.camodel-work/<case>/…/<case>_ordinary/` run on real A5
(device 2) via the VPTO→bisheng→CANN module-launch path, PMU captured with
`msprof --aic-metrics=PipeUtilization --aic-mode=task-based`.

**Same physical counter on both sides.** simpler's `pmu_idc_aic_vec_busy_o`
(event 0x501) and msprof's `aiv_vec_time(us)` derive from the same AIC vec-busy
hardware counter. msprof cycles = `aiv_vec_time(us)` × 1650.

**Correctness is out of scope** for this comparison (see MD #2). The VPTO
kernels' numerical output is NOT validated (`GOLDEN_MODE=skip`), and rms_norm is
known to return inf/NaN on real A5. vec_busy numbers below are therefore
**contaminated** — a kernel that computes garbage without faulting still
produces a vec_busy number. Read every comparison with that caveat.

---

## 1. Task-count alignment plan (same-口径 comparison)

The 1-task sweep (`results.txt`, recorded in `FINDINGS.md`) exercises each
kernel once. The baseline is multi-task (rmsnorm = 16 tasks). To put vec_busy
on the same basis, we loop the single `LaunchXxx(...)` call in `main.cpp` N
times (`for (int _ai = 0; _ai < N; ++_ai) { Launch…; }`) — the same 16-block
loop pattern used for the rms_norm alignment. This exercises the same
single-block kernel N times = N tasks, matching the baseline task count.

**Alignable set** (these kernels have a 1:1 mapping to a single baseline
func_id via the hc_pre name_map
`{0:hc_pre_rms, 1:hc_pre_seed, 2:hc_pre_linear, 3:split_pre_post,
4:comb_sinkhorn, 5:mix_x}`, plus decode_rmsnorm → rmsnorm func0):

| kernel (case) | baseline func | baseline tasks | baseline vec_busy | baseline total_cyc | 1-task vec_busy (predict ≈×N) |
|---|---|---|---|---|---|
| decode_rmsnorm | rmsnorm f0 | 16 | 471173 | 729191 | 34356 |
| hc_pre_rms | hc_pre f0 | 16 | 425925 | 674707 | 65374 |
| hc_pre_seed | hc_pre f1 | 1 | 1902 | 10395 | 300 |
| hc_pre_linear | hc_pre f2 | 32 | 0 | 892694 | 0 |
| split_pre_post | hc_pre f3 | 16 | 10660 | 54520 | 3004 |
| comb_sinkhorn | hc_pre f4 | 16 | 286271 | 341293 | 88393 |
| mix_x | hc_pre f5 | 112 | 302150 | 581944 | 12201 |

- `hc_pre_linear` (func2) is a cube kernel → vec_busy = 0 on both sides (it is
  a matmul; vec is idle). The baseline total_cyc is the meaningful number.
- `hc_pre_seed` is 1 task in the baseline, so no scaling is needed — the 1-task
  sweep number IS the aligned number (300 cyc vs baseline 1902 cyc).
- Runner: `run_aligned_one.sh` (wraps the Launch call, rebuilds, msprof,
  extracts, restores `main.cpp` from `.orig_backup`). **Not yet executed** —
  results table below to be filled after running.

### Aligned vec_busy results (to run)

| kernel | N | aligned vec_busy_cyc | baseline vec_busy | ratio (align/base) | aligned total_cyc | baseline total_cyc | note |
|---|---|---|---|---|---|---|---|
| decode_rmsnorm | 16 | _pending_ | 471173 | _pending_ | _pending_ | 729191 | |
| hc_pre_rms | 16 | _pending_ | 425925 | _pending_ | _pending_ | 674707 | |
| hc_pre_seed | 1 | _pending_ | 1902 | _pending_ | _pending_ | 10395 | no scaling (1 task) |
| hc_pre_linear | 32 | _pending_ (expect 0) | 0 | — | _pending_ | 892694 | cube kernel, vec idle |
| split_pre_post | 16 | _pending_ | 10660 | _pending_ | _pending_ | 54520 | |
| comb_sinkhorn | 16 | _pending_ | 286271 | _pending_ | _pending_ | 341293 | |
| mix_x | 112 | _pending_ | 302150 | _pending_ | _pending_ | 581944 | |

**How to run** (after the env is sourced):
```
cd /data/liuzidi/PTOAS/dsv4-vmi-lowering-lab/.vpto_pmu_align
R=aligned_results.txt
run_aligned_one.sh <decode_rmsnorm_dir> 16 $R
run_aligned_one.sh <hc_pre_rms_dir>      16 $R
run_aligned_one.sh <hc_pre_seed_dir>      1 $R
run_aligned_one.sh <hc_pre_linear_dir>   32 $R
run_aligned_one.sh <split_pre_post_dir>  16 $R
run_aligned_one.sh <comb_sinkhorn_dir>   16 $R
run_aligned_one.sh <mix_x_dir>          112 $R
```
where `<xxx_dir>` = `.camodel-work/<case>/ordinary/generated/ordinary/<case>_ordinary`.

---

## 2. Full 27-op rollup (decode-shape, 1 task) — NOT aligned

This is the raw 1-task sweep from `results.txt` / `aggregated_by_op.csv`,
recorded here for completeness. **Do not compare these vec_busy_cyc_sum numbers
to baseline vec_busy_sum directly** — they are per-kernel single-task snapshots
on a different work amount (see caveat 2 in FINDINGS.md). The aligned table in
§1 is the same-口径 comparison.

| baseline_op | baseline tasks | baseline vec_busy | VPTO 1-task vec_busy | ran_ok / total |
|---|---|---|---|---|
| rmsnorm | 16 | 471173 | 34356 | 1/1 |
| qkv_proj_rope | 154 | 3433762 | 35046 | 10/10 |
| mtp_projection | 945 | 5502419 | — | 0/5 (no lab coverage) |
| gate | 59 | 51333 | — | 0/6 (no lab coverage) |
| expert_shared | 57 | 36650 | — | 0/5 (no lab coverage) |
| expert_routed | 693 | 1743072 | — | 0/6 (no lab coverage) |
| hc_pre | 193 | 1026908 | 169272 | 6/7 |
| hc_post | 8 | 62521 | 567 | 1/3 |
| hc_head | 13 | 33926 | — | 0/5 (no lab coverage) |
| decode_attention_swa | 630 | 678526 | — | 0/2 (no lab coverage) |
| decode_attention_csa | 767 | 1832676 | 2941 | 3/3 |
| decode_attention_hca | 659 | 948846 | — | 0/3 (no lab coverage) |
| decode_sparse_attn | 659 | 948846 | 5511 | 3/9 (5 crash) |
| decode_sparse_attn_swa | 659 | 948846 | 0 | 1/9 (5 crash) |
| decode_sparse_attn_hca | 659 | 948846 | 0 | 1/9 (5 crash) |
| decode_compressor_ratio4 | 18 | 18564 | 800 | 2/3 (1 crash) |
| decode_compressor_ratio128 | 10 | 57509 | 800 | 2/3 (1 crash) |
| decode_indexer | 116 | 377340 | 2296 | 8/10 (2 crash) |
| decode_indexer_compressor | 12 | 20072 | 3745 | 4/5 (1 crash) |
| prefill_attention_swa | 1425 | 14752015 | — | 0/4 (no lab coverage) |
| prefill_attention_csa | 2650 | 18914799 | 10704 | 3/3 |
| prefill_attention_hca | 1579 | 15089681 | — | 0/2 (no lab coverage) |
| prefill_sparse_attn | 1579 | 15089681 | 133628 | 4/10 (5 crash) |
| prefill_compressor_ratio4 | 229 | 859370 | 22453 | 4/6 (2 crash) |
| prefill_compressor_ratio128 | 156 | 340879 | — | 0/7 (no lab coverage) |
| prefill_indexer | 942 | 3208094 | 773 | 2/9 (6 crash) |
| prefill_indexer_compressor | 653 | 714578 | 16961 | 5/8 (2 crash) |

### Coverage summary
- **17 / 27 baseline ops (63%) have at least 1 kernel with VPTO PMU data.**
- **10 ops have zero lab coverage** — their inner kernels were never prepared
  in `.camodel-work/`: mtp_projection, gate, expert_shared, expert_routed,
  hc_head, decode_attention_swa, decode_attention_hca, prefill_attention_swa,
  prefill_attention_hca, prefill_compressor_ratio128.
- **20 inner kernels crashed at runtime** (device AICore exception) — see
  MD #2 for the errcode breakdown.

---

## 3. Caveats

1. **Contamination.** `GOLDEN_MODE=skip` was used throughout. rms_norm VPTO is
   proven to return inf/NaN on real A5 with real inputs (see MD #2 §2). The
   57 "OK" kernels executed and produced PMU, but their output is unverified —
   some likely also compute garbage without faulting. A garbage-output kernel
   still has a (meaningless) vec_busy number. Treat all VPTO vec_busy here as
   **upper-bound / contaminated** until precision is validated.
2. **Loop-N alignment ≠ baseline work distribution.** The loop wraps one
   single-block kernel N times to match the baseline task count. This matches
   the *task count* but not necessarily the baseline's multi-block work
   distribution (block_idx-dependent shapes, tiling, tail blocks). It is the
   same-口径 on task count, which is what the baseline PMU CSV sums over.
3. **hc_pre_linear is a cube kernel** — vec_busy = 0 by construction on both
   sides; compare total_cyc instead.
4. **11 ops have no lab coverage** so no VPTO number exists for them at all;
   they are blank in §2 and excluded from §1.

## 4. Files

- Baseline CSVs: `baselines/nofusion_a5_pmu2/*.pmu.csv` + `baseline_summary.csv`
- VPTO 1-task raw: `.vpto_pmu_run/results.txt`, `aggregated_by_op.csv`,
  `FINDINGS.md`
- VPTO aligned (pending): `.vpto_pmu_align/aligned_results.txt`,
  `run_aligned_one.sh`, `logs/<case>.{build,msprof}.log`,
  `msprof/<case>/op_summary_*.csv`
