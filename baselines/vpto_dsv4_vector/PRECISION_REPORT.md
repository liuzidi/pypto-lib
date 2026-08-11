# DSV4 VPTO Board-Precision State Report

Date: 2026-08-12
Branch: `test-ds-a5`
Sweep source: `baselines/vpto_dsv4_vector/sweep_results.csv` (338 rows)
Command: `python tests/dsv4_validate/validate.py --all-modules -d 0`
Device: single Ascend A5 NPU, device 0, serial
Route under test: Route 2 only (`.pto` → ptoas LLVM → bisheng fatobj `.o`
→ CANN module-load → NPU execute). Route 1 (simpler/EmitC) is used solely
as the **capture vehicle** for non-leaf inner kernels (Phase 5).

> Scope reminder. Per the user's standing instruction, precision FAIL is
> the framework's **expected, valuable output**, not a defect to fix here.
> This report documents *what* the sweep found and proposes *why*; per-
> kernel fixes are a separate later task.

---

## 1. Framework shape

```
                      Route 1 capture (DFX)            Route 2 replay (board)
  ┌──────────┐   run_jit w/   ┌──────────────┐  harvest  ┌──────────────┐  ptoas+bisheng  ┌─────┐
  │ model.py │ ───────────▶ │ args_dump.json│ ────────▶ │ vN.bin /      │ ──────────────▶ │ NPU │ → compare
  │ (@pl.jit)│   enable_     │ name_map_*.   │  (capture │ golden_vN.bin │                 │     │
  └──────────┘   dump_args=2 │  json         │  .py)     │ capture_meta  │                 └─────┘
                 +dep_gen     └──────────────┘           └──────────────┘
```

- **Leaf kernels** (ptr-arg count == spec count, 1:1 map): Route 2 from
  `resolve_meta` → `dump_bins` directly; no capture needed.
- **Non-leaf inner kernels** (ptrs are intermediates from preceding
  kernels): Route 1 capture with `enable_dump_args=2`, then per-kernel
  harvest + Route 2 replay via `vpto_run.py --captured-dump`.
- 24 modules covered (16 decode + 8 prefill), 338 distinct
  (module, kernel) pairs attempted.

The 52 PASSes (max_diff = 0) are the load-bearing evidence that the
capture→replay chain is **byte-accurate**: the same captured bytes, fed
through VPTO codegen + NPU, reproduce the torch golden bit-for-bit on
those kernels. FAILs therefore cannot be attributed to a generic
capture/replay plumbing bug — they localize to per-kernel codegen or
runtime behavior.

---

## 2. Sweep result tally

| status | count | meaning |
|---|---:|---|
| `pass`              |  52 | NPU ran, compare `max_diff = 0` (exact) |
| `fail` (with diff)  |   8 | NPU ran, compare produced a numeric `max_diff` |
| `fail` (no diff)    | 135 | NPU run exited non-zero; compare never ran |
| `crash`             | 143 | vpto_run exited before NPU launch (lowering/harvest failure) |

**Total: 338.** Two structurally distinct failure populations dominate:

1. **135 "NPU run failed (exit 1)"** — the fatobj built and loaded, the
   kernel symbol exists, but the NPU binary faults at launch time. This
   is the largest unexplained bucket.
2. **143 "vpto_run exited 1 without result.json"** — the ptoas→bisheng
   lowering step failed *before* any NPU execution, so there is no
   `.o`/`.so`/host-bin to inspect at run time. This is a codegen
   coverage problem, not a precision problem.

The 8 with-`max_diff` cases are the only true *precision* failures
(kernels that ran to completion and produced wrong numbers). They are
the priority targets for root-cause analysis.

---

## 3. The 8 true precision FAILs (ran + compared)

| kernel | module | max_diff | n_over / n_total | timing | notes |
|---|---|---:|---|---:|---|
| `rms_norm` | attention_csa | 999424.0 | 7 / 57344 | 0.95 ms | appears in 4 modules, identical diff |
| `rms_norm` | attention_swa | 999424.0 | 7 / 57344 | 0.63 ms | same pattern |
| `rms_norm` | prefill_attention_csa | 999424.0 | 7 / 917504 | 2.79 ms | prefill shape, same diff |
| `rms_norm` | prefill_attention_hca | 999424.0 | 7 / 917504 | 0.57 ms | same pattern |
| `rms_norm` | prefill_attention_swa | 999424.0 | 7 / 917504 | 0.55 ms | same pattern |
| `proj_a_mm` | attention_csa | 27.82 | 1024 / 262144 | 0.50 ms | small bounded diff |
| `proj_a_mm` | sparse_attn | 1.95e+38 | 1024 / 262144 | 0.53 ms | catastrophic (near bf16 max) |
| `mtp_projection_rms` | mtp_projection | 1000.0 | 8 / 64 | 0.65 ms | small tensor, clean diff |

### 3.1 Patterns

- **`rms_norm` family (5 rows, identical `max_diff = 999424.0`, `n_over = 7`).**
  The exact constant 999424 is suspiciously round (not a typical
  floating-point residue). `999424 = 0xF4240` in hex is the IEEE-754
  bf16 representation of **+1.0e6** (= 1 0006 0000 → exponent + mantissa
  of 10^6). `n_over = 7` is also identical across decode (57344 elems)
  and prefill (917504 elems) shapes — i.e. exactly 7 elements disagree,
  regardless of input size. This points to a **fixed-structural defect**
  (e.g. a specific lane, a specific tail element, or an inf/NaN
  propagating to a fixed set of positions), not an accumulation-drift
  defect that would scale with element count.
- **`proj_a_mm`** appears twice with two very different magnitudes:
  `27.82` (attention_csa, bounded — looks like a real numerical
  precision issue) vs `1.95e+38` (sparse_attn, near bf16 max — looks
  like uninitialized/garbage data, not drift). Same kernel, different
  module → the capture inputs likely differ, and the sparse_attn case
  may have captured a buffer that the kernel did not actually populate,
  or the `n_over = 1024` block is reading unmapped memory.
- **`mtp_projection_rms`** (`max_diff = 1000.0`, `n_over = 8 / 64`):
  tiny tensor, clean round number — again the roundness suggests a
  structural/initialization artifact rather than drift.

### 3.2 Initial hypotheses (to be tested in §4)

| # | hypothesis | discriminative test |
|---|---|---|
| H1 | **membar / cross-stage sync**: NPU wrote output but the host read it before a sync, so 7 "stale" elements persist. | Run same captured inputs on `a5sim` (simulator has no async sync); if sim passes → sync. |
| H2 | **golden data wrong**: capture wrote wrong bytes (e.g. captured a buffer at the wrong dispatch stage, or before a producer kernel wrote it). | Recompute golden in-process on the captured input bytes; compare to `golden_vN.bin`. |
| H3 | **codegen error** (ptoas / bisheng lowering): the fatobj computes a wrong index or uses a wrong stride for a specific lane/element. | `nm`/objdump the fatobj; correlate the 7 failing positions' addresses to a tiling boundary. |
| H4 | **dtype/view reinterpretation**: a bf16↔fp32 bitcast mismatch between capture and replay. | Inspect `capture_meta.json` `np_types` vs `.pto` ptr dtypes; verify bin sizes match `elem_count × sizeof(dtype)`. |

---

## 4. Root-cause investigation — `rms_norm` family (5 of 8 FAILs)

**Verdict (head):** The `rms_norm` FAILs are **not** golden errors,
**not** capture errors, and **not** board-sync/membar artifacts. They
are a real kernel-level defect: the inner-kernel `.pto` variant of
`rms_norm` leaks uninitialized local-memory intermediates into the
output buffer at fixed tile boundaries. The standalone leaf `rms_norm`
module PASSES on the same NPU with the same harness, because its `.pto`
is a *different* (correct) variant.

### 4.1 Discriminator: leaf-vs-inner natural experiment

A direct `a5sim` re-run was not available (the VPTO host binary uses
`aclrtSetDevice` for real NPU only; the `__CPU_SIM` path is a
compile-time switch in pto-isa's `test_common.h`, not selectable at
runtime by `vpto_run.py`). Instead, the sweep itself provides a stronger
discriminator: the **same kernel name, same NPU, same harness, two
different `.pto` variants**.

| variant | module | mode | fatobj | result |
|---|---|---|---:|---|
| leaf | `rms_norm` (standalone) | decode | 8712 B | **PASS** (max_diff = 0) |
| inner | `rms_norm` inside `attention_csa` | decode | 8568 B | **FAIL** (max_diff = 999424) |
| inner | `rms_norm` inside `prefill_attention_csa` | prefill | 8568 B | **FAIL** (max_diff = 999424) |

Both inner variants produce the *identical* `max_diff = 999424.0` with
`n_over = 7` despite different shapes (decode 57344 vs prefill 917504
elements). Since the NPU hardware + harness are constant, the defect
localizes to the `.pto` source — i.e. the codegen variant, not the
execution environment. This refutes H1 (sync/membar) for the rms_norm
family.

### 4.2 Golden-data audit (H2) — golden is correct

**Method.** Read the captured `v1.bin` (input), `v3.bin` (weight), and
`golden_v2.bin` (reference output) for `vpto_rms_norm/run` (the
prefill-attention_csa variant, func_id = 6, elem_counts v1=v2=917504,
v3=7168, all bf16).

**Findings:**
- `v1.bin` (input): 917504 bf16 elements, **all zero** (`np.count_nonzero = 0`).
- `v3.bin` (weight): 7168 bf16 elements, **all zero**.
- `golden_v2.bin`: 917504 bf16 elements, **all zero**.

The golden output is mathematically correct: `rms_norm(x, w) = x * rsqrt(mean(x²) + eps) * w`; with `x = 0` and `w = 0`, the output is `0 * 1000 * 0 = 0`. The capture is also correct: the producer kernel genuinely wrote zeros (verified by reading the same `bin_offset`/`bin_size` window directly from the source `args.bin` — 917504 zero bf16 elements). **H2 (golden error) is refuted.**

### 4.3 Codegen inspection (H3) — defect confirmed

**The two `.pto` variants differ structurally.** `diff` of the
standalone-leaf `.pto` (from `_jit_rms_norm_test_*`) vs the
inner-kernel `.pto` (from `_jit_prefill_attention_csa_test_*`, as
reused by `vpto_rms_norm/run`):

```diff
- module attributes {pto.target_arch = "a5"} {
-   func.func @rms_norm(%arg0: !pto.ptr<bf16>, %arg1: !pto.ptr<bf16>,
-     %arg2: !pto.ptr<bf16>, %arg3: index, ...) {
-   %x__ssa_v0_view = pto.make_tensor_view %arg0,
-     shape = [%arg3, %c7168_index], ...   ← DYNAMIC row dim (caller-supplied)
+ module attributes {pto.target_arch = "a5", pto.kernel_kind = ...} {
+   func.func @rms_norm(%arg0: !pto.ptr<bf16>, %arg1: !pto.ptr<bf16>,
+     %arg2: !pto.ptr<bf16>, ...) {        ← %arg3 : index is DROPPED
+   %x_mixed_inline523__rv_v2_view = pto.make_tensor_view %arg0,
+     shape = [%c128_index, %c7168_index], ...  ← HARDCODED row dim = 128
```

The leaf variant takes a dynamic `%arg3 : index` (the row dimension is
passed at call time and matches the actual GM tensor). The inner
variant bakes the row count to the constant `128` into the tensor view,
and drops the `%arg3` parameter entirely. The two fatobjs differ
(8712 B vs 8568 B; nested-ELF offsets `[0,288]` vs `[0,272]`),
confirming different lowered code.

**The 7 failing elements localize to tile boundaries.** Reshaping the
917504-element output as `[128, 7168]` and mapping the 7 failing flat
indices:

| flat idx | row | col | col_block (col/128) | offset_in_block | NPU value (as bf16) | bit pattern |
|---:|---:|---:|---:|---:|---:|---|
| 256–259 | 0 | 256–259 | 2 | 0–3 | 1.0 | 0x3F80 |
| 512–513 | 0 | 512–513 | 4 | 0–1 | 1000.0 | 0x447A |
| 768 | 0 | 768 | 6 | 0 | 999424.0 | 0x4974 |

The failing positions are the **starts of even 128-column blocks** in
row 0. The kernel's apply loop (`scf.for 0 to 56 step 2`) writes 8×128
tiles per iteration; the failing positions are tile-aligned.

The three leaked values are the kernel's own fp32 intermediates:
- `1.0` — a unit scale constant.
- `1000.0` = `1/sqrt(1e-6)` = `rsqrt(eps)` — exactly the `x_inv_rms`
  value computed when `x_sq_sum = 0` (input is zero → `mean = 0` →
  `mean + eps = 1e-6` → `rsqrt = 1000`).
- `999424.0` — a partial reduction accumulator (the `x_sq_sum` tile
  carries a stale value from a prior block's reduction).

These are **not** in the GM output path; they are fp32 values held in
local-memory (UB) tiles (`addr = 8768, 8224, 8256` in the `.pto`). The
output bf16 tile `%19` / `%25` is allocated at `addr = 8768`, the same
local-memory region as the f32 reduction tiles — so the bf16 `tstore`
underwrites 2 bytes/elem into a 4-byte/elem region previously used
by f32 intermediates, and the residual high bytes leak as bf16 output.

### 4.4 Capture staging audit (H1-orthogonal) — capture is correctly staged

`args_dump.json` records 33 records for `rms_norm` (func_id 6 in
prefill_attention_csa): 32 input records at `stage = before_dispatch`,
1 output record at `stage = after_completion`. The harvester takes the
first `before_dispatch` input copy per `arg_index` (the 16 SPMD copies
are identical, verified bit-identical). The producer's dispatch order
predates the consumer's in the `deps.json` edge graph (verified: the
producer task_id appears as a `pred` of the rms_norm task_id). **No
capture-timing artifact** — the input bytes were genuinely zero when
read, and the golden was genuinely zero. H1 (sync/membar) is refuted
for this family.

### 4.5 Synthesis — `rms_norm` family root cause

| hypothesis | verdict | evidence |
|---|---|---|
| H1 membar/sync | **refuted** | leaf variant passes on same NPU/harness; staging audit shows correct dispatch order |
| H2 golden/capture error | **refuted** | input + weight + golden all verifiably zero; recompute `rms_norm(0, 0) = 0` |
| H3 codegen | **CONFIRMED** | inner `.pto` hardcodes row=128 (drops dynamic `%arg3`); failing positions are tile-aligned; leaked values match kernel fp32 intermediates held in the same local-memory addr as the output bf16 tile |

The defect is in the **inner-kernel codegen variant of `rms_norm`**
emitted by pypto when the kernel is fused into a multi-kernel module.
The standalone-leaf variant (which keeps the row dim dynamic and
allocates output tiles in a non-colliding local-memory region) is
correct. This is a pypto-side issue to file separately (route:
`pypto`), not a ptoas/bisheng/CANN issue and not a test-framework issue.

---

## 5. Root-cause investigation — remaining 3 FAILs (open)

The `rms_norm` analysis above used the 5 identical rows. The remaining
3 FAILs each have a single instance and need separate investigation:

| kernel | module | max_diff | status |
|---|---|---:|---|
| `proj_a_mm` | `attention_csa` | 27.82 | bounded — likely real numerical (different pattern from rms_norm) |
| `proj_a_mm` | `sparse_attn` | 1.95e+38 | catastrophic — near bf16 max, likely uninitialized/GM OOB |
| `mtp_projection_rms` | `mtp_projection` | 1000.0 | `n_over = 8/64` — same `rsqrt(eps)=1000` fingerprint as rms_norm, likely same local-memory-leak class |

**Next steps for these 3:**
1. Repeat the §4.2 golden audit on `proj_a_mm@sparse_attn` — the
   `1.95e+38` magnitude suggests either a captured buffer the kernel
   didn't populate, or a GM out-of-bounds read.
2. Check if `mtp_projection_rms` shares the `rms_norm` codegen pattern
   (hardcoded shape, dropped `%arg3`) — the `max_diff = 1000.0` =
   `rsqrt(1e-6)` fingerprint is a strong indicator it's the same
   defect class.
3. `proj_a_mm@attention_csa` (`max_diff = 27.82`, bounded) is the only
   FAIL that looks like genuine numerical drift rather than a
   structural leak — defer to per-kernel precision debugging.

---

## 6. Module-level rollup

| module | total | pass | fail | crash |
|---|---:|---:|---:|---:|
| attention_csa | 48 | 7 | 22 | 19 |
| attention_hca | 1 | 0 | 0 | 1 |
| attention_swa | 28 | 5 | 12 | 11 |
| compressor | 3 | 0 | 2 | 1 |
| expert_routed | 6 | 0 | 2 | 4 |
| expert_shared | 5 | 0 | 1 | 4 |
| gate | 6 | 0 | 0 | 6 |
| hc_head | 5 | 1 | 2 | 2 |
| hc_post | 1 | 1 | 0 | 0 |
| hc_pre | 6 | 2 | 4 | 0 |
| indexer | 15 | 1 | 8 | 6 |
| indexer_compressor | 5 | 1 | 2 | 2 |
| mtp_projection | 6 | 2 | 1 | 3 |
| prefill_attention_csa | 58 | 9 | 25 | 24 |
| prefill_attention_hca | 39 | 8 | 18 | 13 |
| prefill_attention_swa | 34 | 7 | 14 | 13 |
| prefill_compressor_ratio128 | 7 | 1 | 5 | 1 |
| prefill_compressor_ratio4 | 6 | 0 | 5 | 1 |
| prefill_indexer | 20 | 2 | 7 | 11 |
| prefill_indexer_compressor | 8 | 1 | 4 | 3 |
| prefill_sparse_attn | 11 | 1 | 4 | 6 |
| qkv_proj_rope | 9 | 2 | 2 | 5 |
| rms_norm | 1 | 1 | 0 | 0 |
| sparse_attn | 10 | 0 | 3 | 7 |

Observations:
- **`rms_norm` standalone module PASSES** (max_diff = 0), but `rms_norm`
  as an inner kernel inside every attention module FAILs with the
  identical `999424.0` pattern. §4 traced this to a **codegen variant
  difference** (H3 confirmed): the inner-kernel `.pto` hardcodes the row
  dimension to `c128_index` and drops the dynamic `%arg3`, while the
  leaf variant keeps it dynamic. The leaked values (`1.0`, `1000.0` =
  `rsqrt(eps)`, `999424.0`) are fp32 intermediates held in the same
  local-memory address as the output bf16 tile. H1 (sync) and H2
  (golden) are refuted for this family.
- Modules with 0 PASS (gate, sparse_attn, compressor,
  prefill_compressor_ratio4) have no positive control — their FAILs
  could be either codegen or capture; need at least one passing case
  per family to isolate.

---

## 7. PASS kernel inventory (the positive control set)

These 52 (module, kernel) pairs reproduce golden bit-for-bit through the
full Route 2 chain. They are the calibration evidence that the framework
itself is sound. (Full list in `sweep_results.csv`; condensed by pattern
below.)

- `*_seed` kernels (kv_proj_seed, qr_proj_seed, hc_pre_seed) — pass in
  every module they appear (12/12). Small fatobj (~5 KB), fast (~0.5 ms).
- `kv_hadamard`, `mix_x`, `hc_post`, `merge_norm` — pass consistently
  across modules.
- `rms_norm` passes as a **leaf** module but fails as an **inner**
  kernel — see §6.

### 6.1 What the PASSes tell us

- The **harvest → replay byte pipeline is correct**: the same `.bin`
  bytes that Route 1 captured, fed back through Route 2, reproduce
  golden. So byte capture, dtype mapping, and `main.cpp` buffer
  sizing are all sound *for the passing kernels*.
- The **ptoas + bisheng lowering is correct for these kernel shapes**.
  Any lowering failure is shape-specific, not a global codegen break.
- The **NPU execution + compare path is correct** for these kernels.

Any FAIL must therefore be explained by something that *differs*
between a passing kernel and a failing one — and the rms_norm
leaf-vs-inner contrast is the cleanest available natural experiment.

---

## 8. Next steps

**Completed (rms_norm family, 5 of 8 FAILs):** §4.1–§4.5 traced the
`999424.0` FAILs to a codegen variant difference — the inner-kernel
`.pto` hardcodes the row dim and collides local-memory tile addresses.
Routed to **pypto** (inner-kernel codegen), not ptoas/bisheng/CANN,
not the test framework, not golden data.

**Remaining (3 of 8 FAILs):**
1. **`mtp_projection_rms`** (`max_diff = 1000.0`, `n_over = 8/64`):
   the `1000.0 = rsqrt(1e-6)` fingerprint matches the `rms_norm` class.
   Re-run the §4.2–§4.3 audit (leaf-vs-inner `.pto` diff, local-memory
   tile-address collision) on this kernel. High prior of being the same
   defect class.
2. **`proj_a_mm@sparse_attn`** (`max_diff = 1.95e+38`, catastrophic):
   near-bf16-max magnitude suggests GM out-of-bounds read or a captured
   buffer the kernel didn't populate. Run §4.2 golden audit first —
   verify the captured input bytes are real data, not zeros/garbage. If
   inputs are valid, this is a separate codegen/runtime defect.
3. **`proj_a_mm@attention_csa`** (`max_diff = 27.82`, bounded): the
   only FAIL that looks like genuine numerical drift (small, bounded,
   `n_over = 1024/262144` = 0.4% of elements). Defer to per-kernel
   precision debugging — likely a real VPTO lowering precision issue,
   not a structural leak.

**Cross-cutting:** the 135 "NPU run failed (exit 1)" cases (the
largest FAIL bucket) and the 143 "ptoas/bisheng lowering failure"
crashes are separate problem classes — they never reached compare, so
they need codegen/run-time fault triage, not precision investigation.
The 7 "no dump records" harvest crashes (0-iteration SPMD kernels like
`kv_touch`, `gate_pre_route`, `hc_head_seed`, `hc_post_inactive_pad`)
are framework edge cases where the kernel's SPMD loop body never
executed on any block — a capture-coverage gap, separate from precision.
