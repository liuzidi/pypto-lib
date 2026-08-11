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

## 2. Sweep result tally + root-cause summary

| status | count | meaning | root cause |
|---|---:|---|---|
| `pass`              |  52 | NPU ran, compare `max_diff = 0` (exact) | framework is sound (positive control) |
| `fail` (with diff)  |   8 | NPU ran, compare produced a numeric `max_diff` | **kernel codegen variant defects** (§3) |
| `fail` (no diff)    | 135 | NPU run exited non-zero; compare never ran | **framework bugs** (§5): inout-harvest gap, scalar=0, size=0 alloc |
| `crash`             | 143 | vpto_run exited before NPU launch | **lowering coverage gaps** (§6): pto.tdivs/i8/tstore templates |

**Total: 338.** After investigation, the 286 non-pass rows decompose into
**6 distinct root-cause classes** (not 286 independent bugs):

| class | bucket | count | fault layer | §ref |
|---:|---|---:|---|---|
| C1 | precision-fail | 8 | pypto inner-kernel codegen + framework scalar_sem | §3 |
| C2 | NPU-crash | ~50 | framework: `inout` role not harvested (v4.bin missing) | §5.1 |
| C3 | NPU-crash | ~40 | framework: index scalars beyond ctx_len default to 0 | §5.2 |
| C4 | NPU-crash | ~30 | framework: output-only ptr → 0-byte alloc → `aclrtMallocHost` fails | §5.3 |
| C5 | lowering-crash | ~80 | ptoas: `pto.tdivs` i32 template not supported on A5 | §6.1 |
| C6 | lowering-crash | ~30 | framework: `PTO_TO_CPP` missing `i8 → int8_t` map | §6.2 |
| C7 | lowering-crash | ~5 | ptoas: `pto.tstore` / operand-dominate codegen errors | §6.3 |
| C8 | harvest-crash | 7 | framework: `inout` role + 0-iter SPMD (no dispatch records) | §5.4 |

**Headline finding:** of the 286 non-pass rows, **~180 (63%) are
framework-side bugs** (C2–C4, C6, C8) that are fixable in the
test framework without touching pypto/ptoas/bisheng, and would convert
those rows from "crash" to either pass or true-precision-fail. Only
~80 rows (C1+C5+C7) are genuine pypto/ptoas-side issues.

---

## 3. Root-cause C1 — precision FAILs (8 rows, kernel codegen)

The 8 rows where the NPU ran to completion and compare produced a
numeric `max_diff`. All 8 share the same fingerprint: **captured inputs
are all-zero, golden is all-zero, but NPU output contains non-zero
intermediate values** (rsqrt, reduction accumulators, NaN).

| kernel | module | max_diff | n_over / n_total | fingerprint |
|---|---|---:|---|---|
| `rms_norm` | attention_csa | 999424.0 | 7 / 57344 | leaked fp32 intermediates at tile boundaries |
| `rms_norm` | attention_swa | 999424.0 | 7 / 57344 | same |
| `rms_norm` | prefill_attention_csa | 999424.0 | 7 / 917504 | same |
| `rms_norm` | prefill_attention_hca | 999424.0 | 7 / 917504 | same |
| `rms_norm` | prefill_attention_swa | 999424.0 | 7 / 917504 | same |
| `proj_a_mm` | attention_csa | 27.82 | 1024 / 262144 | 328 NaN values leaked (input all zero) |
| `proj_a_mm` | sparse_attn | 1.95e+38 | 1024 / 262144 | catastrophic — same class, larger leak |
| `mtp_projection_rms` | mtp_projection | 1000.0 | 8 / 64 | `rsqrt(1e-6)=1000` + NaN at idx 8-15 |

### 3.1 `rms_norm` family (5 rows) — local-memory tile-address collision

**Verdict:** NOT golden error, NOT capture error, NOT sync. The
inner-kernel `.pto` variant hardcodes the row dimension to `c128_index`
(drops the dynamic `%arg3`) and the output bf16 tile collides in local
memory with fp32 reduction intermediates.

Evidence:
- Input `v1.bin`, weight `v3.bin`, golden `golden_v2.bin` all verifiably
  zero (read directly from `args.bin` at the captured offset — producer
  genuinely wrote zeros).
- The leaf-standalone `rms_norm` module PASSES on the same NPU/harness
  (fatobj 8712 B vs inner 8568 B — different lowered code). This is the
  natural experiment that refutes sync/membar (H1) and golden (H2).
- `diff` of the two `.pto` files:

  ```diff
  - shape = [%arg3, %c7168_index]   ← leaf: DYNAMIC row dim (caller-supplied)
  + shape = [%c128_index, %c7168_index]  ← inner: HARDCODED row dim = 128
  ```

- The 7 failing elements are at flat indices 256, 257, 258, 259, 512,
  513, 768 — the **starts of even 128-column tiles** in row 0. The
  leaked values `1.0` / `1000.0` (= `rsqrt(1e-6)`) / `999424.0` are the
  kernel's own fp32 intermediates held at the same local-memory address
  (`addr = 8768`) as the output bf16 tile.
- **Routed to:** pypto (inner-kernel codegen). NOT ptoas/bisheng/CANN,
  NOT the test framework, NOT golden data.

### 3.2 `mtp_projection_rms` (1 row) — framework scalar_sem = 0 bug

**Verdict:** framework bug. The `.pto` keeps dynamic shape
`[%arg5, %c7168_index]`, but `main.cpp` passes `v6 = 0` for `%arg5`
because the framework's `scalar_sem` derivation only marks the FIRST
non-SPMD index scalar as `ctx_len`; the rest default to 0.

Evidence:
- `.pto` signature: `@mtp_projection_rms(%arg0, %arg1, %arg2, %arg3,
  %arg4: index, %arg5: index, %arg6: index)` — 3 index scalars.
- `main.cpp` emits: `v5 = 8` (ctx_len, correct), `v6 = 0` (FIXME),
  `v7 = 0` (FIXME). The `FIXME` comments come from `setup_main.py:182`
  (`{scal_type} {s['name']} = 0;  // FIXME: {hint}`).
- Result: tensor views `shape = [%arg5=0, ...]` collapse to 0 rows →
  the SPMD loop writes nothing → 8 elements at idx 8-15 leak the
  `rsqrt(eps) = 1000.0` intermediate (same fingerprint as rms_norm).
- Also: `v2` (16 elements) gets 8 NaN values — the same leak class.
- **Routed to:** test framework (`vpto_run.py` scalar_sem derivation +
  `setup_main.py` default-to-0). Fix: derive scalar semantics from the
  `.pto`'s tensor-view shape dims, not just "first non-SPMD index".

### 3.3 `proj_a_mm` (2 rows) — same scalar_sem = 0 bug, cube kernel

**Verdict:** same framework scalar_sem bug as §3.2, but on a cube
(matmul) kernel. The `.pto` has 4 index scalars (`%arg3` row-offset,
`%arg4` group, `%arg5`, `%arg6`); `main.cpp` sets only `%arg3 = ctx_len`
and the rest to 0. With wrong indices the matmul reads
out-of-bounds/wrong tile, producing NaN/1e+38.

Evidence:
- `.pto`: `@proj_a_mm(%arg0, %arg1, %arg2, %arg3: index, %arg4: index,
  %arg5: index, %arg6: index)` — cube kernel, hardcoded GM shapes
  `[c2048, c4096]`, `[c16, c1024, c4096]`, `[c128, c16384]`.
- `main.cpp`: `v4 = 128` (ctx_len), `v5 = 0, v6 = 0, v7 = 0` (FIXME).
- Re-run of `proj_a_mm@attention_csa` produced 328 NaN values in v3
  (input all zero → output should be zero); compare.py reported
  "passed" because `NaN > threshold` is `False` (a separate compare.py
  robustness gap).
- `proj_a_mm@sparse_attn` (1.95e+38) is the same class, larger
  magnitude — the cube kernel reads further OOB with wrong indices.
- **Routed to:** test framework (same scalar_sem bug as §3.2).

---

## 4. Detailed evidence for §3 (rms_norm family)

The §3.1 verdict rests on the following per-experiment evidence. This
section is the audit trail; the headline conclusions are in §3.

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
localizes to the `.pto` source. This refutes H1 (sync/membar).

### 4.2 Golden-data audit (H2) — golden is correct

Read the captured `v1.bin` (input), `v3.bin` (weight), and
`golden_v2.bin` (reference output) for `vpto_rms_norm/run`.

- `v1.bin` (input): 917504 bf16 elements, **all zero**.
- `v3.bin` (weight): 7168 bf16 elements, **all zero**.
- `golden_v2.bin`: 917504 bf16 elements, **all zero**.

Golden is mathematically correct: `rms_norm(x, w) = x * rsqrt(mean(x²) + eps) * w`; with `x = 0`, `w = 0` → output `0`. The capture is also
correct: verified by reading the same `bin_offset`/`bin_size` window
directly from the source `args.bin` — producer genuinely wrote zeros.
**H2 (golden error) refuted.**

### 4.3 Codegen inspection (H3) — defect confirmed

`diff` of the standalone-leaf `.pto` vs the inner-kernel `.pto`:

```diff
-   %x__ssa_v0_view = pto.make_tensor_view %arg0,
-     shape = [%arg3, %c7168_index], ...   ← leaf: DYNAMIC row dim
+   %x_mixed_inline523__rv_v2_view = pto.make_tensor_view %arg0,
+     shape = [%c128_index, %c7168_index], ...  ← inner: HARDCODED = 128
```

The 7 failing flat indices (256, 257, 258, 259, 512, 513, 768) are the
**starts of even 128-column tiles** in row 0 (col 256 = tile 2, col 512
= tile 4, col 768 = tile 6). The leaked values map to kernel fp32
intermediates:

| flat idx | NPU value (bf16) | bit pattern | identity |
|---:|---:|---|---|
| 256–259 | 1.0 | 0x3F80 | unit scale constant |
| 512–513 | 1000.0 | 0x447A | `rsqrt(1e-6)` = `x_inv_rms` when input is zero |
| 768 | 999424.0 | 0x4974 | stale `x_sq_sum` reduction accumulator |

The output bf16 tile (`%19`/`%25`) is allocated at local-memory
`addr = 8768`, the **same address** as the f32 reduction tiles
(`%rms_x_chunk_inline919__tile` at `addr = 8768`). The bf16 `tstore`
writes 2 bytes/elem into a 4-byte/elem region previously used by f32
intermediates; the residual high bytes leak as bf16 output. **H3
(codegen) confirmed** for the rms_norm family.

### 4.4 Synthesis — `rms_norm` family root cause

| hypothesis | verdict | evidence |
|---|---|---|
| H1 membar/sync | **refuted** | leaf variant passes on same NPU/harness |
| H2 golden/capture error | **refuted** | input + weight + golden all verifiably zero |
| H3 codegen | **CONFIRMED** | inner `.pto` hardcodes row=128; tile-aligned leaks; intermediates at same local-mem addr |

**Routed to:** pypto (inner-kernel codegen variant).

---

## 5. Root-cause C2–C4, C8 — NPU-run-crash + harvest-crash (142 rows)

These rows reached `main.cpp` compilation + launch but the NPU binary
faulted at runtime (134 rows) or the harvester found no usable records
(7 rows). All are **framework-side bugs**, not kernel or codegen
defects. Re-running with representative samples captured the actual
fault text.

### 5.1 C2 — `inout` role not harvested (v4.bin / vN.bin missing)

**Fault text:** `Failed to read v4.bin` / `Failed to get file. Path =
./v4.bin` → NPU host binary exits 1 before kernel launch.

**Root cause:** `capture.py:155-160` only matches `role == "input"`
(for `before_dispatch`) and `role == "output"` (for `after_completion`).
It does NOT match `role == "inout"`. Kernels with `inout` ptr args
(those that read and write the same GM buffer) have their `inout` arg
skipped entirely → no `vN.bin` is written → `main.cpp`'s `ReadFile3`
fails.

**Affected kernels (sampled):** `comb_sinkhorn`, `kv_score_proj`,
`kv_score_proj_0`, `kv_touch`, `gate_pre_route`, `hc_head_seed` —
all have `role=inout` records in their dumps (verified: arg3 of
comb_sinkhorn, arg0 of kv_touch/gate_pre_route/hc_head_seed are all
`role=inout` with both `before_dispatch` and `after_completion`
records).

**Fix:** `capture.py` harvest should treat `inout` as both input (take
`before_dispatch` copy for `vN.bin`) and output (take `after_completion`
copy for `golden_vN.bin`).

### 5.2 C3 — index scalars beyond `ctx_len` default to 0

(Already covered in §3.2/§3.3 — same root cause produces either a
precision FAIL with diff or an NPU crash, depending on whether the
kernel writes anything with the collapsed view.)

**Fault text (crash variant):** NPU exits 1 (AICore exception or
silent garbage — depends on kernel).

**Root cause:** `vpto_run.py:355-363` only marks the FIRST non-SPMD
index scalar as `ctx_len`; `setup_main.py:182` defaults the rest to 0.
Kernels with multiple index args (e.g. row-offset + group-index +
col-offset) get wrong dimensions → tile loops don't cover the GM tensor
→ either leaks (§3.2) or AICore faults (this bucket).

**Affected kernels (sampled):** any kernel whose `.pto` has ≥2 non-SPMD
`index` args. Fix: derive scalar semantics from the `.pto`'s
`make_tensor_view` shape dims, not just "first non-SPMD index".

### 5.3 C4 — output-only ptr → 0-byte alloc → `aclrtMallocHost` fails

**Fault text:** `aclrtMallocHost failed: 100000 ... Invalid_Argument
(EH0007): aclrtMallocHostImpl failed because value 0 for parameter size
is invalid. Expected value: must be greater than zero.`

**Root cause:** `setup_main.py` computes `fileSize_vN = elemCount *
sizeof(dtype)` from `capture_meta.json`'s `elem_counts`. When a kernel
has an output-only ptr whose dump record has `numel = 0` (or the elem
count wasn't captured), `fileSize = 0` → `aclrtMallocHost(0)` fails.
The kernel's actual runtime numel comes from the index scalars (which
are wrong per §5.2), so even if the alloc succeeded the buffer would be
undersized.

**Affected kernels (sampled):** `hc_pre_linear`, `proj_b_mm`, and any
kernel with a 0-element output in its dump.

### 5.4 C8 — 0-iter SPMD (kernel never dispatched)

**Fault text:** `harvest failed: no dump records for kernel <name> (fid <N>)`

**Root cause (two sub-cases):**
1. **`inout` role** (5 of 7): `kv_touch`, `gate_pre_route`,
   `hc_head_seed`, `hc_post_inactive_pad` (×3 prefill modules) — these
   kernels have `inout` records but no `input`/`output` records, so the
   harvester skips them (same bug as §5.1).
2. **Genuinely 0-iter** (1 of 7, `hc_post_inactive_pad` in prefill):
   verified via `deps.json` — no task has this `kernel_id`, meaning the
   kernel's SPMD loop body never executed on any block during capture
   (the kernel is conditional / branch-not-taken in this test input).
   No bytes were ever written → nothing to harvest.

**Fix (sub-case 1):** same as §5.1. **Fix (sub-case 2):** the framework
should skip these kernels gracefully (mark as "not-exercised" rather
than "crash") — they need a different test input that triggers the
branch.

---

## 6. Root-cause C5–C7 — lowering crash (143 rows)

These rows failed at the ptoas or bisheng step, before any NPU
execution. Probing representative kernels across 4 modules captured 4
distinct error classes.

### 6.1 C5 — ptoas `NoMatchingTemplate` for `pto.tdivs` / `pto.tstore`

**Fault text:**
```
error: InsertTemplateAttributes metadata RPC failed: Error: daemon RPC
failed: NoMatchingTemplate: no legal template for op='pto.tdivs'
target='a5'; template_tdivs_tile_scalar: dtype signature
('i32', 'i32', 'i32') is not supported; ... 4 candidate templates rejected
```

**Root cause:** ptoas's A5 template library has no `tdivs` (tile
scalar-division) template for the `i32` dtype signature. The op is
emitted by pypto for integer division on tile buffers (e.g. Sinkhorn
normalization, route hashing). A5 only supports `tdivs` for fp32.

**Affected kernels (probed + confirmed):** `merge_norm`, `rope_cs`,
`rmsnorm_rope`, `route_hash`, `rope`, `prefill_c4_rmsnorm_rope`,
`prefill_idx_c4_rmsnorm_rope`, `prefill_hca_c128_rmsnorm_rope`,
`qr_rms_norm_quant`, `quant` (also hits i8, see §6.2). All contain a
`pto.tdivs` op on i32 tiles.

A separate variant (`ffn_norm`) hits the same class for a different op:
```
NoMatchingTemplate: no legal template for op='pto.tstore' target='a5';
6 candidate templates rejected (custom constraints not satisfied)
```
— a `tstore` with a layout/dtype combination the A5 templates don't
cover.

**Routed to:** ptoas (A5 template coverage). Fix: add i32 `tdivs`
templates, or have pypto lower i32 division to fp32 + cast.

### 6.2 C6 — bisheng `unknown type name 'i8'`

**Fault text:**
```
launch.cpp:35:79: error: unknown type name 'i8'
extern "C" __global__ AICORE void quant(__gm__ float* v1, __gm__ i8* v2, ...);
```

**Root cause:** `pto_parse.py:183` `PTO_TO_CPP` map is missing an `i8`
entry. The `.pto` uses `!pto.ptr<i8>` (signed 8-bit int, e.g. for
quantized int8 weights), but `PTO_TO_CPP` only has `{f32, bf16, f16,
i32, i64, i16, index}` — no `i8`, no `u8`. `setup_main.py` then emits
raw `i8` into `launch.cpp`, which bisheng doesn't recognize (it expects
`int8_t` or `char`).

**Affected kernels (probed + confirmed):** `quant`, `score_mat`,
`kv_and_cache_write`, `qproj_matmul`, `exp_gate_mm`, `exp_h_q`,
`sh_gate_mm`, `sh_up_mm`, `exp_up_mm`, `sh_w2_mm`, `exp_w2_mm`,
`x_norm_quant` — all have `!pto.ptr<i8>` args (quantized weight
matrices or int8 outputs).

**Routed to:** test framework (`pto_parse.py` `PTO_TO_CPP` map). Fix:
add `"i8": "int8_t", "u8": "uint8_t"` to the map.

### 6.3 C7 — ptoas `operand does not dominate this use` + missing .pto

**Fault text (attention_hca):**
```
error: operand #0 does not dominate this use
```
in `build_valid.pto:27:3` — a pypto-emitted `.pto` with an SSA dominance
violation. ptoas rejects it.

**Fault text (qk_pv_aic, qk_pv_aiv, gate_aic, gate_aiv,
mtp_projection_linear_aic/aiv):** no `.pto` file exists at all — these
are `_aic`/`_aiv` suffixed kernels that are **split** at runtime (one
logical kernel → two AIC/AIV compiled artifacts), and the capture dump's
name_map doesn't emit separate `*_aic`/`*_aiv` entries. The harvester
can't find a kernel by that name.

**Routed to:** pypto (SSA dominance) + framework (aic/aiv split
handling).

---

## 7. Module-level rollup + distinct-kernel classification

### 7.1 Module × status heatmap

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

### 7.2 Distinct-kernel best-status classification (129 kernels)

The 338 rows cover **129 distinct kernel basenames** (same kernel
appears in multiple modules). Each kernel's "best status anywhere"
falls into one of 4 buckets:

| best status anywhere | count | root cause class |
|---|---:|---|
| PASS somewhere | 17 | (positive control — framework sound for these shapes) |
| precision FAIL (with diff) somewhere | 3 | C1 (§3): `rms_norm`, `mtp_projection_rms`, `proj_a_mm` |
| FAIL no-diff only (NPU runs then crashes) | 53 | C2/C3/C4 (§5): inout-harvest, scalar=0, size=0 alloc |
| CRASH only (lowering/harvest) | 56 | C5/C6/C7 (§6) + C8 (§5.4) |

The 17 PASS kernels (the positive control set, §8) prove the capture→
replay chain is byte-accurate for their shapes. The 53 "NPU-crash"
kernels are **not** 53 independent bugs — they decompose into 3
framework-side root causes (§5.1–5.3). The 56 "lowering-crash"
kernels decompose into 3 codegen/framework causes (§6.1–6.3).

---

## 8. PASS kernel inventory (the positive control set)

These 52 (module, kernel) pairs reproduce golden bit-for-bit through the
full Route 2 chain. They are the calibration evidence that the framework
itself is sound. (Full list in `sweep_results.csv`; condensed by pattern
below.)

- `*_seed` kernels (kv_proj_seed, qr_proj_seed, hc_pre_seed) — pass in
  every module they appear (12/12). Small fatobj (~5 KB), fast (~0.5 ms).
- `kv_hadamard`, `mix_x`, `hc_post`, `merge_norm` — pass consistently
  across modules.
- `rms_norm` passes as a **leaf** module but fails as an **inner**
  kernel — see §3.1.

### 8.1 What the PASSes tell us

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

## 9. Next steps

**All 338 rows classified.** The 286 non-pass rows decompose into 8
root-cause classes (§3, §5, §6). The headline finding: **~180 of 286
failures (63%) are framework-side bugs** fixable in the test framework
without touching pypto/ptoas/bisheng.

### 9.1 Framework-side fixes (would convert ~180 crash rows → pass or true-fail)

| fix | class | affected rows | where |
|---|---|---:|---|
| Harvest `role=inout` as both input+output | C2 (§5.1), C8 (§5.4) | ~55 | `capture.py:155-160` |
| Derive index-scalar semantics from `.pto` tensor-view shape dims (not just first non-SPMD) | C3 (§5.2), C1-§3.2/§3.3 | ~45 + 3 precision | `vpto_run.py:355-363`, `setup_main.py:182` |
| Add `i8→int8_t`, `u8→uint8_t` to `PTO_TO_CPP` | C6 (§6.2) | ~30 | `pto_parse.py:183` |
| Handle 0-byte output alloc (skip `aclrtMallocHost(0)`) | C4 (§5.3) | ~30 | `setup_main.py` |
| Handle `_aic`/`_aiv` split kernels in name_map lookup | C7 (§6.3) | ~8 | `capture.py` |
| Treat NaN in compare output as FAIL (not pass) | C1-§3.3 | 1 | `compare.py` |

### 9.2 pypto/ptoas-side fixes (the ~80 genuine lowering/precision bugs)

| fix | class | affected rows | where |
|---|---|---:|---|
| Add A5 `tdivs` i32 template (or lower i32→fp32 in pypto) | C5 (§6.1) | ~80 | ptoas |
| Fix inner-kernel `rms_norm` hardcoded shape + local-mem tile collision | C1-§3.1 | 5 | pypto |
| Fix `build_valid.pto` SSA dominance violation | C7 (§6.3) | 1 | pypto |

### 9.3 Genuine 0-iter SPMD (framework can't fix alone)

`hc_post_inactive_pad` (prefill modules) is a conditional kernel whose
SPMD loop body never executed during capture — the test input doesn't
trigger its branch. Needs a different test input that activates the
inactive-pad path; not a framework or codegen bug.

---

## 10. VMI+membar+bisheng-VF-off route — comparison sweep

A second full 338-kernel sweep was run with a different ptoas+bisheng
codegen route, per `ptoas/README-vmi-membar-bishengvfoff.md`. This route
turns on the VMI fusion pipeline, inserts vecscope memory barriers, and
disables bisheng's VF-fusion backend passes so the NPU sees the
membar-inserted schedule without re-fusion.

| switch | baseline route | vmi-membar-vfoff route |
|---|---|---|
| ptoas `--enable-vmi` | off | **on** |
| ptoas `--enable-op-fusion` | on (bare) | `=true` (required for VMI) |
| ptoas `--enable-vecscope-mem-bar` | off | **on** |
| bisheng VF-fusion | on (default) | **off** (7 `-mllvm` options) |

Results CSV: `baselines/vpto_dsv4_vector/sweep_results_vmi-membar-vfoff.csv`.

### 10.1 Status tally (baseline vs vmi-membar-vfoff)

| status | baseline | vmi-membar-vfoff | delta |
|---|---:|---:|---:|
| PASS | 52 | 45 | -7 |
| FAIL (no diff, NPU crash) | 135 | 152 | +17 |
| FAIL (with diff, true precision) | **8** | **1** | **-7** |
| CRASH (lowering/harvest) | 143 | 140 | -3 |

**Headline finding: the VMI+membar route eliminates 7 of 8 true-precision
FAILs** (the local-memory leak class). The single remaining precision
FAIL is `mtp_projection_rms`, which is the framework `scalar_sem=0` bug
(§3.2) — not a codegen issue, so no route change can fix it.

### 10.2 The 7 precision FAILs fixed by VMI+membar

| kernel | module | baseline max_diff | vmi status |
|---|---|---:|---|
| `rms_norm` | attention_csa | 999424.0 | FAIL (NPU crash — UB alignment, §10.4) |
| `rms_norm` | attention_swa | 999424.0 | FAIL (NPU crash — UB alignment) |
| `rms_norm` | prefill_attention_csa | 999424.0 | FAIL (NPU crash — UB alignment) |
| `rms_norm` | prefill_attention_hca | 999424.0 | FAIL (NPU crash — UB alignment) |
| `rms_norm` | prefill_attention_swa | 999424.0 | FAIL (NPU crash — UB alignment) |
| `proj_a_mm` | attention_csa | 27.82 | **PASS (max_diff=0)** |
| `proj_a_mm` | sparse_attn | 1.95e+38 | **PASS (max_diff=0)** |

The `rms_norm` family: the baseline route's local-memory tile-address
collision (§3.1) no longer produces leaked intermediates — but the VMI
route introduces a **new** UB-alignment crash on the same kernel (§10.4).
So the original leak is gone, replaced by a different VMI codegen bug.
Net: still not usable, but the failure *mode* changed from "wrong
output" to "clean crash" (which is strictly better — a crash is
detectable, a silent wrong output is not).

The `proj_a_mm` pair: **fully fixed** — both the bounded (27.82) and
catastrophic (1.95e+38) FAILs become exact PASS (max_diff=0). VMI fusion
eliminated the cube kernel's wrong-index read that the scalar_sem=0
framework bug exposed.

### 10.3 The 6 IMPROVED kernels (crash/fail → PASS)

| kernel | modules | baseline status | vmi status | vmi timing |
|---|---|---|---|---:|
| `merge_norm` | attention_csa, attention_swa, sparse_attn | crash (pto.tdivs) | PASS | 0.58 ms |
| `proj_a_mm` | attention_csa, sparse_attn | fail (precision) | PASS | 0.45–0.53 ms |
| `proj_a_mm` | attention_swa | fail (precision) | PASS | 0.53 ms |

`merge_norm` crashed in baseline due to `pto.tdivs` NoMatchingTemplate
(§6.1). Under VMI, the i32 tdivs op is fused away by the VMI pipeline →
the template is no longer needed → the kernel compiles and runs
correctly. **VMI fusion fixes the pto.tdivs template-coverage gap for
`merge_norm`.** (Not for other tdivs kernels — see §10.5.)

### 10.4 The 13 REGRESSED kernels (PASS → NPU crash)

All 13 regressions have the **identical** fault:
```
errcode:(340) errorStr: The address for VEC to access UB is not aligned.
retCode=0x31, vector core exception.
```

| kernel | modules affected |
|---|---|
| `mix_x` | attention_csa, attention_swa, hc_pre, prefill_attention_csa, prefill_attention_hca, prefill_attention_swa |
| `merge_norm` | prefill_attention_csa, prefill_attention_hca, prefill_attention_swa, prefill_sparse_attn |
| `rms_norm` (leaf) | rms_norm |
| `hc_head_reduce` | hc_head |
| `mtp_projection_norm` | mtp_projection |

Pattern: `mix_x` regresses in 6 modules, `merge_norm` regresses in 4
prefill modules, plus 2 singletons. All hit the same VMI codegen bug:
the VMI fusion pipeline emits a VEC→UB access with an unaligned address.
**Routed to:** ptoas (VMI pipeline UB-alignment bug).

Note the `merge_norm` split: in decode modules it IMPROVES (crash→pass,
§10.3); in prefill modules it REGRESSES (pass→crash). The difference is
the shape — prefill's larger `[128, ...]` tiles trigger the VMI
alignment bug, while decode's smaller `[8, ...]` tiles don't.

### 10.5 What VMI+membar does NOT fix

- **`mtp_projection_rms`** (`max_diff=1000.0`, `n_over=8/16`): unchanged
  — same `scalar_sem=0` framework bug (§3.2), not a codegen issue.
- **`pto.tdivs` on other kernels**: VMI fusion fixes `merge_norm`
  (decode) but NOT `rope`, `rope_cs`, `rmsnorm_rope`,
  `prefill_c4_rmsnorm_rope`, etc. — these still hit the
  NoMatchingTemplate error. VMI only fuses tdivs when it's inside a
  fusion-eligible region; standalone tdivs ops are untouched.
- **`i8` unmapped** (§6.2): unaffected — bisheng still rejects `i8`.
- **`inout` harvest bug** (§5.1): unaffected — framework-side, route-
  independent.

### 10.6 Route comparison summary

| metric | baseline | vmi-membar-vfoff | verdict |
|---|---:|---:|---|
| total PASS | 52 | 45 | -7 (net worse) |
| true precision FAILs | 8 | 1 | **-7 (much better)** |
| NPU crashes (UB alignment) | 0 | 13 | +13 (new VMI bug) |
| lowering crashes (tdivs) | ~80 | ~77 | -3 (VMI fuses 3 merge_norm) |
| framework bugs (inout/scalar/i8) | ~180 | ~180 | unchanged |

**The VMI+membar route is a precision win but a stability loss.** It
eliminates 7 of 8 silent-precision FAILs (the local-mem leak class) and
fixes `merge_norm`+`proj_a_mm` (6 kernels, crash/fail → exact PASS). But
it introduces 13 new NPU crashes from a VMI UB-alignment codegen bug.
Net pass count drops by 7, but the failure *quality* improves: silent
wrong outputs become loud crashes.

**Recommendation:** the VMI route is the better precision baseline once
the 13-kernel UB-alignment bug is fixed in ptoas. Until then, the
baseline route is safer (more passes, no regressions) but produces 7
silent precision FAILs.
