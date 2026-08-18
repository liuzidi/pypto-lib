# Fix C6, C7b, C8b in the DSV4 VPTO validation framework

## Summary

Three framework-side fixes to `baselines/vpto_dsv4_vector/FIX_HANDOFF.md`. C6 (missing i8/u8 type mapping) and C7b (`_aic`/`_aiv` split kernels) are the highest-leverage: C6 unblocks ~30 kernels that use `!pto.ptr<i8>` (quant weights), C7b fixes ~6 split kernels that currently silently compile/run the wrong half. C8b is a small correctness fix so the sweep reports "not-exercised" instead of "crash" for dead-branch kernels.

Note: C2/C8a (inout harvest) and C1b/C3 (scalar_sem derivation) are already fixed in commits `dcc5568` / `aefe714`. C5 (ptoas tdivs) is already implemented in the ptoas source tree. Only C6/C7b/C8b remain.

---

## Fix 1 — C6: i8→int8_t / u8→uint8_t type mapping (P0, ~30 lines)

**Files:** `.claude/skills/vpto-board-validate/lib/pto_parse.py`

### `PTO_TO_CPP` (line ~249)
Add `i8`/`u8` entries:
```python
PTO_TO_CPP = {"f32": "float", "bf16": "bfloat16_t", "f16": "half",
              "i32": "int32_t", "i64": "int64_t", "i16": "int16_t",
              "i8": "int8_t", "u8": "uint8_t",
              "index": "int64_t"}
```

### `pto_type_to_c` (lines 174-188)
Add `i8`/`u8` to both the `mapping` dict and the explicit branches so the `__gm__` pointer type is correct too:
```python
mapping = {
    "f32": ("float", "__gm__ float*"),
    "f16": ("uint16_t", "__gm__ bfloat16_t*"),
    "bf16": ("uint16_t", "__gm__ bfloat16_t*"),
    "i32": ("int32_t", "__gm__ int32_t*"),
    "i64": ("int64_t", "__gm__ int64_t*"),
    "i16": ("int16_t", "__gm__ int16_t*"),
    "i8": ("int8_t", "__gm__ int8_t*"),
    "u8": ("uint8_t", "__gm__ uint8_t*"),
}
```
(host type for i8 is `int8_t`, not the raw `i8`).

### `capture.py` `_DTYPE_TO_NP` — already has `"INT8": "int8"` ✅, no change needed
Confirmed: dump `dtype` strings are `INT8`/`INT32`/`INT64`/`FLOAT32`/`BFLOAT16`, all already mapped.

**Verification:** re-run a `quant` kernel replay; `launch.cpp` should now emit `__gm__ int8_t* v2` instead of `__gm__ i8* v2`, and bisheng should no longer error `unknown type name 'i8'`.

---

## Fix 2 — C7b: per-kernel function extraction for `_aic`/`_aiv` split kernels (P3, ~6 lines)

**Root cause:** `parse_pto` uses `re.search` to grab the **first** `func.func` in the file. When `validate.py` prefix-matches `qk_pv_aic`→`qk_pv.pto` and `qk_pv_aiv`→`qk_pv.pto`, both kernels parse/compile the `_aic` (first) function. `detect_kernel_kind` also greps the whole file and returns "cube" for the `_aiv` case (since "cube" appears in the `_aic` func's attrs). `preprocess_pto`'s `replace_all`-style func attr edit marks *both* funcs for codegen.

**Files:** `pto_parse.py`, `setup_vpto.py`, `vpto_run.py`

### `pto_parse.py` — `parse_pto` gains an optional `kernel` arg
```python
def parse_pto(pto_path: Path, kernel: str | None = None) -> dict:
```
- If `kernel` is given, anchor the `func.func @<kernel>(...)` regex on that exact name (escape it), so the right half of a split `.pto` is parsed. The `pto.make_tensor_view` regex below already scans the whole file text, which is fine — view names are param-local and the `%argN` indices line up 1:1 with the matched func's signature (both halves of a split kernel share the same ptr/scalar signature, confirmed on `qk_pv`/`mtp_projection_linear`).
- If `kernel` is None, keep current behavior (first func) for backward compat with Mode A `golden-lib` callers that pass a single-func `.pto`.

### `setup_vpto.py` — `detect_kernel_kind` and `preprocess_pto` become per-kernel
```python
def detect_kernel_kind(pto_text: str, kernel: str | None = None) -> str:
```
- When `kernel` is given, narrow the "cube" search to the substring after `func.func @<kernel>` (up to the next `func.func` or EOF), so a vector `_aiv` half is not misdetected as cube just because the `_aic` half has "cube".

```python
def preprocess_pto(pto_src, pto_dst, kernel_kind, kernel=None):
```
- When `kernel` is given, the func-attr `replace_all` is replaced with a **scoped** edit: only add `pto.kernel` to the `attributes {pto.kernel_kind` occurrence belonging to `func.func @<kernel>`. The module-level `pto.kernel_kind` edit is left as-is (it's a module default; ptoas still emits the targeted kernel symbol based on `pto.kernel`).

### `vpto_run.py` — pass `kernel` through
- `info = parse_pto(args.pto, kernel)` (line ~379) — `kernel` already resolved as `args.kernel or args.pto.stem` at line 331.
- `kind = setup_vpto.detect_kernel_kind(pto_text, kernel)` (line 384).
- `setup_vpto.preprocess_pto(args.pto, pto_file, kind, kernel)` (line 529).

**Verification:** re-run `qk_pv_aic` and `qk_pv_aiv` separately; their `result.json` `has_kernel_sym` should each report the *distinct* symbol (`qk_pv_aic` / `qk_pv_aiv`), and the `_aiv` run should lower as a vector kernel (not cube).

---

## Fix 3 — C8b: mark 0-iter SPMD kernels as "not-exercised" (P3, 2 lines)

**Root cause:** `hc_post_inactive_pad` has no task in `deps.json` for the prefill test input → no dump records → `harvest_kernel` raises `RuntimeError("no dump records ...")` → `validate.py` records it as `crash`.

**Files:** `tests/dsv4_validate/capture.py`, `tests/dsv4_validate/validate.py`

### `capture.py` — raise a typed exception
```python
class NotExercised(Exception):
    """Kernel branch not triggered by the test input (0-iter SPMD, no task)."""
```
In `harvest_kernel`, change the "no dump records" raise to `raise NotExercised(...)`. Export it.

### `validate.py` — catch `NotExercised` and report `not-exercised` (not `crash`)
In `run_phase5_module`'s `harvest_kernel` try/except (lines 254-262), add a branch:
```python
except _capture.NotExercised as e:
    results.append(_phase5_error_row(
        module, model_py, mode, device, kname,
        f"not-exercised: {e}", route=route))
    # and set compare_status = "not-exercised" on this row
    continue
```
`_phase5_error_row` currently hardcodes `compare_status="crash"`; extend it to accept a `compare_status` arg defaulting to `"crash"`, and pass `"not-exercised"` here.

**Verification:** `--module hc_post --mode prefill` should now list `hc_post_inactive_pad` as `not-exercised` (not `crash`).

---

## Fix 4 — Update FIX_HANDOFF.md

Mark C6/C7b/C8b sections as "已实施" with the fix actually applied (mirroring how C2/C5/C1b were annotated). Adjust the priority table at the bottom to reflect the three new ✅ entries.

---

## Non-goals (explicitly left to other layers, per FIX_HANDOFF.md)
- C1a (pypto inner-kernel hardcoded shape) — pypto side.
- C7a (SSA dominance in `build_valid.pto`) — pypto side.
- VMI-UB (13 regressions) — ptoas VMI pipeline side.
- C4 (0-byte alloc for output-only numel=0 ptrs) — separate framework issue, not in the C6/C7/C8 scope of this task.
