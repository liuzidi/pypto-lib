#!/usr/bin/python3
"""Compare NPU output (.bin) vs golden (golden_*.bin).

Default (Mode A — test_for_ptoas golden_lib): exact-match for f32, ULP<=1
for bf16. Strict, as in the original harness.

Mode B (DSV4 run_jit golden): if VPTO_COMPARE_RTOL / VPTO_COMPARE_ATOL env
vars are set (the skill exports them for --model-py runs), do a torch-style
allclose: |out - golden| <= atol + rtol * |golden|. Reports max_diff +
the tolerance threshold + pass/fail. This matches the DSV4 models' own
run_jit tolerances (rtol=5e-3 atol=5e-3 by default).
"""
import os, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation_runtime import load_case_meta, bf16_to_float32


def _tol_env():
    """Return (rtol, atol) from env, or None if not set (Mode A)."""
    rtol = os.environ.get("VPTO_COMPARE_RTOL")
    atol = os.environ.get("VPTO_COMPARE_ATOL")
    if rtol is None and atol is None:
        return None
    return (float(rtol) if rtol is not None else 0.0,
            float(atol) if atol is not None else 0.0)


def _to_float_view(arr: np.ndarray, np_type_str):
    """View bf16 (uint16) as float32 for comparison; pass through f32."""
    if "uint16" in str(np_type_str):  # bf16 stored as uint16
        return bf16_to_float32(arr.reshape(-1, 1)).ravel()
    return arr.astype(np.float64).ravel()


def main():
    meta = load_case_meta()
    tol = _tol_env()
    failed = False
    for name in meta.outputs:
        golden_path = Path(f"golden_{name}.bin")
        output_path = Path(f"{name}.bin")
        if not golden_path.exists() or not output_path.exists():
            print(f"[WARN] {name}: missing file, skip")
            continue

        np_t = meta.np_types.get(name)
        golden = np.fromfile(golden_path, dtype=np_t)
        output = np.fromfile(output_path, dtype=np_t)

        if len(golden) != len(output):
            print(f"[ERROR] {name}: size mismatch golden={len(golden)} output={len(output)}")
            failed = True
            continue

        # --- Mode B: tolerance-based compare ---
        if tol is not None:
            rtol, atol = tol
            gf = _to_float_view(golden, str(np_t))
            of = _to_float_view(output, str(np_t))
            abs_diff = np.abs(of - gf)
            max_diff = float(np.max(abs_diff))
            threshold = atol + rtol * np.max(np.abs(gf))
            n_over = int(np.sum(abs_diff > threshold))
            if n_over == 0:
                print(f"[INFO] {name} compare passed: max_diff={max_diff:.6g} "
                      f"threshold={threshold:.6g} (rtol={rtol} atol={atol})")
            else:
                print(f"[ERROR] {name} compare failed: max_diff={max_diff:.6g} "
                      f"threshold={threshold:.6g} n_over={n_over}/{len(gf)} "
                      f"(rtol={rtol} atol={atol})")
                failed = True
            continue

        # --- Mode A: exact/ULP compare (original) ---
        if golden.dtype == np.uint16 and "uint16" in str(np_t):
            gf = bf16_to_float32(golden.reshape(-1, 1)).ravel()
            of = bf16_to_float32(output.reshape(-1, 1)).ravel()
            mismatches = np.where(gf != of)[0]
            if len(mismatches) == 0:
                print(f"[INFO] {name} compare passed (exact match)")
            else:
                max_ulp = 0
                max_idx = 0
                for idx in mismatches[:1000]:
                    ulp = abs(int(golden.ravel()[idx]) - int(output.ravel()[idx]))
                    if ulp > max_ulp:
                        max_ulp = ulp
                        max_idx = idx
                if max_ulp <= 1:
                    print(f"[INFO] {name} compare passed (max_ulp={max_ulp} idx={max_idx})")
                else:
                    gb = golden.ravel()[max_idx]
                    ob = output.ravel()[max_idx]
                    print(f"[ERROR] bf16 compare failed ({name}): max_ulp={max_ulp} "
                          f"idx={max_idx} golden_bits={gb} output_bits={ob} "
                          f"golden={gf[max_idx]} output={of[max_idx]}")
                    failed = True
        else:
            if np.array_equal(golden, output):
                print(f"[INFO] {name} compare passed (exact match)")
            else:
                max_diff = np.max(np.abs(golden.astype(np.float64) - output.astype(np.float64)))
                print(f"[ERROR] {name} compare failed: max_diff={max_diff}")
                failed = True

    if failed:
        print("[ERROR] compare failed")
        sys.exit(1)
    else:
        print("[INFO] compare passed")


if __name__ == "__main__":
    main()
