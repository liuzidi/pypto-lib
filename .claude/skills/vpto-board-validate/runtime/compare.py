#!/usr/bin/python3
"""对比 NPU 输出 (.bin) 与 golden (golden_*.bin)。"""
import os, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation_runtime import load_case_meta, bf16_to_float32

def ulp_diff(golden: np.ndarray, output: np.ndarray) -> int:
    ga = golden.ravel().view(np.uint16) if golden.dtype == np.uint16 else golden.ravel()
    oa = output.ravel().view(np.uint16) if output.dtype == np.uint16 else output.ravel()
    if ga.dtype != oa.dtype:
        ga = ga.astype(np.int64)
        oa = oa.astype(np.int64)
    return int(np.max(np.abs(ga.astype(np.int64) - oa.astype(np.int64))))

def main():
    meta = load_case_meta()
    failed = False
    for name in meta.outputs:
        golden_path = Path(f"golden_{name}.bin")
        output_path = Path(f"{name}.bin")
        if not golden_path.exists() or not output_path.exists():
            print(f"[WARN] {name}: missing file, skip")
            continue

        golden = np.fromfile(golden_path, dtype=meta.np_types[name])
        output = np.fromfile(output_path, dtype=meta.np_types[name])

        if len(golden) != len(output):
            print(f"[ERROR] {name}: size mismatch golden={len(golden)} output={len(output)}")
            failed = True
            continue

        # bf16 special handling
        if golden.dtype == np.uint16 and 'bf16' in str(meta.np_types[name]):
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
                    print(f"[ERROR] bf16 compare failed ({name}): max_ulp={max_ulp} idx={max_idx} golden_bits={gb} output_bits={ob} golden={gf[max_idx]} output={of[max_idx]}")
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
