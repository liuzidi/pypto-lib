#!/usr/bin/env python3
"""Diagnose every DSV4 VPTO failure with the actual error class.

Re-runs each kernel through vpto_run.py individually (avoiding the
run_all.py daemon-socket race), captures the full stderr, and classifies
the failure into:
  - pass
  - precision_fail (NPU ran, compare mismatch)
  - npu_crash_mte_ddr (errcode 95, vector core exception)
  - npu_crash_fixpipe (errcode 161, aicore exception)
  - npu_crash_other (retCode 0x31/0x26 without known errcode)
  - ptoas_tdivs (NoMatchingTemplate pto.tdivs)
  - ptoas_expand (ExpandTileOp instantiation failed)
  - ptoas_metadata (PTODSL metadata query failed)
  - ptoas_pass_failed (Pass execution failed, other)
  - ptoas_vcvt (pto.vmi.vcvt op error)
  - timeout
  - other
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
VPTO_RUN = REPO / ".claude/skills/vpto-board-validate/vpto_run.py"
GOLDEN_LIB = ROOT / "dsv4_golden_lib.py"
PTO_DIR = ROOT / ".pto"

def classify(stdout, stderr, rc):
    combined = (stdout or "") + "\n" + (stderr or "")
    # success
    if "compare passed" in combined and rc == 0:
        return "pass", ""
    # precision fail
    m = re.search(r"max_ulp=(\d+)", combined)
    md = re.search(r"max_diff=([\d.eE+-]+)", combined)
    if (m or md) and "compare failed" in combined:
        return "precision_fail", f"max_ulp={m.group(1) if m else ''} max_diff={md.group(1) if md else ''}"
    # NPU crash — look for errcode
    if "errcode:(95)" in combined or "MTE instruction is out of range" in combined:
        return "npu_crash_mte_ddr", "errcode:95 MTE DDR out of range, retCode=0x31"
    if "errcode:(161)" in combined or "fixpipe to write GM is invalid" in combined:
        return "npu_crash_fixpipe", "errcode:161 fixpipe GM invalid, retCode=0x26"
    if "retCode=0x31" in combined:
        return "npu_crash_vec_other", "retCode=0x31 vector core exception (no errcode:95/161)"
    if "retCode=0x26" in combined:
        return "npu_crash_aic_other", "retCode=0x26 aicore exception (no errcode:95/161)"
    # ptoas compile-time failures
    if "NoMatchingTemplate" in combined and "tdivs" in combined:
        m = re.search(r"NoMatchingTemplate:.*", combined)
        return "ptoas_tdivs", (m.group(0)[:120] if m else "tdivs NoMatchingTemplate")
    if "ExpandTileOp" in combined or "ExpandTileOp" in combined:
        return "ptoas_expand", "ExpandTileOp template instantiation failed"
    if "PTODSL metadata query" in combined or "metadata query raised" in combined:
        return "ptoas_metadata", "PTODSL metadata query failed"
    if "pto.vmi.vcvt" in combined:
        return "ptoas_vcvt", "pto.vmi.vcvt op error"
    if "Pass execution failed" in combined:
        # extract the first error line
        m = re.search(r"error:.*", combined)
        return "ptoas_pass_failed", (m.group(0)[:120] if m else "Pass execution failed")
    if "FAILED step" in combined:
        m = re.search(r"FAILED step:.*", combined)
        return "ptoas_failed", (m.group(0)[:120] if m else "vpto_run FAILED")
    if "timed out" in combined.lower() or rc == 124:
        return "timeout", ""
    return "other", f"rc={rc} {combined[-200:]}"


def find_kernels():
    kernels = []
    for pto in sorted(PTO_DIR.glob("*.pto")):
        k = pto.stem
        # split kernels: add _aic and _aiv variants
        text = pto.read_text()
        if "_aic" in text and "func.func" in text:
            kernels.append(f"{k}_aic")
            kernels.append(f"{k}_aiv")
        else:
            kernels.append(k)
    return kernels


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "6"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    kernels = find_kernels()
    if only:
        kernels = [k for k in kernels if k == only]
    print(f"# Diagnosing {len(kernels)} kernels on device {device}")
    results = []
    for i, k in enumerate(kernels, 1):
        # find the .pto file (strip _aic/_aiv)
        base = re.sub(r"_(aic|aiv)$", "", k)
        pto = PTO_DIR / f"{base}.pto"
        if not pto.exists():
            print(f"[{i}/{len(kernels)}] {k}: SKIP (no .pto)")
            continue
        # clean daemon sockets
        for sock in Path("/tmp").glob("tilelib_daemon_*.sock"):
            try: sock.unlink()
            except OSError: pass
        cmd = [
            "bash", "-c",
            f"source {REPO}/scripts/vpto_env.sh 2>/dev/null; "
            f"exec python3 {VPTO_RUN} "
            f"--pto {pto} --golden-lib {GOLDEN_LIB} "
            f"--kernel {k} --mode decode --device {device} --route baseline"
        ]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0
            cls, detail = classify(r.stdout, r.stderr, r.returncode)
        except subprocess.TimeoutExpired:
            elapsed = 120
            cls, detail = "timeout", ""
        except Exception as e:
            elapsed = time.time() - t0
            cls, detail = "other", str(e)[:120]
        results.append({"kernel": k, "class": cls, "detail": detail, "elapsed": round(elapsed,1)})
        status = "PASS" if cls == "pass" else f"FAIL({cls})"
        print(f"[{i}/{len(kernels)}] {k}: {status} ({elapsed:.1f}s) {detail[:60]}")
    # summary
    from collections import Counter
    c = Counter(r["class"] for r in results)
    print(f"\n# Summary: {len(results)} kernels")
    for cls, n in c.most_common():
        print(f"  {cls}: {n}")
    # write json
    out = ROOT / "crash_diagnosis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# Detail: {out}")


if __name__ == "__main__":
    main()
