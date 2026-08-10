#!/usr/bin/env python
"""Collect a PMU baseline for the 27 deepseek_v4_pro single-card operators.

A "baseline" here is, per operator, the sum of the ``pmu_idc_aic_vec_busy_o``
(vector-pipe busy cycles) column across every task row of that operator's
``pmu.csv``, plus the matching ``pmu_total_cycles`` sum for context. The
baseline is collected with A5 tile op-fusion **disabled** (``--no-fusion``)
and PMU event type 2 (``PIPE_UTILIZATION``), which is the configuration all
27 operators pass precision validation under.

The output is the comparison anchor for a later PTOAS version's end-to-end
run: re-collect with the new PTOAS, diff the per-op ``vec_busy_sum`` column.

Usage:
    source .env_a5.sh
    python scripts/collect_pmu_baseline.py -p a5 -d 6             # all 27
    python scripts/collect_pmu_baseline.py -p a5 -d 6 --only gate,rmsnorm
    python scripts/collect_pmu_baseline.py -p a5 -d 6 --pmu 2     # default
    python scripts/collect_pmu_baseline.py -p a5 -d 6 --tag new_ptoas_v0.55

Outputs (under baselines/<tag>/):
    <op>.pmu.csv          raw per-task PMU CSV, copied from build_output
    baseline_summary.csv one row per op: vec_busy_sum, total_cycles_sum, ...
    baseline_summary.txt human-readable table (same data, fixed-width)
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve()
PYPTO_LIB_ROOT = HERE.parent.parent
SCRIPTS = HERE.parent
MODEL_DIR = PYPTO_LIB_ROOT / "models" / "deepseek_v4_pro"
RUN_OP_PMU = SCRIPTS / "run_op_pmu.py"
BASELINES_ROOT = PYPTO_LIB_ROOT / "baselines"

# The 27 single-card operators that pass under --no-fusion on A5. Order groups
# them by function (norm/proj, gate/experts, hc, decode attn, ...) mirroring
# docs/run-deepseek-v4-pro-single-ops.md §5. This is the canonical baseline set.
OPS: list[str] = [
    # norm / proj
    "rmsnorm", "qkv_proj_rope", "mtp_projection",
    # gate / experts
    "gate", "expert_shared", "expert_routed",
    # hc (hidden-compact)
    "hc_pre", "hc_post", "hc_head",
    # decode attention
    "decode_attention_swa", "decode_attention_csa", "decode_attention_hca",
    # decode sparse attn
    "decode_sparse_attn", "decode_sparse_attn_swa", "decode_sparse_attn_hca",
    # decode compressor / indexer
    "decode_compressor_ratio4", "decode_compressor_ratio128",
    "decode_indexer", "decode_indexer_compressor",
    # prefill attention
    "prefill_attention_swa", "prefill_attention_csa", "prefill_attention_hca",
    "prefill_sparse_attn",
    # prefill compressor / indexer
    "prefill_compressor_ratio4", "prefill_compressor_ratio128",
    "prefill_indexer", "prefill_indexer_compressor",
]

# The vector-pipe busy column we are baselining. On A5 PIPE_UTILIZATION this is
# the only vec-busy counter; "rvec_busy" in conversation maps to this column.
VEC_BUSY_COL = "pmu_idc_aic_vec_busy_o"
TOTAL_CYCLES_COL = "pmu_total_cycles"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect PMU (vec_busy) baseline for the 27 DSV4-Pro ops.")
    ap.add_argument("-p", "--platform", default="a5")
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--pmu", type=int, default=2,
                    help="PMU event type (default 2 = PIPE_UTILIZATION, the "
                         "group that carries vec_busy).")
    ap.add_argument("--no-fusion", action="store_true", default=True,
                    help="disable A5 op-fusion (default ON for this baseline; "
                         "pass --fusion to keep fusion enabled).")
    ap.add_argument("--fusion", dest="no_fusion", action="store_false",
                    help="keep A5 op-fusion ENABLED (not the baseline default; "
                         "18/27 ops are expected to fail).")
    ap.add_argument("--tag", default="nofusion_a5_pmu2",
                    help="subdir under baselines/ for this collection "
                         "(default: nofusion_a5_pmu2).")
    ap.add_argument("--only", default="",
                    help="comma-separated op names to run (default: all 27).")
    return ap.parse_args()


def run_one(op: str, args: argparse.Namespace) -> tuple[int, str]:
    """Run run_op_pmu.py for one op. Returns (exit_code, pmu_csv_path_or_empty).

    We do NOT rely on run_op_pmu.py copying pmu.csv next to the operator (that
    default was removed to avoid polluting the operator directory across a
    27-op sweep). Instead we glob build_output for the newest pmu.csv — the
    same lookup run_op_pmu.py itself uses to report the path.
    """
    cmd = [sys.executable, str(RUN_OP_PMU), op,
           "-p", args.platform, "-d", str(args.device),
           "--pmu", str(args.pmu)]
    if args.no_fusion:
        cmd.append("--no-fusion")
    proc = subprocess.run(cmd, cwd=str(PYPTO_LIB_ROOT))
    bo = PYPTO_LIB_ROOT / "build_output"
    pmus = (sorted(bo.glob("**/dfx_outputs/pmu.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
            if bo.is_dir() else [])
    return proc.returncode, (str(pmus[0]) if pmus else "")


def sum_columns(pmu_csv: pathlib.Path) -> dict | None:
    """Sum vec_busy and total_cycles across all task rows of one pmu.csv."""
    if not pmu_csv.is_file():
        return None
    with open(pmu_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    def col_sum(name: str) -> int:
        total = 0
        for r in rows:
            try:
                total += int(float(r.get(name, 0) or 0))
            except (TypeError, ValueError):
                pass
        return total
    return {
        "rows": len(rows),
        VEC_BUSY_COL + "_sum": col_sum(VEC_BUSY_COL),
        TOTAL_CYCLES_COL + "_sum": col_sum(TOTAL_CYCLES_COL),
    }


def main() -> int:
    args = parse_args()

    only = {o.strip() for o in args.only.split(",") if o.strip()}
    ops = [o for o in OPS if (not only or o in only)]
    unknown = only - set(OPS)
    if unknown:
        print(f"[baseline] WARN: unknown ops ignored: {sorted(unknown)}",
              flush=True)
    if not ops:
        print("[baseline] no ops to run", flush=True)
        return 1

    out_dir = BASELINES_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] tag={args.tag} platform={args.platform} "
          f"device={args.device} pmu={args.pmu} "
          f"fusion={'OFF' if args.no_fusion else 'ON'}", flush=True)
    print(f"[baseline] ops={len(ops)} -> {out_dir}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    n_pass = 0
    for i, op in enumerate(ops, 1):
        print(f"\n[baseline] ({i}/{len(ops)}) {op} ...", flush=True)
        code, op_csv = run_one(op, args)
        passed = code == 0
        if passed:
            n_pass += 1
        # archive the raw pmu.csv next to the operator -> baselines/<tag>/
        archived = ""
        if op_csv:
            src = pathlib.Path(op_csv)
            dst = out_dir / f"{op}.pmu.csv"
            shutil.copy2(src, dst)
            archived = str(dst)
        sums = sum_columns(pathlib.Path(op_csv)) if op_csv else None
        row = {
            "op": op,
            "pass": "PASS" if passed else "FAIL",
            "rows": sums["rows"] if sums else 0,
            "vec_busy_sum": sums[VEC_BUSY_COL + "_sum"] if sums else 0,
            "total_cycles_sum": sums[TOTAL_CYCLES_COL + "_sum"] if sums else 0,
            "pmu_csv": archived,
        }
        rows.append(row)
        print(f"[baseline]   {op}: {row['pass']} rows={row['rows']} "
              f"vec_busy_sum={row['vec_busy_sum']} "
              f"total_cycles_sum={row['total_cycles_sum']}", flush=True)

    elapsed = time.time() - t0

    # write summary CSV
    summary_csv = out_dir / "baseline_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # write fixed-width table for quick eyeballing
    summary_txt = out_dir / "baseline_summary.txt"
    cols = ["op", "pass", "rows", "vec_busy_sum", "total_cycles_sum"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    with open(summary_txt, "w") as f:
        f.write("  ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("  ".join("-" * widths[c] for c in cols) + "\n")
        for r in rows:
            f.write("  ".join(str(r[c]).ljust(widths[c]) for c in cols) + "\n")
        f.write("\n")
        f.write(f"platform={args.platform} device={args.device} "
                f"pmu={args.pmu} fusion={'OFF' if args.no_fusion else 'ON'}\n")
        f.write(f"ops={len(rows)} pass={n_pass}/{len(rows)} "
                f"elapsed={elapsed:.1f}s\n")
        f.write(f"vec_busy col = {VEC_BUSY_COL}\n")

    # print the table to stdout
    print(f"\n[baseline] === summary ({args.tag}) ===", flush=True)
    print(open(summary_txt).read(), flush=True)
    print(f"[baseline] summary CSV: {summary_csv}", flush=True)
    print(f"[baseline] raw PMU CSVs: {out_dir}/*.pmu.csv", flush=True)
    return 0 if n_pass == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
