# DSV4 VPTO Precision Test Suite

End-to-end precision validation for DSV4 (DeepSeek-V4 Pro) VPTO kernels on Ascend A5 NPU.
Each kernel has a **pure-numpy CPU golden reference** — no torch, no GM dump, no NPU capture needed.

## Structure

```
test_for_dsv4/
├── .pto/                        # 110 DSV4 .pto kernel files (MLIR pto IR)
├── dsv4_golden_lib.py           # BUILDERS loader + run_case(name)
├── golden_parts/                # per-module golden build_ functions (pure numpy)
│   ├── attention_kernels.py     #   19 builders (CSA/HCA/SWA attention + rms/rope)
│   ├── hc_gate_moe_kernels.py   #   28 builders (hc_pre/hc_post/hc_head + gate/MoE)
│   ├── prefill_misc_kernels.py  #   43 builders (prefill variants + misc)
│   └── qkv_mtp_compress_kernels.py  # 25 builders (qkv_proj/indexer/mtp_projection)
├── golden/
│   ├── validation_runtime.py    # CaseMeta, rng (seed=19), write_buffers, bf16 helpers
│   └── compare.py               # bf16 ULP + fp32/int compare (tolerance via env vars)
├── lib/                         # symlinks → ../../.claude/skills/vpto-board-validate/lib/
├── run_all.py                   # one-command sweep driver
├── diagnose_crashes.py         # re-run + classify every failure by error type
├── emitc_run.py                 # EmitC route runner (alternative to VPTO)
├── parallel_emitc_sweep.py     # multi-device parallel EmitC sweep
├── setup_all.sh                # orchestrator: generate run dirs
├── kernel_signatures.json       # parsed .pto signatures (reference snapshot)
├── kernel_outputs.json          # output buffer name lists (reference snapshot)
└── README.md
```

## Prerequisites

You need an Ascend A5 NPU machine with CANN, bisheng, and ptoas installed.

Set these environment variables before running:

| Variable | Description | Example |
|----------|-------------|---------|
| `PTOAS_BIN` | Path to ptoas binary | `/path/to/PTOAS/build/tools/ptoas/ptoas` |
| `ASCEND_HOME_PATH` | CANN install root | `/usr/local/Ascend/cann-9.1.0-beta.3` |
| `PTO_ISA_PATH` | Path to pto-isa repo | `/path/to/pto-isa` |
| `BISHENG_BIN` | Path to bisheng compiler | `$ASCEND_HOME_PATH/bin/bisheng` |
| `PTOAS_SOURCE` | Path to PTOAS source tree (for ptodsl) | `/path/to/PTOAS` |
| `LLVM_BUILD` | Path to LLVM build (for LD_LIBRARY_PATH) | `/path/to/llvm/build` |

Optional (only for ptoas < 0.54):

| Variable | Description |
|----------|-------------|
| `TILELANG_PATH` | Path to TileOps directory |
| `TILELANG_PKG` | Path to tilelang-dsl/python |

You can set them manually or source your own env script:

```bash
export PTOAS_BIN=/path/to/PTOAS/build/tools/ptoas/ptoas
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
export PTO_ISA_PATH=/path/to/pto-isa
export PTOAS_SOURCE=/path/to/PTOAS
export LLVM_BUILD=/path/to/llvm-build
source $ASCEND_HOME_PATH/set_env.sh
export BISHENG_BIN=$ASCEND_HOME_PATH/bin/bisheng
```

ptoas 0.59+ is cpython-3.12 native — system `python3` works directly.
For ptoas < 0.59 (cpython-3.11), you need a shim that runs `python3.11`.

## Quick Start

```bash
# 1. Set your environment (see Prerequisites above)

# 2. Run all 110 kernels on device 0 (baseline route: VPTO + insert-sync + op-fusion, bisheng VF ON)
python3 test_for_dsv4/run_all.py --device 0 --route baseline

# 3. Run a single kernel
python3 test_for_dsv4/run_all.py --device 0 --kernel rms_norm

# 4. Results are written to test_for_dsv4/sweep_results.csv
```

## How it works

For each kernel:

1. `vpto_run.py --golden-lib` calls `dsv4_golden_lib.run_case("<kernel>")`
2. `run_case` generates random inputs (seeded numpy RNG, seed=19) + computes golden output **on CPU**
3. Writes `vN.bin` (inputs) + `golden_vN.bin` (expected outputs) — no NPU needed for golden
4. `vpto_run.py` runs ptoas (VPTO backend) → bisheng → NPU execute
5. Compares NPU output vs golden (bf16 ULP or fp32 atol)

The golden is deterministic (seeded RNG) and self-contained (no torch, no NPU dependency).

## Route flags

| Route | ptoas flags | bisheng flags | Notes |
|-------|------------|---------------|-------|
| `baseline` (default) | `--enable-op-fusion --enable-insert-sync` | VF-fusion on (default) | Works on main branch |
| `vmi-membar-vfoff` | `--enable-vmi --enable-vecscope-mem-bar` | 7× `-mllvm -cce-vf-enable-*=false` | Requires feature-vmi-vf branch |

## Compare tolerance

| Env var | Default | Effect |
|---------|---------|--------|
| `VPTO_COMPARE_MAX_ULP` | 3 | Max bf16 ULP difference allowed |
| `VPTO_COMPARE_ATOL` | 1e-3 | fp32 absolute tolerance |

## Diagnosing failures

```bash
# Re-run every kernel individually with full stderr, classify by error type
python3 test_for_dsv4/diagnose_crashes.py 0

# Output: crash_diagnosis.json with per-kernel classification
# Categories: pass, precision_fail, npu_crash_mte_ddr, npu_crash_fixpipe,
#   npu_crash_vec_other, npu_crash_aic_other, ptoas_tdivs, ptoas_metadata,
#   ptoas_failed, ptoas_pass_failed, ptoas_expand, timeout, other
```

## Latest results (baseline route, ptoas 0.59 main, Aug 2026)

**16/117 PASS** (all bit-exact, max_diff=0.0):

`gate_pre_route`, `hc_head_linear`, `hc_head_seed`, `hc_post_inactive_pad`, `hc_post_prefill`, `hc_pre_seed`, `kv_proj_seed`, `kv_touch`, `mtp_projection_output`, `prefill_csa_cache_write`, `prefill_hca_c128_norm_pad_init`, `prefill_hca_cache_write`, `prefill_idx_score_init`, `prefill_idx_score_out`, `qr_proj_matmul`, `qr_proj_seed`

Failure breakdown (101 FAIL):

| Category | Count | Description |
|----------|-------|-------------|
| ptoas_tdivs | 40 | `NoMatchingTemplate: op='pto.tdivs'` — tdivs templates missing on main |
| ptoas_metadata | 22 | PTODSL metadata query failed — tilelib template matching |
| npu_vec_other | 19 | `retCode=0x31` vector core exception |
| other | 10 | `InsertTemplateAttrib` / `vecscope` / `tgather` pass errors |
| precision_fail | 6 | NPU ran but max_diff exceeded threshold |
| npu_aic_other | 4 | `retCode=0x26` aicore exception |
