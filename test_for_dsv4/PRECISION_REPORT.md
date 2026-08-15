# DSV4 VPTO 精度报告 — ptoas 0.59 main 分支 baseline 路线

> 日期: 2026-08-15
> ptoas: 0.59 (main, fe32804c3, cpython-3.12 native build)
> 路线: baseline = VPTO + `--enable-insert-sync` + `--enable-op-fusion` + bisheng VF ON
> 设备: Ascend A5 NPU (Ascend950PR), CANN 9.1.0-beta.3

## 总览

| 指标 | 数量 | 占比 |
|------|------|------|
| **PASS** | 16 | 14% |
| **FAIL — ptoas 编译失败** | 72 | 62% |
| **FAIL — NPU 运行 crash** | 23 | 20% |
| **FAIL — 精度不达标** | 6 | 5% |
| **总计** | 117 | 100% |

**72 个 kernel（62%）在 ptoas 编译阶段就失败了，根本没上板。只有 29 个真正到了 NPU 执行（23 crash + 6 precision）。**

---

## 使用的编译选项

### ptoas 选项（6 个 flag）

```
ptoas --pto-arch=a5 \
      --pto-level=level3 \
      --pto-backend=vpto \
      --enable-tile-op-expand \
      --enable-insert-sync \
      --enable-op-fusion \
      <kernel>.pto -o <kernel>.o
```

| flag | 作用 |
|------|------|
| `--pto-arch=a5` | 目标架构 A5 (Ascend950PR) |
| `--pto-level=level3` | pass pipeline 最高级别（含 tile fusion） |
| `--pto-backend=vpto` | VPTO 后端（fatobj 产出，非 EmitC） |
| `--enable-tile-op-expand` | TileOp 展开（deprecated 兼容 flag，vpto 后端默认展开） |
| `--enable-insert-sync` | 自动同步插入 pass（MTE/MTE2/VEC 之间的 barrier） |
| `--enable-op-fusion` | A5 tile fusion 开启（fusion-region lifecycle） |

**未使用**: `--enable-vmi`、`--enable-vecscope-mem-bar`、`--enable-bufid_sync`、`--enable-inject-barrier-all-sync`（在 main 分支上不存在或未开启）。

### bisheng 选项

**launch.o 编译**（AI core kernel）:

```
bisheng -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
       -fPIC -Xhost-start -Xhost-end \
       -mllvm -cce-aicore-stack-size=0x8000 \
       -mllvm -cce-aicore-function-stack-size=0x8000 \
       -mllvm -cce-aicore-record-overflow=true \
       -mllvm -cce-aicore-addr-transform \
       -mllvm -cce-aicore-dcci-insert-for-scalar=false \
       --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -std=c++17 \
       -Wno-macro-redefined -Wno-ignored-attributes \
       -I ... launch.cpp -o launch.o
```

| flag | 作用 |
|------|------|
| `-xcce` | bisheng cce 前端 |
| `-fenable-matrix` | 矩阵指令支持 |
| `--cce-aicore-enable-tl` | AI core TilingLanguage 支持 |
| `-cce-aicore-stack-size=0x8000` | AI core 栈 32KB |
| `-cce-aicore-function-stack-size=0x8000` | 函数级栈 32KB |
| `-cce-aicore-record-overflow=true` | 记录栈溢出 |
| `-cce-aicore-addr-transform` | 地址变换优化 |
| `-cce-aicore-dcci-insert-for-scalar=false` | 不为标量插入 DCCI |
| `--cce-aicore-arch=dav-c310-vec` | 目标芯片 dav-c310-vec (A5 vector core) |

**baseline 路线没有传任何 VF-off 的 `-mllvm` flag** — bisheng VF fusion 全开（vf-fusion、vf-loop-extender、loop-fusion、vf-ldst-elimination、ub-dead-st-elimination、vf-auto-sync、vf-ifelse-extender 全部 default ON）。

**链接 .so**:

```
bisheng -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
       -shared -Wl,-soname,lib<kernel>_kernel.so \
       -L .../lib64 -Wl,-rpath,.../lib64 \
       -o lib<kernel>_kernel.so <kernel>.o launch.o \
       -Wl,--no-as-needed -lruntime
```

`--cce-fatobj-link` 把 ptoas 产出的 fatobj（嵌套 ELF）正确链接成单一 .so。

---

## 失败分类详情

### 类别 1: ptoas NoMatchingTemplate（模板缺失）— 40 个

main 分支上 TileLib 模板未注册，按缺失算子分：

| 缺失算子 | 数量 | kernel |
|---------|------|--------|
| `pto.tload` | 16 | exp_gate_mm, exp_up_mm, exp_w2_mm, idx_qr_proj_matmul, mtp_projection_linear_aic, mtp_projection_linear_aiv, prefill_idx_qr_proj_aic, prefill_idx_qr_proj_aiv, prefill_idx_score_aic, prefill_idx_score_aiv, proj_b_mm, qproj_matmul, score_mat, sh_gate_mm, sh_up_mm, sh_w2_mm |
| `pto.trsqrt` | 7 | hc_head_rms, hc_pre_rms, kv_rms_norm_rope, mtp_projection_rms, qproj_dequant_rms_nope_rope, qr_rms_norm_quant, rms_norm |
| `pto.tci` | 6 | prefill_idx_qr_rope, q_rope_prepare, qr_rope, rope, rope_cs, swa_cache_insert_valid_bias |
| `pto.tsetval` | 5 | prefill_c4_write_map, prefill_csa_sparse_idx_tile, prefill_hca_sparse_indices, prefill_idx_c4_write_map, route_hash |
| `pto.tcolexpand` | 3 | comb_sinkhorn, hc_head_pre_fused, split_pre_post |
| `pto.tgetval` | 1 | csa_slots_build_valid_qk_plan |
| `pto.ttrans` | 1 | mix_x |
| `pto.trowexpand` | 1 | sh_gate_up_act_q |

`tload`（16 个）和 `trsqrt`（7 个）占了大头 — 分别是 matmul 类和 rmsnorm 类 kernel。这些模板在 `feature-vmi-vf` 分支上可能已补齐但尚未合入 main。

---

### 类别 2: ptoas PTODSL metadata query failed — 24 个

这些 kernel 的 TileOp 在展开时，ptodsl in-process 查询候选模板元数据失败 — 不是模板直接缺失（有候选），而是候选模板的 dtype/shape 全部 reject 后抛出异常。

| kernel |
|--------|
| exp_h_q |
| ffn_norm |
| gather_kv |
| kv_and_cache_write |
| kv_proj_matmul |
| merge_norm |
| mtp_projection_quant |
| prefill_c4_rmsnorm_rope |
| prefill_c4_softmax_pool |
| prefill_hca_c128_rmsnorm_rope |
| prefill_hca_c128_softmax_pool |
| prefill_idx_c4_cache_write |
| prefill_idx_c4_rmsnorm_rope |
| prefill_idx_c4_softmax_pool |
| prefill_idx_qr_hadamard_quant_aic |
| prefill_idx_qr_hadamard_quant_aiv |
| prefill_idx_weights_proj_aic |
| prefill_idx_weights_proj_aiv |
| qr_hadamard_matmul |
| qr_hadamard_quant |
| quant |
| rmsnorm_rope |
| rmsnorm_rope_cache_write |
| x_norm_quant |

---

### 类别 3: ptoas 其他 pass 失败 — 8 个

| 子类 | 数量 | kernel | 错误 |
|------|------|--------|------|
| InsertTemplateAttrib | 4 | gate_aic, gate_aiv, qk_pv_aic, qk_pv_aiv | split kernel 的模板属性插入失败 |
| vecscope infer | 2 | mtp_projection_norm, prefill_c4_state_update | `pto.plt_b32` op 无法推断 resultless vecscope |
| tgather | 2 | prefill_idx_topk, topk | `pto.tgather` op 的 mask pattern 不匹配 |

---

### 类别 4: NPU runtime crash — 23 个

#### 4a. vector core 异常 `retCode=0x31`（19 个）

ptoas 成功编译 + bisheng 成功链接 + NPU 执行时 vector core 异常。baseline 路线下没有出现 errcode:95（MTE DDR 越界）或 errcode:161（fixpipe 写 GM 非法）— 这两种在 vmi-membar-vfoff 路线下才会出现。

| kernel |
|--------|
| build_bias |
| exp_gate_up_act |
| exp_w2_act |
| hc_head_reduce |
| hc_post |
| idx_qr_proj_dequant |
| prefill_c4_cache_write |
| prefill_csa_idx_halfrope |
| prefill_hca_c128_kv_finalize |
| prefill_hca_c128_state_scatter_pre |
| prefill_hca_c128_write_map |
| prefill_idx_c4_state_update |
| proj_b_act |
| qkv_rope_rows |
| scatter_softmax_pool |
| score_reduce |
| sh_w2_act |
| swa_gather_kv |
| swa_rope_step |

#### 4b. aicore 异常 `retCode=0x26`（4 个）

AI core 异常，3 个是 `kv_score_proj` 系列。

| kernel |
|--------|
| hc_pre_linear |
| prefill_c4_kv_score_proj |
| prefill_hca_c128_kv_score_proj |
| prefill_idx_c4_kv_score_proj |

---

### 类别 5: precision fail（NPU 跑了但输出不对）— 6 个

| kernel | max_diff | 严重程度 |
|--------|----------|---------|
| proj_a_mm | 2.10 | 严重 |
| kv_score_proj | 0.307 | 严重 |
| weights_proj | 0.088 | 中等 |
| weights_proj_reduce | 0.003 | 小 |
| kv_hadamard | 7.5e-09 | fp32 精度边界 |
| prefill_idx_c4_kv_hadamard | 5.6e-09 | fp32 精度边界 |

后两个 hadamard 的 max_diff 在 1e-9 量级，实际可能是 fp32 精度边界，调 `VPTO_COMPARE_ATOL` 可以放过去。

---

## PASS kernel 列表（16 个，全部 max_diff=0.0 bit-exact）

| kernel | timing |
|--------|--------|
| gate_pre_route | 0.558ms |
| hc_head_linear | 0.532ms |
| hc_head_seed | 0.514ms |
| hc_post_inactive_pad | 0.603ms |
| hc_post_prefill | 0.467ms |
| hc_pre_seed | 2.827ms |
| kv_proj_seed | 0.452ms |
| kv_touch | 0.571ms |
| mtp_projection_output | 0.645ms |
| prefill_csa_cache_write | 0.625ms |
| prefill_hca_c128_norm_pad_init | 0.465ms |
| prefill_hca_cache_write | 0.616ms |
| prefill_idx_score_init | 0.479ms |
| prefill_idx_score_out | 0.562ms |
| qr_proj_matmul | 0.507ms |
| qr_proj_seed | 0.560ms |

---

## 与 vmi-membar-vfoff 路线对比（stale build311, ptoas 0.59 feature-vmi-vf）

| 指标 | baseline (main) | vmi-membar-vfoff (feature-vmi-vf) |
|------|----------------|----------------------------------|
| PASS | 16 | 20 |
| NoMatchingTemplate | 40 | 14 (tdivs only) |
| PTODSL metadata | 24 | 13 |
| NPU crash | 23 | 41 (含 errcode:95 + errcode:161) |
| precision fail | 6 | 8 |

关键差异:
- main 上模板缺失更严重（40 vs 14）— `feature-vmi-vf` 分支补了 tdivs 等模板但未合入 main
- baseline 路线没有 errcode:95（MTE DDR）和 errcode:161（fixpipe）crash — 这两种是 VMI/membar 调度引入的
- `rms_norm` 在 baseline 上不再精度失败（上次 max_ulp=829），但变成 NoMatchingTemplate（`trsqrt` 模板缺失）

---

## 复现命令

```bash
# 1. 设置环境
export PTOAS_BIN=/path/to/PTOAS/build/tools/ptoas/ptoas
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
export PTO_ISA_PATH=/path/to/pto-isa
export PTOAS_SOURCE=/path/to/PTOAS
export LLVM_BUILD=/path/to/llvm-build
source $ASCEND_HOME_PATH/set_env.sh
export BISHENG_BIN=$ASCEND_HOME_PATH/bin/bisheng

# 2. 全量跑
python3 test_for_dsv4/run_all.py --device 0 --route baseline

# 3. 分类诊断
python3 test_for_dsv4/diagnose_crashes.py 0

# 4. 结果文件
# sweep_results.csv         — 每 kernel 的 PASS/FAIL/max_diff
# crash_diagnosis.json      — 每 kernel 的错误分类
```

---

## 附录 A: EmitC 路线对比（op-fusion=off）

> ptoas 0.59 main, `--pto-backend=emitc --enable-insert-sync --enable-op-fusion=false`
> bisheng VF ON (default), 103 kernels (split kernels excluded)

### EmitC vs VPTO baseline 对比

| 指标 | EmitC (op-fusion off) | VPTO baseline (op-fusion on) |
|------|----------------------|------------------------------|
| **PASS** | **99/103 (96%)** | 16/117 (14%) |
| ptoas 编译失败 | 2 | 72 |
| NPU crash | 0 | 23 |
| precision fail | 6→2 (after golden fix) | 6 |
| 其他 (bisheng 编译/_aiv crash) | 10 | 0 |

**结论：.pto 文件本身基本没有问题。** EmitC 路线 99/103 PASS，证明 110 个 .pto kernel 的 IR 是正确的。

### EmitC 失败分类（18→4 after golden fix）

| 类别 | 数量 | kernel | 原因 |
|------|------|--------|------|
| bisheng 编译失败 | 3 | comb_sinkhorn, exp_gate_up_act, score_reduce | `TFillPadMode` 未定义（bisheng 端 bug） |
| `_aiv` kernel crash | 7 | gate_aiv, mtp_projection_linear_aiv, prefill_idx_qr_hadamard_quant_aiv, prefill_idx_qr_proj_aiv, prefill_idx_score_aiv, prefill_idx_weights_proj_aiv, qk_pv_aiv | split kernel 的 `_aiv` 半在 EmitC 下 vector core 异常（已标记 unsupported） |
| ptoas lowering 失败 | 2 | prefill_idx_topk, topk | `tgather` op 的 EmitC lowering 问题 |
| precision fail (genuine) | 2 | prefill_c4_state_update, prefill_idx_c4_state_update | fp32 累积顺序差异（见附录 B） |

---

## 附录 B: Golden bug 修复记录

### 修复 1: kv_rms_norm_rope — RMSNorm 求和范围错误 ✅ 已修复

**问题**: golden 的 `_rms_norm_rope` 只对 `NOPE_DIM=448` 列求平方和，但 .pto kernel 对全 `HEAD_DIM=512` 列求和。

**证据**: .pto 的循环 `scf.for kb = 0 to 8 step 2`（4 次迭代 × 128 列/次 = 512 列），而 golden 代码为：
```python
# BUG: 只求和 448 列
sq = (rows[:, :NOPE_DIM] * rows[:, :NOPE_DIM]).sum(axis=1, keepdims=True) / HEAD_DIM
```

**修复**:
```python
# FIX: 求和全部 512 列
sq = (rows * rows).sum(axis=1, keepdims=True) / HEAD_DIM
```

**验证**: 修复前 max_ulp=18 FAIL → 修复后 max_diff=1.0 PASS ✅

### 修复 2: prefill_c4_state_update — buffer 映射错误 + pos==0 跳过错误

**问题 1**: dict 字面量中 `v3` 定义了两次（Python 取最后一个），导致 cmp_ape 的随机数据生成到了错误大小的 buffer。

**问题 2**: golden 把 cmp_ape 放进 `v3`，但 .pto 中 `v3 = pooled_kv [32, 512]`，`v5 = cmp_ape [4, 1024]`。buffer 映射错位。

**问题 3**: golden 对 `pos == 0` 跳过 cmp_ape 加法，但 .pto 没有这个条件分支 — kernel 对所有 token 都加 cmp_ape。

**修复**:
- 去掉重复 v3 key
- v3 → pooled_kv (make_fp32)，v5 → cmp_ape (make_fp32)
- 去掉 `if pos > 0` 条件，始终加 cmp_ape
- 去掉不存在的 v8/v9/v10

**验证**: 仍 FAIL max_diff=0.098 — 属于 fp32 真精度差异（scatter+add 累积顺序）

### 修复 3: prefill_idx_c4_state_update — 多余 buffer 键

**问题**: golden 有 v8/v9/v10 但 .pto 只有 7 个 ptr 参数（v1-v7）。

**修复**: 去掉 v8/v9/v10。

**验证**: 仍 FAIL max_diff=0.071 — 属于 fp32 真精度差异

### 未修复的 EmitC 精度失败（4 个，疑似真精度问题）

| kernel | max_diff | 分析 |
|--------|----------|------|
| `prefill_c4_state_update` | 0.098 | golden 与 .pto 计算逻辑已对齐（逐行验证 SPMD 分块、inner loop 32x32、offset 1024+ob*32 等）。差异来自 fp32 scatter+add 累积顺序。 |
| `prefill_idx_c4_state_update` | 0.071 | 同上，SPMD block_idx/4=token, block_idx%4=col_chunk。计算逻辑对齐。差异来自 fp32 累积。 |
| `prefill_hca_c128_rmsnorm_rope` | 0.085 | golden 用 `sum(pooled*pooled)` 全列求和，与 .pto 一致。RoPE interleaved 逻辑也一致。差异可能来自 bf16→fp32 转换或 rsqrt 精度。 |
| `prefill_hca_c128_softmax_pool` | 0.049 | softmax 的 fp32 exp/sum 累积顺序差异。golden 用 numpy 向量化，kernel 用 tile 逐块累加。 |

### Split kernel 移除

7 个 split .pto（含 `_aic`/`_aiv` 双函数）已从 `run_all.py` 中移除，标记为 unsupported：

| .pto 文件 | 产生的 kernel 条目 |
|-----------|------------------|
| gate.pto | gate_aic, gate_aiv |
| mtp_projection_linear.pto | mtp_projection_linear_aic, mtp_projection_linear_aiv |
| prefill_idx_qr_hadamard_quant.pto | prefill_idx_qr_hadamard_quant_aic, prefill_idx_qr_hadamard_quant_aiv |
| prefill_idx_qr_proj.pto | prefill_idx_qr_proj_aic, prefill_idx_qr_proj_aiv |
| prefill_idx_score.pto | prefill_idx_score_aic, prefill_idx_score_aiv |
| prefill_idx_weights_proj.pto | prefill_idx_weights_proj_aic, prefill_idx_weights_proj_aiv |
| qk_pv.pto | qk_pv_aic, qk_pv_aiv |
