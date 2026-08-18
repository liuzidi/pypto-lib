# DSV4 VPTO 精度全量扫描汇总

> 两条路线 × 24 模块 × 338 kernel。报告全文见 `PRECISION_REPORT.md`。

## 1. 两条路线对比总表

| 指标 | baseline 路线 | vmi-membar-vfoff 路线 | 说明 |
|---|---:|---:|---|
| **PASS（max_diff=0，精确通过）** | 52 | 45 | vmi 少 7，是新增 UB 崩溃导致 |
| **真精度 FAIL（有 max_diff）** | 8 | 1 | vmi 修了 7/8（rms_norm 泄漏 + proj_a_mm 越界） |
| **NPU 运行崩溃（exit 1）** | 135 | 152 | vmi +17（其中 13 是 VMI UB 对齐新 bug） |
| **lowering/harvest 崩溃** | 143 | 140 | vmi -3（merge_norm 的 tdivs 被融合掉） |
| **总 kernel 数** | 338 | 338 | 24 模块全量 |

## 2. 129 个不同 kernel 的分布（按"最好成绩"分类）

| 类别 | kernel 数 | 根因 | 责任层 |
|---|---:|---|---|
| **PASS（正对照，框架链路正确）** | 17 | — | — |
| **真精度 FAIL** | 3 | local-mem 泄漏 + scalar=0 | pypto + 框架 |
| **NPU 崩溃** | 53 | inout harvest 漏 + scalar=0 + 0 字节 alloc | **框架** |
| **lowering 崩溃** | 56 | pto.tdivs 无 template + i8 未映射 + SSA | ptoas + **框架** |

## 3. 精度 FAIL 的 8 个 case（baseline 路线，全部已定位根因）

| kernel | 模块 | max_diff | 根因 | vmi 路线结果 |
|---|---|---:|---|---|
| `rms_norm` | attention_csa | 999424 | inner .pto 硬编码形状 + local-mem 地址冲突 | 泄漏消失 → UB 崩溃 |
| `rms_norm` | attention_swa | 999424 | 同上 | 同上 |
| `rms_norm` | prefill_csa/hca/swa | 999424 | 同上 | 同上 |
| `proj_a_mm` | attention_csa | 27.82 | scalar=0 → cube 越界读 | **PASS** ✅ |
| `proj_a_mm` | sparse_attn | 1.95e+38 | 同上 | **PASS** ✅ |
| `mtp_projection_rms` | mtp_projection | 1000 | scalar=0 → view 坍缩 | 不变（框架 bug） |

## 4. 286 个失败按根因分解

| 根因类 | 数量 | 责任层 | 是否框架可修 |
|---|---:|---|---|
| C1 local-mem 泄漏 + scalar=0 | 8 | pypto + 框架 | 部分 |
| C2 inout role 未 harvest | ~50 | **框架** | ✅ |
| C3 index scalar 默认填 0 | ~45 | **框架** | ✅ |
| C4 0 字节 alloc 失败 | ~30 | **框架** | ✅ |
| C5 pto.tdivs 无 A5 template | ~80 | ptoas | ❌ |
| C6 i8→int8_t 映射缺失 | ~30 | **框架** | ✅ |
| C7 SSA + aic/aiv split | ~6 | pypto + 框架 | 部分 |
| C8 0-iter SPMD | 7 | 框架 + 输入 | 部分 |

## 5. 一句话结论

**~180/286（63%）失败是框架侧 bug**（inout harvest、scalar=0、i8 映射），修完后这些行会从 crash 变成 pass 或真精度 FAIL。**~80 是 ptoas 的 tdivs template 缺口**（VMI 路线能修掉其中 3 个 merge_norm）。**8 个真精度 FAIL 里 7 个是 local-mem 泄漏，VMI+membar 路线能修掉**，但该路线引入 13 个新的 UB 对齐崩溃（待 ptoas 修）。
