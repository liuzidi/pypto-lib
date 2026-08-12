# DSV4 VPTO 精度问题调查 + 修复交接文档

> **目的**：把 338-kernel 全量扫描发现的所有问题按"背景 → 复现 →
> 根因 → 修复方案"分类写清楚，让接手的人能独立定位和修复每个问题。
>
> **配套文件**：
> - 完整调查报告：`PRECISION_REPORT.md`
> - baseline 扫描结果：`sweep_results.csv`（338 行）
> - vmi-membar-vfoff 扫描结果：`sweep_results_vmi-membar-vfoff.csv`（338 行）
> - 汇总表：`SUMMARY_TABLE.md`

---

## 0. 背景知识

### 0.1 VPTO 编译路线（Route 2）

DSV4 kernel 从 pypto 的 `.pto` 到 NPU 执行经过三步：

```
.pto (pypto IR)  ──ptoas──▶  fatobj .o (LLVM bitcode + 嵌套 ELF)
                             ──bisheng──▶  .so + host binary
                                          ──CANN──▶  NPU 执行
```

- **ptoas**：把 `.pto`（tile dialect IR）编译成 A5 fatobj。内部走 ptodsl
  daemon 的模板匹配（NoMatchingTemplate = 该 op 在 A5 上没有模板）。
- **bisheng**：CANN 的编译器，把 fatobj + launch.cpp 链接成可执行 .so
  + host binary。支持 VF（Vector Fusion）后端优化。
- **CANN**：`aclrtMalloc` / `aclrtMemcpy` / `aclrtSynchronizeStream` 等
  runtime API，host binary 通过它们驱动 NPU。

### 0.2 测试框架架构

```
                          Route 1 capture (DFX)           Route 2 replay (board)
  ┌──────────┐   run_jit w/   ┌──────────────┐  harvest  ┌──────────────┐  ptoas+bisheng  ┌─────┐
  │ model.py │ ───────────▶ │ args_dump.json│ ────────▶ │ vN.bin /      │ ──────────────▶ │ NPU │ → compare
  │ (@pl.jit)│   enable_     │ name_map_*.   │  (capture │ golden_vN.bin │                 │     │
  └──────────┘   dump_args=2 │  json         │  .py)     │ capture_meta  │                 └─────┘
                 +dep_gen     └──────────────┘           └──────────────┘
```

- **Route 1 capture**：用 simpler 的 `run_jit` 跑模型，开 `enable_dump_args=2`
  + `enable_dep_gen=True`，把每个 kernel 的 GM 输入/输出 dump 到
  `args.bin` + `args_dump.json`。
- **harvest**：`capture.py` 从 dump 里按 kernel name + func_id 提取单个
  kernel 的输入/输出字节，写成 `vN.bin` / `golden_vN.bin`。
- **Route 2 replay**：`vpto_run.py` 用同一份 `.pto` 走 ptoas→bisheng→NPU，
  喂入 harvest 出来的 `vN.bin`，NPU 输出和 `golden_vN.bin` 比对。

### 0.3 关键概念

| 术语 | 含义 |
|---|---|
| **leaf kernel** | ptr-arg 与 module TensorSpec 一一对应的 kernel（可直接从模型生成 golden） |
| **inner kernel** | ptr-arg 是前序 kernel 的中间结果，需要 Phase 5 capture |
| **SPMD** | Single Program Multiple Data，一个 kernel 多 block 并行 |
| **inout ptr** | 同一个 GM buffer 既是输入又是输出（read-modify-write） |
| **UB** | Unified Buffer，A5 的 local memory（vector core 私有） |
| **membar** | vecscope memory barrier，ptoas 的 `--enable-vecscope-mem-bar` 插入的同步 |
| **VMI** | Vector Memory Interface fusion，ptoas 的 `--enable-vmi` 融合流水线 |
| **VF-fusion** | bisheng 的 Vector Fusion 后端优化（可关） |

### 0.4 如何运行

```bash
# 前置：激活 VPTO 环境
source scripts/vpto_env.sh
export PTOAS_ROOT=$(dirname $PTOAS_BIN)

# 全量扫描（baseline 路线）
.venv/bin/python3 tests/dsv4_validate/validate.py --all-modules -d 0 --route baseline

# 全量扫描（vmi-membar-vfoff 路线）
.venv/bin/python3 tests/dsv4_validate/validate.py --all-modules -d 0 --route vmi-membar-vfoff

# 单模块调试
.venv/bin/python3 tests/dsv4_validate/validate.py --module attention_csa -d 0 --route baseline

# 单 kernel 直接调用 vpto_run（绕过 validate.py 框架）
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_*/ptoas/merge_norm.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel merge_norm --route baseline \
  --captured-dump build_output/vpto_merge_norm/run
```

### 0.5 问题总览

338 个 (module, kernel) 尝试，129 个不同 kernel basename。286 个非 pass 行
分解为 **8 个根因类**，按责任层分两组：

| 责任层 | 根因类 | 数量 | 修了能转化多少行 |
|---|---|---:|---:|
| **测试框架** | C2/C3/C4/C6/C8 | ~180 | crash→pass 或 crash→真精度 FAIL |
| **pypto/ptoas** | C1/C5/C7 + VMI UB | ~106 | 需要在编译器侧修 |

> **当前状态**：框架侧 C2/C3/C4/C6/C7b/C8b 全部修完。pypto 侧剩
> C1a/C7a，ptoas 侧剩 VMI-UB（C5 已在 ptoas 源树修完）。
> 注：C7 实际跨两层——C7a（SSA dominance）在 pypto，C7b（aic/aiv split
> 按 func 锚定）在框架已修。

---

## C1 — 精度 FAIL：local-mem 泄漏 + scalar=0（8 行）

### 背景

这是唯一一类"kernel 跑完了但输出错"的真精度问题。8 个 case 共 3 个
kernel basename：

- `rms_norm`（5 行，attention_csa/swa + prefill_csa/hca/swa）
- `proj_a_mm`（2 行，attention_csa + sparse_attn）
- `mtp_projection_rms`（1 行，mtp_projection）

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)

# rms_norm（inner kernel，来自 attention_csa 模块）
# 检查输入/输出
.venv/bin/python3 -c "
import numpy as np
v2 = np.fromfile('build_output/vpto_rms_norm/run/v2.bin', dtype=np.uint16)
gv2 = np.fromfile('build_output/vpto_rms_norm/run/golden_v2.bin', dtype=np.uint16)
print(f'NPU out nonzero: {np.count_nonzero(v2)}, golden nonzero: {np.count_nonzero(gv2)}')
# 预期：NPU 有 7 个非零（1.0, 1000.0, 999424.0），golden 全零
"

# mtp_projection_rms
.venv/bin/python3 -c "
import numpy as np
v3 = np.fromfile('build_output/vpto_mtp_projection_rms/run/v3.bin', dtype=np.float32)
print(f'NPU v3: {v3[:16]}')  # 预期：idx 8-15 是 1000.0
"
```

### 根因

**两个不同的 bug，碰巧产生同一个指纹（输入全零 → NPU 泄漏中间值）。**

#### C1a — `rms_norm` family（5 行）：pypto inner-kernel 硬编码形状 + local-mem 地址冲突

**文件**：pypto 的 inner-kernel codegen（不是本仓库代码）

**根因**：
- standalone leaf `rms_norm` 的 `.pto` 用动态行维 `shape = [%arg3, ...]`
  （caller 传入），PASS。
- inner-kernel `rms_norm` 的 `.pto` 把行维硬编码成 `shape = [%c128_index, ...]`
  并删掉了 `%arg3` 参数。fatobj 不同（8712B vs 8568B）。
- 硬编码 128 行 + 单 block（spmd_block_num=1）→ 只有 8/128 行被写入，
  其余 120 行的 output tile 在 local memory（UB）里与 fp32 reduction 中间值
  共用同一地址（`addr=8768`），bf16 tstore 只写 2 bytes/elem，fp32 残留的
  高 2 bytes 泄漏成 bf16 输出。
- 泄漏值：`1.0`(0x3F80)、`1000.0`(0x447A = `rsqrt(1e-6)`)、
  `999424.0`(0x4974 = stale reduction accumulator)
- 位置：flat idx 256/257/258/259/512/513/768 — 128-col tile 的起始位置

**diff 验证**：
```bash
diff build_output/_jit_rms_norm_test_*/ptoas/rms_norm.pto \
     build_output/vpto_rms_norm/rms_norm.pto
# 关键差异：
# - shape = [%arg3, %c7168_index]        ← leaf：动态行维（PASS）
# + shape = [%c128_index, %c7168_index]  ← inner：硬编码 128（FAIL）
```

#### C1b — `mtp_projection_rms` + `proj_a_mm`（3 行）：框架 scalar_sem=0 bug

**文件**：
- `.claude/skills/vpto-board-validate/vpto_run.py:402-415`（scalar_sem 推导）
- `.claude/skills/vpto-board-validate/lib/setup_main.py:182`（默认填 0）

**根因**：
- `vpto_run.py` 的 scalar_sem 推导只把**第一个**非 SPMD index 标为 `ctx_len`，
  其余 index 标为 `None`。
- `setup_main.py:182` 对 `None` 语义的 scalar 默认填 `0`：
  ```python
  param_decls.append(f"    {scal_type} {s['name']} = 0;  // FIXME: {hint}")
  ```
- `mtp_projection_rms` 的 `.pto` 有 3 个 index scalar（`%arg4`=ctx_len,
  `%arg5`=行数, `%arg6`=列数），但 `main.cpp` 生成 `v5=8`（正确）、
  `v6=0`、`v7=0`（错误）。tensor view `shape=[%arg5=0, ...]` 坍缩为 0 行 →
  SPMD 循环不写任何元素 → `rsqrt(eps)=1000.0` 中间值泄漏到 output。
- `proj_a_mm` 同理：4 个 index scalar，只第一个被填对，其余 3 个填 0 →
  cube kernel 越界读 → NaN / 1.95e+38。

### 修复方案

#### C1a（pypto 侧，5 行）

路由到 **pypto**。inner-kernel codegen 应保持动态行维（像 leaf 变体那样），
或者在硬编码形状时确保 output tile 的 local-mem 地址与 fp32 中间值 tile
不冲突。这需要 pypto 侧的 codegen 修复，不在本仓库。

#### C1b（框架侧，3 行）

路由到 **测试框架**。`vpto_run.py` 的 scalar_sem 推导应从 `.pto` 的
`make_tensor_view` shape 维度推导每个 index scalar 的语义，而不是只标第一个：

```python
# 当前（vpto_run.py:407-413，有 bug）：
for p in info["params"]:
    if p["pto_type"] in ("i32", "index"):
        sig = p.get("sig_name", "") or p["name"]
        if not ctx_marked and "spmd" not in sig:
            scalar_sem.append("ctx_len")     # ← 只有第一个
            ctx_marked = True
        else:
            scalar_sem.append(None)          # ← 其余 → setup_main 填 0

# 修复方向：解析 .pto 的 make_tensor_view shape，把每个 index 参数
# 匹配到它对应的 tensor 维度大小（从 capture_meta elem_counts 反推），
# 或从 args_dump.json 的 shape 字段直接读
```

**验证**：修完后 `mtp_projection_rms` 应该 PASS（输入全零 → 输出全零，
不再泄漏 1000.0）。`proj_a_mm` 应该 PASS 或变成真精度 FAIL（取决于 cube
kernel 在正确索引下是否精确）。

---

## C2 — inout role 未 harvest → vN.bin 缺失（~50 行）

### 背景

`capture.py` 的 harvest 逻辑只匹配 `role=="input"` 和 `role=="output"`，
不匹配 `role=="inout"`。很多 kernel 有 inout ptr（同一个 GM buffer
既读又写），这些 arg 的 dump 记录被完全跳过 → `vN.bin` 不生成 →
`main.cpp` 的 `ReadFile3("./vN.bin")` 失败 → NPU host binary 退出 1。

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)

# comb_sinkhorn（attention_csa，decode）— 典型 inout kernel
.venv/bin/python3 -c "
import json, glob
DUMP='build_output/_jit_attention_csa_test_20260811_224123'
d = json.load(open(f'{DUMP}/dfx_outputs/args_dump/args_dump.json'))
args = d['args']
nm = json.load(open(glob.glob(f'{DUMP}/dfx_outputs/name_map_*.json')[0]))
fid = [int(k) for k,v in nm['callable_id_to_name'].items() if v=='comb_sinkhorn'][0]
recs = [a for a in args if fid in a['func_id']]
from collections import Counter
print('roles:', Counter(a['role'] for a in recs))
# 预期：role=inout（arg3），但 harvest 只找 role=input/output
# → arg3 被跳过 → v4.bin 不生成
"

# 运行 vpto_run 看错误
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_20260811_224123/ptoas/comb_sinkhorn.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel comb_sinkhorn --route baseline \
  --captured-dump build_output/vpto_comb_sinkhorn/run \
  --build-dir build_output/_probe_comb_sinkhorn 2>&1 | grep "Failed to read\|v4.bin"
# 预期：Failed to read v4.bin / Failed to get file. Path = ./v4.bin
```

### 根因

**文件**：`tests/dsv4_validate/capture.py:155-160`

```python
# 当前（有 bug）：
if a["role"] == "input" and a["stage"] == "before_dispatch":
    if ai not in inputs:
        inputs[ai] = a
elif a["role"] == "output" and a["stage"] == "after_completion":
    if ai not in outputs:
        outputs[ai] = a
# ← 没有 role=="inout" 的分支！
```

dump 里的 role 有三种：`input`、`output`、`inout`。`inout` 的 arg 在
`before_dispatch` 阶段有输入快照，在 `after_completion` 阶段有输出快照。
harvest 应该把 `inout` 当作既 input 又 output。

### 修复方案

路由到 **测试框架**。`capture.py:155-160` 加 `inout` 分支：

```python
# 修复：
if a["role"] in ("input", "inout") and a["stage"] == "before_dispatch":
    if ai not in inputs:
        inputs[ai] = a
if a["role"] in ("output", "inout") and a["stage"] == "after_completion":
    if ai not in outputs:
        outputs[ai] = a
```

**受影响 kernel**：`comb_sinkhorn`、`kv_score_proj`、`kv_score_proj_0`、
`kv_touch`、`gate_pre_route`、`hc_head_seed` 等所有含 inout ptr 的 kernel。

**验证**：修完后 `comb_sinkhorn` 应该能读到 `v4.bin`，然后要么 PASS
要么变成真精度 FAIL（取决于 kernel 本身是否正确）。

---

## C3 — index scalar 默认填 0 → tensor view 坍缩（~45 行）

### 背景

与 C1b 同一个 bug，但这里表现为 NPU 崩溃而非精度 FAIL。多个 index
scalar 的 kernel 中，只有第一个被填对（ctx_len），其余填 0，导致
tensor view 维度坍缩 → SPMD 循环不执行 → AICore 异常或空输出。

### 复现

```bash
# mtp_projection_rms 的 main.cpp — 看 scalar 赋值
grep "int64_t v\|FIXME" build_output/vpto_mtp_projection_rms/run/main.cpp
# 预期：
#   int64_t v5 = 8;   // ctx_len（正确）
#   int64_t v6 = 0;   // FIXME: sig=arg5（应为行数，被填 0）
#   int64_t v7 = 0;   // FIXME: sig=arg6（应为列数，被填 0）
```

### 根因

**文件**：
- `.claude/skills/vpto-board-validate/vpto_run.py`（scalar_sem 推导，
  只标第一个非 SPMD index 为 ctx_len）
- `.claude/skills/vpto-board-validate/lib/setup_main.py`（None → 填 0）

```python
# setup_main.py（旧，有 bug）：
param_decls.append(f"    {scal_type} {s['name']} = 0;  // FIXME: {hint}")
```

### 修复方案（已实施）

路由到 **测试框架**。✅ 已完整实施，采用 `derived` + `dumped` 双路径，
按优先级 `derived > dumped > ctx_len(首个非spmd) > None(=0)` 解析每个
index scalar：

1. **`derived`**（commit `aefe714`）——从 `.pto` 的 `make_tensor_view`
   shape 推导：解析 `pto.make_tensor_view %argN, shape=[%argM, %cK, ...]`，
   对每个 `%argM: index`，从 `capture_meta.json` 的 `elem_counts` 反推
   `value = elem_count // product(other static dims)`。覆盖出现在 tensor
   view shape 维度里的 scalar（如 `mtp_projection_rms` 的 `%arg5`/`%arg6`、
   `hc_pre_linear` 的 `%arg4`/`%arg5`）。
2. **`dumped`**——从 args_dump 的 `value` 字段捕获运行时 scalar 值。覆盖
   **不在任何 tensor_view shape 里**的 scalar（partition_view offsets、
   loop bounds、RNG seeds 等），如 `route_hash/v6`、`comb_sinkhorn/v5`、
   `split_pre_post/v6,v7`（13 个残余 case）。
   - `capture.py:harvest_kernel` 从 args_dump 的 `kind=="scalar"` 记录读
     `value` 字段，写入 `capture_meta.json` 的 `scalar_values {vN: int}`。
   - `vpto_run.py` 读 `scalar_values` → 标 sem=`"dumped"`。
   - `setup_main.py` 在 `sem=="dumped"` 时 emit 真实值。
3. **`ctx_len`**——首个非 SPMD scalar 仍走 `ctx_len`（= `B*S`，从 config.py
   读），覆盖 trailing `T_DYN` index。
4. **`None`**——其余（SPMD `block_idx`/`block_num` 等）走 setup_main
   特殊分支：`spmd_block_num → 1`（单 block 启动），其余 → `0`。

**验证**：含多个 index 的 kernel（`mtp_projection_rms`、`hc_pre_linear`、
`route_hash`、`comb_sinkhorn` 等）不再因 view 坍缩而崩溃。`validate.py`
在每次 sweep 时重新生成 `capture_meta.json`（含最新 `scalar_values`），
无 stale 数据风险。

---

## C4 — output-only ptr 的 0 字节 alloc → aclrtMallocHost 失败（~30 行）

### 背景

`setup_main.py` 从 `capture_meta.json` 的 `elem_counts` 计算 buffer 大小
`fileSize = elemCount × sizeof(dtype)`。当一个 output-only ptr 的
`numel=0`（dump 记录里 shape=[] 或 elem_counts 没有该 ptr），`fileSize=0`
→ `aclrtMallocHost(0)` 失败：

```
aclrtMallocHost failed: 100000
Invalid_Argument(EH0007): aclrtMallocHostImpl failed because value 0
for parameter size is invalid. Expected value: must be greater than zero.
```

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_20260811_224123/ptoas/hc_pre_linear.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel hc_pre_linear --route baseline \
  --captured-dump build_output/vpto_hc_pre_linear/run \
  --build-dir build_output/_probe_hc_pre_linear 2>&1 | grep "aclrtMallocHost\|size is invalid"
```

### 根因

**文件**：`.claude/skills/vpto-board-validate/lib/setup_main.py`（ptr 循环）

```python
param_decls.append(f"    size_t elemCount_{n} = {e};" + ...)
param_decls.append(f"    size_t fileSize_{n} = elemCount_{n} * sizeof({ct});")
# 当 e=0 时 fileSize=0 → main.cpp 里 aclrtMallocHost(&ptr, 0) 失败
```

### 修复方案（已实施）

路由到 **测试框架**。✅ 采用"简单修"思路——当 `elemCount=0` 时跳过该
ptr 的 alloc/memcpy/read/free，只传 nullptr 给 kernel。

**文件**：`.claude/skills/vpto-board-validate/lib/setup_main.py`

ptr 循环里，`e = ec.get(n, 0)` 之后若 `e == 0`：
- 仍 emit `elemCount_{n} = 0; // skipped: 0-elem` + `fileSize_{n}` +
  `{ct} *{n}Host = nullptr;` + `{ct} *{n}Device = nullptr;`（保持 nullptr）
- **跳过** `aclrtMallocHost`/`aclrtMalloc`（避免 size=0 失败）
- **跳过** `ReadFile3`（避免 0 字节文件返回 false）
- **跳过** `aclrtMemcpy`（0 字节拷贝无意义）
- **跳过** `aclrtFree`/`aclrtFreeHost`（对 nullptr 安全但保持一致跳过）
- kernel 收到 null GM ptr（单 block 启动在 0-elem 下不会访问它）

非 0-elem ptr 的逻辑保持不变。

**注意**：当前 sweep 里带 `elemCount=0` 的 kernel（`q_rope_prepare` v3/v4、
`qr_rms_norm_quant` v3）都在 C5（ptoas tdivs lowering）阶段崩溃，到不了
NPU run。C4 是 **latent 代码缺陷**：一旦 C5 修完（ptoas 侧已修），这些
kernel 会到达 NPU run 并触发 C4——本修复提前堵住这个路径。

**验证**：对 `q_rope_prepare.pto` 用 `elem_counts_override={v3:0, v4:0}`
调用 `gen_main_cpp()`，确认 0-elem ptr 不再 emit `aclrtMallocHost`/
`ReadFile3`，改传 `nullptr`；非 0-elem ptr 仍有完整 alloc/read/copy。

---

## C5 — ptoas pto.tdivs / pto.tstore 无 A5 template（~80 行）

### 背景

ptoas 的 A5 模板库没有 `pto.tdivs`（tile 标量除法）的 i32 dtype 签名
模板。pypto 在整数除法场景（Sinkhorn 归一化、route hashing 等）会发出
`pto.tdivs` op，ptoas 找不到匹配模板 → lowering 失败。

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)
# merge_norm（典型 tdivs kernel）
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_20260811_224123/ptoas/merge_norm.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel merge_norm --route baseline \
  --captured-dump build_output/vpto_merge_norm/run \
  --build-dir build_output/_probe_merge_norm 2>&1 | grep "NoMatchingTemplate"
```

预期错误：
```
NoMatchingTemplate: no legal template for op='pto.tdivs' target='a5';
  template_tdivs_tile_scalar: dtype signature ('i32', 'i32', 'i32') is not supported;
  template_tdivs_scalar_tile: ... not supported;
  vmi_tdivs: ... not supported;
  vmi_tdivs_scalar_tile: ... not supported
```

### 根因

**文件**：`PTOAS/ptodsl/ptodsl/tilelib/templates/a5/tdivs.py` 的 `_DTYPES`
只注册了 `f16`/`f32`，没有 `i32` 签名 → 模板匹配在 dtype 签名阶段就被拒。

**修正一个文档里的误述**：原报告说"A5 only supports tdivs for fp32"——
这只对 ptodsl 模板层。硬件/IR 层 i32 其实是通的：
- pto-isa A5 `TDivS.hpp` 的 `TDIVS_IMPL` static_assert 允许
  `int32_t/uint32_t/int16_t/uint16_t`，整数走 `TDivs_naive` 标量循环。
- ptodsl `pto.vdiv` verifier 接受 `si32/i32/f16/f32`。

但 **A5 vector core 没有整数 `vdiv` 指令**：直接对 i32 用 `pto.vdiv` 会
lower 成不存在的 `llvm.hivm.vdiv.vNi32.x` builtin，链接/bisheng 阶段会失败。
正确的整数除法路径是 **fp32 绕行**（`vcvt i32→f32, round` → `vdiv` →
`vcvt f32→i32, truncate, NOSAT`）——这跟同仓库 `template_tcolexpanddiv_i32`
的既有约定一致。`lib/TileOps/math.py` 里还有更完整的 `_tl_soft_vdiv_i32`
软件模拟（处理符号/除零/精化），但目前没接到任何 tdivs/tdiv 模板上。

**受影响 kernel（已确认，全部是 i32 tdivs）**：`merge_norm`、`rope_cs`、
`rmsnorm_rope`、`route_hash`、`rope`、`prefill_c4_rmsnorm_rope`、
`prefill_idx_c4_rmsnorm_rope`、`prefill_hca_c128_rmsnorm_rope` 等。用例都是
索引算术（`gather_lin2d / cols → 行号`），C 截断整除语义。
`qr_rms_norm_quant`/`ffn_norm` 用的是 `pto.tdiv`（tile-tile）且 dtype=f32，
**不是** i32 tdivs 问题——见下方 tstore 变体。

另一个变体：`ffn_norm` 命中 `pto.tstore` 的 NoMatchingTemplate（6 个候选
模板的 custom constraints 不满足）。

### 修复方案（已实施）

**文件**：`PTOAS/ptodsl/ptodsl/tilelib/templates/a5/tdivs.py`

1. `_DTYPES` 加 `("i32","i32","i32")`（只加 i32——扫描确认 pypto 对 tdivs
   只发 i32，不发 ui32/i16/ui16；避免无符号转换的正确性问题）。
2. 新增 `_div_i32()`：`vcvt(i32→f32, R)` → `vdiv` → `vcvt(f32→i32, Z, NOSAT)`，
   1:1 lane 映射，照搬 `tcolexpanddiv._divide_i32` 的写法。
3. `_div()` 加 i32 分支，并把 high-precision 路径显式限定为 float-only
   （防止 i32 op 误带 `precisionType=high_precision` 时崩）。
4. VMI 模板（`vmi_tdivs`/`vmi_tdivs_scalar_tile`）保持 f32-only——VMI 融合
   是浮点专属；非 VMI 的 i32 走新加的标量模板。

`tdiv.py`（tile-tile）不改——扫描确认 pypto 对 `pto.tdiv` 只发 f32，
i32 tile-tile 除法不是真实用例。

**同步**：改的是 `ptodsl/` 源树（daemon 的 PYTHONPATH 指向源树，直接生效）；
为保险把 `install311/`、`install/`、`build311/python/` 三个安装副本也同步
了，并清了 `__pycache__`。

**验证**：
- `ptodsl/tests/test_tilelib_catalog.py` 新增 `test_tdivs_i32_routes_through_fp32_divide`，
  断言 i32 两个候选模板都生成 `vcvt`+`vdiv`。PASS（含全量 catalog sweep
  `test_each_catalog_entry_selects_and_renders`）。
- 端到端复现：`merge_norm`（baseline 路线）的 `NoMatchingTemplate` 消失，
  ptoas→bisheng 全过，到了 NPU run 才因 `v2.bin` 缺失（C2/C3 框架问题）停。
  `rope_cs` 同样：tdivs lowering 过了，停在 `v3.bin`（C2/C3）。

**剩余**：`ffn_norm` 的 `tstore` NoMatchingTemplate 是**另一个根因**
（`_check_store_bounds` 形状约束，非 dtype 缺口），不在此修，需单独跟进。

---

## C6 — PTO_TO_CPP 缺 i8→int8_t 映射（~30 行）

### 背景

`.pto` 用 `!pto.ptr<i8>`（int8 量化权重 / int8 输出）。`pto_parse.py` 的
`PTO_TO_CPP` 映射表没有 `i8` 条目 → `setup_main.py` 生成 `launch.cpp`
时写出原始 `i8` → bisheng 不认 `i8`（要 `int8_t`）→ 编译失败。

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_20260811_224123/ptoas/quant.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel quant --route baseline \
  --captured-dump build_output/vpto_quant/run \
  --build-dir build_output/_probe_quant 2>&1 | grep "unknown type name"
```

预期错误：
```
launch.cpp:35:79: error: unknown type name 'i8'
extern "C" __global__ AICORE void quant(__gm__ float* v1, __gm__ i8* v2, ...);
```

### 根因

**文件**：`.claude/skills/vpto-board-validate/lib/pto_parse.py:183-185`

```python
# 当前（有 bug）：
PTO_TO_CPP = {"f32": "float", "bf16": "bfloat16_t", "f16": "half",
              "i32": "int32_t", "i64": "int64_t", "i16": "int16_t",
              "index": "int64_t"}
# ← 缺 "i8": "int8_t", "u8": "uint8_t"
```

**受影响 kernel**：`quant`、`score_mat`、`kv_and_cache_write`、
`qproj_matmul`、`exp_gate_mm`、`exp_h_q`、`sh_gate_mm`、`sh_up_mm`、
`exp_up_mm`、`sh_w2_mm`、`exp_w2_mm`、`x_norm_quant` 等所有含
`!pto.ptr<i8>` 的 kernel。

### 修复方案（已实施）

路由到 **测试框架**。

**文件**：`.claude/skills/vpto-board-validate/lib/pto_parse.py`

两处映射都补齐了 `i8`/`u8`（`PTO_TO_CPP` 和 `pto_type_to_c` 的 `mapping`
字典都要加——后者决定 `launch.cpp` 的 `__gm__` 指针类型，不加的话 host
类型仍是裸 `i8`，bisheng 同样不认）：

```python
# PTO_TO_CPP：
PTO_TO_CPP = {"f32": "float", "bf16": "bfloat16_t", "f16": "half",
              "i32": "int32_t", "i64": "int64_t", "i16": "int16_t",
              "i8": "int8_t", "u8": "uint8_t",   # ← 新增
              "index": "int64_t"}

# pto_type_to_c 的 mapping：
"i8": ("int8_t", "__gm__ int8_t*"),
"u8": ("uint8_t", "__gm__ uint8_t*"),
```

`capture.py` 的 `_DTYPE_TO_NP` 已有 `"INT8": "int8"`——dump 里 dtype 字符串
是 `INT8`/`INT32`/`INT64`/`FLOAT32`/`BFLOAT16`，全部已映射，无需改动。

**验证**：`pto_type_to_c("i8")` 现在返回 `("int8_t", "__gm__ int8_t*")`，
`pto_type_to_c("u8")` 返回 `("uint8_t", "__gm__ uint8_t*")`。`quant` 的
`launch.cpp` 将生成 `__gm__ int8_t* v2` 而非 `__gm__ i8* v2`，bisheng
不再报 `unknown type name 'i8'`。

---

## C7 — SSA dominance + _aic/_aiv split（~6 行）

### 背景

两个子问题：

### C7a — `build_valid.pto` 的 SSA dominance 违规

**复现**：`attention_hca` 模块的 capture 步骤就失败（pypto 发出的 .pto
本身有 SSA 违规）：

```bash
cat build_output/_jit_attention_hca_test_*/report/codegen_errors.txt
# 预期：
# ptoas compilation failed: error: operand #0 does not dominate this use
```

**根因**：pypto 生成的 `build_valid.pto:27:3` 有一个 SSA 值在定义前被使用。
路由到 **pypto**。

### C7b — `_aic`/`_aiv` split kernel 总是被解析/编译成 `_aic` 半

**受影响**：`qk_pv_aic`、`qk_pv_aiv`、`gate_aic`、`gate_aiv`、
`mtp_projection_linear_aic`、`mtp_projection_linear_aiv`

**根因**：`name_map` 里确实有 `*_aic`/`*_aiv` 各自的 callable_id
（不是"找不到 .pto"），`validate.py` 的前缀匹配也能找到共享的
`qk_pv.pto` / `mtp_projection_linear.pto`。真正的 bug 是：**这些
.pto 文件里有两个 `func.func`**（`@qk_pv_aic` cube + `@qk_pv_aiv` vector，
签名相同），而框架的三个函数都按"整个文件"操作，导致 `_aiv` 被当成 `_aic`：

1. `pto_parse.parse_pto` 用 `re.search(r'func\.func\s+@(\w+)\((.*?)\)', text)`
   只抓**第一个** func —— 无论 `--kernel` 传的是 `qk_pv_aic` 还是
   `qk_pv_aiv`，`info["func_name"]` 永远是 `qk_pv_aic`。
2. `setup_vpto.detect_kernel_kind` 对整文件 grep "cube" —— `_aiv` 半
   因为文件里存在 `_aic` 的 cube 属性，被误判为 cube。
3. `setup_vpto.preprocess_pto` 的 func-attr `replace` 是全文件替换 ——
   给 `_aic` 和 `_aiv` 两个 func 都打了 `pto.kernel` attr，ptoas 会
   生成两个 symbol，但 `launch.cpp` 的 `extern` 声明只匹配第一个。

```bash
# 验证 split .pto 有两个 func.func：
grep "func.func @" build_output/_jit_attention_csa_test_*/ptoas/qk_pv.pto
# func.func @qk_pv_aic(...) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
# func.func @qk_pv_aiv(...) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
```

### 修复方案（已实施）

路由到 **测试框架**。三个函数都加 `kernel: str | None = None` 参数，
当指定 kernel 时按 func 名锚定，只操作目标 func：

- `pto_parse.parse_pto(pto_path, kernel)`：正则锚定
  `func.func @<kernel>(...)`，解析正确的半。
- `setup_vpto.detect_kernel_kind(pto_text, kernel)`：把"cube" 搜索范围
  缩到 `func.func @<kernel>` 的 body（到下一个 `func.func`/`}` 为止），
  不再因为文件里有 `_aic` 就把 `_aiv` 误判成 cube。
- `setup_vpto.preprocess_pto(..., kernel)`：func-attr 编辑改成用
  `re.sub(count=1)` 只在 `func.func @<kernel>` 的 `attributes {` 里插入
  `pto.kernel`，另一半不打 attr → ptoas 只生成目标 symbol。

`vpto_run.py` 三处调用点传入 `kernel`（`args.kernel or args.pto.stem`，
line 331 已解析）。

**验证**（`mtp_projection_linear.pto`，含 `_aic` cube + `_aiv` vector）：
```
parse_pto(pto)                          func_name=mtp_projection_linear_aic  (第一个)
parse_pto(pto, 'mtp_projection_linear_aic')  func_name=mtp_projection_linear_aic
parse_pto(pto, 'mtp_projection_linear_aiv')  func_name=mtp_projection_linear_aiv  ✓
detect_kernel_kind(text, '_aic') → cube
detect_kernel_kind(text, '_aiv') → vector  ✓  (修前两个都是 cube)
preprocess_pto(..., '_aiv') → pto.kernel 只插入到 _aiv 的 attributes，_aic 不动 ✓
```

**C7a（SSA dominance）**仍在 **pypto** 侧，不在本仓库修。

---

## C8 — 0-iter SPMD / kernel 未被调度（7 行）

### 背景

两类：

### C8a — inout role 未匹配（5/7 行，与 C2 同根因）

`kv_touch`、`gate_pre_route`、`hc_head_seed` 在 dump 里有 `role=inout`
记录但无 `role=input/output` 记录 → harvest 跳过。修 C2 的同时自动修复。

### C8b — 真正的 0-iter SPMD（2/7 行）

`hc_post_inactive_pad` 在 prefill 模块的 `deps.json` 里**没有任何 task
含该 kernel_id** — 该 kernel 的分支在当前测试输入下未被触发，SPMD 循环
体从未执行 → 没有 dump 数据。

### 修复方案

- C8a：修 C2 自动修复。✅（commit `dcc5568`）
- C8b：路由到 **测试框架**。✅（已实施）

#### C8b 实施

**文件**：`tests/dsv4_validate/capture.py`、`tests/dsv4_validate/validate.py`

`capture.harvest_kernel` 在"no dump records"时不再抛 `RuntimeError`，
改抛一个专门的 `NotExercised` 异常（`capture.py` 顶层定义）：

```python
class NotExercised(Exception):
    """Kernel branch not triggered by the test input (0-iter SPMD, no task)."""

# harvest_kernel:
if not all_indices:
    raise NotExercised(
        f"kernel {kernel!r} (fid {target_fid}) produced no dump records — "
        f"its branch was not triggered by the test input (0-iter SPMD)")
```

`validate.run_phase5_module` 的 harvest try/except 增加 `NotExercised`
分支（必须在通用 `except Exception` **之前**），用
`compare_status="not-exercised"` 记录：

```python
except _capture.NotExercised as e:
    results.append(_phase5_error_row(
        module, model_py, mode, device, kname,
        f"not-exercised: {e}", route=route,
        compare_status="not-exercised"))
    continue
```

`_phase5_error_row` 加 `compare_status: str = "crash"` 形参（向后兼容）。

**验证**：`--module hc_post --mode prefill` 时 `hc_post_inactive_pad`
在 `sweep_results.csv` 里会记 `compare_status=not-exercised` 而非
`crash`。要真正跑通它仍需换一个能触发该分支的测试输入（不在框架侧）。

---

## VMI-UB — VMI+membar 路线的 UB 对齐崩溃（13 行，仅 vmi-membar-vfoff 路线）

### 背景

vmi-membar-vfoff 路线新引入的 13 个回归：baseline PASS 的 kernel 在
VMI 路线下 AICore 崩溃。

### 复现

```bash
source scripts/vpto_env.sh && export PTOAS_ROOT=$(dirname $PTOAS_BIN)
# mix_x（attention_csa，decode）
.venv/bin/python3 .claude/skills/vpto-board-validate/vpto_run.py \
  --pto build_output/_jit_attention_csa_test_20260811_224123/ptoas/mix_x.pto \
  --model-py models/deepseek_v4_pro/decode_attention_csa.py \
  --mode decode --device 0 --kernel mix_x --route vmi-membar-vfoff \
  --captured-dump build_output/vpto_mix_x/run \
  --build-dir build_output/_probe_vmi_mixx 2>&1 | grep "not aligned\|AICore\|exception"
```

预期错误：
```
errcode:(340) errorStr: The address for VEC to access UB is not aligned.
retCode=0x31, vector core exception.
fault kernel_name=mix_x
```

### 根因

**文件**：ptoas 的 VMI 流水线 codegen（不在本仓库）

**受影响 kernel**：`mix_x`（6 模块）、`merge_norm`（4 个 prefill 模块）、
`rms_norm` leaf、`hc_head_reduce`、`mtp_projection_norm`

VMI 融合流水线在处理这些 kernel 形状时生成了 UB 未对齐访存指令。
注意 `merge_norm` 的 decode 变体在 VMI 下 PASS（小 tile `[8,...]`），
但 prefill 变体崩溃（大 tile `[128,...]`）—— 说明是 tile 形状触发的
VMI 代码生成 bug。

### 修复方案

路由到 **ptoas**（VMI 流水线）。修复后 VMI 路线应从 45 PASS 回升到
~58 PASS（恢复 13 个回归），同时保持 7/8 精度 FAIL 的修复。

---

## 修复优先级建议

| 优先级 | 修复项 | 责任层 | 预期转化 | 难度 |
|---|---|---|---|---|
| **P0** | C6: 加 i8→int8_t 映射 | 框架 | ~30 crash→pass/fail | ✅ 已实施（`pto_parse.py`） |
| **P0** | C2: 加 inout harvest 分支 | 框架 | ~50 crash→pass/fail | ✅ 已实施（commit `dcc5568`） |
| **P1** | C3: 从 .pto 推导 scalar 语义 | 框架 | ~45 crash + 3 精度 | ✅ 已实施（`derived` shape + `dumped` runtime value 双路径） |
| **P1** | C4: 跳过 0 字节 alloc | 框架 | ~30 crash→pass/fail | ✅ 已实施（`setup_main.py` 跳过 0-elem alloc） |
| **P2** | C5: 加 A5 tdivs i32 template | ptoas | ~80 crash→pass/fail | ✅ 已实施（fp32 绕行） |
| **P2** | C1a: 修 inner-kernel 硬编码形状 | pypto | 5 精度 FAIL→pass | 需 pypto 侧 |
| **P3** | VMI-UB: 修 VMI UB 对齐 | ptoas | 13 回归→pass | 需 ptoas 侧 |
| **P3** | C7a: SSA dominance | pypto | 部分 crash | 需 pypto 侧 |
| **P3** | C7b: aic/aiv split 按 func 锚定 | 框架 | ~6 crash | ✅ 已实施（`pto_parse`/`setup_vpto`/`vpto_run`） |
| **P3** | C8b: 0-iter SPMD 标记 | 框架 | 2 crash→skip | ✅ 已实施（`NotExercised`） |

**框架侧已修完 C2/C3/C4/C6/C7b/C8b**。剩余框架项无。
pypto 侧剩 C1a/C7a，ptoas 侧剩 VMI-UB。
