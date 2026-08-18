# C3/C4 修复计划

## 调查结论

经过深入调查，C3 和 C4 的实际状态与 `FIX_HANDOFF.md` 文档描述有重要偏差：

### C3（index scalar 默认填 0）— **已在工作树完整实现，无需额外代码修改**

文档说 C3 已实施（`derive_scalar_values`，commit `aefe714`）。但文档只描述了 `derived` 分支（shape-derivable scalar）。实际上工作树里还实现了第二个分支 **`dumped`**，覆盖了文档没提到的那 13 个残余 case：

- `capture.py:199-208`：从 args_dump 的 `value` 字段 harvest scalar 值 → 写入 `capture_meta.json` 的 `scalar_values`
- `vpto_run.py:413,423,431`：读 `scalar_values` → 标 sem=`"dumped"`
- `setup_main.py:180-186`：`sem=="dumped"` 时 emit 真实值
- `pto_parse.py:373-374`：`compute_scalar_comment` 已有 `dumped` 注释

优先级链：`derived > dumped > ctx_len(首个非spmd) > None(=0)`。残余 13 个 case（`route_hash/v6`、`comb_sinkhorn/v5`、`split_pre_post/v6,v7` 等）的 dump value 全会被 `dumped` 覆盖。**C3 代码侧已完整**。

### C4（0 字节 alloc → aclrtMallocHost 失败）— **代码未修，是唯一真正需要动的框架项**

- `setup_main.py:145-163`：ptr 循环无条件 emit alloc/read/copy，当 `elemCount=0` 时 `fileSize=0` → `aclrtMallocHost(0)` 失败（且 `ReadFile3` 对 0 字节文件返回 false）。
- 调查发现当前 sweep 里 **没有运行结果实际触发 C4 错误**（76 个 result.json 无一含 alloc 错误）。原因是带 `elemCount=0` 的 kernel（`q_rope_prepare` v3/v4、`qr_rms_norm_quant` v3）都先在 C5（ptoas tdivs lowering）阶段崩溃了，到不了 NPU run。
- 但 C4 是 **latent 代码缺陷**：一旦 C5 修完（ptoas 侧已修），这些 kernel 会到达 NPU run 并触发 C4。`hc_pre_linear` 的旧 `_probe` main.cpp（`elemCount_v3=0`）证明该代码路径确实会生成。

## 修改计划

### 1. 修改 `setup_main.py`（C4 代码修复）— 唯一代码改动

**文件**：`.claude/skills/vpto-board-validate/lib/setup_main.py`

在 ptr 循环（`setup_main.py:145-163`）里，当 `e == 0`（elemCount 为 0）时，跳过 alloc/read/copy/free，只传 nullptr 给 kernel。具体改法（对每个 ptr `p`）：

- `e = ec.get(n, 0)` 之后，若 `e == 0`：
  - `param_decls`：仍 emit `elemCount_{n} = 0; // skipped: 0-elem (dynamic shape)` + `fileSize_{n} = 0;`
  - `ptr_decls`：仍 emit `{ct} *{n}Host = nullptr;` + `{ct} *{n}Device = nullptr;`（保持 nullptr，kernel 收到空 GM ptr）
  - **跳过** `alloc_host`/`alloc_dev`（不 emit `aclrtMallocHost(0)` / `aclrtMalloc(0)`）
  - **跳过** `reads`（不 emit `ReadFile3`，因为 0 字节文件会让 `ReadFile` 返回 false）
  - **跳过** `copy_in`（0 字节 memcpy 无意义）
  - `copy_out`/`writes`：output ptr 的 0-elem 也跳过（`WriteFile3` 对 0 字节是 benign 的，但保持一致跳过更干净）
  - `free_dev`/`free_host`：对 nullptr 调 `aclrtFree(nullptr)` / `aclrtFreeHost(nullptr)` 是安全的，但为干净起见也跳过

- 若 `e > 0`：保持现有逻辑不变。

实现方式：在循环开头算 `e` 后加 `if not e:` 分支，用 `continue` 跳过后续 append（但 `param_decls`/`ptr_decls` 仍需 append 基础声明，所以不用 continue，而是用 if/else 分叉）。

这跟文档 §C4 的"简单修"思路一致："当 elemCount=0 时，main.cpp 跳过该 ptr 的 alloc + memcpy + read，只传一个 nullptr 给 kernel。"

### 2. 更新 `FIX_HANDOFF.md` — 反映 C3/C4 的实际状态

**文件**：`baselines/vpto_dsv4_vector/FIX_HANDOFF.md`

- **§C3（约 332-343 行）**：补充说明 `dumped` 分支已实现（capture.py harvest scalar `value` 字段 → setup_main.py emit），覆盖文档原本没提到的 13 个残余 case。把"修复方案"从"与 C1b 同一个修复"更新为"已完整实施：`derived`（shape）+ `dumped`（runtime value）双路径"。
- **§C4（约 384-398 行）**：更新修复方案为"已实施"并描述实际改法（skip alloc/read/copy for 0-elem ptr，传 nullptr）。
- **第 99 行**：`仅剩 C4` → `框架侧全部修完`。
- **第 760 行优先级表**：C4 行的"简单（未实施）"→ "✅ 已实施（setup_main.py 跳过 0-elem alloc）"。
- **第 768 行**：`剩余框架项仅 C4` → `框架侧全部修完（C2/C3/C4/C6/C7b/C8b）`。

### 3. 验证

由于 C4 的两个触发 kernel（`q_rope_prepare`、`qr_rms_norm_quant`）当前在 C5 阶段崩溃（tdivs lowering），无法直接端到端验证 C4 修复。验证策略：

- **生成验证**：对一个有 `elemCount=0` 的 main.cpp（如 `vpto_q_rope_prepare`），重新跑 `vpto_run.py` 的 gen 阶段，确认新 main.cpp 对 0-elem ptr 不再 emit `aclrtMallocHost(0)` / `ReadFile3`，改为 nullptr 透传。
- **grep 验证**：确认新生成的 main.cpp 里 `elemCount_v3 = 0` 的 ptr 不再有 `aclrtMallocHost`/`ReadFile3` 行。
- **hc_pre_linear**：用磁盘上最新的 capture_meta（v3=512, 非 0）跑一次确认不回归（应继续 PASS）。

## 不做的事

- **不改 `test_for_ptoas_extracted/` 的 vendored 副本**：它是独立的 diverged snapshot（`_gen_main_cpp_onboard`，不同签名），不在 VPTO board-validation 流程里。
- **不重新跑 338-kernel 全量 sweep**：那需要 NPU + VPTO 环境，超出当前修复范围。C5 修完后才能端到端验证 C4 的两个触发 kernel。
- **不改 C1a/C5/C7/VMI-UB**：这些是 pypto/ptoas 侧，不在框架。