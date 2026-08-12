#!/usr/bin/env python3
"""Shared PTO / EmitC parsing helpers for test_for_ptoas generators.

Previously every setup_*.py carried its own copy of parse_pto / pto_type_to_c /
parse_cpp / split_cpp_args / PTO_TO_CPP / get_outputs_from_golden_lib /
patch_kernel_for_camodel. They are unified here so a single fix propagates to
all backends (on-board, CAModel docker, CAModel remote).

Usage in a generator:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lib.pto_parse import parse_pto, pto_type_to_c, ...
"""

import re
from pathlib import Path

__all__ = [
    "parse_pto",
    "derive_scalar_values",
    "pto_type_to_c",
    "parse_cpp",
    "split_cpp_args",
    "PTO_TO_CPP",
    "get_outputs_from_golden_lib",
    "get_scalar_semantic_names",
    "get_golden_constants",
    "compute_scalar_comment",
    "patch_kernel_for_camodel",
]


def parse_pto(pto_path: Path, kernel: str | None = None) -> dict:
    """Extract from a .pto file: func_name, params, dims, elem_counts,
    scalar_dims.

    Each param is {"name": "vN", "pto_type": <dtype>, "arg": "%argN"}.
    elem_counts is derived from pto.make_tensor_view shapes that reference
    static %cN_index constants; dynamic (non-constant) shapes are skipped.
    scalar_dims maps each index scalar's %argN to the list of tensor params
    whose make_tensor_view shape mentions it: [("v1", [other_static_dim_sizes])].
    Used by derive_scalar_values() to recover an index scalar's value from
    a tensor's elem_count when it controls a single dynamic tensor dimension.

    `kernel` selects which func.func to parse in a multi-func .pto. Split
    kernels (e.g. qk_pv_aic + qk_pv_aiv) live in one .pto as two func.func
    blocks with the same ptr/scalar signature. Without `kernel`, the first
    func.func in the file is parsed (backward compat for single-func .pto).
    """
    text = pto_path.read_text(encoding="utf-8")
    info = {"func_name": "", "params": [], "dims": {}, "elem_counts": {},
            "scalar_dims": {}}

    # Anchor on the specific kernel func when requested so a split .pto's
    # second half (e.g. qk_pv_aiv) is not mis-parsed as the first (qk_pv_aic).
    if kernel:
        m = re.search(
            rf'func\.func\s+@{re.escape(kernel)}\((.*?)\)', text)
        if not m:
            return info
        func_name = kernel
        raw_params = m.group(1)
    else:
        m = re.search(r'func\.func\s+@(\w+)\((.*?)\)', text)
        if not m:
            return info
        func_name = m.group(1)
        raw_params = m.group(2)
    info["func_name"] = func_name

    pt = re.findall(r'%\w+:\s*!pto\.ptr<(\w+)>', raw_params)
    # tensor-view name per ptr: from "%<view>__ssa_vN_view = pto.make_tensor_view %argM"
    # the view stem (before __ssa) often encodes the originating jit-fn param
    # name, which matches the model's TensorSpec name. Used by resolve_meta to
    # build a ptr->spec mapping when ptr order != spec order.
    view_names: dict[int, str] = {}
    for vm in re.finditer(
        r'%(\w+)__ssa_\w*\s*=\s*pto\.make_tensor_view\s+(%arg\d+)', text
    ):
        stem = vm.group(1)
        # strip a trailing _inlineN (codegen rename) -> reveals the param stem
        stem = re.sub(r'_inline\d+$', '', stem)
        arg_idx = int(vm.group(2)[len('%arg'):])
        view_names.setdefault(arg_idx, stem)  # first view wins if multiple
    for i, dtype in enumerate(pt):
        info["params"].append({
            "name": f"v{i+1}", "pto_type": dtype, "arg": f"%arg{i}",
            "view_name": view_names.get(i, ""),
        })

    # 匹配所有非 ptr 标量 (i32, index 等)，按签名顺序
    scalar_matchers = [
        (r'(%\w+):\s*index', "index"),
        (r'(%\w+):\s*i32', "i32"),
    ]
    scalars_raw = []  # (pos_in_raw, arg_name, pto_type)
    for pattern, ptype in scalar_matchers:
        for mobj in re.finditer(pattern, raw_params):
            scalars_raw.append((mobj.start(), mobj.group(1)[1:], ptype))
    scalars_raw.sort(key=lambda x: x[0])  # 按签名中位置排序

    for j, (_, sig_name, ptype) in enumerate(scalars_raw):
        idx = len(pt) + j + 1
        info["params"].append({"name": f"v{idx}", "pto_type": ptype,
                               "arg": f"%arg{len(pt)+j}", "sig_name": sig_name})

    for m in re.finditer(r'%c(\d+)_index\s*=\s*arith\.constant\s+(\d+)', text):
        info["dims"][f"%c{m.group(1)}_index"] = int(m.group(2))

    for p in info["params"]:
        if p["pto_type"] in ("i32", "index"):
            continue
        arg = p["arg"]
        pattern = rf'{re.escape(arg)},\s*shape\s*=\s*\[(.*?)\]'
        m2 = re.search(pattern, text)
        if not m2:
            continue
        shapes = [s.strip() for s in m2.group(1).split(',')]
        # Static elem_count: product when every dim is a %cN_index constant.
        ec = 1
        ok = True
        for s in shapes:
            v = info["dims"].get(s)
            if v is None:
                ok = False
                break
            ec *= v
        if ok:
            info["elem_counts"][p["name"]] = ec
        # Dynamic dims: record each %argN-shaped dimension so its value can be
        # recovered later from this tensor's elem_count (total / product of the
        # other static dims). A scalar may appear in multiple tensors' shapes;
        # collect all of them so derive_scalar_values can cross-check.
        for s in shapes:
            if re.fullmatch(r'%arg\d+', s):
                other = [info["dims"][x] for x in shapes if x != s]
                info["scalar_dims"].setdefault(s, []).append(
                    (p["name"], other))
    return info


def derive_scalar_values(info: dict, elem_counts: dict | None = None) -> dict:
    """Derive each index scalar's runtime value from tensor-view shapes.

    For each index scalar %argN that appears in at least one
    `pto.make_tensor_view shape=[...]`, recover its value as
    `elem_count[ptr_name] // product(other_static_dim_sizes)`, where the
    other dims are the static %cN_index dims of that tensor. If the scalar
    appears in multiple tensors, require them to agree; on disagreement (or a
    missing elem_count, or a non-clean division) skip that scalar so it falls
    back to the setup_main `= 0; // FIXME` path rather than emitting a wrong
    value.

    Returns {scalar_param_name ("v6"): int_value} for the derivable scalars.
    Scalars with no shape occurrence (e.g. partition_view offsets, loop
    bounds) are absent — they are not derivable from shapes alone.
    """
    ec = dict(info.get("elem_counts", {}))
    if elem_counts:
        ec.update(elem_counts)
    # Map %argN -> scalar param name (vN) in .pto signature order.
    arg_to_name = {p["arg"]: p["name"] for p in info["params"]
                   if p["pto_type"] in ("i32", "index")}
    out: dict[str, int] = {}
    for scalar_arg, occurrences in info.get("scalar_dims", {}).items():
        name = arg_to_name.get(scalar_arg)
        if not name:
            continue
        resolved: int | None = None
        consistent = True
        for ptr_name, other_dims in occurrences:
            total = ec.get(ptr_name)
            if total is None:
                continue
            denom = 1
            for d in other_dims:
                denom *= d
            if denom == 0 or total % denom != 0:
                continue
            cand = total // denom
            if resolved is None:
                resolved = cand
            elif resolved != cand:
                consistent = False
                break
        if resolved is not None and consistent:
            out[name] = resolved
    return out


def pto_type_to_c(pto_type: str) -> tuple:
    """Map a PTO element type to (host_c_type, device_gm_ptr_type)."""
    mapping = {
        "f32": ("float", "__gm__ float*"),
        "f16": ("uint16_t", "__gm__ bfloat16_t*"),
        "bf16": ("uint16_t", "__gm__ bfloat16_t*"),
        "i32": ("int32_t", "__gm__ int32_t*"),
        "i64": ("int64_t", "__gm__ int64_t*"),
        "i16": ("int16_t", "__gm__ int16_t*"),
        # i8/u8: quantized weights and int8 outputs. Without this, bisheng
        # rejects the raw `i8` token ("unknown type name 'i8'") in launch.cpp.
        "i8": ("int8_t", "__gm__ int8_t*"),
        "u8": ("uint8_t", "__gm__ uint8_t*"),
    }
    if pto_type == "i32":
        return ("int32_t", "int32_t")
    if pto_type == "index":
        return ("int64_t", "int64_t")
    return mapping.get(pto_type, (pto_type, f"__gm__ {pto_type}*"))


def split_cpp_args(text: str) -> list:
    """Split a C++ parameter list on top-level commas (ignoring <...> and (...))."""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth = max(depth - 1, 0)
        elif ch == ',' and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def parse_cpp(cpp_path: Path) -> dict:
    """Extract from an EmitC .cpp: kernel signature (with C++ types) + output buffer names.

    Outputs are detected via TSTORE(tile, gt) where gt is a GlobalTensor bound
    to a pointer argument.
    """
    text = cpp_path.read_text(encoding="utf-8")
    info = {"params": [], "outputs": []}

    m = re.search(r'(?:extern\s+"C"\s+)?__global__\s+AICORE\s+void\s+(\w+)\s*\(([^)]*)\)', text)
    if not m:
        return info
    info["func_name"] = m.group(1)
    raw = m.group(2)

    for part in split_cpp_args(raw):
        part = part.strip()
        gm = re.match(r'__gm__\s+(\w+)\s*\*\s*(\w+)', part)
        if gm:
            info["params"].append({"name": gm.group(2), "kind": "ptr", "cpp_type": gm.group(1)})
        else:
            m2 = re.match(r'(\w+)\s+(\w+)', part)
            if m2:
                info["params"].append({"name": m2.group(2), "kind": "scalar", "cpp_type": m2.group(1)})

    gt_to_ptr = {}
    for m in re.finditer(r'GlobalTensor<[^;]*>\s*\(\s*(\w+)\s*\+', text):
        lookback = text[:m.start()]
        match_gt = re.search(r'(\w+)\s*=\s*$', lookback.rstrip())
        if match_gt:
            gt_to_ptr[match_gt.group(1)] = m.group(1)

    tstores = re.findall(r'TSTORE\s*\(\s*\w+\s*,\s*(\w+)\s*\)', text)
    for gt in tstores:
        ptr = gt_to_ptr.get(gt)
        if ptr and ptr not in info["outputs"]:
            info["outputs"].append(ptr)

    return info


PTO_TO_CPP = {"f32": "float", "bf16": "bfloat16_t", "f16": "half",
              "i32": "int32_t", "i64": "int64_t", "i16": "int16_t",
              "i8": "int8_t", "u8": "uint8_t",
              "index": "int64_t"}


def get_outputs_from_golden_lib(root: Path, kernel: str) -> list:
    """Extract golden dict keys from the build fn's `return buffers, {...}`.

    Resolves the build fn name via the BUILDERS map first
    (`"kernel": build_fn`), since the build fn is often named after the
    *operation* (e.g. `build_q_proj`) rather than the kernel entry
    (`qwen3_decode_incore_1`). Falls back to `def build_<kernel>` for libs
    where they happen to match (e.g. rmsnorm -> build_rmsnorm).
    """
    for lib in root.glob("*_golden_lib.py"):
        text = lib.read_text(encoding="utf-8")
        build_fn = None
        bm = re.search(rf'"{re.escape(kernel)}"\s*:\s*(build_\w+)', text)
        if bm:
            build_fn = bm.group(1)
        else:
            build_fn = f"build_{kernel}"
        pattern = rf'def {build_fn}\(.*?\):(.*?)(?=\ndef\s|$)'
        m = re.search(pattern, text, re.S)
        if m:
            body = m.group(1)
            ret = re.search(r'return\s+buffers\s*,\s*\{([^}]+)\}', body)
            if ret:
                keys = re.findall(r'"(\w+)"', ret.group(1))
                return keys
    return []


def get_scalar_semantic_names(root: Path, kernel: str) -> list:
    """Return ordered semantic names of int32/index scalars for `kernel`.

    Parsed from the golden_lib BUILDERS map + the build function `ints[:N]`
    unpacking. Names are in .pto signature order (ctx scalars before spmd), so
    they align positionally with the non-spmd i32/index params from parse_pto.
    Returns [] if the kernel is not in BUILDERS or no unpacking is found.
    """
    for lib in root.glob("*_golden_lib.py"):
        text = lib.read_text(encoding="utf-8")
        bm = re.search(r'"%s"\s*:\s*(build_\w+)' % re.escape(kernel), text)
        if not bm:
            continue
        build_fn = bm.group(1)
        fb = re.search(r"def\s+%s\(.*?\):(.*?)(?=\ndef\s|$)" % re.escape(build_fn), text, re.S)
        if not fb:
            return []
        um = re.search(r"^\s*([A-Za-z_][\w,\s]*?)\s*=\s*ints\[(?::)?(\d+)\]", fb.group(1), re.M)
        if not um:
            return []
        return [n.strip() for n in um.group(1).split(",") if n.strip()]
    return []


def get_golden_constants(root: Path) -> dict:
    """Extract MAX_SEQ / SEQ_TILE / MAX_CTX_BLOCKS from the golden_lib.

    MAX_CTX_BLOCKS is computed as ceil(MAX_SEQ / SEQ_TILE) when the lib expresses
    it as an expression. Returns {} if MAX_SEQ or SEQ_TILE is absent.
    """
    for lib in root.glob("*_golden_lib.py"):
        text = lib.read_text(encoding="utf-8")
        ms = re.search(r"MAX_SEQ\s*=\s*(\d+)", text)
        st = re.search(r"SEQ_TILE\s*=\s*(\d+)", text)
        if not (ms and st):
            continue
        max_seq = int(ms.group(1)); seq_tile = int(st.group(1))
        mcb = re.search(r"MAX_CTX_BLOCKS\s*=\s*(\d+)", text)
        max_ctx_blocks = int(mcb.group(1)) if mcb else (max_seq + seq_tile - 1) // seq_tile
        return {"MAX_SEQ": max_seq, "SEQ_TILE": seq_tile, "MAX_CTX_BLOCKS": max_ctx_blocks}
    return {}


def compute_scalar_comment(sem_name: str, consts: dict,
                           load_vals: dict | None = None,
                           sig_name: str = "") -> str:
    """根据 golden_lib 中提取的语义名，返回注释字符串。

    包含参数含义、推荐值或取值范围。
    consts 来自 get_golden_constants() (MAX_SEQ/SEQ_TILE/MAX_CTX_BLOCKS)。
    load_vals 是当前 load 模式下的实际值 ({ctx_len, ctx_blocks})。
    sig_name 是备用的参数名（当 sem_name 为空时用作 hint）。

    返回空字符串当 sem_name 和 sig_name 都为空时。
    """
    if not sem_name and not sig_name:
        return ""

    if sem_name == "ctx_blocks":
        v = (load_vals.get("ctx_blocks")
             if load_vals and load_vals.get("ctx_blocks") is not None
             else consts.get("MAX_CTX_BLOCKS", "?"))
        return f"ctx_blocks: KV cache block数 = ceil(MAX_SEQ/SEQ_TILE) = {v}"
    elif sem_name == "ctx_len":
        v = (load_vals.get("ctx_len")
             if load_vals and load_vals.get("ctx_len") is not None
             else consts.get("MAX_SEQ", "?"))
        return f"ctx_len: 总序列长度 = {v}"
    elif sem_name == "derived":
        return "derived: 从 .pto tensor-view shape 推导的张量维大小"
    elif sem_name == "dumped":
        return "dumped: 从 args_dump.json 捕获的运行时 scalar 值"
    elif sem_name == "pair_index":
        return "pair_index: KV head pair索引, 范围 0..NUM_KV_HEADS-1, 建议 0"
    elif sem_name == "block_base":
        return "block_base: 全局输出起始block偏移, 建议 0"
    elif sem_name == "local_block":
        return "local_block: SPMD块内block偏移, 建议 0"

    # sem_name 无法识别或为空 → 尝试 sig_name 回退 (SPMD 等硬件参数)
    if sig_name:
        if "spmd_block_idx" in sig_name:
            return "SPMD block index, 单block时为0"
        if "spmd_block_num" in sig_name:
            return "SPMD block总数, 单block时为1"
        return f"sig={sig_name}: 无 golden_lib 语义"
    if sem_name:
        return f"sem={sem_name}: 参考 golden_lib build 函数"
    return ""


def patch_kernel_for_camodel(cpp_path: Path, func_name: str) -> bool:
    """Add an `extern "C" __global__` qualifier to a ptoas EmitC kernel signature.

    ptoas EmitC emits:  AICORE void func_name(...)
    CAModel needs:      extern "C" __global__ AICORE void func_name(...)
    Returns True when a patch was applied.
    """
    text = cpp_path.read_text(encoding="utf-8")
    old = f"AICORE void {func_name}("
    new = f'extern "C" __global__ {old}'
    if old in text and new not in text:
        text = text.replace(old, new)
        cpp_path.write_text(text, encoding="utf-8")
        return True
    return False
