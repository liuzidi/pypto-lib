#!/usr/bin/python3
"""Golden 基础设施 — 自动解析 main.cpp，生成/比对 .bin 文件。"""
import os, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

SEED = 19

_HOST_TYPE_TO_NP = {
    "aclFloat16": np.float16, "bfloat16_t": np.uint16, "bool": np.bool_,
    "double": np.float64,   "float": np.float32,       "half": np.float16,
    "int": np.int32,        "int8_t": np.int8,         "int16_t": np.int16,
    "int32_t": np.int32,    "int64_t": np.int64,       "size_t": np.uint64,
    "uint8_t": np.uint8,    "uint16_t": np.uint16,     "uint32_t": np.uint32,
    "uint64_t": np.uint64,  "unsigned": np.uint32,
}

@dataclass
class CaseMeta:
    elem_counts: Dict[str, int]
    np_types: Dict[str, np.dtype]
    read_order: List[str]
    outputs: List[str]

    @property
    def inputs(self) -> List[str]:
        return [n for n in self.read_order if n not in self.outputs]


def load_case_meta(main_cpp: str = "main.cpp", outputs_txt: str = "outputs.txt") -> CaseMeta:
    text = Path(main_cpp).read_text(encoding="utf-8")
    elem_counts = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r'size_t\s+elemCount_(\w+)\s*=\s*(\d+);', text)
    }
    np_types = {
        m.group(1): np.dtype(_HOST_TYPE_TO_NP[m.group(2).strip()])
        for m in re.finditer(
            r'size_t\s+fileSize_(\w+)\s*=\s*elemCount_\1\s*\*\s*sizeof\(([^)]+)\);',
            text,
        )
    }
    read_order = re.findall(r'ReadFile3?\("\./([^"]+)\.bin"', text)
    outputs_path = Path(outputs_txt)
    outputs = []
    if outputs_path.is_file():
        outputs = [line.strip() for line in outputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return CaseMeta(elem_counts=elem_counts, np_types=np_types, read_order=read_order, outputs=outputs)


def load_int32_assignments(main_cpp: str = "main.cpp") -> List[int]:
    text = Path(main_cpp).read_text(encoding="utf-8")
    return [int(m.group(1)) for m in re.finditer(r'int(?:32|64)_t\s+\w+\s*=\s*(-?\d+);', text)]


def rng():
    return np.random.default_rng(SEED)


def write_buffers(meta: CaseMeta, buffers: dict):
    for name in meta.read_order:
        if name in buffers:
            buffers[name].astype(meta.np_types[name]).tofile(f"{name}.bin")


def write_golden(meta: CaseMeta, golden: dict):
    for name, data in golden.items():
        np_type = meta.np_types.get(name)
        if np_type is not None:
            data = data.astype(np_type)
        data.tofile(f"golden_{name}.bin")


def bf16_to_float32(x: np.ndarray) -> np.ndarray:
    x32 = np.zeros(x.shape, dtype=np.float32)
    x_flat = x.ravel().view(np.uint16)
    y_flat = x32.ravel().view(np.uint16)
    y_flat[1::2] = x_flat
    return x32


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    y = np.zeros(x32.shape, dtype=np.uint16)
    y_flat = y.ravel()
    y_flat[...] = x32.ravel().view(np.uint16)[1::2]
    return y


def load_strided_2d(buf, offset, rows, cols, row_stride):
    flat = buf.ravel()
    start = offset
    out = np.zeros((rows, cols), dtype=flat.dtype)
    for r in range(rows):
        out[r, :] = flat[start + r * row_stride : start + r * row_stride + cols]
    return out


def store_strided_2d(dst, src, offset, row_stride):
    flat = dst.ravel()
    rows, cols = src.shape
    start = offset
    for r in range(rows):
        flat[start + r * row_stride : start + r * row_stride + cols] = src[r, :]
    return flat.reshape(dst.shape)
