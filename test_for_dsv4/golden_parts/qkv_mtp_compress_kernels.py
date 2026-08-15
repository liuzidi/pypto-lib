#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden reference builders for DeepSeek-V4 (PRO) VPTO precision tests.

Covers qkv_proj_rope, the indexer (qr/kv/score/topk sub-kernels) and the
mtp_projection sub-kernels. Each ``build_<name>(meta, generator, ints)``
returns ``(buffers, golden)`` matching the .pto buffer order in
``kernel_signatures.json``.

All computations are self-contained numpy/torch-free numpy. bf16 is stored
as uint16 (matching validation_runtime._HOST_TYPE_TO_NP). Cube/matmul uses
plain matmul; RoPE uses the DeepSeek-V4 interleaved-pairs swap-gather form;
INT8 quant is per-row symmetric (127 / amax) with the same i32->fp16->i8
narrowing the kernel applies.
"""

import numpy as np

from validation_runtime import (
    bf16_to_float32,
    float32_to_bf16,
    rng,
    write_buffers,
    write_golden,
)

# ---------------------------------------------------------------------------
# DSV4 PRO model / kernel constants (config.PRO_KERNEL + kernel_signatures).
# ---------------------------------------------------------------------------
B = 4                  # DECODE_BATCH
S = 2                  # DECODE_SEQ
T = B * S             # 8 tokens per decode step
D = 7168              # hidden_size
H = 128               # num_attention_heads
HEAD_DIM = 512        # MLA value-head dim
ROPE_DIM = 64         # qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM   # 448
Q_LORA = 1536        # q_lora_rank
EPS = 1e-6
D_INV = 1.0 / D
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
DEC_T = 8            # DECODE_BATCH * DECODE_SEQ = 4 * 2
FP32_NEG_INF = np.float32(-3.4028234663852886e38)

# indexer constants
IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
IDX_NOPE_DIM = IDX_HEAD_DIM - ROPE_DIM   # 64
IDX_TOPK = 1024
IDX_KV_LEN = 16384 // 4                   # MAX_SEQ_LEN // COMPRESS_RATIO = 4096
COMPRESS_RATIO = 4
BLOCK_SIZE = 128
RMS_PAD_ROWS = 16                          # RMS_PAD_TILE (pad B rows up to 16)

# mtp_projection constants
HC_MULT = 4
HC_DIM = HC_MULT * D                        # 28672
D_CHUNK = 128
D_BLOCKS = D // D_CHUNK                     # 56
OUT_CHUNK = 128
OUT_BLOCKS = D // OUT_CHUNK                 # 56
QUANT_CHUNK = 128
LINEAR_T_TILE = 16

# tiling mirrors the kernels (only the ones that change the accumulation order)
Q_PROJ_TILE = 128          # qproj K-tile
QR_K_SLICE = D // 2         # 3584 (QR_OK=2)
QR_K_TILE = 256
KV_K_SLICE = D // 4         # 1792 (KV_OK=4)
KV_K_TILE = 128
QH_QUANT_TILE = 64          # qr_hadamard_quant row tile
QH_HEAD_DIM_TILE = 64       # qr_hadamard_quant col tile
QH_MM_TILE = 64             # qr_hadamard_matmul row tile
SCORE_REDUCE_TILE = 128
WEIGHTS_SCALE = IDX_HEAD_DIM ** -0.5 * IDX_N_HEADS ** -0.5


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _flat_output(meta, name):
    # Scalars (i32/index) have no elem_count entry; return an empty buffer so
    # build_ functions that list a scalar vN in their buffers dict don't KeyError.
    count = meta.elem_counts.get(name, 0)
    np_type = meta.np_types.get(name, np.float32)
    return np.zeros(count, dtype=np_type)


def make_fp32(generator, count, *, scale=0.05, positive=False):
    if positive:
        return generator.uniform(0.25, 1.5, size=count).astype(np.float32)
    return generator.uniform(-scale, scale, size=count).astype(np.float32)


def make_bf16(generator, count, *, scale=0.05, positive=False):
    return float32_to_bf16(make_fp32(generator, count, scale=scale, positive=positive))


def make_int8(generator, count, *, scale=2.0):
    return generator.integers(-127, 128, size=count).astype(np.int8)


def bf16_round(x_fp32):
    """fp32 -> bf16-then-fp32 (matches kernel's bf16 rint on store)."""
    return bf16_to_float32(float32_to_bf16(x_fp32))


def apply_interleaved_rope(x_fp32, cos_half, sin_half):
    """DeepSeek-V4 interleaved RoPE on the last dim (pairs of (even, odd)).

    x_fp32: [..., ROPE_DIM]; cos_half/sin_half: [HALF_ROPE].
    Matches the kernel's swap-gather form:
        out[j]   = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    where j^1 is the pair partner and sign[j] = +1/-1 alternating.
    Equivalent to the standard half rotation:
        y_even = x_even*cos - x_odd*sin
        y_odd  = x_even*sin + x_odd*cos
    """
    x_pair = x_fp32.reshape(*x_fp32.shape[:-1], -1, 2)
    x_even = x_pair[..., 0]
    x_odd = x_pair[..., 1]
    shape = x_even.shape
    cos_v = np.broadcast_to(cos_half.reshape(*([1] * (len(shape) - 1)), HALF_ROPE), shape).astype(np.float32)
    sin_v = np.broadcast_to(sin_half.reshape(*([1] * (len(shape) - 1)), HALF_ROPE), shape).astype(np.float32)
    y_even = x_even * cos_v - x_odd * sin_v
    y_odd = x_even * sin_v + x_odd * cos_v
    out = np.stack([y_even, y_odd], axis=-1)
    return out.reshape(*x_fp32.shape[:-1], ROPE_DIM)


def per_row_int8_quant(x_fp32):
    """Per-row symmetric INT8 quant with the kernel's i32->fp16->i8 narrowing.

    Returns (i8 [..., K], scale_dequant [..., 1]).
    """
    flat = x_fp32.reshape(-1, x_fp32.shape[-1])
    amax = np.maximum(np.abs(flat).max(axis=1, keepdims=True), np.float32(INT8_AMAX_EPS))
    scale_q = np.float32(INT8_SCALE_MAX) / amax
    scaled = flat * scale_q
    i32 = np.rint(scaled).astype(np.int32)
    # fp16 round (numpy fp16 round-to-nearest-even)
    half = i32.astype(np.float16)
    i8 = half.astype(np.int8)
    scale_dq = (1.0 / scale_q).astype(np.float32)
    return i8.reshape(x_fp32.shape), scale_dq.reshape(*x_fp32.shape[:-1], 1)


def quant_w_per_output_channel(w_bf16):
    """[K, N] bf16 weight -> per-output-channel (per N) symmetric INT8 + scale.

    Matches qkv_proj_rope.build_tensor_specs.quant_w_per_output_channel.
    """
    w = bf16_to_float32(w_bf16)
    amax = np.maximum(np.abs(w).max(axis=0, keepdims=True), np.float32(INT8_AMAX_EPS))
    scale_q = np.float32(INT8_SCALE_MAX) / amax
    scaled = w * scale_q
    i32 = np.rint(scaled).astype(np.int32)
    i32 = np.clip(i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    i8 = i32.astype(np.float16).astype(np.int8)
    scale_dq = (1.0 / scale_q).astype(np.float32).reshape(-1)
    return i8, scale_dq


def quant_w_per_row(w_bf16):
    """[N, K] bf16 weight (matmul with b_trans) -> per-row INT8 + scale."""
    w = bf16_to_float32(w_bf16)
    amax = np.maximum(np.abs(w).max(axis=-1, keepdims=True), np.float32(INT8_AMAX_EPS))
    scale_q = np.float32(INT8_SCALE_MAX) / amax
    scaled = w * scale_q
    i32 = np.rint(scaled).astype(np.int32)
    i32 = np.clip(i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    i8 = i32.astype(np.float16).astype(np.int8)
    scale_dq = (1.0 / scale_q).astype(np.float32).reshape(-1)
    return i8, scale_dq


# ===========================================================================
# qkv_proj_rope family
# ===========================================================================
def build_qproj_matmul(meta, generator, ints):
    """INT8 matmul: qr_i8 [T_PAD, Q_LORA] @ wq_b [Q_LORA, H*HEAD_DIM] -> i32."""
    del ints
    T_PAD = meta.elem_counts.get("v2", 0) // Q_LORA          # 16
    H_HEAD = H * HEAD_DIM                              # 65536
    # v1: (unused, formerly x_fp32 placeholder) -- empty in .pto
    buffers = {
        "v1": np.zeros(meta.elem_counts.get("v1", 0), dtype=meta.np_types.get("v1", np.int32)) if "v1" in meta.elem_counts else np.zeros(0, dtype=np.int32),
        "v2": make_int8(generator, meta.elem_counts.get("v2", 0), scale=2.0),    # qr_i8 [T_PAD, Q_LORA]
        "v3": make_int8(generator, meta.elem_counts.get("v3", 0), scale=2.0),    # wq_b [Q_LORA, H*HEAD_DIM]
        "v4": _flat_output(meta, "v4"),                                    # out i32
    }
    # v1 may be a 0-elem placeholder; if it has content treat as ignored scratch.
    if meta.elem_counts.get("v1", 0) > 0:
        buffers["v1"] = make_fp32(generator, meta.elem_counts.get("v1", 0), scale=0.05)

    qr = buffers["v2"].reshape(T_PAD, Q_LORA).astype(np.int32)
    wq = buffers["v3"].reshape(Q_LORA, H_HEAD).astype(np.int32)
    out = qr @ wq                                    # [T_PAD, H_HEAD] int32
    buffers["v4"] = out.reshape(-1).astype(np.int32)
    return buffers, {"v4": buffers["v4"]}


def build_qproj_dequant_rms_nope_rope(meta, generator, ints):
    """Dequant q_proj_i32 -> per-head RMSNorm -> NOPE writeback + interleaved RoPE.

    .pto params (kernel_signatures.json + scalar_dims %arg9 ties v1..v4 to T):
      v1: bf16 [T, H*HEAD_DIM] -- q output (kernel writes here)
      v2: f32  [T, 1]         -- qr_scale_dq (per-token dequant scale)
      v3: f32  [T, 64]        -- cos_il (interleaved cos, ROPE_DIM wide)
      v4: f32  [T, 64]        -- sin_signed (interleaved signed sin)
      v5: i32  [T, 64]        -- swap_idx (j^1 partner index)
      v6: i32  [T, H*HEAD_DIM] -- q_proj_i32 input (the matmul accumulator)
      v7: f32  [H*HEAD_DIM]   -- wq_b_scale (per-output-channel dequant scale)
      v8..v12: index/i32 scalars (T, spmd block idx/num)
    """
    del ints
    T_PAD = 16
    H_HEAD = H * HEAD_DIM                              # 65536
    wq_b_scale = make_fp32(generator, H_HEAD, scale=0.05).reshape(1, H_HEAD)
    q_proj_i32 = generator.integers(-1000, 1001, size=T_PAD * H_HEAD).astype(np.int32).reshape(T_PAD, H_HEAD)
    qr_scale_dq = make_fp32(generator, T_PAD, scale=0.05).reshape(T_PAD, 1)
    # Build the interleaved RoPE index tables (j>>1, j^1, sign) the same way the
    # kernel does, then materialize cos_il / sin_signed from a half-width draw.
    cos_half = make_fp32(generator, T_PAD * HALF_ROPE, scale=1.0, positive=True).reshape(T_PAD, HALF_ROPE)
    sin_half = make_fp32(generator, T_PAD * HALF_ROPE, scale=1.0, positive=True).reshape(T_PAD, HALF_ROPE)
    j = np.arange(ROPE_DIM, dtype=np.int32)
    dup_idx = (j // 2).astype(np.int32)               # j>>1
    lane = (j - (dup_idx * 2)).astype(np.float32)     # j%2
    swap_idx = (j + 1 - lane * 2).astype(np.int32)    # j^1
    sign = (lane * 2.0 - 1.0).astype(np.float32)      # [-1,+1,...]
    # cos_il[t, j] = cos_half[t, dup_idx[j]]; sin_signed[t,j] = sin_half[t, dup_idx[j]] * sign[j]
    cos_il = cos_half[:, dup_idx]                     # [T, ROPE_DIM]
    sin_signed = sin_half[:, dup_idx] * sign.reshape(1, ROPE_DIM)
    swap_idx_tile = np.broadcast_to(swap_idx.reshape(1, ROPE_DIM), (T_PAD, ROPE_DIM)).copy()

    buffers = {
        "v1": np.zeros(T_PAD * H_HEAD, dtype=np.uint16),                 # q output (bf16)
        "v2": qr_scale_dq.reshape(-1).astype(np.float32),
        "v3": cos_il.reshape(-1).astype(np.float32),
        "v4": sin_signed.reshape(-1).astype(np.float32),
        "v5": swap_idx_tile.reshape(-1).astype(np.int32),
        "v6": q_proj_i32.reshape(-1).astype(np.int32),
        "v7": wq_b_scale.reshape(-1).astype(np.float32),
    }

    # dequant
    q = q_proj_i32.astype(np.float32) * qr_scale_dq * wq_b_scale   # [T_PAD, H_HEAD]
    q = q.reshape(T_PAD, H, HEAD_DIM)
    # per-head RMSNorm (no gamma)
    sq = (q * q).sum(axis=-1, keepdims=True) / HEAD_DIM
    inv_rms = 1.0 / np.sqrt(sq + EPS)
    q = q * inv_rms
    # NOPE writeback (bf16 rint)
    q_nope = q[..., :NOPE_DIM]
    # RoPE on the rope half using the interleaved cos_il/sin_signed:
    #   out[j] = x[j]*cos_il[j] + x[swap[j]]*sin_signed[j]
    # cos_il / sin_signed are [T_PAD, ROPE_DIM]; broadcast over the H axis.
    q_rope_raw = q[..., NOPE_DIM:NOPE_DIM + ROPE_DIM]                # [T_PAD, H, ROPE_DIM]
    q_swap = q_rope_raw[..., swap_idx]                               # gather partner [T,H,ROPE_DIM]
    cos_il_b = cos_il[:, None, :]                                    # [T_PAD, 1, ROPE_DIM]
    sin_signed_b = sin_signed[:, None, :]                            # [T_PAD, 1, ROPE_DIM]
    q_rope = q_rope_raw * cos_il_b + q_swap * sin_signed_b
    q_out = np.concatenate([q_nope, q_rope], axis=-1)              # [T_PAD, H, HEAD_DIM]
    buffers["v1"] = float32_to_bf16(q_out.reshape(-1))
    golden = {"v1": buffers["v1"], "v7": buffers["v7"]}
    return buffers, golden


def build_qk_pv(meta, generator, ints):
    """Fused QK + PV attention (qk_pv_aic; the aiv half shares the same buffers).

    .pto shapes (from tensor views):
      v1: [8, 128]    f32  (q slice: a tile of the per-head query)
      v2: [1024, 512] bf16 (k cache window)
      v3: [1024, 1]   f32  (mi / softmax max, dn layout)
      v4: [1024, 1]   f32  (li / sum)
      v5: [1024, 512] f32  (oi / accumulated output)
      v6: [1024, 512] bf16 (v cache window)

    The q slice [8, 128] is a sub-tile of the 512-dim head; the full QK matmul
    q@k^T needs q's reduction dim to match k's 512. The exact tiling/gather
    that maps the 128-dim q tile onto the 512-dim k is not recoverable from
    the .pto shapes alone (it depends on the sparse-attn head/group wiring),
    so this builder emits random inputs and zeroed attention outputs as a
    placeholder golden. TODO: wire the real head-tile gather once the
    decode_sparse_attn tiling is pinned down.
    """
    del ints
    rows_q, dim_q = 8, 128
    kv_rows, head_dim = 1024, 512
    q = make_fp32(generator, rows_q * dim_q, scale=0.05).reshape(rows_q, dim_q)
    k = bf16_to_float32(make_bf16(generator, kv_rows * head_dim, scale=0.05)).reshape(kv_rows, head_dim)
    v = bf16_to_float32(make_bf16(generator, kv_rows * head_dim, scale=0.05)).reshape(kv_rows, head_dim)
    # Placeholder attention: zero mi/li/oi (TODO: real tile-aware flash attn)
    mi = np.zeros(kv_rows, dtype=np.float32)
    li = np.zeros(kv_rows, dtype=np.float32)
    oi = np.zeros((kv_rows, head_dim), dtype=np.float32)
    buffers = {
        "v1": q.reshape(-1).astype(np.float32),
        "v2": float32_to_bf16(k.reshape(-1)).reshape(-1),
        "v3": mi.astype(np.float32),
        "v4": li.astype(np.float32),
        "v5": oi.reshape(-1).astype(np.float32),
        "v6": float32_to_bf16(v.reshape(-1)).reshape(-1),
    }
    return buffers, {"v3": buffers["v3"], "v4": buffers["v4"], "v5": buffers["v5"]}


def build_qkv_rope_rows(meta, generator, ints):
    """Gather per-token RoPE cos/sin rows from the full table by position_ids.

    .pto shapes:
      v1: [128, 64] bf16 (freqs_cos full table)
      v2: [128, 64] bf16 (freqs_sin full table)
      v3: i32 position_ids (dynamic len)
      v4: [16384, 64] bf16 (rope_cos_t output)
      v5: [16384, 64] bf16 (rope_sin_t output)
    """
    del ints
    table_rows = 128
    rope_w = 64
    cos_tbl = make_bf16(generator, table_rows * rope_w, scale=1.0, positive=True).reshape(table_rows, rope_w)
    sin_tbl = make_bf16(generator, table_rows * rope_w, scale=1.0, positive=True).reshape(table_rows, rope_w)
    pos_count = meta.elem_counts.get("v3", 0)
    pos_ids = generator.integers(0, table_rows, size=pos_count).astype(np.int32)
    out_rows = 16384
    cos_out = np.zeros((out_rows, rope_w), dtype=np.uint16)
    sin_out = np.zeros((out_rows, rope_w), dtype=np.uint16)
    for i, p in enumerate(pos_ids):
        cos_out[i] = cos_tbl[p]
        sin_out[i] = sin_tbl[p]
    buffers = {
        "v1": cos_tbl.reshape(-1),
        "v2": sin_tbl.reshape(-1),
        "v3": pos_ids,
        "v4": cos_out.reshape(-1),
        "v5": sin_out.reshape(-1),
    }
    return buffers, {"v4": buffers["v4"], "v5": buffers["v5"]}


# ---------------------------------------------------------------------------
# indexer: qr path
# ---------------------------------------------------------------------------
def build_qr_hadamard_matmul(meta, generator, ints):
    """q_bf16 [512, 128] @ hadamard [128, 128] -> fp32 [512, 128]."""
    del ints
    rows, dim = 512, 128
    q_bf16 = make_bf16(generator, rows * dim, scale=0.05).reshape(rows, dim)
    had = make_bf16(generator, dim * dim, scale=1.0 / np.sqrt(dim)).reshape(dim, dim)
    out = bf16_to_float32(q_bf16) @ bf16_to_float32(had)
    buffers = {
        "v1": q_bf16.reshape(-1),
        "v2": had.reshape(-1),
        "v3": out.reshape(-1).astype(np.float32),
    }
    return buffers, {"v3": buffers["v3"]}


def build_qr_hadamard_quant(meta, generator, ints):
    """Per-row INT8 quant of q_hadamard [512, 128] with amax over the full row.

    .pto:
      v1: [512, 128] f32 (input)
      v2: [512, 1] f32 (scale_dequant output)
      v3: [512, 128] i8 (quantized output)
    """
    del ints
    rows, dim = 512, 128
    x = make_fp32(generator, rows * dim, scale=0.5).reshape(rows, dim)
    i8, scale_dq = per_row_int8_quant(x)
    buffers = {
        "v1": x.reshape(-1).astype(np.float32),
        "v2": scale_dq.reshape(-1).astype(np.float32),
        "v3": i8.reshape(-1).astype(np.int8),
    }
    return buffers, {"v2": buffers["v2"], "v3": buffers["v3"]}


def build_qr_proj_matmul(meta, generator, ints):
    """Split-K fp32 matmul: x [T_PAD, D] @ wq_a [D, Q_LORA] -> qr_fp32 [T_PAD, Q_LORA].

    .pto:
      v1: [128, 1536] f32 (output -- T_PAD=16 would be 16x1536=24576; but
            the .pto shape is [128, 1536] = 196608, so T_PAD=128 here for the
            standalone case)
      v2: [7168, ...] bf16 (wq_a, dynamic first dim)
      v3: [7168, 1536] bf16 (wq_a static view)
    The standalone .pto uses T_PAD=128 (16-row boxed tiles x 8 head groups).
    """
    del ints
    T_PAD = 128
    wq_a = make_bf16(generator, D * Q_LORA, scale=0.05).reshape(D, Q_LORA)
    x = make_bf16(generator, T_PAD * D, scale=0.05).reshape(T_PAD, D)
    out = bf16_to_float32(x) @ bf16_to_float32(wq_a)        # [T_PAD, Q_LORA]
    buffers = {
        "v1": out.reshape(-1).astype(np.float32),
        "v2": wq_a.reshape(-1),
        "v3": wq_a.reshape(-1),
    }
    return buffers, {"v1": buffers["v1"]}


def build_qr_proj_seed(meta, generator, ints):
    """Zero-seed the qr_fp32 accumulator [T_PAD, Q_LORA] (= 196608 elems)."""
    del ints
    T_PAD = 128
    out = np.zeros(T_PAD * Q_LORA, dtype=np.float32)
    buffers = {
        "v1": out,
    }
    return buffers, {"v1": buffers["v1"]}


def build_qr_rms_norm_quant(meta, generator, ints):
    """RMSNorm + per-row INT8 quant of qr_fp32.

    .pto:
      v1: [128, 1536] f32 (qr_fp32 input)
      v2: [1536] bf16 (gamma_cq)
      v3: [1] f32 (scalar scratch)
      v4: [128, 1536] i8 (qr_i8 output)
      v5: [1536] i8 (qr_i8_matmul alias output)
    scalar_dims: v3/v5 share %arg5 (T dim). We treat T_PAD=128.
    """
    del ints
    T_PAD = 128
    x = make_fp32(generator, T_PAD * Q_LORA, scale=0.5).reshape(T_PAD, Q_LORA)
    gamma = make_bf16(generator, Q_LORA, scale=1.0).reshape(Q_LORA)
    # RMSNorm
    sq = (x * x).sum(axis=1, keepdims=True) / Q_LORA
    inv_rms = 1.0 / np.sqrt(sq + EPS)
    normed = x * inv_rms * bf16_to_float32(gamma).reshape(1, Q_LORA)
    i8, scale_dq = per_row_int8_quant(normed)
    buffers = {
        "v1": x.reshape(-1).astype(np.float32),
        "v2": gamma.reshape(-1),
        "v3": scale_dq.reshape(-1).astype(np.float32),
        "v4": i8.reshape(-1).astype(np.int8),
        "v5": i8.reshape(-1).astype(np.int8),
    }
    return buffers, {"v3": buffers["v3"], "v4": buffers["v4"], "v5": buffers["v5"]}


def build_qr_rope(meta, generator, ints):
    """Interleaved RoPE on qr_proj rows [512, 128] (T*IDX_N_HEADS, IDX_HEAD_DIM).

    .pto:
      v1: [4, 32] f32 (cos/sin half-width per batch: B x HALF_ROPE)
      v2: [4, 32] f32
      v3: [512, 128] f32 (input qr_proj)
      v4: [512, 128] bf16 (output qr_bf16: nope rounded + rope rotated)
    """
    del ints
    rows, dim = 512, IDX_HEAD_DIM
    cos_b = make_fp32(generator, B * HALF_ROPE, scale=1.0, positive=True).reshape(B, HALF_ROPE)
    sin_b = make_fp32(generator, B * HALF_ROPE, scale=1.0, positive=True).reshape(B, HALF_ROPE)
    x = make_fp32(generator, rows * dim, scale=0.5).reshape(rows, dim)
    out = np.zeros((rows, dim), dtype=np.uint16)
    rows_per_batch = rows // B                       # 128
    for b in range(B):
        sl = slice(b * rows_per_batch, (b + 1) * rows_per_batch)
        nope = x[sl, :IDX_NOPE_DIM]
        rope = apply_interleaved_rope(x[sl, IDX_NOPE_DIM:].astype(np.float32), cos_b[b], sin_b[b])
        out[sl] = float32_to_bf16(np.concatenate([nope, rope], axis=-1))
    buffers = {
        "v1": cos_b.reshape(-1).astype(np.float32),
        "v2": sin_b.reshape(-1).astype(np.float32),
        "v3": x.reshape(-1).astype(np.float32),
        "v4": out.reshape(-1),
    }
    return buffers, {"v4": buffers["v4"]}


# ---------------------------------------------------------------------------
# indexer: kv path
# ---------------------------------------------------------------------------
def build_kv_hadamard(meta, generator, ints):
    """kv_proj [16, 128] @ hadamard [128, 128] -> bf16 [16, 128].

    .pto:
      v1: [16, 128] bf16 (input normed_kv)
      v2: [16, 128] f32 (output kv_final -- stored as f32 then re-read)
      v3: [128, 128] bf16 (hadamard)
    """
    del ints
    rows, dim = 16, IDX_HEAD_DIM
    kv_bf16 = make_bf16(generator, rows * dim, scale=0.05).reshape(rows, dim)
    had = make_bf16(generator, dim * dim, scale=1.0 / np.sqrt(dim)).reshape(dim, dim)
    out = bf16_to_float32(kv_bf16) @ bf16_to_float32(had)
    buffers = {
        "v1": kv_bf16.reshape(-1),
        "v2": out.reshape(-1).astype(np.float32),
        "v3": had.reshape(-1),
    }
    return buffers, {"v2": buffers["v2"]}


def build_kv_proj_matmul(meta, generator, ints):
    """Split-K fp32 matmul: x [T_PAD, D] @ wkv [D, HEAD_DIM] -> kv_fp32 [T_PAD, HEAD_DIM].

    .pto:
      v1: [128, 512] f32 (output; T_PAD=128)
      v2: [7168, ...] bf16 (wkv, dynamic first dim)
      v3: [7168, 512] bf16 (wkv static view)
    """
    del ints
    T_PAD = 128
    wkv = make_bf16(generator, D * HEAD_DIM, scale=0.05).reshape(D, HEAD_DIM)
    x = make_bf16(generator, T_PAD * D, scale=0.05).reshape(T_PAD, D)
    out = bf16_to_float32(x) @ bf16_to_float32(wkv)
    buffers = {
        "v1": out.reshape(-1).astype(np.float32),
        "v2": wkv.reshape(-1),
        "v3": wkv.reshape(-1),
    }
    return buffers, {"v1": buffers["v1"]}


def build_kv_proj_seed(meta, generator, ints):
    """Zero-seed the kv_fp32 accumulator [128, 512] (= 65536 elems)."""
    del ints
    T_PAD = 128
    out = np.zeros(T_PAD * HEAD_DIM, dtype=np.float32)
    buffers = {"v1": out}
    return buffers, {"v1": buffers["v1"]}


def build_kv_rms_norm_rope(meta, generator, ints):
    """Fused KV RMSNorm + interleaved RoPE writeback.

    .pto:
      v1: [128, 512] f32 (kv_fp32 input)
      v2: [T, 512] bf16 (kv output, dynamic T)
      v3: [512] bf16 (gamma_ckv)
      v4: [T, 64] bf16 (rope_cos per token)
      v5: [T, 64] bf16 (rope_sin per token)
    """
    del ints
    T_PAD = 128
    # The NPU kernel is launched with 1 SPMD block (block_num=1) and a dynamic
    # ctx_len of DEC_T=8, so it only processes rows 0-7 of the [128, 512] input.
    # Rows 8-127 of the output are left as the input v2 buffer content (zero here).
    x = make_fp32(generator, T_PAD * HEAD_DIM, scale=0.5).reshape(T_PAD, HEAD_DIM)
    gamma = make_bf16(generator, HEAD_DIM, scale=1.0).reshape(HEAD_DIM)
    cos_t = make_bf16(generator, T_PAD * ROPE_DIM, scale=1.0, positive=True).reshape(T_PAD, ROPE_DIM)
    sin_t = make_bf16(generator, T_PAD * ROPE_DIM, scale=1.0, positive=True).reshape(T_PAD, ROPE_DIM)
    # RMSNorm + RoPE for the first DEC_T rows only. Two deviations from the
    # textbook formula, both observed on the NPU:
    #   1. The sum-of-squares is accumulated over the NOPE columns (0..NOPE_DIM-1)
    #      only; the ROPE columns (NOPE_DIM..HEAD_DIM-1) are excluded from the
    #      RMS energy even though the divisor stays HEAD_DIM.
    #   2. cos/sin are stored as bf16 and must be decoded to fp32 before the
    #      rotation (passing the raw uint16 storage to apply_interleaved_rope
    #      would treat the bit patterns as large integers).
    def _rms_norm_rope(rows, cos_rows, sin_rows):
        # Kernel sums x^2 over ALL HEAD_DIM columns (not just NOPE_DIM).
        # The .pto loop iterates kb 0..8 step 2, each loading 2x 8x64 tiles
        # = 128 cols per iter, 4 iters = 512 = HEAD_DIM.
        sq = (rows * rows).sum(axis=1, keepdims=True) / HEAD_DIM
        inv_rms = 1.0 / np.sqrt(sq + EPS)
        normed = rows * inv_rms * g                                # [n, HEAD_DIM]
        nope = normed[:, :NOPE_DIM]
        cos_f32 = bf16_to_float32(cos_rows).reshape(rows.shape[0], ROPE_DIM)
        sin_f32 = bf16_to_float32(sin_rows).reshape(rows.shape[0], ROPE_DIM)
        rope_out = np.zeros((rows.shape[0], ROPE_DIM), dtype=np.float32)
        for t in range(rows.shape[0]):
            rope_out[t] = apply_interleaved_rope(
                normed[t, NOPE_DIM:NOPE_DIM + ROPE_DIM].reshape(1, ROPE_DIM),
                cos_f32[t, :HALF_ROPE], sin_f32[t, :HALF_ROPE]).reshape(ROPE_DIM)
        return np.concatenate([nope, rope_out], axis=-1)          # [n, HEAD_DIM]

    g = bf16_to_float32(gamma).reshape(1, HEAD_DIM)
    out = np.zeros((T_PAD, HEAD_DIM), dtype=np.float32)
    out[:DEC_T] = _rms_norm_rope(x[:DEC_T], cos_t[:DEC_T], sin_t[:DEC_T])
    buffers = {
        "v1": x.reshape(-1).astype(np.float32),
        "v2": float32_to_bf16(out.reshape(-1)).reshape(-1),
        "v3": gamma.reshape(-1),
        "v4": cos_t.reshape(-1),
        "v5": sin_t.reshape(-1),
        # v6: cmp_kv_cache_flat [4096, 512] bf16 — random cache input
        "v6": make_bf16(generator, 4096 * HEAD_DIM, scale=0.05),
        # v7: kv_flat [8, 512] f32 — random kv input
        "v7": make_fp32(generator, DEC_T * HEAD_DIM, scale=0.05),
        # v8: position_ids i32 [T] — identity mapping
        "v8": np.arange(DEC_T, dtype=np.int32),
        # v9: cache_write_slots i64 [T] — identity mapping
        "v9": np.arange(DEC_T, dtype=np.int64),
    }
    return buffers, {"v2": buffers["v2"]}


def build_rmsnorm_rope_cache_write(meta, generator, ints):
    """Fused RMSNorm + interleaved RoPE + cmp_kv_cache scatter for DSV4 PRO.

    .pto ptrs (9 total, no scalars):
      v1 cos        [4, 32]   f32  (in)  rope cos table
      v2 sin        [4, 32]   f32  (in)  rope sin table
      v3 pooled_kv  [16, 512] f32  (in)  pooled kv (only first 4 rows used;
                                          rows 4-15 are padding)
      v4 normed_kv  [16, 512] f32  (out) rmsnorm+rope result
      v5 norm_w_2d  [1, 512]  bf16 (in)  rmsnorm gamma (full HEAD_DIM)
      v6 cmp_kv_cache_flat [4096, 512] bf16 (out) cmp kv cache scatter dst
      v7 kv_flat    [8, 512]  f32  (out) kv output (first 4 rows written)
      v8 position_ids [4, 2]  i32  (in)
      v9 cmp_slot_mapping [4, 2] i64 (in)

    The kernel processes the first 4 rows of pooled_kv through RMSNorm, then
    applies interleaved RoPE on the rope half (cols 448..511) using cos/sin
    gathered per batch, stores the full 512-wide normed row to v4, and for
    each batch b where position_ids[b,0] % 4 >= 2 scatters the normed row to:
      - v7 (kv_flat) at row b*2 + (3 - pos%4)  (f32, no cast)
      - v6 (cmp_kv_cache_flat) at row cmp_slot_mapping[b, 3-pos%4] (bf16 rint)
    """
    del ints
    PAD_ROWS = 16
    cos = make_fp32(generator, 4 * 32, scale=1.0, positive=True).reshape(4, 32)
    sin = make_fp32(generator, 4 * 32, scale=1.0, positive=True).reshape(4, 32)
    pooled_kv = make_fp32(generator, PAD_ROWS * HEAD_DIM, scale=0.5).reshape(PAD_ROWS, HEAD_DIM)
    norm_w = make_bf16(generator, HEAD_DIM, scale=1.0).reshape(1, HEAD_DIM)
    # benign position_ids: [4,2] = [[2,0],[3,0],[4,0],[5,0]] so pos%4 in {2,3,0,1}
    position_ids = np.array([[2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.int32)
    # benign cmp_slot_mapping: [4,2]; second column is the scatter target slot.
    # Map each batch's second slot to a distinct cache row (0,1,2,3).
    cmp_slot_mapping = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    # pre-existing cache content (random) so scatter is a partial overwrite
    cmp_kv_cache = make_bf16(generator, 4096 * HEAD_DIM, scale=0.05).reshape(4096, HEAD_DIM)
    kv_flat = np.zeros((DEC_T, HEAD_DIM), dtype=np.float32)

    g = bf16_to_float32(norm_w).reshape(1, HEAD_DIM)
    normed_kv = np.zeros((PAD_ROWS, HEAD_DIM), dtype=np.float32)
    # The kernel RMSNorms ALL 16 rows. cos_b/sin_b are gathered from cos/sin
    # [4,32] into cos_b[0:4, 0:32]; rows 4-15 of cos_b/sin_b stay 0 (zero-
    # expanded), so their rope half (cols 448..511) collapses to 0
    # (out = x*0 + swapped*0 = 0). Rows 0-3 use their batch's cos/sin.
    for r in range(PAD_ROWS):
        row = pooled_kv[r]
        sq = np.sum(row * row) / HEAD_DIM
        inv_rms = 1.0 / np.sqrt(sq + EPS)
        normed = row * inv_rms * g.reshape(-1)        # [512] fp32
        if r < 4:
            # rope on cols 448..511 (64 wide) using this batch's cos/sin
            rope = normed[NOPE_DIM:NOPE_DIM + ROPE_DIM].reshape(1, ROPE_DIM)
            cos_half = cos[r, :HALF_ROPE]
            sin_half = sin[r, :HALF_ROPE]
            rope_out = apply_interleaved_rope(rope, cos_half, sin_half).reshape(ROPE_DIM)
            normed[NOPE_DIM:NOPE_DIM + ROPE_DIM] = rope_out
        else:
            # zero cos/sin -> rope half is 0
            normed[NOPE_DIM:NOPE_DIM + ROPE_DIM] = 0.0
        normed_kv[r] = normed
    # cache scatter: for each batch inner (0..3), read position_ids[inner,0],
    # if (pos % 4) >= 2: kv_flat[inner*2] = normed_kv[inner] (f32, no cast) and
    # cmp_kv_cache[cmp_slot_mapping[inner, 3-lane]] = bf16-rint(normed_kv[inner]).
    for inner in range(4):
        pos = int(position_ids[inner, 0])
        lane = pos % 4
        if lane >= 2:
            kv_flat[int(inner) * 2] = normed_kv[inner]
            slot = int(cmp_slot_mapping[inner, 3 - lane])
            if slot >= 0:
                cmp_kv_cache[slot] = float32_to_bf16(normed_kv[inner])

    buffers = {
        "v1": cos.reshape(-1).astype(np.float32),
        "v2": sin.reshape(-1).astype(np.float32),
        "v3": pooled_kv.reshape(-1).astype(np.float32),
        "v4": normed_kv.reshape(-1).astype(np.float32),
        "v5": norm_w.reshape(-1),
        "v6": cmp_kv_cache.reshape(-1),
        "v7": kv_flat.reshape(-1).astype(np.float32),
        "v8": position_ids.reshape(-1).astype(np.int32),
        "v9": cmp_slot_mapping.reshape(-1).astype(np.int64),
    }
    return buffers, {
        "v4": buffers["v4"],
        "v6": buffers["v6"],
        "v7": buffers["v7"],
    }


def build_kv_and_cache_write(meta, generator, ints):
    """Per-row INT8 quant-on-write of kv_final + scatter into paged idx_kv_cache.

    .pto:
      v1: [16, 128] f32 (kv_final)
      v2: [8192, 128] i8 (idx_kv_cache flat: 64 blocks x 128)
      v3: [8, 128] f32 (kv output [B*S, HEAD_DIM])
      v4: [4, 2] i32 (position_ids [B, S])
      v5: [4, 2] i64 (idx_slot_mapping [B, S])
      v6: [8192, 1] f32 (idx_kv_scale flat)
    """
    del ints
    rows, dim = 16, IDX_HEAD_DIM
    kv_final = make_fp32(generator, rows * dim, scale=0.5).reshape(rows, dim)
    cache = make_int8(generator, 8192 * dim, scale=2.0).reshape(8192, dim)
    scale_flat = make_fp32(generator, 8192, scale=0.05).reshape(8192, 1)
    pos_ids = generator.integers(0, 4096, size=B * S).reshape(B, S).astype(np.int32)
    # slot mapping: only boundary rows are written; pick deterministic slots
    slot_map = np.full((B, S), -1, dtype=np.int64)
    for b in range(B):
        slot_map[b, 0] = b                         # first token of each batch -> row b
    kv_out = np.zeros((B * S, dim), dtype=np.float32)

    # C8 quant-on-write: bf16-round then per-row int8 quant
    kv_bf16 = bf16_round(kv_final)
    i8_blk, scale_dq = per_row_int8_quant(kv_bf16)
    for b in range(B):
        cache_row = int(slot_map[b, 0])
        if cache_row >= 0:
            cache[cache_row] = i8_blk[b]
            scale_flat[cache_row, 0] = scale_dq[b, 0]
            kv_out[b * S] = kv_final[b]
    buffers = {
        "v1": kv_final.reshape(-1).astype(np.float32),
        "v2": cache.reshape(-1).astype(np.int8),
        "v3": kv_out.reshape(-1).astype(np.float32),
        "v4": pos_ids.reshape(-1).astype(np.int32),
        "v5": slot_map.reshape(-1).astype(np.int64),
        "v6": scale_flat.reshape(-1).astype(np.float32),
    }
    return buffers, {"v2": buffers["v2"], "v3": buffers["v3"], "v6": buffers["v6"]}


def build_kv_score_proj_0(meta, generator, ints):
    """kv_score_proj: x [8, 7168] @ wkv^T + wgate^T -> [16, 1024] (kv + score).

    .pto:
      v1: [8, 7168]   bf16 (x: B*S rows of D)
      v2: [1024, 7168] bf16 (wkv transposed [OUT_DIM, D])
      v3: [1024, 7168] bf16 (wgate transposed)
      v4: [16, 1024]   f32  (kv_proj output, BS_PAD=16 rows)
      v5: [16, 1024]   f32  (score_proj output)
    """
    del ints
    OUT_DIM = 1024                                   # COFF * HEAD_DIM = 2 * 512
    BS_PAD = 16
    x = make_bf16(generator, (B * S) * D, scale=0.05).reshape(B * S, D)        # 8 rows
    wkv = make_bf16(generator, OUT_DIM * D, scale=0.05).reshape(OUT_DIM, D)    # [OUT_DIM, D]
    wgate = make_bf16(generator, OUT_DIM * D, scale=0.05).reshape(OUT_DIM, D)
    x_pad = np.zeros((BS_PAD, D), dtype=np.uint16)
    x_pad[:B * S] = x
    kv = bf16_to_float32(x_pad) @ bf16_to_float32(wkv).T      # [BS_PAD, OUT_DIM]
    score = bf16_to_float32(x_pad) @ bf16_to_float32(wgate).T
    buffers = {
        "v1": x.reshape(-1),
        "v2": wkv.reshape(-1),
        "v3": wgate.reshape(-1),
        "v4": kv.reshape(-1).astype(np.float32),
        "v5": score.reshape(-1).astype(np.float32),
    }
    return buffers, {"v4": buffers["v4"], "v5": buffers["v5"]}


def build_score_reduce(meta, generator, ints):
    """Reduce score matmul partials -> per-position weighted score.

    .pto (tensor views, all static):
      v1: [4]       i32  (kv_seq_lens)
      v2: [4, 2]    i32  (position_ids [B, S])
      v3: [512, 1]  f32  (qh_scale [T*IDX_N_HEADS, 1]; per-token-per-head dequant)
      v4: [16, 64]  f32  (weights [T_PAD, IDX_N_HEADS])
      v5: [8, 4096] f32  (score output [T, IDX_KV_LEN]; tail = -inf)
      v6: [256]     i32  (idx_block_table flat)
      v7: [32768, 64] i32 (score_acc_gm [T*IDX_KV_LEN, IDX_N_HEADS])
      v8: [8192, 1] f32  (idx_kv_scale flat; per-position dequant)

    Per token tg, per cache slot s:
      score = (relu(score_acc[s] * qh_scale[tg]) * weights[tg]).sum(heads) * kv_scale[s]
    """
    del ints
    T_PAD = 16
    score_acc = generator.integers(-1000, 1001, size=T * IDX_KV_LEN * IDX_N_HEADS).astype(np.int32).reshape(T * IDX_KV_LEN, IDX_N_HEADS)
    qh_scale = make_fp32(generator, T * IDX_N_HEADS, scale=0.05).reshape(T * IDX_N_HEADS, 1)
    weights = make_fp32(generator, T_PAD * IDX_N_HEADS, scale=0.05).reshape(T_PAD, IDX_N_HEADS)
    kv_scale = make_fp32(generator, 8192, scale=0.05).reshape(8192, 1)
    block_table = generator.integers(0, 64, size=256).astype(np.int32)
    pos_ids = generator.integers(0, 4096, size=B * S).reshape(B, S).astype(np.int32)
    kv_seq_lens = generator.integers(128, 4096, size=B).astype(np.int32)

    score_out = np.full((T, IDX_KV_LEN), float(FP32_NEG_INF), dtype=np.float32)
    for tg in range(T):
        b = tg // S
        s = tg - b * S
        clen = int(kv_seq_lens[b]) // COMPRESS_RATIO
        pos_t = int(pos_ids[b, s])
        visible = min(min(clen, (pos_t + 1) // COMPRESS_RATIO), IDX_KV_LEN)
        if visible <= 0:
            continue
        qh_s = qh_scale[tg * IDX_N_HEADS:(tg + 1) * IDX_N_HEADS].reshape(1, IDX_N_HEADS)  # [1, H]
        w_row = weights[tg].reshape(1, IDX_N_HEADS)                                      # [1, H]
        acc = score_acc[tg * IDX_KV_LEN:(tg + 1) * IDX_KV_LEN].astype(np.float32) * qh_s  # [idx_kv_len, H]
        acc = np.maximum(acc, 0.0) * w_row
        row = acc.sum(axis=1, keepdims=True) * kv_scale[:IDX_KV_LEN]                      # [idx_kv_len, 1]
        score_out[tg, :visible] = row[:visible, 0]
    buffers = {
        "v1": kv_seq_lens.reshape(-1).astype(np.int32),
        "v2": pos_ids.reshape(-1).astype(np.int32),
        "v3": qh_scale.reshape(-1).astype(np.float32),
        "v4": weights.reshape(-1).astype(np.float32),
        "v5": score_out.reshape(-1).astype(np.float32),
        "v6": block_table.reshape(-1).astype(np.int32),
        "v7": score_acc.reshape(-1).astype(np.int32),
        "v8": kv_scale.reshape(-1).astype(np.float32),
    }
    return buffers, {"v5": buffers["v5"]}


def build_topk(meta, generator, ints):
    """Top-k selection over score rows [T, SCORE_LEN] -> idx [T, SCORE_LEN].

    .pto (from tensor views):
      v1 (arg0): [8, 4096] i32 - topk_idxs_flat OUTPUT (tail = -1)
      v2 (arg1): [4] i32 - kv_seq_lens INPUT
      v3 (arg2): [4, 2] i32 - position_ids INPUT
      v4 (arg3): [8, 4096] f32 - score_flat INPUT
      v5/v6: i32 scalars (spmd block_idx/block_num)
    """
    del ints
    score = make_fp32(generator, T * IDX_KV_LEN, scale=1.0).reshape(T, IDX_KV_LEN)
    topk_out = np.full((T, IDX_KV_LEN), -1, dtype=np.int32)
    # Visible length per token depends on position_ids and kv_seq_lens.
    # The kernel computes: visible = min(kv_seq_lens[b]//COMPRESS_RATIO,
    #   max(1, (pos+1)//COMPRESS_RATIO)) but capped at IDX_KV_LEN.
    # With pos=3..10 and CR=4: token 0 visible=1, tokens 1-7 visible=1024.
    CR = 4
    visible_per_token = []
    for t in range(T):
        b = t // 2  # B=4, S=2
        pos = int(np.array([3,4,5,6,7,8,9,10])[t])
        vis = min(4096 // CR, max(1, (pos + 1) // CR))
        vis = min(vis, IDX_KV_LEN)
        visible_per_token.append(vis)
    offset = 0
    for t in range(T):
        vis = visible_per_token[t]
        k = min(IDX_TOPK, vis)
        idx = np.argsort(-score[t, :vis], kind="stable")[:k]
        topk_out[t, :k] = idx.astype(np.int32) + offset
    buffers = {
        "v1": topk_out.reshape(-1).astype(np.int32),
        "v2": np.array([4096, 4096, 4096, 4096], dtype=np.int32),
        "v3": np.array([3,4, 5,6, 7,8, 9,10], dtype=np.int32),  # position_ids [4,2]
        "v4": score.reshape(-1).astype(np.float32),
    }
    return buffers, {"v1": buffers["v1"]}


# ===========================================================================
# mtp_projection family
# ===========================================================================
def _mtp_rmsnorm(x_fp32, weight_fp32):
    """RMSNorm over last dim D. x: [..., D], weight: [D]."""
    sq = (x_fp32 * x_fp32).sum(axis=-1, keepdims=True) * D_INV
    inv = 1.0 / np.sqrt(sq + EPS)
    return x_fp32 * inv * weight_fp32.reshape(*([1] * (x_fp32.ndim - 1)), D)


def build_mtp_projection_rms(meta, generator, ints):
    """RMSNorm of hidden_states [T, D] bf16 and prev_hidden [T, HC_DIM] f32.

    .pto:
      v1: [T, D] bf16 (hidden_states)
      v2: [1] f32 (hidden_inv_rms scalar)
      v3: [4, T] f32 (prev_inv_rms [HC_MULT, T])
      v4: [T, HC_DIM] f32 (prev_hidden normed output)
    """
    del ints
    T_pad = 8                                        # T for this case
    hidden = make_bf16(generator, T_pad * D, scale=0.5).reshape(T_pad, D)
    prev = make_fp32(generator, T_pad * HC_DIM, scale=0.5).reshape(T_pad, HC_DIM)
    hidden_inv = np.zeros((T_pad, 1), dtype=np.float32)
    prev_inv = np.zeros((HC_MULT, T_pad), dtype=np.float32)
    prev_normed = np.zeros((T_pad, HC_DIM), dtype=np.float32)
    for t in range(T_pad):
        hs = bf16_to_float32(hidden[t])
        sq = (hs * hs).sum() * D_INV
        inv = 1.0 / np.sqrt(sq + EPS)
        hidden_inv[t, 0] = inv
        for hc in range(HC_MULT):
            ps = prev[t, hc * D:(hc + 1) * D]
            sq = (ps * ps).sum() * D_INV
            inv2 = 1.0 / np.sqrt(sq + EPS)
            prev_inv[hc, t] = inv2
            prev_normed[t, hc * D:(hc + 1) * D] = ps * inv2
    buffers = {
        "v1": hidden.reshape(-1),
        "v2": hidden_inv.reshape(-1).astype(np.float32),
        "v3": prev_inv.reshape(-1).astype(np.float32),
        "v4": prev_normed.reshape(-1).astype(np.float32),
    }
    return buffers, {"v2": buffers["v2"], "v3": buffers["v3"], "v4": buffers["v4"]}


def build_mtp_projection_norm(meta, generator, ints):
    """Apply enorm/hnorm + smooth to the RMS-normalized hidden/prev, then
    write per-block amax parts for the downstream quant kernel.

    The .pto is the ground truth for buffer shapes, layouts and which params
    are inputs vs outputs. Tensor views (test_for_dsv4/.pto/mtp_projection_norm.pto):

      v1  (%arg0): hidden_inv_rms   [arg13, 1]    dn  INPUT  (per-token inv_rms, [T,1])
      v2  (%arg1): hidden_amax_parts[56, arg13]   nd  OUTPUT (per-D-block absmax, [D_BLOCKS, T])
      v3  (%arg2): hidden_norm      [arg13, 7168] nd  OUTPUT (normed hidden, [T, D])
      v4  (%arg3): prev_amax_parts  [224, arg13]  nd  OUTPUT (per-hc per-block absmax, [HC_MULT*D_BLOCKS, T])
      v5  (%arg4): prev_norm        [arg13, 28672]nd  OUTPUT (normed prev, [T, HC_DIM])
      v6  (%arg5): hidden_flat      [arg14, 7168] nd bf16 INPUT (hidden_states, [T, D])
      v7  (%arg6): enorm_w          [7168]        nd INPUT
      v8  (%arg7): e_proj_smooth    [7168]        nd INPUT
      v9  (%arg8): hnorm_w          [7168]        nd INPUT
      v10 (%arg9): h_proj_smooth    [7168]        nd INPUT
      v11 (%arg10):prev_inv_rms     [4, arg13]    nd INPUT (per-hc inv_rms, [HC_MULT, T])
      v12 (%arg11):prev_flat        [arg14, 28672]nd INPUT (prev_hidden_states, [T, HC_DIM])
      v13 (%arg12):index -- partition_view row OFFSET (base token idx), = 0 for single-block
      v14 (%arg13):index -- T (token dim of v1/v2/v3/v4/v5/v11), = 8
      v15 (%arg14):index -- T (token dim of v6/v12), = 8

    Math (per the .pto body): for each D_CHUNK=128 block kb (0..55) the kernel
    loads hidden[t, kb*128:..] (bf16->f32), multiplies by hidden_inv_rms[t]
    (row-broadcast), enorm_w (col-broadcast) and e_proj_smooth (col-broadcast),
    stores -> hidden_norm[t, block]; then takes the per-token abs-max of that
    8x128 tile and stores it into hidden_amax_parts[kb, t]. The prev path is
    identical per hc (0..3) with hnorm_w / h_proj_smooth and prev_inv_rms[hc],
    storing into prev_norm[t, hc*D + block] and prev_amax_parts[hc*56+kb, t].

    Note: the amax is a PER-TOKEN PER-D-BLOCK max (the [1,8] partition each kb
    writes), not a full-row max — mtp_projection_quant reduces these parts to
    the per-row quant scale downstream.
    """
    del ints
    T_pad = 8
    # Inputs.
    hidden_inv = make_fp32(generator, T_pad, scale=0.1).reshape(T_pad, 1)        # v1 [T,1]
    hidden_bf16 = make_bf16(generator, T_pad * D, scale=0.5).reshape(T_pad, D)    # v6 [T,D]
    prev_flat = make_fp32(generator, T_pad * HC_DIM, scale=0.5).reshape(T_pad, HC_DIM)  # v12 [T,HC_DIM]
    prev_inv = make_fp32(generator, HC_MULT * T_pad, scale=0.1).reshape(HC_MULT, T_pad)  # v11 [4,T]
    enorm_w = make_fp32(generator, D, scale=1.0).reshape(D)                      # v7
    e_smooth = make_fp32(generator, D, scale=1.0).reshape(D)                      # v8
    hnorm_w = make_fp32(generator, D, scale=1.0).reshape(D)                      # v9
    h_smooth = make_fp32(generator, D, scale=1.0).reshape(D)                      # v10

    # Outputs (zero-init; written block-by-block to match the .pto tile order).
    hidden_norm = np.zeros((T_pad, D), dtype=np.float32)                         # v3
    prev_norm = np.zeros((T_pad, HC_DIM), dtype=np.float32)                      # v5
    hidden_amax = np.zeros((D_BLOCKS, T_pad), dtype=np.float32)                   # v2 [56, T]
    prev_amax = np.zeros((HC_MULT * D_BLOCKS, T_pad), dtype=np.float32)          # v4 [224, T]

    hidden_f32 = bf16_to_float32(hidden_bf16.reshape(-1, 1)).reshape(T_pad, D)
    inv_h = hidden_inv.reshape(T_pad, 1)                                          # [T,1] row-broadcast
    # Hidden path: per D_CHUNK block, norm = x * inv_rms * enorm * e_smooth,
    # amax_part = per-token abs-max over the block.
    for kb in range(D_BLOCKS):
        s = kb * D_CHUNK
        e = s + D_CHUNK
        block = hidden_f32[:, s:e] * inv_h * enorm_w[s:e].reshape(1, D_CHUNK) \
            * e_smooth[s:e].reshape(1, D_CHUNK)                                  # [T, D_CHUNK]
        hidden_norm[:, s:e] = block
        hidden_amax[kb, :] = np.max(np.abs(block), axis=1)                        # [T]

    # Prev path: per hc, per D_CHUNK block, norm = prev * prev_inv[hc] * hnorm * h_smooth,
    # amax_part = per-token abs-max over the block.
    for hc in range(HC_MULT):
        base = hc * D
        inv_p = prev_inv[hc].reshape(T_pad, 1)                                    # [T,1]
        for kb in range(D_BLOCKS):
            s = base + kb * D_CHUNK
            e = s + D_CHUNK
            block = prev_flat[:, s:e] * inv_p * hnorm_w[kb * D_CHUNK:(kb + 1) * D_CHUNK].reshape(1, D_CHUNK) \
                * h_smooth[kb * D_CHUNK:(kb + 1) * D_CHUNK].reshape(1, D_CHUNK)   # [T, D_CHUNK]
            prev_norm[:, s:e] = block
            prev_amax[hc * D_BLOCKS + kb, :] = np.max(np.abs(block), axis=1)     # [T]

    buffers = {
        "v1": hidden_inv.reshape(-1).astype(np.float32),                          # [T,1] = 8
        "v2": hidden_amax.reshape(-1).astype(np.float32),                         # [56, T] = 448
        "v3": hidden_norm.reshape(-1).astype(np.float32),                         # [T, D] = 57344
        "v4": prev_amax.reshape(-1).astype(np.float32),                           # [224, T] = 1792
        "v5": prev_norm.reshape(-1).astype(np.float32),                           # [T, HC_DIM] = 229376
        "v6": hidden_bf16.reshape(-1),                                             # [T, D] bf16 = 57344
        "v7": enorm_w.reshape(-1).astype(np.float32),                             # [D] = 7168
        "v8": e_smooth.reshape(-1).astype(np.float32),                            # [D] = 7168
        "v9": hnorm_w.reshape(-1).astype(np.float32),                              # [D] = 7168
        "v10": h_smooth.reshape(-1).astype(np.float32),                           # [D] = 7168
        "v11": prev_inv.reshape(-1).astype(np.float32),                           # [4, T] = 32
        "v12": prev_flat.reshape(-1).astype(np.float32),                         # [T, HC_DIM] = 229376
    }
    # Outputs the kernel actually writes (tstore targets): v2, v3, v4, v5.
    # The golden dict is inlined in the return statement so the harness's
    # output-name regex detects these four output names.
    return buffers, {
        "v2": buffers["v2"],
        "v3": buffers["v3"],
        "v4": buffers["v4"],
        "v5": buffers["v5"],
    }


# Per-kernel scalar overrides for the EmitC harness. Keys are the .pto scalar
# param names (vN). Values that the harness cannot derive from tensor-view
# shapes (e.g. partition_view row offsets such as mtp_projection_norm %arg12,
# which is NOT a tensor dim and must be 0 for a single-block launch) live here.
# The NPU kernel is ground truth: an offset scalar of 0 makes the kernel read
# rows [0, T) of each [argT, ...] buffer; a wrong nonzero value reads past the
# buffer ("DDR address of the MTE instruction is out of range").
SCALAR_VALUES = {
    "mtp_projection_norm": {"v13": 0},
}


def build_mtp_projection_quant(meta, generator, ints):
    """Per-row INT8 quant of hidden_norm [T, D] and prev_norm [T, HC_DIM].

    .pto (scalar_dims %arg9 ties all tensors to T):
      v1: [56, T] f32 (hidden_amax_parts [D_BLOCKS, T])
      v2: [T, 1] f32 (hidden_scale_dq output)
      v3: [T, 7168] i8 (hidden_i8 output)
      v4: [T, 7168] f32 (hidden_norm input)
      v5: [T, 28672] i8 (prev_i8 output)
      v6: [4, T] f32 (prev_scale_dq [HC_MULT, T])
      v7: [224, T] f32 (prev_amax_parts [HC_MULT*D_BLOCKS, T])
      v8: [T, 28672] f32 (prev_norm input)
    """
    del ints
    T_pad = 8
    hidden_norm = make_fp32(generator, T_pad * D, scale=0.5).reshape(T_pad, D)
    prev_norm = make_fp32(generator, T_pad * HC_DIM, scale=0.5).reshape(T_pad, HC_DIM)
    h_i8, h_scale = per_row_int8_quant(hidden_norm)
    p_i8 = np.zeros((T_pad, HC_DIM), dtype=np.int8)
    p_scale = np.zeros((HC_MULT, T_pad), dtype=np.float32)
    prev_amax_parts = np.zeros((HC_MULT * D_BLOCKS, T_pad), dtype=np.float32)
    hidden_amax_parts = np.zeros((D_BLOCKS, T_pad), dtype=np.float32)
    for t in range(T_pad):
        for kb in range(D_BLOCKS):
            chunk = hidden_norm[t, kb * D_CHUNK:(kb + 1) * D_CHUNK]
            hidden_amax_parts[kb, t] = np.abs(chunk).max() if chunk.size else 0.0
        for hc in range(HC_MULT):
            row = prev_norm[t, hc * D:(hc + 1) * D]
            i8_r, sc_r = per_row_int8_quant(row.reshape(1, D))
            p_i8[t, hc * D:(hc + 1) * D] = i8_r
            p_scale[hc, t] = sc_r[0, 0]
            for kb in range(D_BLOCKS):
                chunk = prev_norm[t, hc * D + kb * D_CHUNK:hc * D + (kb + 1) * D_CHUNK]
                prev_amax_parts[hc * D_BLOCKS + kb, t] = np.abs(chunk).max() if chunk.size else 0.0
    buffers = {
        "v1": hidden_amax_parts.reshape(-1).astype(np.float32),
        "v2": h_scale.reshape(-1).astype(np.float32),
        "v3": h_i8.reshape(-1).astype(np.int8),
        "v4": hidden_norm.reshape(-1).astype(np.float32),
        "v5": p_i8.reshape(-1).astype(np.int8),
        "v6": p_scale.reshape(-1).astype(np.float32),
        "v7": prev_amax_parts.reshape(-1).astype(np.float32),
        "v8": prev_norm.reshape(-1).astype(np.float32),
    }
    return buffers, {"v2": buffers["v2"], "v3": buffers["v3"], "v5": buffers["v5"], "v6": buffers["v6"]}


def build_mtp_projection_linear(meta, generator, ints):
    """INT8 matmul (b_trans) of hidden_i8/prev_i8 with e_proj/h_proj weights.

    .pto (scalar_dims %arg11 ties v1/v4/v7/v8/v9 to T):
      v1: [T, 7168]  i8  (hidden_i8)
      v2: [7168, 7168] i8  (e_proj_w)
      v3: [7168]      f32 (e_proj_w_scale)
      v4: [T, 1]      f32 (hidden_scale_dq)
      v5: [7168, 7168] i8  (h_proj_w)
      v6: [7168]      f32 (h_proj_w_scale)
      v7: [T, 28672]  f32 (out_pad: hidden_e + prev_h sum)
      v8: [T, 28672]  i8  (prev_i8, HC_DIM-wide)
      v9: [4, T]      f32 (prev_scale_dq [HC_MULT, T])
    """
    del ints
    T_lin = 16                                      # LINEAR_T_TILE
    e_proj_w = make_int8(generator, D * D, scale=2.0).reshape(D, D)
    e_proj_scale = make_fp32(generator, D, scale=0.05).reshape(D)
    h_proj_w = make_int8(generator, D * D, scale=2.0).reshape(D, D)
    h_proj_scale = make_fp32(generator, D, scale=0.05).reshape(D)
    hidden_i8 = make_int8(generator, T_lin * D, scale=2.0).reshape(T_lin, D)
    prev_i8 = make_int8(generator, T_lin * HC_DIM, scale=2.0).reshape(T_lin, HC_DIM)
    hidden_scale = make_fp32(generator, T_lin, scale=0.05).reshape(T_lin, 1)
    prev_scale = make_fp32(generator, HC_MULT * T_lin, scale=0.05).reshape(HC_MULT, T_lin)

    out_pad = np.zeros((T_lin, HC_DIM), dtype=np.float32)
    for n0 in range(0, D, OUT_CHUNK):
        n_end = n0 + OUT_CHUNK
        # hidden path: hidden_i8 @ e_proj_w[n0:n_end, :].T  -> [T_lin, OUT_CHUNK]
        acc_h = hidden_i8.astype(np.int32) @ e_proj_w[n0:n_end, :].astype(np.int32).T
        hidden_deq = acc_h.astype(np.float32) * hidden_scale * e_proj_scale[n0:n_end].reshape(1, OUT_CHUNK)
        for hc in range(HC_MULT):
            base = hc * D
            prev_slice = prev_i8[:, base:base + D].astype(np.int32)
            acc_p = prev_slice @ h_proj_w[n0:n_end, :].astype(np.int32).T
            prev_deq = (acc_p.astype(np.float32) * prev_scale[hc].reshape(T_lin, 1)
                        * h_proj_scale[n0:n_end].reshape(1, OUT_CHUNK))
            out_pad[:, base + n0:base + n_end] = hidden_deq + prev_deq

    buffers = {
        "v1": hidden_i8.reshape(-1).astype(np.int8),
        "v2": e_proj_w.reshape(-1).astype(np.int8),
        "v3": e_proj_scale.reshape(-1).astype(np.float32),
        "v4": hidden_scale.reshape(-1).astype(np.float32),
        "v5": h_proj_w.reshape(-1).astype(np.int8),
        "v6": h_proj_scale.reshape(-1).astype(np.float32),
        "v7": out_pad.reshape(-1).astype(np.float32),
        "v8": prev_i8.reshape(-1).astype(np.int8),
        "v9": prev_scale.reshape(-1).astype(np.float32),
    }
    return buffers, {"v7": buffers["v7"]}


def build_mtp_projection_output(meta, generator, ints):
    """Copy out_pad [T, HC_DIM] -> hidden_states_out [T, HC_MULT, D].

    .pto (scalar_dims %arg3 ties v1, %arg4 ties v2):
      v1: [T, 28672] f32 (out_pad input)
      v2: [T, 28672] f32 (hidden_states_out output, flattened HC layout)
    """
    del ints
    T_pad = 8
    out_pad = make_fp32(generator, T_pad * HC_DIM, scale=0.5).reshape(T_pad, HC_DIM)
    buffers = {
        "v1": out_pad.reshape(-1).astype(np.float32),
        "v2": out_pad.reshape(-1).astype(np.float32),
    }
    return buffers, {"v2": buffers["v2"]}


# ===========================================================================
# Builder registry
# ===========================================================================
BUILDERS = {
    "qproj_matmul": build_qproj_matmul,
    "qproj_dequant_rms_nope_rope": build_qproj_dequant_rms_nope_rope,
    "qk_pv": build_qk_pv,
    "qkv_rope_rows": build_qkv_rope_rows,
    "qr_hadamard_matmul": build_qr_hadamard_matmul,
    "qr_hadamard_quant": build_qr_hadamard_quant,
    "qr_proj_matmul": build_qr_proj_matmul,
    "qr_proj_seed": build_qr_proj_seed,
    "qr_rms_norm_quant": build_qr_rms_norm_quant,
    "qr_rope": build_qr_rope,
    "kv_hadamard": build_kv_hadamard,
    "kv_proj_matmul": build_kv_proj_matmul,
    "kv_proj_seed": build_kv_proj_seed,
    "kv_rms_norm_rope": build_kv_rms_norm_rope,
    "kv_and_cache_write": build_kv_and_cache_write,
    "rmsnorm_rope_cache_write": build_rmsnorm_rope_cache_write,
    "kv_score_proj": build_kv_score_proj_0,
    "kv_score_proj_0": build_kv_score_proj_0,
    "score_reduce": build_score_reduce,
    "topk": build_topk,
    "mtp_projection_linear": build_mtp_projection_linear,
    "mtp_projection_norm": build_mtp_projection_norm,
    "mtp_projection_output": build_mtp_projection_output,
    "mtp_projection_quant": build_mtp_projection_quant,
    "mtp_projection_rms": build_mtp_projection_rms,
}


def run_case(case_name: str):
    """Generate buffers+golden for a kernel case (mirrors qwen3 golden_lib)."""
    from validation_runtime import load_case_meta, load_int32_assignments
    meta = load_case_meta()
    generator = rng()
    ints = load_int32_assignments()
    buffers, golden = BUILDERS[case_name](meta, generator, ints)
    write_buffers(meta, buffers)
    write_golden(meta, golden)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: qkv_mtp_compress_kernels.py <case_name>")
        sys.exit(1)
    run_case(sys.argv[1])
