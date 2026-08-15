#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""DSV4 golden reference for attention_csa/hca/swa family kernels.

Covers the sparse-attention merge/RoPE/output-projection sub-kernels, the
ratio-4 compressor sub-kernels, the hc_pre split/comb/mix sub-kernels, and the
SWA cache/rope sub-kernels. Each ``build_<name>(meta, generator, ints)``
returns ``(buffers, golden)`` matching the .pto buffer order in
``kernel_signatures.json``.

All computations are self-contained numpy. bf16 is stored as uint16 (matching
validation_runtime._HOST_TYPE_TO_NP). Cube/matmul uses plain matmul; RoPE uses
the DeepSeek-V4 interleaved-pairs swap-gather form; INT8 quant is per-row
symmetric (127 / amax) with the same i32->fp16->i8 narrowing the kernel applies.
"""

import numpy as np

from validation_runtime import (
    bf16_to_float32,
    float32_to_bf16,
    rng,
    write_buffers,
    write_golden,
    load_case_meta,
    load_int32_assignments,
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
O_LORA = 1024        # o_lora_rank
O_GROUPS = 16        # o_groups
HEADS_PER_GROUP = H // O_GROUPS    # 8
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM   # 4096
EPS = 1e-6
D_INV = 1.0 / D
HEAD_DIM_INV = 1.0 / HEAD_DIM
SOFTMAX_SCALE = HEAD_DIM ** -0.5
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = np.float32(-3.4028234663852886e38)
NEG_INF = np.float32(-1.0e20)

# compressor (ratio-4 overlap) constants
COMPRESS_RATIO = 4
COFF = 2                                   # 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM                   # 1024
COMPRESS_STATE_BLOCK_SIZE = 4               # C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM            # 2048
BLOCK_SIZE = 128                            # paged-KV page size
BS_PAD = 16                                 # padded B*S up to one 16-row cube tile
RMS_PAD_ROWS = 16

# hc_pre constants
HC_MULT = 4
HC_DIM = HC_MULT * D                         # 28672
MIX_HC = (2 + HC_MULT) * HC_MULT            # 24
HC_EPS = 1e-6
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = 20
MIX_PAD = 32                                # mix_hc (24) padded to 32-wide
HC_PAD = 8                                  # hc (4) padded for 32B-aligned vector ops

# indexer constants (for score_mat)
IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
IDX_TOPK = 1024
IDX_KV_LEN = 16384 // 4                     # 4096

# sparse-attn constants
WIN = 128                                   # sliding_window
ATTN_K_TILE = 128
H_TILE = 16
QK_M_TILE = 32
SPARSE_BLOCKS = 1                           # SWA: single block
T_PAD = ((T + 15) // 16) * 16              # 16
PADDED_TOPK = WIN                           # SWA: no compressed tail

# proj tiling
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_D_CHUNK = 512
PROJ_B_ACT_N_TILE = 512
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TBLK = 8
QUANT_TOKEN_TILE = 8


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


def apply_inverse_rope(x_fp32, cos_half, sin_half):
    """Inverse (conjugate) interleaved RoPE: swaps the sin sign.

        y_even = x_even*cos + x_odd*sin
        y_odd  = -x_even*sin + x_odd*cos
    Matches golden_sparse_attn's inv_even/inv_odd.
    """
    x_pair = x_fp32.reshape(*x_fp32.shape[:-1], -1, 2)
    x_even = x_pair[..., 0]
    x_odd = x_pair[..., 1]
    shape = x_even.shape
    cos_v = np.broadcast_to(cos_half.reshape(*([1] * (len(shape) - 1)), HALF_ROPE), shape).astype(np.float32)
    sin_v = np.broadcast_to(sin_half.reshape(*([1] * (len(shape) - 1)), HALF_ROPE), shape).astype(np.float32)
    y_even = x_even * cos_v + x_odd * sin_v
    y_odd = -x_even * sin_v + x_odd * cos_v
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
    half = i32.astype(np.float16)
    i8 = half.astype(np.int8)
    scale_dq = (1.0 / scale_q).astype(np.float32)
    return i8.reshape(x_fp32.shape), scale_dq.reshape(*x_fp32.shape[:-1], 1)


def quant_w_per_output_channel(w_bf16):
    """[K, N] bf16 weight -> per-output-channel (per N) symmetric INT8 + scale."""
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


def _softmax(x, axis=-1):
    """Numerically stable softmax over the given axis."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sinkhorn(comb_logits, n_iter=HC_SINKHORN_ITER):
    """Sinkhorn normalization of [T, HC_MULT, HC_MULT] logits.

    Mirrors hc_pre: softmax over last dim + eps, then alternating column/row
    normalization for n_iter-1 steps (column-first).
    """
    comb = _softmax(comb_logits, axis=-1) + HC_EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(n_iter - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    return comb


# ===========================================================================
# rms_norm
# ===========================================================================
def build_rms_norm(meta, generator, ints):
    """RMSNorm of x [8, 7168] bf16 with gamma [7168] -> x_normed [8, 7168] bf16.

    .pto:
      v1: [8, 7168] bf16 (x)
      v2: [8, 7168] bf16 (x_normed output)
      v3: [7168] bf16 (attn_norm_w / gamma)
    """
    del ints
    T_PAD = 8
    x = make_bf16(generator, T_PAD * D, scale=0.5).reshape(T_PAD, D)
    gamma = make_bf16(generator, D, scale=1.0).reshape(D)
    x_fp32 = bf16_to_float32(x)
    g_fp32 = bf16_to_float32(gamma).reshape(1, D)
    sq = (x_fp32 * x_fp32).sum(axis=1, keepdims=True) * D_INV
    inv_rms = 1.0 / np.sqrt(sq + EPS)
    normed = x_fp32 * inv_rms * g_fp32
    out = float32_to_bf16(normed)
    buffers = {
        "v1": x.reshape(-1),
        "v2": out.reshape(-1),
        "v3": gamma.reshape(-1),
    }
    return buffers, {"v2": buffers["v2"]}


# ===========================================================================
# rope (inverse RoPE rotation fused into o_packed rope columns)
# ===========================================================================
def build_rope(meta, generator, ints):
    """Inverse-RoPE rotation of attn_rope_stage -> o_packed rope columns.

    .pto:
      v1: [2048, 4096] bf16 (o_packed: O_GROUPS*T rows, O_GROUP_IN cols)
      v2: [128, 128, 64] f32 (attn_rope_stage_3d [T, H, ROPE_DIM], flat 1048576)
      v3: [128, 64] f32 (rope_cos_il [T, ROPE_DIM])
      v4: [128, 64] f32 (rope_sin_signed [T, ROPE_DIM])

    The kernel computes the head-invariant swap index j^1 = j+1 - 2*(j%2), then
        r_rot[j] = r[j]*cos_il[j] + r[j^1]*sin_signed[j]
    rounded to bf16 and packed into o_packed's rope columns (NOPE cols are
    untouched -- written by merge_norm).

    The .pto v1 is statically [2048, 4096] => O_GROUPS*T_pref=2048 (T_pref=128),
    v2 is [128, 128, 64] => [T_pref, H, ROPE_DIM], v3/v4 are [128, 64].
    """
    del ints
    T_pref = 128                                # 2048 / O_GROUPS, matches .pto static
    rope_stage = make_fp32(generator, T_pref * H * ROPE_DIM, scale=0.5).reshape(T_pref, H, ROPE_DIM)
    cos_il = make_fp32(generator, T_pref * ROPE_DIM, scale=1.0, positive=True).reshape(T_pref, ROPE_DIM)
    sin_signed = make_fp32(generator, T_pref * ROPE_DIM, scale=1.0, positive=True).reshape(T_pref, ROPE_DIM)
    o_packed = make_bf16(generator, (O_GROUPS * T_pref) * O_GROUP_IN, scale=0.5).reshape(O_GROUPS * T_pref, O_GROUP_IN)

    # swap index j^1 = j + 1 - 2*(j%2) over ROPE_DIM
    j = np.arange(ROPE_DIM, dtype=np.float32)
    lane = j - np.floor(j * 0.5) * 2.0
    swap_idx = (j + 1.0 - lane * 2.0).astype(np.int32)

    out_packed = np.array(bf16_to_float32(o_packed), copy=True).reshape(O_GROUPS * T_pref, O_GROUP_IN)
    for rp_hg in range(H // 4):
        for rp_hl in range(4):
            rp_gh = rp_hg * 4 + rp_hl
            rp_g = rp_gh // HEADS_PER_GROUP
            rp_hh = rp_gh - rp_g * HEADS_PER_GROUP
            rp_col = rp_hh * HEAD_DIM + NOPE_DIM
            for r_r0 in range(0, HALF_ROPE, 32):
                c0 = 2 * r_r0
                tile = rope_stage[:, rp_gh, c0:c0 + 64].astype(np.float32)  # [T_pref, 64]
                r_cos = cos_il[:, c0:c0 + 64]
                r_sin = sin_signed[:, c0:c0 + 64]
                r_swapped = tile[:, swap_idx]
                r_rot = tile * r_cos + r_swapped * r_sin
                r_rot_bf16 = float32_to_bf16(r_rot)
                for rp_tt in range(T_pref):
                    rp_o0 = rp_g * T_pref + rp_tt
                    out_packed[rp_o0, rp_col + c0:rp_col + c0 + 64] = bf16_to_float32(r_rot_bf16[rp_tt])
    out_bf16 = float32_to_bf16(out_packed)
    buffers = {
        "v1": out_bf16.reshape(-1),
        "v2": rope_stage.reshape(-1).astype(np.float32),
        "v3": cos_il.reshape(-1).astype(np.float32),
        "v4": sin_signed.reshape(-1).astype(np.float32),
    }
    return buffers, {"v1": buffers["v1"]}


# ===========================================================================
# rope_cs (CSA/HCA head-invariant interleaved cos/signed-sin row builder)
# ===========================================================================
def build_rope_cs(meta, generator, ints):
    """Build rope_swap_idx, rope_cos_il, rope_sin_signed from freqs tables.

    .pto:
      v1: [16, 64] i32 (rope_swap_idx [H_TILE, ROPE_DIM])
      v2: [8, 64] f32 (rope_cos_il [T, ROPE_DIM])
      v3: [8, 64] f32 (rope_sin_signed [T, ROPE_DIM])
      v4: [8, 64] bf16 (rope_cos_t input [T, ROPE_DIM])
      v5: [8, 64] bf16 (rope_sin_t input [T, ROPE_DIM])

    The kernel builds:
      dup_idx[j] = j >> 1 (trunc)
      sign[j]    = -(2*(j%2) - 1)   # +1,-1,...  (conjugate)
      cos_il[j]  = cos_half[dup_idx[j]]
      sin_signed[j] = sin_half[dup_idx[j]] * sign[j]
      swap_idx[j] = j + 1 - 2*(j%2) = j^1
    """
    del ints
    rope_w = ROPE_DIM
    cos_t = make_bf16(generator, T * rope_w, scale=1.0, positive=True).reshape(T, rope_w)
    sin_t = make_bf16(generator, T * rope_w, scale=1.0, positive=True).reshape(T, rope_w)

    j = np.arange(rope_w, dtype=np.float32)
    dup_f = np.floor(j * 0.5)
    dup_idx = dup_f.astype(np.int32)
    lane = j - dup_f * 2.0
    sign = -(lane * 2.0 - 1.0)
    swap_idx = (j + 1.0 - lane * 2.0).astype(np.int32)

    cos_half = bf16_to_float32(cos_t[:, :HALF_ROPE]).astype(np.float32)
    sin_half = bf16_to_float32(sin_t[:, :HALF_ROPE]).astype(np.float32)
    cos_il = cos_half[:, dup_idx]
    sin_signed = sin_half[:, dup_idx] * sign.reshape(1, rope_w)

    swap_idx_tile = np.broadcast_to(swap_idx.reshape(1, rope_w), (H_TILE, rope_w)).astype(np.int32)
    buffers = {
        "v1": swap_idx_tile.reshape(-1).astype(np.int32),
        "v2": cos_il.reshape(-1).astype(np.float32),
        "v3": sin_signed.reshape(-1).astype(np.float32),
        "v4": cos_t.reshape(-1),
        "v5": sin_t.reshape(-1),
    }
    return buffers, {"v1": buffers["v1"], "v2": buffers["v2"], "v3": buffers["v3"]}


# ===========================================================================
# q_rope_prepare (Q head-invariant rope tables: cos_il/sin_signed/swap_idx)
# ===========================================================================
def build_q_rope_prepare(meta, generator, ints):
    """Build q_rope_cos_il / q_rope_sin_signed / q_rope_swap_idx from rope tables.

    .pto (scalar_dims %arg5 = T ties v1/v2/v3/v4 to [T, 64]):
      v1: [T, 64] bf16 (rope_cos input)
      v2: [T, 64] bf16 (rope_sin input)
      v3: [T, 64] f32 (q_rope_cos_il output)
      v4: [T, 64] f32 (q_rope_sin_signed output)
      v5: [T, 64] i32 (q_rope_swap_idx output)

    All buffer shapes are dynamic (T = %arg5). The runtime main.cpp sizes them
    via elem_counts_override (Mode B) or skips 0-elem ptrs (Mode A). We size to
    the standard decode T=8 when meta.elem_counts is 0/absent (the golden file
    is still written; callers that bind T=8 will match).
    """
    del ints
    rope_w = ROPE_DIM
    # Resolve T from meta if any vN elem_count is set; else default to 8.
    # Note: the preliminary main.cpp sets placeholder elem_count=1 for
    # dynamic-shaped ptrs, so treat count <= 1 as "not set".
    ec_v3 = meta.elem_counts.get("v3", 0)
    T_qrp = (ec_v3 // rope_w) if ec_v3 > 1 else 8
    cos_t = make_bf16(generator, T_qrp * rope_w, scale=1.0, positive=True).reshape(T_qrp, rope_w)
    sin_t = make_bf16(generator, T_qrp * rope_w, scale=1.0, positive=True).reshape(T_qrp, rope_w)

    j = np.arange(rope_w, dtype=np.float32)
    dup_f = np.floor(j * 0.5)
    dup_idx = dup_f.astype(np.int32)
    lane = j - dup_f * 2.0
    sign = (lane * 2.0 - 1.0)
    swap_idx = (j + 1.0 - lane * 2.0).astype(np.int32)

    cos_half = bf16_to_float32(cos_t[:, :HALF_ROPE]).astype(np.float32)
    sin_half = bf16_to_float32(sin_t[:, :HALF_ROPE]).astype(np.float32)
    cos_il = cos_half[:, dup_idx]
    sin_signed = sin_half[:, dup_idx] * sign.reshape(1, rope_w)

    buffers = {
        "v1": cos_t.reshape(-1),
        "v2": sin_t.reshape(-1),
        "v3": cos_il.reshape(-1).astype(np.float32),
        "v4": sin_signed.reshape(-1).astype(np.float32),
        "v5": np.broadcast_to(swap_idx.reshape(1, rope_w), (T_qrp, rope_w)).reshape(-1).astype(np.int32),
    }
    return buffers, {"v3": buffers["v3"], "v4": buffers["v4"], "v5": buffers["v5"]}


# ===========================================================================
# kv_touch (no-op self-copy to mark ori_kv add_inout for WAR)
# ===========================================================================
def build_kv_touch(meta, generator, ints):
    """No-op self-touch of the ori_kv cache [16384, 512] bf16.

    .pto:
      v1: [16384, 512] bf16 (ori_kv_flat)

    The kernel loads tile [0:8, 0:512] and stores it back unchanged (a WAR
    marker so the enclosing layer's in-place KV-cache writeback gets its WAR
    edge against the gather read). The golden is the input unchanged.
    """
    del ints
    cache = make_bf16(generator, meta.elem_counts.get("v1", 0), scale=0.05)
    buffers = {"v1": cache}
    return buffers, {"v1": buffers["v1"].copy()}


# ===========================================================================
# merge_norm (online-softmax merge + sink-norm + inverse-RoPE -> o_packed)
# ===========================================================================
def build_merge_norm(meta, generator, ints):
    """Merge sparse_blk stats, sink-norm, inverse RoPE, pack into o_packed.

    .pto:
      v1: [1024, 1] f32 (sparse_blk_mi, dn layout)
      v2: [1024, 1] f32 (sparse_blk_li, dn layout)
      v3: [1024, 512] f32 (sparse_blk_oi)
      v4: [128] f32 (attn_sink [H])
      v5: [16, 64] i32 (rope_swap_idx [H_TILE, ROPE_DIM])
      v6: [8, 64] f32 (rope_cos_il [T, ROPE_DIM])
      v7: [8, 64] f32 (rope_sin_signed [T, ROPE_DIM])
      v8: [1024, 512] bf16 (o_packed_heads output)

    For SWA (SPARSE_BLOCKS==1) the merge loop is a no-op: the single block's
    (mi, li, oi) is sink-normalized directly. Inverse RoPE rotates the rope
    half and packs NOPE+rope into o_packed_heads[g*T + t, hh*HEAD_DIM:...].
    """
    del ints
    sparse_blocks = SPARSE_BLOCKS
    blk_rows = T * (H // H_TILE) * sparse_blocks * H_TILE   # 8 * 8 * 1 * 16 = 1024
    mi = make_fp32(generator, blk_rows, scale=1.0).reshape(blk_rows, 1)
    li = make_fp32(generator, blk_rows, scale=1.0, positive=True).reshape(blk_rows, 1)
    oi = make_fp32(generator, blk_rows * HEAD_DIM, scale=0.5).reshape(blk_rows, HEAD_DIM)
    attn_sink = make_fp32(generator, H, scale=1.0).reshape(H)
    swap_idx = make_fp32(generator, H_TILE * ROPE_DIM, scale=1.0).reshape(H_TILE, ROPE_DIM).astype(np.int32)
    cos_il = make_fp32(generator, T * ROPE_DIM, scale=1.0, positive=True).reshape(T, ROPE_DIM)
    sin_signed = make_fp32(generator, T * ROPE_DIM, scale=1.0, positive=True).reshape(T, ROPE_DIM)

    o_packed = np.zeros((O_GROUPS * T, HEAD_DIM * HEADS_PER_GROUP), dtype=np.uint16)

    for m_idx in range(T * (H // H_TILE)):
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * sparse_blocks * H_TILE
        cur_mi = mi[m_blk_base:m_blk_base + H_TILE]
        cur_li = li[m_blk_base:m_blk_base + H_TILE]
        cur_oi = oi[m_blk_base:m_blk_base + H_TILE]
        # SWA: single block, no merge loop
        n_sink = attn_sink[m_h0:m_h0 + H_TILE].reshape(H_TILE, 1)
        n_denom = cur_li + np.exp(n_sink - cur_mi)
        n_full = cur_oi / n_denom
        # inverse RoPE on rope half
        m_rope = n_full[:, NOPE_DIM:HEAD_DIM].astype(np.float32)
        r_cos = cos_il[m_t]
        r_sin = sin_signed[m_t]
        r_swapped = m_rope[:, swap_idx[0]]
        m_rot = m_rope * r_cos + r_swapped * r_sin
        n_bf16 = float32_to_bf16(n_full[:, :NOPE_DIM])
        n_rope_bf16 = float32_to_bf16(m_rot)
        for n_hi in range(H_TILE):
            n_gh = m_h0 + n_hi
            n_g = n_gh // HEADS_PER_GROUP
            n_hh = n_gh - n_g * HEADS_PER_GROUP
            n_pack_row = n_g * T + m_t
            n_col = n_hh * HEAD_DIM
            o_packed[n_pack_row, n_col:n_col + NOPE_DIM] = n_bf16[n_hi]
            o_packed[n_pack_row, n_col + NOPE_DIM:n_col + HEAD_DIM] = n_rope_bf16[n_hi]

    buffers = {
        "v1": mi.reshape(-1).astype(np.float32),
        "v2": li.reshape(-1).astype(np.float32),
        "v3": oi.reshape(-1).astype(np.float32),
        "v4": attn_sink.reshape(-1).astype(np.float32),
        "v5": swap_idx.reshape(-1).astype(np.int32),
        "v6": cos_il.reshape(-1).astype(np.float32),
        "v7": sin_signed.reshape(-1).astype(np.float32),
        "v8": o_packed.reshape(-1),
    }
    return buffers, {"v8": buffers["v8"]}


# ===========================================================================
# proj_a_mm (grouped output projection A: o_packed @ wo_a^T -> o_r_pad)
# ===========================================================================
def build_proj_a_mm(meta, generator, ints):
    """Per-group fp32 matmul: o_packed[g*T:g*T+T, :O_GROUP_IN] @ wo_a[g]^T -> o_r_pad.

    .pto:
      v1: [128, 4096] bf16 (o_packed [O_GROUPS*T, O_GROUP_IN])
      v2: [16, 1024, 4096] bf16 (wo_a [O_GROUPS, O_LORA, O_GROUP_IN])
      v3: [16, 16384] f32 (o_r_pad [T_PAD, O_GROUPS*O_LORA])

    wo_a stored [G, O_LORA, O_GROUP_IN]; matmul is o_packed @ wo_a^T (b_trans)
    so the K-contract is over O_GROUP_IN. o_r_pad layout: [T_PAD, G*O_LORA].
    """
    del ints
    T_lin = T                                    # 8 real tokens (o_packed has O_GROUPS*T rows)
    T_pad_out = T_PAD                            # 16 (o_r_pad rows, padded up to 16-row cube tile)
    o_packed = make_bf16(generator, (O_GROUPS * T_lin) * O_GROUP_IN, scale=0.5).reshape(O_GROUPS * T_lin, O_GROUP_IN)
    wo_a = make_bf16(generator, O_GROUPS * O_LORA * O_GROUP_IN, scale=0.05).reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    o_r_pad = np.zeros((T_pad_out, O_GROUPS * O_LORA), dtype=np.float32)
    for g in range(O_GROUPS):
        row_base = g * T_lin
        xa = bf16_to_float32(o_packed[row_base:row_base + T_lin, :O_GROUP_IN])
        wa = bf16_to_float32(wo_a[g])                      # [O_LORA, O_GROUP_IN]
        acc = xa @ wa.T                                    # [T_lin, O_LORA]
        o_r_pad[:T_lin, g * O_LORA:(g + 1) * O_LORA] = acc
    buffers = {
        "v1": o_packed.reshape(-1),
        "v2": wo_a.reshape(-1),
        "v3": o_r_pad.reshape(-1).astype(np.float32),
    }
    return buffers, {"v3": buffers["v3"]}


# ===========================================================================
# proj_b_mm (grouped output projection B: o_r_i8 @ wo_b^T -> INT32 partials)
# ===========================================================================
def build_proj_b_mm(meta, generator, ints):
    """Per-group INT8 matmul: o_r_i8[g] @ wo_b[g]^T -> partials[:, g*D + n].

    .pto:
      v1: [16, 114688] i32 (partials [T_PAD, O_GROUPS*D])
      v2: [16, 16384] i8 (o_r_i8_pad [T_PAD, O_GROUPS*O_LORA])
      v3: [7168, 16384] i8 (wo_b [D, O_GROUPS*O_LORA])

    wo_b stored [D, G*O_LORA]; per-group slice wo_b[:, g*O_LORA:(g+1)*O_LORA].
    """
    del ints
    T_lin = T_PAD
    o_r_i8 = make_int8(generator, T_lin * O_GROUPS * O_LORA, scale=2.0).reshape(T_lin, O_GROUPS * O_LORA)
    wo_b = make_int8(generator, D * O_GROUPS * O_LORA, scale=2.0).reshape(D, O_GROUPS * O_LORA)
    partials = np.zeros((T_lin, O_GROUPS * D), dtype=np.int32)
    for g in range(O_GROUPS):
        col_g = g * O_LORA
        b_act = o_r_i8[:, col_g:col_g + O_LORA].astype(np.int32)
        b_weight = wo_b[:, col_g:col_g + O_LORA].astype(np.int32)
        acc_b = b_act @ b_weight.T                           # [T_lin, D]
        partials[:, g * D:(g + 1) * D] = acc_b
    buffers = {
        "v1": partials.reshape(-1).astype(np.int32),
        "v2": o_r_i8.reshape(-1).astype(np.int8),
        "v3": wo_b.reshape(-1).astype(np.int8),
    }
    return buffers, {"v1": buffers["v1"]}


# ===========================================================================
# proj_b_act (vector dequant+sum of INT32 partials -> attn_out bf16)
# ===========================================================================
def build_proj_b_act(meta, generator, ints):
    """Sum O_GROUPS INT32 partials (each x its group act scale) + weight scale -> bf16.

    .pto:
      v1: [7168] f32 (wo_b_scale [D])
      v2: [16, 114688] i32 (partials [T_PAD, O_GROUPS*D])
      v3: [16, 8] f32 (act_scale_dq [O_GROUPS, T])
      v4: [8, 7168] bf16 (attn_out [T, D])

    out = (sum_g partials[:, g*D + n] * act_scale_dq[g, t]) * wo_b_scale[n] -> bf16
    """
    del ints
    T_lin = T
    wo_b_scale = make_fp32(generator, D, scale=0.05).reshape(D)
    partials = make_int8(generator, T_PAD * O_GROUPS * D, scale=2.0).astype(np.int32).reshape(T_PAD, O_GROUPS * D)
    act_scale_dq = make_fp32(generator, O_GROUPS * T_lin, scale=0.05).reshape(O_GROUPS, T_lin)
    attn_out = np.zeros((T_lin, D), dtype=np.uint16)
    for t in range(T_lin):
        acc = np.zeros(D, dtype=np.float32)
        for g in range(O_GROUPS):
            p = partials[t, g * D:(g + 1) * D].astype(np.float32)
            acc += p * act_scale_dq[g, t]
        out = acc * wo_b_scale
        attn_out[t] = float32_to_bf16(out)
    buffers = {
        "v1": wo_b_scale.reshape(-1).astype(np.float32),
        "v2": partials.reshape(-1).astype(np.int32),
        "v3": act_scale_dq.reshape(-1).astype(np.float32),
        "v4": attn_out.reshape(-1),
    }
    return buffers, {"v4": buffers["v4"]}


# ===========================================================================
# kv_score_proj (ratio-4 compressor kv+score projection)
# ===========================================================================
def build_kv_score_proj(meta, generator, ints):
    """x [8, 7168] @ wkv^T + wgate^T -> kv_proj/score_proj [16, 1024] f32.

    .pto:
      v1: [8, 7168] bf16 (x_flat [B*S, D])
      v2: [1024, 7168] bf16 (wkv [OUT_DIM, D])
      v3: [1024, 7168] bf16 (wgate [OUT_DIM, D])
      v4: [16, 1024] f32 (cmp4_kv_proj_pad [BS_PAD, OUT_DIM])
      v5: [16, 1024] f32 (cmp4_score_proj_pad [BS_PAD, OUT_DIM])

    Weights stored [OUT_DIM, D] and consumed via b_trans=True. BS_PAD=16 rows
    (B*S=8 real rows, 8 pad rows zero-filled past the real count).
    """
    del ints
    OUT_DIM_l = OUT_DIM
    x = make_bf16(generator, (B * S) * D, scale=0.05).reshape(B * S, D)
    wkv = make_bf16(generator, OUT_DIM_l * D, scale=0.05).reshape(OUT_DIM_l, D)
    wgate = make_bf16(generator, OUT_DIM_l * D, scale=0.05).reshape(OUT_DIM_l, D)
    x_pad = np.zeros((BS_PAD, D), dtype=np.uint16)
    x_pad[:B * S] = x
    kv = bf16_to_float32(x_pad) @ bf16_to_float32(wkv).T       # [BS_PAD, OUT_DIM]
    score = bf16_to_float32(x_pad) @ bf16_to_float32(wgate).T
    buffers = {
        "v1": x.reshape(-1),
        "v2": wkv.reshape(-1),
        "v3": wgate.reshape(-1),
        "v4": kv.reshape(-1).astype(np.float32),
        "v5": score.reshape(-1).astype(np.float32),
    }
    return buffers, {"v4": buffers["v4"], "v5": buffers["v5"]}


def build_kv_score_proj_0(meta, generator, ints):
    """kv_score_proj variant 0 (alias of build_kv_score_proj).

    No separate .pto/signature exists for kv_score_proj_0; it is the same
    ratio-4 compressor kv+score projection as kv_score_proj.
    """
    return build_kv_score_proj(meta, generator, ints)


# ===========================================================================
# scatter_softmax_pool (ratio-4 compressor scatter + online-softmax pool)
# ===========================================================================
def build_scatter_softmax_pool(meta, generator, ints):
    """Scatter proj rows into compress_state, online-softmax pool -> pooled_kv.

    .pto:
      v1: [260, 2048] f32 (compress_state_flat [BLK_NUM*BLK_SIZE, STATE_DIM])
      v2: [16, 512] f32 (pooled_kv [RMS_PAD_ROWS, HEAD_DIM])
      v3: [4, 2] i32 (position_ids [B, S])
      v4: [4, 2] i64 (state_slot_mapping [B, S])
      v5: [16, 1024] f32 (cmp4_kv_proj_pad [BS_PAD, OUT_DIM])
      v6: [16, 1024] f32 (cmp4_score_proj_pad [BS_PAD, OUT_DIM])
      v7: [4, 1024] f32 (ape [COMPRESS_RATIO, OUT_DIM])
      v8: [4, 4096] i32 (compress_state_block_table [B, MAX_BLOCKS])

    Pools a per-batch window of 2*COMPRESS_RATIO state rows (front+back slots)
    with online softmax, writes pooled_kv[b] = sum(softmax(score)*kv).
    """
    del ints
    state_rows = 65 * COMPRESS_STATE_BLOCK_SIZE            # 260
    compress_state = make_fp32(generator, state_rows * COMPRESS_STATE_DIM, scale=0.5).reshape(state_rows, COMPRESS_STATE_DIM)
    position_ids = generator.integers(8, 4096, size=B * S).reshape(B, S).astype(np.int32)
    # state_slot_mapping: deterministic mapping; first token of each batch -> a row
    state_slot = np.full((B, S), -1, dtype=np.int64)
    for b in range(B):
        state_slot[b, 0] = b * 4
        state_slot[b, 1] = b * 4 + 1
    kv_proj = make_fp32(generator, BS_PAD * OUT_DIM, scale=0.5).reshape(BS_PAD, OUT_DIM)
    score_proj = make_fp32(generator, BS_PAD * OUT_DIM, scale=0.5).reshape(BS_PAD, OUT_DIM)
    ape = make_fp32(generator, COMPRESS_RATIO * OUT_DIM, scale=0.05).reshape(COMPRESS_RATIO, OUT_DIM)
    block_table = np.zeros((B, 4096), dtype=np.int32)
    for b in range(B):
        for blk in range(65):
            block_table[b, blk] = blk

    # scatter: per token, add ape to score, write kv+score into state row
    new_state = compress_state.copy()
    for b in range(B):
        for s in range(S):
            proj_row = b * S + s
            pos = int(position_ids[b, s])
            token_ape_row = pos % COMPRESS_RATIO
            score_row = score_proj[proj_row] + ape[token_ape_row]
            srow = int(state_slot[b, s])
            if srow >= 0:
                new_state[srow, :OUT_DIM] = kv_proj[proj_row]
                new_state[srow, OUT_DIM:] = score_row

    pooled = np.zeros((RMS_PAD_ROWS, HEAD_DIM), dtype=np.float32)
    for b in range(B):
        first_pos = int(position_ids[b, 0])
        pre_tokens = min(S, COMPRESS_RATIO - (first_pos % COMPRESS_RATIO))
        boundary_s = COMPRESS_RATIO - 1 - (first_pos % COMPRESS_RATIO)
        should_compress = 0 <= boundary_s < S
        if not should_compress:
            continue
        boundary_end = first_pos + pre_tokens - 1
        cur_window_start = boundary_end - COMPRESS_RATIO + 1
        prev_window_start = cur_window_start - COMPRESS_RATIO
        kv_rows = []
        score_rows = []
        for s in range(COMPRESS_RATIO):
            abs_pos = prev_window_start + s
            if abs_pos < 0:
                kv_rows.append(np.zeros(HEAD_DIM, dtype=np.float32))
                score_rows.append(np.full(HEAD_DIM, FP32_NEG_INF, dtype=np.float32))
                continue
            row = new_state[abs_pos]
            kv_rows.append(row[:HEAD_DIM])
            score_rows.append(row[OUT_DIM:OUT_DIM + HEAD_DIM])
        for s in range(COMPRESS_RATIO):
            abs_pos = cur_window_start + s
            row = new_state[abs_pos]
            kv_rows.append(row[HEAD_DIM:OUT_DIM])
            score_rows.append(row[OUT_DIM + HEAD_DIM:COMPRESS_STATE_DIM])
        kvs = np.stack(kv_rows, axis=0)                     # [8, HEAD_DIM]
        scs = np.stack(score_rows, axis=0)                  # [8, HEAD_DIM]
        scs_max = scs.max(axis=0, keepdims=True)
        exp = np.exp(scs - scs_max)
        weights = exp / exp.sum(axis=0, keepdims=True)
        pooled[b] = (kvs * weights).sum(axis=0)

    buffers = {
        "v1": new_state.reshape(-1).astype(np.float32),
        "v2": pooled.reshape(-1).astype(np.float32),
        "v3": position_ids.reshape(-1).astype(np.int32),
        "v4": state_slot.reshape(-1).astype(np.int64),
        "v5": kv_proj.reshape(-1).astype(np.float32),
        "v6": score_proj.reshape(-1).astype(np.float32),
        "v7": ape.reshape(-1).astype(np.float32),
        "v8": block_table.reshape(-1).astype(np.int32),
    }
    return buffers, {"v1": buffers["v1"], "v2": buffers["v2"]}


# ===========================================================================
# score_mat (indexer: kv_cache_i8 @ qr_hadamard_i8^T -> score_acc INT32)
# ===========================================================================
def build_score_mat(meta, generator, ints):
    """INT8 matmul: kv_cache_i8[BLOCK_SIZE, IDX_HEAD_DIM] @ qr_hadamard_i8^T -> i32.

    .pto:
      v1: [4] i32 (kv_seq_lens [B])
      v2: [512, 128] i8 (qr_hadamard_i8 [IDX_N_HEADS, IDX_HEAD_DIM])
      v3: [32768, 64] i32 (score_acc_gm [T*IDX_KV_LEN, IDX_N_HEADS])
      v4: [256] i32 (idx_block_table_flat [B*IDX_CACHE_MAX_BLOCKS])
      v5: [8192, 128] i8 (kv_cache_i8_flat [IDX_CACHE_BLOCK_NUM*BLOCK_SIZE, IDX_HEAD_DIM])
      v6/v7: i32 spmd scalars

    Per token: for each cache block, kv_i8 @ qr^T (b_trans) -> [BLOCK_SIZE, H].
    """
    del ints
    kv_seq_lens = generator.integers(128, 4096, size=B).astype(np.int32)
    # v2 is [512, 128] = [T * IDX_N_HEADS, IDX_HEAD_DIM]
    qr = make_int8(generator, T * IDX_N_HEADS * IDX_HEAD_DIM, scale=2.0).reshape(T * IDX_N_HEADS, IDX_HEAD_DIM)
    block_table = generator.integers(0, 64, size=B * 64).astype(np.int32)
    kv_cache = make_int8(generator, 8192 * IDX_HEAD_DIM, scale=2.0).reshape(8192, IDX_HEAD_DIM)

    score_acc = np.zeros((T * IDX_KV_LEN, IDX_N_HEADS), dtype=np.int32)
    for tg in range(T):
        b = tg // S
        s = tg - b * S
        clen = int(kv_seq_lens[b]) // COMPRESS_RATIO
        cblk = (clen + BLOCK_SIZE - 1) // BLOCK_SIZE
        qb = b * S * IDX_N_HEADS
        qr_full = qr[qb + s * IDX_N_HEADS:qb + (s + 1) * IDX_N_HEADS, :IDX_HEAD_DIM]
        for cb in range(cblk):
            idx_blk_id = int(block_table[b * 64 + cb])
            kv0 = idx_blk_id * BLOCK_SIZE
            kv_i8_mat = kv_cache[kv0:kv0 + BLOCK_SIZE, :]
            acc = kv_i8_mat.astype(np.int32) @ qr_full.astype(np.int32).T
            base = tg * IDX_KV_LEN + cb * BLOCK_SIZE
            score_acc[base:base + BLOCK_SIZE, :] = acc

    buffers = {
        "v1": kv_seq_lens.reshape(-1).astype(np.int32),
        "v2": qr.reshape(-1).astype(np.int8),
        "v3": score_acc.reshape(-1).astype(np.int32),
        "v4": block_table.reshape(-1).astype(np.int32),
        "v5": kv_cache.reshape(-1).astype(np.int8),
        "v6": np.array([0], dtype=np.int32),
        "v7": np.array([1], dtype=np.int32),
    }
    return buffers, {"v3": buffers["v3"]}


# ===========================================================================
# split_pre_post (hc_pre: pre gate -> pre_val_store, post gate -> post)
# ===========================================================================
def build_split_pre_post(meta, generator, ints):
    """inv_rms-scaled pre/post gates from mixes_raw.

    .pto (scalar_dims %arg7 ties v1[1], v3[32], v4[8] to T):
      v1: [T, 1] f32 (inv_rms)
      v2: [24] f32 (hc_base [MIX_HC])
      v3: [T, 32] f32 (mixes_raw [T, MIX_PAD])
      v4: [T, 8] f32 (pre_val_store output [T, HC_PAD])
      v5: [8, 4] f32 (post output [T, HC_MULT]; elem_count 32 = T*HC_MULT)

    pre = sigmoid(mixes[:, :HC_PAD] * inv_rms * scale0 + base[:HC_PAD]) + eps
    post = 2 * sigmoid(mixes[:, HC_MULT:HC_MULT+HC_PAD] * inv_rms * scale1 + base[...])

    T is dynamic; we recover it from v5's elem_count (T*HC_MULT=32 => T=8) or
    v3's (T*MIX_PAD => T=1 if the runtime only captured the static dim). Fall
    back to decode T=8.
    """
    del ints
    # Recover T from v5 elem_count (T*HC_MULT) when the runtime captured it.
    ec_v5 = meta.elem_counts.get("v5", 0)
    T_spp = (ec_v5 // HC_MULT) if ec_v5 > 0 and ec_v5 % HC_MULT == 0 else 8
    inv_rms = make_fp32(generator, T_spp, scale=1.0, positive=True).reshape(T_spp, 1)
    hc_base = make_fp32(generator, MIX_HC, scale=1.0).reshape(MIX_HC)
    mixes = make_fp32(generator, T_spp * MIX_PAD, scale=0.5).reshape(T_spp, MIX_PAD)
    scale0 = np.float32(0.076099)
    scale1 = np.float32(0.032597)

    pre_base = hc_base[:HC_PAD].reshape(1, HC_PAD)
    pre_scaled = mixes[:, :HC_PAD] * inv_rms * scale0
    pre_logits = pre_scaled + pre_base
    pre_sig = 1.0 / (1.0 + np.exp(-pre_logits))
    pre_val = pre_sig + HC_EPS

    post_base = hc_base[HC_MULT:HC_MULT + HC_PAD].reshape(1, HC_PAD)
    post_scaled = mixes[:, HC_MULT:HC_MULT + HC_PAD] * inv_rms * scale1
    post_logits = post_scaled + post_base
    post_sig = 1.0 / (1.0 + np.exp(-post_logits))
    post_pad = post_sig * 2.0
    post_out = post_pad[:, :HC_MULT].reshape(T_spp, HC_MULT)
    buffers = {
        "v1": inv_rms.reshape(-1).astype(np.float32),
        "v2": hc_base.reshape(-1).astype(np.float32),
        "v3": mixes.reshape(-1).astype(np.float32),
        "v4": pre_val.reshape(-1).astype(np.float32),
        "v5": post_out.reshape(-1).astype(np.float32),
    }
    return buffers, {"v4": buffers["v4"], "v5": buffers["v5"]}


# ===========================================================================
# comb_sinkhorn (hc_pre: comb gate + softmax + 20-iter Sinkhorn -> comb)
# ===========================================================================
def build_comb_sinkhorn(meta, generator, ints):
    """comb gate (mixes_raw * inv_rms * scale2 + base) -> softmax -> sinkhorn.

    .pto:
      v1: [T, 1] f32 (inv_rms, scalar_dims %arg5 ties v1 to [1] per token -> [T,1])
      v2: [T, 32] f32 (mixes_raw, scalar_dims %arg5 ties v2 to [32] -> [T, MIX_PAD])
      v3: [1, 24] f32 (hc_base_2d [1, MIX_HC])
      v4: [8, 16] f32 (comb output [T, HC_MULT*HC_MULT])

    comb_off = HC_MULT*2 = 8; 4 comb groups at cols 8/12/16/20 of mixes_raw.
    """
    del ints
    T_PAD = 8
    inv_rms = make_fp32(generator, T_PAD, scale=1.0, positive=True).reshape(T_PAD, 1)
    mixes = make_fp32(generator, T_PAD * MIX_PAD, scale=0.5).reshape(T_PAD, MIX_PAD)
    hc_base = make_fp32(generator, MIX_HC, scale=1.0).reshape(1, MIX_HC)
    scale2 = np.float32(0.226994)

    comb_off = HC_MULT * 2                            # 8
    rows = []
    for k in range(HC_MULT):
        mix_g = mixes[:, comb_off + k * HC_MULT:comb_off + (k + 1) * HC_MULT]
        cb = hc_base[:, comb_off + k * HC_MULT:comb_off + (k + 1) * HC_MULT]
        row = mix_g * inv_rms * scale2 + cb
        rows.append(row)
    logits = np.stack(rows, axis=1)                   # [T, HC_MULT, HC_MULT]
    comb = _sinkhorn(logits, HC_SINKHORN_ITER)
    comb_out = comb.reshape(T_PAD, HC_MULT * HC_MULT)
    buffers = {
        "v1": inv_rms.reshape(-1).astype(np.float32),
        "v2": mixes.reshape(-1).astype(np.float32),
        "v3": hc_base.reshape(-1).astype(np.float32),
        "v4": comb_out.reshape(-1).astype(np.float32),
    }
    return buffers, {"v4": buffers["v4"]}


# ===========================================================================
# mix_x (hc_pre: x_mixed = sum_h pre[:,h] * x[:,h,:])
# ===========================================================================
def build_mix_x(meta, generator, ints):
    """x_mixed = sum_h pre[:,h] * x[:,h,:] -> bf16 [T, D].

    .pto (scalar_dims %arg3 ties v1 to [8] -> [T, HC_PAD]; %arg4 ties v3 to [28672]):
      v1: [T, 8] f32 (pre_val_store [T, HC_PAD])
      v2: [8, 7168] bf16 (x_mixed output [T, D])
      v3: [T, 28672] f32 (x_flat [T, HC_DIM])

    pre has HC_MULT=4 valid cols (HC_PAD=8); x is [T, HC_MULT*D] reshaped.
    """
    del ints
    T_PAD = 8
    pre = make_fp32(generator, T_PAD * HC_PAD, scale=1.0, positive=True).reshape(T_PAD, HC_PAD)
    x_flat = make_fp32(generator, T_PAD * HC_DIM, scale=0.5).reshape(T_PAD, HC_DIM)
    y = np.zeros((T_PAD, D), dtype=np.float32)
    for h in range(HC_MULT):
        x_h = x_flat[:, h * D:(h + 1) * D]
        y += x_h * pre[:, h:h + 1]
    out = float32_to_bf16(y)
    buffers = {
        "v1": pre.reshape(-1).astype(np.float32),
        "v2": out.reshape(-1),
        "v3": x_flat.reshape(-1).astype(np.float32),
    }
    return buffers, {"v2": buffers["v2"]}


# ===========================================================================
# swa_rope_step (gather per-token RoPE cos/sin rows by position_ids)
# ===========================================================================
def build_swa_rope_step(meta, generator, ints):
    """Gather rope_cos_t/rope_sin_t from freqs tables by position_ids.

    .pto:
      v1: [8, 64] bf16 (rope_cos_t output [T, ROPE_HEAD_DIM])
      v2: [8, 64] bf16 (rope_sin_t output [T, ROPE_HEAD_DIM])
      v3: [8] i32 (position_ids [T])
      v4: [16384, 64] bf16 (freqs_cos [MAX_SEQ_LEN, ROPE_HEAD_DIM])
      v5: [16384, 64] bf16 (freqs_sin [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    """
    del ints
    T_PAD = 8
    rope_w = ROPE_DIM
    table_rows = 16384
    cos_tbl = make_bf16(generator, table_rows * rope_w, scale=1.0, positive=True).reshape(table_rows, rope_w)
    sin_tbl = make_bf16(generator, table_rows * rope_w, scale=1.0, positive=True).reshape(table_rows, rope_w)
    pos_ids = generator.integers(0, table_rows, size=T_PAD).astype(np.int32)
    cos_out = np.zeros((T_PAD, rope_w), dtype=np.uint16)
    sin_out = np.zeros((T_PAD, rope_w), dtype=np.uint16)
    for t, p in enumerate(pos_ids):
        cos_out[t] = cos_tbl[int(p)]
        sin_out[t] = sin_tbl[int(p)]
    buffers = {
        "v1": cos_out.reshape(-1),
        "v2": sin_out.reshape(-1),
        "v3": pos_ids.reshape(-1).astype(np.int32),
        "v4": cos_tbl.reshape(-1),
        "v5": sin_tbl.reshape(-1),
    }
    return buffers, {"v1": buffers["v1"], "v2": buffers["v2"]}


# ===========================================================================
# swa_cache_insert_valid_bias (commit decode KV + build softmax valid bias)
# ===========================================================================
def build_swa_cache_insert_valid_bias(meta, generator, ints):
    """Insert current decode KV into cache + build additive valid bias.

    .pto:
      v1: [16384, 512] bf16 (kv_cache_flat [ORI_BLOCK_NUM*BLOCK_SIZE, HEAD_DIM])
      v2: [8] i64 (swa_slot_mapping [T])
      v3: [8, 512] bf16 (kv [T, HEAD_DIM])
      v4: [8] i32 (swa_lens [T])
      v5: [8, 128] f32 (sparse_bias output [T, WIN])

    sparse_bias[t, j] = (valid(j) - 1) * -NEG_INF, where valid(j) = 1 if j < swa_lens[t] else 0.
    """
    del ints
    cache = make_bf16(generator, 16384 * HEAD_DIM, scale=0.05).reshape(16384, HEAD_DIM)
    slot_map = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    kv = make_bf16(generator, T * HEAD_DIM, scale=0.5).reshape(T, HEAD_DIM)
    swa_lens = generator.integers(1, WIN + 1, size=T).astype(np.int32)
    sparse_bias = np.zeros((T, WIN), dtype=np.float32)

    new_cache = cache.copy()
    for t in range(T):
        row = int(slot_map[t])
        if row >= 0:
            new_cache[row] = kv[t]
    for t in range(T):
        for j in range(WIN):
            valid = 1.0 if j < int(swa_lens[t]) else 0.0
            sparse_bias[t, j] = (valid - 1.0) * (-NEG_INF)
    buffers = {
        "v1": new_cache.reshape(-1),
        "v2": slot_map.reshape(-1).astype(np.int64),
        "v3": kv.reshape(-1),
        "v4": swa_lens.reshape(-1).astype(np.int32),
        "v5": sparse_bias.reshape(-1).astype(np.float32),
    }
    return buffers, {"v1": buffers["v1"], "v5": buffers["v5"]}


# ===========================================================================
# swa_gather_kv (gather sliding-window KV rows into swa_kv_flat)
# ===========================================================================
def build_swa_gather_kv(meta, generator, ints):
    """Gather swa_kv_flat[t*WIN : (t+1)*WIN] from kv_cache by swa_indices.

    .pto:
      v1: [1024, 512] bf16 (swa_kv_flat [T*WIN, HEAD_DIM])
      v2: [8, 128] i32 (swa_indices [T, WIN])
      v3: [16384, 512] bf16 (kv_cache_flat [ORI_BLOCK_NUM*BLOCK_SIZE, HEAD_DIM])
      v4/v5: i32 spmd scalars

    Invalid slots (-1) gather a zero row.
    """
    del ints
    swa_kv = np.zeros((T * WIN, HEAD_DIM), dtype=np.uint16)
    swa_indices = generator.integers(-1, 16384, size=T * WIN).reshape(T, WIN).astype(np.int32)
    # ensure each row has at least one valid slot
    for t in range(T):
        if (swa_indices[t] < 0).all():
            swa_indices[t, 0] = t
    kv_cache = make_bf16(generator, 16384 * HEAD_DIM, scale=0.5).reshape(16384, HEAD_DIM)
    for t in range(T):
        for j in range(WIN):
            slot = int(swa_indices[t, j])
            if slot >= 0:
                swa_kv[t * WIN + j] = kv_cache[slot]
            # else: stays zero
    buffers = {
        "v1": swa_kv.reshape(-1),
        "v2": swa_indices.reshape(-1).astype(np.int32),
        "v3": kv_cache.reshape(-1),
        "v4": np.array([0], dtype=np.int32),
        "v5": np.array([1], dtype=np.int32),
    }
    return buffers, {"v1": buffers["v1"]}


# ===========================================================================
# Builder registry
# ===========================================================================
BUILDERS = {
    "comb_sinkhorn": build_comb_sinkhorn,
    "merge_norm": build_merge_norm,
    "kv_score_proj": build_kv_score_proj,
    "kv_score_proj_0": build_kv_score_proj_0,
    "kv_touch": build_kv_touch,
    "mix_x": build_mix_x,
    "proj_a_mm": build_proj_a_mm,
    "proj_b_act": build_proj_b_act,
    "proj_b_mm": build_proj_b_mm,
    "q_rope_prepare": build_q_rope_prepare,
    "rms_norm": build_rms_norm,
    "rope": build_rope,
    "rope_cs": build_rope_cs,
    "scatter_softmax_pool": build_scatter_softmax_pool,
    "score_mat": build_score_mat,
    "split_pre_post": build_split_pre_post,
    "swa_cache_insert_valid_bias": build_swa_cache_insert_valid_bias,
    "swa_gather_kv": build_swa_gather_kv,
    "swa_rope_step": build_swa_rope_step,
}


def run_case(case_name: str):
    """Generate buffers+golden for a kernel case (mirrors qwen3 golden_lib)."""
    meta = load_case_meta()
    generator = rng()
    ints = load_int32_assignments()
    buffers, golden = BUILDERS[case_name](meta, generator, ints)
    write_buffers(meta, buffers)
    write_golden(meta, golden)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: attention_kernels.py <case_name>")
        sys.exit(1)
    run_case(sys.argv[1])
