# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Indexed BCA attention, adapted from Bosch's fused indexed execution strategy.

No routed [queries, budget, heads, dim] K/V or dense attention scores are
materialized. Training uses FP32 atomic accumulation for shared source gradients.
"""

import triton
import triton.language as tl


@triton.jit
def indexed_forward(
    Q,
    K,
    V,
    ROUTE,
    VALID,
    O,
    LSE,
    BASE,
    BASE_LSE,
    FRAME_IDS,
    VIEWS: tl.constexpr,
    MULTI_FRAME: tl.constexpr,
    N: tl.constexpr,
    PIXELS: tl.constexpr,
    ROLES: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    OFFSET,
    BD: tl.constexpr,
    BL: tl.constexpr,
    BQ: tl.constexpr,
    HAS_BASE: tl.constexpr,
):
    # Bosch's four-query CTA amortizes launch/scheduling overhead for B_local=32.
    rows = tl.program_id(0) * BQ + tl.arange(0, BQ)
    head = tl.program_id(1)
    slots, dims = tl.arange(0, BL), tl.arange(0, BD)
    route_rows = (rows // (ROLES * PIXELS) % VIEWS * PIXELS + rows % PIXELS) * L
    route_mask = (rows[:, None] < N) & (slots[None, :] < L)
    valid = tl.load(VALID + route_rows[:, None] + slots[None, :], route_mask, 0).to(tl.int1)
    source = tl.load(ROUTE + route_rows[:, None] + slots[None, :], route_mask, 0)
    if MULTI_FRAME:
        frame = tl.load(FRAME_IDS + rows // (VIEWS * ROLES * PIXELS), rows < N, 0)
        source += frame[:, None] * PIXELS
    else:
        source += OFFSET
    qoffset = (rows[:, None] * HQ + head) * D + dims[None, :]
    qmask = (rows[:, None] < N) & (dims[None, :] < D)
    q = tl.load(Q + qoffset, qmask, 0).to(tl.float32)
    offsets = (source[:, :, None] * HK + head // (HQ // HK)) * D + dims[None, None, :]
    mask = valid[:, :, None] & (dims[None, None, :] < D)
    k = tl.load(K + offsets, mask, 0).to(tl.float32)
    scores = tl.sum(k * q[:, None, :], 2) * (D**-0.5)
    scores = tl.where(valid, scores, -float("inf"))
    maximum = tl.max(scores, 1)
    if HAS_BASE:
        base_lse = tl.load(BASE_LSE + rows * HQ + head, rows < N, -float("inf"))
        maximum = tl.maximum(maximum, base_lse)
    present = maximum != -float("inf")
    maximum = tl.where(present, maximum, 0.0)
    weights = tl.where(valid, tl.exp(scores - maximum[:, None]), 0.0)
    denominator = tl.sum(weights, 1)
    if HAS_BASE:
        base_weight = tl.exp(base_lse - maximum)
        denominator += base_weight
    denominator = tl.where(present, denominator, 1.0)
    probability = weights / denominator[:, None]
    v = tl.load(V + offsets, mask, 0).to(tl.float32)
    out = tl.sum(probability[:, :, None] * v, 1)
    if HAS_BASE:
        base = tl.load(BASE + qoffset, qmask, 0).to(tl.float32)
        base = tl.where((base_lse != -float("inf"))[:, None], base, 0.0)
        out += base * (base_weight / denominator)[:, None]
    tl.store(O + qoffset, out, qmask)
    tl.store(LSE + rows * HQ + head, tl.where(present, maximum + tl.log(denominator), -float("inf")), rows < N)


@triton.jit
def indexed_backward(
    Q,
    K,
    V,
    ROUTE,
    VALID,
    O,
    LSE,
    DO,
    DL,
    DQ,
    DK,
    DV,
    PIXELS: tl.constexpr,
    ROLES: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    OFFSET,
    BD: tl.constexpr,
    BL: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slots, dims = tl.arange(0, BL), tl.arange(0, BD)
    route_row = (row // (ROLES * PIXELS) * PIXELS + row % PIXELS) * L
    valid = tl.load(VALID + route_row + slots, slots < L, 0).to(tl.int1)
    source = tl.load(ROUTE + route_row + slots, slots < L, 0) + OFFSET
    qoffset = (row * HQ + head) * D + dims
    q = tl.load(Q + qoffset, dims < D, 0).to(tl.float32)
    dout = tl.load(DO + qoffset, dims < D, 0).to(tl.float32)
    out = tl.load(O + qoffset, dims < D, 0).to(tl.float32)
    lse = tl.load(LSE + row * HQ + head)
    dlse = tl.load(DL + row * HQ + head)
    offsets = (source[:, None] * HK + head // (HQ // HK)) * D + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < D)
    k = tl.load(K + offsets, mask, 0).to(tl.float32)
    v = tl.load(V + offsets, mask, 0).to(tl.float32)
    scores = tl.sum(k * q[None, :], 1) * (D**-0.5)
    p = tl.where(valid, tl.exp(scores - lse), 0.0)
    ds = p * (tl.sum(v * dout[None, :], 1) - tl.sum(out * dout, 0) + dlse) * (D**-0.5)
    tl.store(DQ + qoffset, tl.sum(ds[:, None] * k, 0), dims < D)
    tl.atomic_add(DK + offsets, ds[:, None] * q[None, :], mask, sem="relaxed")
    tl.atomic_add(DV + offsets, p[:, None] * dout[None, :], mask, sem="relaxed")


@triton.jit
def lse_backward(
    Q,
    K,
    LSE,
    DL,
    GRAD,
    QLEN: tl.constexpr,
    KLEN: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QUERY_GRAD: tl.constexpr,
    BD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """LSE VJP for FlashAttention, tiled without saving/recreating a full matrix."""
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    columns = tl.arange(0, BLOCK)
    dims = tl.arange(0, BD)
    qbase = (batch * HQ + head) * QLEN * D
    kbase = (batch * HK + head // (HQ // HK)) * KLEN * D
    lbase = (batch * HQ + head) * QLEN
    acc = tl.zeros((BLOCK, BD), tl.float32)
    if QUERY_GRAD:
        q = tl.load(Q + qbase + rows[:, None] * D + dims[None, :], (rows[:, None] < QLEN) & (dims[None, :] < D), 0)
        lse = tl.load(LSE + lbase + rows, rows < QLEN, 0)
        dl = tl.load(DL + lbase + rows, rows < QLEN, 0)
        for start in range(tl.cdiv(KLEN, BLOCK)):
            tokens = start * BLOCK + columns
            k = tl.load(
                K + kbase + tokens[:, None] * D + dims[None, :], (tokens[:, None] < KLEN) & (dims[None, :] < D), 0
            )
            scores = tl.dot(q, tl.trans(k)) * (D**-0.5)
            p = tl.exp(scores - lse[:, None])
            p = tl.where((rows[:, None] < QLEN) & (tokens[None, :] < KLEN), p, 0.0)
            ds = p * dl[:, None] * (D**-0.5)
            acc += tl.dot(ds.to(k.dtype), k)
        tl.store(GRAD + qbase + rows[:, None] * D + dims[None, :], acc, (rows[:, None] < QLEN) & (dims[None, :] < D))
    else:
        k = tl.load(K + kbase + rows[:, None] * D + dims[None, :], (rows[:, None] < KLEN) & (dims[None, :] < D), 0)
        for start in range(tl.cdiv(QLEN, BLOCK)):
            tokens = start * BLOCK + columns
            q = tl.load(
                Q + qbase + tokens[:, None] * D + dims[None, :], (tokens[:, None] < QLEN) & (dims[None, :] < D), 0
            )
            lse = tl.load(LSE + lbase + tokens, tokens < QLEN, 0)
            dl = tl.load(DL + lbase + tokens, tokens < QLEN, 0)
            scores = tl.dot(k, tl.trans(q)) * (D**-0.5)
            p = tl.exp(scores - lse[None, :])
            p = tl.where((rows[:, None] < KLEN) & (tokens[None, :] < QLEN), p, 0.0)
            ds = p * dl[None, :] * (D**-0.5)
            acc += tl.dot(ds.to(q.dtype), q)
        # One independent partial per query head; reduce GQA groups in Python.
        target = ((batch * HQ + head) * KLEN + rows[:, None]) * D + dims[None, :]
        tl.store(GRAD + target, acc, (rows[:, None] < KLEN) & (dims[None, :] < D))


@triton.jit
def residual_scores(
    V,
    REP,
    CELL,
    SCORES,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NC: tl.constexpr,
    BD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    dims = tl.arange(0, BD)
    cell = tl.load(CELL + tokens, tokens < N, 0)
    representative = tl.load(REP + bh * NC + cell)
    offset = ((batch * N + tokens[:, None]) * H + head) * D + dims[None, :]
    ref_offset = ((batch * N + representative[:, None]) * H + head) * D + dims[None, :]
    mask = (tokens[:, None] < N) & (dims[None, :] < D)
    value = tl.load(V + offset, mask, 0).to(tl.float32)
    ref = tl.load(V + ref_offset, mask, 0).to(tl.float32)
    score = tl.sum((value - ref) * (value - ref), 1)
    score = tl.where(tokens == representative, float("inf"), score)
    tl.store(SCORES + bh * N + tokens, score, tokens < N)


@triton.jit
def window_residual_scores(
    V,
    REP,
    CELL,
    ALLOWED,
    SCORES,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NC: tl.constexpr,
    FRAMES: tl.constexpr,
    PIXELS: tl.constexpr,
    VIEWS: tl.constexpr,
    ROLES: tl.constexpr,
    BD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    window, view, role = batch // (VIEWS * ROLES), batch // ROLES % VIEWS, batch % ROLES
    source_head = (view * ROLES + role) * H + head
    dims = tl.arange(0, BD)
    cell = tl.load(CELL + tokens, tokens < N, 0)
    representative = tl.load(REP + source_head * NC + cell)
    frame = tl.load(ALLOWED + window * (N // PIXELS) + tokens // PIXELS, tokens < N, 0)
    reference_frame = tl.load(ALLOWED + window * (N // PIXELS) + representative // PIXELS, tokens < N, 0)
    row = ((role * VIEWS + view) * FRAMES + frame) * PIXELS + tokens % PIXELS
    reference_row = ((role * VIEWS + view) * FRAMES + reference_frame) * PIXELS + representative % PIXELS
    offset = (row[:, None] * H + head) * D + dims[None, :]
    reference_offset = (reference_row[:, None] * H + head) * D + dims[None, :]
    mask = (tokens[:, None] < N) & (dims[None, :] < D)
    value = tl.load(V + offset, mask, 0).to(tl.float32)
    reference = tl.load(V + reference_offset, mask, 0).to(tl.float32)
    score = tl.sum((value - reference) * (value - reference), 1)
    score = tl.where(tokens == representative, float("inf"), score)
    tl.store(SCORES + bh * N + tokens, score, tokens < N)


@triton.jit
def pack_window_kv(
    K,
    V,
    RK,
    RV,
    INDICES,
    ALLOWED,
    OTHERS,
    OK,
    OV,
    COUNT: tl.constexpr,
    N_ALLOWED: tl.constexpr,
    VIEWS: tl.constexpr,
    FRAMES: tl.constexpr,
    PIXELS: tl.constexpr,
    ROLES: tl.constexpr,
    KEEP: tl.constexpr,
    REGISTERS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    dim, head = i % DIM, i // DIM % HEADS
    length = REGISTERS + (VIEWS - 1) * ROLES * KEEP
    token, batch = i // (DIM * HEADS) % length, i // (DIM * HEADS * length)
    window, view = batch // VIEWS, batch % VIEWS
    register = valid & (token < REGISTERS)
    register_offset = ((batch * REGISTERS + token) * HEADS + head) * DIM + dim
    key = tl.load(RK + register_offset, register, 0)
    value = tl.load(RV + register_offset, register, 0)
    if KEEP > 0:
        residual = valid & (token >= REGISTERS)
        position = token - REGISTERS
        other = tl.load(OTHERS + view * (VIEWS - 1) + position // (ROLES * KEEP), residual, 0)
        role, retained = position // KEEP % ROLES, position % KEEP
        selected = tl.load(
            INDICES + (((window * VIEWS + other) * ROLES + role) * HEADS + head) * KEEP + retained, residual, 0
        )
        frame = tl.load(ALLOWED + window * N_ALLOWED + selected // PIXELS, residual, 0)
        row = ((role * VIEWS + other) * FRAMES + frame) * PIXELS + selected % PIXELS
        offset = (row * HEADS + head) * DIM + dim
        key = tl.where(residual, tl.load(K + offset, residual, 0), key)
        value = tl.where(residual, tl.load(V + offset, residual, 0), value)
    tl.store(OK + i, key, valid)
    tl.store(OV + i, value, valid)
