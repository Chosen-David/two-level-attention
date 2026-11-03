"""单层 Req-Batched 当前 query NFA selector（生产唯一选择入口）。"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only or missing triton
    triton = None
    tl = None

from ..ops.paged_region_kernels_nfa import INVALID_GATHER_IDX
from .selection.unique_ranking import (
    _fast_launch, _tile_keys_from_scores_body, rank_unique_batched)

def _has_triton() -> bool:
    return triton is not None and tl is not None


class FallbackRequired(Exception):
    """The request-batched selector does not support this geometry."""


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _has_triton():

    @triton.jit
    def _nfa_page_stats_body(
        block_group, row,
        KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
        STATS_LEN, P_MIN, P_MAX, P_SUM,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, D_BLOCK: tl.constexpr, BS: tl.constexpr,
        KT_CAP: tl.constexpr, L: tl.constexpr, B: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        m = row // HK
        kv = row % HK
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)
        active = tl.load(ACTIVE_REQ + req) >= 0
        length = tl.load(LENGTHS + req).to(tl.int32)
        stats_len_offset = (layer + LAYER_OFFSET) * MAX_REQ + req_id
        previous = tl.load(STATS_LEN + stats_len_offset).to(tl.int32)
        incremental = active & (previous > 0) & (length == previous + 1)
        last_blk = (length - 1) // BS
        owner = block_group == last_blk // BLOCKS_PER_PROGRAM

        if active & (~incremental | owner):
            group_start = block_group * BLOCKS_PER_PROGRAM
            blk_start = tl.where(incremental, last_blk, group_start)
            blk_end = tl.where(
                incremental, last_blk + 1,
                tl.minimum(group_start + BLOCKS_PER_PROGRAM, KT_CAP))
            offs_d = tl.arange(0, D_BLOCK)
            dim_mask = offs_d < DIM_NUMEL
            dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0).to(tl.int64)
            kv_addr = tl.load(KV_PTRS + layer)
            kv_ptr = kv_addr.to(tl.pointer_type(P_MIN.dtype.element_ty))
            for blk in range(blk_start, blk_end):
                tok = blk * BS + tl.arange(0, BS)
                visible = tok < length
                lp = tok // PAGE
                off = tok % PAGE
                page_valid = lp < MAX_BLOCKS
                phys = tl.load(
                    BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
                    + req_id * MAX_BLOCKS + lp,
                    mask=page_valid & visible, other=0).to(tl.int64)
                src = (kv_ptr
                       + phys[:, None] * HK * PAGE * HEAD_DIM
                       + kv * PAGE * HEAD_DIM
                       + off[:, None] * HEAD_DIM
                       + dim[None, :])
                read_mask = (visible & page_valid)[:, None] & dim_mask[None, :]
                val = tl.load(src, mask=read_mask, other=0.0).to(tl.float32)
                vmin = tl.min(val, axis=0)
                vmax = tl.max(val, axis=0)
                vsum = tl.sum(val, axis=0)
                p_base = ((layer * MAX_REQ + req_id) * KT_CAP * HK * DIM_NUMEL
                          + blk * HK * DIM_NUMEL + kv * DIM_NUMEL + offs_d)
                tl.store(P_MIN + p_base, vmin.to(P_MIN.dtype.element_ty), mask=dim_mask)
                tl.store(P_MAX + p_base, vmax.to(P_MAX.dtype.element_ty), mask=dim_mask)
                tl.store(P_SUM + p_base, vsum.to(P_SUM.dtype.element_ty), mask=dim_mask)

    @triton.jit
    def _nfa_page_stats_kernel(
        KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
        STATS_LEN, P_MIN, P_MAX, P_SUM,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, D_BLOCK: tl.constexpr, BS: tl.constexpr,
        KT_CAP: tl.constexpr, L: tl.constexpr, B: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """稳态仅一个 program 更新最新块，全量重建按块组并行。"""
        _nfa_page_stats_body(
            tl.program_id(0), tl.program_id(1),
            KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
            STATS_LEN, P_MIN, P_MAX, P_SUM,
            MAX_BLOCKS=MAX_BLOCKS, PAGE=PAGE, HK=HK, HEAD_DIM=HEAD_DIM,
            DIM_NUMEL=DIM_NUMEL, D_BLOCK=D_BLOCK, BS=BS, KT_CAP=KT_CAP,
            L=L, B=B, BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
            MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)

    @triton.jit
    def _nfa_l1_scores_kernel(
        KMIN, KMAX, KSUM, Q, LENGTHS, ACTIVE_REQ, SKIP_FAR, SKIP_NEAR,
        VALID_REQ, STATS_LEN, FAR_OUT, NEAR_OUT, T,
        stride_kt, stride_kh, stride_kt2, stride_kh2,
        stride_kts, stride_khs, stride_qh,
        PAGE: tl.constexpr, BS: tl.constexpr,
        SINK: tl.constexpr, NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
        POOL_NEAR_LEN: tl.constexpr, SINK_POOL_LO: tl.constexpr,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        B: tl.constexpr, STRIDE_KB: tl.constexpr, STRIDE_KB2: tl.constexpr,
        STRIDE_KS: tl.constexpr, STRIDE_KR: tl.constexpr,
        STRIDE_KR2: tl.constexpr, STRIDE_KRS: tl.constexpr,
        HAS_AVG: tl.constexpr,
        FAR_AVG: tl.constexpr, NEAR_AVG: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """一次计算 GQA 粗分、区域可见性和 far/near L1 候选分数。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        batch = pid_h // Hkv
        kv = pid_h % Hkv
        layer = batch // B
        req = batch % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        length = tl.load(LENGTHS + req).to(tl.int32)
        # 页统计已在本 kernel 启动前完成；每个 request 只由首个 L1 program
        # 提交长度，替代独立的逐层 commit launch。
        if (pid_t == 0) & (kv == 0):
            if tl.load(ACTIVE_REQ + req) >= 0:
                tl.store(STATS_LEN + (layer + LAYER_OFFSET) * MAX_REQ + req_id,
                         length)
        skip_far = tl.load(SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        skip_near = tl.load(SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        far_on = skip_far == 0
        near_on = skip_near == 0

        sink_hi = tl.minimum(SINK, length)
        swa_hi = tl.maximum(sink_hi, length - WINDOW)
        far_blk = (length - WINDOW - NEAR_LEN) // PAGE
        orig_far_hi = tl.minimum(swa_hi, tl.maximum(sink_hi, far_blk * PAGE))

        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        block_lo = offs_t * BS
        block_hi = block_lo + BS
        has_sink = block_lo < sink_hi
        has_far = far_on & (block_hi > sink_hi) & (block_lo < orig_far_hi)
        has_near = near_on & (block_hi > orig_far_hi) & (block_lo < swa_hi)
        has_swa = (block_hi > swa_hi) & (block_lo < length)
        block_valid = (block_lo < length) & (
            has_sink | has_far | has_near | has_swa)

        # 与 _batched_geometry 相同的候选池边界；跳过区域和非对齐 bypass
        # 页在这里直接置为 -inf，不再物化扩展后的 bool mask。
        near_blks = tl.maximum(
            (length - WINDOW - POOL_NEAR_LEN) // BS, SINK_POOL_LO)
        kt_eff = (length + BS - 1) // BS
        swa_lo_blk = tl.maximum(near_blks, kt_eff - tl.maximum(1, WINDOW // BS))
        swa_lo_tok = tl.maximum(length - WINDOW, 0)
        swa_byp_blk = swa_lo_tok // BS
        swa_byp_cond = ((swa_lo_tok % BS != 0)
                        & (swa_byp_blk >= SINK_POOL_LO))
        if SINK % BS != 0:
            swa_byp_cond = swa_byp_cond & (swa_byp_blk != SINK_POOL_LO)
        far_pool = ((offs_t >= SINK_POOL_LO) & (offs_t < near_blks)
                    & far_on)
        near_pool = ((offs_t >= near_blks) & (offs_t < swa_lo_blk)
                     & near_on
                     & ~((offs_t == swa_byp_blk) & swa_byp_cond))
        if SINK % BS != 0:
            far_pool = far_pool & (offs_t != SINK_POOL_LO)
            near_pool = near_pool & (offs_t != SINK_POOL_LO)
        far_pool = far_pool & block_valid
        near_pool = near_pool & block_valid

        KMIN += layer * STRIDE_KB + req_id * STRIDE_KR
        KMAX += layer * STRIDE_KB2 + req_id * STRIDE_KR2
        if HAS_AVG:
            KSUM += layer * STRIDE_KS + req_id * STRIDE_KRS
        offs_d = tl.arange(0, D)
        kmin = tl.load(
            KMIN + offs_t[:, None] * stride_kt + kv * stride_kh
            + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        kmax = tl.load(
            KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2
            + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        if HAS_AVG:
            ksum = tl.load(
                KSUM + offs_t[:, None] * stride_kts + kv * stride_khs
                + offs_d[None, :],
                mask=mask_t[:, None], other=0.0)
            # 保持旧 OUT_AVG 的 BF16 中间舍入，再参与 FP32 点积。
            kavg = (ksum.to(tl.float32) / BS).to(
                FAR_OUT.dtype.element_ty).to(tl.float32)

        mm_acc = tl.zeros((BT,), dtype=tl.float32)
        avg_acc = tl.zeros((BT,), dtype=tl.float32)
        for j in tl.static_range(G):
            h = pid_h.to(tl.int64) * G + j
            q = tl.load(Q + h * stride_qh + offs_d)
            q_low = tl.minimum(q, 0.0)
            q_high = tl.maximum(q, 0.0)
            mm = (tl.sum(kmin * q_low[None, :], axis=1)
                  + tl.sum(kmax * q_high[None, :], axis=1))
            # 旧路径先逐 query head 转 BF16/FP16，再对 GQA 求均值。
            mm_acc += mm.to(FAR_OUT.dtype.element_ty).to(tl.float32)
            if HAS_AVG:
                avg = tl.sum(kavg * q[None, :], axis=1)
                avg_acc += avg.to(FAR_OUT.dtype.element_ty).to(tl.float32)
        mm_score = (mm_acc / G).to(FAR_OUT.dtype.element_ty)
        far_score = mm_score
        near_score = mm_score
        if HAS_AVG:
            avg_score = (avg_acc / G).to(FAR_OUT.dtype.element_ty)
            if FAR_AVG:
                far_score = avg_score
            if NEAR_AVG:
                near_score = avg_score
        far_score = tl.where(far_pool, far_score, float("-inf"))
        near_score = tl.where(near_pool, near_score, float("-inf"))
        out_offs = pid_h.to(tl.int64) * T + offs_t
        tl.store(FAR_OUT + out_offs, far_score, mask=mask_t)
        tl.store(NEAR_OUT + out_offs, near_score, mask=mask_t)

    @triton.jit
    def _nfa_fine_scores_paged_kernel(
        KV_PTRS, BLOCK_TABLE, VALID_REQ, DIMS, Q, CAND, KEEP, OWNERS,
        GENERATIONS, UNIQUE, SCORES, PARTIAL_M, PARTIAL_L, MSTAT, LSTAT,
        TILE_COUNTERS, RANK_OUT, LOW, HIGH, QUOTA,
        KV_REF, R, NUM_TILES,
        stride_qm, stride_qh, stride_ch, stride_cr,
        S: tl.constexpr, MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
        D_BLOCK: tl.constexpr, G_PAD: tl.constexpr,
        HKV: tl.constexpr, G: tl.constexpr, B: tl.constexpr,
        TILE: tl.constexpr, MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
        FUSE_STATE: tl.constexpr, FUSE_RANK: tl.constexpr,
        RANK_GROUPS: tl.constexpr, RANK_K: tl.constexpr,
        META_GROUP: tl.constexpr, TOKEN_BITS: tl.constexpr,
        PROB_BF16: tl.constexpr,
    ):
        """按 tile 并行计算 BF16 score；避免为 online softmax 重读 paged K。"""
        row = tl.program_id(0)
        t = tl.program_id(1)
        m = row // HKV
        kv = row % HKV
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        kv_addr = tl.load(KV_PTRS + layer)
        kv_ptr = kv_addr.to(tl.pointer_type(KV_REF.dtype.element_ty))
        offs_d = tl.arange(0, D_BLOCK)
        dim_mask = offs_d < DIM_NUMEL
        dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0).to(tl.int64)

        offs_r = t * TILE + tl.arange(0, TILE)
        mask_r = offs_r < R
        keep = tl.load(KEEP + row * stride_ch + offs_r * stride_cr,
                       mask=mask_r, other=False)
        read_r = mask_r & keep
        cand = tl.load(CAND + row * stride_ch + offs_r * stride_cr,
                       mask=read_r, other=0)
        in_range = (cand >= 0) & (cand < S)
        owner = tl.load(OWNERS + row * S + cand,
                        mask=read_r & in_range, other=-1)
        generation = tl.load(GENERATIONS + row)
        unique = read_r & in_range & (owner == generation * R + offs_r)
        tl.store(UNIQUE + row * R + offs_r, unique, mask=mask_r)
        safe = tl.where(read_r, cand, 0)
        lp = safe // PAGE
        off = safe % PAGE
        lp_valid = lp < MAX_BLOCKS
        phys = tl.load(
            BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
            + req_id * MAX_BLOCKS + lp,
            mask=read_r & lp_valid, other=0).to(tl.int64)
        src = (kv_ptr
               + phys[:, None] * HK * PAGE * HEAD_DIM
               + kv * PAGE * HEAD_DIM
               + off[:, None] * HEAD_DIM
               + dim[None, :])
        k = tl.load(
            src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :],
            other=0.0)
        for qrow in tl.static_range(G):
            q = tl.load(
                Q + m * stride_qm + (kv * G + qrow) * stride_qh + offs_d,
                mask=dim_mask, other=0.0)
            s = tl.sum(k.to(tl.float32) * q[None, :], axis=1)
            s = tl.where(read_r & lp_valid, s, float("-inf"))
            s = s.to(tl.bfloat16).to(tl.float32)
            tl.store(SCORES + (row * G + qrow) * R + offs_r,
                     s.to(tl.bfloat16), mask=mask_r)
            tile_m = tl.max(s, axis=0)
            safe_m = tl.where(tile_m == float("-inf"), 0.0, tile_m)
            tile_l = tl.sum(tl.exp(s - safe_m), axis=0)
            tile_l = tl.where(tile_m == float("-inf"), 0.0, tile_l)
            partial_offset = (row * G + qrow) * NUM_TILES + t
            tl.store(PARTIAL_M + partial_offset, tile_m)
            tl.store(PARTIAL_L + partial_offset, tile_l)

        if FUSE_STATE:
            # 最后一个完成的 tile CTA 直接按原 tile 顺序归并 softmax 状态，
            # 省去独立 state launch；acq_rel 保证其他 CTA 的 partial 写入可见。
            tl.debug_barrier()
            finished = tl.atomic_add(
                TILE_COUNTERS + row, 1, sem="acq_rel", scope="gpu")
            if finished == NUM_TILES - 1:
                # GQA 各 query head 之间没有归约依赖；按 lane 并行，保持每个
                # head 内部仍严格按 tile 顺序执行同一组 FP32 标量运算。
                qrows = tl.arange(0, G_PAD)
                active_q = qrows < G
                mstat = tl.full((G_PAD,), float("-inf"), tl.float32)
                lstat = tl.zeros((G_PAD,), tl.float32)
                for tile_idx in range(NUM_TILES):
                    offset = (row * G * NUM_TILES
                              + qrows * NUM_TILES + tile_idx)
                    part_m = tl.load(PARTIAL_M + offset, mask=active_q,
                                     other=float("-inf"))
                    part_l = tl.load(PARTIAL_L + offset, mask=active_q, other=0.0)
                    new_m = tl.maximum(mstat, part_m)
                    old_corr = tl.exp(mstat - new_m)
                    old_corr = tl.where(
                        (mstat == float("-inf")) & (new_m == float("-inf")),
                        1.0, old_corr)
                    tile_corr = tl.exp(part_m - new_m)
                    tile_corr = tl.where(part_m == float("-inf"), 0.0, tile_corr)
                    lstat = lstat * old_corr + part_l * tile_corr
                    mstat = new_m
                tl.store(MSTAT + row * G + qrows, mstat, mask=active_q)
                tl.store(LSTAT + row * G + qrows, lstat, mask=active_q)
                if FUSE_RANK:
                    tl.debug_barrier()
                    for rank_tile in range(RANK_GROUPS):
                        _tile_keys_from_scores_body(
                            row, rank_tile, CAND, SCORES, MSTAT, LSTAT, UNIQUE,
                            LOW, HIGH, QUOTA, RANK_OUT,
                            R=R, S=S, GROUPS=RANK_GROUPS, K=RANK_K,
                            META_GROUP=META_GROUP, TOKEN_BITS=TOKEN_BITS, G=G,
                            PROB_BF16=PROB_BF16)

    @triton.jit
    def _nfa_fine_softmax_state_kernel(
        PARTIAL_M, PARTIAL_L, MSTAT, LSTAT, NUM_TILES,
        G: tl.constexpr,
    ):
        """按 tile 顺序合并并行计算的部分 max/sum，避免全候选串行 exp。"""
        row = tl.program_id(0) // G
        qrow = tl.program_id(0) % G
        mstat = tl.full((), float("-inf"), tl.float32)
        lstat = tl.zeros((), tl.float32)
        for t in range(NUM_TILES):
            offset = row * G * NUM_TILES + qrow * NUM_TILES + t
            tile_m = tl.load(PARTIAL_M + offset)
            tile_l = tl.load(PARTIAL_L + offset)
            new_m = tl.maximum(mstat, tile_m)
            old_corr = tl.exp(mstat - new_m)
            old_corr = tl.where(
                (mstat == float("-inf")) & (new_m == float("-inf")),
                1.0, old_corr)
            tile_corr = tl.exp(tile_m - new_m)
            tile_corr = tl.where(tile_m == float("-inf"), 0.0, tile_corr)
            lstat = lstat * old_corr + tile_l * tile_corr
            mstat = new_m
        tl.store(MSTAT + row * G + qrow, mstat)
        tl.store(LSTAT + row * G + qrow, lstat)

    @triton.jit
    def _prepare_q_body(
        pid, Q_SRC, REQ_IDS, SCALES, DIM_IDX, OUT, S0, S1, S2,
        HQ: tl.constexpr, D: tl.constexpr, DQ: tl.constexpr,
        B: tl.constexpr, DQ_BLOCK: tl.constexpr,
    ):
        m_idx = pid // HQ
        hq = pid % HQ
        layer = m_idx // B
        req = m_idx % B
        req_id = tl.load(REQ_IDS + req)
        scale = tl.load(SCALES + layer)
        offs = tl.arange(0, DQ_BLOCK)
        mask = offs < DQ
        dim = tl.load(DIM_IDX + offs, mask=mask, other=0)
        src = Q_SRC + layer * S0 + req_id * S1 + hq * S2 + dim
        q = tl.load(src, mask=mask, other=0.0).to(tl.float32)
        q = (q * scale).to(tl.bfloat16).to(tl.float32)
        dst = OUT + m_idx * (HQ * DQ) + hq * DQ + offs
        tl.store(dst, q, mask=mask)

    @triton.jit
    def _prepare_q_kernel(
        Q_SRC, REQ_IDS, SCALES, DIM_IDX, OUT,
        S0, S1, S2,
        HQ: tl.constexpr, D: tl.constexpr, DQ: tl.constexpr,
        B: tl.constexpr, DQ_BLOCK: tl.constexpr,
    ):
        """将当前 post-RoPE Q 按请求行和指定维度收集到 fp32 workspace，
        并复现 ``(q_f32 * scale).to(bf16).to(f32)`` 的生产缩放口径。
        """
        _prepare_q_body(
            tl.program_id(0), Q_SRC, REQ_IDS, SCALES, DIM_IDX, OUT, S0, S1, S2,
            HQ=HQ, D=D, DQ=DQ, B=B, DQ_BLOCK=DQ_BLOCK)

    @triton.jit
    def _nfa_page_stats_prepare_q_kernel(
        KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
        STATS_LEN, P_MIN, P_MAX, P_SUM,
        Q_SRC, Q_REQ, SCALES, Q_OUT, S0, S1, S2,
        KEYS, VALUES, SLOTS, K_POOL, V_POOL,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, D_BLOCK: tl.constexpr, BS: tl.constexpr,
        KT_CAP: tl.constexpr, L: tl.constexpr, B: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
        HQ: tl.constexpr, D: tl.constexpr, DQ: tl.constexpr,
        DQ_BLOCK: tl.constexpr, STATS_PROGRAMS: tl.constexpr,
        STATS_ROWS: tl.constexpr, DV: tl.constexpr,
        BD: tl.constexpr, BV: tl.constexpr, HAS_WRITE: tl.constexpr,
    ):
        """横向融合页统计/Q 预处理；可选由最新块 owner 先写当前 KV。"""
        pid = tl.program_id(0)
        if pid < STATS_PROGRAMS:
            block_group = pid // STATS_ROWS
            row = pid % STATS_ROWS
            if HAS_WRITE:
                m = row // HK
                kv = row % HK
                req = m % B
                active = tl.load(ACTIVE_REQ + req) >= 0
                length = tl.load(LENGTHS + req).to(tl.int32)
                last_blk = (length - 1) // BS
                owner = block_group == last_blk // BLOCKS_PER_PROGRAM
                if active & owner:
                    slot = tl.load(SLOTS + req)
                    if slot >= 0:
                        phys = (slot // PAGE).to(tl.int64)
                        off = (slot % PAGE).to(tl.int64)
                        d = tl.arange(0, BD)
                        vd = tl.arange(0, BV)
                        key = tl.load(
                            KEYS + req * HK * D + kv * D + d,
                            mask=d < D, other=0.0)
                        tl.store(
                            K_POOL + phys * (HK * PAGE * D)
                            + kv * (PAGE * D) + off * D + d,
                            key, mask=d < D)
                        value = tl.load(
                            VALUES + req * HK * DV + kv * DV + vd,
                            mask=vd < DV, other=0.0)
                        tl.store(
                            V_POOL + phys * (HK * PAGE * DV)
                            + kv * (PAGE * DV) + off * DV + vd,
                            value, mask=vd < DV)
                    # 页统计随后在同一 CTA 读取刚写入的最新 K。
                    tl.debug_barrier()
            # L1 与最终 canonical union 均只消费 KV head 0；其他头仍须在
            # 上方写入当前 K/V，但其页 min/max/sum 没有 downstream 消费者。
            if row % HK == 0:
                _nfa_page_stats_body(
                    block_group, row,
                    KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
                    STATS_LEN, P_MIN, P_MAX, P_SUM,
                    MAX_BLOCKS=MAX_BLOCKS, PAGE=PAGE, HK=HK, HEAD_DIM=HEAD_DIM,
                    DIM_NUMEL=DIM_NUMEL, D_BLOCK=D_BLOCK, BS=BS, KT_CAP=KT_CAP,
                    L=L, B=B, BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
                    MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)
        else:
            _prepare_q_body(
                pid - STATS_PROGRAMS, Q_SRC, Q_REQ, SCALES, DIMS, Q_OUT,
                S0, S1, S2, HQ=HQ, D=D, DQ=DQ, B=B, DQ_BLOCK=DQ_BLOCK)

    @triton.jit
    def _candidate_pack_batched_kernel(
        I_F, I_N, FAR_SCORES, NEAR_SCORES, LENGTHS,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS, TILE_COUNTERS,
        K_F: tl.constexpr, K_N: tl.constexpr, KT: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, PAGE: tl.constexpr,
        NEAR_LEN: tl.constexpr, K2: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """Top-K 块单遍展开；保留下游 mask 所需的原始 virtual slot。"""
        row = tl.program_id(0)
        m = row // HKV
        h = row % HKV
        tl.store(TILE_COUNTERS + row, 0)
        req = m % B
        length = tl.load(LENGTHS + req)
        lo = tl.maximum(0, length - SWA)
        width = R + HEAD + TAIL

        # generation 隔离不同 selector 调用；同次调用仍由 atomic_max 选择
        # 最大 candidate slot，不再清空整个 CAP 的 owner 表。
        generation = tl.load(GENERATIONS + row) + 1
        tl.store(GENERATIONS + row, generation)
        for start in range(0, width, 1024):
            r = start + tl.arange(0, 1024)
            tagged_slot = generation * width + r
            token, active = _candidate_pack_virtual_slot(
                I_F, I_N, FAR_SCORES, NEAR_SCORES, row, r, length,
                K_F=K_F, K_N=K_N, KT=KT, R=R, HEAD=HEAD, TAIL=TAIL,
                SINK=SINK, SWA=SWA, SINK_POOL_LO=SINK_POOL_LO, BS=BS,
                live=r < width)
            tl.store(OUT + row * width + r, token, r < width)
            tl.store(VALID + row * width + r, active, r < width)
            tl.atomic_max(OWNERS + row * CAP + token, tagged_slot,
                          mask=active, sem="relaxed")

        if h == 0:
            layer = m // B
            req_id = tl.load(VALID_REQ + req).to(tl.int32)
            skip_far = tl.load(
                SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id) != 0
            skip_near = tl.load(
                SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id) != 0
            lower = SINK_POOL_LO * BS
            upper = tl.maximum(lower, lo)
            quota = tl.maximum(
                tl.minimum(length, K2) - SINK - (length - lo), 0)
            sink_hi = tl.minimum(length, SINK)
            swa_hi = tl.maximum(sink_hi, length - SWA)
            far_blk = (tl.maximum(length - SWA - NEAR_LEN, 0) // PAGE) * PAGE
            far_hi = tl.minimum(swa_hi, tl.maximum(sink_hi, far_blk))
            lower = tl.where(skip_far, far_hi, lower)
            upper = tl.where(skip_near, far_hi, upper)
            upper = tl.maximum(lower, upper)
            quota = tl.minimum(quota, upper - lower)
            tl.store(LOWER + m, lower)
            tl.store(UPPER + m, upper)
            tl.store(QUOTA + m, quota)

    @triton.jit
    def _candidate_pack_virtual_slot(
        I_F, I_N, FAR_SCORES, NEAR_SCORES, row, r, length,
        K_F: tl.constexpr, K_N: tl.constexpr, KT: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, live,
    ):
        lo = tl.maximum(0, length - SWA)
        far = r < K_F * BS
        near = (r >= K_F * BS) & (r < (K_F + K_N) * BS)
        l1 = far | near
        k1_width = (K_F + K_N) * BS
        byp0 = (r >= k1_width) & (r < k1_width + BS)
        byp1 = (r >= k1_width + BS) & (r < R)
        old = r < R

        far_idx = r // BS
        near_idx = (r - K_F * BS) // BS
        blk_f = tl.load(I_F + row * K_F + far_idx, mask=far, other=0)
        blk_n = tl.load(I_N + row * K_N + near_idx, mask=near, other=0)
        blk = tl.where(far, blk_f, blk_n)
        score_f = tl.load(FAR_SCORES + row * KT + blk_f, mask=far,
                          other=float("-inf"))
        score_n = tl.load(NEAR_SCORES + row * KT + blk_n, mask=near,
                          other=float("-inf"))
        score = tl.where(far, score_f, score_n)

        swa_byp_blk = lo // BS
        byp_blk = tl.where(byp0, SINK_POOL_LO, swa_byp_blk)
        byp0_keep = tl.full(r.shape, SINK % BS != 0, tl.int1)
        byp1_keep = ((lo % BS != 0) & (swa_byp_blk >= SINK_POOL_LO))
        if SINK % BS != 0:
            byp1_keep = byp1_keep & (swa_byp_blk != SINK_POOL_LO)
        token = tl.where(l1, blk * BS + r % BS, byp_blk * BS + r % BS)
        keep = tl.where(l1, score != float("-inf"),
                        tl.where(byp0, byp0_keep, byp1_keep))
        active = (old & keep & (token >= SINK) & (token < lo)
                  & (token >= 0) & (token < length))

        head = (r >= R) & (r < R + HEAD)
        tail_id = lo + r - R - HEAD
        new_id = tl.where(head, r - R, tail_id)
        new_valid = ((head & (new_id < length))
                     | ((r >= R + HEAD) & (r < R + HEAD + TAIL)
                        & (new_id < length) & (new_id >= SINK)))
        token = tl.where(old, token, new_id)
        active = tl.where(old, active, new_valid)
        active = active & live
        return token, active

    @triton.jit
    def _final_union_physical_kernel(
        MIDDLE, MIDDLE_COUNTS, LENGTHS, ACTIVE_REQ, VALID_REQ,
        FIRST_DECODE_SEEDED, SEED_IDS, SEED_COUNTS, BLOCK_TABLE,
        GATHER, VIS_LENS,
        W: tl.constexpr, PROT: tl.constexpr, MAX_SEED: tl.constexpr,
        N: tl.constexpr, N_PAD: tl.constexpr, A_PAD: tl.constexpr,
        MERGE_BITS: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        MAX_REQ: tl.constexpr, MAX_VIS: tl.constexpr,
        INVALID: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
        LAYER_OFFSET: tl.constexpr,
    ):
        """middle ∪ protected ∪ seed：一次排序、cumsum 去重并写物理槽。"""
        m = tl.program_id(0)
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)
        length = tl.load(LENGTHS + req).to(tl.int32)
        offs = tl.arange(0, N_PAD)
        in_n = offs < N

        # ranking 的 middle 已经按逻辑 token 升序；只需排序较短的
        # protected+seed，再把第二路反转后与 middle 做 bitonic merge。
        middle_count = tl.load(MIDDLE_COUNTS + m * HKV)
        middle_id = tl.load(MIDDLE + m * (HKV * W) + offs,
                            mask=(offs < W) & (offs < middle_count), other=CAP)

        aux_offs = tl.arange(0, A_PAD)
        prot = aux_offs < PROT
        sink_count = tl.minimum(length, SINK)
        total_count = tl.minimum(length, SINK + SWA)
        in_sink = prot & (aux_offs < sink_count)
        in_swa = (prot & (aux_offs >= sink_count)
                  & (aux_offs < total_count))
        swa_id = length - SWA + aux_offs - sink_count
        protected_id = tl.where(in_sink, aux_offs, swa_id)

        seed_col = aux_offs - PROT
        seed_count = tl.load(
            SEED_COUNTS + (layer + LAYER_OFFSET) * MAX_REQ + req_id)
        seed_needed = ((tl.load(FIRST_DECODE_SEEDED + req_id) == 0)
                       & (tl.load(ACTIVE_REQ + req) >= 0))
        seed = ((seed_col >= 0) & (seed_col < MAX_SEED)
                & (seed_col < seed_count) & seed_needed)
        seed_id = tl.load(
            SEED_IDS + ((layer + LAYER_OFFSET) * MAX_REQ + req_id) * MAX_SEED
            + seed_col,
            mask=seed, other=CAP)
        aux_id = tl.where(prot, protected_id, seed_id)
        aux_valid = in_sink | in_swa | seed
        aux_id = tl.where(aux_valid & (aux_id >= 0) & (aux_id < CAP),
                          aux_id, CAP)
        aux_sorted = tl.sort(aux_id, descending=False)

        if A_PAD > W:
            # The bitonic fast path requires each sorted run to fit in W.
            # Larger protection/seed capacities need the complete union.
            aux_idx = tl.minimum(tl.maximum(offs - W, 0), A_PAD - 1)
            aux_forward = tl.gather(aux_sorted, aux_idx, 0)
            combined = tl.where(offs < W, middle_id,
                                tl.where(offs - W < A_PAD, aux_forward, CAP))
            ordered = tl.sort(combined, descending=False)
        else:
            reverse_idx = 2 * W - 1 - offs
            aux_reverse = tl.gather(
                aux_sorted, tl.minimum(reverse_idx, A_PAD - 1), 0)
            aux_reverse = tl.where(
                (offs >= W) & (reverse_idx < A_PAD), aux_reverse, CAP)
            ordered = tl.where(offs < W, middle_id, aux_reverse)
            for bit in tl.static_range(MERGE_BITS - 1, -1, -1):
                stride = 1 << bit
                peer = tl.gather(ordered, offs ^ stride, axis=0)
                ordered = tl.where((offs & stride) == 0,
                                   tl.minimum(ordered, peer),
                                   tl.maximum(ordered, peer))
        prev_idx = tl.maximum(offs - 1, 0)
        previous = tl.gather(ordered, prev_idx, 0)
        previous = tl.where(offs == 0, -1, previous)
        unique = (ordered < CAP) & (ordered != previous)
        destination = tl.cumsum(unique.to(tl.int32), 0) - 1
        count = tl.sum(unique.to(tl.int32), 0)

        # 无效尾槽与有效 destination 不重叠，可在同一 program 内直接写入。
        tail = in_n & (offs < MAX_VIS) & (offs >= count)
        dst = GATHER + layer * (MAX_REQ * MAX_VIS) + req * MAX_VIS
        tl.store(dst + offs, INVALID, mask=tail)
        emit = unique & (destination >= 0) & (destination < MAX_VIS)
        safe_id = tl.where(emit, ordered, 0)
        page_idx = safe_id // PAGE
        phys = tl.load(
            BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
            + req_id * MAX_BLOCKS + page_idx,
            mask=emit, other=0).to(tl.int32)
        slot = phys * PAGE + safe_id % PAGE
        tl.store(dst + destination, slot, mask=emit)
        tl.store(VIS_LENS + layer * MAX_REQ + req, count)

    @triton.jit
    def _final_physical_map_kernel(
        IDS, COUNTS, BLOCK_TABLE, VALID_REQ, GATHER, VIS_LENS,
        W: tl.constexpr, MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        MAX_REQ: tl.constexpr, MAX_VIS: tl.constexpr,
        INVALID: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
        BLOCK_W: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """Head-0 selection -> absolute physical slot, written in-place.

        ``IDS`` is ``canonical_selection``'s contiguous ``[M, HKV, W]`` int32
        output and ``COUNTS`` is its ``[M, HKV]`` int32 count.  Each program
        handles one (layer, request) row and ``BLOCK_W`` columns, converting a
        logical token id to ``phys_page * PAGE + offset`` through the request
        block table and writing directly into the persistent
        ``gather_indices_buf``/``visible_lens_buf``.  No ``gather``/``where``/
        ``index_copy_`` materialization is needed.
        """
        pid_m = tl.program_id(0)
        pid_w = tl.program_id(1)
        layer = pid_m // B
        req = pid_m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_w = (offs_w < W) & (offs_w < MAX_VIS)
        sel = tl.load(
            IDS + pid_m * (HKV * W) + offs_w, mask=mask_w, other=INVALID
        ).to(tl.int32)

        cnt = tl.load(COUNTS + pid_m * HKV)
        valid = (sel >= 0) & (offs_w < cnt)
        page_idx = tl.where(valid, sel // PAGE, 0)
        page_idx = tl.minimum(tl.maximum(page_idx, 0), MAX_BLOCKS - 1)
        phys = tl.load(
            BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
            + req_id * MAX_BLOCKS + page_idx,
            mask=mask_w & valid, other=0,
        ).to(tl.int32)

        slot = phys * PAGE + (sel % PAGE)
        slot = tl.where(valid, slot, INVALID)
        # 输出由 attention 按 batch 行消费；request id 只用于查页表。
        dst = GATHER + layer * (MAX_REQ * MAX_VIS) + req * MAX_VIS + offs_w
        tl.store(dst, slot, mask=mask_w)

        if pid_w == 0:
            tl.store(VIS_LENS + layer * MAX_REQ + req, cnt)

    @triton.jit
    def _topk_indices_body(
        SCORES, OUT,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K: tl.constexpr, K_PAD: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, KT_PAD)
        mask = offs < KT
        s = tl.load(SCORES + row * KT + offs, mask=mask, other=float("-inf"))
        bits = s.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
        sign = bits & 0x8000
        key = tl.where(sign != 0, (~bits) & 0xFFFF, bits | 0x8000)
        key_signed = key - 0x8000
        idx = KT - 1 - offs
        packed = (key_signed << 16) | (idx & 0xFFFF)
        packed = tl.where(mask, packed, -0x80000000)
        ordered = tl.sort(packed, descending=True)
        top_idx = tl.arange(0, K_PAD)
        top = tl.gather(ordered, top_idx, 0)
        out_idx = (KT - 1 - (top & 0xFFFF)).to(OUT.dtype.element_ty)
        tl.store(OUT + row * K + top_idx, out_idx, mask=top_idx < K)

    @triton.jit
    def _topk_indices_pair_kernel(
        FAR_SCORES, NEAR_SCORES, FAR_OUT, NEAR_OUT,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K_F: tl.constexpr, K_N: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
    ):
        """同一次 launch 完成 far/near 两路独立 Top-K。"""
        if tl.program_id(1) == 0:
            _topk_indices_body(
                FAR_SCORES, FAR_OUT, KT=KT, KT_PAD=KT_PAD,
                K=K_F, K_PAD=K_F_PAD)
        else:
            _topk_indices_body(
                NEAR_SCORES, NEAR_OUT, KT=KT, KT_PAD=KT_PAD,
                K=K_N, K_PAD=K_N_PAD)

    @triton.jit
    def _topk_candidate_batched_kernel(
        FAR_SCORES, NEAR_SCORES, I_F, I_N, LENGTHS,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS, TILE_COUNTERS,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K_F: tl.constexpr, K_N: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, PAGE: tl.constexpr,
        NEAR_LEN: tl.constexpr, K2: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """eager 单 CTA 顺序完成两路 Top-K 与 candidate expansion。"""
        _topk_indices_body(
            FAR_SCORES, I_F, KT=KT, KT_PAD=KT_PAD, K=K_F, K_PAD=K_F_PAD)
        _topk_indices_body(
            NEAR_SCORES, I_N, KT=KT, KT_PAD=KT_PAD, K=K_N, K_PAD=K_N_PAD)
        tl.debug_barrier()
        _candidate_pack_batched_kernel(
            I_F, I_N, FAR_SCORES, NEAR_SCORES, LENGTHS,
            SKIP_FAR, SKIP_NEAR, VALID_REQ,
            OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS, TILE_COUNTERS,
            K_F=K_F, K_N=K_N, KT=KT, R=R, HEAD=HEAD, TAIL=TAIL,
            SINK=SINK, SWA=SWA, CAP=CAP, SINK_POOL_LO=SINK_POOL_LO,
            BS=BS, PAGE=PAGE, NEAR_LEN=NEAR_LEN, K2=K2, HKV=HKV, B=B,
            MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)

# ---------------------------------------------------------------------------
# Host-side batched helpers
# ---------------------------------------------------------------------------

def _scalar(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(int(x), device=device, dtype=dtype)


def _workspace_empty(workspace, name, shape, dtype, device):
    """复用同 stream 的层间临时 buffer，消除逐层 allocator host 开销。"""
    if workspace is None:
        return torch.empty(shape, dtype=dtype, device=device)
    key = (name, tuple(shape), dtype, device)
    out = workspace.get(key)
    if out is None:
        out = workspace[key] = torch.empty(shape, dtype=dtype, device=device)
    return out


def _workspace_zeros(workspace, name, shape, dtype, device):
    if workspace is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    key = (name, tuple(shape), dtype, device)
    out = workspace.get(key)
    if out is None:
        out = workspace[key] = torch.zeros(shape, dtype=dtype, device=device)
    return out


def _topk_indices_pair(far_scores, near_scores, k_f, k_n, workspace=None, launch=True):
    """一次 launch 生成 far/near 两路 int64 Top-K 索引。"""
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    m, hkv, kt_cap = far_scores.shape
    k_f = min(int(k_f), kt_cap)
    k_n = min(int(k_n), kt_cap)
    out_f = _workspace_empty(
        workspace, 'topk_far', (m, hkv, k_f), torch.int64, far_scores.device)
    out_n = _workspace_empty(
        workspace, 'topk_near', (m, hkv, k_n), torch.int64, far_scores.device)
    if k_f + k_n == 0:
        return out_f, out_n
    if not launch:
        return out_f, out_n
    if kt_cap == 1:
        out_f.zero_()
        out_n.zero_()
        return out_f, out_n
    _fast_launch(workspace, 'topk_pair', _topk_indices_pair_kernel,
                 (m * hkv, 2),
        far_scores.reshape(m * hkv, kt_cap),
        near_scores.reshape(m * hkv, kt_cap), out_f, out_n,
        KT=kt_cap, KT_PAD=triton.next_power_of_2(kt_cap),
        K_F=k_f, K_N=k_n,
        K_F_PAD=triton.next_power_of_2(max(1, k_f)),
        K_N_PAD=triton.next_power_of_2(max(1, k_n)),
        num_warps=8,
    )
    return out_f, out_n


def _page_stats_all_layers(cache, req_ids, layer_ptrs, dims, bs, pad, lengths,
                           layer_offset=0, workspace=None, q_job=None,
                           write_job=None):
    """持久化 request-slot 页统计；连续 decode 只更新最新逻辑块。"""
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    layer0 = cache.layers[0]
    device = layer0.kv_pool_k.device
    hk = layer0.num_kv_heads
    head_dim = layer0.head_dim_k
    b = lengths.numel() if torch.is_tensor(lengths) else len(lengths)
    l = len(layer_ptrs)
    m = l * b
    kt_cap = pad // bs
    dim_numel = dims.numel()

    valid_req_t = (req_ids if torch.is_tensor(req_ids) else
                   torch.as_tensor(req_ids, dtype=torch.int64, device=device))
    active_req_t = cache.active_req_ids_buf[:b]
    block_table = cache.block_table_buf
    lengths_t = (lengths if torch.is_tensor(lengths) else
                 torch.as_tensor(lengths, dtype=torch.int32, device=device))
    dims_t = dims
    kv_ptrs = cache.layer_kv_ptrs_buf[layer_offset:layer_offset + l]

    out_dtype = layer0.kv_pool_k.dtype
    stats_len = getattr(cache, '_nfa_page_stats_lengths', None)
    if stats_len is None:
        stats_len = cache._nfa_page_stats_lengths = torch.zeros(
            (cache.n_layers, cache.max_requests), dtype=torch.int32,
            device=device)
    persistent = getattr(cache, '_nfa_page_stats_persistent', None)
    if persistent is None:
        persistent = cache._nfa_page_stats_persistent = {}
    key = (layer_offset, l, dim_numel, out_dtype)
    buffers = persistent.get(key)
    if buffers is None:
        p_shape = (l, cache.max_requests, kt_cap, hk, dim_numel)
        buffers = tuple(torch.empty(
            p_shape, dtype=out_dtype, device=device) for _ in range(3))
        persistent[key] = buffers
    p_min, p_max, p_sum = buffers

    blocks_per_program = 16
    stats_rows = m * hk
    stats_programs = triton.cdiv(kt_cap, blocks_per_program) * stats_rows
    if q_job is None:
        _fast_launch(workspace, ('page_stats', layer_offset), _nfa_page_stats_kernel,
                     (triton.cdiv(kt_cap, blocks_per_program), stats_rows),
            kv_ptrs, block_table, lengths_t, active_req_t, valid_req_t, dims_t,
            stats_len, p_min, p_max, p_sum,
            MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
            HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dim_numel,
            D_BLOCK=triton.next_power_of_2(dim_numel), BS=bs,
            KT_CAP=kt_cap, L=l, B=b, BLOCKS_PER_PROGRAM=blocks_per_program,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
            num_warps=4,
        )
    else:
        q_src, q_req_ids, scales, q_out, s0, s1, s2, hq, d_full = q_job
        layer = cache.layers[layer_offset]
        if write_job is None:
            # constexpr 分支会被裁掉；dummy 指针不会解引用。
            keys = values = k_pool = v_pool = q_src
            slots = lengths_t
        else:
            keys, values, slots = write_job
            k_pool, v_pool = layer.kv_pool_k, layer.kv_pool_v
        _fast_launch(workspace,
                     ('page_stats_prepare_q', layer_offset, write_job is not None),
                     _nfa_page_stats_prepare_q_kernel,
                     (stats_programs + m * hq,),
            kv_ptrs, block_table, lengths_t, active_req_t, valid_req_t, dims_t,
            stats_len, p_min, p_max, p_sum,
            q_src, q_req_ids, scales, q_out, s0, s1, s2,
            keys, values, slots, k_pool, v_pool,
            MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
            HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dim_numel,
            D_BLOCK=triton.next_power_of_2(dim_numel), BS=bs,
            KT_CAP=kt_cap, L=l, B=b, BLOCKS_PER_PROGRAM=blocks_per_program,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
            HQ=hq, D=d_full, DQ=dim_numel,
            DQ_BLOCK=triton.next_power_of_2(dim_numel),
            STATS_PROGRAMS=stats_programs, STATS_ROWS=stats_rows,
            DV=layer.head_dim_v, BD=triton.next_power_of_2(head_dim),
            BV=triton.next_power_of_2(layer.head_dim_v),
            HAS_WRITE=write_job is not None,
            num_warps=4,
        )
    return p_min, p_max, p_sum


def _fused_fine_scores_paged(
        cache, req_ids, layer_ptrs, dims, q_fine, cand_idx2, keep2, owners,
        generations, tile_counters, capacity, group_size, tile=128, num_warps=4,
        fuse_state=True, fuse_rank=False, rank_out=None, lower=None, upper=None,
        quota=None, layer_offset=0, workspace=None):
    """Candidate gather -> paged K load -> BF16 score 与 online-softmax 状态。

    Reads candidate tokens straight from ``kv_pool_k`` through the request
    block table; the old ``projected[:, :s_cap]`` fine-K materialization is
    gone from this path.
    """
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    layer0 = cache.layers[0]
    device = layer0.kv_pool_k.device
    hk = layer0.num_kv_heads
    head_dim = layer0.head_dim_k
    b = req_ids.numel() if torch.is_tensor(req_ids) else len(req_ids)
    m, hkv, r = cand_idx2.shape
    if hkv not in (1, hk):
        raise FallbackRequired("KV-head geometry mismatch")

    valid_req_t = (req_ids if torch.is_tensor(req_ids) else
                   torch.as_tensor(req_ids, dtype=torch.int64, device=device))
    block_table = cache.block_table_buf
    dims_t = dims
    kv_ptrs = cache.layer_kv_ptrs_buf[
        layer_offset:layer_offset + len(layer_ptrs)]

    unique = _workspace_empty(
        workspace, 'fine_unique', keep2.shape, keep2.dtype, device)
    scores = _workspace_empty(
        workspace, 'fine_scores', (m, hkv, group_size, r), torch.bfloat16,
        device)
    mstat = _workspace_empty(
        workspace, 'fine_mstat', (m, hkv, group_size), torch.float32, device)
    lstat = _workspace_empty(
        workspace, 'fine_lstat', mstat.shape, torch.float32, device)

    grid_prob = (m * hkv, triton.cdiv(r, tile))
    num_tiles = triton.cdiv(r, tile)
    partial_m = _workspace_empty(
        workspace, 'fine_partial_m', (m, hkv, group_size, num_tiles),
        torch.float32, device)
    partial_l = _workspace_empty(
        workspace, 'fine_partial_l', partial_m.shape, torch.float32, device)

    rank_groups = rank_out.shape[1]
    rank_k = rank_out.shape[2]
    token_bits = 32 if capacity >= 65536 else 16
    _fast_launch(workspace, ('fine_score', layer_offset, fuse_state, fuse_rank),
                 _nfa_fine_scores_paged_kernel, grid_prob,
        kv_ptrs, block_table, valid_req_t, dims_t, q_fine, cand_idx2,
        keep2, owners, generations, unique, scores, partial_m, partial_l,
        mstat, lstat, tile_counters, rank_out,
        lower, upper, quota,
        layer0.kv_pool_k,
        R=r, NUM_TILES=num_tiles,
        stride_qm=q_fine.stride(0), stride_qh=q_fine.stride(1),
        stride_ch=cand_idx2.stride(1), stride_cr=cand_idx2.stride(2),
        S=capacity, MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dims.numel(),
        D_BLOCK=triton.next_power_of_2(dims.numel()),
        G_PAD=triton.next_power_of_2(max(16, group_size)),
        HKV=hkv, G=group_size, B=b, TILE=tile,
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        FUSE_STATE=fuse_state, FUSE_RANK=fuse_rank,
        RANK_GROUPS=rank_groups, RANK_K=rank_k,
        META_GROUP=hkv,
        TOKEN_BITS=token_bits,
        PROB_BF16=layer0.kv_pool_k.dtype == torch.bfloat16,
        num_warps=num_warps,
    )
    if not fuse_state:
        _fast_launch(workspace, 'fine_state', _nfa_fine_softmax_state_kernel,
                     (m * hkv * group_size,),
            partial_m, partial_l, mstat, lstat, NUM_TILES=num_tiles,
            G=group_size, num_warps=1)
    return scores, mstat, lstat, unique


def _l1_scores(k_min, k_max, k_sum, b_q_full, group_size, indexer, dtype,
               cache, valid_req, lengths, page, bs, sink, near_len, window,
               layer_offset=0, workspace=None):
    """直接读取持久页统计并生成 far/near Top-K 输入。"""
    _, _, kt_cap, hkv, d = k_min.shape
    m = b_q_full.shape[0]
    hq = b_q_full.shape[1]
    q_flat = b_q_full.reshape(m * hq, d).contiguous()
    far_scored = _workspace_empty(
        workspace, 'l1_far', (m, 1, kt_cap), dtype, k_min.device)
    near_scored = _workspace_empty(
        workspace, 'l1_near', (m, 1, kt_cap), dtype, k_min.device)
    bt, nw = indexer._TILE_COARSE.get(d, (64, 4))
    lengths_t = torch.as_tensor(lengths, dtype=torch.int32, device=k_min.device)
    b = lengths.numel() if torch.is_tensor(lengths) else len(lengths)
    far_avg = indexer.far_method == "avg"
    near_avg = indexer.near_method != "minmax"
    has_avg = far_avg or near_avg
    k_sum_arg = k_sum if has_avg else k_min
    _fast_launch(workspace, ('l1', layer_offset), _nfa_l1_scores_kernel,
                 (triton.cdiv(kt_cap, bt), m),
        k_min, k_max, k_sum_arg, q_flat, lengths_t,
        cache.active_req_ids_buf[:b], cache.skip_far_layer_buf,
        cache.skip_near_layer_buf, valid_req, cache._nfa_page_stats_lengths,
        far_scored, near_scored, kt_cap,
        k_min.stride(2), k_min.stride(3),
        k_max.stride(2), k_max.stride(3),
        k_sum_arg.stride(2), k_sum_arg.stride(3), q_flat.stride(0),
        PAGE=page, BS=bs, SINK=sink, NEAR_LEN=near_len, WINDOW=window,
        POOL_NEAR_LEN=max(bs, int(indexer.nt_near_len)),
        SINK_POOL_LO=int(indexer._sink_pool_lo_blk),
        Hkv=1, G=group_size, D=d, BT=bt, B=b,
        STRIDE_KB=k_min.stride(0), STRIDE_KB2=k_max.stride(0),
        STRIDE_KS=k_sum_arg.stride(0), STRIDE_KR=k_min.stride(1),
        STRIDE_KR2=k_max.stride(1), STRIDE_KRS=k_sum_arg.stride(1),
        HAS_AVG=has_avg,
        FAR_AVG=far_avg, NEAR_AVG=near_avg,
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        num_warps=nw,
    )
    return far_scored, near_scored


def _candidate_pack_batched(i_f, i_n, far_scored, near_scored, lengths,
                            cache, valid_req, indexer, bs, capacity,
                            layer_offset=0, workspace=None, fuse_topk=False):
    """直接生成 complete-pack 候选及 rank 元数据，无 aten 中间张量。"""
    m, hkv, k_f = i_f.shape
    k_n = i_n.shape[-1]
    sink = int(indexer.sink_tokens)
    swa = int(indexer.sliding_window_size)
    head = min(sink, capacity)
    tail = min(swa, capacity)
    r = (k_f + k_n + 2) * bs
    width = r + head + tail
    device = i_f.device
    out = _workspace_empty(
        workspace, 'candidate_ids', (m, hkv, width), torch.int64, device)
    valid = _workspace_empty(
        workspace, 'candidate_valid', (m, hkv, width), torch.bool, device)
    lower = _workspace_empty(
        workspace, 'candidate_lower', (m,), torch.int64, device)
    upper = _workspace_empty(
        workspace, 'candidate_upper', (m,), torch.int64, device)
    quota = _workspace_empty(
        workspace, 'candidate_quota', (m,), torch.int64, device)
    owners = _workspace_zeros(
        workspace, 'fine_owners', (m, hkv, capacity), torch.int64, device)
    generations = _workspace_zeros(
        workspace, 'fine_owner_generations', (m, hkv), torch.int64, device)
    tile_counters = _workspace_empty(
        workspace, 'fine_tile_counters', (m, hkv), torch.int32, device)
    b = lengths.numel() if torch.is_tensor(lengths) else len(lengths)
    lengths_t = (lengths if torch.is_tensor(lengths) else
                 torch.as_tensor(lengths, dtype=torch.int64, device=device))
    if fuse_topk:
        _fast_launch(workspace, ('topk_candidate', layer_offset),
                     _topk_candidate_batched_kernel, (m * hkv,),
            far_scored, near_scored, i_f, i_n, lengths_t,
            cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
            out, valid, lower, upper, quota, owners, generations, tile_counters,
            KT=far_scored.shape[-1],
            KT_PAD=triton.next_power_of_2(far_scored.shape[-1]),
            K_F=k_f, K_N=k_n,
            K_F_PAD=triton.next_power_of_2(max(1, k_f)),
            K_N_PAD=triton.next_power_of_2(max(1, k_n)),
            R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=int(indexer._sink_pool_lo_blk), BS=bs,
            PAGE=cache.geometry.page, NEAR_LEN=cache.geometry.near,
            K2=int(indexer.args.tia_level2_topk),
            HKV=hkv, B=b,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
            num_warps=8)
    else:
        _fast_launch(workspace, ('candidate', layer_offset),
                     _candidate_pack_batched_kernel, (m * hkv,),
            i_f, i_n, far_scored, near_scored, lengths_t,
            cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
            out, valid, lower, upper, quota, owners, generations, tile_counters,
            K_F=k_f, K_N=k_n, KT=far_scored.shape[-1],
            R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=int(indexer._sink_pool_lo_blk), BS=bs,
            PAGE=cache.geometry.page, NEAR_LEN=cache.geometry.near,
            K2=int(indexer.args.tia_level2_topk),
            HKV=hkv, B=b,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
            num_warps=4)
    return out, valid, lower, upper, quota, owners, generations, tile_counters


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def _replay_eager_selector_call_plan(workspace, call_plan, q_src, decode_write):
    """按固定几何原序重发 selector；仅当前 Q/K/V 与 slot 来自本步。"""
    keys = values = slots = None
    if decode_write is not None:
        keys, values, slots = decode_write
        slots = slots.contiguous()
    stream, enter_hook, exit_hook, _ = workspace['_nfa_launch_context']
    for (key, kernel, grid, args, kwargs, run, function, packed_metadata,
         compiled, launch_values, grid_1, grid_2) in call_plan[1]:
        if kernel is _nfa_page_stats_prepare_q_kernel:
            args = args[:10] + (q_src,) + args[11:]
            if run is not None:
                launch_values = (launch_values[:10] + (q_src,)
                                 + launch_values[11:])
            if decode_write is not None:
                args = args[:17] + (keys, values, slots) + args[20:]
                if run is not None:
                    launch_values = (launch_values[:17]
                                     + (keys, values, slots)
                                     + launch_values[20:])
        if run is not None and enter_hook is None:
            run(grid[0], grid_1, grid_2, stream, function, packed_metadata,
                None, None, exit_hook, *launch_values)
        else:
            _fast_launch(workspace, key, kernel, grid, *args, **kwargs)


def select_topk_for_layer_batched(cache, layer_idx, req_ids,
                                  query_states, attention=None, decode_write=None):
    """单层 Req-Batched 当前 query 选择（生产唯一入口）。

    ``query_states`` 必须是本层当前 token 的 post-RoPE Q，形状 ``[B,HQ,D]``。
    输出直接写入当前 visible bucket 的 batch 行，支持非连续 request slot。
    """
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    if query_states is None:
        raise ValueError('production selection requires current post-RoPE query')

    # 生产 Cache 的配置在 decode 期间固定。稳态可在公共入口直接重放本层
    # 计划，避免再次组织页表、子空间、scale 与 ranking 的 host 参数；动态
    # batch/bucket/Q 几何仍逐项守卫，capture 与任何失配均回退完整路径。
    workspace = getattr(cache, '_nfa_selector_workspace', None)
    if workspace is not None:
        call_plan = workspace.get(('eager_selector_call_plan', layer_idx))
        geometry = workspace.get('_nfa_launch_geometry')
        if (call_plan is not None and call_plan[0] is geometry
                and not torch.cuda.is_current_stream_capturing()
                and len(req_ids) == geometry[0]
                and cache.active_visible_bucket == geometry[12]
                and query_states.shape ==
                    (geometry[0], geometry[2], cache.layers[0].head_dim_k)):
            q_src = query_states.unsqueeze(0)
            q_geometry = (q_src.dtype, q_src.data_ptr() % 16,
                          q_src.stride(0), q_src.stride(1), q_src.stride(2))
            if q_geometry == geometry[27:]:
                _replay_eager_selector_call_plan(
                    workspace, call_plan, q_src, decode_write)
                return

    return _layer_batched_select_impl(
        cache, req_ids, layer_indices=[layer_idx],
        query_states=query_states, attention=attention, decode_write=decode_write)


def _layer_batched_select_impl(cache, req_ids, layer_indices,
                               query_states, attention, decode_write=None):
    """在当前层、当前 post-RoPE Q 上执行 Req-Batched 选择。"""
    if cache.layers[0].kv_pool_k.device.type != "cuda":
        raise FallbackRequired("CUDA required")

    b = len(req_ids)
    if b == 0:
        return

    if layer_indices is None:
        layer_indices = list(range(cache.n_layers))
    if not layer_indices:
        return
    n_layers = len(layer_indices)
    layer_offset = layer_indices[0]

    device = cache.layers[0].kv_pool_k.device
    workspace = getattr(cache, '_nfa_selector_workspace', None)
    if workspace is None:
        workspace = cache._nfa_selector_workspace = {}
    indexer = attention.indexer
    if not hasattr(indexer, "args") or indexer.args is None:
        raise FallbackRequired("missing indexer config")
    # 生产路径由每层 0 的 attention 入口按 model forward 刷新上下文；
    # 独立 selector 调用没有该入口时保留本地回退。
    capturing = torch.cuda.is_current_stream_capturing()
    workspace['_nfa_capturing'] = capturing
    if '_nfa_launch_context' not in workspace or capturing:
        from triton import knobs
        workspace['_nfa_launch_context'] = (
            torch.cuda.current_stream().cuda_stream,
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
            knobs.runtime.debug,
        )

    if b > cache.planned_lengths_buf.numel():
        raise ValueError('selector batch exceeds fixed query metadata capacity')
    lengths = cache.planned_lengths_buf[:b]
    m = n_layers * b
    hq = cache.num_query_heads
    hkv = cache.layers[0].num_kv_heads
    group_size = hq // hkv
    bs = int(indexer.args.tia_block_size)
    page = cache.geometry.page
    # Graph 形状只取决于固定容量，真实长度全部从设备 buffer 读取。
    # 逻辑序列容量来自页表；max_visible_len 只限制选中输出宽度，二者不能混用。
    pad = cache.max_blocks * bs
    kt_cap = cache.max_blocks
    s_cap = pad

    dims = indexer._subspace_indices(device).long()
    if indexer.enable_subspace:
        q_dims = dims
    else:
        q_dims = None

    # PAD 行使用 request 0 的安全页表地址，但 length=0 且不消费 seed。
    active_req_t = cache.active_req_ids_buf[:b]
    valid_req_t = cache.valid_req_ids_buf[:b]
    scale_bufs = getattr(cache, '_attention_scale_bufs', None)
    if scale_bufs is None:
        scale_bufs = cache._attention_scale_bufs = {}
    scales = scale_bufs.get(layer_offset)
    if scales is None:
        # warmup/eager 时创建一次；capture/replay 仅复用固定地址。
        scales = torch.tensor(
            [attention.scaling], dtype=torch.float32, device=device)
        scale_bufs[layer_offset] = scales
    if n_layers != 1 or query_states.shape != (b, hq, cache.layers[0].head_dim_k):
        raise ValueError(
            f'current query must be [{b},{hq},{cache.layers[0].head_dim_k}] '
            f'for one layer, got {tuple(query_states.shape)}')
    q_src = query_states.unsqueeze(0)
    q_req_ids = cache.query_req_ids_buf[:b]
    d_full = q_src.shape[-1]
    s0, s1, s2 = q_src.stride(0), q_src.stride(1), q_src.stride(2)
    # 同一 Cache 的生产几何固定；用一个严格令牌替代每次 launch 的逐张量键。
    # 独立 ranking 调用没有该令牌，仍使用完整逐参数 specialization 键。
    geometry = (
        b, n_layers, hq, hkv, group_size, bs, page, pad, kt_cap, s_cap,
        capturing,
        cache.max_requests, cache.active_visible_bucket,
        cache.prefill_seed_ids_buf.shape[-1], dims.numel(),
        indexer.enable_subspace, indexer.l2_subspace, indexer.cmp_ratio,
        indexer.far_method, indexer.near_method, indexer.nt_near_pages,
        indexer.nt_near_len, indexer.args.tia_level1_topk,
        indexer.args.tia_level2_topk, indexer.sink_tokens,
        indexer.sliding_window_size, indexer._sink_pool_lo_blk,
        q_src.dtype, q_src.data_ptr() % 16, s0, s1, s2,
    )
    previous_geometry = workspace.get('_nfa_launch_geometry')
    if previous_geometry == geometry:
        geometry = previous_geometry
    workspace['_nfa_launch_geometry'] = geometry

    plan_key = ('eager_selector_call_plan', layer_offset)
    call_plan = workspace.get(plan_key)
    if not capturing and call_plan is not None and call_plan[0] is geometry:
        _replay_eager_selector_call_plan(
            workspace, call_plan, q_src, decode_write)
        return
    if not capturing:
        workspace['_nfa_eager_record'] = []

    def _prepare_q(dim_idx, d_q, name, launch=True):
        out = _workspace_empty(
            workspace, name, (m, hq, d_q), torch.float32, device)
        if launch:
            _fast_launch(workspace, ('prepare_q', layer_offset, name),
                         _prepare_q_kernel, (m * hq,),
                q_src, q_req_ids, scales, dim_idx, out,
                s0, s1, s2,
                HQ=hq, D=d_full, DQ=d_q, B=b,
                DQ_BLOCK=triton.next_power_of_2(d_q), num_warps=4)
        return out

    fuse_prepare_q = q_dims is not None and indexer.l2_subspace != "full"
    if q_dims is not None:
        b_q_full = _prepare_q(
            dims, dims.numel(), 'q_full', launch=not fuse_prepare_q)
    else:
        full_dims = torch.arange(d_full, dtype=torch.int64, device=device)
        b_q_full = _prepare_q(full_dims, d_full, 'q_full')

    if indexer.l2_subspace == "full":
        full_dims = torch.arange(d_full, dtype=torch.int64, device=device)
        b_q_fine = _prepare_q(full_dims, d_full, 'q_fine')
    elif q_dims is not None:
        b_q_fine = b_q_full
    else:
        qat = indexer._qat_indices(device).long()
        b_q_fine = _prepare_q(qat, qat.numel(), 'q_fine')
    q_job = ((q_src, q_req_ids, scales, b_q_full,
              s0, s1, s2, hq, d_full) if fuse_prepare_q else None)

    write_job = None
    if decode_write is not None:
        keys, values, slots = decode_write
        if not fuse_prepare_q:
            cache.write_decode(layer_offset, keys, values, slots)
        else:
            if keys.ndim != 4 or keys.shape[-2] != 1:
                raise ValueError('graph-ready decode KV must be one token per row')
            if values.ndim != 4 or values.shape[-2] != 1:
                raise ValueError('paged write decode expects [B,HK,1,DV] values')
            if keys.shape[0] != b or keys.shape[0] != slots.shape[0]:
                raise ValueError('request batch must match KV batch')
            if keys.shape[1] != hkv:
                raise ValueError('KV head count does not match arena')
            if slots.device != device:
                slots = slots.to(device)
            write_job = (keys, values, slots.contiguous())

    # Layer pointers and persistent tensor-only skip flags [layer, request].
    # Skip/block-table lookups now read the full persistent buffers directly
    # via ``valid_req_t``, so no ``.contiguous()`` column copy is materialized.
    layer_ptrs = [cache.layers[li].kv_pool_k.data_ptr() for li in layer_indices]

    # ---- one launch: in-place paged block stats, no projection -----------
    k_min, k_max, k_sum = _page_stats_all_layers(
        cache, valid_req_t, layer_ptrs, dims, bs, pad,
        lengths, layer_offset=layer_offset, workspace=workspace,
        q_job=q_job, write_job=write_job)  # [L,max_req,kt_cap,Hkv,d]

    # ---- fused L1 scores + region/candidate-pool masks -------------------
    far_scored, near_scored = _l1_scores(
        k_min, k_max, k_sum, b_q_full, group_size, indexer,
        q_src.dtype, cache=cache, valid_req=valid_req_t,
        lengths=lengths, page=page, bs=bs, sink=cache.geometry.sink,
        near_len=cache.geometry.near, window=cache.geometry.window,
        layer_offset=layer_offset, workspace=workspace)

    nb_near = max(0, int(indexer.nt_near_pages))
    k1 = int(indexer.args.tia_level1_topk)
    nb_far = max(0, k1 - nb_near)

    i_f, i_n = _topk_indices_pair(
        far_scored, near_scored, min(nb_far, kt_cap), min(nb_near, kt_cap),
        workspace=workspace, launch=capturing)

    # ---- candidate expansion + completion + rank metadata ---------------
    cand_idx2, keep2, lower, upper, quota, owners, generations, tile_counters = _candidate_pack_batched(
        i_f, i_n, far_scored, near_scored, lengths, cache, valid_req_t,
        indexer, bs, s_cap, layer_offset=layer_offset, workspace=workspace,
        fuse_topk=not capturing)

    # final canonical union 只消费 request-level KV head 0；其余头的
    # fine/state/ranking 结果没有 downstream 消费者，无需物化。
    cand_idx2 = cand_idx2[:, :1]
    keep2 = keep2[:, :1]
    owners = owners[:, :1]
    generations = generations[:, :1]
    tile_counters = tile_counters[:, :1]

    # ---- fused fine score + local unique topk ----------------------------
    width = 1024
    rank_groups = triton.cdiv(max(cand_idx2.shape[-1], width), 2 * width)
    rank_dtype = torch.int64 if s_cap >= 65536 else torch.int32
    rank_tile = _workspace_empty(
        workspace, 'rank_tile', (m, rank_groups, width), rank_dtype, device)
    scores, mstat, lstat, unique = _fused_fine_scores_paged(
        cache, valid_req_t, layer_ptrs, dims, b_q_fine,
        cand_idx2, keep2, owners, generations, tile_counters, s_cap, group_size,
        tile=256 if capturing else 128, fuse_state=not capturing,
        fuse_rank=not capturing, rank_out=rank_tile,
        lower=lower, upper=upper, quota=quota,
        layer_offset=layer_offset, workspace=workspace)

    fixed_candidate_buffers = cand_idx2.shape[-1] >= width
    if cand_idx2.shape[-1] < width:
        # 短 prompt 的 L1 候选数可小于固定 ranking 宽度；补无效项不改变
        # Top-K 语义，也避免 max_new_tokens=1/首 token 等短序列被误拒绝。
        pad_width = width - cand_idx2.shape[-1]
        cand_idx2 = torch.cat([
            cand_idx2,
            torch.zeros((*cand_idx2.shape[:-1], pad_width),
                        dtype=cand_idx2.dtype, device=device)], dim=-1)
        scores = torch.cat([
            scores,
            torch.full((*scores.shape[:-1], pad_width), float('-inf'),
                       dtype=scores.dtype, device=device)], dim=-1)
        unique = torch.cat([
            unique,
            torch.zeros((*unique.shape[:-1], pad_width),
                        dtype=unique.dtype, device=device)], dim=-1)
    middle = rank_unique_batched(
        cand_idx2, None, unique, lower, upper, quota,
        capacity=s_cap, width=width, sink=int(indexer.sink_tokens),
        validate=False, prevalidated=True, fuse_tree=not capturing,
        fuse_tiles=not capturing, workspace=workspace,
        score_state=(scores, mstat, lstat, group_size,
                     cache.layers[0].kv_pool_k.dtype))

    # ---- final canonical union + physical mapping ------------------------
    # 只消费 request-level head 0；保护集与 seed 在 kernel 内按原语义生成。
    gather_out = cache.gather_indices_for_layer(layer_offset).unsqueeze(0)
    vis_out = cache.visible_lens_for_layer(layer_offset).unsqueeze(0)
    output_width = gather_out.shape[-1]
    prot_width = min(
        s_cap, int(indexer.sink_tokens + indexer.sliding_window_size))
    max_seed = cache.prefill_seed_ids_buf.shape[-1]
    union_width = middle.indices.shape[-1] + prot_width + max_seed
    _fast_launch(workspace, ('final_union', layer_offset),
                 _final_union_physical_kernel, (m,),
        middle.indices, middle.counts, lengths, active_req_t, valid_req_t,
        cache.first_decode_seeded_buf, cache.prefill_seed_ids_buf,
        cache.prefill_seed_counts_buf, cache.block_table_buf,
        gather_out, vis_out,
        W=middle.indices.shape[-1], PROT=prot_width, MAX_SEED=max_seed,
        N=union_width, N_PAD=triton.next_power_of_2(union_width),
        A_PAD=triton.next_power_of_2(prot_width + max_seed),
        MERGE_BITS=(triton.next_power_of_2(union_width)).bit_length() - 1,
        SINK=int(indexer.sink_tokens), SWA=int(indexer.sliding_window_size),
        CAP=s_cap, MAX_BLOCKS=cache.max_blocks, PAGE=page,
        MAX_REQ=cache.max_requests, MAX_VIS=output_width,
        INVALID=INVALID_GATHER_IDX, HKV=1, B=b,
        LAYER_OFFSET=layer_offset, num_warps=8,
    )

    record = workspace.pop('_nfa_eager_record', None)
    if record is not None and fixed_candidate_buffers:
        prepared = []
        for key, kernel, grid, args, kwargs in record:
            launch = workspace.get(('triton_plan', key))
            if launch is not None and launch[0] is geometry:
                (_, _, run, function, packed_metadata, compiled, tail,
                 grid_1, grid_2) = launch
                launch_values = args + tail
            else:
                run = function = packed_metadata = compiled = None
                launch_values = ()
                grid_1 = grid_2 = 1
            prepared.append((key, kernel, grid, args, kwargs, run, function,
                             packed_metadata, compiled, launch_values,
                             grid_1, grid_2))
        workspace[plan_key] = (geometry, prepared)


def mark_first_decode_seeded(cache, req_ids):
    """Consume prefill seeds once, after all layers completed selection."""
    valid_req_ids = [
        rid for rid in req_ids
        if rid is not None and rid >= 0 and rid in cache.states
        and not cache.states[rid].first_decode_seeded
    ]
    if not valid_req_ids:
        return
    device = cache.first_decode_seeded_buf.device
    valid_req_t = torch.as_tensor(valid_req_ids, dtype=torch.int64, device=device)
    cache.first_decode_seeded_buf[valid_req_t] = 1
    for rid in valid_req_ids:
        cache.states[rid].first_decode_seeded = True

