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
from .selection.selector import TokenSelection
from .selection.unique_ranking import _fast_launch

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
        BLOCKS_PER_PROGRAM: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
        BLOCK_GROUP_STRIDE: tl.constexpr,
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
        owner = (block_group
                 == (last_blk // BLOCKS_PER_PROGRAM) % BLOCK_GROUP_STRIDE)

        if active & (~incremental | owner):
            # 稳态 owner 只处理最新块；重建时固定少量 CTA 按 stride 覆盖
            # 全部块组，grid 不再随上下文容量增长。
            group_end = tl.where(incremental, block_group + 1, BLOCK_GROUPS)
            for group in range(block_group, group_end, BLOCK_GROUP_STRIDE):
                group_start = group * BLOCKS_PER_PROGRAM
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
        BLOCKS_PER_PROGRAM: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
        BLOCK_GROUP_STRIDE: tl.constexpr,
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
            BLOCK_GROUPS=BLOCK_GROUPS, BLOCK_GROUP_STRIDE=BLOCK_GROUP_STRIDE,
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
        # 每种统计只服务实际消费它的候选池；skip、bypass 和保护区
        # 在 load 前排除，避免先读取 Min/Max/Sum 再把分数置为 -inf。
        mm_pool = mask_t & ((far_pool & (not FAR_AVG))
                            | (near_pool & (not NEAR_AVG)))
        if not FAR_AVG or not NEAR_AVG:
            kmin = tl.load(
                KMIN + offs_t[:, None] * stride_kt + kv * stride_kh
                + offs_d[None, :],
                mask=mm_pool[:, None], other=0.0).to(tl.float32)
            kmax = tl.load(
                KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2
                + offs_d[None, :],
                mask=mm_pool[:, None], other=0.0).to(tl.float32)
        if HAS_AVG:
            avg_pool = mask_t & ((far_pool & FAR_AVG)
                                 | (near_pool & NEAR_AVG))
            ksum = tl.load(
                KSUM + offs_t[:, None] * stride_kts + kv * stride_khs
                + offs_d[None, :],
                mask=avg_pool[:, None], other=0.0)
            # 保持旧 OUT_AVG 的 BF16 中间舍入，再参与 FP32 点积。
            kavg = (ksum.to(tl.float32) / BS).to(
                FAR_OUT.dtype.element_ty).to(tl.float32)

        if not FAR_AVG or not NEAR_AVG:
            mm_acc = tl.zeros((BT,), dtype=tl.float32)
        if HAS_AVG:
            avg_acc = tl.zeros((BT,), dtype=tl.float32)
        for j in tl.static_range(G):
            h = pid_h.to(tl.int64) * G + j
            q = tl.load(Q + h * stride_qh + offs_d)
            if not FAR_AVG or not NEAR_AVG:
                q_low = tl.minimum(q, 0.0)
                q_high = tl.maximum(q, 0.0)
                mm = (tl.sum(kmin * q_low[None, :], axis=1)
                      + tl.sum(kmax * q_high[None, :], axis=1))
                # 旧路径先逐 query head 转 BF16/FP16，再对 GQA 求均值。
                mm_acc += mm.to(FAR_OUT.dtype.element_ty).to(tl.float32)
            if HAS_AVG:
                avg = tl.sum(kavg * q[None, :], axis=1)
                avg_acc += avg.to(FAR_OUT.dtype.element_ty).to(tl.float32)
        if not FAR_AVG or not NEAR_AVG:
            mm_score = (mm_acc / G).to(FAR_OUT.dtype.element_ty)
        if HAS_AVG:
            avg_score = (avg_acc / G).to(FAR_OUT.dtype.element_ty)
        if FAR_AVG:
            far_score = avg_score
        else:
            far_score = mm_score
        if NEAR_AVG:
            near_score = avg_score
        else:
            near_score = mm_score
        far_score = tl.where(far_pool, far_score, float("-inf"))
        near_score = tl.where(near_pool, near_score, float("-inf"))
        out_offs = pid_h.to(tl.int64) * T + offs_t
        tl.store(FAR_OUT + out_offs, far_score, mask=mask_t)
        tl.store(NEAR_OUT + out_offs, near_score, mask=mask_t)

    @triton.jit
    def _nfa_fine_scores_paged_kernel(
        KV_PTRS, BLOCK_TABLE, VALID_REQ, DIMS, Q, CAND, KEEP, OWNERS,
        GENERATIONS, LOGITS, LOW, HIGH, QUOTA, KV_REF, R,
        stride_qm, stride_qh, stride_ch, stride_cr,
        S: tl.constexpr, MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
        D_BLOCK: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
        TILE: tl.constexpr, MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
        TOKEN_BITS: tl.constexpr,
    ):
        """按 tile 并行计算 grouped-Q 的 BF16 Logits 或有序 packed key。"""
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
        safe = tl.where(read_r, cand, 0)
        lp = safe // PAGE
        off = safe % PAGE
        lp_valid = lp < MAX_BLOCKS
        low = tl.load(LOW + m)
        high = tl.load(HIGH + m)
        quota = tl.load(QUOTA + m)
        # protected、越界、非 owner 与 quota=0 的候选不参与 middle 竞争，
        # 在物理页表和低维 K 读取之前过滤；NaN 仍在打分后按原规则排除。
        load_r = (unique & lp_valid & (cand >= low) & (cand < high)
                  & (quota > 0))
        phys = tl.load(
            BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
            + req_id * MAX_BLOCKS + lp,
            mask=load_r, other=0).to(tl.int64)
        src = (kv_ptr
               + phys[:, None] * HK * PAGE * HEAD_DIM
               + kv * PAGE * HEAD_DIM
               + off[:, None] * HEAD_DIM
               + dim[None, :])
        k = tl.load(
            src, mask=load_r[:, None] & dim_mask[None, :],
            other=0.0)
        q = tl.load(
            Q + m * stride_qm + kv * stride_qh + offs_d,
            mask=dim_mask, other=0.0)
        score = tl.sum(k.to(tl.float32) * q[None, :], axis=1)
        score_bf16 = score.to(tl.bfloat16)
        score = score_bf16.to(tl.float32)
        good = load_r & (score == score)
        score = tl.where(good, score, float("-inf"))
        if TOKEN_BITS:
            # 与原 mono Top-K 完全相同的 signed-BF16 ordered key 和
            # 同分新 token 优先编码；长序列使用 int64，保留完整 token 域。
            bits = score_bf16.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
            negative = (bits & 0x8000) != 0
            ordered = tl.where(
                negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
            packed_good = good & (score != float("-inf"))
            delta = S - cand.to(tl.int32)
            if TOKEN_BITS == 32:
                ordered = ordered.to(tl.int64)
                delta = delta.to(tl.int64)
            invalid = -32768 << TOKEN_BITS
            key = tl.where(
                packed_good, (ordered << TOKEN_BITS) + delta, invalid)
            tl.store(LOGITS + row * R + offs_r, key, mask=mask_r)
        else:
            tl.store(LOGITS + row * R + offs_r, score.to(tl.bfloat16), mask=mask_r)

    @triton.jit
    def _finish_sorted_logits_key(
        h, key, LOW, HIGH, QUOTA, OUT, COUNT,
        S: tl.constexpr, K: tl.constexpr, SINK: tl.constexpr,
        META_GROUP: tl.constexpr, TOKEN_BITS: tl.constexpr,
    ):
        """解码 signed-Logits key，并保留 canonical middle 的补洞语义。"""
        i = tl.arange(0, K)
        b = 0 if META_GROUP == 0 else h // META_GROUP
        q = tl.load(QUOTA + b).to(tl.int32)
        if q == 0:
            tl.store(OUT + h * K + i, 0)
            tl.store(COUNT + h, 0)
        else:
            invalid = (-32768 << TOKEN_BITS)
            valid = key > invalid
            token_mask = (1 << TOKEN_BITS) - 1
            chosen = tl.where(
                (i < q) & valid, S - (key & token_mask), S).to(tl.int32)
            nvalid = tl.sum(((i < q) & valid).to(tl.int32), 0)
            if nvalid == q:
                fast_packed = tl.sort(tl.where(chosen >= SINK, chosen, S),
                                      descending=False)
                tl.store(OUT + h * K + i,
                         tl.where(fast_packed < S, fast_packed, 0))
                tl.store(COUNT + h,
                         tl.sum((fast_packed < S).to(tl.int32), 0))
            else:
                chosen = tl.sort(chosen, descending=False)
                low = tl.load(LOW + b).to(tl.int32)
                high = tl.load(HIGH + b).to(tl.int32)
                boundaries = tl.where(chosen < S, chosen - low - i, S)
                lo = tl.zeros((K,), tl.int32)
                hi = tl.full((K,), K, tl.int32)
                for _ in range(11):
                    mid = lo + (hi - lo) // 2
                    value = tl.gather(boundaries, tl.minimum(mid, K - 1), axis=0)
                    right = value <= i
                    active = lo < hi
                    lo = tl.where(active & right, mid + 1, lo)
                    hi = tl.where(active & ~right, mid, hi)
                zero = low + i + lo
                zero = tl.where((i < q - nvalid) & (zero < high), zero, S)
                combined = tl.cat(chosen, zero, can_reorder=True)
                combined = tl.where(combined >= SINK, combined, S)
                packed = tl.sort(combined, descending=False)
                j = tl.arange(0, 2 * K)
                tl.store(OUT + h * K + j, tl.where(packed < S, packed, 0), j < K)
                tl.store(COUNT + h, tl.sum((packed < S).to(tl.int32), 0))

    @triton.jit
    def _nfa_filtered_topk_kernel(
        LOGITS, CAND, LOW, HIGH, QUOTA, OUT, COUNT, R,
        R_PAD: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        SINK: tl.constexpr, META_GROUP: tl.constexpr,
        TOKEN_BITS: tl.constexpr,
    ):
        """单 CTA 对 signed BF16 Logits 做 Top-K，同分优先新 token。"""
        h = tl.program_id(0)
        b = 0 if META_GROUP == 0 else h // META_GROUP
        offs = tl.arange(0, R_PAD)
        mask = offs < R
        token = tl.load(CAND + h * R + offs, mask=mask, other=0).to(tl.int32)
        logits = tl.load(LOGITS + h * R + offs, mask=mask,
                         other=float("-inf"))
        score = logits.to(tl.float32)
        good = (mask & (token >= tl.load(LOW + b))
                & (token < tl.load(HIGH + b))
                & (tl.load(QUOTA + b) > 0)
                & (score == score) & (score != float("-inf")))
        bits = logits.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
        negative = (bits & 0x8000) != 0
        ordered = tl.where(
            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
        delta = S - token
        if TOKEN_BITS == 32:
            ordered = ordered.to(tl.int64)
            delta = delta.to(tl.int64)
        invalid = -32768 << TOKEN_BITS
        key = tl.where(good, (ordered << TOKEN_BITS) + delta, invalid)
        if K == 1:
            key = tl.max(key, axis=0, keep_dims=True)
        else:
            key = tl.topk(key, K)
        _finish_sorted_logits_key(
            h, key, LOW, HIGH, QUOTA, OUT, COUNT,
            S=S, K=K, SINK=SINK, META_GROUP=META_GROUP,
            TOKEN_BITS=TOKEN_BITS)

    @triton.jit
    def _nfa_filtered_topk_radix_kernel(
        LOGITS, CAND, LOW, HIGH, QUOTA, TOP, R,
        R_PAD: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        META_GROUP: tl.constexpr,
    ):
        """Radix-select 第 K 个 signed-BF16 key，再精确压实 Top-K。

        与整行 ``tl.topk`` 使用同一 packed key；输出无需有序，因为 finish
        随后会按逻辑 token 重排并执行 quota 补洞。
        """
        h = tl.program_id(0)
        b = 0 if META_GROUP == 0 else h // META_GROUP
        offs = tl.arange(0, R_PAD)
        mask = offs < R
        token = tl.load(CAND + h * R + offs, mask=mask, other=0).to(tl.int32)
        logits = tl.load(LOGITS + h * R + offs, mask=mask,
                         other=float("-inf"))
        score = logits.to(tl.float32)
        good = (mask & (token >= tl.load(LOW + b))
                & (token < tl.load(HIGH + b))
                & (tl.load(QUOTA + b) > 0)
                & (score == score) & (score != float("-inf")))
        bits = logits.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
        negative = (bits & 0x8000) != 0
        ordered = tl.where(
            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
        delta = S - token
        invalid = -32768 << 16
        key = tl.where(good, (ordered << 16) + delta, invalid)
        out_offs = tl.arange(0, K)
        if tl.load(QUOTA + b) > 0:
            nvalid = tl.sum(good.to(tl.int32), axis=0)
            if nvalid >= K:
                # 在 packed int32 全值域二分第 K 个 key；packed key 与
                # 原 tl.topk 完全一致，因此阈值和压实结果也保持一致。
                lo = tl.full((), -2147483648, tl.int64)
                hi = tl.full((), 2147483647, tl.int64)
                for _ in tl.static_range(32):
                    mid = (lo + hi + 1) // 2
                    count = tl.sum(
                        (good & (key >= mid.to(tl.int32))).to(tl.int32), axis=0)
                    take = count >= K
                    lo = tl.where(take, mid, lo)
                    hi = tl.where(take, hi, mid - 1)
                threshold = lo.to(tl.int32)
            else:
                threshold = invalid
            greater = good & (key > threshold)
            greater_pos = tl.cumsum(greater.to(tl.int32), axis=0) - 1
            tl.store(TOP + h * K + greater_pos, key,
                     mask=greater & (greater_pos < K))
            equal = good & (key == threshold)
            ngreater = tl.sum(greater.to(tl.int32), axis=0)
            equal_pos = ngreater + tl.cumsum(equal.to(tl.int32), axis=0) - 1
            tl.store(TOP + h * K + equal_pos, key,
                     mask=equal & (equal_pos < K))
            nequal = tl.sum(equal.to(tl.int32), axis=0)
            total = tl.minimum(ngreater + nequal, K)
            tl.store(TOP + h * K + out_offs, invalid, mask=out_offs >= total)
        else:
            tl.store(TOP + h * K + out_offs, invalid)

    @triton.jit
    def _nfa_finish_topk_kernel(
        TOP, LOW, HIGH, QUOTA, OUT, COUNT,
        S: tl.constexpr, K: tl.constexpr, SINK: tl.constexpr,
        META_GROUP: tl.constexpr, TOKEN_BITS: tl.constexpr,
    ):
        h = tl.program_id(0)
        i = tl.arange(0, K)
        key = tl.load(TOP + h * K + i)
        _finish_sorted_logits_key(
            h, key, LOW, HIGH, QUOTA, OUT, COUNT,
            S=S, K=K, SINK=SINK, META_GROUP=META_GROUP,
            TOKEN_BITS=TOKEN_BITS)

    @triton.jit
    def _nfa_filtered_topk_partial_kernel(
        LOGITS, PARTIAL, R,
        TILE: tl.constexpr, NTILES: tl.constexpr,
        S: tl.constexpr, K: tl.constexpr, META_GROUP: tl.constexpr,
        TOKEN_BITS: tl.constexpr,
    ):
        """直接对 fine-score 阶段生成的 packed key 做局部 Top-K。"""
        h = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * TILE + tl.arange(0, TILE)
        mask = offs < R
        invalid = -32768 << TOKEN_BITS
        key = tl.load(LOGITS + h * R + offs, mask=mask, other=invalid)
        if K == 1:
            key = tl.max(key, axis=0, keep_dims=True)
        else:
            key = tl.topk(key, K)
        tl.store(PARTIAL + (h * NTILES + tile) * K + tl.arange(0, K), key)

    @triton.jit
    def _nfa_filtered_topk_merge_kernel(
        PARTIAL, LOW, HIGH, QUOTA, OUT, COUNT,
        NTILES: tl.constexpr, MERGE_PAD: tl.constexpr, LOG2_K: tl.constexpr,
        S: tl.constexpr, K: tl.constexpr, SINK: tl.constexpr,
        META_GROUP: tl.constexpr, TOKEN_BITS: tl.constexpr,
    ):
        h = tl.program_id(0)
        if K == 1:
            offs = tl.arange(0, MERGE_PAD)
            invalid = -32768 << TOKEN_BITS
            key = tl.load(PARTIAL + h * NTILES * K + offs,
                          mask=offs < NTILES * K, other=invalid)
            key = tl.max(key, axis=0, keep_dims=True)
        else:
            # 各 partial run 已按降序排列。原实现将 A 与反转后的 B 拼成
            # 2K bitonic 序列；其前半项逐元素恰为 max(A[i], B[K-1-i])，
            # 因此直接生成这 K 项，删除等价的 2K gather/select 中间网络。
            # 后续 K-wide 排序与 run 归并顺序不变，packed key 逐位一致。
            i = tl.arange(0, K)
            key = tl.load(PARTIAL + h * NTILES * K + i)
            for group in tl.static_range(1, NTILES):
                incoming = tl.load(
                    PARTIAL + (h * NTILES + group) * K + (K - 1 - i))
                key = tl.maximum(key, incoming)
                for bit in tl.static_range(LOG2_K - 1, -1, -1):
                    stride = 1 << bit
                    peer = tl.gather(key, i ^ stride, axis=0)
                    key = tl.where(
                        (i & stride) == 0,
                        tl.maximum(key, peer), tl.minimum(key, peer))
        _finish_sorted_logits_key(
            h, key, LOW, HIGH, QUOTA, OUT, COUNT,
            S=S, K=K, SINK=SINK, META_GROUP=META_GROUP,
            TOKEN_BITS=TOKEN_BITS)

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
    def _prepare_q_grouped_kernel(
        Q_SRC, REQ_IDS, SCALES, DIM_IDX, OUT, S0, S1, S2,
        HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        DQ: tl.constexpr, B: tl.constexpr, DQ_BLOCK: tl.constexpr,
    ):
        """逐 query head 保留生产缩放/BF16 舍入，再在 FP32 中求和。"""
        pid = tl.program_id(0)
        m_idx = pid // HKV
        kv = pid % HKV
        layer = m_idx // B
        req = m_idx % B
        req_id = tl.load(REQ_IDS + req)
        scale = tl.load(SCALES + layer)
        offs = tl.arange(0, DQ_BLOCK)
        mask = offs < DQ
        dim = tl.load(DIM_IDX + offs, mask=mask, other=0)
        grouped = tl.zeros((DQ_BLOCK,), tl.float32)
        for j in tl.static_range(G):
            hq = kv * G + j
            src = Q_SRC + layer * S0 + req_id * S1 + hq * S2 + dim
            q = tl.load(src, mask=mask, other=0.0).to(tl.float32)
            grouped += (q * scale).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + m_idx * (HKV * DQ) + kv * DQ + offs,
                 grouped, mask=mask)

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
        BLOCKS_PER_PROGRAM: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
        BLOCK_GROUP_STRIDE: tl.constexpr,
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
                owner = (block_group
                 == (last_blk // BLOCKS_PER_PROGRAM) % BLOCK_GROUP_STRIDE)
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
            # 每个 KV group 都有独立的 L1/L2 与 attention 消费者。
            _nfa_page_stats_body(
                block_group, row,
                KV_PTRS, BLOCK_TABLE, LENGTHS, ACTIVE_REQ, VALID_REQ, DIMS,
                STATS_LEN, P_MIN, P_MAX, P_SUM,
                MAX_BLOCKS=MAX_BLOCKS, PAGE=PAGE, HK=HK, HEAD_DIM=HEAD_DIM,
                DIM_NUMEL=DIM_NUMEL, D_BLOCK=D_BLOCK, BS=BS, KT_CAP=KT_CAP,
                L=L, B=B, BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
                BLOCK_GROUPS=BLOCK_GROUPS, BLOCK_GROUP_STRIDE=BLOCK_GROUP_STRIDE,
                MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)
        else:
            _prepare_q_body(
                pid - STATS_PROGRAMS, Q_SRC, Q_REQ, SCALES, DIMS, Q_OUT,
                S0, S1, S2, HQ=HQ, D=D, DQ=DQ, B=B, DQ_BLOCK=DQ_BLOCK)

    @triton.jit
    def _candidate_pack_batched_kernel(
        I_F, I_N, FAR_SCORES, NEAR_SCORES, LENGTHS,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
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
        req = m % B
        length = tl.load(LENGTHS + req)
        lo = tl.maximum(0, length - SWA)
        width = R + HEAD + TAIL

        # generation 隔离不同 selector 调用；同次调用仍由 atomic_max 选择
        # 最大 candidate slot，不再清空整个 CAP 的 owner 表。
        generation = tl.load(GENERATIONS + row) + 1
        # int32 owner 表将去重 workspace/atomic 位宽减半。极限计数到达前
        # 逐行清空并重新计代，避免 generation*width 溢出后误复用旧 owner。
        generation_limit = 2147483647 // width
        if generation > generation_limit:
            generation = 1
            clear_offs = tl.arange(0, 1024)
            for clear_start in range(0, CAP, 1024):
                clear = clear_start + clear_offs
                tl.store(OWNERS + row * CAP + clear, 0, mask=clear < CAP)
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
        MIDDLE, MIDDLE_COUNTS, LENGTHS, VALID_REQ, BLOCK_TABLE,
        GATHER, VIS_LENS,
        W: tl.constexpr, PROT: tl.constexpr,
        N: tl.constexpr, N_PAD: tl.constexpr, A_PAD: tl.constexpr,
        MERGE_BITS: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        MAX_REQ: tl.constexpr, MAX_VIS: tl.constexpr,
        INVALID: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
        LAYER_OFFSET: tl.constexpr, EMPTY: tl.constexpr,
    ):
        """逐 KV group 做 middle ∪ protected 并写物理槽。"""
        row = tl.program_id(0)
        m = row // HKV
        kv = row % HKV
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)
        length = tl.load(LENGTHS + req).to(tl.int32)
        offs = tl.arange(0, N_PAD)
        in_n = offs < N

        # ranking 的 middle 已经按逻辑 token 升序；只需排序较短的
        # protected，再把第二路反转后与 middle 做 bitonic merge。
        if EMPTY:
            middle_count = 0
            middle_id = tl.full(offs.shape, CAP, tl.int32)
        else:
            middle_count = tl.load(MIDDLE_COUNTS + row)
            middle_id = tl.load(
                MIDDLE + row * W + offs,
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

        aux_id = tl.where((in_sink | in_swa) & (protected_id >= 0)
                          & (protected_id < CAP), protected_id, CAP)
        aux_sorted = tl.sort(aux_id, descending=False)

        reverse_idx = 2 * W - 1 - offs
        # N_PAD 尾部 lane 的 reverse_idx 可能为负；where 的掩码不能撤销
        # tl.gather 本身，因此索引必须先夹到合法范围再按原条件丢弃。
        aux_gather_idx = tl.maximum(
            tl.minimum(reverse_idx, A_PAD - 1), 0)
        aux_reverse = tl.gather(aux_sorted, aux_gather_idx, 0)
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
        dst = (GATHER + layer * (MAX_REQ * HKV * MAX_VIS)
               + req * (HKV * MAX_VIS) + kv * MAX_VIS)
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
        tl.store(VIS_LENS + (layer * MAX_REQ + req) * HKV + kv, count)

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
        # 只需有序 Top-K；直接部分排序，避免对 KT_PAD 个整数做全排序。
        # Triton 3.4 的 topk(k=1) reshape 有缺陷，单项直接 max。
        if K_PAD == 1:
            top = tl.max(packed, axis=0, keep_dims=True)
        else:
            top = tl.topk(packed, K_PAD)
        top_idx = tl.arange(0, K_PAD)
        out_idx = (KT - 1 - (top & 0xFFFF)).to(OUT.dtype.element_ty)
        tl.store(OUT + row * K + top_idx, out_idx, mask=top_idx < K)

    @triton.jit
    def _l1_topk_partial_body(
        SCORES, OUT, row, tile,
        KT: tl.constexpr, TILE: tl.constexpr, NTILES: tl.constexpr,
        K_PAD: tl.constexpr,
    ):
        """每个 CTA 只排序 512 项；packed key 与整行实现逐位一致。"""
        offs = tile * TILE + tl.arange(0, TILE)
        mask = offs < KT
        s = tl.load(SCORES + row * KT + offs, mask=mask, other=float("-inf"))
        bits = s.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
        sign = bits & 0x8000
        key = tl.where(sign != 0, (~bits) & 0xFFFF, bits | 0x8000)
        key_signed = key - 0x8000
        idx = KT - 1 - offs
        packed = (key_signed << 16) | (idx & 0xFFFF)
        packed = tl.where(mask, packed, -0x80000000)
        if K_PAD == 1:
            top = tl.max(packed, axis=0, keep_dims=True)
        else:
            top = tl.topk(packed, K_PAD)
        top_idx = tl.arange(0, K_PAD)
        tl.store(OUT + (row * NTILES + tile) * K_PAD + top_idx, top)

    @triton.jit
    def _l1_row_skip_flags(
        SKIP_FAR, SKIP_NEAR, VALID_REQ, row,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        m = row // HKV
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)
        skip_far = tl.load(
            SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id) != 0
        skip_near = tl.load(
            SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id) != 0
        return skip_far, skip_near

    @triton.jit
    def _l1_topk_partial_kernel(
        FAR_SCORES, NEAR_SCORES, FAR_PARTIAL, NEAR_PARTIAL,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        KT: tl.constexpr, TILE: tl.constexpr, NTILES: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        skip_far, skip_near = _l1_row_skip_flags(
            SKIP_FAR, SKIP_NEAR, VALID_REQ, row,
            HKV=HKV, B=B, MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)
        if skip_far == 0:
            _l1_topk_partial_body(
                FAR_SCORES, FAR_PARTIAL, row, tile,
                KT=KT, TILE=TILE, NTILES=NTILES, K_PAD=K_F_PAD)
        if skip_near == 0:
            _l1_topk_partial_body(
                NEAR_SCORES, NEAR_PARTIAL, row, tile,
                KT=KT, TILE=TILE, NTILES=NTILES, K_PAD=K_N_PAD)

    @triton.jit
    def _topk_indices_merge_body(
        PARTIAL, OUT,
        KT: tl.constexpr, NTILES: tl.constexpr, NTILES_PAD: tl.constexpr,
        K: tl.constexpr, K_PAD: tl.constexpr, LOG2_K: tl.constexpr,
    ):
        """归并已排序的局部 run，精确保留全局 Top-K 与同分顺序。"""
        row = tl.program_id(0)
        if K_PAD == 1:
            offs = tl.arange(0, NTILES_PAD)
            key = tl.load(PARTIAL + row * NTILES + offs,
                          mask=offs < NTILES, other=-0x80000000)
            top = tl.max(key, axis=0, keep_dims=True)
        else:
            i = tl.arange(0, K_PAD)
            top = tl.load(PARTIAL + row * NTILES * K_PAD + i)
            for tile in tl.static_range(1, NTILES):
                incoming = tl.load(
                    PARTIAL + (row * NTILES + tile) * K_PAD + (K_PAD - 1 - i))
                top = tl.maximum(top, incoming)
                for bit in tl.static_range(LOG2_K - 1, -1, -1):
                    stride = 1 << bit
                    peer = tl.gather(top, i ^ stride, axis=0)
                    top = tl.where(
                        (i & stride) == 0,
                        tl.maximum(top, peer), tl.minimum(top, peer))
        top_idx = tl.arange(0, K_PAD)
        out_idx = (KT - 1 - (top & 0xFFFF)).to(OUT.dtype.element_ty)
        tl.store(OUT + row * K + top_idx, out_idx, mask=top_idx < K)

    @triton.jit
    def _l1_skipped_topk_body(
        OUT, row, K: tl.constexpr, K_PAD: tl.constexpr,
    ):
        # host 已保证 K<=KT；被 skip 的池由 L1 score 置为全 -inf，
        # 原 packed-key Top-K 的逆序索引恰好产生 0..K-1。
        offs = tl.arange(0, K_PAD)
        tl.store(OUT + row * K + offs, offs.to(OUT.dtype.element_ty),
                 mask=offs < K)

    @triton.jit
    def _l1_candidate_topk_body(
        FAR_TOPK, NEAR_TOPK, I_F, I_N, skip_far, skip_near,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K_F: tl.constexpr, K_N: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
        TILED: tl.constexpr, NTILES: tl.constexpr, NTILES_PAD: tl.constexpr,
        LOG2_F: tl.constexpr, LOG2_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if skip_far:
            _l1_skipped_topk_body(I_F, row, K=K_F, K_PAD=K_F_PAD)
        elif TILED:
            _topk_indices_merge_body(
                FAR_TOPK, I_F, KT=KT, NTILES=NTILES, NTILES_PAD=NTILES_PAD,
                K=K_F, K_PAD=K_F_PAD, LOG2_K=LOG2_F)
        else:
            _topk_indices_body(
                FAR_TOPK, I_F, KT=KT, KT_PAD=KT_PAD, K=K_F, K_PAD=K_F_PAD)
        if skip_near:
            _l1_skipped_topk_body(I_N, row, K=K_N, K_PAD=K_N_PAD)
        elif TILED:
            _topk_indices_merge_body(
                NEAR_TOPK, I_N, KT=KT, NTILES=NTILES, NTILES_PAD=NTILES_PAD,
                K=K_N, K_PAD=K_N_PAD, LOG2_K=LOG2_N)
        else:
            _topk_indices_body(
                NEAR_TOPK, I_N, KT=KT, KT_PAD=KT_PAD, K=K_N, K_PAD=K_N_PAD)

    @triton.jit
    def _topk_candidate_batched_kernel(
        FAR_TOPK, NEAR_TOPK, FAR_SCORES, NEAR_SCORES, I_F, I_N, LENGTHS,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K_F: tl.constexpr, K_N: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
        TILED: tl.constexpr, NTILES: tl.constexpr, NTILES_PAD: tl.constexpr,
        LOG2_F: tl.constexpr, LOG2_N: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, PAGE: tl.constexpr,
        NEAR_LEN: tl.constexpr, K2: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """eager 单 CTA 顺序完成两路 Top-K 与 candidate expansion。"""
        row = tl.program_id(0)
        skip_far, skip_near = _l1_row_skip_flags(
            SKIP_FAR, SKIP_NEAR, VALID_REQ, row,
            HKV=HKV, B=B, MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)
        _l1_candidate_topk_body(
            FAR_TOPK, NEAR_TOPK, I_F, I_N, skip_far, skip_near,
            KT=KT, KT_PAD=KT_PAD, K_F=K_F, K_N=K_N,
            K_F_PAD=K_F_PAD, K_N_PAD=K_N_PAD, TILED=TILED,
            NTILES=NTILES, NTILES_PAD=NTILES_PAD,
            LOG2_F=LOG2_F, LOG2_N=LOG2_N)
        tl.debug_barrier()
        _candidate_pack_batched_kernel(
            I_F, I_N, FAR_SCORES, NEAR_SCORES, LENGTHS,
            SKIP_FAR, SKIP_NEAR, VALID_REQ,
            OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
            K_F=K_F, K_N=K_N, KT=KT, R=R, HEAD=HEAD, TAIL=TAIL,
            SINK=SINK, SWA=SWA, CAP=CAP, SINK_POOL_LO=SINK_POOL_LO,
            BS=BS, PAGE=PAGE, NEAR_LEN=NEAR_LEN, K2=K2, HKV=HKV, B=B,
            MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)

    @triton.jit
    def _topk_candidate_prepare_kernel(
        FAR_SCORES, NEAR_SCORES, I_F, I_N, LENGTHS,
        SKIP_FAR, SKIP_NEAR, VALID_REQ,
        LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K_F: tl.constexpr, K_N: tl.constexpr,
        K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
        TILED: tl.constexpr, NTILES: tl.constexpr, NTILES_PAD: tl.constexpr,
        LOG2_F: tl.constexpr, LOG2_N: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, PAGE: tl.constexpr,
        NEAR_LEN: tl.constexpr, K2: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """完成两路精确 Top-K，并在并行展开前稳定 generation/metadata。"""
        row = tl.program_id(0)
        skip_far, skip_near = _l1_row_skip_flags(
            SKIP_FAR, SKIP_NEAR, VALID_REQ, row,
            HKV=HKV, B=B, MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)
        _l1_candidate_topk_body(
            FAR_SCORES, NEAR_SCORES, I_F, I_N, skip_far, skip_near,
            KT=KT, KT_PAD=KT_PAD, K_F=K_F, K_N=K_N,
            K_F_PAD=K_F_PAD, K_N_PAD=K_N_PAD, TILED=TILED,
            NTILES=NTILES, NTILES_PAD=NTILES_PAD,
            LOG2_F=LOG2_F, LOG2_N=LOG2_N)

        m = row // HKV
        h = row % HKV
        req = m % B
        length = tl.load(LENGTHS + req)
        lo = tl.maximum(0, length - SWA)
        width = R + HEAD + TAIL

        # 与 fused kernel 相同：每行每次 selector 调用只推进一次 generation。
        # 独立 kernel 边界保证后续所有展开 CTA 读到同一个稳定值，也保证
        # 极端回绕时的 owner 清零先于任何新 tagged atomic_max。
        generation = tl.load(GENERATIONS + row) + 1
        generation_limit = 2147483647 // width
        if generation > generation_limit:
            generation = 1
            clear_offs = tl.arange(0, 1024)
            for clear_start in range(0, CAP, 1024):
                clear = clear_start + clear_offs
                tl.store(OWNERS + row * CAP + clear, 0, mask=clear < CAP)
        tl.store(GENERATIONS + row, generation)

        if h == 0:
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
    def _candidate_expand_batched_kernel(
        I_F, I_N, FAR_SCORES, NEAR_SCORES, LENGTHS,
        OUT, VALID, OWNERS, GENERATIONS,
        K_F: tl.constexpr, K_N: tl.constexpr, KT: tl.constexpr,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        SINK_POOL_LO: tl.constexpr, BS: tl.constexpr,
        HKV: tl.constexpr, B: tl.constexpr,
    ):
        """按 1024-slot chunk 并行展开；每个 virtual slot 的语义不变。"""
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        m = row // HKV
        req = m % B
        length = tl.load(LENGTHS + req)
        generation = tl.load(GENERATIONS + row)
        width = R + HEAD + TAIL
        r = chunk * 1024 + tl.arange(0, 1024)
        token, active = _candidate_pack_virtual_slot(
            I_F, I_N, FAR_SCORES, NEAR_SCORES, row, r, length,
            K_F=K_F, K_N=K_N, KT=KT, R=R, HEAD=HEAD, TAIL=TAIL,
            SINK=SINK, SWA=SWA, SINK_POOL_LO=SINK_POOL_LO, BS=BS,
            live=r < width)
        tl.store(OUT + row * width + r, token, r < width)
        tl.store(VALID + row * width + r, active, r < width)
        # max 对重复 token 的结果与 CTA/原子到达顺序无关；tagged_slot 仍
        # 精确等于 fused 单 CTA 版本，最大 candidate slot 去重规则不变。
        tagged_slot = generation * width + r
        tl.atomic_max(OWNERS + row * CAP + token, tagged_slot,
                      mask=active, sem="relaxed")

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


def _page_stats_all_layers(cache, req_ids, n_layers, dims, bs, pad, lengths,
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
    l = n_layers
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
    block_groups = triton.cdiv(kt_cap, blocks_per_program)
    # 固定 4 路覆盖重建；稳态只有最新块 owner 工作，避免按 KT_CAP
    # 启动大量读取元数据后立即退出的空 CTA。
    stats_group_count = min(4, block_groups)
    stats_rows = m * hk
    stats_programs = stats_group_count * stats_rows
    if q_job is None:
        _fast_launch(workspace, ('page_stats', layer_offset), _nfa_page_stats_kernel,
                     (stats_group_count, stats_rows),
            kv_ptrs, block_table, lengths_t, active_req_t, valid_req_t, dims_t,
            stats_len, p_min, p_max, p_sum,
            MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
            HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dim_numel,
            D_BLOCK=triton.next_power_of_2(dim_numel), BS=bs,
            KT_CAP=kt_cap, L=l, B=b, BLOCKS_PER_PROGRAM=blocks_per_program,
            BLOCK_GROUPS=block_groups, BLOCK_GROUP_STRIDE=stats_group_count,
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
            BLOCK_GROUPS=block_groups, BLOCK_GROUP_STRIDE=stats_group_count,
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
        cache, req_ids, n_layers, dims, q_fine, cand_idx2, keep2, owners,
        generations, capacity, lower, upper, quota,
        tile=128, num_warps=4, layer_offset=0, workspace=None):
    """Candidate gather -> paged K load -> grouped-Q BF16 Logits/packed key。"""
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
    token_bits = 32 if capacity >= 65536 else 16
    logits = _workspace_empty(
        workspace, 'fine_logits', (m, hkv, r),
        torch.int64 if token_bits == 32 else torch.int32, device)
    kv_ptrs = cache.layer_kv_ptrs_buf[
        layer_offset:layer_offset + n_layers]
    _fast_launch(workspace, ('fine_logits', layer_offset),
                 _nfa_fine_scores_paged_kernel,
                 (m * hkv, triton.cdiv(r, tile)),
        kv_ptrs, cache.block_table_buf, valid_req_t, dims, q_fine, cand_idx2,
        keep2, owners, generations, logits, lower, upper, quota,
        layer0.kv_pool_k, r,
        stride_qm=q_fine.stride(0), stride_qh=q_fine.stride(1),
        stride_ch=cand_idx2.stride(1), stride_cr=cand_idx2.stride(2),
        S=capacity, MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dims.numel(),
        D_BLOCK=triton.next_power_of_2(dims.numel()),
        HKV=hkv, B=b, TILE=tile,
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        TOKEN_BITS=token_bits,
        num_warps=num_warps,
    )
    return logits


def _nfa_filtered_topk(logits, cand_idx2, lower, upper, quota, capacity, width,
                       sink, layer_offset=0, workspace=None):
    """将 grouped-Q Logits 直接转换为 canonical ``TokenSelection``。"""
    m, hkv, r = cand_idx2.shape
    out = _workspace_empty(
        workspace, 'middle_indices', (m, hkv, width), torch.int32, logits.device)
    count = _workspace_empty(
        workspace, 'middle_counts', (m, hkv), torch.int32, logits.device)
    token_bits = 32 if capacity >= 65536 else 16
    # 两种 token 域都按 2048 候选并行局部 Top-K，再精确归并；mono kernel
    # 保留为 bit-exact reference，避免长序列退化为单 CTA 的 16384-wide 排序。
    tile = 2048
    ntiles = triton.cdiv(r, tile)
    partial = _workspace_empty(
        workspace, 'filtered_topk_partial', (m, hkv, ntiles, width),
        logits.dtype, logits.device)
    _fast_launch(workspace, ('filtered_topk_partial', layer_offset),
                 _nfa_filtered_topk_partial_kernel, (m * hkv, ntiles),
        logits, partial, r,
        TILE=tile, NTILES=ntiles,
        S=capacity, K=width, META_GROUP=hkv, TOKEN_BITS=token_bits,
        num_warps=16,
    )
    _fast_launch(workspace, ('filtered_topk_merge', layer_offset),
                 _nfa_filtered_topk_merge_kernel, (m * hkv,),
        partial, lower, upper, quota, out, count,
        NTILES=ntiles, MERGE_PAD=triton.next_power_of_2(ntiles * width),
        LOG2_K=width.bit_length() - 1,
        S=capacity, K=width, SINK=sink,
        META_GROUP=hkv, TOKEN_BITS=token_bits,
        num_warps=16,
    )
    return TokenSelection(out, count)


def _l1_scores(k_min, k_max, k_sum, b_q_full, group_size, indexer, dtype,
               cache, valid_req, lengths, page, bs, sink, near_len, window,
               layer_offset=0, workspace=None):
    """直接读取持久页统计并生成 far/near Top-K 输入。"""
    _, _, kt_cap, hkv, d = k_min.shape
    m = b_q_full.shape[0]
    hq = b_q_full.shape[1]
    q_flat = b_q_full.reshape(m * hq, d).contiguous()
    far_scored = _workspace_empty(
        workspace, 'l1_far', (m, hkv, kt_cap), dtype, k_min.device)
    near_scored = _workspace_empty(
        workspace, 'l1_near', (m, hkv, kt_cap), dtype, k_min.device)
    bt, nw = indexer._TILE_COARSE.get(d, (64, 4))
    lengths_t = lengths
    b = lengths.numel()
    far_avg = indexer.far_method == "avg"
    near_avg = indexer.near_method != "minmax"
    has_avg = far_avg or near_avg
    k_sum_arg = k_sum if has_avg else k_min
    _fast_launch(workspace, ('l1', layer_offset), _nfa_l1_scores_kernel,
                 (triton.cdiv(kt_cap, bt), m * hkv),
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
        Hkv=hkv, G=group_size, D=d, BT=bt, B=b,
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
                            layer_offset=0, workspace=None, fuse_topk=False,
                            parallel_expand=False):
    """直接生成 complete-pack 候选及过滤元数据，无 aten 中间张量。"""
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
        workspace, 'fine_owners', (m, hkv, capacity), torch.int32, device)
    generations = _workspace_zeros(
        workspace, 'fine_owner_generations', (m, hkv), torch.int32, device)
    b = lengths.numel() if torch.is_tensor(lengths) else len(lengths)
    lengths_t = (lengths if torch.is_tensor(lengths) else
                 torch.as_tensor(lengths, dtype=torch.int64, device=device))
    if fuse_topk:
        kt = far_scored.shape[-1]
        k_f_pad = triton.next_power_of_2(max(1, k_f))
        k_n_pad = triton.next_power_of_2(max(1, k_n))
        # 长行整行 Top-K 每个 KV head 只有一个 CTA，SM 覆盖不足。
        # 512 项局部排序保持相同 packed key，随后精确归并；短行仍走旧路径。
        # eager 每层多一次 launch 的 host 成本会超过 device 收益，保留单 kernel；
        # Graph/parallel_expand 路径通过 capture 消除 launch 成本，再启用 tiled。
        tiled = kt > 512 and parallel_expand
        ntiles = triton.cdiv(kt, 512) if tiled else 1
        if tiled:
            far_topk = _workspace_empty(
                workspace, 'l1_far_topk_partial',
                (m, hkv, ntiles, k_f_pad), torch.int32, device)
            near_topk = _workspace_empty(
                workspace, 'l1_near_topk_partial',
                (m, hkv, ntiles, k_n_pad), torch.int32, device)
            _fast_launch(workspace, ('l1_topk_partial', layer_offset),
                         _l1_topk_partial_kernel, (m * hkv, ntiles),
                far_scored, near_scored, far_topk, near_topk,
                cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
                KT=kt, TILE=512, NTILES=ntiles,
                K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                HKV=hkv, B=b, MAX_REQ=cache.max_requests,
                LAYER_OFFSET=layer_offset, num_warps=8)
        else:
            far_topk, near_topk = far_scored, near_scored
        common = dict(
            KT=kt, KT_PAD=triton.next_power_of_2(kt),
            K_F=k_f, K_N=k_n, K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
            TILED=tiled, NTILES=ntiles,
            NTILES_PAD=triton.next_power_of_2(ntiles),
            LOG2_F=k_f_pad.bit_length() - 1,
            LOG2_N=k_n_pad.bit_length() - 1,
            R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=int(indexer._sink_pool_lo_blk), BS=bs,
            PAGE=cache.geometry.page, NEAR_LEN=cache.geometry.near,
            K2=int(indexer.args.tia_level2_topk),
            HKV=hkv, B=b,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset)
        if parallel_expand:
            # 相互独立的 candidate chunk 在下一 kernel 并行展开；多一次
            # launch 换取更高 SM 覆盖，不改变排序、打分或去重语义。
            _fast_launch(workspace, ('topk_candidate_prepare', layer_offset),
                         _topk_candidate_prepare_kernel, (m * hkv,),
                far_topk, near_topk, i_f, i_n, lengths_t,
                cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
                lower, upper, quota, owners, generations,
                **common, num_warps=8)
            _fast_launch(workspace, ('candidate_expand', layer_offset),
                         _candidate_expand_batched_kernel,
                         (m * hkv, triton.cdiv(width, 1024)),
                i_f, i_n, far_scored, near_scored, lengths_t,
                out, valid, owners, generations,
                K_F=k_f, K_N=k_n, KT=far_scored.shape[-1],
                R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
                SINK_POOL_LO=int(indexer._sink_pool_lo_blk), BS=bs,
                HKV=hkv, B=b, num_warps=4)
        else:
            # eager 仍由单 fused kernel 完成归并、metadata 与 expansion，
            # 避免再拆一个 candidate kernel 的 host 开销。
            _fast_launch(workspace, ('topk_candidate', layer_offset),
                         _topk_candidate_batched_kernel, (m * hkv,),
                far_topk, near_topk, far_scored, near_scored, i_f, i_n, lengths_t,
                cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
                out, valid, lower, upper, quota, owners, generations,
                **common, num_warps=8)
    else:
        _fast_launch(workspace, ('candidate', layer_offset),
                     _candidate_pack_batched_kernel, (m * hkv,),
            i_f, i_n, far_scored, near_scored, lengths_t,
            cache.skip_far_layer_buf, cache.skip_near_layer_buf, valid_req,
            out, valid, lower, upper, quota, owners, generations,
            K_F=k_f, K_N=k_n, KT=far_scored.shape[-1],
            R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=int(indexer._sink_pool_lo_blk), BS=bs,
            PAGE=cache.geometry.page, NEAR_LEN=cache.geometry.near,
            K2=int(indexer.args.tia_level2_topk),
            HKV=hkv, B=b,
            MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
            num_warps=4)
    return out, valid, lower, upper, quota, owners, generations


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
        if kernel in (_prepare_q_kernel, _prepare_q_grouped_kernel):
            args = (q_src,) + args[1:]
            if run is not None:
                launch_values = (q_src,) + launch_values[1:]
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


def _store_eager_selector_call_plan(workspace, plan_key, geometry):
    """固定几何下保存本层 launch 序列，供后续 eager step 原序重放。"""
    record = workspace.pop('_nfa_eager_record', None)
    if record is None:
        return
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
                and tuple(req_ids) == workspace.get('_nfa_launch_req_ids')
                and cache.active_visible_bucket == geometry[12]
                and query_states.shape ==
                    (geometry[0], geometry[2], cache.layers[0].head_dim_k)):
            has_live_req = any(
                rid is not None and rid >= 0 and rid in cache.states
                for rid in req_ids)
            protected_only = has_live_req and all(
                rid is None or rid < 0 or rid not in cache.states
                or (cache.states[rid].skip_far_by_layer[layer_idx]
                    and cache.states[rid].skip_near_by_layer[layer_idx])
                for rid in req_ids)
            q_src = query_states.unsqueeze(0)
            q_geometry = (q_src.dtype, q_src.data_ptr() % 16,
                          q_src.stride(0), q_src.stride(1), q_src.stride(2))
            if (q_geometry == geometry[26:31]
                    and protected_only == geometry[31]):
                if geometry[31] and decode_write is not None:
                    # both-skip 轻路径没有 page-stats/Q fused kernel 可携带
                    # decode 写入，重放 final union 前必须显式写当前 K/V。
                    cache.write_decode(layer_idx, *decode_write)
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

    # PAD 行使用 request 0 的安全页表地址，但 length=0，因而不产生可见 token。
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
    # 所有有效请求都跳过 far/near 时 middle 恒为空；无效 padding 行不参与
    # 判定。全 padding 的 Graph capture 必须走通用路径，否则会把真实 replay
    # 请求错误特化成空 middle。该状态进入 eager 几何，防止 mixed batch 误重放。
    has_live_req = any(
        rid is not None and rid >= 0 and rid in cache.states
        for rid in req_ids)
    protected_only = n_layers == 1 and has_live_req and all(
        rid is None or rid < 0 or rid not in cache.states
        or (cache.states[rid].skip_far_by_layer[layer_offset]
            and cache.states[rid].skip_near_by_layer[layer_offset])
        for rid in req_ids)
    # 同一 Cache 的生产几何固定；用一个严格令牌替代每次 launch 的逐张量键。
    # 独立 ranking 调用没有该令牌，仍使用完整逐参数 specialization 键。
    geometry = (
        b, n_layers, hq, hkv, group_size, bs, page, pad, kt_cap, s_cap,
        capturing,
        cache.max_requests, cache.active_visible_bucket, dims.numel(),
        indexer.enable_subspace, indexer.l2_subspace, indexer.cmp_ratio,
        indexer.far_method, indexer.near_method, indexer.nt_near_pages,
        indexer.nt_near_len, indexer.args.tia_level1_topk,
        indexer.args.tia_level2_topk, indexer.sink_tokens,
        indexer.sliding_window_size, indexer._sink_pool_lo_blk,
        q_src.dtype, q_src.data_ptr() % 16, s0, s1, s2,
        protected_only,
    )
    previous_geometry = workspace.get('_nfa_launch_geometry')
    if previous_geometry == geometry:
        geometry = previous_geometry
    workspace['_nfa_launch_geometry'] = geometry
    workspace['_nfa_launch_req_ids'] = tuple(req_ids)

    plan_key = ('eager_selector_call_plan', layer_offset)
    call_plan = workspace.get(plan_key)
    if not capturing and call_plan is not None and call_plan[0] is geometry:
        _replay_eager_selector_call_plan(
            workspace, call_plan, q_src, decode_write)
        return
    if not capturing:
        workspace['_nfa_eager_record'] = []

    if protected_only:
        # middle quota 恒为 0：跳过页统计、L1/L2、candidate 与 ranking，
        # 但仍复用 canonical union，保证保护集去重和物理映射一致。
        if decode_write is not None:
            # paged write 由公共 eager replay 入口显式执行，不能录进本层
            # selector 计划，否则旧 K/V 会在当前写入之后再次覆盖同一 slot。
            record = workspace.pop('_nfa_eager_record', None)
            cache.write_decode(layer_offset, *decode_write)
            if record is not None:
                workspace['_nfa_eager_record'] = record
        gather_out = cache.gather_indices_for_layer(layer_offset).unsqueeze(0)
        vis_out = cache.visible_lens_for_layer(layer_offset).unsqueeze(0)
        output_width = gather_out.shape[-1]
        prot_width = min(
            s_cap, int(indexer.sink_tokens + indexer.sliding_window_size))
        aux_width = prot_width
        empty_middle = _workspace_empty(
            workspace, 'empty_middle', (m * hkv, aux_width), torch.int32, device)
        empty_counts = _workspace_empty(
            workspace, 'empty_middle_counts', (m * hkv,), torch.int32, device)
        # bitonic merge 的两路各占 W；空 middle 也必须保留与 aux 等宽
        # 的第一路，不能用 W=1 压缩掉第二路 reversing 所需空间。
        union_width = 2 * aux_width
        _fast_launch(workspace, ('final_union', layer_offset, 'empty'),
                     _final_union_physical_kernel, (m * hkv,),
            empty_middle, empty_counts, lengths, valid_req_t, cache.block_table_buf,
            gather_out, vis_out,
            W=aux_width, PROT=prot_width,
            N=union_width, N_PAD=triton.next_power_of_2(union_width),
            A_PAD=triton.next_power_of_2(prot_width),
            MERGE_BITS=(triton.next_power_of_2(union_width)).bit_length() - 1,
            SINK=int(indexer.sink_tokens), SWA=int(indexer.sliding_window_size),
            CAP=s_cap, MAX_BLOCKS=cache.max_blocks, PAGE=page,
            MAX_REQ=cache.max_requests, MAX_VIS=output_width,
            INVALID=INVALID_GATHER_IDX, HKV=hkv, B=b,
            LAYER_OFFSET=layer_offset, EMPTY=True, num_warps=8,
        )
        if not capturing:
            _store_eager_selector_call_plan(workspace, plan_key, geometry)
        return

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

    def _prepare_q_grouped(dim_idx, d_q):
        out = _workspace_empty(
            workspace, 'q_fine_grouped', (m, hkv, d_q), torch.float32, device)
        _fast_launch(workspace, ('prepare_q_grouped', layer_offset),
                     _prepare_q_grouped_kernel, (m * hkv,),
            q_src, q_req_ids, scales, dim_idx, out, s0, s1, s2,
            HKV=hkv, G=group_size, D=d_full, DQ=d_q, B=b,
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
        fine_dims = torch.arange(d_full, dtype=torch.int64, device=device)
    elif q_dims is not None:
        fine_dims = dims
    else:
        fine_dims = indexer._qat_indices(device).long()
    b_q_fine = _prepare_q_grouped(fine_dims, fine_dims.numel())
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

    # Persistent tensor-only skip flags [layer, request] are read directly via
    # ``valid_req_t``；不再逐层收集 Python data_ptr 或复制列。
    # ---- one launch: in-place paged block stats, no projection -----------
    k_min, k_max, k_sum = _page_stats_all_layers(
        cache, valid_req_t, n_layers, dims, bs, pad,
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

    k_f = min(nb_far, kt_cap)
    k_n = min(nb_near, kt_cap)
    i_f = _workspace_empty(
        workspace, 'topk_far', (m, hkv, k_f), torch.int64, device)
    i_n = _workspace_empty(
        workspace, 'topk_near', (m, hkv, k_n), torch.int64, device)

    # ---- candidate expansion + completion --------------------------------
    cand_idx2, keep2, lower, upper, quota, owners, generations = _candidate_pack_batched(
        i_f, i_n, far_scored, near_scored, lengths, cache, valid_req_t,
        indexer, bs, s_cap, layer_offset=layer_offset, workspace=workspace,
        fuse_topk=True, parallel_expand=capturing)

    # ---- grouped-Q Logits + signed packed-key Top-K ----------------------
    width = 1024
    logits = _fused_fine_scores_paged(
        cache, valid_req_t, n_layers, dims, b_q_fine,
        cand_idx2, keep2, owners, generations, s_cap, lower, upper, quota,
        tile=128, num_warps=2 if capturing else 4,
        layer_offset=layer_offset, workspace=workspace)
    middle = _nfa_filtered_topk(
        logits, cand_idx2, lower, upper, quota, s_cap, width,
        sink=int(indexer.sink_tokens), layer_offset=layer_offset,
        workspace=workspace)

    # ---- final canonical union + physical mapping ------------------------
    # 每个 KV group 独立合并 middle 与保护集，并按真实 request slot 写回。
    gather_out = cache.gather_indices_for_layer(layer_offset).unsqueeze(0)
    vis_out = cache.visible_lens_for_layer(layer_offset).unsqueeze(0)
    output_width = gather_out.shape[-1]
    prot_width = min(
        s_cap, int(indexer.sink_tokens + indexer.sliding_window_size))
    union_width = middle.indices.shape[-1] + prot_width
    _fast_launch(workspace, ('final_union', layer_offset),
                 _final_union_physical_kernel, (m * hkv,),
        middle.indices, middle.counts, lengths, valid_req_t, cache.block_table_buf,
        gather_out, vis_out,
        W=middle.indices.shape[-1], PROT=prot_width,
        N=union_width, N_PAD=triton.next_power_of_2(union_width),
        A_PAD=triton.next_power_of_2(prot_width),
        MERGE_BITS=(triton.next_power_of_2(union_width)).bit_length() - 1,
        SINK=int(indexer.sink_tokens), SWA=int(indexer.sliding_window_size),
        CAP=s_cap, MAX_BLOCKS=cache.max_blocks, PAGE=page,
        MAX_REQ=cache.max_requests, MAX_VIS=output_width,
        INVALID=INVALID_GATHER_IDX, HKV=hkv, B=b,
        LAYER_OFFSET=layer_offset, EMPTY=False, num_warps=8,
    )

    record = workspace.get('_nfa_eager_record')
    if record is not None:
        _store_eager_selector_call_plan(workspace, plan_key, geometry)
    else:
        workspace.pop('_nfa_eager_record', None)
