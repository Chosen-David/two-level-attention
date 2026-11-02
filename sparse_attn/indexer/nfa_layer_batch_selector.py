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
from ..ops.paged_region_kernels_nfa import _protected_selection_batch
from ..indexer.nfa_indexer import _score_coarse_kernel, _score_fine_kernel
from .selection.selector import canonical_selection
from .selection.unique_ranking import rank_unique_batched

def _has_triton() -> bool:
    return triton is not None and tl is not None


class FallbackRequired(Exception):
    """The request-batched selector does not support this geometry."""


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _has_triton():

    @triton.jit
    def _nfa_page_stats_kernel(
        KV_PTRS, BLOCK_TABLE, LENGTHS, SKIP_FAR, SKIP_NEAR, VALID_REQ, DIMS,
        OUT_MIN, OUT_MAX, OUT_SUM,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        SINK: tl.constexpr, NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, D_BLOCK: tl.constexpr, BS: tl.constexpr,
        KT_CAP: tl.constexpr, L: tl.constexpr, B: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        blk = tl.program_id(0)
        row = tl.program_id(1)
        m = row // HK
        kv = row % HK
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        length = tl.load(LENGTHS + req).to(tl.int32)
        skip_far = tl.load(SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        skip_near = tl.load(SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        far_on = skip_far == 0
        near_on = skip_near == 0

        sink_hi = tl.minimum(SINK, length)
        near_hi = tl.maximum(sink_hi, length - WINDOW)
        far_blk = (length - WINDOW - NEAR_LEN) // PAGE
        far_hi = tl.minimum(near_hi, tl.maximum(sink_hi, far_blk * PAGE))

        tok = blk * BS + tl.arange(0, BS)
        visible = tok < sink_hi
        visible = visible | (far_on & (tok >= sink_hi) & (tok < far_hi))
        visible = visible | (near_on & (tok >= far_hi) & (tok < near_hi))
        visible = visible | ((tok >= near_hi) & (tok < length))

        lp = tok // PAGE
        off = tok % PAGE
        page_valid = lp < MAX_BLOCKS
        phys = tl.load(
            BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
            + req_id * MAX_BLOCKS + lp,
            mask=page_valid, other=0).to(tl.int64)

        offs_d = tl.arange(0, D_BLOCK)
        dim_mask = offs_d < DIM_NUMEL
        dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0).to(tl.int64)
        kv_addr = tl.load(KV_PTRS + layer)
        kv_ptr = kv_addr.to(tl.pointer_type(OUT_MIN.dtype.element_ty))
        src = (kv_ptr
               + phys[:, None] * HK * PAGE * HEAD_DIM
               + kv * PAGE * HEAD_DIM
               + off[:, None] * HEAD_DIM
               + dim[None, :])
        load_mask = visible & page_valid
        read_mask = load_mask[:, None] & dim_mask[None, :]
        val = tl.load(src, mask=read_mask, other=0.0).to(tl.float32)

        vmin = tl.min(val, axis=0)
        vmax = tl.max(val, axis=0)
        vsum = tl.sum(val, axis=0)

        out_base = (m * KT_CAP * HK * DIM_NUMEL
                    + blk * HK * DIM_NUMEL
                    + kv * DIM_NUMEL
                    + offs_d)
        tl.store(OUT_MIN + out_base, vmin.to(OUT_MIN.dtype.element_ty), mask=dim_mask)
        tl.store(OUT_MAX + out_base, vmax.to(OUT_MAX.dtype.element_ty), mask=dim_mask)
        tl.store(OUT_SUM + out_base, vsum.to(OUT_SUM.dtype.element_ty), mask=dim_mask)

    @triton.jit
    def _nfa_coarse_score_kernel(
        KMIN, KMAX, Q, LENGTHS, SKIP_FAR, SKIP_NEAR, VALID_REQ, OUT,
        T,
        stride_kt, stride_kh, stride_kt2, stride_kh2,
        stride_qh, stride_oh, stride_ot,
        PAGE: tl.constexpr, BS: tl.constexpr,
        SINK: tl.constexpr, NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        B: tl.constexpr, STRIDE_KB: tl.constexpr, STRIDE_KB2: tl.constexpr,
        MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        """Batched coarse min/max score with per-request dynamic region flags.

        ``pid_h`` enumerates (layer, request, KV head).  Unlike the single-
        request ``_score_coarse_kernel``, skipped far/near regions are masked
        to ``-inf`` inside this kernel so they can never enter L1 Top-K.
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        batch = pid_h // Hkv
        kv = pid_h % Hkv
        layer = batch // B
        req = batch % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        length = tl.load(LENGTHS + req).to(tl.int32)
        skip_far = tl.load(SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        skip_near = tl.load(SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
        far_on = skip_far == 0
        near_on = skip_near == 0

        sink_hi = tl.minimum(SINK, length)
        swa_hi = tl.maximum(sink_hi, length - WINDOW)
        far_blk = (length - WINDOW - NEAR_LEN) // PAGE
        orig_far_hi = tl.minimum(swa_hi, tl.maximum(sink_hi, far_blk * PAGE))
        orig_near_hi = swa_hi

        offs_t = pid_t * BT + tl.arange(0, BT)
        block_lo = offs_t * BS
        block_hi = block_lo + BS
        has_sink = block_lo < sink_hi
        has_far = far_on & (block_hi > sink_hi) & (block_lo < orig_far_hi)
        has_near = near_on & (block_hi > orig_far_hi) & (block_lo < orig_near_hi)
        has_swa = (block_hi > orig_near_hi) & (block_lo < length)
        block_valid = (block_lo < length) & (
            has_sink | has_far | has_near | has_swa)

        KMIN += batch * STRIDE_KB
        KMAX += batch * STRIDE_KB2
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        kmin = tl.load(
            KMIN + offs_t[:, None] * stride_kt + kv * stride_kh
            + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        kmax = tl.load(
            KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2
            + offs_d[None, :],
            mask=mask_t[:, None], other=0.0).to(tl.float32)

        for j in tl.static_range(G):
            h = pid_h.to(tl.int64) * G + j
            q = tl.load(Q + h * stride_qh + offs_d)
            q_low = tl.minimum(q, 0.0)
            q_high = tl.maximum(q, 0.0)
            s = (tl.sum(kmin * q_low[None, :], axis=1)
                 + tl.sum(kmax * q_high[None, :], axis=1))
            s = tl.where(mask_t & block_valid, s, float("-inf"))
            s_o = s.to(OUT.dtype.element_ty)
            tl.store(
                OUT + h * stride_oh + offs_t * stride_ot,
                s_o, mask=mask_t)

    @triton.jit
    def _nfa_fine_paged_kernel(
        KV_PTRS, BLOCK_TABLE, VALID_REQ, DIMS, Q, CAND, CLEN, OUT,
        R, NUM_TILES,
        stride_qm, stride_qh, stride_ch, stride_cr,
        stride_om, stride_or,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
        D_BLOCK: tl.constexpr,
        HKV: tl.constexpr, G: tl.constexpr, B: tl.constexpr,
        TILE: tl.constexpr, MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    ):
        row = tl.program_id(0)
        m = row // HKV
        kv = row % HKV
        layer = m // B
        req = m % B
        req_id = tl.load(VALID_REQ + req).to(tl.int32)

        kv_addr = tl.load(KV_PTRS + layer)
        kv_ptr = kv_addr.to(tl.pointer_type(OUT.dtype.element_ty))
        offs_d = tl.arange(0, D_BLOCK)
        dim_mask = offs_d < DIM_NUMEL
        dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0).to(tl.int64)
        qrow = tl.arange(0, G)

        mstat = tl.full((G,), float("-inf"), tl.float32)
        lstat = tl.zeros((G,), tl.float32)
        cl = tl.load(CLEN + row)

        for t in range(NUM_TILES):
            offs_r = t * TILE + tl.arange(0, TILE)
            mask_r = offs_r < R
            read_r = mask_r & (offs_r < cl)
            cand = tl.load(CAND + row * stride_ch + offs_r * stride_cr,
                           mask=read_r, other=0)
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
            k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + m * stride_qm + (kv * G + j) * stride_qh + offs_d,
                            mask=dim_mask, other=0.0)
                s = tl.sum(k * q[None, :], axis=1)
                s = tl.where(read_r & lp_valid, s, float("-inf"))
                s = s.to(tl.bfloat16).to(tl.float32)
                smax = tl.max(s, axis=0)
                new_m = tl.maximum(mstat, tl.where(qrow == j, smax, float("-inf")))
                corr = tl.exp(mstat - new_m)
                corr = tl.where((mstat == float("-inf")) & (new_m == float("-inf")),
                                1.0, corr)
                m_j = tl.sum(tl.where(qrow == j, new_m, 0.0), axis=0)
                l_j_update = tl.sum(tl.exp(s - m_j), axis=0)
                lstat = lstat * corr + tl.where(qrow == j, l_j_update, 0.0)
                mstat = new_m

        for t in range(NUM_TILES):
            offs_r = t * TILE + tl.arange(0, TILE)
            mask_r = offs_r < R
            read_r = mask_r & (offs_r < cl)
            cand = tl.load(CAND + row * stride_ch + offs_r * stride_cr,
                           mask=read_r, other=0)
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
            k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            prob_sum = tl.zeros((TILE,), tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + m * stride_qm + (kv * G + j) * stride_qh + offs_d,
                            mask=dim_mask, other=0.0)
                s = tl.sum(k * q[None, :], axis=1)
                s = tl.where(read_r & lp_valid, s, float("-inf"))
                s = s.to(tl.bfloat16).to(tl.float32)
                m_j = tl.sum(tl.where(qrow == j, mstat, 0.0), axis=0)
                l_j = tl.sum(tl.where(qrow == j, lstat, 0.0), axis=0)
                safe_l = tl.where(l_j > 0.0, l_j, 1.0)
                head_p = tl.exp(s - m_j) / safe_l
                head_p = tl.where(l_j > 0.0, head_p, 0.0)
                prob_sum += head_p.to(tl.bfloat16).to(tl.float32)
            out = prob_sum * (1.0 / G)
            tl.store(OUT + row * stride_om + offs_r * stride_or,
                     out.to(OUT.dtype.element_ty), mask=mask_r)

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
        pid = tl.program_id(0)
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
    def _complete_pack_batched_kernel(
        IDS, KEEP, LENGTHS, OUT, VALID, COUNTS,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
        HKV: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        m = row // HKV
        h = row % HKV
        length = tl.load(LENGTHS + m)
        lo = tl.maximum(0, length - SWA)
        width = R + HEAD + TAIL

        total = tl.full((), 0, tl.int32)
        for start in range(0, width, BLOCK):
            r = start + tl.arange(0, BLOCK)
            active = r < width
            token, active = _layer_batch_virtual_slot(
                IDS, KEEP, row, r, length, R, HEAD, TAIL, SINK, SWA, active)
            total += tl.sum(((r < width) & active).to(tl.int32), 0)

        carry = tl.full((), 0, tl.int32)
        for start in range(0, width, BLOCK):
            r = start + tl.arange(0, BLOCK)
            active = r < width
            token, active = _layer_batch_virtual_slot(
                IDS, KEEP, row, r, length, R, HEAD, TAIL, SINK, SWA, active)
            active = (r < width) & active
            prefix = tl.cumsum(active.to(tl.int32), 0)
            destination = tl.where(active, carry + prefix - 1,
                                   total + r - carry - prefix)
            tl.store(OUT + row * width + destination,
                     tl.where(active, token, CAP - 1), r < width)
            tl.store(VALID + row * width + destination, active, r < width)
            carry += tl.sum(active.to(tl.int32), 0)
        tl.store(COUNTS + row, total.to(tl.int64))

    @triton.jit
    def _layer_batch_virtual_slot(
        IDS, KEEP, row, r, length,
        R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
        SINK: tl.constexpr, SWA: tl.constexpr, live,
    ):
        lo = tl.maximum(0, length - SWA)
        old = r < R
        token = tl.load(IDS + row * R + r, mask=old, other=0)
        keep = tl.load(KEEP + row * R + r, mask=old, other=False)
        active = keep & (token >= SINK) & (token < lo) & (token >= 0)
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
    def _candidate_owner_batched_kernel(
        IDS, KEEP, OWNERS,
        R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        ids = tl.load(IDS + row * R + r, mask=mask, other=0)
        keep = tl.load(KEEP + row * R + r, mask=mask, other=False)
        valid = mask & keep & (ids >= 0) & (ids < S)
        tl.atomic_max(OWNERS + row * S + ids, r, mask=valid, sem="relaxed")

    @triton.jit
    def _owner_validity_batched_kernel(
        IDS, KEEP, OWNERS, VALID, UNIQUE,
        R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        ids = tl.load(IDS + row * R + r, mask=mask, other=0)
        keep = tl.load(KEEP + row * R + r, mask=mask, other=False)
        valid = mask & keep & (ids >= 0) & (ids < S)
        owner = tl.load(OWNERS + row * S + ids, mask=valid, other=-1)
        unique = valid & (owner == r)
        tl.store(VALID + row * R + r, valid, mask=mask)
        tl.store(UNIQUE + row * R + r, unique, mask=mask)

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
    def _topk_indices_kernel(
        SCORES, OUT,
        KT: tl.constexpr, KT_PAD: tl.constexpr,
        K: tl.constexpr, K_PAD: tl.constexpr,
    ):
        """Per-row Top-K over ``KT`` packed 16-bit scores.

        The packed key encodes the float total order in the high 16 bits
        (monotonic with ``(score_key << 16)``) and ``KT - 1 - index`` in the
        low 16 bits, so a single descending ``tl.sort`` reproduces
        ``torch.topk`` semantics (largest score first, smallest index wins
        ties).  Masked ``-inf`` entries naturally sort to the tail.
        """
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

# ---------------------------------------------------------------------------
# Host-side batched helpers
# ---------------------------------------------------------------------------

def _scalar(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(int(x), device=device, dtype=dtype)


def _topk_indices(scores, k):
    """Fused Triton replacement for ``torch.topk(..., dim=-1).indices``.

    ``scores`` is a contiguous ``[M, HKV, KT]`` 16-bit tensor and the result is
    ``[M, HKV, K]`` int64, matching the old ``torch.topk`` output dtype so the
    downstream ``torch.gather`` masks keep working unchanged.
    """
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    m, hkv, kt_cap = scores.shape
    k = min(int(k), kt_cap)
    if k <= 0:
        return torch.empty((m, hkv, 0), dtype=torch.int64, device=scores.device)
    if kt_cap == 1:
        # Triton 3.4 的 tl.sort 对单元素张量会触发 reshape 编译错误；
        # 单候选 Top-1 的索引恒为 0，无需进入排序 kernel。
        return torch.zeros((m, hkv, 1), dtype=torch.int64, device=scores.device)
    kt_pad = triton.next_power_of_2(kt_cap)
    k_pad = triton.next_power_of_2(k)
    out = torch.empty((m, hkv, k), dtype=torch.int64, device=scores.device)
    _topk_indices_kernel[(m * hkv,)](
        scores.reshape(m * hkv, kt_cap), out,
        KT=kt_cap, KT_PAD=kt_pad, K=k, K_PAD=k_pad,
        num_warps=8,
    )
    return out


def _page_stats_all_layers(cache, req_ids, layer_ptrs, dims, bs, pad, lengths,
                           layer_offset=0):
    """Compute per-logical-block min/max/sum directly from paged KV.

    No intermediate [L,B,pad,HK,D] projection is materialized.  Each
    program handles one logical token block for one (layer, request, KV
    head) and reduces inside registers, mirroring the old projected
    ``amin/amax/mean`` contract exactly.
    """
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

    valid_req_t = torch.as_tensor(req_ids, dtype=torch.int64, device=device)
    block_table = cache.block_table_buf
    lengths_t = torch.as_tensor(lengths, dtype=torch.int32, device=device)
    skip_far_t = cache.skip_far_layer_buf
    skip_near_t = cache.skip_near_layer_buf
    dims_t = dims
    kv_ptrs = cache.layer_kv_ptrs_buf[layer_offset:layer_offset + l]

    out_dtype = layer0.kv_pool_k.dtype
    k_min = torch.empty(
        (m, kt_cap, hk, dim_numel), dtype=out_dtype, device=device)
    k_max = torch.empty_like(k_min)
    k_sum = torch.empty_like(k_min)

    _nfa_page_stats_kernel[(kt_cap, m * hk)](
        kv_ptrs, block_table, lengths_t, skip_far_t, skip_near_t, valid_req_t,
        dims_t,
        k_min, k_max, k_sum,
        MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        SINK=cache.geometry.sink, NEAR_LEN=cache.geometry.near,
        WINDOW=cache.geometry.window,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dim_numel,
        D_BLOCK=triton.next_power_of_2(dim_numel), BS=bs,
        KT_CAP=kt_cap, L=l, B=b,
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        num_warps=4,
    )
    k_avg = (k_sum.to(torch.float32) / bs).to(out_dtype)
    return k_min, k_max, k_avg


def _fused_fine_probabilities_paged(
        cache, req_ids, layer_ptrs, dims, q_fine, cand_idx2, keep2,
        cand_len, capacity, group_size, tile=64, num_warps=16,
        layer_offset=0):
    """Candidate gather -> paged K load -> online softmax -> GQA mean.

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
    if hkv != hk:
        raise FallbackRequired("KV-head geometry mismatch")

    valid_req_t = torch.as_tensor(req_ids, dtype=torch.int64, device=device)
    block_table = cache.block_table_buf
    dims_t = dims
    kv_ptrs = cache.layer_kv_ptrs_buf[
        layer_offset:layer_offset + len(layer_ptrs)]

    owners = torch.full((m, hkv, capacity), -1, dtype=torch.int32, device=device)
    valid = torch.empty_like(keep2)
    unique = torch.empty_like(keep2)
    probs = torch.empty((m, hkv, r), dtype=layer0.kv_pool_k.dtype, device=device)

    block = 256
    grid_owner = (m * hkv, triton.cdiv(r, block))
    grid_prob = (m * hkv,)
    num_tiles = triton.cdiv(r, tile)

    _candidate_owner_batched_kernel[grid_owner](
        cand_idx2, keep2, owners, R=r, S=capacity, BLOCK=block)
    _owner_validity_batched_kernel[grid_owner](
        cand_idx2, keep2, owners, valid, unique, R=r, S=capacity, BLOCK=block)
    _nfa_fine_paged_kernel[grid_prob](
        kv_ptrs, block_table, valid_req_t, dims_t, q_fine, cand_idx2,
        cand_len, probs,
        R=r, NUM_TILES=num_tiles,
        stride_qm=q_fine.stride(0), stride_qh=q_fine.stride(1),
        stride_ch=cand_idx2.stride(1), stride_cr=cand_idx2.stride(2),
        stride_om=probs.stride(1), stride_or=probs.stride(2),
        MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dims.numel(),
        D_BLOCK=triton.next_power_of_2(dims.numel()),
        HKV=hkv, G=group_size, B=b, TILE=tile,
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        num_warps=num_warps,
    )
    return probs, unique


def _batched_geometry(cache, req_ids, lengths, indexer, device, kt_cap, s_cap):
    bs = int(indexer.args.tia_block_size)
    sink = int(indexer.sink_tokens)
    swa = int(indexer.sliding_window_size)
    nl = max(bs, int(indexer.nt_near_len))
    sink_pool_lo = int(indexer._sink_pool_lo_blk)
    k2 = int(indexer.args.tia_level2_topk)

    s_real = torch.as_tensor(lengths, dtype=torch.long, device=device)
    b = s_real.numel()
    swa_lo_tok = torch.clamp(s_real - swa, min=0)
    kt_eff = (s_real + bs - 1) // bs
    near_blks = torch.clamp((s_real - swa - nl) // bs, min=sink_pool_lo)
    swa_lo_blk = torch.maximum(near_blks, kt_eff - max(1, swa // bs))
    swa_byp_blk = swa_lo_tok // bs
    swa_byp_cond = (swa_lo_tok % bs != 0) & (swa_byp_blk >= sink_pool_lo)
    if sink % bs != 0:
        swa_byp_cond = swa_byp_cond & (swa_byp_blk != sink_pool_lo)

    ar_blk = torch.arange(kt_cap, dtype=torch.long, device=device)
    far_pool_mask = (ar_blk[None] >= sink_pool_lo) & (ar_blk[None] < near_blks[:, None])
    near_pool_mask = (ar_blk[None] >= near_blks[:, None]) & (ar_blk[None] < swa_lo_blk[:, None])
    if sink % bs != 0:
        far_pool_mask = far_pool_mask & (ar_blk[None] != sink_pool_lo)
    byp_excl = (ar_blk[None] == swa_byp_blk[:, None]) & swa_byp_cond[:, None]
    near_pool_mask = near_pool_mask & ~byp_excl
    if sink % bs != 0:
        sink_in_near = ((ar_blk[None] == sink_pool_lo)
                        & (ar_blk[None] >= near_blks[:, None])
                        & (ar_blk[None] < swa_lo_blk[:, None]))
        near_pool_mask = near_pool_mask & ~sink_in_near

    ar_tok = torch.arange(s_cap, dtype=torch.long, device=device)
    far_tok_lo = sink_pool_lo * bs
    mid_mask = (ar_tok[None] >= far_tok_lo) & (ar_tok[None] < swa_lo_tok[:, None])
    swa_force = (ar_tok[None] >= swa_lo_tok[:, None]) & (ar_tok[None] < s_real[:, None])
    k2_mid = torch.clamp(torch.clamp(s_real, max=k2) - sink - (s_real - swa_lo_tok), min=0)

    ar_bs = torch.arange(bs, dtype=torch.long, device=device)
    sink_cross = torch.full(
        (), sink % bs != 0, dtype=torch.bool, device=device)
    slot0 = s_real.new_full((), sink_pool_lo if sink % bs != 0 else 0)
    byp_blk = torch.stack([slot0.expand(b), swa_byp_blk], dim=1)  # [B,2]
    byp_tok = byp_blk[:, :, None] * bs + ar_bs[None, None, :]
    byp_keep = byp_tok < s_real[:, None, None]
    byp_keep = byp_keep & torch.stack(
        [sink_cross.expand(b), swa_byp_cond], dim=1)[:, :, None]

    return {
        "s_real": s_real,
        "swa_lo_tok": swa_lo_tok,
        "far_pool_mask": far_pool_mask,
        "near_pool_mask": near_pool_mask,
        "mid_mask": mid_mask,
        "swa_force": swa_force,
        "k2_mid": k2_mid,
        "far_tok_lo": s_real.new_full((), far_tok_lo),
        "byp_tok": byp_tok,
        "byp_keep": byp_keep,
    }


def _coarse_scores(k_min, k_max, k_avg, b_q_full, group_size, indexer, dtype,
                   cache=None, valid_req=None, lengths=None,
                   page=None, bs=None, sink=None, near_len=None, window=None,
                   layer_offset=0):
    # One launch per coarse feature instead of a Python row loop.  The new
    # batched kernel reads per-request skip flags and masks skipped far/near
    # blocks to -inf before L1 Top-K, so skip policies stay tensor-only.
    m, kt_cap, hkv, d = k_min.shape
    hq = b_q_full.shape[1]
    q_flat = b_q_full.reshape(m * hq, d).contiguous()
    coarse = torch.empty((m, hq, kt_cap), dtype=dtype, device=k_min.device)
    bt, nw = indexer._TILE_COARSE.get(d, (64, 4))
    lengths_t = torch.as_tensor(lengths, dtype=torch.int32, device=k_min.device)
    skip_far_t = cache.skip_far_layer_buf
    skip_near_t = cache.skip_near_layer_buf
    b = lengths.numel() if torch.is_tensor(lengths) else len(lengths)
    _nfa_coarse_score_kernel[(triton.cdiv(kt_cap, bt), m * hkv)](
        k_min, k_max, q_flat, lengths_t, skip_far_t, skip_near_t, valid_req,
        coarse,
        kt_cap,
        k_min.stride(1), k_min.stride(2),
        k_max.stride(1), k_max.stride(2),
        q_flat.stride(0), coarse.stride(1), coarse.stride(2),
        PAGE=page, BS=bs, SINK=sink, NEAR_LEN=near_len, WINDOW=window,
        Hkv=hkv, G=group_size, D=d, BT=bt, B=b,
        STRIDE_KB=k_min.stride(0), STRIDE_KB2=k_max.stride(0),
        MAX_REQ=cache.max_requests, LAYER_OFFSET=layer_offset,
        num_warps=nw,
    )
    sc = coarse.view(m, hkv, group_size, kt_cap).mean(dim=2)

    if k_avg is None:
        return sc, None

    fine = torch.empty_like(coarse)
    bt_f, nw_f = indexer._TILE_FINE.get(d, (64, 4))
    _score_fine_kernel[(triton.cdiv(kt_cap, bt_f), m * hkv)](
        k_avg, q_flat, fine, fine,
        kt_cap,
        k_avg.stride(1), k_avg.stride(2), q_flat.stride(0),
        fine.stride(1), fine.stride(2),
        fine.stride(1), fine.stride(2),
        Hkv=hkv, G=group_size, D=d, BT=bt_f, HAS_MEAN=False,
        BATCHED=True, STRIDE_KB=k_avg.stride(0),
        num_warps=nw_f,
    )
    sc_avg = fine.view(m, hkv, group_size, kt_cap).mean(dim=2)
    # Mask the mean path with the same tensor-only region visibility used by
    # the min/max kernel.  This is especially important when far_method=avg.
    valid_blocks = _region_block_mask(
        lengths, cache, valid_req, m, kt_cap, page, bs,
        sink, near_len, window, k_min.device, layer_offset=layer_offset)
    sc_avg = torch.where(
        valid_blocks.unsqueeze(1).expand(m, hkv, kt_cap),
        sc_avg, torch.full_like(sc_avg, float("-inf")))
    return sc, sc_avg


def _region_block_mask(lengths, cache, valid_req, m, kt_cap, page, bs,
                       sink, near_len, window, device, layer_offset=0):
    """[m, kt_cap] boolean mask for blocks with at least one visible token."""
    lengths_t = torch.as_tensor(lengths, dtype=torch.int64, device=device)
    b = lengths_t.numel()
    n_layers = m // b
    req_idx = torch.arange(b, dtype=torch.int64, device=device).repeat(n_layers)
    layer_idx = (torch.arange(n_layers, dtype=torch.int64, device=device)
                 .repeat_interleave(b) + layer_offset)
    rid = valid_req[req_idx]
    skip_far_t = cache.skip_far_layer_buf[layer_idx, rid].to(torch.int64)
    skip_near_t = cache.skip_near_layer_buf[layer_idx, rid].to(torch.int64)
    ar_blk = torch.arange(kt_cap, dtype=torch.int64, device=device)
    block_lo = ar_blk[None, :] * bs
    block_hi = block_lo + bs
    length = lengths_t.repeat(n_layers)[:, None]
    sink_hi = torch.minimum(lengths_t.new_full((), sink), length)
    swa_hi = torch.maximum(sink_hi, length - window)
    far_blk = ((length - window - near_len).clamp(min=0) // page) * page
    orig_far_hi = torch.minimum(swa_hi, torch.maximum(sink_hi, far_blk))
    orig_near_hi = swa_hi
    far_vis = (skip_far_t[:, None] == 0) & (block_hi > sink_hi) & (block_lo < orig_far_hi)
    near_vis = (skip_near_t[:, None] == 0) & (block_hi > orig_far_hi) & (block_lo < orig_near_hi)
    valid = (block_lo < length) & (
        (block_lo < sink_hi)
        | far_vis
        | near_vis
        | ((block_hi > orig_near_hi) & (block_lo < length))
    )
    return valid


def _candidate_assembly(i_f, i_n, geo, req_idx, m, hkv, bs, device,
                        far_mask, near_mask):
    ar_bs = torch.arange(bs, dtype=torch.long, device=device)
    s_real_m = geo["s_real"][req_idx]
    k1 = i_f.shape[-1] + i_n.shape[-1]

    blk = torch.cat([i_f, i_n], dim=-1)
    ok = torch.cat([
        torch.gather(far_mask, -1, i_f),
        torch.gather(near_mask, -1, i_n),
    ], dim=-1)
    tok = blk.unsqueeze(-1) * bs + ar_bs
    kp = (tok < s_real_m[:, None, None, None]) & ok.unsqueeze(-1)
    l1_ids = tok.reshape(m, hkv, k1 * bs)
    l1_keep = kp.reshape(m, hkv, k1 * bs)

    byp_tok = geo["byp_tok"][req_idx].unsqueeze(1).expand(m, hkv, -1, -1).reshape(m, hkv, -1)
    byp_keep = geo["byp_keep"][req_idx].unsqueeze(1).expand(m, hkv, -1, -1).reshape(m, hkv, -1)
    cand_idx2 = torch.cat([l1_ids, byp_tok], dim=-1).to(torch.int64)
    keep2 = torch.cat([l1_keep, byp_keep], dim=-1).to(torch.bool)
    # selection_mask for non-skip layers is token < length.
    keep2 = keep2 & (cand_idx2 >= 0) & (cand_idx2 < s_real_m[:, None, None])
    cand_idx2 = torch.where(keep2, cand_idx2, torch.zeros_like(cand_idx2))
    return cand_idx2.contiguous(), keep2.contiguous()


def _complete_pack_batched(cand_idx2, keep2, lengths, sink, swa, capacity):
    m, hkv, r = cand_idx2.shape
    head = min(sink, capacity)
    tail = min(swa, capacity)
    width = r + head + tail
    out = torch.empty((m, hkv, width), device=cand_idx2.device, dtype=cand_idx2.dtype)
    valid = torch.empty((m, hkv, width), device=cand_idx2.device, dtype=torch.bool)
    counts = torch.empty((m, hkv), device=cand_idx2.device, dtype=torch.int64)
    lengths_t = torch.as_tensor(lengths, dtype=torch.int64, device=cand_idx2.device)
    _complete_pack_batched_kernel[(m * hkv,)](
        cand_idx2, keep2, lengths_t, out, valid, counts,
        R=r, HEAD=head, TAIL=tail, SINK=sink, SWA=swa, CAP=capacity,
        HKV=hkv, BLOCK=1024, num_warps=4)
    return out, valid, counts


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def select_topk_for_layer_batched(cache, layer_idx, req_ids,
                                  query_states, attention=None):
    """单层 Req-Batched 当前 query 选择（生产唯一入口）。

    ``query_states`` 必须是本层当前 token 的 post-RoPE Q，形状 ``[B,HQ,D]``。
    输出直接写入当前 visible bucket 的 batch 行，支持非连续 request slot。
    """
    if not _has_triton():
        raise FallbackRequired("Triton unavailable")
    if query_states is None:
        raise ValueError('production selection requires current post-RoPE query')
    return _layer_batched_select_impl(
        cache, req_ids, layer_indices=[layer_idx],
        query_states=query_states, attention=attention)


def _layer_batched_select_impl(cache, req_ids, layer_indices,
                               query_states, attention):
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
    indexer = attention.indexer
    if not hasattr(indexer, "args") or indexer.args is None:
        raise FallbackRequired("missing indexer config")

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

    if indexer.enable_subspace:
        dims = indexer._subspace_indices(device).long()
    else:
        # _subspace_indices also supplies the legacy tail/QAT coordinates;
        # disabling L1 projection must instead read full-width K and Q.
        dims = torch.arange(cache.layers[0].head_dim_k,
                            dtype=torch.int64, device=device)

    # PAD 行使用 request 0 的安全页表地址，但 length=0 且不消费 seed。
    active_req_t = cache.active_req_ids_buf[:b]
    valid_req_t = torch.where(
        active_req_t >= 0, active_req_t,
        torch.zeros_like(active_req_t))
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
    q_req_ids = torch.arange(b, dtype=torch.int64, device=device)
    d_full = q_src.shape[-1]
    s0, s1, s2 = q_src.stride(0), q_src.stride(1), q_src.stride(2)

    def _prepare_q(dim_idx, d_q):
        out = torch.empty((m, hq, d_q), dtype=torch.float32, device=device)
        _prepare_q_kernel[(m * hq,)](
            q_src, q_req_ids, scales, dim_idx, out,
            s0, s1, s2,
            HQ=hq, D=d_full, DQ=d_q, B=b,
            DQ_BLOCK=triton.next_power_of_2(d_q), num_warps=4)
        return out

    b_q_full = _prepare_q(dims, dims.numel())
    if indexer.l2_subspace == "full":
        fine_dims = (torch.arange(d_full, dtype=torch.int64, device=device)
                     if indexer.enable_subspace else dims)
    elif indexer.enable_subspace:
        fine_dims = dims
    else:
        fine_dims = indexer._qat_indices(device).long()
    # Each stage uses the same ordered coordinates for K and Q. Reuse the
    # current query workspace only when the coordinate object is shared.
    b_q_fine = (b_q_full if fine_dims is dims else
                _prepare_q(fine_dims, fine_dims.numel()))

    # Layer pointers and persistent tensor-only skip flags [layer, request].
    # Skip/block-table lookups now read the full persistent buffers directly
    # via ``valid_req_t``, so no ``.contiguous()`` column copy is materialized.
    layer_ptrs = [cache.layers[li].kv_pool_k.data_ptr() for li in layer_indices]

    # ---- one launch: in-place paged block stats, no projection -----------
    k_min, k_max, k_avg = _page_stats_all_layers(
        cache, valid_req_t, layer_ptrs, dims, bs, pad,
        lengths, layer_offset=layer_offset)  # each [M,kt_cap,Hkv,d]

    # ---- coarse scores ---------------------------------------------------
    b_q = b_q_full.reshape(m, hkv, group_size, dims.numel())
    sc, sc_avg = _coarse_scores(
        k_min, k_max, k_avg, b_q_full, group_size, indexer,
        q_src.dtype, cache=cache, valid_req=valid_req_t,
        lengths=lengths, page=page, bs=bs, sink=cache.geometry.sink,
        near_len=cache.geometry.near, window=cache.geometry.window,
        layer_offset=layer_offset)

    geo = _batched_geometry(cache, valid_req_t, lengths, indexer, device, kt_cap, s_cap)
    req_idx = torch.arange(b, dtype=torch.long, device=device).repeat(n_layers)
    layer_idx = torch.arange(n_layers, dtype=torch.long, device=device).repeat_interleave(b) + layer_offset
    far_pool_m = geo["far_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)
    near_pool_m = geo["near_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)
    rid = valid_req_t[req_idx]
    far_on_m = (cache.skip_far_layer_buf[layer_idx, rid] == 0)[:, None, None]
    near_on_m = (cache.skip_near_layer_buf[layer_idx, rid] == 0)[:, None, None]
    far_pool_m = far_pool_m & far_on_m
    near_pool_m = near_pool_m & near_on_m

    nb_near = max(0, int(indexer.nt_near_pages))
    k1 = int(indexer.args.tia_level1_topk)
    nb_far = max(0, k1 - nb_near)
    k1_eff = nb_far + nb_near

    far_src = sc_avg if (indexer.far_method == "avg" and sc_avg is not None) else sc
    near_src = sc if (indexer.near_method == "minmax" or sc_avg is None) else sc_avg
    neg_inf = torch.full((), float("-inf"), dtype=far_src.dtype, device=device)
    far_scored = torch.where(far_pool_m, far_src, neg_inf)
    near_scored = torch.where(near_pool_m, near_src, neg_inf)
    i_f = _topk_indices(far_scored, min(nb_far, kt_cap))
    i_n = _topk_indices(near_scored, min(nb_near, kt_cap))

    cand_idx2, keep2 = _candidate_assembly(
        i_f, i_n, geo, req_idx, m, hkv, bs, device,
        far_pool_m, near_pool_m)

    # ---- candidate-domain completion -------------------------------------
    cand_idx2, keep2, cand_len = _complete_pack_batched(
        cand_idx2, keep2, lengths.repeat(n_layers),
        int(indexer.sink_tokens), int(indexer.sliding_window_size), s_cap)

    # ---- fused fine score + local unique topk ----------------------------
    probs, unique = _fused_fine_probabilities_paged(
        cache, valid_req_t, layer_ptrs, fine_dims, b_q_fine,
        cand_idx2, keep2, cand_len, s_cap, group_size,
        layer_offset=layer_offset)

    lower = geo["far_tok_lo"].expand(m).contiguous()
    upper = torch.maximum(lower, geo["swa_lo_tok"][req_idx])
    quota = geo["k2_mid"][req_idx]
    # rank_unique 在候选不足时会从 [lower, upper) 补齐 K2。skip far/near
    # 后仍使用完整 mid 区间会把已跳过 token 重新选回；far+near 本身连续，
    # 因此分别收缩左/右边界即可同时约束候选排名与空位补齐。
    length_m = geo["s_real"][req_idx]
    sink_hi = length_m.clamp(max=cache.geometry.sink)
    swa_hi = torch.maximum(sink_hi, length_m-cache.geometry.window)
    far_blk = ((length_m-cache.geometry.window-cache.geometry.near)
               .clamp(min=0) // cache.geometry.page) * cache.geometry.page
    far_hi = torch.minimum(swa_hi, torch.maximum(sink_hi, far_blk))
    skip_far_m = cache.skip_far_layer_buf[layer_idx, rid] != 0
    skip_near_m = cache.skip_near_layer_buf[layer_idx, rid] != 0
    lower = torch.where(skip_far_m, far_hi, lower)
    upper = torch.where(skip_near_m, far_hi, upper)
    upper = torch.maximum(lower, upper)
    quota = torch.minimum(quota, upper-lower)
    width = 1024
    if cand_idx2.shape[-1] < width:
        # 短 prompt 的 L1 候选数可小于固定 ranking 宽度；补无效项不改变
        # Top-K 语义，也避免 max_new_tokens=1/首 token 等短序列被误拒绝。
        pad_width = width - cand_idx2.shape[-1]
        cand_idx2 = torch.cat([
            cand_idx2,
            torch.zeros((*cand_idx2.shape[:-1], pad_width),
                        dtype=cand_idx2.dtype, device=device)], dim=-1)
        probs = torch.cat([
            probs,
            torch.zeros((*probs.shape[:-1], pad_width),
                        dtype=probs.dtype, device=device)], dim=-1)
        unique = torch.cat([
            unique,
            torch.zeros((*unique.shape[:-1], pad_width),
                        dtype=unique.dtype, device=device)], dim=-1)
    middle = rank_unique_batched(
        cand_idx2, probs, unique, lower, upper, quota,
        capacity=s_cap, width=width, sink=int(indexer.sink_tokens))

    # ---- protected sink + SWA merge --------------------------------------
    protected_ids, protected_counts = _protected_selection_batch(
        lengths, indexer, s_cap)
    prot_width = protected_ids.shape[-1]
    if prot_width > 0:
        prot = protected_ids.unsqueeze(0).expand(n_layers, -1, -1).reshape(m, -1)
        prot_valid = (
            torch.arange(prot_width, device=device)[None, None, :]
            < protected_counts.unsqueeze(0).unsqueeze(-1)
        ).expand(n_layers, -1, -1).reshape(m, -1)
        prot = prot.unsqueeze(1).expand(m, hkv, -1).to(torch.int32)
        prot_valid = prot_valid.unsqueeze(1).expand(m, hkv, -1)
    else:
        prot = torch.empty((m, hkv, 0), dtype=torch.int32, device=device)
        prot_valid = torch.empty((m, hkv, 0), dtype=torch.bool, device=device)

    # ---- first-decode prefill seed merge ---------------------------------
    # Seed IDs are merged AFTER top-k, preserving the legacy semantics of
    # ``selection ∪ prefill_seed``.  They never re-enter the candidate pool.
    seed_needed = ((cache.first_decode_seeded_buf[valid_req_t] == 0)
                   & (active_req_t >= 0))  # [B]，PAD 行不消费 request 0 的 seed
    seed_ids = cache.prefill_seed_ids_buf[
        layer_offset:layer_offset + n_layers][:, valid_req_t]  # [L,B,S]
    seed_counts = cache.prefill_seed_counts_buf[
        layer_offset:layer_offset + n_layers][:, valid_req_t]  # [L,B]
    max_seed = seed_ids.shape[-1]
    seed_ids = seed_ids.reshape(m, max_seed)
    seed_counts = seed_counts.reshape(m)
    seed_valid = (
        torch.arange(max_seed, device=device)[None, :] < seed_counts[:, None]
    ) & seed_needed.repeat(n_layers)[:, None]
    seed = seed_ids.unsqueeze(1).expand(m, hkv, -1).to(torch.int32)
    seed_valid = seed_valid.unsqueeze(1).expand(m, hkv, -1)

    middle_valid = (
        torch.arange(middle.indices.shape[-1], device=device)[None, None, :]
        < middle.counts[:, :, None])
    ids = torch.cat([middle.indices.to(torch.int32), prot, seed], dim=-1)
    valid = torch.cat([middle_valid, prot_valid, seed_valid], dim=-1)
    final = canonical_selection(ids, valid, s_cap)

    # ---- physical mapping for request-level (head 0) selection -----------
    # One Triton launch converts the canonical head-0 selection to absolute
    # physical slots and writes both persistent buffers in-place, removing the
    # prior ``contiguous/gather/where/full/index_copy_`` chain.
    width_final = final.indices.shape[-1]
    # 层内选择直接写 Graph 当前 bucket，attention 随后读取同一地址。
    gather_out = cache.gather_indices_for_layer(layer_offset).unsqueeze(0)
    vis_out = cache.visible_lens_for_layer(layer_offset).unsqueeze(0)
    output_width = gather_out.shape[-1]
    vis_out.zero_()
    _final_physical_map_kernel[
        (m, triton.cdiv(width_final, 256))
    ](
        final.indices, final.counts, cache.block_table_buf, valid_req_t,
        gather_out, vis_out,
        W=width_final, MAX_BLOCKS=cache.max_blocks, PAGE=page,
        MAX_REQ=cache.max_requests, MAX_VIS=output_width,
        INVALID=INVALID_GATHER_IDX, HKV=hkv, B=b, BLOCK_W=256,
        LAYER_OFFSET=layer_offset,
        num_warps=4,
    )



def mark_first_decode_seeded(cache, req_ids):
    """Consume prefill seeds once, after all layers completed selection."""
    valid_req_ids = [
        rid for rid in req_ids
        if rid is not None and rid >= 0 and rid in cache.states
    ]
    if not valid_req_ids:
        return
    device = cache.first_decode_seeded_buf.device
    valid_req_t = torch.as_tensor(valid_req_ids, dtype=torch.int64, device=device)
    cache.first_decode_seeded_buf[valid_req_t] = 1
    for rid in valid_req_ids:
        cache.states[rid].first_decode_seeded = True

