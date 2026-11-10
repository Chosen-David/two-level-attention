"""GraphReady 固定物理页写入与选中词元 gather attention。"""
import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _paged_write_kernel(
        KEYS, VALUES, SLOTS, K_POOL, V_POOL,
        B,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        PAD_SLOT_SAFE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
    ):
        b = tl.program_id(0)
        if b >= B:
            return
        slot = tl.load(SLOTS + b)
        if slot < 0:
            return
        safe = tl.where(slot >= 0, slot, PAD_SLOT_SAFE)
        phys = (safe // PAGE).to(tl.int64)
        off = (safe % PAGE).to(tl.int64)

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        for h in tl.static_range(HK):
            key = tl.load(
                KEYS + b * HK * D + h * D + d,
                mask=d < D, other=0.0)
            k_ptr = (
                K_POOL
                + phys * (HK * PAGE * D)
                + h * (PAGE * D)
                + off * D
                + d
            )
            tl.store(k_ptr, key, mask=d < D)

            val = tl.load(
                VALUES + b * HK * DV + h * DV + vd,
                mask=vd < DV, other=0.0)
            v_ptr = (
                V_POOL
                + phys * (HK * PAGE * DV)
                + h * (PAGE * DV)
                + off * DV
                + vd
            )
            tl.store(v_ptr, val, mask=vd < DV)


    @triton.jit
    def _paged_write_prefill_kernel(
        KEYS, VALUES, BLOCK_TABLE, K_POOL, V_POOL, N,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
    ):
        # 一个 program 搬运单个 KV head 的二维 token/dim tile；同一 tile
        # 可以跨逻辑页，物理页号逐 token 查表，尾部仅由 N mask。
        tile = tl.program_id(0)
        h = tl.program_id(1)
        token = tile * BT + tl.arange(0, BT)
        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        token_mask = token < N
        logical_page = token // PAGE
        offset = token % PAGE
        physical = tl.load(
            BLOCK_TABLE + logical_page, mask=token_mask, other=0).to(tl.int64)
        token64 = token.to(tl.int64)

        key = tl.load(
            KEYS + h * N * D + token64[:, None] * D + d[None, :],
            mask=token_mask[:, None] & (d[None, :] < D), other=0.0)
        k_ptr = (
            K_POOL + physical[:, None] * (HK * PAGE * D)
            + h * (PAGE * D) + offset[:, None] * D + d[None, :])
        tl.store(k_ptr, key, mask=token_mask[:, None] & (d[None, :] < D))

        value = tl.load(
            VALUES + h * N * DV + token64[:, None] * DV + vd[None, :],
            mask=token_mask[:, None] & (vd[None, :] < DV), other=0.0)
        v_ptr = (
            V_POOL + physical[:, None] * (HK * PAGE * DV)
            + h * (PAGE * DV) + offset[:, None] * DV + vd[None, :])
        tl.store(v_ptr, value, mask=token_mask[:, None] & (vd[None, :] < DV))


    @triton.jit
    def _gather_attend_kernel(
        Q,
        K_POOL,
        V_POOL,
        GATHER,          # [B, HK, MAX_VISIBLE] absolute physical slots
        VISIBLE_LENS,    # [B, HK]
        QUERY_TO_REQ,    # [TOTAL_QUERIES]
        OUT,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        LOG2_PAGE: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
        USE_BF16: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        # 2D grid: one program per (flat query row, query head).  This avoids
        # the flattened pid division/modulo on the inner path and keeps page
        # loads private to each q/head program.
        q_idx = tl.program_id(0)
        head = tl.program_id(1)
        if q_idx >= TOTAL_QUERIES:
            return

        # Host precomputes q_idx -> req_idx, eliminating the per-program
        # linear scan over query_start_loc in the inner kernel.
        req_idx = tl.load(QUERY_TO_REQ + q_idx, cache_modifier='.cg')
        kv_head = head // (HQ // HK)

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)

        if req_idx < 0:
            # Padded batch rows have no request slot.  Their q rows are never
            # consumed by the scheduler, but write zeros to keep graph replay
            # deterministic and avoid an out-of-bounds VISIBLE_LENS load.
            tl.store(
                OUT + q_idx * HQ * DV + head * DV + vd,
                tl.zeros((BV,), tl.float32).to(OUT.dtype.element_ty),
                mask=vd < DV,
            )
            return
        vis_len = tl.load(VISIBLE_LENS + req_idx * HK + kv_head, cache_modifier='.cg')

        # Keep Q and K in the KV dtype for tl.dot.  bf16 and fp16 both map to
        # Tensor Core MMA; the online softmax accumulators stay in fp32.
        q_vec = tl.load(
            Q + q_idx * HQ * D + head * D + d,
            mask=d < D, other=0.0, cache_modifier='.ca')
        if USE_DOT:
            if USE_BF16:
                q_vec = q_vec.to(tl.bfloat16)
            else:
                q_vec = q_vec.to(tl.float16)
            # tl.dot requires N >= 16.  Replicate the single query vector to
            # a [BD, 16] tile and reduce the identical columns afterwards.
            q_tile = tl.broadcast_to(
                tl.reshape(q_vec, (BD, 1)), (BD, 16))
        else:
            q_vec = q_vec.to(tl.float32)

        m = tl.full((), float('-inf'), tl.float32)
        z = tl.zeros((), tl.float32)
        acc = tl.zeros((BV,), tl.float32)

        # Triton 3.4 does not accept ``break`` inside ``tl.range``.  The
        # equivalent dynamic upper bound below stops at the first block beyond
        # this request's actual visible length, so short requests do not ride
        # along with the fixed MAX_VISIBLE tail.
        num_blocks = tl.cdiv(tl.minimum(vis_len, MAX_VISIBLE), BT)
        for block_idx in range(0, num_blocks):
            start = block_idx * BT
            col = tl.max_contiguous(start + t, BT)
            col_valid = col < vis_len
            slot = tl.load(
                GATHER + (req_idx * HK + kv_head) * MAX_VISIBLE + col,
                mask=col_valid, other=-1, cache_modifier='.cg')
            valid = col_valid & (slot >= 0)

            phys_page = (slot >> LOG2_PAGE).to(tl.int64)
            offset = slot & (PAGE - 1)

            k_base = (
                K_POOL
                + phys_page * (HK * PAGE * D)
                + kv_head * (PAGE * D)
                + offset * D
            )
            v_base = (
                V_POOL
                + phys_page * (HK * PAGE * DV)
                + kv_head * (PAGE * DV)
                + offset * DV
            )

            k = tl.load(
                k_base[:, None] + d[None, :],
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0, cache_modifier='.ca')
            if USE_DOT:
                if USE_BF16:
                    k = k.to(tl.bfloat16)
                else:
                    k = k.to(tl.float16)
                score_tile = tl.dot(k, q_tile)
                score = tl.max(score_tile, 1) * SCALE
            else:
                k = k.to(tl.float32)
                score = tl.sum(k * q_vec[None, :], axis=1) * SCALE
            score = tl.where(valid, score, float('-inf'))

            new_m = tl.maximum(m, tl.max(score, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score - safe_m)

            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0, cache_modifier='.ca').to(tl.float32)

            acc = acc * alpha + tl.sum(v * prob[:, None], axis=0)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        out = tl.where(z == 0.0, 0.0, acc / z)
        tl.store(
            OUT + q_idx * HQ * DV + head * DV + vd,
            out.to(OUT.dtype.element_ty),
            mask=vd < DV,
        )



    @triton.jit
    def _gather_attend_grouped_kernel(
        Q,
        K_POOL,
        V_POOL,
        GATHER,
        VISIBLE_LENS,
        QUERY_TO_REQ,
        OUT,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        LOG2_PAGE: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        HEADS_PER_PROGRAM: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        """Grouped Tensor-Core gather attention over all visible blocks."""
        q_idx = tl.program_id(0)
        head_base = tl.program_id(1) * HEADS_PER_PROGRAM
        kv_head = head_base // HEAD_GROUP
        if q_idx >= TOTAL_QUERIES:
            return

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)
        h = tl.arange(0, HEAD_TILE)
        req_idx = tl.load(QUERY_TO_REQ + q_idx, cache_modifier='.cg')
        out_base = OUT + q_idx * HQ * DV + head_base * DV
        out_ptrs = out_base + h[:, None] * DV + vd[None, :]
        out_mask = (h[:, None] < HEADS_PER_PROGRAM) & (vd[None, :] < DV)
        if req_idx < 0:
            tl.store(
                out_ptrs,
                tl.zeros((HEAD_TILE, BV), tl.float32).to(OUT.dtype.element_ty),
                mask=out_mask,
            )
            return

        vis_len = tl.load(VISIBLE_LENS + req_idx * HK + kv_head, cache_modifier='.cg')
        q_ptrs = Q + q_idx * HQ * D + d[:, None] + (head_base + h)[None, :] * D
        q_tile = tl.load(
            q_ptrs,
            mask=(d[:, None] < D) & (h[None, :] < HEADS_PER_PROGRAM),
            other=0.0,
            cache_modifier='.ca',
        )
        if USE_BF16:
            q_tile = q_tile.to(tl.bfloat16)
        else:
            q_tile = q_tile.to(tl.float16)

        m = tl.full((HEAD_TILE,), float('-inf'), tl.float32)
        z = tl.zeros((HEAD_TILE,), tl.float32)
        acc = tl.zeros((HEAD_TILE, BV), tl.float32)
        num_blocks = tl.cdiv(tl.minimum(vis_len, MAX_VISIBLE), BT)
        for block_idx in range(0, num_blocks):
            start = block_idx * BT
            col = tl.max_contiguous(start + t, BT)
            col_valid = col < vis_len
            slot = tl.load(
                GATHER + (req_idx * HK + kv_head) * MAX_VISIBLE + col,
                mask=col_valid, other=-1, cache_modifier='.cg')
            valid = col_valid & (slot >= 0)
            phys_page = (slot >> LOG2_PAGE).to(tl.int64)
            offset = slot & (PAGE - 1)
            k_base = (
                K_POOL
                + phys_page * (HK * PAGE * D)
                + kv_head * (PAGE * D)
                + offset * D
            )
            v_base = (
                V_POOL
                + phys_page * (HK * PAGE * DV)
                + kv_head * (PAGE * DV)
                + offset * DV
            )
            k = tl.load(
                k_base[:, None] + d[None, :],
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0, cache_modifier='.ca')
            if USE_BF16:
                k = k.to(tl.bfloat16)
            else:
                k = k.to(tl.float16)
            score_tile = tl.dot(k, q_tile) * SCALE
            score_valid = valid[:, None] & (h[None, :] < HEADS_PER_PROGRAM)
            score_tile = tl.where(score_valid, score_tile, float('-inf'))
            new_m = tl.maximum(m, tl.max(score_tile, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score_tile - safe_m[None, :])
            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0, cache_modifier='.ca').to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(tl.trans(prob), v)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        out = tl.where(z[:, None] == 0.0, 0.0, acc / z[:, None])
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=out_mask)


    @triton.jit
    def _gather_attend_grouped_split_kernel(
        Q,
        K_POOL,
        V_POOL,
        GATHER,          # [B, HK, MAX_VISIBLE] absolute physical slots
        VISIBLE_LENS,    # [B, HK]
        QUERY_TO_REQ,    # [TOTAL_QUERIES]
        OUT,
        PART_ACC, PART_M, PART_Z,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        LOG2_PAGE: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        USE_BF16: tl.constexpr,
        NSPLIT: tl.constexpr,
        WRITE_PARTIAL: tl.constexpr,
    ):
        """Grouped Tensor-Core gather attention.

        One program owns one ``(query row, KV head)`` pair and computes all
        ``HEAD_GROUP`` query heads that share that KV head.  The Q tile is
        therefore ``[BD, HEAD_TILE]``, where ``HEAD_TILE`` is padded to at
        least 16 so both QK and PV use MMA instructions without the previous
        per-head 16-column replication.
        """
        q_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)
        if q_idx >= TOTAL_QUERIES:
            return

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)
        h = tl.arange(0, HEAD_TILE)

        req_idx = tl.load(QUERY_TO_REQ + q_idx, cache_modifier='.cg')
        out_base = OUT + q_idx * HQ * DV + kv_head * HEAD_GROUP * DV
        out_ptrs = out_base + h[:, None] * DV + vd[None, :]
        out_mask = (h[:, None] < HEAD_GROUP) & (vd[None, :] < DV)

        if req_idx < 0:
            # PAD 不触碰负 request 地址；split 路径写空部分和供统一归并。
            if WRITE_PARTIAL:
                part = ((split * TOTAL_QUERIES + q_idx) * HQ
                        + kv_head * HEAD_GROUP + h)
                tl.store(PART_ACC + part[:, None] * DV + vd[None, :],
                         tl.zeros((HEAD_TILE, BV), tl.float32), mask=out_mask)
                tl.store(PART_M + part, tl.full((HEAD_TILE,), float('-inf'), tl.float32),
                         mask=h < HEAD_GROUP)
                tl.store(PART_Z + part, tl.zeros((HEAD_TILE,), tl.float32),
                         mask=h < HEAD_GROUP)
            else:
                tl.store(
                    out_ptrs,
                    tl.zeros((HEAD_TILE, BV), tl.float32).to(OUT.dtype.element_ty),
                    mask=out_mask,
                )
            return

        vis_len = tl.load(VISIBLE_LENS + req_idx * HK + kv_head, cache_modifier='.cg')

        q_ptrs = Q + q_idx * HQ * D + d[:, None] + (kv_head * HEAD_GROUP + h)[None, :] * D
        q_tile = tl.load(
            q_ptrs,
            mask=(d[:, None] < D) & (h[None, :] < HEAD_GROUP),
            other=0.0,
            cache_modifier='.ca',
        )
        if USE_BF16:
            q_tile = q_tile.to(tl.bfloat16)
        else:
            q_tile = q_tile.to(tl.float16)

        m = tl.full((HEAD_TILE,), float('-inf'), tl.float32)
        z = tl.zeros((HEAD_TILE,), tl.float32)
        acc = tl.zeros((HEAD_TILE, BV), tl.float32)

        # Dynamic upper bound is the functional replacement for ``break`` on
        # Triton 3.4: a short request never walks the fixed MAX_VISIBLE tail.
        # split-KV 各 program 只处理连续的一段 block，保持分区内原运算顺序。
        num_blocks = tl.cdiv(tl.minimum(vis_len, MAX_VISIBLE), BT)
        if WRITE_PARTIAL:
            # 用 floor 边界均衡分配，避免 num_blocks=10、NSPLIT=8 时仅前 5 个
            # split 获得工作；所有 split 均参与，且单个 split 最多多处理一块。
            block_begin = (split * num_blocks) // NSPLIT
            block_end = ((split + 1) * num_blocks) // NSPLIT
        else:
            block_begin = 0
            block_end = num_blocks
        for block_idx in range(block_begin, block_end):
            start = block_idx * BT
            col = tl.max_contiguous(start + t, BT)
            col_valid = col < vis_len
            slot = tl.load(
                GATHER + (req_idx * HK + kv_head) * MAX_VISIBLE + col,
                mask=col_valid, other=-1, cache_modifier='.cg')
            valid = col_valid & (slot >= 0)

            phys_page = (slot >> LOG2_PAGE).to(tl.int64)
            offset = slot & (PAGE - 1)

            k_base = (
                K_POOL
                + phys_page * (HK * PAGE * D)
                + kv_head * (PAGE * D)
                + offset * D
            )
            v_base = (
                V_POOL
                + phys_page * (HK * PAGE * DV)
                + kv_head * (PAGE * DV)
                + offset * DV
            )

            k = tl.load(
                k_base[:, None] + d[None, :],
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0, cache_modifier='.ca')
            if USE_BF16:
                k = k.to(tl.bfloat16)
            else:
                k = k.to(tl.float16)

            score_tile = tl.dot(k, q_tile) * SCALE
            score_valid = valid[:, None] & (h[None, :] < HEAD_GROUP)
            score_tile = tl.where(score_valid, score_tile, float('-inf'))

            new_m = tl.maximum(m, tl.max(score_tile, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score_tile - safe_m[None, :])

            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0, cache_modifier='.ca').to(tl.float32)

            acc = acc * alpha[:, None] + tl.dot(tl.trans(prob), v)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        if WRITE_PARTIAL:
            part = ((split * TOTAL_QUERIES + q_idx) * HQ
                    + kv_head * HEAD_GROUP + h)
            tl.store(PART_ACC + part[:, None] * DV + vd[None, :], acc,
                     mask=out_mask)
            tl.store(PART_M + part, m, mask=h < HEAD_GROUP)
            tl.store(PART_Z + part, z, mask=h < HEAD_GROUP)
        else:
            out = tl.where(z[:, None] == 0.0, 0.0, acc / z[:, None])
            tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=out_mask)


    @triton.jit
    def _gather_attend_regions_combine_kernel(
        PROT_ACC, PROT_M, PROT_Z, MIDDLE_ACC, MIDDLE_M, MIDDLE_Z,
        OUT, TOTAL_QUERIES,
        HQ: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
        PROT_NSPLIT: tl.constexpr, MIDDLE_NSPLIT: tl.constexpr,
    ):
        """按 protected→middle 固定顺序合并两路 online-softmax 状态。"""
        q_idx = tl.program_id(0)
        head = tl.program_id(1)
        if q_idx >= TOTAL_QUERIES:
            return
        vd = tl.arange(0, BV)
        mask = vd < DV
        acc = tl.zeros((BV,), tl.float32)
        m = float('-inf')
        z = 0.0
        for split in tl.static_range(PROT_NSPLIT):
            part = (split * TOTAL_QUERIES + q_idx) * HQ + head
            part_m = tl.load(PROT_M + part)
            part_z = tl.load(PROT_Z + part)
            part_acc = tl.load(PROT_ACC + part * DV + vd, mask=mask, other=0.0)
            new_m = tl.maximum(m, part_m)
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            beta = tl.exp(part_m - safe_m)
            acc = acc * alpha + part_acc * beta
            z = z * alpha + part_z * beta
            m = new_m
        for split in tl.static_range(MIDDLE_NSPLIT):
            part = (split * TOTAL_QUERIES + q_idx) * HQ + head
            part_m = tl.load(MIDDLE_M + part)
            part_z = tl.load(MIDDLE_Z + part)
            part_acc = tl.load(MIDDLE_ACC + part * DV + vd, mask=mask, other=0.0)
            new_m = tl.maximum(m, part_m)
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            beta = tl.exp(part_m - safe_m)
            acc = acc * alpha + part_acc * beta
            z = z * alpha + part_z * beta
            m = new_m
        out = tl.where(z == 0.0, 0.0, acc / z)
        tl.store(OUT + (q_idx * HQ + head) * DV + vd,
                 out.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _gather_attend_split_combine_kernel(
        PART_ACC, PART_M, PART_Z, OUT, TOTAL_QUERIES,
        HQ: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
        NSPLIT: tl.constexpr,
    ):
        """按 split 序号归并 online-softmax 状态，保持确定性。"""
        q_idx = tl.program_id(0)
        head = tl.program_id(1)
        if q_idx >= TOTAL_QUERIES:
            return
        vd = tl.arange(0, BV)
        mask = vd < DV
        acc = tl.zeros((BV,), tl.float32)
        m = float('-inf')
        z = 0.0
        for split in tl.static_range(NSPLIT):
            part = (split * TOTAL_QUERIES + q_idx) * HQ + head
            part_m = tl.load(PART_M + part)
            part_z = tl.load(PART_Z + part)
            part_acc = tl.load(PART_ACC + part * DV + vd, mask=mask, other=0.0)
            new_m = tl.maximum(m, part_m)
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            beta = tl.exp(part_m - safe_m)
            acc = acc * alpha + part_acc * beta
            z = z * alpha + part_z * beta
            m = new_m
        out = tl.where(z == 0.0, 0.0, acc / z)
        tl.store(OUT + (q_idx * HQ + head) * DV + vd,
                 out.to(OUT.dtype.element_ty), mask=mask)


def paged_write_prefill(cache_layer, key_states, value_states, block_table, *,
                        workspace=None):
    """Write an unpadded B1 prefill with one batched paged-scatter launch.

    ``key_states`` / ``value_states`` are [1,HK,N,D] / [1,HK,N,DV], and
    ``block_table`` is the current layer/request logical-to-physical row.
    """
    if key_states.ndim != 4 or key_states.shape[0] != 1 or key_states.shape[2] < 1:
        raise ValueError('paged write prefill expects [1,HK,N,D] keys')
    if value_states.ndim != 4 or value_states.shape[:3] != key_states.shape[:3]:
        raise ValueError('paged write prefill expects [1,HK,N,DV] values')
    if key_states.shape[1] != cache_layer.num_kv_heads:
        raise ValueError('KV head count does not match arena')
    if key_states.shape[-1] != cache_layer.head_dim_k:
        raise ValueError('key head dimension does not match arena')
    if value_states.shape[-1] != cache_layer.head_dim_v:
        raise ValueError('value head dimension does not match arena')
    if block_table.ndim != 1 or block_table.device != key_states.device:
        raise ValueError('prefill block table must be a device-matched 1D row')

    n = key_states.shape[2]
    hk = cache_layer.num_kv_heads
    d = cache_layer.head_dim_k
    dv = cache_layer.head_dim_v
    page = cache_layer.geometry.page
    if n > block_table.numel() * page:
        raise ValueError('prefill length exceeds block-table capacity')

    if key_states.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for paged write prefill')
        from ..indexer.selection.unique_ranking import _fast_launch
        bt = 16
        _fast_launch(
            workspace, ('paged_write_prefill', id(cache_layer)),
            _paged_write_prefill_kernel, (triton.cdiv(n, bt), hk),
            key_states.contiguous(), value_states.contiguous(),
            block_table.contiguous(), cache_layer.kv_pool_k,
            cache_layer.kv_pool_v, n,
            hk, d, dv, page, bt,
            triton.next_power_of_2(d), triton.next_power_of_2(dv))
        return key_states, value_states

    # CPU reference 与 Triton 使用同一 logical->physical 映射。
    for logical_page in range((n + page - 1) // page):
        physical_id = int(block_table[logical_page])
        lo = logical_page * page
        hi = min(lo + page, n)
        cache_layer.kv_pool_k[physical_id, 0, :, :hi-lo, :].copy_(
            key_states[0, :, lo:hi, :])
        cache_layer.kv_pool_v[physical_id, 0, :, :hi-lo, :].copy_(
            value_states[0, :, lo:hi, :])
    return key_states, value_states


def paged_write_decode(cache_layer, key_states, value_states, slot_mapping, *,
                       workspace=None):
    """Write one decode token per batch row into a fixed physical arena.

    ``key_states`` / ``value_states`` are [B, HK, 1, D] / [B, HK, 1, DV].
    ``slot_mapping`` is [B] int32.  Invalid slots (-1) are routed to the PAD
    page so writes are safe discards and the kernel has no data-dependent
    branch.
    """
    if key_states.ndim != 4 or key_states.shape[-2] != 1:
        raise ValueError('paged write decode expects [B,HK,1,D] keys')
    if value_states.ndim != 4 or value_states.shape[-2] != 1:
        raise ValueError('paged write decode expects [B,HK,1,DV] values')
    if key_states.shape[0] != slot_mapping.shape[0]:
        raise ValueError('slot mapping batch must match KV batch')
    if key_states.shape[1] != cache_layer.num_kv_heads:
        raise ValueError('KV head count does not match arena')
    if key_states.device != slot_mapping.device:
        slot_mapping = slot_mapping.to(key_states.device)

    B = key_states.shape[0]
    HK = cache_layer.num_kv_heads
    D = cache_layer.head_dim_k
    DV = cache_layer.head_dim_v
    PAGE = cache_layer.geometry.page
    PAD_SLOT_SAFE = cache_layer.PAD_PAGE_ID * PAGE

    if key_states.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for paged write decode')
        from ..indexer.selection.unique_ranking import _fast_launch
        _fast_launch(
            workspace, 'paged_write', _paged_write_kernel, (B,),
            # The kernel uses packed [B, HK, D] offsets.  GLM's V is a
            # strided split view of the joint K/V projection, even at T=1.
            key_states.contiguous(),
            value_states.contiguous(),
            slot_mapping.contiguous(),
            cache_layer.kv_pool_k,
            cache_layer.kv_pool_v,
            B,
            HK,
            D,
            DV,
            PAGE,
            PAD_SLOT_SAFE,
            triton.next_power_of_2(D),
            triton.next_power_of_2(DV),
        )
        return key_states, value_states

    # CPU reference: PAD writes are intentionally allowed and discarded.
    for b in range(B):
        slot = int(slot_mapping[b])
        if slot < 0:
            slot = PAD_SLOT_SAFE
        physical_id, offset = divmod(slot, PAGE)
        cache_layer.kv_pool_k[physical_id, 0, :, offset, :] = key_states[b, :, 0, :]
        cache_layer.kv_pool_v[physical_id, 0, :, offset, :] = value_states[b, :, 0, :]
    return key_states, value_states


def sparse_paged_attention_gather_partial(
    cache_layer, q, gather_indices, visible_lens, query_to_req, *,
    scale, workspace, name, nsplit,
):
    """只写一个区域的 FP32 split 状态，不执行区域间归并。"""
    if not q.is_cuda or triton is None:
        raise RuntimeError('split regions require CUDA Triton')
    if q.ndim != 3 or q.shape[-1] != cache_layer.head_dim_k:
        raise ValueError('split regions expect flat [T,HQ,D] query')
    hq = q.shape[1]
    hk = cache_layer.num_kv_heads
    if hq % hk != 0:
        raise ValueError('GQA requires HQ % HK == 0')
    total_q = q.shape[0]
    max_visible = gather_indices.shape[-1]
    page = cache_layer.geometry.page
    if (page & (page - 1)) != 0:
        raise ValueError('split regions require power-of-two page size')
    head_group = hq // hk
    head_tile = max(16, triton.next_power_of_2(head_group))
    if (cache_layer.head_dim_k < 16 or cache_layer.head_dim_v < 16
            or head_group > head_tile):
        raise RuntimeError('split regions require grouped attention geometry')
    if cache_layer.head_dim_k >= 256 or cache_layer.head_dim_v >= 256:
        block_tokens = 64
        kernel_kwargs = {'num_stages': 1}
    else:
        block_tokens = 128 if total_q <= 8 else 64
        kernel_kwargs = {}
    from ..indexer.selection.unique_ranking import _fast_launch, _workspace_empty
    part_acc = _workspace_empty(
        workspace, f'gather_split_{name}_acc',
        (nsplit, total_q, hq, cache_layer.head_dim_v), torch.float32, q.device)
    part_m = _workspace_empty(
        workspace, f'gather_split_{name}_m', (nsplit, total_q, hq),
        torch.float32, q.device)
    part_z = _workspace_empty(
        workspace, f'gather_split_{name}_z', (nsplit, total_q, hq),
        torch.float32, q.device)
    q_arg = q if q.is_contiguous() else q.contiguous()
    _fast_launch(
        workspace, ('gather_attend_region_split', name),
        _gather_attend_grouped_split_kernel, (total_q, hk, nsplit),
        q_arg, cache_layer.kv_pool_k, cache_layer.kv_pool_v,
        gather_indices, visible_lens, query_to_req, q_arg,
        part_acc, part_m, part_z, total_q,
        hq, hk, cache_layer.head_dim_k, cache_layer.head_dim_v,
        page, page.bit_length() - 1, max_visible, visible_lens.shape[0],
        block_tokens, scale,
        triton.next_power_of_2(cache_layer.head_dim_k),
        triton.next_power_of_2(cache_layer.head_dim_v),
        head_group, head_tile, cache_layer.kv_pool_k.dtype == torch.bfloat16,
        nsplit, True, **kernel_kwargs)
    return part_acc, part_m, part_z


def sparse_paged_attention_regions_combine(
    q, protected_parts, middle_parts, *, workspace,
):
    """合并 disjoint 区域；serial 与 overlap 共用完全相同的归并顺序。"""
    from ..indexer.selection.unique_ranking import _fast_launch
    total_q, hq = q.shape[:2]
    dv = protected_parts[0].shape[-1]
    out = q.new_empty((total_q, hq, dv))
    _fast_launch(
        workspace, 'gather_attend_regions_combine',
        _gather_attend_regions_combine_kernel, (total_q, hq),
        *protected_parts, *middle_parts, out, total_q,
        hq, dv, triton.next_power_of_2(dv),
        protected_parts[0].shape[0], middle_parts[0].shape[0])
    return out


def sparse_paged_attention_gather(
    cache_layer,
    q,
    gather_indices,
    visible_lens,
    query_to_req,
    *,
    scale,
    workspace=None,
    split_kv=False,
):
    """Sparse gather paged attention over a GraphReadyPagedRegionLayer arena.

    ``q`` is flat [T, HQ, D].  ``gather_indices`` is [B, HK, MAX_VISIBLE]
    absolute physical slots (``physical_page * PAGE + offset``),
    ``visible_lens`` is [B, HK], and ``query_to_req`` maps each flat query row
    to its request slot. Each query head consumes its own KV group's selection.
    The kernel performs no block-table lookup and no integer division.
    ``split_kv=True`` 是实验对照路径：按 split 顺序归并 online-softmax，
    数值归约顺序不同于默认 grouped 单 program 路径，不作为语义不变优化。
    """
    if q.ndim != 3 or q.shape[-1] != cache_layer.head_dim_k:
        raise ValueError('sparse gather attention expects flat [T,HQ,D] query')

    hq = q.shape[1]
    hk = cache_layer.num_kv_heads
    if hq % hk != 0:
        raise ValueError('GQA requires HQ % HK == 0')
    if (gather_indices.ndim != 3 or gather_indices.shape[1] != hk
            or visible_lens.shape != gather_indices.shape[:2]):
        raise ValueError('gather attention expects indices [B,HK,K] and lens [B,HK]')
    total_q = q.shape[0]
    max_visible = gather_indices.shape[-1]
    page = cache_layer.geometry.page
    log2_page = page.bit_length() - 1
    if (page & (page - 1)) != 0:
        raise ValueError('sparse gather attention requires power-of-two page size')
    use_bf16 = cache_layer.kv_pool_k.dtype == torch.bfloat16
    head_group = hq // hk
    head_tile = max(16, triton.next_power_of_2(head_group)) if triton is not None else 16
    # B1 的 grouped grid 仅有 HK 个 CTA，无法填满 GPU。按 query head 拆分
    # 后仍保持每个 head 内完全相同的 online-softmax 循环与 tile 几何。
    heads_per_program = 1 if total_q == 1 and head_group > 1 else head_group
    use_grouped = (
        cache_layer.head_dim_k >= 16
        and cache_layer.head_dim_v >= 16
        and head_group <= head_tile
    )

    # 生产稳态下 selector 已建立固定几何与 launch context。直接重放 grouped
    # gather 的预绑定 launch，仅替换当前 Q 和新输出，避免重复组织 host 参数。
    if (not split_kv and workspace is not None and q.is_cuda
            and workspace.get('_nfa_capturing') is False):
        context = workspace.get('_nfa_launch_context')
        geometry = workspace.get('_nfa_launch_geometry')
        plan = workspace.get(('eager_gather_call_plan', id(cache_layer)))
        if (context is not None and context[1] is None and plan is not None
                and plan[0] is geometry):
            args, launch = plan[1], plan[2]
            if launch[1] == (_gather_attend_grouped_kernel.debug or context[3]):
                out = q.new_empty(args[6].shape)
                values = ((q.contiguous(),) + args[1:6] + (out,)
                          + args[7:] + launch[6])
                launch[2](args[7], launch[7], launch[8], context[0], launch[3],
                          launch[4], None, None, context[2], *values)
                return out

    if q.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for sparse gather attention')
        out = q.new_empty((total_q, hq, cache_layer.head_dim_v))
        from ..indexer.selection.unique_ranking import _fast_launch, _workspace_empty
        capturing = (workspace.get('_nfa_capturing') if workspace is not None
                     else None)
        if capturing is None:
            capturing = torch.cuda.is_current_stream_capturing()
        if use_grouped:
            # Small batches are latency-bound and prefer a wider token tile;
            # larger batches are occupancy-bound and prefer BT=64.  GLM's
            # D=DV=256 tile would otherwise exceed the L2/SMEM budget, so use a
            # narrower tile and single-stage pipelining there.
            if cache_layer.head_dim_k >= 256 or cache_layer.head_dim_v >= 256:
                block_tokens = 64
                kernel_kwargs = {'num_stages': 1}
            else:
                block_tokens = 128 if total_q <= 8 else 64
                # B1 已按 query head 拆成 32 个 CTA；BT128 下 8 warps
                # 保持 online-softmax 循环与输出逐项一致，并降低单 CTA 延迟。
                kernel_kwargs = ({'num_warps': 8}
                                 if heads_per_program == 1 else {})
            if not split_kv:
                gather_args = (
                    q.contiguous(),
                    cache_layer.kv_pool_k,
                    cache_layer.kv_pool_v,
                    gather_indices,
                    visible_lens,
                    query_to_req,
                    out,
                    total_q,
                    hq, hk,
                    cache_layer.head_dim_k,
                    cache_layer.head_dim_v,
                    page,
                    log2_page,
                    max_visible,
                    visible_lens.shape[0],
                    block_tokens,
                    scale,
                    triton.next_power_of_2(cache_layer.head_dim_k),
                    triton.next_power_of_2(cache_layer.head_dim_v),
                    head_group,
                    head_tile,
                    heads_per_program,
                    use_bf16,
                )
                _fast_launch(
                    workspace, 'gather_attend_grouped',
                    _gather_attend_grouped_kernel,
                    (total_q, hq // heads_per_program),
                    *gather_args, **kernel_kwargs,
                )
                if not capturing and workspace is not None:
                    geometry = workspace.get('_nfa_launch_geometry')
                    launch = workspace.get(('triton_plan', 'gather_attend_grouped'))
                    if launch is not None and launch[0] is geometry:
                        workspace[('eager_gather_call_plan', id(cache_layer))] = (
                            geometry, gather_args, launch)
                return out

            nsplit = 8
            part_acc = _workspace_empty(
                workspace, 'gather_split_acc',
                (nsplit, total_q, hq, cache_layer.head_dim_v),
                torch.float32, q.device)
            part_m = _workspace_empty(
                workspace, 'gather_split_m', (nsplit, total_q, hq),
                torch.float32, q.device)
            part_z = _workspace_empty(
                workspace, 'gather_split_z', (nsplit, total_q, hq),
                torch.float32, q.device)
            _fast_launch(
                workspace, 'gather_attend_grouped_split',
                _gather_attend_grouped_split_kernel, (total_q, hk, nsplit),
                q.contiguous(),
                cache_layer.kv_pool_k,
                cache_layer.kv_pool_v,
                gather_indices,
                visible_lens,
                query_to_req,
                out,
                part_acc, part_m, part_z,
                total_q,
                hq, hk,
                cache_layer.head_dim_k,
                cache_layer.head_dim_v,
                page,
                log2_page,
                max_visible,
                visible_lens.shape[0],
                block_tokens,
                scale,
                triton.next_power_of_2(cache_layer.head_dim_k),
                triton.next_power_of_2(cache_layer.head_dim_v),
                head_group,
                head_tile,
                use_bf16,
                nsplit,
                True,
                **kernel_kwargs,
            )
            _fast_launch(
                workspace, 'gather_attend_split_combine',
                _gather_attend_split_combine_kernel, (total_q, hq),
                part_acc, part_m, part_z, out, total_q,
                hq, cache_layer.head_dim_v,
                triton.next_power_of_2(cache_layer.head_dim_v), nsplit,
            )
        else:
            # Small-head fallback keeps the scalar-FMA path because tl.dot
            # requires N >= 16 and would otherwise fabricate dummy columns.
            _fast_launch(
                workspace, 'gather_attend', _gather_attend_kernel,
                (total_q, hq),
                q.contiguous(),
                cache_layer.kv_pool_k,
                cache_layer.kv_pool_v,
                gather_indices,
                visible_lens,
                query_to_req,
                out,
                total_q,
                hq, hk,
                cache_layer.head_dim_k,
                cache_layer.head_dim_v,
                page,
                log2_page,
                max_visible,
                visible_lens.shape[0],
                32,
                scale,
                triton.next_power_of_2(cache_layer.head_dim_k),
                triton.next_power_of_2(cache_layer.head_dim_v),
                use_bf16,
                False,
            )
        return out

    # CPU reference: physical-slot gather, loops only visible_lens slots.
    page = cache_layer.geometry.page
    out = q.new_zeros((total_q, hq, cache_layer.head_dim_v))
    for q_idx in range(total_q):
        req_idx = int(query_to_req[q_idx])
        if req_idx < 0:
            # Padded batch rows have no request slot; leave their output zero.
            continue
        for h in range(hq):
            kh = h // (hq // hk)
            vis_len = int(visible_lens[req_idx, kh].item())
            m = float('-inf')
            z = 0.0
            acc = torch.zeros(cache_layer.head_dim_v, dtype=torch.float32,
                              device=q.device)
            for col in range(vis_len):
                slot = int(gather_indices[req_idx, kh, col].item())
                if slot < 0:
                    continue
                phys, offset = divmod(slot, page)
                k = cache_layer.kv_pool_k[phys, 0, kh, offset].float()
                v = cache_layer.kv_pool_v[phys, 0, kh, offset].float()
                score = (q[q_idx, h].float() * k).sum() * scale
                new_m = max(m, score)
                if new_m == float('-inf'):
                    continue
                alpha = math.exp(m - new_m)
                prob = math.exp(score - new_m)
                acc = acc * alpha + prob * v
                z = z * alpha + prob
                m = new_m
            if z != 0:
                out[q_idx, h] = (acc / z).to(q.dtype)
    return out
