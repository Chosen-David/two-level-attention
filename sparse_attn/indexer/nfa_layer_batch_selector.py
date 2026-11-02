"""Opt-in layer-batched NFA selector.

Removes the 36-layer Python loop from GraphRunner.step by folding the
layer/request/KV-head dimensions into Triton grid dimensions.  It reads the
previous-step per-layer query rows from
``GraphReadyPagedRegionCache._last_q_buf`` and, in a few launches, produces
final physical gather slots for every layer at once.

Opt-in through ``NFA_LAYER_BATCH_SELECTOR=1`` and fail-closes to the
authoritative per-layer selector when geometry is unsupported.
"""
from __future__ import annotations

import os

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

_ENV = "NFA_LAYER_BATCH_SELECTOR"


def is_enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _has_triton() -> bool:
    return triton is not None and tl is not None


class FallbackRequired(Exception):
    """Signal GraphRunner to use the authoritative per-layer selector."""


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _has_triton():

    @triton.jit
    def _nfa_page_stats_kernel(
        KV_PTRS, BLOCK_TABLE, LENGTHS, SKIP_FAR, SKIP_NEAR, DIMS,
        OUT_MIN, OUT_MAX, OUT_SUM,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        SINK: tl.constexpr, NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, D_BLOCK: tl.constexpr, BS: tl.constexpr,
        KT_CAP: tl.constexpr, L: tl.constexpr, B: tl.constexpr,
    ):
        blk = tl.program_id(0)
        row = tl.program_id(1)
        m = row // HK
        kv = row % HK
        layer = m // B
        req = m % B

        length = tl.load(LENGTHS + req).to(tl.int32)
        skip_far = tl.load(SKIP_FAR + layer * B + req).to(tl.int32)
        skip_near = tl.load(SKIP_NEAR + layer * B + req).to(tl.int32)
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
            BLOCK_TABLE + req * MAX_BLOCKS + lp,
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
    def _nfa_fine_paged_kernel(
        KV_PTRS, BLOCK_TABLE, DIMS, Q, CAND, CLEN, OUT,
        R, NUM_TILES,
        stride_qm, stride_qh, stride_ch, stride_cr,
        stride_om, stride_or,
        MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
        HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
        D_BLOCK: tl.constexpr,
        HKV: tl.constexpr, G: tl.constexpr, B: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        m = row // HKV
        kv = row % HKV
        layer = m // B
        req = m % B

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
            phys = tl.load(BLOCK_TABLE + req * MAX_BLOCKS + lp,
                           mask=read_r & lp_valid, other=0).to(tl.int64)
            src = (kv_ptr
                   + phys[:, None] * HK * PAGE * HEAD_DIM
                   + kv * PAGE * HEAD_DIM
                   + off[:, None] * HEAD_DIM
                   + dim[None, :])
            k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + m * stride_qm + (kv * G + j) * stride_qh + offs_d)
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
            phys = tl.load(BLOCK_TABLE + req * MAX_BLOCKS + lp,
                           mask=read_r & lp_valid, other=0).to(tl.int64)
            src = (kv_ptr
                   + phys[:, None] * HK * PAGE * HEAD_DIM
                   + kv * PAGE * HEAD_DIM
                   + off[:, None] * HEAD_DIM
                   + dim[None, :])
            k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            prob_sum = tl.zeros((TILE,), tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + m * stride_qm + (kv * G + j) * stride_qh + offs_d)
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

# ---------------------------------------------------------------------------
# Host-side batched helpers
# ---------------------------------------------------------------------------

def _scalar(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(int(x), device=device, dtype=dtype)


def _page_stats_all_layers(cache, req_ids, layer_ptrs, dims, bs, pad,
                           lengths, skip_far, skip_near):
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
    b = len(req_ids)
    l = cache.n_layers
    m = l * b
    kt_cap = pad // bs
    dim_numel = dims.numel()

    block_table = cache.block_table_buf[req_ids].contiguous()
    lengths_t = torch.as_tensor(lengths, dtype=torch.int32, device=device)
    skip_far_t = torch.as_tensor(
        skip_far, dtype=torch.int32, device=device).contiguous()
    skip_near_t = torch.as_tensor(
        skip_near, dtype=torch.int32, device=device).contiguous()
    dims_t = dims.to(device=device).contiguous()
    kv_ptrs = torch.as_tensor(layer_ptrs, dtype=torch.int64, device=device)

    out_dtype = layer0.kv_pool_k.dtype
    k_min = torch.empty(
        (m, kt_cap, hk, dim_numel), dtype=out_dtype, device=device)
    k_max = torch.empty_like(k_min)
    k_sum = torch.empty_like(k_min)

    _nfa_page_stats_kernel[(kt_cap, m * hk)](
        kv_ptrs, block_table, lengths_t, skip_far_t, skip_near_t, dims_t,
        k_min, k_max, k_sum,
        MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        SINK=cache.geometry.sink, NEAR_LEN=cache.geometry.near,
        WINDOW=cache.geometry.window,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dim_numel,
        D_BLOCK=triton.next_power_of_2(dim_numel), BS=bs,
        KT_CAP=kt_cap, L=l, B=b,
        num_warps=4,
    )
    k_avg = (k_sum.to(torch.float32) / bs).to(out_dtype)
    return k_min, k_max, k_avg


def _fused_fine_probabilities_paged(
        cache, req_ids, layer_ptrs, dims, q_fine, cand_idx2, keep2,
        cand_len, capacity, group_size, tile=64, num_warps=16):
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
    b = len(req_ids)
    m, hkv, r = cand_idx2.shape
    if hkv != hk:
        raise FallbackRequired("KV-head geometry mismatch")

    block_table = cache.block_table_buf[req_ids].contiguous()
    dims_t = dims.to(device=device).contiguous()
    kv_ptrs = torch.as_tensor(layer_ptrs, dtype=torch.int64, device=device)

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
        kv_ptrs, block_table, dims_t, q_fine, cand_idx2, cand_len, probs,
        R=r, NUM_TILES=num_tiles,
        stride_qm=q_fine.stride(0), stride_qh=q_fine.stride(1),
        stride_ch=cand_idx2.stride(1), stride_cr=cand_idx2.stride(2),
        stride_om=probs.stride(1), stride_or=probs.stride(2),
        MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        HK=hk, HEAD_DIM=head_dim, DIM_NUMEL=dims.numel(),
        D_BLOCK=triton.next_power_of_2(dims.numel()),
        HKV=hkv, G=group_size, B=b, TILE=tile,
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
    sink_cross = torch.tensor(sink % bs != 0, dtype=torch.bool, device=device)
    slot0 = torch.where(sink_cross, torch.tensor(sink_pool_lo, device=device),
                        torch.tensor(0, device=device))
    byp_blk = torch.stack([slot0.expand(len(req_ids)), swa_byp_blk], dim=1)  # [B,2]
    byp_tok = byp_blk[:, :, None] * bs + ar_bs[None, None, :]
    byp_keep = byp_tok < s_real[:, None, None]
    byp_keep = byp_keep & torch.stack(
        [sink_cross.expand(len(req_ids)), swa_byp_cond], dim=1)[:, :, None]

    return {
        "s_real": s_real,
        "swa_lo_tok": swa_lo_tok,
        "far_pool_mask": far_pool_mask,
        "near_pool_mask": near_pool_mask,
        "mid_mask": mid_mask,
        "swa_force": swa_force,
        "k2_mid": k2_mid,
        "far_tok_lo": torch.tensor(far_tok_lo, dtype=torch.long, device=device),
        "byp_tok": byp_tok,
        "byp_keep": byp_keep,
    }


def _coarse_scores(k_min, k_max, k_avg, b_q_full, group_size, indexer, dtype):
    # One launch per coarse feature instead of a Python row loop.  We reuse
    # the authoritative BATCHED Triton score kernels and then reduce the
    # q-head dimension with the same torch.mean PyTorch uses in
    # compute_mask_graph, so the numerical contract stays bit-identical.
    m, kt_cap, hkv, d = k_min.shape
    hq = b_q_full.shape[1]
    q_flat = b_q_full.reshape(m * hq, d).contiguous()
    coarse = torch.empty((m, hq, kt_cap), dtype=dtype, device=k_min.device)
    bt, nw = indexer._TILE_COARSE.get(d, (64, 4))
    _score_coarse_kernel[(triton.cdiv(kt_cap, bt), m * hkv)](
        k_min, k_max, q_flat, coarse, coarse,
        kt_cap,
        k_min.stride(1), k_min.stride(2),
        k_max.stride(1), k_max.stride(2),
        q_flat.stride(0), coarse.stride(1), coarse.stride(2),
        coarse.stride(1), coarse.stride(2),
        Hkv=hkv, G=group_size, D=d, BT=bt, HAS_MEAN=False,
        BATCHED=True,
        STRIDE_KB=k_min.stride(0), STRIDE_KB2=k_max.stride(0),
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
    return sc, sc_avg


def _candidate_assembly(i_f, i_n, geo, req_idx, m, hkv, bs, device):
    ar_bs = torch.arange(bs, dtype=torch.long, device=device)
    far_mask = geo["far_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)
    near_mask = geo["near_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)
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

def layer_batched_select(cache, model, req_ids):
    """Fill ``gather_indices_buf``/``visible_lens_buf`` for all layers at once.

    Raises ``FallbackRequired`` when the batch contains the first decode step
    of a request, any partial/complete skip policy, or unsupported geometry.
    """
    if not is_enabled() or not _has_triton():
        raise FallbackRequired("disabled")
    if cache.layers[0].kv_pool_k.device.type != "cuda":
        raise FallbackRequired("CUDA required")

    valid_slots = [b for b, rid in enumerate(req_ids)
                   if rid is not None and rid >= 0 and rid in cache.states]
    valid_req_ids = [req_ids[b] for b in valid_slots]
    if not valid_req_ids:
        return
    if any(not cache.states[rid].first_decode_seeded for rid in valid_req_ids):
        raise FallbackRequired("first decode seed uses per-layer fallback")

    n_layers = cache.n_layers
    device = cache.layers[0].kv_pool_k.device
    indexer = model.model.layers[0].self_attn.indexer
    if not hasattr(indexer, "args") or indexer.args is None:
        raise FallbackRequired("missing indexer config")

    # Only the fixed non-skip production geometry is supported here.
    for rid in valid_req_ids:
        state = cache.states[rid]
        if any(state.skip_far_by_layer) or any(state.skip_near_by_layer):
            raise FallbackRequired("skip policies unsupported")

    lengths = [
        cache.pending_lengths.get(rid, cache.states[rid].logical_length)
        for rid in valid_req_ids
    ]
    b = len(valid_req_ids)
    m = n_layers * b
    hq = cache.num_query_heads
    hkv = cache.layers[0].num_kv_heads
    group_size = hq // hkv
    bs = int(indexer.args.tia_block_size)
    page = cache.geometry.page
    max_len = max(lengths)
    pad = ((max_len + bs - 1) // bs) * bs
    kt_cap = pad // bs
    s_cap = pad

    dims = indexer._subspace_indices(device).long()
    if indexer.enable_subspace:
        q_dims = dims
    else:
        q_dims = None

    # Per-layer previous-step Q: [L,B,Hq,D].
    q_buf = cache._last_q_buf[:, valid_req_ids].contiguous()
    scales = _layer_scales(cache, model, device)
    q_full = (q_buf.to(torch.float32) * scales.view(n_layers, 1, 1, 1)).to(q_buf.dtype).to(torch.float32).reshape(m, hq, -1)
    if q_dims is not None:
        b_q_full = q_full[..., q_dims].contiguous()
    else:
        b_q_full = q_full.contiguous()
    if indexer.l2_subspace == "full":
        b_q_fine = q_full.contiguous()
    elif q_dims is not None:
        b_q_fine = b_q_full.contiguous()
    else:
        b_q_fine = q_full[..., indexer._qat_indices(device)].contiguous()

    # Layer pointers and skip flags.
    layer_ptrs = [cache.layers[li].kv_pool_k.data_ptr() for li in range(n_layers)]
    skip_far = [[0 for _ in range(b)] for _ in range(n_layers)]
    skip_near = [[0 for _ in range(b)] for _ in range(n_layers)]

    # ---- one launch: in-place paged block stats, no projection -----------
    k_min, k_max, k_avg = _page_stats_all_layers(
        cache, valid_req_ids, layer_ptrs, dims, bs, pad,
        lengths, skip_far, skip_near)  # each [M,kt_cap,Hkv,d]

    # ---- coarse scores ---------------------------------------------------
    b_q = b_q_full.reshape(m, hkv, group_size, dims.numel())
    sc, sc_avg = _coarse_scores(
        k_min, k_max, k_avg, b_q_full, group_size, indexer, cache._last_q_buf.dtype)

    geo = _batched_geometry(cache, valid_req_ids, lengths, indexer, device, kt_cap, s_cap)
    req_idx = torch.arange(b, dtype=torch.long, device=device).repeat(n_layers)
    far_pool_m = geo["far_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)
    near_pool_m = geo["near_pool_mask"][req_idx].unsqueeze(1).expand(m, hkv, -1)

    nb_near = max(0, int(indexer.nt_near_pages))
    k1 = int(indexer.args.tia_level1_topk)
    nb_far = max(0, k1 - nb_near)
    k1_eff = nb_far + nb_near

    far_src = sc_avg if (indexer.far_method == "avg" and sc_avg is not None) else sc
    near_src = sc if (indexer.near_method == "minmax" or sc_avg is None) else sc_avg
    neg_inf = torch.tensor(float("-inf"), dtype=far_src.dtype, device=device)
    far_scored = torch.where(far_pool_m, far_src, neg_inf)
    near_scored = torch.where(near_pool_m, near_src, neg_inf)
    i_f = torch.topk(
        far_scored.unsqueeze(1), min(nb_far, kt_cap), dim=-1).indices.squeeze(1)
    i_n = torch.topk(
        near_scored.unsqueeze(1), min(nb_near, kt_cap), dim=-1).indices.squeeze(1)

    cand_idx2, keep2 = _candidate_assembly(
        i_f, i_n, geo, req_idx, m, hkv, bs, device)

    # ---- candidate-domain completion -------------------------------------
    cand_idx2, keep2, cand_len = _complete_pack_batched(
        cand_idx2, keep2, lengths * n_layers,
        int(indexer.sink_tokens), int(indexer.sliding_window_size), s_cap)

    # ---- fused fine score + local unique topk ----------------------------
    probs, unique = _fused_fine_probabilities_paged(
        cache, valid_req_ids, layer_ptrs, dims, b_q_fine,
        cand_idx2, keep2, cand_len, s_cap, group_size)

    lower = geo["far_tok_lo"].expand(m).contiguous()
    upper = torch.maximum(lower, geo["swa_lo_tok"][req_idx])
    quota = geo["k2_mid"][req_idx]
    width = 1024
    middle = rank_unique_batched(
        cand_idx2, probs, unique, lower, upper, quota,
        capacity=s_cap, width=width, sink=int(indexer.sink_tokens))

    # ---- protected sink + SWA merge --------------------------------------
    protected_ids, protected_counts = _protected_selection_batch(
        cache, 0, valid_req_ids, indexer)
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

    middle_valid = (
        torch.arange(middle.indices.shape[-1], device=device)[None, None, :]
        < middle.counts[:, :, None])
    ids = torch.cat([middle.indices.to(torch.int32), prot], dim=-1)
    valid = torch.cat([middle_valid, prot_valid], dim=-1)
    final = canonical_selection(ids, valid, s_cap)

    # ---- physical mapping for request-level (head 0) selection -----------
    selected_ids = final.indices[:, 0].contiguous()   # [M, width]
    counts = final.counts[:, 0].contiguous()          # [M]
    width_final = selected_ids.shape[-1]
    req_idx_m = req_idx.long()
    bt = cache.block_table_buf[valid_req_ids]          # [B, max_blocks]
    bt = bt.unsqueeze(0).expand(n_layers, -1, -1).reshape(m, -1)
    pages = selected_ids >> (page.bit_length() - 1) if (page & (page - 1)) == 0 else torch.div(selected_ids, page, rounding_mode='floor')
    offsets = selected_ids & (page - 1) if (page & (page - 1)) == 0 else torch.remainder(selected_ids, page)
    safe_pages = torch.clamp(pages, min=0, max=cache.max_blocks - 1)
    phys = bt.gather(1, safe_pages.to(torch.int64))
    slots = phys.to(torch.int32) * page + offsets.to(torch.int32)
    slots = torch.where(selected_ids >= 0, slots,
                        torch.full_like(slots, INVALID_GATHER_IDX))
    slots = slots.reshape(n_layers, b, width_final)
    counts = counts.reshape(n_layers, b)

    valid_req_t = torch.as_tensor(valid_req_ids, dtype=torch.long, device=device)
    slots_full = torch.full(
        (n_layers, b, cache.max_visible_len), INVALID_GATHER_IDX,
        dtype=slots.dtype, device=device)
    slots_full[:, :, :width_final] = slots
    # Padding slots must not inherit a previous step's visible length; the
    # authoritative per-layer gather path zeroes this buffer before writing.
    cache.visible_lens_buf.zero_()
    cache.gather_indices_buf.index_copy_(1, valid_req_t, slots_full)
    cache.visible_lens_buf.index_copy_(
        1, valid_req_t, counts.to(torch.int32).contiguous())

    for rid in valid_req_ids:
        cache.states[rid].first_decode_seeded = True


def _layer_scales(cache, model, device):
    """Cache per-layer attention scales on the cache to avoid a hot-path list."""
    scales = getattr(cache, "_layer_batch_scales", None)
    if scales is None or scales.device != device or scales.numel() != cache.n_layers:
        vals = [module.self_attn.scaling for module in model.model.layers]
        scales = torch.as_tensor(vals, dtype=torch.float32, device=device)
        cache._layer_batch_scales = scales
    return scales
