"""Opt-in fused NFA fine scorer + local Top-K prototype.

This module is intentionally *not* on the production path by default.  Set
``NFA_FINE_TOPK_TRITON=1`` to enable it.  It replaces the host-side
``scatter_/masked_fill_/softmax`` candidate-normalization chain with a single
Triton pass:

    candidate gather -> QK fine score -> online softmax -> GQA group mean

and then ranks candidates with the pure-Triton packed-key Top-K used by the
NFA unique-ranking path.  The output can be either logical token ids (feeding
the existing cache page-table projection) or absolute ``physical_slot`` ids.

The implementation deliberately keeps the numerical contract close to the
accepted NFA path:

  * BF16 model dtype and RNE rounding on output probabilities,
  * per query-head softmax over the *capacity* denominator (missing candidates
    contribute zero, never enter the normalizer),
  * GQA mean over ``G`` query heads, with each head normalized independently,
  * duplicate logical-token candidates reduced to the last candidate slot
    (atomic_max owner), and
  * (probability descending, logical id ascending) tie order for BF16/FP16
    inputs by packing ``(prob_bits << 16) + (capacity - token_id)``.

If any supported geometry guard fails, the wrapper raises (fail-closed) so the
caller can fall back to the existing implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only or missing triton
    triton = None
    tl = None


_ENV = "NFA_FINE_TOPK_TRITON"
INVALID_SLOT = -1


def is_enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _has_triton() -> bool:
    return triton is not None and tl is not None


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _has_triton():
    @triton.jit
    def _candidate_owner_kernel(
        IDS, KEEP, OWNERS,
        R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Record the last candidate slot owning each logical token.

        Mirrors the accepted `_candidate_owners` contract: repeated IDs have
        identical fine scores, so the last slot (atomic max of the candidate
        column index) is the canonical owner.
        """
        kv = tl.program_id(0)
        r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        ids = tl.load(IDS + kv * R + r, mask=mask, other=0)
        keep = tl.load(KEEP + kv * R + r, mask=mask, other=False)
        valid = mask & keep & (ids >= 0) & (ids < S)
        tl.atomic_max(OWNERS + kv * S + ids, r, mask=valid, sem="relaxed")

    @triton.jit
    def _owner_validity_kernel(
        IDS, KEEP, OWNERS, VALID, UNIQUE,
        R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Emit `valid` and `unique` masks for candidate slots."""
        kv = tl.program_id(0)
        r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        ids = tl.load(IDS + kv * R + r, mask=mask, other=0)
        keep = tl.load(KEEP + kv * R + r, mask=mask, other=False)
        valid = mask & keep & (ids >= 0) & (ids < S)
        owner = tl.load(OWNERS + kv * S + ids, mask=valid, other=-1)
        unique = valid & (owner == r)
        tl.store(VALID + kv * R + r, valid, mask=mask)
        tl.store(UNIQUE + kv * R + r, unique, mask=mask)

    @triton.jit
    def _fused_fine_prob_kernel(
        KQ, Q, CAND, CLEN, OUT,
        R, NUM_TILES,
        stride_kt, stride_kh, stride_qh, stride_ch, stride_cr,
        stride_oh, stride_or,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        BT: tl.constexpr,
    ):
        """Fused candidate score -> online softmax -> GQA mean.

        One persistent program per KV head.  It makes two passes over the
        candidate tiles without materializing the [H, R] score matrix:

          1. streaming online softmax stats m/l per query head,
          2. tile-major probability recomputation and GQA mean.

        This keeps the numerical denominator over the full candidate domain
        (missing candidates contribute zero), which is the NFA contract.
        """
        kv = tl.program_id(0)
        offs_d = tl.arange(0, D)
        row = tl.arange(0, G)

        m = tl.full((G,), float("-inf"), tl.float32)
        l = tl.zeros((G,), tl.float32)

        cl = tl.load(CLEN + kv)

        # ---- pass 1: global max and denominator per query head ----
        for t in range(NUM_TILES):
            offs_r = t * BT + tl.arange(0, BT)
            mask_r = offs_r < R
            read_r = mask_r & (offs_r < cl)
            cand = tl.load(CAND + kv * stride_ch + offs_r * stride_cr,
                           mask=read_r, other=0)
            k = tl.load(KQ + cand[:, None] * stride_kt + kv * stride_kh
                        + offs_d[None, :],
                        mask=read_r[:, None], other=0.0).to(tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + (kv * G + j) * stride_qh + offs_d)
                s = tl.sum(k * q[None, :], axis=1)          # [BT]
                s = tl.where(read_r, s, float("-inf"))
                s = s.to(tl.bfloat16).to(tl.float32)
                smax = tl.max(s, axis=0)
                new_m = tl.maximum(m, tl.where(row == j, smax, float("-inf")))
                corr = tl.exp(m - new_m)                     # [G]
                corr = tl.where((m == float("-inf")) & (new_m == float("-inf")),
                                1.0, corr)
                m_j = tl.sum(tl.where(row == j, new_m, 0.0), axis=0)
                l_j_update = tl.sum(tl.exp(s - m_j), axis=0)
                l = l * corr + tl.where(row == j, l_j_update, 0.0)
                m = new_m

        # ---- pass 2: tile-major probabilities and group mean ----
        for t in range(NUM_TILES):
            offs_r = t * BT + tl.arange(0, BT)
            mask_r = offs_r < R
            read_r = mask_r & (offs_r < cl)
            cand = tl.load(CAND + kv * stride_ch + offs_r * stride_cr,
                           mask=read_r, other=0)
            k = tl.load(KQ + cand[:, None] * stride_kt + kv * stride_kh
                        + offs_d[None, :],
                        mask=read_r[:, None], other=0.0).to(tl.float32)
            prob_sum = tl.zeros((BT,), tl.float32)
            for j in tl.static_range(G):
                q = tl.load(Q + (kv * G + j) * stride_qh + offs_d)
                s = tl.sum(k * q[None, :], axis=1)
                s = tl.where(read_r, s, float("-inf"))
                s = s.to(tl.bfloat16).to(tl.float32)
                m_j = tl.sum(tl.where(row == j, m, 0.0), axis=0)
                l_j = tl.sum(tl.where(row == j, l, 0.0), axis=0)
                safe_l = tl.where(l_j > 0.0, l_j, 1.0)
                head_p = tl.exp(s - m_j) / safe_l
                head_p = tl.where(l_j > 0.0, head_p, 0.0)
                prob_sum += head_p.to(tl.bfloat16).to(tl.float32)
            out = prob_sum * (1.0 / G)
            tl.store(OUT + kv * stride_oh + offs_r * stride_or,
                     out.to(OUT.dtype.element_ty), mask=mask_r)

    @triton.jit
    def _packed_topk_kernel(
        IDS, PROB, VALID, LOW, HIGH, QUOTA, OUT, COUNT,
        R,
        stride_ih, stride_ir, stride_ph, stride_pr, stride_vh, stride_vr,
        stride_oh, stride_or,
        S: tl.constexpr, K: tl.constexpr, BLOCK_R: tl.constexpr,
        SINK: tl.constexpr,
    ):
        """Pure-Triton packed-key local Top-K.

        One program per KV head.  Keys pack positive BF16 probability bits and
        the inverse logical id; ascending sort gives smallest keys first, so
        the top ``K`` descending-probability tokens are the tail of the sorted
        vector.  Deduplicated/validated candidate slots are supplied through
        ``VALID``.
        """
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK_R)
        mask = offs < R

        ids = tl.load(IDS + h * stride_ih + offs * stride_ir, mask=mask,
                      other=S)
        p = tl.load(PROB + h * stride_ph + offs * stride_pr, mask=mask,
                    other=0.0)
        valid = tl.load(VALID + h * stride_vh + offs * stride_vr, mask=mask,
                        other=False)

        low = tl.load(LOW).to(tl.int32)
        high = tl.load(HIGH).to(tl.int32)
        quota = tl.load(QUOTA).to(tl.int32)
        token = ids.to(tl.int32)
        good = valid & (token >= low) & (token < high) & (p > 0.0) & (quota > 0)

        bits = p.to(tl.uint16, bitcast=True).to(tl.int32)
        key = tl.where(good, (bits << 16) + (S - token), 0)
        ordered = tl.sort(key, descending=False)

        top_key = tl.gather(ordered, BLOCK_R - K + tl.arange(0, K), axis=0)
        positive = (top_key >> 16) > 0
        top_token = tl.where(positive, S - (top_key & 65535), S)
        top_token = tl.where((tl.arange(0, K) < quota), top_token, S)
        top_token = tl.sort(top_token, descending=False)

        final = tl.where(top_token >= SINK, top_token, S)
        valid_final = final < S
        count = tl.sum(valid_final.to(tl.int32), axis=0)
        final = tl.where(valid_final, final, 0)

        tl.store(OUT + h * stride_oh + tl.arange(0, K) * stride_or, final)
        tl.store(COUNT + h, count)

    @triton.jit
    def _logical_to_physical_kernel(
        IDS, BLOCK_TABLE, OUT,
        K: tl.constexpr, PAGE: tl.constexpr, LOG2_PAGE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, PAD_PAGE: tl.constexpr,
        stride_ih, stride_ir, stride_bt_req, stride_bt_page,
        stride_oh, stride_or,
    ):
        """Map packed logical token ids to absolute physical slots.

        ``BLOCK_TABLE`` is [num_requests, max_blocks] int32.  One program per
        KV head, one vector of width ``K``.
        """
        h = tl.program_id(0)
        offs = tl.arange(0, K)
        ids = tl.load(IDS + h * stride_ih + offs * stride_ir)
        valid = (ids >= 0)
        safe = tl.where(valid, ids, 0)

        if LOG2_PAGE >= 0:
            pages = safe >> LOG2_PAGE
            offsets = safe & (PAGE - 1)
        else:
            pages = safe // PAGE
            offsets = safe % PAGE

        pages = tl.minimum(pages, MAX_BLOCKS - 1)
        phys = tl.load(BLOCK_TABLE + pages * stride_bt_page, mask=valid,
                       other=PAD_PAGE)
        slot = phys.to(tl.int32) * PAGE + offsets.to(tl.int32)
        slot = tl.where(valid, slot, -1)
        tl.store(OUT + h * stride_oh + offs * stride_or, slot)


@dataclass
class FineTopKResult:
    probabilities: torch.Tensor | None
    unique: torch.Tensor | None
    logical_ids: torch.Tensor | None
    physical_slots: torch.Tensor | None
    counts: torch.Tensor | None
    fallback_reason: str | None = None


def _pow2_or_minus_one(page_size: int):
    if page_size > 0 and (page_size & (page_size - 1)) == 0:
        return page_size.bit_length() - 1
    return -1


def _validate_fine_inputs(kq, q_fine, cand_idx2, keep2, cand_len,
                          capacity, group_size):
    if not _has_triton():
        raise RuntimeError("Triton is not available")
    if not all(t.is_cuda for t in (kq, q_fine, cand_idx2, keep2, cand_len)):
        raise ValueError("all fine inputs must be CUDA tensors")
    if not all(t.is_contiguous() for t in (cand_idx2, keep2, cand_len)):
        raise ValueError("candidate metadata must be contiguous")
    if kq.stride(-1) != 1 or q_fine.stride(-1) != 1:
        raise ValueError("K/Q last dimension must be contiguous")
    if kq.ndim != 3 or q_fine.ndim != 2 or cand_idx2.ndim != 2:
        raise ValueError("expected kq [S,Hkv,D], q [H,D], cand [Hkv,R]")
    hk, r = cand_idx2.shape
    if keep2.shape != cand_idx2.shape or cand_len.shape != (hk,):
        raise ValueError("candidate metadata shape mismatch")
    if cand_idx2.dtype != torch.int64 or cand_len.dtype != torch.int64:
        raise ValueError("candidate ids/lengths must be int64")
    if keep2.dtype != torch.bool:
        raise ValueError("keep2 must be bool")
    if q_fine.shape[0] % hk:
        raise ValueError("query heads must be an integer multiple of KV heads")
    g = q_fine.shape[0] // hk
    if group_size != g:
        raise ValueError("group_size does not match q_fine geometry")
    if kq.shape[1] != hk or kq.shape[2] != q_fine.shape[1]:
        raise ValueError("K/Q head/dim geometry mismatch")
    if capacity < 1 or capacity >= 65536:
        raise ValueError("BF16 packed Top-K requires capacity < 65536")
    if q_fine.dtype != torch.float32:
        raise ValueError("fine query must be fp32 and pre-scaled")
    return hk, r, g


def fused_fine_probabilities(
    kq: torch.Tensor,
    q_fine: torch.Tensor,
    cand_idx2: torch.Tensor,
    keep2: torch.Tensor,
    cand_len: torch.Tensor,
    capacity: int,
    group_size: int,
    *,
    stream: torch.cuda.Stream | None = None,
    tile: int = 64,
    num_warps: int = 4,
):
    """Candidate gather -> score -> online softmax -> GQA mean in one kernel.

    Returns ``(probabilities [Hkv,R] bf16, unique [Hkv,R] bool)``.
    """
    if not _has_triton():
        raise RuntimeError("Triton is not available")
    hk, r, g = _validate_fine_inputs(
        kq, q_fine, cand_idx2, keep2, cand_len, capacity, group_size)

    dev = kq.device
    if r == 0:
        probs = torch.empty((hk, 0), dtype=kq.dtype, device=dev)
        unique = torch.empty((hk, 0), dtype=torch.bool, device=dev)
        return probs, unique
    with torch.cuda.device(dev):
        cur_stream = torch.cuda.current_stream()
        if stream is not None and stream is not cur_stream:
            stream.wait_stream(cur_stream)
        use_stream = stream if stream is not None else cur_stream

        owners = torch.full((hk, capacity), -1, dtype=torch.int32, device=dev)
        valid = torch.empty_like(keep2)
        unique = torch.empty_like(keep2)
        probs = torch.empty((hk, r), dtype=kq.dtype, device=dev)

        block = 256
        grid_owner = (hk, triton.cdiv(r, block))
        grid_prob = (hk,)
        num_tiles = triton.cdiv(r, tile)

        with torch.cuda.stream(use_stream):
            _candidate_owner_kernel[grid_owner](
                cand_idx2, keep2, owners, R=r, S=capacity, BLOCK=block)
            _owner_validity_kernel[grid_owner](
                cand_idx2, keep2, owners, valid, unique,
                R=r, S=capacity, BLOCK=block)
            _fused_fine_prob_kernel[grid_prob](
                kq, q_fine, cand_idx2, cand_len, probs,
                R=r, NUM_TILES=num_tiles,
                stride_kt=kq.stride(0), stride_kh=kq.stride(1),
                stride_qh=q_fine.stride(0),
                stride_ch=cand_idx2.stride(0),
                stride_cr=cand_idx2.stride(1),
                stride_oh=probs.stride(0), stride_or=probs.stride(1),
                Hkv=hk, G=g, D=kq.shape[2], BT=tile,
                num_warps=num_warps)

        if stream is not None and stream is not cur_stream:
            cur_stream.wait_stream(stream)
        return probs, unique


def topk_selection_from_probabilities(
    cand_idx2: torch.Tensor,
    probabilities: torch.Tensor,
    unique: torch.Tensor,
    lower: torch.Tensor | int,
    upper: torch.Tensor | int,
    quota: torch.Tensor | int,
    capacity: int,
    width: int,
    sink: int = 0,
    *,
    stream: torch.cuda.Stream | None = None,
):
    """Pure-Triton local Top-K on deduplicated candidates.

    ``lower``/``upper``/``quota`` are device or host scalar ints.  Returns
    ``(logical_ids [Hkv,width] int64, counts [Hkv] int32)``.
    """
    if not _has_triton():
        raise RuntimeError("Triton is not available")
    hk, r = cand_idx2.shape
    if probabilities.shape != (hk, r) or unique.shape != (hk, r):
        raise ValueError("probability/unique shape mismatch")
    if not all(t.is_cuda and t.is_contiguous()
               for t in (cand_idx2, probabilities, unique)):
        raise ValueError("candidate ranking inputs must be contiguous CUDA tensors")
    if width < 1 or width > 65536:
        raise ValueError("invalid packed Top-K width")
    if capacity < 1 or capacity >= 65536:
        raise ValueError("packed Top-K requires capacity < 65536")
    if probabilities.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("packed Top-K requires BF16/FP16 probabilities")

    dev = cand_idx2.device

    def scalar(x):
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError("ranking bounds must be scalar")
            return x.to(device=dev, dtype=torch.int64).reshape(())
        return torch.tensor(int(x), dtype=torch.int64, device=dev)

    low = scalar(lower)
    high = scalar(upper)
    q = scalar(quota)

    next_r = max(1, triton.next_power_of_2(r))
    out = torch.empty((hk, width), dtype=torch.int64, device=dev)
    counts = torch.empty((hk,), dtype=torch.int32, device=dev)

    with torch.cuda.device(dev):
        cur_stream = torch.cuda.current_stream()
        if stream is not None and stream is not cur_stream:
            stream.wait_stream(cur_stream)
        use_stream = stream if stream is not None else cur_stream
        with torch.cuda.stream(use_stream):
            _packed_topk_kernel[(hk,)](
                cand_idx2, probabilities, unique, low, high, q, out, counts,
                R=r,
                stride_ih=cand_idx2.stride(0), stride_ir=cand_idx2.stride(1),
                stride_ph=probabilities.stride(0),
                stride_pr=probabilities.stride(1),
                stride_vh=unique.stride(0), stride_vr=unique.stride(1),
                stride_oh=out.stride(0), stride_or=out.stride(1),
                S=capacity, K=width, BLOCK_R=next_r, SINK=sink)
        if stream is not None and stream is not cur_stream:
            cur_stream.wait_stream(stream)
    return out, counts


def logical_to_physical(
    logical_ids: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    *,
    request_idx: int = 0,
    stream: torch.cuda.Stream | None = None,
):
    """Convert logical token ids to absolute physical slots."""
    if not _has_triton():
        raise RuntimeError("Triton is not available")
    if logical_ids.ndim != 2 or logical_ids.dtype != torch.int64:
        raise ValueError("logical_ids must be [Hkv,K] int64")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be [num_requests,max_blocks] int32")
    hk, width = logical_ids.shape
    pad = block_table.shape[1]
    out = torch.empty_like(logical_ids, dtype=torch.int32)

    dev = logical_ids.device
    with torch.cuda.device(dev):
        cur_stream = torch.cuda.current_stream()
        if stream is not None and stream is not cur_stream:
            stream.wait_stream(cur_stream)
        use_stream = stream if stream is not None else cur_stream
        row = block_table[request_idx]
        with torch.cuda.stream(use_stream):
            _logical_to_physical_kernel[(hk,)](
                logical_ids, row, out,
                K=width, PAGE=page_size,
                LOG2_PAGE=_pow2_or_minus_one(page_size),
                MAX_BLOCKS=pad,
                PAD_PAGE=0,
                stride_ih=logical_ids.stride(0),
                stride_ir=logical_ids.stride(1),
                stride_bt_req=block_table.stride(0),
                stride_bt_page=block_table.stride(1),
                stride_oh=out.stride(0), stride_or=out.stride(1))
        if stream is not None and stream is not cur_stream:
            cur_stream.wait_stream(stream)
    return out


def fused_fine_topk(
    kq: torch.Tensor,
    q_fine: torch.Tensor,
    cand_idx2: torch.Tensor,
    keep2: torch.Tensor,
    cand_len: torch.Tensor,
    capacity: int,
    group_size: int,
    lower: torch.Tensor | int,
    upper: torch.Tensor | int,
    quota: torch.Tensor | int,
    width: int,
    *,
    sink: int = 0,
    block_table: torch.Tensor | None = None,
    page_size: int = 64,
    request_idx: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> FineTopKResult:
    """End-to-end opt-in fused fine scorer, local Top-K and optional physical map.

    This is the API intended to replace the production candidate-normalization
    + ranking chain when ``NFA_FINE_TOPK_TRITON=1``.
    """
    probs, unique = fused_fine_probabilities(
        kq, q_fine, cand_idx2, keep2, cand_len, capacity, group_size,
        stream=stream)
    logical, counts = topk_selection_from_probabilities(
        cand_idx2, probs, unique, lower, upper, quota, capacity, width,
        sink=sink, stream=stream)
    physical = None
    if block_table is not None:
        physical = logical_to_physical(
            logical, block_table, page_size, request_idx=request_idx,
            stream=stream)
    return FineTopKResult(
        probabilities=probs, unique=unique, logical_ids=logical,
        physical_slots=physical, counts=counts)
