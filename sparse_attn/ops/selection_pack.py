"""Fuse rank filtering and canonical token-ID packing after unique TopK."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(TOP, QUOTA, MID, LENGTH, IDS, COUNTS,
          K: tl.constexpr, CAP: tl.constexpr, SINK: tl.constexpr,
          SWA: tl.constexpr, PER_HEAD: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    token = tl.load(TOP + h * K + r, r < K, other=CAP)
    quota = tl.load(QUOTA + (h if PER_HEAD else 0))
    length = tl.load(LENGTH)
    in_range = (token >= 0) & (token < CAP)
    mid = tl.load(MID + token, in_range & (r < K), other=0)
    prefix = tl.minimum(length, SINK)
    start = tl.maximum(prefix, length - SWA)
    keep = ((r < K) & (r < quota) & in_range & mid
            & (token >= prefix) & (token < start) & (token < length))
    ordered = tl.sort(tl.where(keep, token, CAP), descending=False)
    valid = ordered < CAP
    tl.store(IDS + h * K + r, tl.where(valid, ordered, 0), r < K)
    tl.store(COUNTS + h, tl.sum(valid.to(tl.int32), 0))


def pack_unique_top(top, quota, mid, length, capacity, sink, swa):
    """Return None for unsupported layouts; never change the TopK tie order.

    TopK supplies distinct positions. Only after rank gating is applied do we
    sort logical IDs, exactly as the Torch canonical-selection reference does.
    """
    if not all(torch.is_tensor(x) for x in (top, quota, mid, length)):
        return None
    if not all(x.is_cuda and x.is_contiguous() and x.device == top.device
               for x in (top, quota, mid, length)):
        return None
    if (top.ndim != 2 or not 0 < top.shape[-1] <= 4096
            or top.dtype not in (torch.int32, torch.int64)
            or quota.dtype not in (torch.int32, torch.int64)
            or length.dtype not in (torch.int32, torch.int64)
            or mid.dtype != torch.bool or mid.ndim != 1
            or mid.numel() != capacity or length.numel() != 1
            or quota.numel() not in (1, top.shape[0])
            or not 0 < capacity < 2**31):
        return None
    heads, width = top.shape
    ids = torch.empty((1, heads, width), dtype=torch.int32, device=top.device)
    counts = torch.empty((1, heads), dtype=torch.int32, device=top.device)
    _pack[(heads,)](top, quota, mid, length, ids, counts, width, capacity,
                   sink, swa, quota.numel() > 1,
                   BLOCK=triton.next_power_of_2(width))
    return ids, counts
