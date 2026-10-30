"""Complete the protected domain and stably compact it in one launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _virtual_slot(IDS, KEEP, h, r, length, R: tl.constexpr,
                  HEAD: tl.constexpr, TAIL: tl.constexpr,
                  SINK: tl.constexpr, SWA: tl.constexpr):
    lo = tl.maximum(0, length - SWA)
    old = r < R
    token = tl.load(IDS + h * R + r, old, other=0)
    active = tl.load(KEEP + h * R + r, old, other=0)
    active = active & (token >= SINK) & (token < lo) & (token >= 0)
    head = (r >= R) & (r < R + HEAD)
    tail_id = lo + r - R - HEAD
    new_id = tl.where(head, r - R, tail_id)
    new_valid = ((head & (new_id < length))
                 | ((r >= R + HEAD) & (r < R + HEAD + TAIL)
                    & (new_id < length) & (new_id >= SINK)))
    return tl.where(old, token, new_id), tl.where(old, active, new_valid)


@triton.jit
def _complete_pack(IDS, KEEP, LENGTH, OUT, VALID, COUNTS,
                   R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
                   SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
                   BLOCK: tl.constexpr):
    h = tl.program_id(0)
    width = R + HEAD + TAIL
    length = tl.load(LENGTH)
    total = tl.full((), 0, tl.int32)
    for start in range(0, width, BLOCK):
        r = start + tl.arange(0, BLOCK)
        token, active = _virtual_slot(IDS, KEEP, h, r, length, R, HEAD, TAIL, SINK, SWA)
        total += tl.sum(((r < width) & active).to(tl.int32), 0)
    carry = tl.full((), 0, tl.int32)
    for start in range(0, width, BLOCK):
        r = start + tl.arange(0, BLOCK)
        token, active = _virtual_slot(IDS, KEEP, h, r, length, R, HEAD, TAIL, SINK, SWA)
        active = (r < width) & active
        prefix = tl.cumsum(active.to(tl.int32), 0)
        destination = tl.where(active, carry + prefix - 1,
                               total + r - carry - prefix)
        tl.store(OUT + h * width + destination, tl.where(active, token, CAP - 1), r < width)
        tl.store(VALID + h * width + destination, active, r < width)
        carry += tl.sum(active.to(tl.int32), 0)
    tl.store(COUNTS + h, total.to(tl.int64))


def complete_and_pack(ids, keep, length, sink, swa, capacity):
    """Return None on unsupported layout; integer output equals stable Torch pack."""
    if not all(x.is_cuda and x.is_contiguous() and x.device == ids.device
               for x in (ids, keep, length)):
        return None
    if (ids.ndim != 2 or keep.shape != ids.shape or length.numel() != 1
            or ids.dtype != torch.int64 or keep.dtype != torch.bool
            or length.dtype != torch.int64 or capacity < 1 or sink < 0 or swa < 0):
        return None
    head, tail = min(sink, capacity), min(swa, capacity)
    width = ids.shape[-1] + head + tail
    out = torch.empty((ids.shape[0], width), device=ids.device, dtype=ids.dtype)
    valid = torch.empty_like(out, dtype=torch.bool)
    counts = torch.empty((ids.shape[0],), device=ids.device, dtype=torch.int64)
    _complete_pack[(ids.shape[0],)](ids, keep, length, out, valid, counts,
                                  ids.shape[-1], head, tail, sink, swa, capacity,
                                  BLOCK=1024)
    return out, valid, counts
