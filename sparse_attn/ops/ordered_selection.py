"""Stable middle compaction for producers promising sorted, unique valid IDs.

Invalid slots may appear anywhere. Valid IDs must already be strictly increasing
within each row; generic selectors must keep canonical_selection instead.
"""
import torch
from .token_selection import TokenSelection, middle_selection
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _compact(IDS, VALID, LENGTH, OUT, COUNT, WIDTH:tl.constexpr,
                 HEADS:tl.constexpr, SINK:tl.constexpr, SWA:tl.constexpr,
                 CAP:tl.constexpr, BLOCK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)
        r=tl.arange(0,BLOCK)
        length=tl.load(LENGTH+row//HEADS)
        valid=tl.load(VALID+row*WIDTH+r,r<WIDTH,0)
        ids=tl.load(IDS+row*WIDTH+r,(r<WIDTH)&valid,0)
        lo=tl.minimum(length,SINK)
        hi=tl.maximum(lo,length-SWA)
        keep=(r<WIDTH)&valid&(ids>=0)&(ids<CAP)&(ids>=lo)&(ids<hi)&(ids<length)
        rank=tl.cumsum(keep.to(tl.int32),0)-1
        count=tl.sum(keep.to(tl.int32),0)
        # Prefix writes and tail zeroing target disjoint locations. No
        # initialization/scatter race between warps and no separate fill kernel.
        tl.store(OUT+row*WIDTH+r,0,(r<WIDTH)&(r>=count))
        tl.store(OUT+row*WIDTH+rank,ids.to(tl.int32),keep)
        tl.store(COUNT+row,count)


def ordered_middle_selection(ids, valid, lengths, sink, swa, capacity):
    """Exactly matches middle_selection on the sorted-unique producer contract."""
    if (ids.ndim!=3 or valid.shape!=ids.shape or lengths.numel()!=ids.shape[0]
            or type(capacity) is not int or not 1<=capacity<=torch.iinfo(torch.int32).max):
        raise ValueError('invalid ordered selection dimensions/capacity')
    supported=(triton is not None and ids.is_cuda and ids.shape[0]>0 and ids.shape[1]>0 and 0<ids.shape[-1]<=8192
               and ids.dtype in (torch.int32,torch.int64) and valid.dtype==torch.bool
               and lengths.dtype==torch.int64
               and all(t.device==ids.device and t.is_contiguous() for t in (valid,lengths))
               and ids.is_contiguous() and type(sink) is int and sink>=0
               and type(swa) is int and swa>=0)
    if not supported:
        return middle_selection(ids,valid,lengths,sink,swa,capacity)
    b,h,width=ids.shape
    out=torch.empty_like(ids,dtype=torch.int32)
    counts=torch.empty((b,h),device=ids.device,dtype=torch.int32)
    _compact[(b*h,)](ids,valid,lengths,out,counts,WIDTH=width,HEADS=h,SINK=sink,
                     SWA=swa,CAP=capacity,BLOCK=triton.next_power_of_2(width),
                     num_warps=4 if width<=2048 else 8)
    return TokenSelection(out,counts)
