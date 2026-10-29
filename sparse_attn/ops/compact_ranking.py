"""Pinned PyTorch 2.8 CUDA TopK order from a sparse nonnegative domain.

Experimental: compact threshold search followed by token-order gathering and
native small-sort. Input IDs must be unique where keep=True; outside those IDs
p must be +0 (or -inf outside mid). Signed -0 and NaN are unsupported;
probabilities are rounded upstream and NaNs removed, producing +0.
This still scans full p once; it does not claim fully compact normalization.
"""
import os
from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

_last_kth_route = None


@lru_cache(maxsize=1)
def build():
    os.environ.setdefault('MAX_JOBS', '1')
    return load(name='tli_compact_rank_v1',
                sources=[str(Path(__file__).parent/'csrc/rank_binding.cpp')],
                extra_cflags=['-O3'], verbose=True)


@triton.jit
def _gather_candidates(P, IDS, KEEP, C, S:tl.constexpr, R:tl.constexpr,
                       W:tl.constexpr, BLOCK:tl.constexpr):
    h=tl.program_id(0);r=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(IDS+h*R+r,r<R,other=0)
    active=tl.load(KEEP+h*R+r,r<R,other=0)&(token>=0)&(token<S)
    v=tl.load(P+h*S+token,(r<R)&active,other=-float('inf'))
    # K virtual zeros suffice to account for every omitted zero-valued token
    # when finding a positive kth value. The threshold correction below handles
    # fewer than K eligible tokens and never emits these virtual positions.
    v=tl.where(r<R,v,0.)
    tl.store(C+h*W+r,v,r<W)


@triton.jit
def _canonical_gather(P, THRESHOLD, MID, VALUES, IDS, S:tl.constexpr,
                      K:tl.constexpr, STRIDE:tl.constexpr, BLOCK:tl.constexpr):
    h=tl.program_id(0);t=tl.arange(0,BLOCK)
    v=tl.load(P+h*S+t,t<S,other=-float('inf'))
    eligible=tl.load(MID+t,t<S,other=0)
    kth=tl.load(THRESHOLD+h*STRIDE+STRIDE-1)
    kth=tl.where(tl.sum(eligible.to(tl.int32),0)<K,-float('inf'),kth)
    greater=(t<S)&(v>kth);equal=(t<S)&(v==kth)
    ng=tl.sum(greater.to(tl.int32),0)
    rank=tl.where(greater,tl.cumsum(greater.to(tl.int32),0)-1,
                  ng+tl.cumsum(equal.to(tl.int32),0)-1)
    valid=(greater|equal)&(rank<K)
    tl.store(VALUES+h*K+rank,v,valid)
    tl.store(IDS+h*K+rank,t.to(tl.int64),valid)


def compact_topk(p, ids, keep, mid, k):
    """Return native-order token IDs, or None for unsupported static geometry."""
    if (torch.__version__.split('+')[0]!='2.8.0' or p.dtype!=torch.bfloat16
            or p.ndim!=2 or p.shape[0]==0 or ids.ndim!=2 or keep.shape!=ids.shape
            or ids.shape[0]!=p.shape[0] or ids.dtype!=torch.int64
            or keep.dtype!=torch.bool or mid.dtype!=torch.bool
            or mid.shape!=(p.shape[1],) or not 0<k<=min(4096,p.shape[1])
            or p.shape[1]>65536 or ids.shape[1]>=p.shape[1]
            or not all(t.is_cuda and t.is_contiguous() and t.device==p.device
                       for t in (p,ids,keep,mid))):
        return None
    global _last_kth_route
    with torch.cuda.device(p.device):
        if torch.cuda.is_current_stream_capturing() and build.cache_info().currsize==0:
            raise RuntimeError('warm up compact ranking before graph capture')
        extension=build()
        h,s=p.shape;r=ids.shape[1];w=r+k
        if os.environ.get('TLI_COMPACT_KTH', 'histogram') == 'topk':
            compact=torch.empty((h,w),device=p.device,dtype=p.dtype)
            _gather_candidates[(h,triton.cdiv(w,256))](p,ids,keep,compact,s,r,w,BLOCK=256)
            threshold=compact.topk(k,dim=-1).values
            stride=k
            _last_kth_route="compact_torch_topk"
        else:
            from .compact_rank_threshold import bf16_kth
            threshold=bf16_kth(p,ids,keep,k)
            stride=1
            _last_kth_route="bf16_radix_histogram"
        values=torch.empty((h,k),device=p.device,dtype=p.dtype)
        result=torch.empty((h,k),device=p.device,dtype=torch.int64)
        _canonical_gather[(h,)](p,threshold,mid,values,result,s,k,stride,
                               BLOCK=triton.next_power_of_2(s),num_warps=8)
        extension.sort_rank(values,result)
        return result
