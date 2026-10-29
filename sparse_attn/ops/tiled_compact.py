"""Optional candidate-domain stable partition with parallel row tiles.

Two passes, O(H*ceil(R/1024)) count scratch. Inactive IDs are redirected before
MID loads. Keep both the active prefix and inactive suffix in original order.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _count(KEEP,COUNTS,R:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(0).to(tl.int64);t=tl.program_id(1)
    r=t*B+tl.arange(0,B)
    keep=tl.load(KEEP+h*R+r,r<R,other=0).to(tl.int32)
    tl.store(COUNTS+h*T+t,tl.sum(keep,0))

@triton.jit
def _partition(IDS,KEEP,MID,COUNTS,OUT,KO,MO,N,R:tl.constexpr,T:tl.constexpr,
               TB:tl.constexpr,S:tl.constexpr,B:tl.constexpr,WITH_MID:tl.constexpr):
    h=tl.program_id(0).to(tl.int64);t=tl.program_id(1)
    tiles=tl.arange(0,TB)
    counts=tl.load(COUNTS+h*T+tiles,tiles<T,other=0)
    total=tl.sum(counts,0);before=tl.sum(tl.where(tiles<t,counts,0),0)
    r=t*B+tl.arange(0,B);valid=r<R
    keep=tl.load(KEEP+h*R+r,valid,other=0).to(tl.int32)
    ids=tl.load(IDS+h*R+r,valid,other=0)
    ids=tl.where(keep!=0,ids,S-1)
    local=tl.cumsum(keep,0)
    position=tl.where(keep!=0,before+local-1,total.to(tl.int64)+r-before-local)
    tl.store(OUT+h*R+position,ids,valid)
    tl.store(KO+h*R+position,keep!=0,valid)
    if WITH_MID:
        mid=tl.load(MID+ids,valid,other=0)
        tl.store(MO+h*R+position,mid!=0,valid)
    if t==0:tl.store(N+h,total.to(tl.int64))


def compact_tiled(ids,keep,mid,capacity):
    if (ids.ndim!=2 or keep.shape!=ids.shape or ids.dtype!=torch.int64
            or keep.dtype!=torch.bool or not ids.is_contiguous() or not keep.is_contiguous()
            or keep.device!=ids.device or ids.shape[0]<1
            or type(capacity) is not int or capacity<1 or ids.shape[1]>65536):
        raise ValueError('tiled compact requires contiguous int64 IDs and bool keep[H,R]')
    if mid is not None and (mid.shape!=(capacity,) or mid.dtype!=torch.bool
            or mid.device!=ids.device or not mid.is_contiguous()):
        raise ValueError('mid must be contiguous bool[capacity]')
    if not ids.is_cuda:raise ValueError('tiled compact requires CUDA')
    h,r=ids.shape
    out=torch.empty_like(ids);ko=torch.empty_like(keep)
    mo=torch.empty_like(keep) if mid is not None else None
    if r==0:return out,ko,mo,torch.zeros(h,device=ids.device,dtype=torch.int64)
    n=torch.empty(h,device=ids.device,dtype=torch.int64)
    b=1024;t=triton.cdiv(r,b)
    counts=torch.empty(h,t,device=ids.device,dtype=torch.int32)
    _count[(h,t)](keep,counts,r,t,b,num_warps=4)
    _partition[(h,t)](ids,keep,mid if mid is not None else keep,counts,out,ko,
        mo if mo is not None else ko,n,r,t,triton.next_power_of_2(t),capacity,b,mid is not None,num_warps=4)
    return out,ko,mo,n
