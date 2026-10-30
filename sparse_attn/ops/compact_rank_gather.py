"""Parallel canonical TopK gathering, preserving native token-order prefixes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _counts(P, THRESHOLD, MID, COUNTS, S:tl.constexpr, N:tl.constexpr,
            STRIDE:tl.constexpr, BLOCK:tl.constexpr):
    h=tl.program_id(0);b=tl.program_id(1);t=b*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(P+h*S+t,t<S,other=-float('inf'))
    mid=tl.load(MID+t,t<S,other=0)
    kth=tl.load(THRESHOLD+h*STRIDE+STRIDE-1)
    base=(h*N+b)*3
    tl.store(COUNTS+base,tl.sum(((t<S)&(v>kth)).to(tl.int32),0))
    tl.store(COUNTS+base+1,tl.sum(((t<S)&(v==kth)).to(tl.int32),0))
    tl.store(COUNTS+base+2,tl.sum(mid.to(tl.int32),0))


@triton.jit
def _write(P, THRESHOLD, MID, COUNTS, VALUES, IDS, S:tl.constexpr,
           N:tl.constexpr, K:tl.constexpr, STRIDE:tl.constexpr,
           BLOCK:tl.constexpr, CB:tl.constexpr):
    h=tl.program_id(0);b=tl.program_id(1);c=tl.arange(0,CB)
    gc=tl.load(COUNTS+(h*N+c)*3,c<N,other=0)
    ec=tl.load(COUNTS+(h*N+c)*3+1,c<N,other=0)
    mc=tl.load(COUNTS+(h*N+c)*3+2,c<N,other=0)
    short=tl.sum(mc,0)<K
    # If fewer than K eligible tokens, native kth is -inf. All eligible
    # positions precede the excluded -inf tie group, including +0 values.
    gc=tl.where(short,mc,gc)
    ec=tl.where(short,tl.where(c<N,tl.minimum(BLOCK,S-c*BLOCK)-mc,0),ec)
    before_g=tl.sum(tl.where(c<b,gc,0),0)
    before_e=tl.sum(tl.where(c<b,ec,0),0)
    total_g=tl.sum(gc,0)
    kth=tl.load(THRESHOLD+h*STRIDE+STRIDE-1)
    kth=tl.where(short,-float('inf'),kth)
    t=b*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(P+h*S+t,t<S,other=-float('inf'))
    greater=(t<S)&(v>kth);equal=(t<S)&(v==kth)
    rank=tl.where(greater,before_g+tl.cumsum(greater.to(tl.int32),0)-1,
                  total_g+before_e+tl.cumsum(equal.to(tl.int32),0)-1)
    valid=(greater|equal)&(rank<K)
    tl.store(VALUES+h*K+rank,v,valid)
    tl.store(IDS+h*K+rank,t.to(tl.int64),valid)


def canonical_gather(p,threshold,mid,values,ids,stride):
    h,s=p.shape;k=values.shape[1];n=triton.cdiv(s,1024)
    counts=torch.empty((h,n,3),device=p.device,dtype=torch.int32)
    _counts[(h,n)](p,threshold,mid,counts,s,n,stride,BLOCK=1024,num_warps=4)
    _write[(h,n)](p,threshold,mid,counts,values,ids,s,n,k,stride,
                   BLOCK=1024,CB=triton.next_power_of_2(n),num_warps=4)
