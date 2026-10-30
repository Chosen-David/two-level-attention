"""Exact positive BF16 kth threshold; zeros are implicit, IDs unique."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_kth(P, IDS, KEEP, OUT, S:tl.constexpr, R:tl.constexpr,
              K:tl.constexpr, BLOCK:tl.constexpr):
    h=tl.program_id(0);r=tl.arange(0,BLOCK)
    token=tl.load(IDS+h*R+r,r<R,other=0)
    active=tl.load(KEEP+h*R+r,r<R,other=0)&(token>=0)&(token<S)
    value=tl.load(P+h*S+token,(r<R)&active,other=0.)
    positive=(r<R)&active&(value>0.)
    code=value.to(tl.uint16,bitcast=True).to(tl.int32)
    bins=tl.arange(0,256)
    upper=code>>8
    hist=tl.histogram(upper,256,mask=positive)
    tail=tl.cumsum(hist,0,reverse=True)
    high=tl.max(tl.where(tail>=K,bins,0),0)
    above=tl.sum(tl.where(bins>high,hist,0),0)
    low_hist=tl.histogram(code&255,256,mask=positive&(upper==high))
    low_tail=tl.cumsum(low_hist,0,reverse=True)
    low=tl.max(tl.where(low_tail>=K-above,bins,0),0)
    kth=tl.where(tl.sum(positive.to(tl.int32),0)>=K,(high<<8)|low,0)
    tl.store(OUT+h,kth.to(tl.uint16).to(tl.bfloat16,bitcast=True))


def bf16_kth(p,ids,keep,k):
    result=torch.empty((p.shape[0],),device=p.device,dtype=p.dtype)
    _bf16_kth[(p.shape[0],)](p,ids,keep,result,p.shape[1],ids.shape[1],k,
                            BLOCK=max(1,triton.next_power_of_2(ids.shape[1])),num_warps=8)
    return result
