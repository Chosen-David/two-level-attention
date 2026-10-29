"""Fused pinned BF16/FP16 unique-candidate ranking for <=65535 tokens."""
import torch
import triton
import triton.language as tl
from .token_selection import TokenSelection


@triton.jit
def _tile_keys(IDS,P,VALID,LOW,HIGH,QUOTA,OUT,R:tl.constexpr,S:tl.constexpr,
               GROUPS:tl.constexpr,K:tl.constexpr):
    h=tl.program_id(0);tile=tl.program_id(1);i=tl.arange(0,2*K)
    r=tile*(2*K)+i
    token=tl.load(IDS+h*R+r,r<R,other=0)
    probability=tl.load(P+h*R+r,r<R,other=0)
    keep=tl.load(VALID+h*R+r,r<R,other=0)
    good=keep&(token>=tl.load(LOW))&(token<tl.load(HIGH))&(probability>0)&(tl.load(QUOTA)>0)
    bits=probability.to(tl.uint16,bitcast=True).to(tl.int32)
    key=tl.where(good,(bits<<16)+(S-token).to(tl.int32),0)
    ordered=tl.sort(key,descending=True)
    tl.store(OUT+(h*GROUPS+tile)*K+i,ordered,i<K)


@triton.jit
def _merge_topk(IN,OUT,GROUPS:tl.constexpr,NEXT:tl.constexpr,K:tl.constexpr):
    h=tl.program_id(0);tile=tl.program_id(1);i=tl.arange(0,2*K)
    group=2*tile+(i>=K).to(tl.int32)
    rank=tl.where(i<K,i,2*K-1-i)
    value=tl.load(IN+(h*GROUPS+group)*K+rank,group<GROUPS,other=0)
    # Two descending runs become a bitonic sequence by reversing the second.
    for bit in tl.static_range(10,-1,-1):
        stride=1<<bit
        peer=tl.gather(value,i^stride,axis=0)
        value=tl.where((i&stride)==0,tl.maximum(value,peer),tl.minimum(value,peer))
    tl.store(OUT+(h*NEXT+tile)*K+i,value,i<K)


def tree_topk(ids,p,valid,lower,upper,quota,capacity,width):
    hk,r=ids.shape
    groups=triton.cdiv(r,2*width)
    current=torch.empty((hk,groups,width),device=ids.device,dtype=torch.int32)
    _tile_keys[(hk,groups)](ids,p,valid,lower,upper,quota,current,r,capacity,groups,width,num_warps=8)
    while groups>1:
        following=triton.cdiv(groups,2)
        output=torch.empty((hk,following,width),device=ids.device,dtype=torch.int32)
        _merge_topk[(hk,following)](current,output,groups,following,width,num_warps=8)
        current=output;groups=following
    return current.view(hk,width)


@triton.jit
def _finish(TOP,LOW,HIGH,QUOTA,OUT,COUNT,S:tl.constexpr,K:tl.constexpr,SINK:tl.constexpr):
    h=tl.program_id(0);i=tl.arange(0,K)
    q=tl.load(QUOTA).to(tl.int32)
    if q==0:
        tl.store(OUT+h*K+i,0)
        tl.store(COUNT+h,0)
    else:
        key=tl.load(TOP+h*K+i)
        positive=(key>>16)>0
        chosen=tl.where((i<q)&positive,S-(key&65535),S)
        npositive=tl.sum(((i<q)&positive).to(tl.int32),0)
        if npositive==q:
            fast_packed=tl.sort(tl.where(chosen>=SINK,chosen,S),descending=False)
            tl.store(OUT+h*K+i,tl.where(fast_packed<S,fast_packed,0))
            tl.store(COUNT+h,tl.sum((fast_packed<S).to(tl.int32),0))
        else:
            chosen=tl.sort(chosen,descending=False)
            low=tl.load(LOW).to(tl.int32);high=tl.load(HIGH).to(tl.int32)
            boundaries=tl.where(chosen<S,chosen-low-i,S)
            lo=tl.zeros((K,),tl.int32);hi=tl.full((K,),K,tl.int32)
            for _ in range(11):
                mid=lo+(hi-lo)//2
                value=tl.gather(boundaries,tl.minimum(mid,K-1),axis=0)
                right=value<=i
                active=lo<hi
                lo=tl.where(active&right,mid+1,lo)
                hi=tl.where(active&~right,mid,hi)
            zero=low+i+lo
            zero=tl.where((i<q-npositive)&(zero<high),zero,S)
            combined=tl.cat(chosen,zero,can_reorder=True)
            combined=tl.where(combined>=SINK,combined,S)
            packed=tl.sort(combined,descending=False)
            j=tl.arange(0,2*K)
            tl.store(OUT+h*K+j,tl.where(packed<S,packed,0),j<K)
            tl.store(COUNT+h,tl.sum((packed<S).to(tl.int32),0))


def rank_unique(ids,p,valid,lower,upper,quota,*,capacity,width,sink):
    hk,r=ids.shape
    top=tree_topk(ids,p,valid,lower,upper,quota,capacity,width)
    out=torch.empty((1,hk,width),device=ids.device,dtype=torch.int32)
    counts=torch.empty((1,hk),device=ids.device,dtype=torch.int32)
    _finish[(hk,)](top,lower,upper,quota,out,counts,capacity,width,sink,num_warps=8)
    return TokenSelection(out,counts)
