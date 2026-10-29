"""Direct token IDs into the original eager consumer's ordered reduction.

The union pack replaces only its full-width boolean-mask scan. Kernel B, query
scale rounding, tile width and launch geometry remain the reference operations.
"""
import os
import torch
import triton
import triton.language as tl
from .eager_decoding import _eager_attn_kernel_b, _err_flag, _triton_kv_layout_compatible


@triton.jit
def _union(SELECTED,COUNTS,LENGTH,IDS,OUT_COUNTS,FLAG,
           H:tl.constexpr,K:tl.constexpr,CAP:tl.constexpr,SINK:tl.constexpr,
           SWA:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);r=tl.arange(0,BLOCK)
    length=tl.load(LENGTH+row//H);count=tl.load(COUNTS+row)
    token=tl.load(SELECTED+row*K+r,r<K,other=CAP)
    active=(r<K)&(r<count)
    bad=(count<0)|(count>K)|(length<1)|(length>CAP)
    bad=bad|(tl.sum((active&((token<0)|(token>=length))).to(tl.int32),0)>0)
    if bad:tl.atomic_max(FLAG,4)
    prefix=tl.minimum(tl.maximum(length,0),SINK)
    start=tl.maximum(prefix,length-SWA)
    protected=tl.where(r<K+SINK,r-K,start+r-K-SINK)
    keep_protected=((r>=K)&(r<K+SINK)&(protected<prefix))|((r>=K+SINK)&(r<K+SINK+SWA)&(protected<length))
    chosen=tl.where(active&(token>=0)&(token<length)&(token<CAP),token,CAP)
    chosen=tl.where(keep_protected&(protected>=0)&(protected<CAP),protected,chosen)
    ordered=tl.sort(chosen,descending=False)
    previous=tl.gather(ordered,tl.maximum(r-1,0),0)
    unique=(ordered<CAP)&((r==0)|(ordered!=previous))
    positions=tl.cumsum(unique.to(tl.int32),0)-1
    total=tl.sum(unique.to(tl.int32),0)
    if total==0:tl.atomic_max(FLAG,1)
    if total>W:tl.atomic_max(FLAG,2)
    tl.store(IDS+row*W+positions,ordered,unique&(positions<W))
    tl.store(OUT_COUNTS+row,total)


def _int32_addressable(t):
    # Kernel B multiplies program/token IDs by strides in int32. numel alone
    # cannot bound a strided view's relative addresses (storage holes count).
    return (all(s>=0 for s in t.stride()) and
            sum((n-1)*s for n,s in zip(t.shape,t.stride()))<=torch.iinfo(torch.int32).max)


def ordered_attention(q,k,v,selection,lengths,*,sink,swa,scale):
    ids,counts=selection
    if (q.ndim!=4 or q.shape[1]!=1 or k.ndim!=4 or q.shape[0]!=k.shape[0]
            or q.shape[-1]!=128 or k.shape[-1]!=128 or k.shape[2]<1
            or q.shape[2]<1 or q.shape[2]%k.shape[2] or q.dtype not in (torch.bfloat16,torch.float16)
            or k.dtype!=q.dtype or v.dtype!=q.dtype or not q.is_cuda
            or not _triton_kv_layout_compatible(k,v) or k.stride(0)!=v.stride(0)
            or not _int32_addressable(k) or q.numel()>torch.iinfo(torch.int32).max
            or q.stride(-1)!=1
            or k.stride(-1)!=1 or not 1<=q.shape[0]<=8
            or any(t.requires_grad for t in (q,k,v)) or not 16384<=k.shape[1]<65536):
        raise ValueError('ordered attention requires supported fixed BF16/FP16 D128 B1..8 CUDA geometry')
    b,h= k.shape[0],k.shape[2]
    if (ids.ndim!=3 or ids.shape[:2]!=(b,h) or not 0<ids.shape[-1]<=4096
            or ids.dtype!=torch.int32 or counts.shape!=(b,h) or counts.dtype!=torch.int32
            or lengths.shape!=(b,) or lengths.dtype!=torch.int64
            or any(t.device!=q.device for t in (k,v,ids,counts,lengths))
            or any(not t.is_contiguous() for t in (ids,counts,lengths))
            or type(sink) is not int or type(swa) is not int or min(sink,swa)<0
            or ids.shape[-1]+sink+swa>8192):
        raise ValueError('invalid ordered selection metadata')
    width=min(int(os.environ.get('TLI_E127_W_UPPER','2048')),k.shape[1])
    if width<1:raise ValueError('positive attention width required')
    packed=torch.empty((b,h,width),device=q.device,dtype=torch.int32)
    packed_counts=torch.empty_like(counts)
    _union[(b*h,)](ids,counts,lengths,packed,packed_counts,_err_flag(q.device),
        h,ids.shape[-1],k.shape[1],sink,swa,width,
        BLOCK=triton.next_power_of_2(ids.shape[-1]+sink+swa),num_warps=8)
    # Preserve the model-dtype multiplication BEFORE conversion to FP32.
    qs=(q[:,0]*scale).to(torch.float32).contiguous()
    out=torch.empty((b,q.shape[2],128),device=q.device,dtype=q.dtype)
    group=q.shape[2]//h
    _eager_attn_kernel_b[(b*h*group,)](packed,packed_counts,qs,k,v,out,h,
        qs.stride(0),k.stride(0),k.stride(1),k.stride(2),qs.stride(1),out.stride(0),
        G=group,D=128,PER_QH=False,BW=256,W=width,num_warps=4)
    return out[:,None]
