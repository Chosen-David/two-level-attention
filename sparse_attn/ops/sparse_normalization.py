"""Undispatched prototype: candidate-only storage with native reduction order.

The token domain is virtual: missing -inf scores contribute zero exponentials.
Map each present token to its pinned native thread and traversal ordinal, then
reduce only present values in that same order. Repeated IDs have equal scores.
"""
from functools import lru_cache
from pathlib import Path
import os
import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def build():
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parent/'csrc'
    os.environ.setdefault('MAX_JOBS','1')
    return load(name='tli_sparse_softmax_v1',sources=[str(root/'sparse_binding.cpp'),str(root/'sparse_softmax.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-std=c++17'],verbose=True)


@triton.jit
def _mean(CP,KEEP,OUT,R:tl.constexpr,G:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0);r=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    valid=(r<R)&tl.load(KEEP+h*R+r,r<R,other=0)
    acc=tl.zeros((BLOCK,),tl.float32)
    for g in range(G):
        v=tl.load(CP+(h*G+g)*R+r,valid,other=0.).to(tl.float32)
        acc+=tl.where(v!=v,0.,v)
    tl.store(OUT+h*R+r,(acc/G).to(tl.bfloat16),r<R)


@triton.jit
def _prepare_keys(IDS,KEEP,KEYS,VALID,R:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0);r=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(IDS+h*R+r,r<R,other=0)
    valid=(r<R)&tl.load(KEEP+h*R+r,r<R,other=0)&(token>=0)&(token<S)
    token=tl.where(valid,token,0)
    end:tl.constexpr=(S//8192)*8192
    worker=tl.where(token<end,(token//8)%1024,(token-end)%1024)
    ordinal=tl.where(token<end,(token//8192)*8+token%8,end//1024+(token-end)//1024)
    key=tl.where(valid,worker*S+ordinal,1024*S)
    tl.store(KEYS+h*R+r,key,r<R)
    tl.store(VALID+h*R+r,valid,r<R)


@triton.jit
def _offsets_and_keys(KEYS,PERM,LONG_KEYS,OFFSETS,UNIQUE,R:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0);r=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    key=tl.load(KEYS+h*R+r,r<R,other=0)
    tl.store(LONG_KEYS+h*R+r,key.to(tl.int64),r<R)
    previous=tl.load(KEYS+h*R+r-1,(r>0)&(r<R),other=-1)
    original=tl.load(PERM+h*R+r,r<R,other=0)
    tl.store(UNIQUE+h*R+original,(key<1024*S)&(key!=previous),r<R)
    valid=r<1025
    target=r.to(tl.int64)*S
    lo=tl.zeros((BLOCK,),tl.int32)
    hi=tl.full((BLOCK,),R,tl.int32)
    while tl.sum((valid&(lo<hi)).to(tl.int32),0)>0:
        active=valid&(lo<hi)
        mid=lo+(hi-lo)//2
        value=tl.load(KEYS+h*R+mid,active,other=0)
        smaller=value<target
        lo=tl.where(active&smaller,mid+1,lo)
        hi=tl.where(active&~smaller,mid,hi)
    tl.store(OFFSETS+h*1025+r,lo,valid)


def candidate_probabilities(scores,ids,keep,capacity,group,*,return_unique=False):
    if torch.__version__.split('+')[0]!='2.8.0' or scores.dtype!=torch.bfloat16:
        raise ValueError('pinned torch2.8.0 BF16 prototype only')
    if not all(t.is_cuda and t.is_contiguous() and t.device==scores.device for t in (scores,ids,keep)):
        raise ValueError('same-device contiguous CUDA inputs required')
    if ids.dtype != torch.int64 or keep.dtype != torch.bool or ids.ndim != 2:
        raise ValueError('int64 IDs and bool keep required')
    hk,r=ids.shape
    if type(capacity) is not int or not 0 < capacity < 2**31-1 or hk < 1 or hk*r >= 2**31-1 or hk*group*r >= 2**31-1 or hk*group > 65535:
        raise ValueError('unsupported capacity/grid geometry')
    if scores.shape!=(hk*group,r) or keep.shape!=ids.shape or group<1 or group&(group-1):
        raise ValueError('invalid candidate geometry')
    if r == 0:
        result=scores.new_empty((hk,0))
        return (result,keep) if return_unique else result
    prop=torch.cuda.get_device_properties(scores.device)
    if capacity%8 or capacity<(prop.shared_memory_per_block-128)//2 or capacity<=9216:
        raise ValueError('only aligned generic BF16 softmax supported')
    with torch.cuda.device(scores.device):
        if torch.cuda.is_current_stream_capturing() and build.cache_info().currsize==0:
            raise RuntimeError('warm up sparse normalization before capture')
        extension = build()
    key_dtype=torch.int32 if 1024*capacity<2**31 else torch.int64
    keys=torch.empty_like(ids,dtype=key_dtype)
    valid=torch.empty_like(keep)
    _prepare_keys[(hk,triton.cdiv(r,256))](ids,keep,keys,valid,r,capacity,BLOCK=256)
    keys,perm=keys.sort(dim=-1,stable=True)
    long_keys=torch.empty_like(ids)
    unique=torch.empty_like(keep)
    offsets=torch.empty((hk,1025),device=ids.device,dtype=torch.int64)
    _offsets_and_keys[(hk,triton.cdiv(max(r,1025),256))](
        keys,perm,long_keys,offsets,unique,r,capacity,BLOCK=256)
    keys=long_keys
    with torch.cuda.device(scores.device):
        cp=extension.sparse_softmax(scores,perm,keys,offsets,group)
        result=torch.empty((hk,r),device=scores.device,dtype=scores.dtype)
        if r: _mean[(hk,triton.cdiv(r,256))](cp,valid,result,r,group,BLOCK=256)
    return (result,unique) if return_unique else result


def supported_geometry(dtype, device, capacity, group, heads):
    if (torch.__version__.split('+')[0] != '2.8.0' or dtype != torch.bfloat16
            or device.type != 'cuda' or not 9216 < capacity < 2**31-1
            or capacity % 8 or group < 1 or group & (group-1)
            or not 1 <= heads*group <= 65535):
        return False
    prop = torch.cuda.get_device_properties(device)
    return capacity >= (prop.shared_memory_per_block-128)//2
