"""Experimental exact BF16 normalization; full-domain ranking unchanged.

Repeated IDs must have identical scores, as produced by the TLI fine scorer.
Pinned to PyTorch 2.8 generic/shared CUDA softmax; warm up before graph capture.
"""
import os
from pathlib import Path
from functools import lru_cache
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load


@triton.jit
def _candidate_owners(IDS, KEEP, OWNERS, R: tl.constexpr, S: tl.constexpr,
                      BLOCK: tl.constexpr):
    kv = tl.program_id(0)
    r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + kv * R + r, r < R, other=0)
    keep = tl.load(KEEP + kv * R + r, r < R, other=0)
    # Integer ownership gives each token exactly one writer, even when repeated
    # slots cross CTA boundaries. No concurrent same-value floating-point store.
    tl.atomic_max(OWNERS + kv * S + ids, r,
                  (r < R) & keep & (ids >= 0) & (ids < S), sem="relaxed")



@triton.jit
def _compact_gqa_mean(CP,OWNER,OUT,S:tl.constexpr,R:tl.constexpr,G:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0);w=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    owner=tl.load(OWNER+h*S+w,w<S,other=-1)
    valid=(w<S)&(owner>=0)
    acc=tl.zeros((BLOCK,),tl.float32)
    # Same sequential GQA sum and BF16 cast as the accepted main kernel.
    for g in range(G):
        v=tl.load(CP+(h*G+g)*R+owner,valid,other=0.).to(tl.float32)
        v=tl.where(v!=v,0.,v)
        acc+=v
    val=(acc/G).to(tl.bfloat16)
    tl.store(OUT+h*S+w,val,w<S)

@lru_cache(maxsize=1)
def build():
    if torch.__version__.split('+')[0]!='2.8.0':raise RuntimeError('requires pinned torch2.8.0 generic softmax')
    os.environ.setdefault('MAX_JOBS','1');root=Path(__file__).resolve().parent/'csrc'
    return load(name='tli_virtual_softmax_compact_v5',sources=[str(root/'binding.cpp'),str(root/'virtual_softmax_compact.cu')],
                extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-std=c++17'],verbose=True)

def probabilities(extension,scores,ids,keep,capacity,group,excluded=None):
    if scores.device != ids.device or scores.device != keep.device:
        raise ValueError('same CUDA device required')
    if scores.dtype!=torch.bfloat16:raise ValueError('compact mean mirrors BF16-only native mean; other dtypes keep accepted fallback')
    if not all(t.is_cuda and t.is_contiguous() for t in (scores,ids,keep)):raise ValueError('contiguous CUDA tensors required')
    hk,r=ids.shape
    if group<1 or group&(group-1) or scores.shape!=(hk*group,r) or keep.shape!=ids.shape:raise ValueError('GQA candidate shapes invalid')
    if excluded is not None and (excluded.device != scores.device or excluded.dtype != torch.bool
            or not excluded.is_contiguous() or excluded.shape != (capacity,)):
        raise ValueError('contiguous same-device token-domain bool exclusion mask required')
    if r==0:
        p = torch.zeros((hk,capacity),device=scores.device,dtype=scores.dtype)
        return p if excluded is None else p.masked_fill(excluded, -torch.inf)
    mask = torch.empty(0, device=scores.device, dtype=torch.bool) if excluded is None else excluded
    owner=torch.full((hk,capacity),-1,device=scores.device,dtype=torch.int32)
    _candidate_owners[(hk,triton.cdiv(r,256))](ids,keep,owner,r,capacity,BLOCK=256)
    if os.environ.get('TLI_COMPACT_FUSED_MEAN', '1') == '1':
        return extension.virtual_compact_softmax(scores,owner,capacity,group,True,mask)
    compact=extension.virtual_compact_softmax(scores,owner,capacity,group,False,mask)
    p=torch.empty((hk,capacity),device=scores.device,dtype=scores.dtype)
    _compact_gqa_mean[(hk,triton.cdiv(capacity,1024))](compact,owner,p,capacity,r,group,BLOCK=1024,num_warps=4)
    return p if excluded is None else p.masked_fill(excluded, -torch.inf)


def supported(scores, ids, keep, capacity, group):
    """Pinned generic/shared softmax geometry preserves reference rounding."""
    if torch.__version__.split('+')[0] != '2.8.0' or scores.dtype != torch.bfloat16:
        return False
    if scores.device != ids.device or scores.device != keep.device:
        return False
    if not all(t.is_cuda and t.is_contiguous() for t in (scores, ids, keep)):
        return False
    if ids.dtype not in (torch.int32, torch.int64) or keep.dtype != torch.bool:
        return False
    if ids.ndim != 2 or keep.shape != ids.shape or scores.shape != (ids.shape[0]*group, ids.shape[1]):
        return False
    if group < 1 or group & (group-1) or not 9216 < capacity < 2**31:
        return False
    if ids.shape[0]*capacity >= 2**31 or ids.shape[1] >= 2**31:
        return False
    return True


def compact_probabilities(scores, ids, keep, capacity, group, excluded=None):
    if not supported(scores, ids, keep, capacity, group):
        raise ValueError('unsupported exact compact normalization geometry')
    with torch.cuda.device(scores.device):
        if torch.cuda.is_current_stream_capturing() and build.cache_info().currsize == 0:
            raise RuntimeError('warm up compact normalization before CUDA Graph capture')
        return probabilities(build(), scores, ids, keep, capacity, group, excluded)
