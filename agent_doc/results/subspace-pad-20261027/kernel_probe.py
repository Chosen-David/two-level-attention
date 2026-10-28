"""Existing Triton full-stat kernel interpreter pairs; not hardware evidence."""
import json
import torch
from exp.bench.subspace_pad_compare import make_indexer
from sparse_attn.indexer.utils import prepare_pad_mask,pad_tensor
from sparse_attn.indexer.tli_indexer import _HAS_TRITON
assert _HAS_TRITON
rows=[]
for dtype in (torch.float32,torch.float16):
    for mode in ('tail','mla_rope','full'):
        for lengths in ([1],[63],[64],[65],[33,31,7]):
            torch.manual_seed(271027)
            dim=192 if mode=='mla_rope' else 128
            k=torch.randn(1,2,sum(lengths),dim,dtype=dtype).transpose(1,2)
            cu=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0)),dtype=torch.int32)
            ix=make_indexer(mode)
            ids=ix._subspace_indices(k.device) if ix.enable_subspace else None
            mask,_=prepare_pad_mask(64,cu)
            padded=pad_tensor(k,mask)
            old=padded[...,ids] if ids is not None else padded
            new=pad_tensor(k[...,ids] if ids is not None else k,mask)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            a=ix._fast_stat_full(old,None,True)
            b=ix._fast_stat_full(new,None,True)
            for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
            rows.append(dict(dtype=str(dtype),mode=mode,lengths=lengths,
                             old_stride=list(old.stride()),new_stride=list(new.stride()),exact=True))
print(json.dumps(dict(scope='Triton interpreter CPU fp32/fp16 only; no CUDA/BF16 or performance claim',cases=rows),indent=2))
