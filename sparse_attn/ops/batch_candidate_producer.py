"""Experimental exact batch L1/candidate producer, preserving primary ordering.

Index-feature maintenance is owned elsewhere. This module performs one batched
coarse/average score, native rounded GQA mean, two last-axis TopKs, integer page
expansion and one flattened stable compaction. Native tie equivalence remains a
GPU gate; this must not be substituted for the old numerical parallel indexer.
"""
from typing import NamedTuple
import torch
from .batch_coarse_score import score_coarse_batched, score_average_batched
from .cuda_graph_state import geometry_cache_key


class BatchedCandidateInputs(NamedTuple):
    k_fine: torch.Tensor
    q_fine: torch.Tensor
    ids: torch.Tensor
    keep: torch.Tensor
    counts: torch.Tensor
    lengths: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    quota: torch.Tensor
    capacity: int
    group_size: int
    output_dtype: torch.dtype
    rank_width: int
    sink: int
    swa: int


class CandidateGeometry(NamedTuple):
    far_pool: torch.Tensor       # [B,P]
    near_pool: torch.Tensor      # [B,P]
    bypass_ids: torch.Tensor     # [B,2,bs]
    bypass_keep: torch.Tensor
    lower: torch.Tensor         # [B]
    upper: torch.Tensor
    quota: torch.Tensor


def candidate_geometry(lengths, pages, offsets, *, block_size, sink, swa, near_len, k2):
    """Tensor-only batching of _build_graph_geometry's compact-domain fields."""
    bs=block_size
    if (type(bs) is not int or bs<1 or lengths.ndim!=1 or lengths.dtype!=torch.int64
            or pages.ndim!=1 or pages.dtype!=torch.int64 or offsets.shape!=(bs,)
            or offsets.dtype!=torch.int64 or min(lengths.numel(),pages.numel())<1
            or any(t.device!=lengths.device for t in (pages,offsets))):
        raise ValueError('invalid batched candidate geometry metadata')
    lo=sink//bs
    swalo=(lengths-swa).clamp(min=0)
    effective=(lengths+bs-1)//bs
    near=((lengths-swa-max(bs,near_len))//bs).clamp(min=lo)
    stop=torch.maximum(near,effective-max(1,swa//bs))
    bypage=swalo//bs
    byok=(swalo%bs!=0)&(bypage>=lo)
    if sink%bs:byok=byok&(bypage!=lo)
    far=(pages[None]>=lo)&(pages[None]<near[:,None])
    near_pool=(pages[None]>=near[:,None])&(pages[None]<stop[:,None])
    if sink%bs:far=far&(pages[None]!=lo)
    near_pool=near_pool&~((pages[None]==bypage[:,None])&byok[:,None])
    if sink%bs:
        sink_in_near=(pages[None]==lo)&(pages[None]>=near[:,None])&(pages[None]<stop[:,None])
        near_pool=near_pool&~sink_in_near
    slot0=torch.full_like(bypage,lo if sink%bs else 0)
    blocks=torch.stack((slot0,bypage),dim=-1)
    byids=blocks[...,None]*bs+offsets[None,None]
    conditions=torch.stack((torch.full_like(byok,bool(sink%bs)),byok),dim=-1)
    bykeep=(byids<lengths[:,None,None])&conditions[...,None]
    lower=torch.full_like(lengths,lo*bs)
    upper=torch.maximum(lower,swalo)
    quota=(lengths.clamp(max=k2)-sink-(lengths-swalo)).clamp(min=0)
    return CandidateGeometry(far,near_pool,byids,bykeep,lower,upper,quota)


def compact_candidates_exact(ids,keep,capacity):
    """Same sentinel and stable ordering as primary _iter3_compact, B*H CTAs."""
    if (ids.ndim!=3 or keep.shape!=ids.shape or ids.dtype!=torch.int64
            or keep.dtype!=torch.bool or ids.device!=keep.device
            or not ids.is_contiguous() or not keep.is_contiguous()
            or min(ids.shape)<1 or ids.numel()>=2**31):
        raise ValueError('invalid batched stable-compaction inputs')
    b,h,r=ids.shape
    if ids.is_cuda:
        from ..indexer.tli_indexer import _iter3_compact
        out,valid,_,counts=_iter3_compact(ids.view(b*h,r),keep.view(b*h,r),None,
                                        r,b*h,capacity,ids.device)
        return out.view(b,h,r),valid.view(b,h,r),counts.view(b,h)
    # CPU reference of the exact primary fallback, including inactive tail IDs.
    safe=torch.where(keep,ids,capacity-1)
    order=torch.argsort(~keep,dim=-1,stable=True)
    return safe.gather(-1,order),keep.gather(-1,order),keep.sum(-1).to(torch.int64)


def select_candidates(coarse,average,geometry,lengths,offsets,*,capacity,
                      block_size,k1,near_pages,far_method,near_method,sink,swa):
    """One batch pool/TopK/assembly/compaction, no Python request iteration.

    Inputs are already rounded and GQA-averaged [B,Hkv,P] scores. Preserve a
    singleton query axis as in primary TopK, including its invalid-pool slots.
    """
    if coarse.ndim!=3 or min(coarse.shape)<1:
        raise ValueError('coarse scores require [B,Hkv,P]')
    b,h,p=coarse.shape;bs=block_size
    if (type(bs) is not int or bs<1 or type(capacity) is not int or capacity<1
            or lengths.shape!=(b,) or lengths.dtype!=torch.int64
            or geometry.far_pool.shape!=(b,p) or geometry.near_pool.shape!=(b,p)
            or geometry.bypass_ids.shape!=(b,2,bs) or geometry.bypass_keep.shape!=(b,2,bs)
            or offsets.shape!=(bs,) or coarse.dtype not in (torch.float16,torch.bfloat16,torch.float32)
            or (average is not None and (average.shape!=coarse.shape or average.dtype!=coarse.dtype))):
        raise ValueError('incompatible batched candidate pools/scores')
    if far_method not in ('minmax','avg') or near_method not in ('minmax','avg'):
        raise ValueError('batched producer supports minmax/avg pools only')
    far_source=average if far_method=='avg' and average is not None else coarse
    near_source=coarse if near_method=='minmax' or average is None else average
    nf=min(max(0,k1-max(0,near_pages)),p);nn=min(max(0,near_pages),p)
    if b*h*(nf+nn+2)*bs>=2**31:
        raise ValueError('batched candidate address range exceeds supported compaction')
    fp=geometry.far_pool[:,None,None].expand(b,1,h,p)
    np=geometry.near_pool[:,None,None].expand(b,1,h,p)
    fs=torch.where(fp,far_source[:,None],torch.full_like(far_source[:,None],-torch.inf))
    ns=torch.where(np,near_source[:,None],torch.full_like(near_source[:,None],-torch.inf))
    fi=torch.topk(fs,nf,dim=-1).indices
    ni=torch.topk(ns,nn,dim=-1).indices
    blocks=torch.cat((fi,ni),dim=-1)
    in_pool=torch.cat((fp.gather(-1,fi),np.gather(-1,ni)),dim=-1)
    l1_ids=blocks[...,None]*bs+offsets[None,None,None,None]
    l1_keep=(l1_ids<lengths[:,None,None,None,None])&in_pool[...,None]
    bypass_ids=geometry.bypass_ids[:,None,None].expand(b,1,h,2,bs)
    bypass_keep=geometry.bypass_keep[:,None,None].expand(b,1,h,2,bs)
    ids=torch.cat((l1_ids.reshape(b,h,(nf+nn)*bs),bypass_ids.reshape(b,h,2*bs)),dim=-1)
    keep=torch.cat((l1_keep.reshape(b,h,(nf+nn)*bs),bypass_keep.reshape(b,h,2*bs)),dim=-1)
    # Complete the same unique protected denominator as the single-row/eager
    # producer BEFORE fine scoring; protection is not only an attention mask.
    from ..indexer.candidate_domain import complete_protected_candidates
    ids,keep=complete_protected_candidates(ids,keep,lengths,sink,swa,capacity)
    return compact_candidates_exact(ids,keep,capacity)


class BatchedCandidateProducer:
    def __init__(self,indexer,k):
        self.capacity=k.shape[1];self.heads=k.shape[2];self.bs=indexer.args.tia_block_size
        self.k1=indexer.args.tia_level1_topk;self.k2=indexer.args.tia_level2_topk
        self.near_pages=max(0,int(indexer.nt_near_pages));self.near_len=max(self.bs,int(indexer.nt_near_len))
        self.sink=indexer.sink_tokens;self.swa=indexer.sliding_window_size
        self.far_method=indexer.far_method;self.near_method=indexer.near_method
        if self.far_method not in ('minmax','avg') or self.near_method not in ('minmax','avg'):
            raise ValueError('unsupported exact batch pool scoring method')
        self.coarse_dims=indexer._subspace_indices(k.device) if indexer.enable_subspace else None
        self.fine_dims=(None if indexer.l2_subspace=='full' else self.coarse_dims
                        if self.coarse_dims is not None else indexer._qat_indices(k.device))
        self.pages=torch.arange((self.capacity+self.bs-1)//self.bs,device=k.device)
        self.offsets=torch.arange(self.bs,device=k.device)

    def geometry(self,lengths):
        # Reuse primary mutation/capture/stream safety rules; prefix avoids
        # collision with scalar-row geometry stored under the same base pointer.
        from ..indexer.tli_indexer import _graph_geo_lookup,_graph_geo_store
        capturing=lengths.is_cuda and torch.cuda.is_current_stream_capturing()
        key=('exact-batch-candidates-v1',tuple(lengths.shape),str(lengths.device),
             self.capacity,self.bs,self.sink,self.swa,self.near_len,self.k2)
        key=geometry_cache_key(key,capturing=capturing,
            stream=torch.cuda.current_stream(lengths.device).cuda_stream if capturing else None)
        geo=None if key is None else _graph_geo_lookup(lengths,key)
        if geo is None:
            geo=candidate_geometry(lengths,self.pages,self.offsets,block_size=self.bs,
                sink=self.sink,swa=self.swa,near_len=self.near_len,k2=self.k2)
            if key is not None:geo=_graph_geo_store(lengths,key,geo)
        return geo

    def __call__(self,q,fine,summaries,lengths,scale):
        b,_,hq,_=q.shape;g=hq//self.heads;p=self.pages.numel()
        # Elementwise scale is rounded to the model dtype BEFORE FP32 projection.
        full=(q[:,0]*scale).to(torch.float32)
        qc=full if self.coarse_dims is None else full[...,self.coarse_dims]
        # The default L2 subspace reuses the exact L1 index object. Both
        # scoring stages only read Q; sharing avoids a second advanced gather.
        qf=(qc if self.fine_dims is self.coarse_dims else
            full if self.fine_dims is None else full[...,self.fine_dims])
        # Use the same separate native Torch mean as primary graph selection,
        # not the optional sequential-mean output of the coarse primitives.
        raw=score_coarse_batched(summaries['k_min'],summaries['k_max'],qc.contiguous(),out_dtype=q.dtype)
        coarse=raw.view(b,self.heads,g,p).mean(dim=2)
        average=None
        if summaries.get('k_avg') is not None:
            ar=score_average_batched(summaries['k_avg'],qc.contiguous(),out_dtype=q.dtype)
            average=ar.view(b,self.heads,g,p).mean(dim=2)
        geo=self.geometry(lengths)
        ids,keep,counts=select_candidates(coarse,average,geo,lengths,self.offsets,
            capacity=self.capacity,block_size=self.bs,k1=self.k1,near_pages=self.near_pages,
            far_method=self.far_method,near_method=self.near_method,sink=self.sink,swa=self.swa)
        return BatchedCandidateInputs(fine,qf.contiguous(),ids,keep,counts,lengths,
            geo.lower,geo.upper,geo.quota,self.capacity,g,q.dtype,min(self.capacity,self.k2),
            self.sink,self.swa)
