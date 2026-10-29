"""Batched candidate scoring with the existing selector's reduction order.

Queries already contain the model-dtype scale rounding, then FP32 conversion.
Only this scoring stage is batched here; this is not a full batched indexer.
"""
import torch


def score_candidates_batched(k, query, ids, counts, *, out_dtype):
    if (k.ndim != 4 or query.ndim != 3 or ids.ndim != 3 or counts.ndim != 2
            or k.shape[0] < 1 or k.shape[1] < 1 or k.shape[2] < 1
            or k.shape[-1] not in (16,32,64,128,256)
            or query.shape[0] != k.shape[0] or query.shape[-1] != k.shape[-1]
            or query.shape[1] < 1 or query.shape[1] % k.shape[2]
            or ids.shape[:2] != (k.shape[0],k.shape[2]) or counts.shape != ids.shape[:2]
            or ids.shape[-1] < 1 or k.shape[0]*k.shape[2] > 65535
            or k.dtype not in (torch.bfloat16,torch.float16)
            or out_dtype not in (torch.bfloat16,torch.float16)
            or query.dtype != torch.float32 or ids.dtype != torch.int64 or counts.dtype != torch.int64
            or any(t.device != k.device for t in (query,ids,counts))
            or k.stride(-1) != 1 or not all(t.is_contiguous() for t in (query,ids,counts))):
        raise ValueError('unsupported batched candidate score shape/dtype/stride')
    if not k.is_cuda:
        raise ValueError('batched candidate scoring requires CUDA')
    from sparse_attn.indexer.tli_indexer import _score_fine_cand_kernel, TLIIndexer
    import triton
    b,cap,h,d=k.shape;hq=query.shape[1];width=ids.shape[-1]
    rank=torch.arange(width,device=k.device)
    torch._assert_async(((counts>=0)&(counts<=width)).all(),'invalid candidate count')
    torch._assert_async(((rank>=counts[...,None])|((ids>=0)&(ids<cap))).all(),
                        'active candidate ID out of bounds')
    out=torch.empty((b,hq,width),device=k.device,dtype=out_dtype)
    tile,warps=TLIIndexer._TILE_FINE.get(d,(64,4))
    _score_fine_cand_kernel[(triton.cdiv(width,tile),b*h)](
        k,query,ids,counts,out,width,k.stride(1),k.stride(2),query.stride(1),
        ids.stride(1),ids.stride(2),out.stride(1),out.stride(2),
        Hkv=h,G=hq//h,D=d,BT=tile,BATCHED=True,STRIDE_KB=k.stride(0),CAP=cap,
        num_warps=warps)
    return out
