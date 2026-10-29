"""Exact batching of the primary min/max coarse kernel, including rounded mean.

Only the coarse score stage is batched; callers retain pool/candidate semantics.
Queries must already have the primary path's scale rounding and FP32 conversion.
"""
import torch


def score_coarse_batched(kmin, kmax, query, *, out_dtype, return_mean=False):
    if (kmin.ndim != 4 or kmax.shape != kmin.shape or query.ndim != 3
            or min(kmin.shape[:3]) < 1 or kmin.shape[-1] not in (16, 32, 64, 128, 256)
            or query.shape[0] != kmin.shape[0] or query.shape[-1] != kmin.shape[-1]
            or query.shape[1] < 1 or query.shape[1] % kmin.shape[2]
            or kmin.shape[0] * kmin.shape[2] > 65535
            or kmin.dtype not in (torch.bfloat16, torch.float16) or kmax.dtype != kmin.dtype
            or out_dtype not in (torch.bfloat16, torch.float16) or query.dtype != torch.float32
            or kmax.device != kmin.device or query.device != kmin.device
            or kmin.stride(-1) != 1 or kmax.stride(-1) != 1 or not query.is_contiguous()):
        raise ValueError('unsupported batched coarse score shape/dtype/stride')
    batch, tokens, heads, dim = kmin.shape
    group = query.shape[1] // heads
    if return_mean and group & (group - 1):
        raise ValueError('fused coarse mean requires power-of-two GQA group')
    if not kmin.is_cuda:
        raise ValueError('batched coarse scoring requires CUDA')
    import triton
    from sparse_attn.indexer.tli_indexer import TLIIndexer, _score_coarse_kernel
    out = torch.empty((batch, query.shape[1], tokens), device=kmin.device, dtype=out_dtype)
    mean = torch.empty((batch, heads, tokens), device=kmin.device, dtype=out_dtype) if return_mean else out
    tile, warps = TLIIndexer._TILE_COARSE.get(dim, (64, 4))
    _score_coarse_kernel[(triton.cdiv(tokens, tile), batch * heads)](
        kmin, kmax, query, out, mean, tokens,
        kmin.stride(1), kmin.stride(2), kmax.stride(1), kmax.stride(2),
        query.stride(1), out.stride(1), out.stride(2), mean.stride(1), mean.stride(2),
        Hkv=heads, G=group, D=dim, BT=tile, HAS_MEAN=return_mean,
        BATCHED=True, STRIDE_KB=kmin.stride(0), STRIDE_KB2=kmax.stride(0), num_warps=warps)
    return (out, mean) if return_mean else out
