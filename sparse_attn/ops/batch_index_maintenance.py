"""Exact primary tail32/no-QDQ summary maintenance over B*H programs.

The primary kernel retains its default single-request branch/reduction. This
wrapper validates shared-arena metadata and masks inactive requests. The caller
canonicalizes each row's current block during warmup before skipping dead rows.
"""
import torch


def validate_maintenance(k,positions,alive,indices,kmin,kmax,kavg,fine,*,block_size=64):
    """Host metadata only; no device value reads or implicit copies."""
    if (not all(isinstance(t,torch.Tensor) for t in (k,positions,alive,indices,kmin,kmax,fine))
            or (kavg is not None and not isinstance(kavg,torch.Tensor))):
        raise ValueError('maintenance inputs require tensors')
    if (k.ndim!=4 or min(k.shape[:3])<1 or k.shape[-1]!=128
            or k.dtype!=torch.bfloat16 or k.stride(-1)!=1
            or type(block_size) is not int or block_size!=64
            or k.shape[0]*k.shape[2]>65535):
        raise ValueError('batch maintenance requires BF16 K[B,S,Hkv,128] and BS64')
    b,s,h,_=k.shape;p=(s+block_size-1)//block_size
    summaries=(kmin,kmax) if kavg is None else (kmin,kmax,kavg)
    if (positions.shape!=(b,) or positions.dtype!=torch.int64
            or alive.shape!=(b,) or alive.dtype!=torch.bool
            or indices.shape!=(32,) or indices.dtype!=torch.int64
            or any(not t.is_contiguous() for t in (positions,alive,indices))
            or fine.shape!=(b,s,h,32) or not fine.is_contiguous()
            or any(t.shape!=(b,p,h,32) or not t.is_contiguous() for t in summaries)
            or any(t.dtype!=k.dtype for t in (*summaries,fine))
            or any(t.device!=k.device for t in (positions,alive,indices,*summaries,fine))):
        raise ValueError('incompatible batch maintenance arenas/metadata')
    if k.device.type!='meta':
        # Writable arenas must not overlap one another or raw K storage.
        pointers=[t.untyped_storage().data_ptr() for t in (k,*summaries,fine)]
        if len(set(pointers))!=len(pointers):
            raise ValueError('batch maintenance requires separate writable arenas')


def maintenance_route(indexer,k,positions,alive,indices,summaries,fine):
    """Select once during warmup; fallback is explicit, never kernel-exception based."""
    if (not indexer.enable_subspace or indexer.subspace!='tail' or indexer.cmp_ratio!=4
            or indexer.head_dim!=128 or indexer.l2_subspace!='same' or indexer.l2_quant
            or indexer.args.tia_block_size!=64 or not indexer._triton_ok):
        return 'row_fallback:configuration'
    if not k.is_cuda:
        return 'row_fallback:device'
    try:
        validate_maintenance(k,positions,alive,indices,summaries['k_min'],summaries['k_max'],
                             summaries.get('k_avg'),fine)
    except (ValueError,KeyError):
        return 'row_fallback:layout'
    return 'batched_tail32'


def maintain_batched(k,positions,alive,indices,kmin,kmax,kavg,fine,*,block_size=64,check_values=True):
    """Update only active rows; preserve all bytes of dead-request arenas.

    The exact owner passes check_values=False only after its dynamic position
    guard and a warmup check of immutable feature indices. Metadata is always
    validated. Standalone callers retain the default value guards.
    """
    validate_maintenance(k,positions,alive,indices,kmin,kmax,kavg,fine,block_size=block_size)
    if not k.is_cuda:
        raise ValueError('batched primary maintenance requires CUDA')
    if type(check_values) is not bool:
        raise ValueError('check_values must be bool')
    if check_values:
        torch._assert_async(((~alive)|((positions>=0)&(positions<k.shape[1]))).all(),
                            'active maintenance position out of bounds')
        torch._assert_async(((indices>=0)&(indices<128)).all(),'maintenance feature index out of bounds')
    from ..indexer.tli_indexer import _iter3_maint_kernel
    avg=kmin if kavg is None else kavg
    _iter3_maint_kernel[(k.shape[0]*k.shape[2],)](
        k,positions,indices,kmin,kmax,avg,fine,k.shape[1],k.stride(1),k.stride(2),
        kmin.stride(1),kmin.stride(2),fine.stride(1),fine.stride(2),
        NEED_AVG=kavg is not None,BS=block_size,DSUB=32,BATCHED=True,ALIVE=alive,
        Hkv=k.shape[2],STRIDE_KB=k.stride(0),STRIDE_MINB=kmin.stride(0),
        STRIDE_MAXB=kmax.stride(0),STRIDE_AVGB=avg.stride(0),STRIDE_QB=fine.stride(0),num_warps=4)
