"""Actual TLI selection over low-dimensional scratch, full-width paged attention."""
import argparse
import os
import torch
from sparse_attn.arguments import add_sparse_attn_args
from .paged_region_kernels import paged_attention


def tli_args(subspace='tail'):
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    return parser.parse_args(['--method', 'tli', '--tli_far_method', 'minmax',
        '--tli_near_method', 'avg', '--tli_near_pages', '32',
        '--tli_near_tokens', 'off', '--tli_near_len', '4096',
        '--tli_subspace', subspace, '--tia_level2_cmp_ratio', '4',
        '--tli_sink_tokens', '4', '--tli_swa_size', '128',
        '--tia_level1_topk', '128', '--tia_level2_topk', '1024',
        '--tli_enable_layer_skip', 'false', '--tli_far_skip_backend', 'legacy',
        '--tli_far_skip_cache', 'retain'])


def _validate_indexer(idx):
    if os.environ.get('TLI_FINE_GATHER','1')!='0':
        raise ValueError('paged reference requires TLI_FINE_GATHER=0')
    supported_subspace=(idx.subspace=='tail' or
                        (idx.subspace=='nope' and idx.head_dim==256))
    if (not supported_subspace or idx.l2_subspace!='same' or idx.gamma is not None or idx._basis_all is not None
        or idx.far_select!='4bit' or idx.near_select!='4bit' or idx.per_q_head or idx.enable_async):
        raise ValueError('paged reference supports shared-head tail TLI or GLM D=256 NoPE64 only')


def region_selection(indexer,q,layer,scale):
    idx=indexer
    _validate_indexer(idx)
    length=layer.logical_length
    if layer.skip_near and layer.skip_far:
        # Near/Far 均已物理回收，唯一合法支持集就是缓存实际可见的 Sink+SWA。
        # 直接使用缓存几何，禁止 indexer 的另一套几何重新选中已回收页面。
        # 不构造投影、不运行 L1/L2，也不通过 top-k 重新发现已删除区域。
        protected=torch.zeros(length,device=q.device,dtype=torch.bool)
        for a,b in layer.intervals():protected[a:b]=True
        key_heads=next(iter(layer.pages.values()))[1].shape[1]
        mask=protected[None,None,None].expand(1,1,key_heads,length)
        receipt=dict(subspace=idx.subspace,dimensions=[],logical_tokens=length,
            visible_tokens=int(mask[0,0,0].sum()),selected_per_kv_head=mask.sum(-1).flatten().tolist(),
            scorer='protected_only',selector='protected_only',skip_near=True,skip_far=True,
            l1_calls=0,l2_calls=0,selection_topk_calls=0,kv_route='direct_page_table',
            full_width_gather_bytes=0,index_scratch_bytes=0)
        return mask,receipt
    dims=idx._subspace_indices(q.device);bs=idx.args.tia_block_size
    projected=layer.project_visible(dims,bs);pages=projected.reshape(1,-1,bs,projected.shape[2],dims.numel())
    fine=projected[:,:length]
    if idx.l2_quant:fine=idx.min_max_per_token_quant(fine)
    dictionary=dict(k_min=pages.amin(2),k_max=pages.amax(2),k_avg=pages.mean(2),k_qat=fine,
        cu_seqlens_k_coarse=torch.tensor([0,pages.shape[1]],device=q.device,dtype=torch.int32),
        cu_seqlens_k_fine=torch.tensor([0,length],device=q.device,dtype=torch.int32),valid_length=length)
    qids=torch.tensor([length-1],device=q.device,dtype=torch.int32)
    scores=idx.compute_score(q.transpose(1,2).contiguous(),qids,dictionary,scale)
    if scores.get('cand_idx') is not None:raise ValueError('candidate-only scorer unsupported')
    visible=torch.zeros(length,device=q.device,dtype=torch.bool)
    for a,b in layer.intervals():visible[a:b]=True
    scores['score_fine'].masked_fill_(~visible,float('-inf'))
    mask=idx.compute_mask(qids,scores)&visible
    receipt=dict(subspace=idx.subspace,dimensions=dims.tolist(),logical_tokens=length,visible_tokens=int(visible.sum()),
        selected_per_kv_head=mask.sum(-1).flatten().tolist(),scorer='TLIIndexer.compute_score',selector='TLIIndexer.compute_mask',
        skip_near=layer.skip_near,skip_far=layer.skip_far,kv_route='direct_page_table',
        full_width_gather_bytes=0,index_scratch_bytes=projected.numel()*projected.element_size())
    return mask,receipt


def decode(module,q,k,v,cache,cache_position):
    layer=cache.layers[module.layer_idx]
    if not layer.frozen:raise ValueError('freeze before paged decode')
    # 所有确定性配置/形状准入必须先于 KV 提交；失败后同一 cache_position
    # 仍可修正配置并重试，不得把请求留在已追加但未 decode 的半完成状态。
    _validate_indexer(module.indexer)
    if (q.ndim!=4 or k.ndim!=4 or q.shape[0]!=1 or q.shape[2]!=1
            or k.shape[1]<1 or q.shape[1]%k.shape[1] or q.shape[-1]!=k.shape[-1]):
        raise ValueError('paged decode requires B1 single-token GQA with matching QK width')
    if q.dtype!=k.dtype or q.device!=k.device:
        raise ValueError('paged decode requires matching Q/K dtype and device')
    cache.update_for_paged_decode(k,v,module.layer_idx,dict(cache_position=cache_position))
    module.indexer.skip_far=layer.skip_far
    mask,receipt=region_selection(module.indexer,q,layer,module.scaling)
    layer.tli_receipt=receipt
    return paged_attention(layer,q,mask,module.scaling,ring_arithmetic=True)
