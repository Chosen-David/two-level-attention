"""Actual TLI selection over low-dimensional scratch, full-width paged attention."""
import os
import torch
from .paged_region_kernels import paged_attention


def region_selection(indexer,q,layer,scale):
    idx=indexer
    if os.environ.get('TLI_FINE_GATHER','1')!='0':raise ValueError('paged reference requires TLI_FINE_GATHER=0')
    if (idx.subspace!='tail' or idx.l2_subspace!='same' or idx.gamma is not None or idx._basis_all is not None
        or idx.far_select!='4bit' or idx.near_select!='4bit' or idx.per_q_head or idx.enable_async):
        raise ValueError('paged reference supports ordinary shared-head tail TLI')
    length=layer.logical_length
    if layer.skip_near and layer.skip_far:
        # Near/Far 均已物理回收，唯一合法支持集是永久保护的 Sink+SWA。
        # 不构造投影、不运行 L1/L2，也不通过 top-k 重新发现已删除区域。
        logical=torch.arange(length,device=q.device)
        protected=(logical<idx.sink_tokens)|(logical>=max(0,length-idx.sliding_window_size))
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
    cache.update_for_paged_decode(k,v,module.layer_idx,dict(cache_position=cache_position))
    module.indexer.skip_far=layer.skip_far
    mask,receipt=region_selection(module.indexer,q,layer,module.scaling)
    layer.tli_receipt=receipt
    return paged_attention(layer,q,mask,module.scaling,ring_arithmetic=True)
