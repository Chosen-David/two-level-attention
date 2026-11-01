"""Compact TLI selection over low-dimensional scratch and page-table attention."""
import argparse
import os
import torch
from sparse_attn.arguments import add_sparse_attn_args
from .paged_region_kernels import paged_attention, paged_attention_selected
from .token_selection import TokenSelection, canonical_selection

INVALID_GATHER_IDX = -1


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


def _visible_mask(layer, device):
    length=layer.logical_length
    visible=torch.zeros(length,device=device,dtype=torch.bool)
    for a,b in layer.intervals():visible[a:b]=True
    return visible


def _protected_ids(layer,indexer,device):
    length=layer.logical_length
    tokens=torch.arange(length,device=device)
    visible=_visible_mask(layer,device)
    swa_lo=max(0,length-indexer.sliding_window_size)
    protected=visible & ((tokens < indexer.sink_tokens) | (tokens >= swa_lo))
    return tokens[protected].to(torch.int32)


def _combine_compact(selection, protected_ids, key_heads, length, device):
    """Merge compact middle IDs with protected sink/SWA IDs into one padded set."""
    selected=selection.indices[0]
    selected_counts=selection.counts[0]
    sel_width=selected.shape[-1]
    sel_valid=torch.arange(sel_width,device=device)[None] < selected_counts[:,None]
    prot=protected_ids.expand(key_heads,-1)
    prot_valid=torch.ones_like(prot,dtype=torch.bool)
    ids=torch.cat((selected,prot),dim=-1)
    valid=torch.cat((sel_valid,prot_valid),dim=-1)
    return canonical_selection(ids[None],valid[None],length,width=ids.shape[-1])


def region_selection_compact(indexer,q,layer,scale):
    """Compact paged selection: coarse/fine scoring then selected-token IDs only.

    This is the production paged-region path.  It never materializes the
    legacy [1,1,Hkv,length] bool mask and never calls compute_mask.
    """
    idx=indexer
    _validate_indexer(idx)
    length=layer.logical_length
    device=q.device
    visible=_visible_mask(layer,device)
    key_heads=next(iter(layer.pages.values()))[1].shape[1]
    if layer.skip_near and layer.skip_far:
        ids=torch.arange(length,device=device)[visible].to(torch.int32)
        width=ids.numel()
        selection=TokenSelection(ids[None].expand(1,key_heads,-1).contiguous(),
                                 torch.full((1,key_heads),width,dtype=torch.int32,device=device))
        receipt=dict(subspace=idx.subspace,dimensions=[],logical_tokens=length,
            visible_tokens=width,selected_per_kv_head=[width]*key_heads,
            scorer='protected_only',selector='protected_only',skip_near=True,skip_far=True,
            l1_calls=0,l2_calls=0,selection_topk_calls=0,kv_route='direct_page_table',
            full_width_gather_bytes=0,index_scratch_bytes=0)
        return selection,receipt
    dims=idx._subspace_indices(device);bs=idx.args.tia_block_size
    projected=layer.project_visible(dims,bs)
    pages=projected.reshape(1,-1,bs,projected.shape[2],dims.numel())
    fine=projected[:,:length]
    if idx.l2_quant:fine=idx.min_max_per_token_quant(fine)
    dictionary=dict(k_min=pages.amin(2),k_max=pages.amax(2),k_avg=pages.mean(2),k_qat=fine,
        cu_seqlens_k_coarse=torch.tensor([0,pages.shape[1]],device=device,dtype=torch.int32),
        cu_seqlens_k_fine=torch.tensor([0,length],device=device,dtype=torch.int32),valid_length=length)
    vl=torch.tensor([length],device=device,dtype=torch.int64)
    middle=idx.compute_selection_graph(q.transpose(1,2).contiguous(),dictionary,scale,vl,
                                       selection_mask=visible)
    selection=_combine_compact(middle,_protected_ids(layer,idx,device),key_heads,length,device)
    receipt=dict(subspace=idx.subspace,dimensions=dims.tolist(),logical_tokens=length,
        visible_tokens=int(visible.sum()),selected_per_kv_head=selection.counts[0].tolist(),
        scorer='TLIIndexer.compute_selection_graph',selector='TLIIndexer.compute_selection_graph',
        skip_near=layer.skip_near,skip_far=layer.skip_far,kv_route='direct_page_table',
        full_width_gather_bytes=0,index_scratch_bytes=projected.numel()*projected.element_size())
    return selection,receipt


class _GraphReadyLayerView:
    """Adapt one GraphReady request/layer to the legacy compact selector."""

    def __init__(self, cache, layer_idx, request_id, length):
        self.cache = cache
        self.layer_idx = layer_idx
        self.request_id = request_id
        self._length = length
        state = cache.states[request_id]
        self.skip_far = state.skip_far_by_layer[layer_idx]
        self.skip_near = state.skip_near_by_layer[layer_idx]
        layer = cache.layers[layer_idx]
        self.pages = {0: (0, layer.kv_pool_k[0])}

    @property
    def logical_length(self):
        return self._length

    def intervals(self, length=None, resident=False, request_id=None):
        return self.cache._intervals_for_layer(
            self.cache.states[self.request_id], self.layer_idx,
            self._length if length is None else length)

    def project_visible(self, dims, block_size):
        return self.cache.project_visible_for_request(
            self.layer_idx, self.request_id, dims, block_size,
            length=self._length)


def _selected_logical_ids_graph_ready(cache, layer_idx, request_id, indexer,
                                      q_row, scale, length):
    """Run the production TLI selector and return one shared ID sequence.

    The selector is shared-head in the validated production config, so the
    first KV-head selection is the request-level token set used by the gather
    kernel.
    """
    layer_view = _GraphReadyLayerView(cache, layer_idx, request_id, length)
    selection, _ = region_selection_compact(indexer, q_row, layer_view, scale)
    selected = selection.indices[0, 0]
    count = int(selection.counts[0, 0].item())
    if count <= 0:
        return torch.empty((0,), dtype=torch.int32, device=q_row.device)
    return selected[:count].contiguous()


def _prepare_tli_gather_for_layer(cache, layer_idx, req_ids, q, indexer, scale):
    """Overwrite per-request gather buffers with TLI-selected logical IDs.

    This is the eager TLI integration point.  The fixed buffers are filled on
    GPU from the selector output; the sparse attention kernel still loops only
    ``visible_lens`` (selected count).
    """
    layer_buf = cache.gather_indices_buf[layer_idx]
    vis_buf = cache.visible_lens_buf[layer_idx]
    layer_buf.fill_(INVALID_GATHER_IDX)
    vis_buf.zero_()
    for b, request_id in enumerate(req_ids):
        if request_id < 0 or request_id not in cache.states:
            continue
        state = cache.states[request_id]
        length = cache.pending_lengths.get(request_id, state.logical_length)
        q_row = q[b:b + 1].reshape(1, 1, q.shape[1], q.shape[2])
        ids = _selected_logical_ids_graph_ready(
            cache, layer_idx, request_id, indexer, q_row, scale, length)
        if ids.numel() > cache.max_visible_len:
            raise RuntimeError(
                f'request {request_id} layer {layer_idx} selected count '
                f'{ids.numel()} exceeds max_visible_len {cache.max_visible_len}')
        if ids.numel():
            layer_buf[b, :ids.numel()].copy_(ids, non_blocking=True)
        vis_buf[b] = ids.numel()


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
    selection,receipt=region_selection_compact(module.indexer,q,layer,module.scaling)
    layer.tli_receipt=receipt
    return paged_attention_selected(layer,q,selection,module.scaling,ring_arithmetic=True)


def decode_graph_ready(module, q, cache, *, gather_indices=None,
                       visible_lens=None, block_table=None,
                       query_start_loc=None, scale=None):
    """Graph-ready sparse gather consumer for the new arena cache.

    ``q`` may be flat [T, HQ, D] or [B, 1, HQ, D]; the latter is flattened to
    the kernel contract.  This path never builds a dense attention mask.
    """
    from .paged_region_kernels import sparse_paged_attention_gather

    layer_idx = module.layer_idx
    layer = cache.layers[layer_idx]
    scale = module.scaling if scale is None else scale

    if q.ndim == 4:
        if q.shape[2] == 1 and q.shape[1] != 1:
            q = q[:, :, 0].contiguous()
        elif q.shape[1] == 1:
            q = q[:, 0].contiguous()
        else:
            raise ValueError('graph-ready decode expects one query per row')
    if q.ndim != 3:
        raise ValueError('graph-ready decode expects [T,HQ,D] query')

    gather_indices = (
        cache.gather_indices_buf[layer_idx]
        if gather_indices is None else gather_indices)
    visible_lens = (
        cache.visible_lens_buf[layer_idx]
        if visible_lens is None else visible_lens)
    block_table = cache.block_table_buf if block_table is None else block_table
    query_start_loc = (
        cache.query_start_loc_buf
        if query_start_loc is None else query_start_loc)

    return sparse_paged_attention_gather(
        layer,
        q,
        gather_indices,
        visible_lens,
        block_table,
        query_start_loc,
        scale=scale,
    )
