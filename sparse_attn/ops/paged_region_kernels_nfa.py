"""Compact NFA selection over low-dimensional scratch and page-table attention."""
import argparse
import os
import torch
from sparse_attn.arguments import add_sparse_attn_args
from .paged_region_kernels import paged_attention, paged_attention_selected
from sparse_attn.indexer.selection.selector import TokenSelection, canonical_selection

INVALID_GATHER_IDX = -1


def nfa_args(subspace='tail'):
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    return parser.parse_args(['--method', 'nfa', '--nfa_far_method', 'minmax',
        '--nfa_near_method', 'avg', '--nfa_near_pages', '32',
        '--nfa_near_tokens', 'off', '--nfa_near_len', '4096',
        '--nfa_subspace', subspace, '--tia_level2_cmp_ratio', '4',
        '--nfa_sink_tokens', '4', '--nfa_swa_size', '128',
        '--tia_level1_topk', '128', '--tia_level2_topk', '1024',
        '--nfa_enable_layer_skip', 'false', '--nfa_far_skip_backend', 'legacy',
        '--nfa_far_skip_cache', 'retain'])


def _validate_indexer(idx):
    if os.environ.get('NFA_FINE_GATHER','1')!='0':
        raise ValueError('paged reference requires NFA_FINE_GATHER=0')
    supported_subspace=(idx.subspace=='tail' or
                        (idx.subspace=='nope' and idx.head_dim==256))
    if (not supported_subspace or idx.l2_subspace!='same' or idx.gamma is not None or idx._basis_all is not None
        or idx.far_select!='4bit' or idx.near_select!='4bit' or idx.per_q_head or idx.enable_async):
        raise ValueError('paged reference supports shared-head tail NFA or GLM D=256 NoPE64 only')


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
        selected_per_kv_head=mask.sum(-1).flatten().tolist(),scorer='NFAIndexer.compute_score',selector='NFAIndexer.compute_mask',
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


def region_selection_compact(indexer,q,layer,scale,*,build_receipt=True):
    """Compact paged selection: coarse/fine scoring then selected-token IDs only.

    This is the production paged-region path.  It never materializes the
    legacy [1,1,Hkv,length] bool mask and never calls compute_mask.
    """
    idx=indexer
    _validate_indexer(idx)
    length=layer.logical_length
    device=q.device
    key_heads=next(iter(layer.pages.values()))[1].shape[1]
    if layer.skip_near and layer.skip_far:
        parts=[torch.arange(a,b,device=device,dtype=torch.int32)
               for a,b in layer.intervals()]
        ids=torch.cat(parts) if parts else torch.empty(
            (0,),dtype=torch.int32,device=device)
        width=ids.numel()
        selection=TokenSelection(ids[None].expand(1,key_heads,-1).contiguous(),
                                 torch.full((1,key_heads),width,dtype=torch.int32,device=device))
        receipt=None
        if build_receipt:
            receipt=dict(subspace=idx.subspace,dimensions=[],logical_tokens=length,
                visible_tokens=width,selected_per_kv_head=[width]*key_heads,
                scorer='protected_only',selector='protected_only',skip_near=True,skip_far=True,
                l1_calls=0,l2_calls=0,selection_topk_calls=0,kv_route='direct_page_table',
                full_width_gather_bytes=0,index_scratch_bytes=0)
        return selection,receipt
    dims=idx._subspace_indices(device);bs=idx.args.tia_block_size
    visible=_visible_mask(layer,device)
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
    receipt=None
    if build_receipt:
        receipt=dict(subspace=idx.subspace,dimensions=dims.tolist(),logical_tokens=length,
            visible_tokens=int(visible.sum()),selected_per_kv_head=selection.counts[0].tolist(),
            scorer='NFAIndexer.compute_selection_graph',selector='NFAIndexer.compute_selection_graph',
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


def _protected_selection_batch(cache, layer_idx, req_ids, indexer):
    """Build sink+SWA selections for a whole batch in one vectorized pass.

    This is only valid when every request in this layer has both near and far
    skipped; no indexer scoring or host scalar reads are required.
    """
    batch = len(req_ids)
    lengths = [
        (cache.pending_lengths.get(rid, cache.states[rid].logical_length)
         if rid >= 0 and rid in cache.states else 0)
        for rid in req_ids
    ]
    max_len = max(lengths) if lengths else 0
    protected_width = min(
        max_len, indexer.sink_tokens + indexer.sliding_window_size)
    if max_len == 0 or protected_width == 0:
        device = cache.gather_indices_buf.device
        return (
            torch.zeros((batch, 0), dtype=torch.int32, device=device),
            torch.zeros((batch,), dtype=torch.int32, device=device),
        )

    device = cache.gather_indices_buf.device
    length_t = torch.tensor(lengths, dtype=torch.long, device=device)
    if max_len <= indexer.sink_tokens + indexer.sliding_window_size:
        col_full = torch.arange(max_len, dtype=torch.long, device=device)
        valid = col_full[None, :] < length_t[:, None]
        ids = torch.where(
            valid, col_full[None, :].expand(batch, -1),
            torch.full_like(col_full[None, :].expand(batch, -1), -1))
        counts = length_t.to(torch.int32)
        return ids.to(torch.int32), counts

    col = torch.arange(protected_width, dtype=torch.long, device=device)
    sink_count = torch.minimum(
        length_t, torch.tensor(indexer.sink_tokens, device=device))
    total_count = torch.minimum(
        length_t,
        torch.tensor(indexer.sink_tokens + indexer.sliding_window_size,
                     device=device))
    swa_count = total_count - sink_count

    in_sink = col[None, :] < sink_count[:, None]
    in_swa = (col[None, :] >= sink_count[:, None]) & (
        col[None, :] < total_count[:, None])
    swa_col = col[None, :] - sink_count[:, None]
    # Short rows may overlap sink/SWA even when another row is long.
    swa_id = length_t[:, None] - swa_count[:, None] + swa_col
    ids = torch.where(
        in_sink,
        col[None, :].expand(batch, -1),
        torch.where(in_swa, swa_id, torch.zeros_like(swa_id)))
    ids = ids.to(torch.int32)
    counts = total_count.to(torch.int32)
    return ids, counts


def _selected_logical_ids_graph_ready(cache, layer_idx, request_id, indexer,
                                      q_row, scale, length):
    """Run the production NFA selector and return one shared ID sequence.

    The selector is shared-head in the validated production config, so the
    first KV-head selection is the request-level token set used by the gather
    kernel.
    """
    layer_view = _GraphReadyLayerView(cache, layer_idx, request_id, length)
    selection, _ = region_selection_compact(
        indexer, q_row, layer_view, scale, build_receipt=False)
    ids = selection.indices[0, 0].contiguous()
    count = selection.counts[0, 0].contiguous()
    return ids, count


def _prepare_nfa_gather_for_layer(cache, layer_idx, req_ids, q, indexer, scale):
    """Overwrite per-request gather buffers with NFA-selected logical IDs.

    This is the eager NFA integration point.  The fixed buffers are filled on
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
        ids, count = _selected_logical_ids_graph_ready(
            cache, layer_idx, request_id, indexer, q_row, scale, length)
        torch._assert_async(count <= cache.max_visible_len)
        width = ids.numel()
        valid_cols = torch.arange(width, device=ids.device) < count
        safe_ids = torch.where(valid_cols, ids, INVALID_GATHER_IDX)
        cache._store_gather_slots(layer_idx, b, safe_ids, width)
        vis_buf[b].copy_(count, non_blocking=True)


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
    layer.nfa_receipt=receipt
    return paged_attention_selected(layer,q,selection,module.scaling,ring_arithmetic=True)


def decode_graph_ready(module, q, cache, *, gather_indices=None,
                       visible_lens=None, query_to_req=None, scale=None):
    """Graph-ready sparse gather consumer for the new arena cache.

    ``q`` may be flat [T, HQ, D] or [B, 1, HQ, D]; the latter is flattened to
    the kernel contract.  This path never builds a dense attention mask and
    consumes precomputed absolute physical slots.
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
    query_to_req = cache.query_to_req_buf if query_to_req is None else query_to_req

    # The frozen-runner completion receipt only needs proof that this layer
    # executed the sparse gather path.  The actual selected token count is the
    # fixed-width visible-lens row for this request slot.
    layer.nfa_receipt = {
        'route': 'graph_ready_sparse',
        'visible_lens': visible_lens[:cache.max_requests].detach().cpu().tolist(),
    }

    return sparse_paged_attention_gather(
        layer,
        q,
        gather_indices,
        visible_lens,
        query_to_req,
        scale=scale,
    )



def project_visible_varlen_kernel(kv_pool_k, block_table, lengths,
                                  skip_far, skip_near, dims, pad,
                                  *, sink, near_len, window):
    """Batch low-dimensional visible-K projection over the graph-ready arena.

    Grid is one program per request.  The kernel writes ``[B, pad, HK, D]`` in
    token order, zeroing non-visible positions exactly like the eager
    ``project_visible_for_request`` buffer contract.
    """
    import triton
    import triton.language as tl

    device = kv_pool_k.device
    batch = lengths.numel()
    hk = kv_pool_k.shape[2]
    page = kv_pool_k.shape[3]
    head_dim = kv_pool_k.shape[4]
    dims_numel = dims.numel()

    if device.type != 'cuda':
        raise RuntimeError('project_visible_varlen_kernel requires CUDA')
    if pad <= 0:
        raise ValueError('pad must be positive')

    out = torch.empty(
        (batch, pad, hk, dims_numel),
        dtype=kv_pool_k.dtype, device=device)
    block_table = block_table.contiguous()
    dims = dims.contiguous().to(device)

    @triton.jit
    def _kernel(
        kv_ptr, bt_ptr, len_ptr, skip_far_ptr, skip_near_ptr,
        out_ptr, dims_ptr,
        PAD, PAGE: tl.constexpr, SINK: tl.constexpr,
        NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, HK: tl.constexpr, HEAD_DIM: tl.constexpr,
        DIM_NUMEL: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        req = tl.program_id(1)

        length = tl.load(len_ptr + req).to(tl.int32)
        skip_far = tl.load(skip_far_ptr + req).to(tl.int32)
        skip_near = tl.load(skip_near_ptr + req).to(tl.int32)
        far_on = skip_far == 0
        near_on = skip_near == 0

        sink_hi = tl.minimum(SINK, length)
        near_hi = tl.maximum(sink_hi, length - WINDOW)
        tmp = length - WINDOW - NEAR_LEN
        far_blk = tmp // PAGE
        far_hi = tl.minimum(
            near_hi, tl.maximum(sink_hi, far_blk * PAGE))

        total = PAD * HK * DIM_NUMEL
        row = pid * BLOCK + tl.arange(0, BLOCK)
        out_mask = row < total

        tok = row // (HK * DIM_NUMEL)
        rem = row % (HK * DIM_NUMEL)
        head = rem // DIM_NUMEL
        dim_idx = rem % DIM_NUMEL

        visible = tok < sink_hi
        visible = visible | (
            far_on & (tok >= sink_hi) & (tok < far_hi))
        visible = visible | (
            near_on & (tok >= far_hi) & (tok < near_hi))
        visible = visible | (
            (tok >= near_hi) & (tok < length))

        lp = tok // PAGE
        off = tok % PAGE
        page_valid = lp < MAX_BLOCKS
        phys = tl.load(
            bt_ptr + req * MAX_BLOCKS + lp,
            mask=out_mask & page_valid, other=0).to(tl.int64)
        dim = tl.load(
            dims_ptr + dim_idx, mask=out_mask, other=0).to(tl.int64)

        src = (kv_ptr
               + phys * HK * PAGE * HEAD_DIM
               + head * PAGE * HEAD_DIM
               + off * HEAD_DIM
               + dim)
        load_mask = out_mask & visible & page_valid
        val = tl.load(src, mask=load_mask, other=0.0)
        tl.store(
            out_ptr + req * PAD * HK * DIM_NUMEL + row,
            val, mask=out_mask)

    block = 256
    grid = (triton.cdiv(pad * hk * dims_numel, block), batch)
    _kernel[grid](
        kv_pool_k, block_table, lengths, skip_far, skip_near,
        out, dims,
        PAD=pad, PAGE=page, SINK=sink, NEAR_LEN=near_len, WINDOW=window,
        MAX_BLOCKS=block_table.shape[1], HK=hk, HEAD_DIM=head_dim,
        DIM_NUMEL=dims_numel, BLOCK=block,
        num_warps=4,
    )
    return out
