"""GraphReady NFA 的保护集、页投影和 gather attention 入口。"""
import argparse
import torch
from sparse_attn.arguments import add_sparse_attn_args

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


def _protected_selection_batch(lengths, indexer, capacity):
    """按持久长度 buffer 构建固定宽度的 sink+SWA 保护集。

    宽度只能由配置和 Graph 固定容量决定；不能从 capture 时的 Python
    ``max(lengths)`` 推导，否则后续 replay 会沿用旧形状。
    """
    if lengths.ndim != 1:
        raise ValueError('protected selection lengths must be [B]')
    device = lengths.device
    batch = lengths.numel()
    length_t = lengths.to(dtype=torch.long)
    protected_width = min(
        int(capacity), int(indexer.sink_tokens + indexer.sliding_window_size))
    if protected_width == 0:
        return (
            torch.zeros((batch, 0), dtype=torch.int32, device=device),
            torch.zeros((batch,), dtype=torch.int32, device=device),
        )

    col = torch.arange(protected_width, dtype=torch.long, device=device)
    sink_count = torch.minimum(
        length_t, length_t.new_full((), indexer.sink_tokens))
    total_count = torch.minimum(
        length_t,
        length_t.new_full((), indexer.sink_tokens + indexer.sliding_window_size))

    in_sink = col[None, :] < sink_count[:, None]
    in_swa = (col[None, :] >= sink_count[:, None]) & (
        col[None, :] < total_count[:, None])
    swa_col = col[None, :] - sink_count[:, None]
    swa_id = length_t[:, None] - indexer.sliding_window_size + swa_col
    ids = torch.where(
        in_sink,
        col[None, :].expand(batch, -1),
        torch.where(in_swa, swa_id, torch.zeros_like(swa_id)))
    return ids.to(torch.int32), total_count.to(torch.int32)


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
    # Graph capture 中禁止 D2H 同步；replay 不执行 Python，receipt 由 capture
    # 前的 eager warmup 建立，eager 生产步则继续记录真实 visible_lens。
    if not torch.cuda.is_current_stream_capturing():
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
