"""GraphReady NFA 的保护集和 gather attention 入口。"""
import argparse
import torch
import triton
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


def split_gather_buffers(cache, layer_idx, batch):
    """返回固定地址的 protected/middle 物理槽，capture 期间不分配。"""
    workspace = getattr(cache, '_nfa_selector_workspace', None)
    if workspace is None:
        workspace = cache._nfa_selector_workspace = {}
    key = ('split_gather_buffers', layer_idx, batch,
           cache.active_visible_bucket)
    buffers = workspace.get(key)
    if buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('split gather buffers must be warmed before capture')
        layer = cache.layers[layer_idx]
        device = layer.kv_pool_k.device
        shape = (batch, layer.num_kv_heads, cache.active_visible_bucket)
        buffers = (
            torch.empty((2,) + shape, dtype=torch.int32, device=device),
            torch.empty((2, batch, layer.num_kv_heads),
                        dtype=torch.int32, device=device),
        )
        workspace[key] = buffers
    return buffers[0][0], buffers[1][0], buffers[0][1], buffers[1][1]


def build_protected_gather(cache, layer_idx, indexer, gather_out, vis_out):
    """用原 canonical union 的空 middle 特化生成独立保护集。"""
    from ..indexer.nfa_layer_batch_selector import (
        _final_union_physical_kernel, _workspace_empty)
    from ..indexer.selection.unique_ranking import _fast_launch

    workspace = cache._nfa_selector_workspace
    batch = gather_out.shape[0]
    hkv = gather_out.shape[1]
    output_width = gather_out.shape[-1]
    lengths = cache.planned_lengths_buf[:batch]
    valid_req = cache.valid_req_ids_buf[:batch]
    capacity = cache.max_blocks * int(indexer.args.tia_block_size)
    prot_width = min(
        capacity, int(indexer.sink_tokens + indexer.sliding_window_size))
    empty_middle = _workspace_empty(
        workspace, 'empty_middle', (batch * hkv, prot_width),
        torch.int32, gather_out.device)
    empty_counts = _workspace_empty(
        workspace, 'empty_middle_counts', (batch * hkv,),
        torch.int32, gather_out.device)
    union_width = 2 * prot_width
    _fast_launch(
        workspace, ('final_union', layer_idx, 'empty', 'disjoint'),
        _final_union_physical_kernel, (batch * hkv,),
        empty_middle, empty_counts, lengths, valid_req, cache.block_table_buf,
        gather_out, vis_out,
        W=prot_width, PROT=prot_width,
        N=union_width, N_PAD=triton.next_power_of_2(union_width),
        A_PAD=triton.next_power_of_2(prot_width),
        MERGE_BITS=triton.next_power_of_2(union_width).bit_length() - 1,
        SINK=int(indexer.sink_tokens), SWA=int(indexer.sliding_window_size),
        CAP=capacity, MAX_BLOCKS=cache.max_blocks, PAGE=cache.geometry.page,
        MAX_REQ=cache.max_requests, MAX_VIS=output_width,
        INVALID=INVALID_GATHER_IDX, HKV=hkv, B=batch,
        LAYER_OFFSET=layer_idx, EMPTY=True, num_warps=8)


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

    return sparse_paged_attention_gather(
        layer,
        q,
        gather_indices,
        visible_lens,
        query_to_req,
        scale=scale,
        workspace=getattr(cache, '_nfa_selector_workspace', None),
    )
