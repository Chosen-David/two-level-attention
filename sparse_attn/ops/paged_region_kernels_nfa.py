"""GraphReady NFA 的保护集和 gather attention 入口。"""
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
    # Short rows may overlap sink/SWA even when another row is long.
    swa_id = length_t[:, None] - swa_count[:, None] + swa_col
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

    return sparse_paged_attention_gather(
        layer,
        q,
        gather_indices,
        visible_lens,
        query_to_req,
        scale=scale,
        workspace=getattr(cache, '_nfa_selector_workspace', None),
    )
