"""Request-local hybrid KV storage for Qwen3 batched, single-token decode.

Prefill each request unpadded, then immediately add it to the builder. Gates
and indexer state belong to that request, not to the last prefilled batch row.
The finished cache and its runner are immutable in layout: start a new builder
and capture a new graph for a new batch. No rectangular full-history allocation
is made for compact layers; each row owns either a ring or a linear buffer.
"""
import copy
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .cuda_graph_state import graph_state
from .far_skip_static import FarSkipStaticLayer
from .tli_static_cache import TLIStaticCache, TLIStaticLayer


@contextmanager
def row_graph_state(pos, length, alive):
    """Scope a B=1 indexer's device inputs without leaking them to other rows."""
    gs = graph_state()
    saved = gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf
    gs.enabled, gs.bs = True, 1
    gs.pos_buf, gs.valid_len_buf, gs.alive_buf = pos, length, alive
    try:
        yield
    finally:
        gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf = saved


def _snapshot_indexer(idx):
    # Do not deepcopy an old graph's full-history workspaces. Immutable config
    # tensors can be shared; every mutable request/index cache starts fresh.
    result = copy.copy(idx)
    result.args = copy.deepcopy(idx.args)
    result._ginc = None
    result.clear()
    result._skip_ids = None if idx._skip_ids is None else frozenset(idx._skip_ids)
    result.skip_far = idx.skip_far
    result._dyn_frozen = idx._dyn_frozen
    result._dyn_skip_pending = idx._dyn_skip_pending
    result._pair_idx = None if idx._pair_idx is None else idx._pair_idx.clone()
    result._graph_const_cache = None
    result._far_skip_workspace = None
    return result


class HybridBatchLayer(TLIStaticLayer):
    """Cache metadata plus independent row storage; never a full-width view."""
    def __init__(self, capacity, batch_size):
        super().__init__(capacity, batch_size)
        self.rows = []
        self.indexers = []
        self.cu = None

    def update(self, *args, **kwargs):
        raise ValueError('hybrid rows must be consumed by the compact Qwen3 reader')

    def reset(self):
        raise ValueError('build a fresh hybrid batch and recapture for a new request')


class HybridBatchCacheBuilder:
    """Consume B single-request DynamicCaches in order, releasing each promptly."""
    def __init__(self, model, capacity, batch_size):
        if model.training or torch.is_grad_enabled():
            raise ValueError('hybrid batch requires eval and no_grad')
        if any(os.environ.get(name, '0') == '1' for name in
               ('TLI_DECODE_DENSE', 'TLI_SPARSE_PREFILL')):
            raise ValueError('hybrid batch requires ordinary dense prefill and sparse decode')
        self.model = model
        self.modules = {m.layer_idx: m for m in model.modules()
                        if getattr(getattr(m, 'indexer', None), 'far_skip_backend', '') == 'compact'}
        if not self.modules:
            raise ValueError('hybrid batch requires compact Qwen3 attention')
        self.cache = TLIStaticCache(model.config, capacity, batch_size)
        self.lengths = []
        self.finished = False
        for li, module in self.modules.items():
            if module.indexer.far_skip_cache != 'evict' or module.sliding_window is not None:
                raise ValueError('hybrid batch requires evict and no native sliding window')
            self.cache.layers[li] = HybridBatchLayer(capacity, batch_size)

    def add(self, dyn, length):
        b, cache = len(self.lengths), self.cache
        if self.finished or b >= cache.batch_size:
            raise ValueError('hybrid batch already filled')
        if (type(dyn) is not DynamicCache or getattr(dyn, 'offloading', False)
                or len(dyn.layers) != len(cache.layers) or type(length) is not int
                or not 1 <= length < cache.max_cache_len):
            raise ValueError('expected an unpadded DynamicCache with room for decode')
        for li, source in enumerate(dyn.layers):
            k, v = source.keys, source.values
            if (k is None or v is None or k.ndim != 4 or k.shape != v.shape
                    or k.shape[0] != 1 or k.shape[-2] != length
                    or k.requires_grad or v.requires_grad
                    or (li in self.modules and k.shape[-1] != 128)):
                raise ValueError('prefill must contain one detached, unpadded request')
        # Freeze now: the next prefill resets the model's dynamic gate/pair state.
        for li, layer in enumerate(cache.layers):
            source = dyn.layers[li]
            if isinstance(layer, HybridBatchLayer):
                idx = self.modules[li].indexer
                idx.freeze_dyn_dgate()
                idx._resolve_skip()
                row_idx = _snapshot_indexer(idx)
                if row_idx.skip_far:
                    if row_idx.far_skip_config.near_pages == 0 and row_idx.sink_tokens+row_idx.sliding_window_size == 0:
                        raise ValueError('hybrid far selection needs a nonempty protected or page budget')
                    full = torch.arange(128, device=source.keys.device)
                    coarse = row_idx._subspace_indices(source.keys.device) if row_idx.enable_subspace else full
                    fine = (full if row_idx.l2_subspace == 'full' else coarse
                            if row_idx.enable_subspace else row_idx._qat_indices(source.keys.device))
                    row = FarSkipStaticLayer(source, row_idx.far_skip_config,
                                             layer.max_cache_len, coarse, fine)
                else:
                    row = TLIStaticLayer(layer.max_cache_len, 1)
                    row.lazy_initialization(source.keys)
                    row.keys[..., :length, :].copy_(source.keys)
                    row.values[..., :length, :].copy_(source.values)
                    row.cumulative_length = length
                layer.rows.append(row)
                layer.indexers.append(row_idx)
                if layer.cu is None:
                    layer.cu = torch.tensor([0, layer.max_cache_len], dtype=torch.int32,
                                            device=source.keys.device)
            else:
                if layer.keys is None:
                    # Shape-only expanded view, not a B*prompt temporary tensor.
                    layer.lazy_initialization(source.keys[..., :1, :].expand(
                        cache.batch_size, -1, -1, -1))
                layer.keys[b:b+1, :, :length].copy_(source.keys)
                layer.values[b:b+1, :, :length].copy_(source.values)
            layer.cumulative_length = max(layer.cumulative_length, length)
        self.lengths.append(length)

    def finish(self):
        if self.finished or len(self.lengths) != self.cache.batch_size:
            raise ValueError('fill every hybrid batch row exactly once')
        self.finished = True
        self.cache._hybrid_batch = True
        self.cache._hybrid_lengths = tuple(self.lengths)
        self.cache._far_skip_prepared = True
        self.cache._far_skip_attention_mask = None
        return self.cache


def _attend_linear(q, k, v, mask, budget, scale):
    """Fixed-size gather/SDPA, including short contexts; no nonzero/item sync."""
    h, length, d = k.shape[2], k.shape[1], k.shape[-1]
    count = min(length, budget)
    valid, positions = mask[0, 0].to(torch.int32).topk(count, dim=-1, sorted=False)
    valid = valid.bool()
    heads = torch.arange(h, device=k.device)[:, None]
    kk = k[0][positions, heads].masked_fill(~valid[..., None], 0)
    vv = v[0][positions, heads].masked_fill(~valid[..., None], 0)
    groups = q.shape[2] // h
    out = F.scaled_dot_product_attention(
        q[0, 0].reshape(h, 1, groups, d), kk[:, None], vv[:, None],
        attn_mask=valid[:, None, None], scale=scale, dropout_p=0.0)
    return out.reshape(1, 1, q.shape[2], v.shape[-1])


def try_hybrid_batch_decode(module, q, k, v, cache, mask, ctx):
    if (module.training or torch.is_grad_enabled() or ctx is None
            or q.shape[0] != cache.batch_size or ctx.batch != cache.batch_size
            or q.shape[-2] != 1):
        raise ValueError('hybrid batch requires eval/no_grad single-token BatchCtx')
    if mask is not getattr(cache, '_far_skip_attention_mask', None) or mask is None:
        raise ValueError('hybrid batch requires the registered causal decode mask')
    layer = cache.layers[module.layer_idx]
    if not isinstance(layer, HybridBatchLayer):
        raise ValueError('compact layer missing hybrid batch storage')
    gs = graph_state()
    if not gs.enabled or any(t is None for t in (gs.pos_buf, gs.valid_len_buf, gs.alive_buf)):
        raise ValueError('hybrid batch requires HybridBatchRunner device inputs')
    pos, lengths, alive = gs.pos_buf, gs.valid_len_buf, gs.alive_buf
    outputs = []
    for b, (row, idx) in enumerate(zip(layer.rows, layer.indexers)):
        p, vl, active = pos[b:b+1], lengths[b:b+1], alive[b:b+1]
        slot = row.layout.map(p) if isinstance(row, FarSkipStaticLayer) else p
        # Dead rows keep their last valid KV exactly. Avoid a Python branch on
        # alive: EOS changes during graph replay while kernel shapes stay fixed.
        for storage, incoming in ((row.keys, k), (row.values, v)):
            old = storage.index_select(2, slot)
            storage.index_copy_(2, slot, torch.where(active.view(1, 1, 1, 1), incoming[b:b+1], old))
        qq, kk, vv = q[b:b+1].transpose(1, 2), row.keys.transpose(1, 2), row.values.transpose(1, 2)
        if isinstance(row, FarSkipStaticLayer):
            out = row.workspace(qq, kk, vv, vl, module.scaling)
        else:
            with row_graph_state(p, vl, active):
                index = idx.prepare_index(kk, layer.cu)
                selection = idx.compute_mask_graph(qq, index, module.scaling, vl)
            budget = idx.args.tia_level2_topk + idx.sink_tokens + idx.sliding_window_size
            out = _attend_linear(qq, kk, vv, selection, budget, module.scaling)
        outputs.append(out)
    return torch.cat(outputs, dim=0)
