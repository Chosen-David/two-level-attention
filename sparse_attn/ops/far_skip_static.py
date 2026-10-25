"""Setup outside capture, then fixed-address KV writes and far-only decode.

Used by fill_static_from_dynamic(..., model=model). Rebuild once per request and
recapture any enclosing graph: frozen gate decisions and pointers may change.
Full-model callers must pass explicit cache_position/position_ids, and register
their unpadded causal decode mask by identity in cache._far_skip_attention_mask.
That mask's contents are caller-owned input: arbitrary padding/masking is not
supported. The shipped GraphRunner constructs it from the same position buffer.
"""
import os

import torch
from transformers.cache_utils import DynamicCache

from .far_skip_cache import FarSkipLayer
from .far_skip_graph import FarSkipGraphWorkspace
from .tli_static_cache import TLIStaticLayer


class FarSkipStaticLayer(TLIStaticLayer):
    def __init__(self, source, cfg, max_cache_len, coarse, fine):
        super().__init__(max_cache_len, 1)
        ring = FarSkipLayer.from_layer(source, cfg)
        self.keys, self.values = ring.keys, ring.values
        self.dtype, self.device = ring.dtype, ring.device
        self.num_heads, self.head_dim = self.keys.shape[1], self.keys.shape[-1]
        self.config, self.layout = cfg, ring.layout
        self.cumulative_length = ring.logical_length
        self.workspace = FarSkipGraphWorkspace(cfg, ring.layout, self.num_heads, coarse, fine)
        self.workspace.initialize(self.keys.transpose(1, 2), ring.logical_length)

    def update(self, key_states, value_states, cache_kwargs=None):
        for x, storage in ((key_states, self.keys), (value_states, self.values)):
            if (x.shape != (*storage.shape[:2], 1, storage.shape[-1])
                    or x.dtype != storage.dtype or x.device != storage.device
                    or x.requires_grad):
                raise ValueError('static far KV must match detached single-token storage')
        pos = (cache_kwargs or {}).get('cache_position')
        if pos is None or pos.shape != (1,) or pos.dtype != torch.long or pos.device != self.device:
            raise ValueError('static far KV needs one device int64 cache position')
        slot = self.layout.map(pos)
        self.keys.index_copy_(2, slot, key_states)
        self.values.index_copy_(2, slot, value_states)
        return self.keys, self.values

    def reset(self):
        raise ValueError('refill far static cache from a fresh prefill and recapture')


def prepare_far_skip_static(cache, dyn, seq_lens, model):
    """Return prepared far-off layer indices, or an empty set for legacy mode.

    Preflight every layer before replacing any storage. Far-on layers are reset
    to ordinary static storage if a previous request evicted them. Caller must
    release the original DynamicCache after fill to actually free far history.
    """
    modules = {m.layer_idx: m for m in model.modules()
               if getattr(getattr(m, 'indexer', None), 'far_skip_backend', 'legacy') == 'compact'}
    if not modules:
        return set()
    if (cache.batch_size != 1 or len(seq_lens) != 1 or type(dyn) is not DynamicCache
            or getattr(dyn, 'offloading', False) or model.training or torch.is_grad_enabled()
            or os.environ.get('TLI_DECODE_DENSE', '0') == '1'
            or os.environ.get('TLI_SPARSE_PREFILL', '0') == '1'):
        raise ValueError('static compact far skip needs eval, no_grad, B=1 and ordinary prefill')
    planned = {}
    for li, module in modules.items():
        idx, source = module.indexer, dyn.layers[li]
        if (idx.far_skip_cache != 'evict' or module.sliding_window is not None
                or source.keys.shape[0] != 1 or source.keys.shape[-1] != 128
                or source.keys.shape[-2] != seq_lens[0] or seq_lens[0] < 1
                or seq_lens[0] >= cache.layers[li].max_cache_len):
            raise ValueError('static compact far skip needs evict, unpadded prefill and head_dim=128')
        idx.freeze_dyn_dgate()
        idx._resolve_skip()
        if idx.skip_far:
            if idx.far_skip_config.near_pages == 0 and idx.sink_tokens+idx.sliding_window_size == 0:
                raise ValueError('static far selection needs a nonempty protected or page budget')
            full = torch.arange(128, device=source.keys.device)
            coarse = idx._subspace_indices(source.keys.device) if idx.enable_subspace else full
            fine = (full if idx.l2_subspace == 'full' else
                    coarse if idx.enable_subspace else idx._qat_indices(source.keys.device))
            planned[li] = (source, idx.far_skip_config, coarse, fine)
    new_layers = {}
    for li, (source, cfg, coarse, fine) in planned.items():
        new_layers[li] = FarSkipStaticLayer(source, cfg, cache.layers[li].max_cache_len, coarse, fine)
    for li, old in enumerate(cache.layers):
        if li in new_layers:
            cache.layers[li] = new_layers[li]
        elif isinstance(old, FarSkipStaticLayer):
            cache.layers[li] = TLIStaticLayer(old.max_cache_len, 1)
    cache._far_skip_prepared = True
    cache._far_skip_attention_mask = None
    return set(new_layers)


def try_far_skip_static(module, q, k, v, cache, mask, batch_ctx):
    if not getattr(cache, '_far_skip_prepared', False):
        raise ValueError('prepare compact static cache with fill_static_from_dynamic(..., model=model)')
    if (module.training or torch.is_grad_enabled() or q.shape[:1] != (1,)
            or q.shape[-2] != 1 or batch_ctx is None or batch_ctx.batch != 1
            or os.environ.get('TLI_DECODE_DENSE', '0') == '1'
            or os.environ.get('TLI_SPARSE_PREFILL', '0') == '1'):
        raise ValueError('static far skip requires eval, no_grad, single-token B=1 BatchCtx')
    li, idx = module.layer_idx, module.indexer
    layer = cache.layers[li]
    if not isinstance(layer, FarSkipStaticLayer):
        if idx.skip_far:
            raise ValueError('far gate changed: refill static cache and recapture')
        return None
    if not idx.skip_far or layer.config != idx.far_skip_config or idx.far_skip_cache != 'evict':
        raise ValueError('far gate/geometry changed: refill static cache and recapture')
    if mask is not None and mask is not getattr(cache, '_far_skip_attention_mask', None):
        raise ValueError('static far skip requires the registered unpadded decode mask')
    from .cuda_graph_state import graph_state
    gs = graph_state()
    if gs.enabled:
        pos = gs.pos_buf
    else:
        if not 1 <= batch_ctx.seq_lens[0] <= layer.max_cache_len:
            raise ValueError('decode position exceeds static cache capacity')
        pos = torch.tensor([batch_ctx.seq_lens[0]-1], dtype=torch.long, device=q.device)
    k, v = layer.update(k, v, {'cache_position': pos})
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return layer.workspace(q, k, v, pos+1, module.scaling)
