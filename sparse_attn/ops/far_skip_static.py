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
from .far_skip import select_far_skip, attend_far_skip
from .far_skip_workspace import FarSkipWorkspace
from .far_skip_subspace import same_dimensions
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
        self.coarse = coarse.clone()
        self.fine = self.coarse if same_dimensions(coarse, fine) else fine.clone()
        self.workspace = None
        self.eager_workspace = None
        self.logical_length = ring.logical_length

    def update(self, key_states, value_states, cache_kwargs=None):
        self.validate_update(key_states, value_states)
        pos = (cache_kwargs or {}).get('cache_position')
        if pos is None or pos.shape != (1,) or pos.dtype != torch.long or pos.device != self.device:
            raise ValueError('static far KV needs one device int64 cache position')
        from .far_skip_accel import enabled, write_pair
        if enabled(self.keys):
            write_pair(self.keys, self.values, key_states, value_states, pos, self.layout)
        else:
            slot = self.layout.map(pos)
            self.keys.index_copy_(2, slot, key_states)
            self.values.index_copy_(2, slot, value_states)
        return self.keys, self.values

    def validate_update(self, key_states, value_states):
        for x, storage in ((key_states, self.keys), (value_states, self.values)):
            if (x.shape != (*storage.shape[:2], 1, storage.shape[-1])
                    or x.dtype != storage.dtype or x.device != storage.device
                    or x.requires_grad):
                raise ValueError('static far KV must match detached single-token storage')

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
        if layer.workspace is None:
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError('warm up compact graph workspace before capture')
            layer.workspace = FarSkipGraphWorkspace(
                layer.config, layer.layout, layer.num_heads, layer.coarse, layer.fine)
            layer.workspace.initialize(layer.keys.transpose(1, 2), layer.logical_length)
            layer.eager_workspace = None
        pos = gs.pos_buf
    else:
        length = batch_ctx.seq_lens[0]
        if not 1 <= length <= layer.max_cache_len:
            raise ValueError('decode position exceeds static cache capacity')
        if length != layer.logical_length + 1:
            raise ValueError('staged compact decode requires the next position; refill after replay or rewind')
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise ValueError('enable graph state before compact CUDA capture')
        layer.validate_update(k, v)
        # Host length is already known in staged decode. Do not pay for device
        # boundary arithmetic, padded candidates or per-layer H2D positions.
        slot = layer.layout.slot(length-1)
        from .far_skip_accel import enabled, write_pair
        if enabled(layer.keys):
            write_pair(layer.keys, layer.values, k, v, length-1, layer.layout)
        else:
            layer.keys[:, :, slot:slot+1].copy_(k)
            layer.values[:, :, slot:slot+1].copy_(v)
        layer.logical_length = length
        layer.workspace = None
        if layer.eager_workspace is None:
            layer.eager_workspace = FarSkipWorkspace()
        q, k, v = q.transpose(1, 2), layer.keys.transpose(1, 2), layer.values.transpose(1, 2)
        selection = select_far_skip(q, k, length, layer.config, layer.coarse, layer.fine,
                                    scale=module.scaling, layout=layer.layout,
                                    workspace=layer.eager_workspace)
        return attend_far_skip(q, k, v, selection, layer.config, scale=module.scaling,
                               layout=layer.layout, workspace=layer.eager_workspace)
    k, v = layer.update(k, v, {'cache_position': pos})
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return layer.workspace(q, k, v, pos+1, module.scaling)
