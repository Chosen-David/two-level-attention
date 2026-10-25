"""Request-local, physically bounded KV for irrevocable far-off layers.

Transformers 4.56 DynamicCache adapter. Keys are already RoPE-encoded; logical
positions never change when storage is compacted. No view retains the full KV.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def _reject_legacy_export():
    raise ValueError("legacy KV tuples cannot represent evicted absolute positions")


class FarSkipLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.config = None
        self.logical_length = 0
        self.tail_start = None

    @classmethod
    def from_layer(cls, layer, config):
        if type(layer) is not DynamicLayer and not (
                isinstance(layer, cls) and layer.config is None):
            raise ValueError("far eviction requires an ordinary DynamicLayer")
        result = cls()
        result.config = config
        result.logical_length = layer.get_seq_length()
        if result.logical_length < 1 or layer.keys.shape[0] != 1:
            raise ValueError("far eviction requires a nonempty single-request cache")
        if layer.keys.requires_grad or layer.values.requires_grad:
            raise ValueError("far eviction cannot retain an autograd graph; use inference_mode")
        result.dtype, result.device = layer.keys.dtype, layer.keys.device
        result.tail_start = config.tail_start(result.logical_length)
        prefix = min(config.sink_tokens, result.logical_length)
        # cat allocates independent storage even when the tail is a view.
        result.keys = torch.cat((layer.keys[..., :prefix, :],
                                 layer.keys[..., result.tail_start:, :]), dim=-2)
        result.values = torch.cat((layer.values[..., :prefix, :],
                                   layer.values[..., result.tail_start:, :]), dim=-2)
        return result

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.config is None:
            return super().update(key_states, value_states, cache_kwargs)
        if key_states.shape[0] != 1 or key_states.shape[-2] != 1:
            raise ValueError("evicted cache supports only single-token, batch-one decode")
        if key_states.requires_grad or value_states.requires_grad:
            raise ValueError("far-evicted cache requires detached inference KV")
        length = self.logical_length + 1
        new_tail = self.config.tail_start(length)
        prefix = min(self.config.sink_tokens, self.logical_length)
        # Once full, sink stays fixed; before then, all old tokens are prefix.
        offset = prefix + new_tail - self.tail_start
        if length <= self.config.sink_tokens:
            offset = prefix
        self.keys = torch.cat((self.keys[..., :prefix, :],
                               self.keys[..., offset:, :], key_states), dim=-2)
        self.values = torch.cat((self.values[..., :prefix, :],
                                 self.values[..., offset:, :], value_states), dim=-2)
        self.logical_length, self.tail_start = length, new_tail
        return self.keys, self.values

    def get_seq_length(self):
        return (super().get_seq_length() if self.config is None
                else self.logical_length)

    def reset(self):
        # A new request must prefill and calibrate again, before eviction.
        self.keys = self.values = None
        self.config = None
        self.logical_length = 0
        self.tail_start = None

    def crop(self, max_length):
        if self.config is not None:
            raise ValueError("cannot rewind a far-evicted cache; start a fresh request")
        return super().crop(max_length)

    def _unsupported(self, *args, **kwargs):
        raise ValueError("far-evicted cache does not support beams, batching or offload")

    reorder_cache = batch_repeat_interleave = batch_select_indices = _unsupported
    offload = prefetch = _unsupported


def evict_far(cache, layer_idx, config):
    if type(cache) is not DynamicCache or getattr(cache, "offloading", False):
        raise ValueError("far eviction requires a non-offloaded HF DynamicCache")
    layer = cache.layers[layer_idx]
    if isinstance(layer, FarSkipLayer) and layer.config is not None:
        if layer.config != config:
            raise ValueError("cannot change geometry after far KV eviction")
        return layer
    layer = FarSkipLayer.from_layer(layer, config)
    cache.layers[layer_idx] = layer
    # HF's tuple export reads layer.keys directly and drops all position metadata.
    cache.to_legacy_cache = _reject_legacy_export
    return layer
