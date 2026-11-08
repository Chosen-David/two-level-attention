"""Request-local, physically bounded KV for irrevocable far-off layers.

Transformers 4.56 DynamicCache adapter. Keys are already RoPE-encoded; logical
positions never change when storage is compacted. No view retains the full KV.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .far_skip import FarSkipLayout


def _reject_legacy_export():
    raise ValueError("legacy KV tuples cannot represent evicted absolute positions")


class FarSkipLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.config = None
        self.logical_length = 0
        self.tail_start = None
        self.layout = None

    @classmethod
    def from_layer(cls, layer, config):
        if type(layer) is not DynamicLayer and not (
                isinstance(layer, cls) and layer.config is None):
            raise ValueError("far eviction requires an ordinary DynamicLayer")
        if layer.get_seq_length() < 1 or layer.keys.shape[0] != 1:
            raise ValueError("far eviction requires a nonempty single-request cache")
        return cls.from_prefill(layer.keys, layer.values, config)

    @classmethod
    def from_prefill(cls, keys, values, config):
        """Allocate bounded storage and write one complete unpadded prefill."""
        if (keys.ndim != 4 or values.shape != keys.shape or keys.shape[0] != 1
                or keys.shape[-2] < 1):
            raise ValueError("far eviction requires a nonempty single-request cache")
        if keys.requires_grad or values.requires_grad:
            raise ValueError("far eviction cannot retain an autograd graph; use inference_mode")
        result = cls()
        result.config = config
        result.logical_length = keys.shape[-2]
        result.dtype, result.device = keys.dtype, keys.device
        result.tail_start = config.tail_start(result.logical_length)
        prefix = min(config.sink_tokens, result.logical_length)
        result.layout = FarSkipLayout(config.sink_tokens,
                                      max(config.near_length, config.block_size) + config.window_tokens + config.block_size - 1)
        shape = (*keys.shape[:-2], result.layout.capacity, keys.shape[-1])
        result.keys = keys.new_empty(shape)
        result.values = values.new_empty((*shape[:-1], values.shape[-1]))
        result.keys[..., :prefix, :].copy_(keys[..., :prefix, :])
        result.values[..., :prefix, :].copy_(values[..., :prefix, :])
        # Copy the chronological tail in at most two runs into the cyclic suffix.
        count = result.logical_length - result.tail_start
        slot = result.layout.slot(result.tail_start)
        first = min(count, result.layout.capacity - slot)
        for dst, src, width in ((slot, result.tail_start, first),
                                (config.sink_tokens, result.tail_start + first, count - first)):
            if width:
                result.keys[..., dst:dst+width, :].copy_(keys[..., src:src+width, :])
                result.values[..., dst:dst+width, :].copy_(values[..., src:src+width, :])
        return result

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.config is None:
            return super().update(key_states, value_states, cache_kwargs)
        if key_states.shape[0] != 1 or key_states.shape[-2] != 1:
            raise ValueError("evicted cache supports only single-token, batch-one decode")
        if key_states.requires_grad or value_states.requires_grad:
            raise ValueError("far-evicted cache requires detached inference KV")
        for incoming, storage in ((key_states, self.keys), (value_states, self.values)):
            expected = (*storage.shape[:-2], 1, storage.shape[-1])
            if (incoming.shape != expected or incoming.dtype != storage.dtype
                    or incoming.device != storage.device):
                raise ValueError("new KV must match ring batch, heads, width, dtype and device")
        position = self.logical_length
        slot = self.layout.slot(position)
        # Fixed addresses: no allocation or copy of old KV in steady decode.
        self.keys[..., slot:slot+1, :].copy_(key_states)
        self.values[..., slot:slot+1, :].copy_(value_states)
        self.logical_length += 1
        self.tail_start = self.config.tail_start(self.logical_length)
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
        self.layout = None

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
