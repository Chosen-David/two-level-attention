"""Request-local physical sink -> far -> near -> SWA cache for eager Qwen3.

Far-skip and both-skip physically evict their unused nodes. Near-only skip keeps
an explicit, attention-invisible staging node; its KV ages normally into far.
Near-only skip saves attention work, not resident KV. No hidden full-history
backup is retained. This exploration does not implement TLI selection,
CUDA graphs, batching, offload, rewind, or beam search.
"""
from dataclasses import dataclass
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


@dataclass(frozen=True)
class RegionGeometry:
    sink: int = 4
    near: int = 4096
    window: int = 128
    page: int = 64
    max_new_tokens: int = 512

    def __post_init__(self):
        for name in ('sink', 'near', 'window', 'page', 'max_new_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'sink' else 1):
                raise ValueError(f'invalid {name}: {value}')

    def bounds(self, length):
        sink_hi = min(self.sink, length)
        near_hi = max(sink_hi, length-self.window)
        far_hi = min(near_hi, max(sink_hi, ((length-self.window-self.near)//self.page)*self.page))
        return sink_hi, far_hi, near_hi


class KVNode:
    """An independently owned fixed-capacity FIFO, with chronological reads."""
    def __init__(self, name, capacity, template):
        self.name, self.capacity = name, capacity
        shape = (*template.shape[:-2], capacity, template.shape[-1])
        self.keys = torch.empty(shape, device=template.device, dtype=template.dtype)
        self.values = torch.empty_like(self.keys)
        self.positions = [None]*capacity
        self.head = self.size = 0
        self.next = None
        self.attention_visible = True

    def append(self, keys, values, positions):
        n = keys.shape[-2]
        if n != len(positions) or n > self.capacity-self.size:
            raise ValueError(f'{self.name} capacity exceeded or inconsistent positions')
        if not n:
            return
        tail = (self.head+self.size) % self.capacity
        first = min(n, self.capacity-tail)
        self.keys[..., tail:tail+first, :].copy_(keys[..., :first, :])
        self.values[..., tail:tail+first, :].copy_(values[..., :first, :])
        self.positions[tail:tail+first] = positions[:first]
        if first < n:
            self.keys[..., :n-first, :].copy_(keys[..., first:, :])
            self.values[..., :n-first, :].copy_(values[..., first:, :])
            self.positions[:n-first] = positions[first:]
        self.size += n

    def peek(self):
        if not self.size:
            raise ValueError('empty queue')
        return (self.keys[..., self.head:self.head+1, :],
                self.values[..., self.head:self.head+1, :], self.positions[self.head])

    def pop(self):
        if not self.size:
            raise ValueError('empty queue')
        self.positions[self.head] = None
        self.head = (self.head+1) % self.capacity
        self.size -= 1

    def read(self):
        if self.head+self.size <= self.capacity:
            return (self.keys[..., self.head:self.head+self.size, :],
                    self.values[..., self.head:self.head+self.size, :],
                    self.positions[self.head:self.head+self.size])
        end = (self.head+self.size) % self.capacity
        return (torch.cat((self.keys[..., self.head:, :], self.keys[..., :end, :]), -2),
                torch.cat((self.values[..., self.head:, :], self.values[..., :end, :]), -2),
                self.positions[self.head:]+self.positions[:end])

    def resident_bytes(self):
        return (self.keys.numel()+self.values.numel())*self.keys.element_size()


class RegionCacheLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.nodes = {}
        self.head = None
        self.logical_length = self.prompt_length = 0
        self.frozen = False
        self.skip_far = self.skip_near = False
        self.permanently_dropped = 0

    def lazy_initialization(self, key_states):
        pass  # Shape and length are both needed; initialization occurs in update.

    def _relink(self):
        live = [self.nodes[name] for name in ('sink', 'far', 'near', 'near_staging', 'swa') if name in self.nodes]
        for i, node in enumerate(live):
            node.next = live[i+1] if i+1 < len(live) else None
        self.head = live[0] if live else None

    def freeze(self, skip_near=False, skip_far=False):
        if self.frozen:
            if (skip_near, skip_far) != (self.skip_near, self.skip_far):
                raise ValueError('cannot change frozen request decisions')
            return
        if not self.logical_length:
            raise ValueError('prefill required before freeze')
        for name, evict in (('near', skip_near and skip_far), ('far', skip_far)):
            if evict:
                node = self.nodes.pop(name)
                self.permanently_dropped += node.size
                node.next = None
        if skip_near and not skip_far:
            node = self.nodes.pop('near')
            node.name = 'near_staging'
            node.attention_visible = False
            self.nodes['near_staging'] = node
        self.skip_near, self.skip_far = bool(skip_near), bool(skip_far)
        self.frozen = True
        self._relink()

    def update(self, keys, values, cache_kwargs=None):
        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError('only unpadded B=1, equal-width Qwen K/V supported')
        if torch.is_grad_enabled() and (keys.requires_grad or values.requires_grad):
            raise ValueError('inference-only cache')
        n = keys.shape[-2]
        positions = (cache_kwargs or {}).get('cache_position')
        expected = list(range(self.logical_length, self.logical_length+n))
        if positions is not None and positions.tolist() != expected:
            raise ValueError('non-monotonic or padded positions unsupported')
        if not self.logical_length:
            if self.frozen:
                raise ValueError('cache reset required')
            g = self.geometry
            sink_hi, far_hi, near_hi = g.bounds(n)
            specs = [('sink', g.sink, 0, sink_hi),
                     ('far', far_hi-sink_hi+g.max_new_tokens+g.page, sink_hi, far_hi),
                     ('near', g.near+g.page-1, far_hi, near_hi),
                     ('swa', g.window, near_hi, n)]
            for name, capacity, lo, hi in specs:
                node = KVNode(name, capacity, keys)
                node.append(keys[..., lo:hi, :], values[..., lo:hi, :], list(range(lo, hi)))
                self.nodes[name] = node
            self.logical_length = self.prompt_length = n
            self._relink()
            return keys, values  # Dense prefill sees original, complete K/V.
        if not self.frozen or n != 1:
            raise ValueError('freeze after one-shot prefill; decode must be one token')
        if self.logical_length-self.prompt_length >= self.geometry.max_new_tokens:
            raise ValueError('configured generation capacity exceeded')
        position = self.logical_length
        new_length = position+1
        _, new_far_hi, _ = self.geometry.bounds(new_length)
        near = self.nodes.get('near') or self.nodes.get('near_staging')
        far = self.nodes.get('far')
        if near is not None:
            while near.size and near.peek()[2] < new_far_hi:
                old_k, old_v, old_pos = near.peek()
                if far is not None:
                    far.append(old_k, old_v, [old_pos])
                else:
                    self.permanently_dropped += 1
                near.pop()
        if position < self.geometry.sink:
            self.nodes['sink'].append(keys, values, [position])
        else:
            swa = self.nodes['swa']
            if swa.size == swa.capacity:
                old_k, old_v, old_pos = swa.peek()
                if near is not None:
                    near.append(old_k, old_v, [old_pos])
                else:
                    self.permanently_dropped += 1
                swa.pop()
            swa.append(keys, values, [position])
        self.logical_length = new_length
        k, v, _ = self.gather()
        return k, v

    def gather(self):
        keys, values, positions = [], [], []
        node = self.head
        while node is not None:
            if node.size and node.attention_visible:
                k, v, pos = node.read()
                keys.append(k); values.append(v); positions.extend(pos)
            node = node.next
        return torch.cat(keys, -2), torch.cat(values, -2), positions

    def get_mask_sizes(self, cache_position):
        return self.logical_length + (cache_position if isinstance(cache_position, int)
                                      else cache_position.shape[0]), 0

    def get_seq_length(self):
        return self.logical_length

    def get_max_cache_shape(self):
        return -1

    def reorder_cache(self, beam_idx):
        raise ValueError('beam/reorder unsupported')

    def reset(self):
        raise ValueError('construct a new request cache instead of resetting')

    def report(self):
        order = []
        node = self.head
        while node is not None:
            order.append(node.name)
            node = node.next
        return dict(logical_length=self.logical_length, skip_near=self.skip_near, skip_far=self.skip_far,
                    nodes={name: dict(tokens=node.size, capacity=node.capacity, kv_bytes=node.resident_bytes(),
                                      attention_visible=node.attention_visible) for name, node in self.nodes.items()},
                    stored_tokens=sum(node.size for node in self.nodes.values()),
                    attention_tokens=sum(node.size for node in self.nodes.values() if node.attention_visible),
                    linked_order=order,
                    attention_order=[name for name in order if self.nodes[name].attention_visible],
                    permanently_dropped=self.permanently_dropped)


class FrozenRegionCache(Cache):
    def __init__(self, n_layers, geometry=RegionGeometry()):
        super().__init__(layers=[RegionCacheLayer(geometry) for _ in range(n_layers)])

    def freeze(self, decisions):
        if set(decisions) != set(range(len(self.layers))):
            raise ValueError('every layer must have a prefill decision')
        for li, layer in enumerate(self.layers):
            layer.freeze(**decisions[li])

    def report(self):
        reports = [layer.report() for layer in self.layers]
        return dict(layers=reports, kv_bytes=sum(n['kv_bytes'] for layer in reports for n in layer['nodes'].values()))
