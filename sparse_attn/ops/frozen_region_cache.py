"""Request-local physical sink -> far -> near -> SWA cache for eager Qwen3.

Far-skip and both-skip physically evict their unused nodes. Near-only skip keeps
an explicit, attention-invisible staging node; its KV ages normally into far.
Near-only skip saves attention work, not resident KV. No hidden full-history
backup is retained. This exploration does not implement TLI selection,
CUDA graphs, batching, offload, rewind, or beam search.
"""
from dataclasses import dataclass
import os
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
        self.arena_offset = None

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

    def pop_many(self, n):
        """按时间顺序批量弹出前缀；仅在跨环边界时拼接本次弹出的 KV。"""
        if not 0 <= n <= self.size:
            raise ValueError('invalid prefix pop count')
        if not n:
            return None
        first = min(n, self.capacity-self.head)
        keys = self.keys[..., self.head:self.head+first, :]
        values = self.values[..., self.head:self.head+first, :]
        positions = self.positions[self.head:self.head+first]
        if first < n:
            keys = torch.cat((keys, self.keys[..., :n-first, :]), -2)
            values = torch.cat((values, self.values[..., :n-first, :]), -2)
            positions += self.positions[:n-first]
        self.positions[self.head:self.head+first] = [None]*first
        if first < n:
            self.positions[:n-first] = [None]*(n-first)
        self.head = (self.head+n) % self.capacity
        self.size -= n
        return keys, values, positions

    @property
    def first_position(self):
        return self.positions[self.head] if self.size else None

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
        self.arena_k = self.arena_v = None
        self.arena_offsets = {}

    def lazy_initialization(self, key_states):
        pass  # Shape and length are both needed; initialization occurs in update.

    def _relink(self):
        live = [self.nodes[name] for name in ('sink', 'far', 'near', 'near_staging', 'swa') if name in self.nodes]
        for i, node in enumerate(live):
            node.next = live[i+1] if i+1 < len(live) else None
        self.head = live[0] if live else None

    def freeze(self, skip_near=False, skip_far=False, physical_arena=False):
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
        if physical_arena:
            self._build_physical_arena()

    def _build_physical_arena(self):
        """freeze 边界一次性搬迁；之后所有区域共享一份定容 K/V 存储。

        节点仍保留独立 FIFO 元数据，但 K/V 变为 arena 视图。逻辑位置到
        物理槽位由 (arena_offset, head, capacity) 解析，不在 decode 中 gather。
        """
        nodes = []
        node = self.head
        while node is not None:
            nodes.append(node)
            node = node.next
        total = sum(node.capacity for node in nodes)
        template = nodes[0].keys
        self.arena_k = torch.empty(
            (total, template.shape[1], template.shape[-1]),
            device=template.device, dtype=template.dtype)
        self.arena_v = torch.empty_like(self.arena_k)
        offset = 0
        for node in nodes:
            keys, values, positions = node.read()
            size = node.size
            node_keys = self.arena_k[offset:offset+node.capacity].permute(1, 0, 2).unsqueeze(0)
            node_values = self.arena_v[offset:offset+node.capacity].permute(1, 0, 2).unsqueeze(0)
            node_keys[..., :size, :].copy_(keys)
            node_values[..., :size, :].copy_(values)
            node.keys, node.values = node_keys, node_values
            node.positions = positions + [None]*(node.capacity-size)
            node.head = 0
            node.arena_offset = offset
            self.arena_offsets[node.name] = offset
            offset += node.capacity

    def logical_positions(self, device=None):
        """直接在目标设备构造可见逻辑位置，避免 Python list 的 H2D 重传。"""
        if device is None:
            device = self.arena_k.device if self.arena_k is not None else self.nodes['sink'].keys.device
        ranges = []
        node = self.head
        while node is not None:
            if node.size and node.attention_visible:
                first = node.first_position
                ranges.append(torch.arange(first, first+node.size, device=device, dtype=torch.int64))
            node = node.next
        return torch.cat(ranges)

    def visible_mask(self, positions):
        """绝对逻辑位置是否属于当前 attention-visible 区域。"""
        visible = torch.zeros_like(positions, dtype=torch.bool)
        node = self.head
        while node is not None:
            if node.size and node.attention_visible:
                first = node.first_position
                visible |= (positions >= first) & (positions < first+node.size)
            node = node.next
        return visible

    def physical_positions(self, positions):
        """绝对逻辑 token ID → arena 槽位；全程 device 端整数运算。"""
        if self.arena_k is None:
            raise ValueError('physical arena is not initialized')
        physical = torch.zeros_like(positions)
        node = self.head
        while node is not None:
            if node.size and node.attention_visible:
                first = node.first_position
                in_node = (positions >= first) & (positions < first+node.size)
                slots = node.arena_offset + (node.head+positions-first) % node.capacity
                physical = torch.where(in_node, slots, physical)
            node = node.next
        return physical

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
        self._append_decode(keys, values, n)
        k, v, _ = self.gather()
        return k, v

    def append_decode(self, keys, values):
        """热路径单 token 追加：不构造可见 K/V，也不读取 GPU 位置元数据。"""
        if (keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1
                or keys.shape[-2] != 1):
            raise ValueError('physical decode requires equal-width B1 single-token K/V')
        if torch.is_grad_enabled() and (keys.requires_grad or values.requires_grad):
            raise ValueError('inference-only cache')
        self._append_decode(keys, values, 1)

    def _append_decode(self, keys, values, n):
        if not self.frozen or n != 1:
            raise ValueError('freeze after one-shot prefill; decode must be one token')
        if self.logical_length-self.prompt_length >= self.geometry.max_new_tokens:
            raise ValueError('configured generation capacity exceeded')
        position = self.logical_length
        new_length = position+1
        _, new_far_hi, _ = self.geometry.bounds(new_length)
        near = self.nodes.get('near') or self.nodes.get('near_staging')
        far = self.nodes.get('far')
        if near is not None and near.size:
            aged = min(near.size, max(0, new_far_hi-near.first_position))
            if aged:
                old_k, old_v, old_positions = near.pop_many(aged)
                if far is not None:
                    far.append(old_k, old_v, old_positions)
                else:
                    self.permanently_dropped += aged
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
        # Transformers 5 passes query_length; 4.x passes cache_position.
        query_length = cache_position if isinstance(cache_position, int) else cache_position.shape[0]
        return self.logical_length + query_length, 0

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
        self.tli_regions = False

    def freeze(self, decisions):
        if set(decisions) != set(range(len(self.layers))):
            raise ValueError('every layer must have a prefill decision')
        physical = (self.tli_regions and
                    os.environ.get('TLI_FROZEN_EAGER_REFERENCE', '0') != '1')
        for li, layer in enumerate(self.layers):
            layer.freeze(**decisions[li], physical_arena=physical)

    def report(self):
        reports = [layer.report() for layer in self.layers]
        return dict(layers=reports, kv_bytes=sum(n['kv_bytes'] for layer in reports for n in layer['nodes'].values()))
