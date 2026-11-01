"""Paged KV cache: global physical page pool plus per-request region metadata.

Each logical page owns independently allocated K/V tensors. Reclaiming a page
releases its tensor references to the Torch allocator, and recycles its page ID.
Allocator-reserved memory is not reported as freed driver memory. The legacy
B1 request path is preserved; batch requests share one physical page pool per
layer and keep independent block tables / region state.
"""
import heapq
from collections import deque
from dataclasses import dataclass, field
import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from ..policies.region_geometry import RegionGeometry

INVALID_PAGE_IDX = -1
INVALID_GATHER_IDX = -1
PAD_SLOT = -1

SINK = 0b0001
FAR = 0b0010
NEAR = 0b0100
SWA = 0b1000
REGION_BITS = (SINK, FAR, NEAR, SWA)


@dataclass
class RequestState:
    """One continuous-batching request slot.

    ``block_table`` maps logical page id -> physical page id.  Physical page ids
    are shared by every layer; each layer still owns a private physical arena,
    so the same physical id indexes a different tensor in each layer.

    ``free_phys_slots`` is a request-local deque populated at ``freeze()``.  It
    contains the request's preallocated resident pages that are currently not
    bound to any logical page.  Decode never touches the global ``free_ids``.
    """
    logical_length: int = 0
    prompt_length: int = 0
    max_resident_pages: int = 0
    block_table: dict[int, int] = field(default_factory=dict)
    free_phys_slots: deque = field(default_factory=deque)
    skip_far_by_layer: list = field(default_factory=list)
    skip_near_by_layer: list = field(default_factory=list)
    prefill_topk_ids_by_layer: dict = field(default_factory=dict)
    first_decode_seeded: bool = False
    frozen: bool = False
    active: bool = True
    # Decode-incremental resident-page cache.  ``resident_page_ranges`` is the
    # merged logical-page union across the layer skip-policy groups, and is kept
    # for the last synced length so a +1 token step only diffs interval edges.
    resident_page_ranges: tuple = ()
    last_sync_length: int = 0
    skip_policy_groups: tuple = ()
    # Incremental low-dimensional visible projection, keyed by
    # (layer_idx, dims.data_ptr(), dims.numel(), block_size).
    projection_cache: dict = field(default_factory=dict)


class GraphReadyPagedRegionLayer:
    """Per-layer fixed-size physical arena with a PAD safety page.

    This class deliberately owns only physical storage and the free-list.  The
    cross-layer request state and shared metadata are coordinated by
    ``GraphReadyPagedRegionCache``.
    """

    def __init__(
        self,
        *,
        geometry,
        max_physical_pages,
        num_kv_heads,
        num_query_heads=None,
        head_dim_k,
        head_dim_v,
        dtype,
        device,
    ):
        self.geometry = geometry
        self.max_physical_pages = max_physical_pages
        self.PAD_PAGE_ID = max_physical_pages
        self.num_kv_heads = num_kv_heads
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.dtype = dtype
        self.device = device

        # 物理池：max_physical_pages 个业务页 + 1 个 PAD 页。
        # 布局 [page_id, 1, num_kv_heads, page_size, head_dim]。
        total_pages = max_physical_pages + 1
        page = self.geometry.page
        self.kv_pool_k = torch.empty(
            (total_pages, 1, num_kv_heads, page, head_dim_k),
            dtype=dtype, device=device)
        self.kv_pool_v = torch.empty(
            (total_pages, 1, num_kv_heads, page, head_dim_v),
            dtype=dtype, device=device)

        # PAD 页必须全零。无效 block_table 条目指向它，Kernel 可以不分支地
        # 计算 Base + physical_id * stride。
        self.kv_pool_k[self.PAD_PAGE_ID].zero_()
        self.kv_pool_v[self.PAD_PAGE_ID].zero_()

        # 只把业务页放入全局空闲池。PAD_PAGE_ID 永远不参与分配。
        self.free_ids = list(range(max_physical_pages))
        heapq.heapify(self.free_ids)

    def alloc_pages(self, count):
        if len(self.free_ids) < count:
            raise RuntimeError('graph-ready physical page pool exhausted')
        return [heapq.heappop(self.free_ids) for _ in range(count)]

    def free_pages(self, physical_ids):
        for physical_id in physical_ids:
            if physical_id == self.PAD_PAGE_ID:
                raise RuntimeError('PAD page leaked into free_ids')
            heapq.heappush(self.free_ids, physical_id)


class PagedRegionLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.pages = {}
        self.kv_pool = {}
        self.free_ids = []
        self.next_id = 0
        self.logical_length = self.prompt_length = 0
        self.frozen = self.skip_near = self.skip_far = False
        self.reclaimed_pages = self.reused_page_ids = 0
        self.aging_copy_bytes = 0
        self.write_bytes = 0
        self._tables = None
        self._reclaim_cursor = None
        self.nfa_receipt = None
        self._stream_id = None
        # vLLM-style batch state: one shared physical pool per layer and a
        # block table plus region state for every active request.
        self.block_tables = []
        self.seq_lens = []
        self.prompt_lengths = []
        self.frozen_flags = []
        self.skip_near_flags = []
        self.skip_far_flags = []
        self.reclaim_cursors = []
        self.active_flags = []
        self.nfa_receipts = []
        self.free_slots = []
        self.next_slot = 0
        self._k_ptr = None
        self._v_ptr = None

    def _check_stream(self, device):
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device).cuda_stream
            if self._stream_id is not None and stream != self._stream_id:
                raise ValueError("paged request must stay on its owning CUDA stream")
            self._stream_id = stream

    def lazy_initialization(self, key_states):
        pass

    def intervals(self, length=None, resident=False, request_id=None):
        if request_id is None:
            length = self.logical_length if length is None else length
            skip_near, skip_far = self.skip_near, self.skip_far
        else:
            length = self.seq_lens[request_id] if length is None else length
            skip_near = self.skip_near_flags[request_id]
            skip_far = self.skip_far_flags[request_id]
        s, f, n = self.geometry.bounds(length)
        spans = [(0, s), (n, length)]
        if not skip_far: spans.append((s, f))
        if not skip_near or (resident and not skip_far): spans.append((f, n))
        return [(a,b) for a,b in sorted(spans) if a < b]

    def visible_positions(self, request_id=None):
        return [p for a,b in self.intervals(request_id=request_id) for p in range(a,b)]

    def _request_context(self, request_id=None):
        if request_id is None:
            return self.logical_length, self.skip_near, self.skip_far
        request_id = self._batch_index(request_id)
        return (self.seq_lens[request_id], self.skip_near_flags[request_id],
                self.skip_far_flags[request_id])

    def page_label(self, logical_page, request_id=None):
        length, _, _ = self._request_context(request_id)
        sink_hi, far_hi, near_hi = self.geometry.bounds(length)
        lo = logical_page * self.geometry.page
        hi = min(lo + self.geometry.page, length)
        if hi <= lo:
            return 0
        label = 0
        if lo < sink_hi:
            label |= SINK
        if max(lo, sink_hi) < min(hi, far_hi):
            label |= FAR
        if max(lo, far_hi) < min(hi, near_hi):
            label |= NEAR
        if max(lo, near_hi) < min(hi, length):
            label |= SWA
        return label

    def skip_mask(self, request_id=None):
        _, skip_near, skip_far = self._request_context(request_id)
        mask = 0
        if skip_far:
            mask |= FAR
        if skip_near:
            mask |= NEAR
        return mask

    def resident_mask(self, request_id=None):
        _, skip_near, skip_far = self._request_context(request_id)
        mask = SINK | SWA
        if not skip_far:
            mask |= FAR
        if not skip_near or not skip_far:
            mask |= NEAR
        return mask

    def visible_mask(self, logical_page, request_id=None):
        return self.page_label(logical_page, request_id) & ~self.skip_mask(request_id) & (SINK | FAR | NEAR | SWA)

    def page_reclaimable(self, logical_page, request_id=None):
        return (self.page_label(logical_page, request_id) & self.resident_mask(request_id)) == 0

    def page_labels(self, request_id=None):
        if request_id is None:
            pages = self.pages
        else:
            request_id = self._batch_index(request_id)
            pages = self.block_tables[request_id]
        return {lp: self.page_label(lp, request_id) for lp in pages}

    def _reclaim(self):
        sink_hi, _, _ = self.geometry.bounds(self.logical_length)
        size = self.geometry.page
        if self._reclaim_cursor is None:
            self._reclaim_cursor = (sink_hi + size - 1) // size
        candidates = sorted(lp for lp in self.pages if lp >= self._reclaim_cursor)
        for lp in candidates:
            lo, hi = lp*size, min((lp+1)*size,self.logical_length)
            if self.page_reclaimable(lp):
                physical, _, _ = self.pages.pop(lp)
                self.kv_pool.pop(physical, None)
                heapq.heappush(self.free_ids, physical)
                self.reclaimed_pages += 1
                self._tables = None
                self._reclaim_cursor = lp + 1
                continue
            if lo < sink_hi:
                # Mixed/full sink pages are protected forever by the fixed sink prefix.
                self._reclaim_cursor = lp + 1
                continue
            # Later pages can still age into a reclaimable gap; stop this scan.
            break

    def _batch_index(self, request_id):
        if (request_id < 0 or request_id >= len(self.active_flags)
                or not self.active_flags[request_id]):
            raise ValueError(f'unknown or inactive request {request_id}')
        return request_id

    def _batch_extend(self, request_id):
        while len(self.active_flags) <= request_id:
            self.block_tables.append({})
            self.seq_lens.append(0)
            self.prompt_lengths.append(0)
            self.frozen_flags.append(False)
            self.skip_near_flags.append(False)
            self.skip_far_flags.append(False)
            self.reclaim_cursors.append(None)
            self.active_flags.append(False)
            self.nfa_receipts.append(None)

    def add_request(self, request_id=None):
        if request_id is None:
            if self.free_slots:
                request_id = heapq.heappop(self.free_slots)
            else:
                request_id = self.next_slot
                self.next_slot += 1
        elif request_id < 0:
            raise ValueError('request_id must be non-negative')
        elif request_id in self.free_slots:
            self.free_slots.remove(request_id)
            heapq.heapify(self.free_slots)
        elif request_id < self.next_slot:
            raise ValueError(f'request slot {request_id} is not free')
        else:
            self.next_slot = request_id + 1
        self._batch_extend(request_id)
        if self.active_flags[request_id]:
            raise ValueError(f'request {request_id} already active')
        self.block_tables[request_id] = {}
        self.seq_lens[request_id] = 0
        self.prompt_lengths[request_id] = 0
        self.frozen_flags[request_id] = False
        self.skip_near_flags[request_id] = False
        self.skip_far_flags[request_id] = False
        self.reclaim_cursors[request_id] = None
        self.active_flags[request_id] = True
        self.nfa_receipts[request_id] = None
        return request_id

    def remove_request(self, request_id):
        request_id = self._batch_index(request_id)
        for logical_page, physical in list(self.block_tables[request_id].items()):
            self.kv_pool.pop(physical, None)
            heapq.heappush(self.free_ids, physical)
        self.block_tables[request_id] = {}
        self.seq_lens[request_id] = 0
        self.prompt_lengths[request_id] = 0
        self.frozen_flags[request_id] = False
        self.skip_near_flags[request_id] = False
        self.skip_far_flags[request_id] = False
        self.reclaim_cursors[request_id] = None
        self.active_flags[request_id] = False
        self.nfa_receipts[request_id] = None
        heapq.heappush(self.free_slots, request_id)
        self._k_ptr = self._v_ptr = None

    def _batch_allocate(self, request_id, logical_page, keys, values):
        if logical_page in self.block_tables[request_id]:
            return self.block_tables[request_id][logical_page]
        if self.free_ids:
            physical = heapq.heappop(self.free_ids)
            self.reused_page_ids += 1
        else:
            physical = self.next_id
            self.next_id += 1
        page = self.geometry.page
        expected = (1, keys.shape[1], page, keys.shape[-1])
        if physical in self.kv_pool:
            k, v = self.kv_pool[physical]
            if (tuple(k.shape), tuple(v.shape), k.dtype, v.dtype, k.device, v.device) != (
                    expected, (1, values.shape[1], page, values.shape[-1]),
                    keys.dtype, values.dtype, keys.device, values.device):
                raise ValueError('physical page reuse with incompatible KV shape/dtype/device')
        else:
            k = torch.empty(expected, dtype=keys.dtype, device=keys.device)
            v = torch.empty((1, values.shape[1], page, values.shape[-1]),
                            dtype=values.dtype, device=values.device)
            self.kv_pool[physical] = (k, v)
        self.block_tables[request_id][logical_page] = physical
        self._k_ptr = self._v_ptr = None
        return physical

    def _batch_reclaim(self, request_id):
        request_id = self._batch_index(request_id)
        length = self.seq_lens[request_id]
        if length <= 0:
            return
        sink_hi, _, _ = self.geometry.bounds(length)
        size = self.geometry.page
        if self.reclaim_cursors[request_id] is None:
            self.reclaim_cursors[request_id] = (sink_hi + size - 1) // size
        table = self.block_tables[request_id]
        candidates = sorted(lp for lp in table if lp >= self.reclaim_cursors[request_id])
        for lp in candidates:
            lo, hi = lp * size, min((lp + 1) * size, length)
            if self.page_reclaimable(lp, request_id):
                physical = table.pop(lp)
                self.kv_pool.pop(physical, None)
                heapq.heappush(self.free_ids, physical)
                self.reclaimed_pages += 1
                self.reclaim_cursors[request_id] = lp + 1
                self._k_ptr = self._v_ptr = None
                continue
            if lo < sink_hi:
                self.reclaim_cursors[request_id] = lp + 1
                continue
            break

    def write_request(self, request_id, keys, values, cache_kwargs=None):
        request_id = self._batch_index(request_id)
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3] or keys.shape[0] != 1:
            raise ValueError('paged prefill requires unpadded B1 and matching KV heads/tokens')
        if keys.device != values.device or keys.dtype != values.dtype or keys.shape[-1] < 1 or values.shape[-1] < 1:
            raise ValueError('invalid KV dtype/device/width')
        if keys.requires_grad or values.requires_grad:
            raise ValueError('inference-only cache')
        self._check_stream(keys.device)
        n = keys.shape[-2]
        if n <= 0:
            raise ValueError('empty update')
        start = self.seq_lens[request_id]
        positions = (cache_kwargs or {}).get('cache_position')
        if positions is not None and positions.tolist() != list(range(start, start + n)):
            raise ValueError('nonmonotonic cache positions')
        prefill = start == 0
        if not prefill:
            if not self.frozen_flags[request_id] or n != 1:
                raise ValueError('freeze after one-shot prefill; decode one token')
            if start - self.prompt_lengths[request_id] >= self.geometry.max_new_tokens:
                raise ValueError('generation capacity exceeded')
            sample_k, sample_v = self._request_sample(request_id)
            if (keys.dtype != sample_k.dtype or values.dtype != sample_v.dtype
                    or keys.device != sample_k.device or values.device != sample_v.device):
                raise ValueError('KV dtype/device changed across request')
            if keys.shape[1:] != (sample_k.shape[1], 1, sample_k.shape[-1]) or values.shape[-1] != sample_v.shape[-1]:
                raise ValueError('KV shape changed across request')

        page = self.geometry.page
        new_length = start + n
        allocated = []
        touched = {}
        free_before = list(self.free_ids)
        next_before = self.next_id
        reused_before = self.reused_page_ids
        write_before = self.write_bytes
        try:
            cursor = 0
            while cursor < n:
                absolute = start + cursor
                lp, offset = divmod(absolute, page)
                count = min(page - offset, n - cursor)
                if lp not in self.block_tables[request_id]:
                    self._batch_allocate(request_id, lp, keys, values)
                    allocated.append(lp)
                physical = self.block_tables[request_id][lp]
                k, v = self.kv_pool[physical]
                if lp not in allocated:
                    key = (physical, offset, count)
                    if key not in touched:
                        touched[key] = (k[..., offset:offset + count, :].clone(),
                                        v[..., offset:offset + count, :].clone())
                k[..., offset:offset + count, :].copy_(keys[..., cursor:cursor + count, :])
                v[..., offset:offset + count, :].copy_(values[..., cursor:cursor + count, :])
                self.write_bytes += (k.shape[1] * k.shape[-1] + v.shape[1] * v.shape[-1]) * count * k.element_size()
                cursor += count
        except Exception:
            for (physical, offset, count), (old_k, old_v) in touched.items():
                k, v = self.kv_pool[physical]
                k[..., offset:offset + count, :].copy_(old_k)
                v[..., offset:offset + count, :].copy_(old_v)
            for lp in allocated:
                physical = self.block_tables[request_id].pop(lp, None)
                self.kv_pool.pop(physical, None)
            self.free_ids = free_before
            heapq.heapify(self.free_ids)
            self.next_id = next_before
            self.reused_page_ids = reused_before
            self.write_bytes = write_before
            self._k_ptr = self._v_ptr = None
            raise
        self.seq_lens[request_id] = new_length
        if prefill:
            self.prompt_lengths[request_id] = new_length
        if self.frozen_flags[request_id]:
            self._batch_reclaim(request_id)
        return keys, values

    def _request_sample(self, request_id):
        if not self.block_tables[request_id]:
            raise ValueError('request has no resident KV pages')
        physical = next(iter(self.block_tables[request_id].values()))
        return self.kv_pool[physical]

    def freeze_request(self, request_id, skip_near=False, skip_far=False):
        request_id = self._batch_index(request_id)
        if self.frozen_flags[request_id]:
            if (skip_near, skip_far) != (self.skip_near_flags[request_id], self.skip_far_flags[request_id]):
                raise ValueError('cannot change frozen gates')
            return
        if not self.seq_lens[request_id]:
            raise ValueError('prefill before freeze')
        sample_k, _ = self._request_sample(request_id)
        self._check_stream(sample_k.device)
        self.skip_near_flags[request_id], self.skip_far_flags[request_id] = bool(skip_near), bool(skip_far)
        self.frozen_flags[request_id] = True
        self._batch_reclaim(request_id)

    def _allocate_decode_page(self, request_id, logical_page):
        if logical_page in self.block_tables[request_id]:
            return self.block_tables[request_id][logical_page]
        sample_k, sample_v = self._request_sample(request_id)
        return self._batch_allocate(request_id, logical_page,
                                    sample_k[:, :, :1], sample_v[:, :, :1])

    def decode_slot_mapping(self, request_ids=None, device=None):
        if request_ids is None:
            request_ids = list(range(self.next_slot))
        request_ids = list(request_ids)
        page = self.geometry.page
        mapping = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                mapping.append(-1)
                continue
            if not self.frozen_flags[request_id]:
                raise ValueError('freeze all active requests before batch decode')
            length = self.seq_lens[request_id]
            lp, offset = divmod(length, page)
            if lp not in self.block_tables[request_id]:
                self._allocate_decode_page(request_id, lp)
            physical = self.block_tables[request_id].get(lp)
            mapping.append(physical * page + offset if physical is not None else -1)
        return torch.tensor(mapping, dtype=torch.int64, device=device)

    def write_decode(self, request_ids, keys, values, slot_mapping):
        request_ids = list(request_ids)
        # Each active request owns exactly one next-token slot in this API.
        # Duplicate rows would overwrite that slot and advance its length twice.
        active = [r for r in request_ids
                  if 0 <= r < len(self.active_flags) and self.active_flags[r]]
        if len(active) != len(set(active)):
            raise ValueError('duplicate active request_ids in batch decode')
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3] or keys.shape[-2] != 1:
            raise ValueError('batch decode requires matching B1-token KV')
        if keys.shape[0] != len(request_ids):
            raise ValueError('request_ids length must match KV batch')
        if slot_mapping.shape != (len(request_ids),):
            raise ValueError('slot_mapping must be [B]')
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError('invalid batch KV dtype/device')
        self._check_stream(keys.device)
        page = self.geometry.page
        writes = []
        for batch, request_id in enumerate(request_ids):
            mapping = int(slot_mapping[batch])
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                if mapping != -1:
                    raise ValueError(f'inactive request {request_id} must use -1 slot mapping')
                continue
            if not self.frozen_flags[request_id]:
                raise ValueError('freeze all active requests before batch decode')
            length = self.seq_lens[request_id]
            if length - self.prompt_lengths[request_id] >= self.geometry.max_new_tokens:
                raise ValueError('generation capacity exceeded')
            lp, offset = divmod(length, page)
            physical = self.block_tables[request_id].get(lp)
            expected = physical * page + offset if physical is not None else -1
            if mapping != expected:
                raise ValueError(f'stale slot mapping for request {request_id}: {mapping} != {expected}')
            if physical is None:
                raise ValueError(f'missing decode page for request {request_id}')
            k, v = self.kv_pool[physical]
            if k.dtype != keys.dtype or v.dtype != values.dtype or k.device != keys.device or v.device != values.device:
                raise ValueError('batch KV dtype/device mismatch')
            if k.shape[1] != keys.shape[1] or v.shape[1] != values.shape[1] or k.shape[-1] != keys.shape[-1] or v.shape[-1] != values.shape[-1]:
                raise ValueError('batch KV shape mismatch')
            writes.append((request_id, physical, offset, batch))

        # Validate every row before committing any write, so a bad batch cannot
        # partially advance request lengths.
        touched = []
        for request_id, physical, offset, batch in writes:
            k, v = self.kv_pool[physical]
            touched.append((k, v, k[..., offset:offset + 1, :].clone(), v[..., offset:offset + 1, :].clone()))
        try:
            for (request_id, physical, offset, batch), (k, v, _, _) in zip(writes, touched):
                k[..., offset:offset + 1, :].copy_(keys[batch:batch + 1])
                v[..., offset:offset + 1, :].copy_(values[batch:batch + 1])
                self.write_bytes += (k.shape[1] * k.shape[-1] + v.shape[1] * v.shape[-1]) * k.element_size()
        except Exception:
            for (request_id, physical, offset, batch), (k, v, old_k, old_v) in zip(writes, touched):
                k[..., offset:offset + 1, :].copy_(old_k)
                v[..., offset:offset + 1, :].copy_(old_v)
            raise
        for request_id, _, _, _ in writes:
            self.seq_lens[request_id] += 1
            self._batch_reclaim(request_id)
        return keys, values

    def block_table_tensor(self, request_id, device=None):
        request_id = self._batch_index(request_id)
        page = self.geometry.page
        logical_pages = (self.seq_lens[request_id] + page - 1) // page
        table = torch.full((logical_pages,), -1, dtype=torch.int64, device=device)
        for lp, physical in self.block_tables[request_id].items():
            if lp < logical_pages:
                table[lp] = physical
        return table

    def _resolve_request_ids(self, request_ids=None):
        if request_ids is None:
            return list(range(self.next_slot))
        return list(request_ids)

    def _default_decode_query_lengths(self, request_ids):
        lengths = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                lengths.append(0)
            elif not self.frozen_flags[request_id]:
                raise ValueError('freeze all active requests before batch decode')
            else:
                lengths.append(1)
        return lengths

    def query_start_loc(self, request_ids=None, query_lengths=None, device=None):
        """vLLM-style input-side cumsum for a flat multi-request query batch."""
        request_ids = self._resolve_request_ids(request_ids)
        if query_lengths is None:
            query_lengths = self._default_decode_query_lengths(request_ids)
        query_lengths = [int(x) for x in query_lengths]
        if len(query_lengths) != len(request_ids):
            raise ValueError('query_lengths must match request_ids')
        if any(x < 0 for x in query_lengths):
            raise ValueError('query_lengths must be non-negative')
        lengths = torch.tensor(query_lengths, dtype=torch.int64)
        loc = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        loc[1:] = lengths.cumsum(0).to(torch.int32)
        return loc

    def seq_lens_tensor(self, request_ids=None, device=None):
        request_ids = self._resolve_request_ids(request_ids)
        values = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                values.append(0)
            else:
                values.append(self.seq_lens[request_id])
        return torch.tensor(values, dtype=torch.int64, device=device)

    def block_table_batch(self, request_ids=None, device=None, max_blocks=None):
        """Pad per-request block tables to a fixed [B, max_blocks] tensor."""
        request_ids = self._resolve_request_ids(request_ids)
        page = self.geometry.page
        rows = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                rows.append([])
                continue
            logical_pages = (self.seq_lens[request_id] + page - 1) // page
            rows.append([self.block_tables[request_id].get(lp, -1) for lp in range(logical_pages)])
        max_blocks = max((len(row) for row in rows), default=0) if max_blocks is None else max_blocks
        table = torch.full((len(request_ids), max_blocks), -1, dtype=torch.int64, device=device)
        for batch, row in enumerate(rows):
            if row:
                table[batch, :len(row)] = torch.tensor(row, dtype=torch.int64, device=device)
        return table

    def region_label_masks(self, request_ids=None, device=None, max_blocks=None):
        """Per-request page-level region labels, aligned with block_table_batch."""
        request_ids = self._resolve_request_ids(request_ids)
        page = self.geometry.page
        rows = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                rows.append([])
                continue
            logical_pages = (self.seq_lens[request_id] + page - 1) // page
            rows.append([self.page_label(lp, request_id) if lp in self.block_tables[request_id] else 0
                         for lp in range(logical_pages)])
        max_blocks = max((len(row) for row in rows), default=0) if max_blocks is None else max_blocks
        masks = torch.zeros((len(request_ids), max_blocks), dtype=torch.uint8, device=device)
        for batch, row in enumerate(rows):
            if row:
                masks[batch, :len(row)] = torch.tensor(row, dtype=torch.uint8, device=device)
        return masks

    def region_visible_masks(self, request_ids=None, device=None, max_blocks=None):
        """Per-request page-level visible masks, aligned with block_table_batch."""
        request_ids = self._resolve_request_ids(request_ids)
        page = self.geometry.page
        rows = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                rows.append([])
                continue
            logical_pages = (self.seq_lens[request_id] + page - 1) // page
            rows.append([self.visible_mask(lp, request_id) if lp in self.block_tables[request_id] else 0
                         for lp in range(logical_pages)])
        max_blocks = max((len(row) for row in rows), default=0) if max_blocks is None else max_blocks
        masks = torch.zeros((len(request_ids), max_blocks), dtype=torch.uint8, device=device)
        for batch, row in enumerate(rows):
            if row:
                masks[batch, :len(row)] = torch.tensor(row, dtype=torch.uint8, device=device)
        return masks

    def token_visible_mask(self, request_ids=None, device=None):
        """Token-level region visibility for each request, padded to max seq_len."""
        request_ids = self._resolve_request_ids(request_ids)
        max_length = 0
        rows = []
        for request_id in request_ids:
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                rows.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue
            length = self.seq_lens[request_id]
            max_length = max(max_length, length)
            sink_hi, far_hi, near_hi = self.geometry.bounds(length)
            skip_near = self.skip_near_flags[request_id]
            skip_far = self.skip_far_flags[request_id]
            pos = torch.arange(length, device=device)
            visible = ((pos < sink_hi) | (pos >= near_hi)
                       | ((pos >= sink_hi) & (pos < far_hi) & (not skip_far))
                       | ((pos >= far_hi) & (pos < near_hi) & (not skip_near)))
            rows.append(visible)
        mask = torch.zeros((len(request_ids), max_length), dtype=torch.bool, device=device)
        for batch, row in enumerate(rows):
            if row.numel():
                mask[batch, :row.numel()] = row
        return mask

    def slot_mapping_for_write(self, request_ids=None, query_lengths=None, device=None):
        """Flat write locations for the next `query_lengths` tokens per request."""
        request_ids = self._resolve_request_ids(request_ids)
        if query_lengths is None:
            query_lengths = self._default_decode_query_lengths(request_ids)
        query_lengths = [int(x) for x in query_lengths]
        if len(query_lengths) != len(request_ids):
            raise ValueError('query_lengths must match request_ids')
        if any(x < 0 for x in query_lengths):
            raise ValueError('query_lengths must be non-negative')
        page = self.geometry.page
        mapping = []
        for request_id, query_length in zip(request_ids, query_lengths):
            if request_id < 0 or request_id >= len(self.active_flags) or not self.active_flags[request_id]:
                if query_length != 0:
                    raise ValueError(f'inactive request {request_id} cannot accept queries')
                continue
            if not self.frozen_flags[request_id]:
                raise ValueError('freeze all active requests before batch write')
            start = self.seq_lens[request_id]
            if start - self.prompt_lengths[request_id] + query_length > self.geometry.max_new_tokens:
                raise ValueError('generation capacity exceeded')
            for token_offset in range(query_length):
                absolute = start + token_offset
                lp, offset = divmod(absolute, page)
                if lp not in self.block_tables[request_id]:
                    if not self.block_tables[request_id]:
                        raise ValueError('new-request prefill slot mapping requires KV shape; use write_request')
                    self._allocate_decode_page(request_id, lp)
                physical = self.block_tables[request_id][lp]
                mapping.append(physical * page + offset)
        return torch.tensor(mapping, dtype=torch.int64, device=device)

    def region_metadata(self, request_ids=None, query_lengths=None, device=None,
                        include_token_mask=False):
        request_ids = self._resolve_request_ids(request_ids)
        query_start_loc = self.query_start_loc(request_ids, query_lengths, device=device)
        seq_lens = self.seq_lens_tensor(request_ids, device=device)
        block_table = self.block_table_batch(request_ids, device=device)
        slot_mapping = self.slot_mapping_for_write(request_ids, query_lengths, device=device)
        metadata = dict(request_ids=request_ids, query_start_loc=query_start_loc,
                        seq_lens=seq_lens, block_table=block_table,
                        slot_mapping=slot_mapping,
                        region_label_masks=self.region_label_masks(request_ids, device=device),
                        region_visible_masks=self.region_visible_masks(request_ids, device=device))
        if include_token_mask:
            metadata['token_visible_mask'] = self.token_visible_mask(request_ids, device=device)
        return metadata

    def physical_pointer_tables(self):
        if not self.kv_pool:
            raise ValueError('physical page pool is empty')
        device = next(iter(self.kv_pool.values()))[1].device
        if self._k_ptr is None or self._v_ptr is None or self._k_ptr.numel() != self.next_id:
            k_ptr = torch.zeros(self.next_id, dtype=torch.int64, device=device)
            v_ptr = torch.zeros(self.next_id, dtype=torch.int64, device=device)
            for physical, (k, v) in self.kv_pool.items():
                if physical < self.next_id:
                    k_ptr[physical] = k.data_ptr()
                    v_ptr[physical] = v.data_ptr()
            self._k_ptr, self._v_ptr = k_ptr, v_ptr
        return self._k_ptr, self._v_ptr

    def freeze(self, skip_near=False, skip_far=False):
        if self.frozen:
            if (skip_near,skip_far)!=(self.skip_near,self.skip_far):
                raise ValueError('cannot change frozen gates')
            return
        if not self.logical_length: raise ValueError('prefill before freeze')
        self._check_stream(next(iter(self.pages.values()))[1].device)
        self.skip_near,self.skip_far=bool(skip_near),bool(skip_far)
        self.frozen=True
        self._reclaim()

    def _allocate(self, lp, keys, values):
        if self.free_ids:
            physical=heapq.heappop(self.free_ids);self.reused_page_ids+=1
        else:
            physical=self.next_id;self.next_id+=1
        p=self.geometry.page
        k=torch.empty((1,keys.shape[1],p,keys.shape[-1]),dtype=keys.dtype,device=keys.device)
        v=torch.empty((1,values.shape[1],p,values.shape[-1]),dtype=values.dtype,device=values.device)
        self.pages[lp]=(physical,k,v)
        self.kv_pool[physical]=(k,v)
        self._tables=None

    def update(self, keys, values, cache_kwargs=None):
        request_id = (cache_kwargs or {}).get('request_id')
        if request_id is not None:
            return self.write_request(request_id, keys, values, cache_kwargs)
        if keys.ndim!=4 or values.ndim!=4 or keys.shape[:3]!=values.shape[:3] or keys.shape[0]!=1:
            raise ValueError('paged cache requires unpadded B1 and matching KV heads/tokens')
        if keys.device!=values.device or keys.dtype!=values.dtype or keys.shape[-1]<1 or values.shape[-1]<1:
            raise ValueError('invalid KV dtype/device/width')
        if keys.requires_grad or values.requires_grad: raise ValueError('inference-only cache')
        self._check_stream(keys.device)
        n=keys.shape[-2]
        if n<=0: raise ValueError('empty update')
        positions=(cache_kwargs or {}).get('cache_position')
        if positions is not None and positions.tolist()!=list(range(self.logical_length,self.logical_length+n)):
            raise ValueError('nonmonotonic cache positions')
        prefill=not self.logical_length
        if not prefill:
            if not self.frozen or n!=1: raise ValueError('freeze after one-shot prefill; decode one token')
            if self.logical_length-self.prompt_length>=self.geometry.max_new_tokens:
                raise ValueError('generation capacity exceeded')
            _,samplek,samplev=next(iter(self.pages.values()))
            if keys.dtype!=samplek.dtype or values.dtype!=samplev.dtype or keys.device!=samplek.device or values.device!=samplev.device:
                raise ValueError('KV dtype/device changed across request')
            if keys.shape[1:]!=(samplek.shape[1],1,samplek.shape[-1]) or values.shape[-1]!=samplev.shape[-1]:
                raise ValueError('KV shape changed across request')
        p=self.geometry.page
        start=self.logical_length
        new_length=start+n
        allocated=[]
        touched={}
        free_before=list(self.free_ids)
        next_before=self.next_id
        reused_before=self.reused_page_ids
        tables_before=self._tables
        write_before=self.write_bytes
        try:
            cursor=0
            while cursor<n:
                absolute=start+cursor;lp,offset=divmod(absolute,p)
                count=min(p-offset,n-cursor)
                if lp not in self.pages:
                    self._allocate(lp,keys,values)
                    allocated.append(lp)
                _,k,v=self.pages[lp]
                if lp not in allocated and lp not in touched:
                    # 仅备份本次覆盖的尾部槽位；分配、拷贝或回收前异常均可原子回滚。
                    touched[lp]=(offset,count,
                        k[...,offset:offset+count,:].clone(),v[...,offset:offset+count,:].clone())
                k[...,offset:offset+count,:].copy_(keys[...,cursor:cursor+count,:])
                v[...,offset:offset+count,:].copy_(values[...,cursor:cursor+count,:])
                self.write_bytes+=(k.shape[1]*k.shape[-1]+v.shape[1]*v.shape[-1])*count*k.element_size()
                cursor+=count
        except Exception:
            for lp,(offset,count,old_k,old_v) in touched.items():
                _,k,v=self.pages[lp]
                k[...,offset:offset+count,:].copy_(old_k)
                v[...,offset:offset+count,:].copy_(old_v)
            for lp in allocated:
                entry = self.pages.pop(lp, None)
                if entry is not None:
                    self.kv_pool.pop(entry[0], None)
            self.free_ids=free_before
            heapq.heapify(self.free_ids)
            self.next_id=next_before
            self.reused_page_ids=reused_before
            self._tables=tables_before
            self.write_bytes=write_before
            raise
        # 只有全部页写入成功才提交逻辑边界并回收旧页，避免半追加状态。
        self.logical_length=new_length
        if prefill:self.prompt_length=new_length
        if self.frozen:self._reclaim()
        # Prefill returns original complete tensors. Decode is consumed by the
        # paged adapter, which must not interpret these as a gathered history.
        return keys,values

    def pointer_tables(self):
        self._check_stream(next(iter(self.pages.values()))[1].device)
        n=(self.logical_length+self.geometry.page-1)//self.geometry.page
        if self._tables is None or self._tables[0].numel()!=n:
            device=next(iter(self.pages.values()))[1].device
            self._tables=tuple(torch.tensor([self.pages[i][j].data_ptr() if i in self.pages else 0 for i in range(n)],device=device,dtype=torch.int64) for j in (1,2))
        return self._tables

    def gather(self):
        """Concatenate visible full-width K/V for native SDPA consumers."""
        keys, values = [], []
        page = self.geometry.page
        for a, b in self.intervals():
            for lp in range(a // page, (b - 1) // page + 1):
                if lp not in self.pages:
                    continue
                _, k, v = self.pages[lp]
                lo = max(a, lp * page)
                hi = min(b, (lp + 1) * page)
                keys.append(k[..., lo - lp * page:hi - lp * page, :])
                values.append(v[..., lo - lp * page:hi - lp * page, :])
        if not keys:
            raise ValueError('paged cache has no visible KV to gather')
        return torch.cat(keys, 2), torch.cat(values, 2)

    def project_visible(self, dims, block_size):
        """Only low-dimensional index scratch; never a full-width KV gather."""
        from ..ops.paged_region_kernels import project_visible
        return project_visible(self,dims,block_size)

    def get_seq_length(self):return self.logical_length
    def get_max_cache_shape(self):return -1
    def get_mask_sizes(self,cache_position):
        q=cache_position if isinstance(cache_position,int) else cache_position.shape[0]
        return self.logical_length+q,0
    def reset(self):raise ValueError('create a fresh request cache')
    def reorder_cache(self,beam_idx):raise ValueError('beam/reorder unsupported')

    def report(self, request_id=None):
        if request_id is not None:
            request_id = self._batch_index(request_id)
            length = self.seq_lens[request_id]
            spans = self.intervals(resident=True, request_id=request_id)
            logical_retained = sum(b - a for a, b in spans)
            table = self.block_tables[request_id]
            kv = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                     for physical, (k, v) in self.kv_pool.items()
                     if physical in table.values())
            return dict(request_id=request_id, logical_length=length,
                        prompt_length=self.prompt_lengths[request_id],
                        active=self.active_flags[request_id],
                        frozen=self.frozen_flags[request_id],
                        skip_near=self.skip_near_flags[request_id],
                        skip_far=self.skip_far_flags[request_id],
                        skip_mask=self.skip_mask(request_id),
                        resident_mask=self.resident_mask(request_id),
                        region_labels=self.page_labels(request_id),
                        visible_masks={lp: self.visible_mask(lp, request_id) for lp in table},
                        live_pages=len(table),
                        logical_to_physical=dict(table),
                        resident_tokens=logical_retained,
                        physical_token_slots=len(table) * self.geometry.page,
                        inactive_retained_slots=len(table) * self.geometry.page - logical_retained,
                        attention_tokens=sum(b - a for a, b in self.intervals(request_id=request_id)),
                        kv_bytes=kv,
                        device_page_table_bytes=0,
                        metadata_scope='request block table only; global pointer tables excluded',
                        reclaimed_pages=self.reclaimed_pages,
                        reused_page_ids=self.reused_page_ids,
                        aging_copy_bytes=self.aging_copy_bytes,
                        kv_write_bytes=self.write_bytes,
                        allocator_scope='active tensor allocations; caching allocator reserved memory is separate')
        spans=self.intervals(resident=True)
        logical_retained=sum(b-a for a,b in spans)
        occupied=sum(min(self.geometry.page,self.logical_length-lp*self.geometry.page) for lp in self.pages)
        kv=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for _,k,v in self.pages.values())
        return dict(logical_length=self.logical_length,skip_near=self.skip_near,skip_far=self.skip_far,
            live_pages=len(self.pages),logical_to_physical={lp:page[0] for lp,page in self.pages.items()},
            resident_tokens=logical_retained,physical_token_slots=len(self.pages)*self.geometry.page,
            inactive_retained_slots=len(self.pages)*self.geometry.page-logical_retained,
            attention_tokens=sum(b-a for a,b in self.intervals()),kv_bytes=kv,
            device_page_table_bytes=0 if self._tables is None else sum(t.numel()*t.element_size() for t in self._tables),
            metadata_scope='GPU pointer tables exact; Python dict/list overhead excluded',
            reclaimed_pages=self.reclaimed_pages,reused_page_ids=self.reused_page_ids,
            aging_copy_bytes=self.aging_copy_bytes,kv_write_bytes=self.write_bytes,
            allocator_scope='active tensor allocations; caching allocator reserved memory is separate')


class PagedRegionCache(Cache):
    nfa_regions=True
    paged_regions=True

    def __init__(self,n_layers,geometry=RegionGeometry()):
        super().__init__(layers=[PagedRegionLayer(geometry) for _ in range(n_layers)])
        self.active_request_id = None

    def add_request(self):
        request_id = self.layers[0].add_request()
        for layer in self.layers[1:]:
            if layer.add_request(request_id) != request_id:
                raise ValueError('request slots diverged across cache layers')
        return request_id

    def remove_request(self, request_id):
        for layer in self.layers:
            layer.remove_request(request_id)
        if self.active_request_id == request_id:
            self.active_request_id = None

    def set_active_request(self, request_id):
        self.layers[0]._batch_index(request_id)
        self.active_request_id = request_id

    def write_request(self, layer_idx, key_states, value_states, request_id=None, cache_kwargs=None):
        if request_id is None:
            if self.active_request_id is None:
                raise ValueError('no active request selected')
            request_id = self.active_request_id
        return self.layers[layer_idx].write_request(request_id, key_states, value_states, cache_kwargs)

    def freeze_request(self, request_id, decisions):
        if set(decisions) != set(range(len(self.layers))):
            raise ValueError('all layers need frozen decisions')
        for layer_idx, layer in enumerate(self.layers):
            layer.freeze_request(request_id, **decisions[layer_idx])

    def query_start_loc(self, layer_idx, request_ids=None, query_lengths=None, device=None):
        return self.layers[layer_idx].query_start_loc(request_ids, query_lengths, device=device)

    def seq_lens_tensor(self, layer_idx, request_ids=None, device=None):
        return self.layers[layer_idx].seq_lens_tensor(request_ids, device=device)

    def block_table_batch(self, layer_idx, request_ids=None, device=None, max_blocks=None):
        return self.layers[layer_idx].block_table_batch(request_ids, device=device, max_blocks=max_blocks)

    def slot_mapping_for_write(self, layer_idx, request_ids=None, query_lengths=None, device=None):
        return self.layers[layer_idx].slot_mapping_for_write(request_ids, query_lengths, device=device)

    def region_metadata(self, layer_idx, request_ids=None, query_lengths=None, device=None,
                        include_token_mask=False):
        return self.layers[layer_idx].region_metadata(
            request_ids, query_lengths=query_lengths, device=device,
            include_token_mask=include_token_mask)

    def decode_slot_mapping(self, layer_idx, request_ids=None, device=None):
        return self.layers[layer_idx].decode_slot_mapping(request_ids, device=device)

    def write_decode(self, layer_idx, key_states, value_states, request_ids, slot_mapping):
        return self.layers[layer_idx].write_decode(request_ids, key_states, value_states, slot_mapping)

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        cache_kwargs = dict(cache_kwargs or {})
        if self.active_request_id is not None and 'request_id' not in cache_kwargs:
            cache_kwargs['request_id'] = self.active_request_id
        if self.layers[layer_idx].frozen:
            if getattr(self, 'nfa_regions', False):
                # 通用 HF attention 会把返回值当作完整历史；只允许直接分页消费者写入。
                raise ValueError('frozen paged decode requires the patched direct-page consumer')
            self.layers[layer_idx].update(key_states,value_states,cache_kwargs)
            return self.layers[layer_idx].gather()
        return super().update(key_states,value_states,layer_idx,cache_kwargs)

    def update_for_paged_decode(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if (cache_kwargs or {}).get('request_ids') is not None:
            return self.write_decode(layer_idx, key_states, value_states,
                                     cache_kwargs['request_ids'],
                                     cache_kwargs['slot_mapping'])
        if not self.layers[layer_idx].frozen:
            raise ValueError('freeze gates before direct-page decode')
        return super().update(key_states,value_states,layer_idx,cache_kwargs)

    def freeze(self,decisions):
        if set(decisions)!=set(range(len(self.layers))):raise ValueError('all layers need frozen decisions')
        for li,layer in enumerate(self.layers):layer.freeze(**decisions[li])

    def report(self):
        layers=[x.report() for x in self.layers]
        active_requests = [i for i, active in enumerate(self.layers[0].active_flags) if active]
        return dict(layers=layers,active_requests=active_requests,
                    kv_bytes=sum(x['kv_bytes'] for x in layers),
                    device_page_table_bytes=sum(x['device_page_table_bytes'] for x in layers))

    def report_request(self, request_id):
        layers = [layer.report(request_id) for layer in self.layers]
        return dict(request_id=request_id, layers=layers,
                    kv_bytes=sum(x['kv_bytes'] for x in layers),
                    device_page_table_bytes=sum(x['device_page_table_bytes'] for x in layers))


class GraphReadyPagedRegionCache:
    """Continuous-batching coordinator for the graph-ready paged cache.

    The block table and slot mapping are common across layers, while
    ``gather_indices`` and ``visible_lens`` are per layer because different
    layers may skip different near/far regions.
    """

    INVALID_REQ_ID = -1
    paged_regions = True
    nfa_regions = True
    is_compileable = False

    def __init__(
        self,
        n_layers,
        geometry,
        *,
        max_requests,
        max_physical_pages,
        max_blocks,
        max_visible_len,
        max_query_tokens,
        num_kv_heads,
        num_query_heads=None,
        head_dim_k,
        head_dim_v,
        dtype,
        device,
        visible_buckets=None,
    ):
        device = torch.device(device)
        self.geometry = geometry
        self.n_layers = n_layers
        self.max_requests = max_requests
        self.max_blocks = max_blocks
        self.max_visible_len = max_visible_len
        self.max_query_tokens = max_query_tokens
        self.num_query_heads = num_kv_heads if num_query_heads is None else num_query_heads

        self.layers = [
            GraphReadyPagedRegionLayer(
                geometry=geometry,
                max_physical_pages=max_physical_pages,
                num_kv_heads=num_kv_heads,
                head_dim_k=head_dim_k,
                head_dim_v=head_dim_v,
                dtype=dtype,
                device=device,
            )
            for _ in range(n_layers)
        ]

        self.states: dict[int, RequestState] = {}
        self.pending_lengths: dict[int, int] = {}
        self.active_request_id = None
        self.active_req_ids = []
        self._pending_free_pages = []
        self.common_dirty = True

        # 全模型共享一套页表和写入 slot。各层持有自己的物理 Arena，但 page id
        # 对齐，所以同一份 logical->physical 映射对所有层有效。
        self.block_table_buf = torch.full(
            (max_requests, max_blocks),
            self.layers[0].PAD_PAGE_ID,
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_buf = torch.full(
            (max_query_tokens,),
            PAD_SLOT,
            dtype=torch.int32,
            device=device,
        )
        self.query_start_loc_buf = torch.zeros(
            (max_requests + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.query_to_req_buf = torch.full(
            (max_query_tokens,),
            -1,
            dtype=torch.int32,
            device=device,
        )

        # 层间异构计算量：每层一套 gather_indices / visible_lens。
        self.gather_indices_buf = torch.full(
            (n_layers, max_requests, max_visible_len),
            INVALID_GATHER_IDX,
            dtype=torch.int32,
            device=device,
        )
        self.visible_lens_buf = torch.zeros(
            (n_layers, max_requests),
            dtype=torch.int32,
            device=device,
        )

        # CPU staging：capture 后每步只 copy_，不在热路径 torch.tensor()。
        self._cpu_gather_staging = torch.full(
            (max_requests, max_visible_len),
            INVALID_GATHER_IDX,
            dtype=torch.int32,
            device='cpu',
        )
        self._gather_logical_tmp = torch.empty(
            (max_visible_len,),
            dtype=torch.int32,
            device=device,
        )

        # Common metadata CPU staging.  Writing every scalar directly to the
        # device buffers launches one tiny CUDA kernel per element and is the
        # dominant GraphReady metadata cost for long contexts.  Fill these CPU
        # tensors once per step and copy the three small buffers in one shot.
        self._cpu_block_table_staging = torch.full(
            (max_requests, max_blocks),
            self.layers[0].PAD_PAGE_ID,
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_slot_mapping_staging = torch.full(
            (max_query_tokens,),
            PAD_SLOT,
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_query_start_loc_staging = torch.zeros(
            (max_requests + 1,),
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_query_to_req_staging = torch.full(
            (max_query_tokens,),
            -1,
            dtype=torch.int32,
            device='cpu',
        )
        if device.type == 'cuda':
            self._cpu_block_table_staging = self._cpu_block_table_staging.pin_memory()
            self._cpu_slot_mapping_staging = self._cpu_slot_mapping_staging.pin_memory()
            self._cpu_query_start_loc_staging = self._cpu_query_start_loc_staging.pin_memory()
            self._cpu_query_to_req_staging = self._cpu_query_to_req_staging.pin_memory()
        self._last_q_buf = torch.zeros(
            (n_layers, max_requests, self.num_query_heads, head_dim_k),
            dtype=dtype,
            device=device,
        )

        # Optional visible-length graph buckets.  ``max_visible_len`` itself is
        # always served by the primary contiguous buffers; smaller buckets own
        # contiguous scratch tensors so graph capture can select a tighter
        # MAX_VISIBLE constant without slicing the primary non-contiguous rows.
        self._gather_indices_by_vis = {}
        self._visible_lens_by_vis = {}
        self.active_visible_bucket = max_visible_len
        if visible_buckets:
            self.ensure_visible_buckets(visible_buckets)

    # ------------------------------------------------------------------
    # Visible-length graph buckets
    # ------------------------------------------------------------------
    def ensure_visible_buckets(self, buckets):
        for width in buckets:
            width = int(width)
            if width <= 0 or width > self.max_visible_len:
                raise ValueError(
                    f'visible bucket {width} outside (0, {self.max_visible_len}]')
            if width == self.max_visible_len:
                continue
            if width in self._gather_indices_by_vis:
                continue
            self._gather_indices_by_vis[width] = torch.full(
                (self.n_layers, self.max_requests, width),
                INVALID_GATHER_IDX,
                dtype=torch.int32,
                device=self.gather_indices_buf.device,
            )
            self._visible_lens_by_vis[width] = torch.zeros(
                (self.n_layers, self.max_requests),
                dtype=torch.int32,
                device=self.visible_lens_buf.device,
            )

    def set_active_visible_bucket(self, width):
        width = int(width)
        if width != self.max_visible_len and width not in self._gather_indices_by_vis:
            raise ValueError(f'visible bucket {width} has not been prepared')
        self.active_visible_bucket = width

    def gather_indices_for_layer(self, layer_idx):
        width = self.active_visible_bucket
        if width != self.max_visible_len:
            return self._gather_indices_by_vis[width][layer_idx]
        return self.gather_indices_buf[layer_idx]

    def visible_lens_for_layer(self, layer_idx):
        width = self.active_visible_bucket
        if width != self.max_visible_len:
            return self._visible_lens_by_vis[width][layer_idx]
        return self.visible_lens_buf[layer_idx]

    def choose_visible_bucket(self, req_ids, visible_buckets):
        """Smallest bucket >= max coarse visible length for this decode batch.

        The NFA Top-K result is always a subset of the coarse visible
        intervals, so bounding by interval length is conservative and requires
        no device-to-host read.
        """
        max_total = 0
        for request_id in req_ids:
            if request_id is None or request_id < 0 or request_id not in self.states:
                continue
            state = self.states[request_id]
            length = self.pending_lengths.get(request_id, state.logical_length)
            seed_max = max(
                (len(ids) for ids in state.prefill_topk_ids_by_layer.values()),
                default=0)
            for layer_idx in range(self.n_layers):
                intervals = self._intervals_for_layer(state, layer_idx, length)
                total = sum(b - a for a, b in intervals) + seed_max
                if total > max_total:
                    max_total = total
        for width in sorted(visible_buckets):
            if width >= max_total:
                return width
        return self.max_visible_len

    # ------------------------------------------------------------------
    # Physical page allocation (all layers advance together)
    # ------------------------------------------------------------------
    def _alloc_pages(self, count):
        self._reclaim_pending_free()
        if count == 0:
            return []
        for layer in self.layers:
            if len(layer.free_ids) < count:
                raise RuntimeError('graph-ready physical page pool exhausted')
        physical_ids = []
        for _ in range(count):
            physical_id = heapq.heappop(self.layers[0].free_ids)
            for layer in self.layers[1:]:
                other = heapq.heappop(layer.free_ids)
                if other != physical_id:
                    raise RuntimeError('physical page free lists diverged across layers')
            physical_ids.append(physical_id)
        return physical_ids

    def _free_pages(self, physical_ids):
        for physical_id in physical_ids:
            for layer in self.layers:
                layer.free_pages([physical_id])

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    def add_request(self, request_id=None):
        if request_id is None:
            request_id = self._next_free_slot()
        if request_id < 0 or request_id >= self.max_requests:
            raise ValueError('request_id outside fixed batch capacity')
        if request_id in self.states:
            raise ValueError(f'request {request_id} already active')
        self.states[request_id] = RequestState()
        self._cpu_block_table_staging[request_id].fill_(
            self.layers[0].PAD_PAGE_ID)
        return request_id

    def _next_free_slot(self):
        for request_id in range(self.max_requests):
            if request_id not in self.states:
                return request_id
        raise RuntimeError('no free request slot')

    def remove_request(self, request_id):
        state = self.states.pop(request_id, None)
        if state is None:
            return
        # 所有业务物理页（已绑定和未绑定）一次性归还全局池。
        physical_ids = set(state.block_table.values())
        physical_ids.update(state.free_phys_slots)
        self._cpu_block_table_staging[request_id].fill_(
            self.layers[0].PAD_PAGE_ID)
        if self.layers[0].kv_pool_k.is_cuda:
            event = torch.cuda.Event()
            event.record()
            self._pending_free_pages.append((event, physical_ids))
        else:
            self._free_pages(physical_ids)
        if self.active_request_id == request_id:
            self.active_request_id = None

    def _reclaim_pending_free(self):
        if not self._pending_free_pages:
            return
        ready = []
        for event, physical_ids in self._pending_free_pages:
            if event.query():
                self._free_pages(physical_ids)
                ready.append((event, physical_ids))
        for item in ready:
            self._pending_free_pages.remove(item)

    def set_active_request(self, request_id):
        if request_id not in self.states:
            raise ValueError(f'unknown request {request_id}')
        self.active_request_id = request_id

    def record_query(self, layer_idx, request_ids, query_states):
        """Store one decode/prefill query row per request for next-step indexer.

        ``query_states`` may be [B, HQ, 1, D] or [B, HQ, D].  Rows are written
        into a fixed device buffer so ``GraphRunner.step`` can run the NFA
        selector before the next captured model replay without a host sync.
        """
        if query_states.ndim == 4:
            if query_states.shape[-2] != 1:
                raise ValueError('record_query expects one query row')
            query_states = query_states[:, :, 0]
        if query_states.ndim != 3 or query_states.shape[1] != self.num_query_heads:
            raise ValueError('record_query expects [B,HQ,D]')
        for batch_idx, request_id in enumerate(request_ids):
            if request_id is None or request_id < 0 or request_id not in self.states:
                continue
            if batch_idx >= query_states.shape[0]:
                break
            self._last_q_buf[layer_idx, request_id].copy_(
                query_states[batch_idx], non_blocking=True)

    def compute_selected_ids_by_layer(self, model, req_ids):
        """Run the per-layer NFA selector and return ``{layer: {req: ids}}``.

        This is the Top-K fine-selection step used by ``GraphRunner.step``.
        Query rows are read from the fixed ``_last_q_buf`` populated by prefill
        and by the previous decode step, so no per-step host tensor is built.
        """
        from ..indexer.tli_indexer import select_graph_ready_ids
        from ..ops.paged_region_kernels_nfa import _protected_selection_batch

        selected_by_layer = {}
        for layer_idx, module in enumerate(model.model.layers):
            indexer = module.self_attn.indexer
            scale = module.self_attn.scaling

            active_states = [
                self.states[rid] for rid in req_ids
                if rid >= 0 and rid in self.states
            ]
            if active_states and all(
                    state.skip_near_by_layer[layer_idx]
                    and state.skip_far_by_layer[layer_idx]
                    for state in active_states):
                ids, counts = _protected_selection_batch(
                    self, layer_idx, req_ids, indexer)
                self._store_gather_slots_batch(layer_idx, ids, counts, req_ids)
                for rid in req_ids:
                    if rid >= 0 and rid in self.states:
                        self.states[rid].first_decode_seeded = True
                selected_by_layer[layer_idx] = '__prepared__'
                continue

            valid_slots = [
                b for b, rid in enumerate(req_ids)
                if rid >= 0 and rid in self.states
            ]
            valid_req_ids = [req_ids[b] for b in valid_slots]
            lengths = [
                self.pending_lengths.get(rid, self.states[rid].logical_length)
                for rid in valid_req_ids
            ]
            q_flat = self._last_q_buf[layer_idx, valid_req_ids]
            length_t = torch.tensor(
                lengths, dtype=torch.long, device=q_flat.device)
            query_start_loc = torch.arange(
                len(valid_req_ids) + 1, dtype=torch.int32,
                device=q_flat.device)
            seq_idx = torch.arange(
                len(valid_req_ids), dtype=torch.int32,
                device=q_flat.device)
            if valid_req_ids:
                torch._assert_async(length_t > 0)
                torch._assert_async(query_start_loc[-1] == len(valid_req_ids))
                torch._assert_async(seq_idx[-1] == len(valid_req_ids) - 1)

            per_request = {}
            for row_idx, request_id in enumerate(valid_req_ids):
                batch_idx = valid_slots[row_idx]
                state = self.states[request_id]
                length = lengths[row_idx]
                q_row = q_flat[row_idx].reshape(
                    1, self.num_query_heads, 1, -1)
                ids, counts = select_graph_ready_ids(
                    self, layer_idx, request_id, indexer, q_row, scale, length)
                if not state.first_decode_seeded:
                    state.first_decode_seeded = True
                    seed_ids = state.prefill_topk_ids_by_layer.get(layer_idx, [])
                    if seed_ids:
                        seed = torch.as_tensor(seed_ids, dtype=ids.dtype,
                                               device=ids.device)
                        ids = torch.cat((ids, seed))
                        counts = counts + seed.numel()
                per_request[request_id] = (ids, counts)
            selected_by_layer[layer_idx] = per_request
        return selected_by_layer

    # ------------------------------------------------------------------
    # Prefill and freeze
    # ------------------------------------------------------------------
    def write_prefill(self, layer_idx, request_id, keys, values):
        state = self.states[request_id]
        layer = self.layers[layer_idx]
        page = self.geometry.page

        if keys.ndim != 4 or keys.shape[0] != 1 or keys.shape[-2] < 1:
            raise ValueError('graph-ready prefill expects unpadded B1 KV')
        if keys.shape[1] != layer.num_kv_heads:
            raise ValueError('KV head count does not match arena')
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError('KV dtype/device mismatch')

        n = keys.shape[-2]
        if not state.block_table:
            # 首个 layer 负责建立全模型共享的逻辑页->物理页映射。
            logical_pages = list(range(0, (n + page - 1) // page))
            physical_ids = self._alloc_pages(len(logical_pages))
            state.block_table = dict(zip(logical_pages, physical_ids))
            state.prompt_length = n
            state.logical_length = n
        elif n != state.prompt_length:
            raise ValueError('prefill length diverged across layers')

        self._write_block_table_row(request_id, state)

        # 逐页写入该层自己的物理 Arena。
        for logical_page, physical_id in state.block_table.items():
            lo = logical_page * page
            hi = min(lo + page, n)
            if hi <= lo:
                continue
            off = lo - logical_page * page
            layer.kv_pool_k[physical_id, 0, :, off:off + (hi - lo), :].copy_(
                keys[0, :, lo:hi, :])
            layer.kv_pool_v[physical_id, 0, :, off:off + (hi - lo), :].copy_(
                values[0, :, lo:hi, :])

        return keys, values

    def freeze(self, request_id, skip_far_list, skip_near_list,
               prefill_topk_by_layer=None):
        state = self.states[request_id]
        if len(skip_far_list) != self.n_layers or len(skip_near_list) != self.n_layers:
            raise ValueError('freeze expects per-layer skip decisions')
        if state.logical_length == 0:
            raise ValueError('prefill before freeze')

        state.skip_far_by_layer = [bool(x) for x in skip_far_list]
        state.skip_near_by_layer = [bool(x) for x in skip_near_list]
        state.skip_policy_groups = self._build_skip_policy_groups(state)
        if prefill_topk_by_layer is not None:
            state.prefill_topk_ids_by_layer = {
                layer_idx: [int(x) for x in prefill_topk_by_layer.get(layer_idx, [])]
                for layer_idx in range(self.n_layers)
            }

        # 物理木桶效应：按最贪心层预留 resident pages。
        per_layer_pages = [
            self._max_resident_pages_for_layer(state, layer_idx)
            for layer_idx in range(self.n_layers)
        ]
        state.max_resident_pages = max(per_layer_pages)

        # 释放当前非 resident 的 prefill 页，只保留当前 resident 页。
        resident_now_ranges = self._resident_page_ranges_for_length(
            state, state.logical_length)
        resident_now = set(self._pages_in_ranges(resident_now_ranges))
        stale = [
            logical_page for logical_page in state.block_table
            if logical_page not in resident_now
        ]
        if stale:
            self._free_pages([state.block_table.pop(lp) for lp in stale])

        # 补足请求专属 slot 池。decode 阶段只消费这个 deque，不碰全局池。
        deficit = state.max_resident_pages - len(state.block_table)
        if deficit < 0:
            raise RuntimeError('resident page accounting is inconsistent')
        if deficit:
            state.free_phys_slots.extend(self._alloc_pages(deficit))

        # Prefill 已把所有当前 resident 页写入 block_table；这里只记录
        # 增量 diff 的基线，后续 decode 每步仅比较区间边沿。
        state.resident_page_ranges = resident_now_ranges
        state.last_sync_length = state.logical_length
        self._write_block_table_row(request_id, state)
        state.frozen = True

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------
    def _intervals_for_layer(self, state, layer_idx, length):
        sink_hi, far_hi, near_hi = self.geometry.bounds(length)
        spans = [(0, sink_hi), (near_hi, length)]
        if not state.skip_far_by_layer[layer_idx]:
            spans.append((sink_hi, far_hi))
        if not state.skip_near_by_layer[layer_idx]:
            spans.append((far_hi, near_hi))
        return sorted((a, b) for a, b in spans if a < b)

    def _pages_in_intervals(self, intervals):
        page = self.geometry.page
        pages = set()
        for a, b in intervals:
            if a < b:
                pages.update(range(a // page, (b - 1) // page + 1))
        return pages

    @staticmethod
    def _merge_page_ranges(page_ranges):
        if not page_ranges:
            return ()
        page_ranges = sorted(page_ranges)
        merged = []
        cur_lo, cur_hi = page_ranges[0]
        for lo, hi in page_ranges[1:]:
            if lo <= cur_hi + 1:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append((cur_lo, cur_hi))
                cur_lo, cur_hi = lo, hi
        merged.append((cur_lo, cur_hi))
        return tuple(merged)

    def _build_skip_policy_groups(self, state):
        """Group layers by their frozen (skip_far, skip_near) policy.

        Decode metadata only depends on these two flags, so the per-step
        resident-page union can be computed once per unique policy instead of
        once per layer.
        """
        groups = []
        seen = set()
        for layer_idx in range(self.n_layers):
            key = (bool(state.skip_far_by_layer[layer_idx]),
                   bool(state.skip_near_by_layer[layer_idx]))
            if key in seen:
                continue
            seen.add(key)
            groups.append((key, layer_idx))
        return tuple(groups)

    def _resident_page_ranges_for_length(self, state, length):
        page = self.geometry.page
        page_ranges = []
        groups = state.skip_policy_groups
        if not groups:
            groups = self._build_skip_policy_groups(state)
        for _key, layer_idx in groups:
            for a, b in self._intervals_for_layer(state, layer_idx, length):
                if a < b:
                    page_ranges.append((a // page, (b - 1) // page))
        return self._merge_page_ranges(page_ranges)

    def _resident_page_union_for_length(self, state, length):
        # Kept as a compatibility helper.  New hot paths use the range form so
        # long resident intervals do not materialize Python set(range(...)).
        return set(self._pages_in_ranges(
            self._resident_page_ranges_for_length(state, length)))

    @staticmethod
    def _pages_in_ranges(ranges):
        pages = []
        for lo, hi in ranges:
            pages.extend(range(lo, hi + 1))
        return pages

    @staticmethod
    def _interval_difference(a_intervals, b_intervals):
        """Segments covered by ``a_intervals`` but not ``b_intervals``.

        Both inputs are sorted half-open token intervals.  Decode shifts the
        visible SWA/near/far boundaries by only a few edges per step, so this
        is O(len(a)+len(b)) and avoids rebuilding a full visibility mask.
        """
        out = []
        for a_lo, a_hi in a_intervals:
            cur = a_lo
            for b_lo, b_hi in b_intervals:
                if b_hi <= cur:
                    continue
                if b_lo >= a_hi:
                    break
                if b_lo > cur:
                    out.append((cur, min(b_lo, a_hi)))
                cur = max(cur, b_hi)
                if cur >= a_hi:
                    break
            if cur < a_hi:
                out.append((cur, a_hi))
        return out

    @staticmethod
    def _range_diff(a_ranges, b_ranges):
        """Pages present in ``a_ranges`` but absent from ``b_ranges``.

        Both inputs are sorted, disjoint inclusive page ranges.  A +1 token
        decode step normally changes only one or two page edges, so this runs
        in O(len(a)+len(b)) and only materializes those changed pages.
        """
        out = []
        j = 0
        n = len(b_ranges)
        for lo, hi in a_ranges:
            while j < n and b_ranges[j][1] < lo:
                j += 1
            cur = lo
            k = j
            while k < n and b_ranges[k][0] <= hi:
                if b_ranges[k][0] > cur:
                    out.extend(range(cur, min(hi, b_ranges[k][0] - 1) + 1))
                cur = max(cur, b_ranges[k][1] + 1)
                if cur > hi:
                    break
                k += 1
            if cur <= hi:
                out.extend(range(cur, hi + 1))
        return out

    def _max_resident_pages_for_layer(self, state, layer_idx):
        """O(1) upper bound for the resident-page pool.

        skip_far=False means the far region stays resident, so the worst case is
        the whole sequence through prompt_len + max_new_tokens.  skip_far=True
        bounds resident storage to sink + near + SWA.  The extra page is a
        partial-page safety margin for page-boundary growth.
        """
        page = self.geometry.page
        if not state.skip_far_by_layer[layer_idx]:
            resident_tokens = state.prompt_length + self.geometry.max_new_tokens
        else:
            resident_tokens = (
                self.geometry.sink
                + max(self.geometry.near, 0)
                + self.geometry.window
            )
            # sink/near/SWA are three disjoint intervals and can straddle up to
            # two extra page boundaries beyond the token-count page ceiling.
            return (resident_tokens + page - 1) // page + 2
        return (resident_tokens + page - 1) // page + 1

    def _write_block_table_row(self, request_id, state):
        """Refresh one CPU staging row after rare block-table mutations.

        This is deliberately not used on the decode hot path.  The previous
        ``torch.tensor(list(...)) + scatter_`` formulation is replaced by two
        scalar row updates because the row is only refreshed at prefill/freeze.
        """
        row = self._cpu_block_table_staging[request_id]
        pad = self.layers[0].PAD_PAGE_ID
        row.fill_(pad)
        for logical_page, physical_id in state.block_table.items():
            if 0 <= logical_page < self.max_blocks:
                row[logical_page] = physical_id

    def _sync_block_table(self, state, length, request_id=None):
        """Incrementally reconcile the logical->physical table for ``length``.

        ``state.resident_page_ranges`` always corresponds to the last synced
        length.  A decode step advances by one token, so only the interval
        edges differ; ``_range_diff`` materializes those few pages instead of
        rescanning the whole resident set.
        """
        new_ranges = self._resident_page_ranges_for_length(state, length)
        old_ranges = state.resident_page_ranges
        if old_ranges is None:
            old_ranges = ()

        missing = self._range_diff(new_ranges, old_ranges)
        stale = self._range_diff(old_ranges, new_ranges)

        if len(missing) > len(state.free_phys_slots):
            raise RuntimeError('resident slot pool exhausted')
        for logical_page in missing:
            state.block_table[logical_page] = state.free_phys_slots.popleft()
        for logical_page in stale:
            if logical_page in state.block_table:
                state.free_phys_slots.append(state.block_table.pop(logical_page))

        if request_id is not None and (missing or stale):
            row = self._cpu_block_table_staging[request_id]
            pad = self.layers[0].PAD_PAGE_ID
            for logical_page in missing:
                if 0 <= logical_page < self.max_blocks:
                    row[logical_page] = state.block_table[logical_page]
            for logical_page in stale:
                if 0 <= logical_page < self.max_blocks:
                    row[logical_page] = pad

        state.resident_page_ranges = new_ranges
        state.last_sync_length = length

    def _visible_logical_ids_for_layer(self, state, layer_idx, selected_ids=None,
                                       length=None):
        if length is None:
            length = state.logical_length
        intervals = self._intervals_for_layer(state, layer_idx, length)
        count = sum(b - a for a, b in intervals)
        ids = torch.empty((count,), dtype=torch.int32, device='cpu')
        offset = 0
        for a, b in intervals:
            n = b - a
            if n > 0:
                torch.arange(a, b, dtype=torch.int32,
                             out=ids[offset:offset + n])
                offset += n
        if selected_ids is not None:
            if not isinstance(selected_ids, torch.Tensor):
                selected_ids = torch.as_tensor(selected_ids, dtype=torch.int32,
                                               device='cpu')
            elif selected_ids.device != ids.device:
                selected_ids = selected_ids.to(ids.device)
            ids = ids[torch.isin(ids, selected_ids)]
        return ids

    def project_visible_for_request(self, layer_idx, request_id, dims,
                                    block_size, length=None):
        """Incremental low-dimensional visible-K projection.

        Returns ``[1, padded_length, num_kv_heads, dims.numel()]`` in the KV
        dtype.  The buffer is cached per request/layer/subspace and only the
        newly appended decode positions are gathered.  K/V are append-only, so
        old visible positions never need to be recomputed; positions that leave
        the visible intervals are masked by the selector and need not be zeroed
        in this projection.
        """
        state = self.states[request_id]
        layer = self.layers[layer_idx]
        if length is None:
            length = self.pending_lengths.get(request_id, state.logical_length)
        device = layer.kv_pool_k.device
        hk = layer.num_kv_heads
        page = self.geometry.page
        pad = ((length + block_size - 1) // block_size) * block_size
        dims_numel = dims.numel()
        key = (layer_idx, dims.data_ptr(), dims_numel, block_size)
        cached = state.projection_cache.get(key)

        if cached is None or cached['buffer'].dtype != layer.kv_pool_k.dtype                 or cached['buffer'].device != device                 or cached['buffer'].shape[1] < pad:
            out = torch.zeros((1, pad, hk, dims_numel),
                              dtype=layer.kv_pool_k.dtype, device=device)
            start_pos = 0
            if cached is not None and cached['buffer'].shape[1] < pad:
                old_rows = cached['buffer'].shape[1]
                out[0, :old_rows].copy_(cached['buffer'][0, :old_rows])
                start_pos = cached['last_length']
            cached = {
                'buffer': out,
                'last_length': start_pos,
                'dims_ptr': dims.data_ptr(),
                'dims_numel': dims_numel,
                'block_size': block_size,
            }
            state.projection_cache[key] = cached

        out = cached['buffer']
        start_pos = cached['last_length']
        if length <= start_pos:
            cached['last_length'] = length
            return out

        old_intervals = (
            self._intervals_for_layer(state, layer_idx, start_pos)
            if start_pos > 0 else ())
        new_intervals = self._intervals_for_layer(state, layer_idx, length)

        # Tokens that age out of the visible set must be zeroed, otherwise their
        # stale projected values would pollute block min/max/mean statistics
        # for blocks shared with still-visible tokens.
        for a, b in self._interval_difference(old_intervals, new_intervals):
            if a >= b:
                continue
            positions = torch.arange(a, b, device=device, dtype=torch.long)
            out[0, positions] = 0

        # Only fill tokens that are newly visible since the last cached length.
        for a, b in self._interval_difference(new_intervals, old_intervals):
            if a >= b:
                continue
            first_page = a // page
            last_page = (b - 1) // page
            for logical_page in range(first_page, last_page + 1):
                physical_id = state.block_table.get(logical_page)
                if physical_id is None:
                    continue
                lo = max(a, logical_page * page)
                hi = min(b, (logical_page + 1) * page)
                if hi <= lo:
                    continue
                positions = torch.arange(lo, hi, device=device, dtype=torch.long)
                offsets = positions - logical_page * page
                vals = layer.kv_pool_k[physical_id, 0, :, offsets, :]
                # vals: [HK, n, D] -> [n, HK, D]
                vals = vals.permute(1, 0, 2)[..., dims]
                out[0, positions] = vals

        cached['last_length'] = length
        return out

    def project_visible_varlen(self, layer_idx, req_ids, dims, block_size,
                               lengths=None):
        """Batch visible-K projection for the graph-ready NFA selector.

        Unlike :meth:`project_visible_for_request`, this computes every valid
        request's projection with one kernel launch and returns a unified
        ``[B, pad, HK, dims.numel()]`` tensor.  It is a full-visible projection
        intended for the varlen indexer path; the eager incremental path remains
        authoritative for correctness checks.
        """
        from ..ops.paged_region_kernels_nfa import project_visible_varlen_kernel

        if not req_ids:
            raise ValueError('project_visible_varlen requires at least one request')
        if any(rid < 0 or rid not in self.states for rid in req_ids):
            raise ValueError('project_visible_varlen accepts only active request ids')

        layer = self.layers[layer_idx]
        device = layer.kv_pool_k.device
        if device.type != 'cuda':
            raise RuntimeError('project_visible_varlen requires CUDA')

        if lengths is None:
            lengths = [
                self.pending_lengths.get(rid, self.states[rid].logical_length)
                for rid in req_ids
            ]
        if len(lengths) != len(req_ids):
            raise ValueError('lengths must align with req_ids')

        max_len = max(lengths) if lengths else 0
        pad = ((max_len + block_size - 1) // block_size) * block_size
        dims = dims.to(device=device).contiguous()

        lengths_t = torch.tensor(
            lengths, dtype=torch.int32, device=device)
        skip_far_t = torch.tensor(
            [self.states[rid].skip_far_by_layer[layer_idx] for rid in req_ids],
            dtype=torch.int32, device=device)
        skip_near_t = torch.tensor(
            [self.states[rid].skip_near_by_layer[layer_idx] for rid in req_ids],
            dtype=torch.int32, device=device)
        block_table = self.block_table_buf[req_ids].contiguous()

        return project_visible_varlen_kernel(
            layer.kv_pool_k, block_table, lengths_t,
            skip_far_t, skip_near_t, dims, pad,
            sink=self.geometry.sink,
            near_len=self.geometry.near,
            window=self.geometry.window,
        )

    # ------------------------------------------------------------------
    # Metadata preparation (common page table + per-layer sparse gather)
    # ------------------------------------------------------------------
    def _store_gather_slots(self, layer_idx, batch_idx, logical_ids, width,
                            *, request_id):
        """Convert logical token ids to absolute physical slots in-place.

        The attention kernel now consumes ``physical_page * PAGE + offset``
        directly, so it neither reads ``block_table`` nor performs integer
        division/modulo.  ``PAGE`` is power-of-two in production and uses the
        corresponding bit mask; the non-power-of-two path is retained for CPU
        unit tests.
        """
        if width <= 0:
            return
        ids = logical_ids[:width]
        if ids.device != self.gather_indices_buf.device:
            ids = ids.to(self.gather_indices_buf.device)
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()

        page_size = self.geometry.page
        is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
        if is_pow2:
            pages = ids >> (page_size.bit_length() - 1)
            offsets = ids & (page_size - 1)
        else:
            pages = torch.div(ids, page_size, rounding_mode='floor')
            offsets = torch.remainder(ids, page_size)

        safe_pages = torch.clamp(pages, min=0, max=self.max_blocks - 1)
        # Page-table rows are persistent request slots; output rows are batch slots.
        phys_pages = self.block_table_buf[request_id].gather(
            0, safe_pages.to(torch.int64))
        slots = phys_pages.to(torch.int32) * page_size + offsets.to(torch.int32)
        slots = torch.where(ids >= 0, slots,
                            torch.full_like(slots, INVALID_GATHER_IDX))
        self.gather_indices_buf[layer_idx, batch_idx, :width].copy_(
            slots, non_blocking=True)

    def _store_gather_slots_batch(self, layer_idx, logical_ids, counts, request_ids):
        """Batch version of ``_store_gather_slots`` for fixed-width varlen rows.

        ``logical_ids`` is ``[B, MAX_WIDTH]`` and ``counts`` is ``[B]``.  Rows
        beyond their count are already sentinel-padded by the caller; this
        helper converts all rows without a per-request Python loop.
        ``request_ids`` maps those batch rows to persistent page-table rows.
        """
        if logical_ids.numel() == 0 or counts.numel() == 0:
            return
        batch, width = logical_ids.shape
        if len(request_ids) != batch:
            raise ValueError('request_ids must align with logical_ids rows')
        ids = logical_ids
        if ids.device != self.gather_indices_buf.device:
            ids = ids.to(self.gather_indices_buf.device)
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()

        page_size = self.geometry.page
        is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
        if is_pow2:
            pages = ids >> (page_size.bit_length() - 1)
            offsets = ids & (page_size - 1)
        else:
            pages = torch.div(ids, page_size, rounding_mode='floor')
            offsets = torch.remainder(ids, page_size)

        safe_pages = torch.clamp(pages, min=0, max=self.max_blocks - 1)
        # Index only selected pages rather than materializing a reordered [B,M]
        # page table. Padding rows use row 0 safely and are masked below.
        request_rows = torch.tensor(
            request_ids, dtype=torch.int64, device=ids.device).unsqueeze(1)
        phys_pages = self.block_table_buf[
            request_rows.clamp(min=0), safe_pages.to(torch.int64)]
        slots = phys_pages.to(torch.int32) * page_size + offsets.to(torch.int32)
        slots = torch.where(
            (ids >= 0) & (request_rows >= 0), slots,
            torch.full_like(slots, INVALID_GATHER_IDX))
        self.gather_indices_buf[layer_idx, :batch, :width].copy_(
            slots, non_blocking=True)
        self.visible_lens_buf[layer_idx, :batch].copy_(
            counts.to(torch.int32), non_blocking=True)

    def new_step(self, active_req_ids):
        self._reclaim_pending_free()
        self.active_req_ids = list(active_req_ids)
        self.common_dirty = True

    def plan_common(self, active_req_ids, query_lengths=None):
        """Prepare common page/slot/query metadata without layer gather.

        This gives ``compute_selected_ids_by_layer`` the planned decode lengths
        before Top-K selection, without an extra all-layer gather pass.
        """
        if query_lengths is None:
            query_lengths = [
                1 if (rid >= 0 and rid in self.states) else 0
                for rid in active_req_ids
            ]
        self.active_req_ids = list(active_req_ids)
        if self.common_dirty:
            self._prepare_common(active_req_ids, query_lengths)

    def _prepare_common(self, req_ids, query_lengths):
        page = self.geometry.page
        batch = len(req_ids)
        pad_page_id = self.layers[0].PAD_PAGE_ID

        # 只清当前 step 会用到的 slot 前缀，避免每步 fill 整个长 buffer。
        self._cpu_slot_mapping_staging[:batch].fill_(PAD_SLOT)
        self.pending_lengths.clear()

        cumsum = 0
        flat = 0
        for b, request_id in enumerate(req_ids):
            self._cpu_query_start_loc_staging[b] = cumsum
            qlen = int(query_lengths[b]) if b < len(query_lengths) else 0
            if request_id < 0 or request_id not in self.states:
                cumsum += qlen
                continue

            state = self.states[request_id]
            planned_length = state.logical_length + qlen

            # 只预分配/更新 block_table，不推进 state.logical_length。
            # 所有层共享同一 planned_length，避免层间状态分叉。
            # ``_sync_block_table`` 现在只 diff 上一个 committed length 的
            # resident 区间，不再每步重建 Python set/list 或 scatter。
            self._sync_block_table(state, planned_length, request_id=request_id)
            self.pending_lengths[request_id] = planned_length

            for t in range(qlen):
                pos = state.logical_length + t
                logical_page, offset = divmod(pos, page)
                physical_id = state.block_table.get(logical_page)
                if physical_id is None:
                    raise RuntimeError(
                        f'missing logical page {logical_page} for request {request_id}')
                self._cpu_slot_mapping_staging[flat] = physical_id * page + offset
                self._cpu_query_to_req_staging[flat] = b
                flat += 1

            cumsum += qlen

        self._cpu_query_start_loc_staging[batch] = cumsum
        self.block_table_buf.copy_(self._cpu_block_table_staging, non_blocking=True)
        self.slot_mapping_buf.copy_(self._cpu_slot_mapping_staging, non_blocking=True)
        self.query_start_loc_buf.copy_(self._cpu_query_start_loc_staging, non_blocking=True)
        self.query_to_req_buf.copy_(self._cpu_query_to_req_staging, non_blocking=True)
        self.common_dirty = False

    def _prepare_layer_gather(self, layer_idx, req_ids, selected_ids_by_req=None):
        self.visible_lens_buf[layer_idx].zero_()
        # Padding columns are masked by visible_lens in the kernel, so the
        # entire MAX_VISIBLE tail does not need to be rewritten every step.

        for batch_idx, request_id in enumerate(req_ids):
            if request_id < 0 or request_id not in self.states:
                continue
            state = self.states[request_id]
            length = self.pending_lengths.get(request_id, state.logical_length)
            intervals = self._intervals_for_layer(state, layer_idx, length)
            total = sum(b - a for a, b in intervals)

            selected = None
            if selected_ids_by_req is not None:
                selected = selected_ids_by_req.get(request_id)

            if selected is None:
                if total > self.max_visible_len:
                    raise RuntimeError(
                        f'request {request_id} layer {layer_idx} visible_len '
                        f'{total} exceeds fixed capacity {self.max_visible_len}')
                offset = 0
                for a, interval_end in intervals:
                    n = interval_end - a
                    if n > 0:
                        torch.arange(a, interval_end, dtype=torch.int32,
                                     out=self._gather_logical_tmp[offset:offset + n])
                        offset += n
                self._store_gather_slots(layer_idx, batch_idx,
                                         self._gather_logical_tmp, total,
                                         request_id=request_id)
                self.visible_lens_buf[layer_idx, batch_idx] = total
            else:
                if isinstance(selected, tuple):
                    selected, selected_count = selected
                    selected_count = selected_count.reshape(())
                else:
                    selected_count = None
                    if not isinstance(selected, torch.Tensor):
                        selected = torch.as_tensor(selected, dtype=torch.int32,
                                                   device=self.gather_indices_buf.device)
                if selected.device != self.gather_indices_buf.device:
                    selected = selected.to(self.gather_indices_buf.device)
                if selected.ndim != 1:
                    selected = selected.reshape(-1)
                width = min(selected.numel(), self.max_visible_len)
                if selected_count is not None:
                    torch._assert_async(selected_count <= self.max_visible_len)
                    self.visible_lens_buf[layer_idx, batch_idx].copy_(
                        selected_count, non_blocking=True)
                else:
                    if selected.numel() > self.max_visible_len:
                        raise RuntimeError(
                            f'request {request_id} layer {layer_idx} visible_len '
                            f'{selected.numel()} exceeds fixed capacity '
                            f'{self.max_visible_len}')
                    self.visible_lens_buf[layer_idx, batch_idx] = width
                if width:
                    self._store_gather_slots(
                        layer_idx, batch_idx, selected, width,
                        request_id=request_id)

    def prepare_metadata(self, active_req_ids, current_layer_id,
                         query_lengths=None, selected_ids_by_req=None):
        if current_layer_id < 0 or current_layer_id >= self.n_layers:
            raise ValueError('invalid layer id')
        if self.common_dirty:
            if query_lengths is None:
                query_lengths = [
                    1 if (rid >= 0 and rid in self.states) else 0
                    for rid in active_req_ids
                ]
            self._prepare_common(active_req_ids, query_lengths)
        self._prepare_layer_gather(
            current_layer_id, active_req_ids, selected_ids_by_req)
        return (
            self.gather_indices_buf[current_layer_id],
            self.visible_lens_buf[current_layer_id],
        )

    def prepare_all_layers(self, active_req_ids, query_lengths=None,
                           selected_ids_by_layer=None, visible_bucket=None):
        """Prepare common metadata once, then every layer's gather buffers.

        If ``visible_bucket`` is supplied, the selected bucket's contiguous
        scratch tensors are filled as well and ``active_visible_bucket`` is set
        so the attention patch reads the graph-captured width.
        """
        if query_lengths is None:
            query_lengths = [
                1 if (rid >= 0 and rid in self.states) else 0
                for rid in active_req_ids
            ]
        self.active_req_ids = list(active_req_ids)
        if self.common_dirty:
            self._prepare_common(active_req_ids, query_lengths)
        for layer_idx in range(self.n_layers):
            selected = None
            if selected_ids_by_layer is not None:
                selected = selected_ids_by_layer.get(layer_idx)
            if selected == '__prepared__':
                continue
            self._prepare_layer_gather(layer_idx, active_req_ids, selected)

        if visible_bucket is not None:
            visible_bucket = int(visible_bucket)
            if visible_bucket != self.max_visible_len:
                self.ensure_visible_buckets((visible_bucket,))
                dst_gather = self._gather_indices_by_vis[visible_bucket]
                dst_lens = self._visible_lens_by_vis[visible_bucket]
                for layer_idx in range(self.n_layers):
                    dst_gather[layer_idx].copy_(
                        self.gather_indices_buf[layer_idx, :, :visible_bucket],
                        non_blocking=True)
                    dst_lens[layer_idx].copy_(
                        self.visible_lens_buf[layer_idx], non_blocking=True)
            self.active_visible_bucket = visible_bucket

    def commit_step(self):
        """Advance request lengths exactly once after all layers finish.

        ``_prepare_common`` only planned the next tokens; committing here keeps
        every layer synchronized and keeps state mutation outside the captured
        model graph.
        """
        for request_id, planned_length in self.pending_lengths.items():
            state = self.states[request_id]
            state.logical_length = planned_length
            self._sync_block_table(state, state.logical_length)
        self.pending_lengths.clear()


    # ------------------------------------------------------------------
    # Decode write
    # ------------------------------------------------------------------
    def write_decode(self, layer_idx, key_states, value_states, slot_mapping):
        layer = self.layers[layer_idx]
        if key_states.ndim != 4 or key_states.shape[-2] != 1:
            raise ValueError('graph-ready decode KV must be one token per row')
        if key_states.shape[0] != slot_mapping.shape[0]:
            raise ValueError('request batch must match KV batch')
        from ..ops.paged_region_kernels import paged_write_decode
        return paged_write_decode(layer, key_states, value_states, slot_mapping)

    def block_table(self):
        return self.block_table_buf

    def slot_mapping(self):
        return self.slot_mapping_buf

    def query_start_loc(self):
        return self.query_start_loc_buf

    def get_max_cache_shape(self):
        return -1

    def get_seq_length(self, layer_idx=0):
        # 公共长度由 scheduler/state 维护；此接口仅兼容 HF attention 前置校验。
        if self.states:
            return max(state.logical_length for state in self.states.values())
        return 0

    def get_mask_sizes(self, cache_position, layer_idx=0):
        """HF ``create_causal_mask`` compatibility for the fixed arena cache."""
        if isinstance(cache_position, int):
            query_len = cache_position
        else:
            query_len = cache_position.shape[0]
        max_len = max((state.logical_length for state in self.states.values()),
                      default=0)
        return max_len + query_len, 0

    def report(self):
        """JSON-safe summary for frozen-region receipts."""
        return {
            "requests": len(self.states),
            "active_requests": list(self.states),
            "n_layers": self.n_layers,
            "layers": [
                {
                    "num_kv_heads": layer.num_kv_heads,
                    "head_dim_k": layer.head_dim_k,
                    "head_dim_v": layer.head_dim_v,
                }
                for layer in self.layers
            ],
        }

