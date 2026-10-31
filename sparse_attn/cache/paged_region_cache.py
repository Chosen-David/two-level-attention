"""Paged KV cache: global physical page pool plus per-request region metadata.

Each logical page owns independently allocated K/V tensors. Reclaiming a page
releases its tensor references to the Torch allocator, and recycles its page ID.
Allocator-reserved memory is not reported as freed driver memory. The legacy
B1 request path is preserved; batch requests share one physical page pool per
layer and keep independent block tables / region state.
"""
import heapq
import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from ..ops.region_geometry import RegionGeometry


SINK = 0b0001
FAR = 0b0010
NEAR = 0b0100
SWA = 0b1000
REGION_BITS = (SINK, FAR, NEAR, SWA)


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
        self.tli_receipt = None
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
        self.tli_receipts = []
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
            self.tli_receipts.append(None)

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
        self.tli_receipts[request_id] = None
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
        self.tli_receipts[request_id] = None
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
    tli_regions=True
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
            if getattr(self, 'tli_regions', False):
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
