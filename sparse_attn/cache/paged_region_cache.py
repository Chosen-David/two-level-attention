"""GraphReady 固定地址分页 KV Cache 与连续批请求生命周期。"""
import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import torch

INVALID_PAGE_IDX = -1
INVALID_GATHER_IDX = -1
PAD_SLOT = -1

SINK = 0b0001
FAR = 0b0010
NEAR = 0b0100
SWA = 0b1000
REGION_BITS = (SINK, FAR, NEAR, SWA)


class RequestPhase(str, Enum):
    """冻结请求在统一 Runner 中的显式生命周期。"""
    PREFILL = 'PREFILL'
    FREEZE = 'FREEZE'
    SUFFIX = 'SUFFIX'
    GENERATE = 'GENERATE'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


_REQUEST_PHASE_TRANSITIONS = {
    RequestPhase.PREFILL: {RequestPhase.FREEZE, RequestPhase.FAILED},
    RequestPhase.FREEZE: {
        RequestPhase.SUFFIX, RequestPhase.GENERATE,
        RequestPhase.FINISHED, RequestPhase.FAILED},
    RequestPhase.SUFFIX: {
        RequestPhase.GENERATE, RequestPhase.FINISHED, RequestPhase.FAILED},
    RequestPhase.GENERATE: {RequestPhase.FINISHED, RequestPhase.FAILED},
    RequestPhase.FINISHED: set(),
    RequestPhase.FAILED: set(),
}


@dataclass
class RequestState:
    """One continuous-batching request slot.

    ``block_tables[layer]`` maps logical page id -> physical page id.  Every
    layer owns its page ids independently, so a keep layer cannot pin pages in
    a layer whose near/far regions were skipped.

    ``free_phys_slots_by_layer`` contains each layer's request-local resident
    pages that are not currently bound to a logical page.  Decode never touches
    the global ``free_ids``.
    """
    logical_length: int = 0
    prompt_length: int = 0
    phase: RequestPhase = RequestPhase.PREFILL
    max_resident_pages_by_layer: list = field(default_factory=list)
    block_tables: list = field(default_factory=list)
    free_phys_slots_by_layer: list = field(default_factory=list)
    skip_far_by_layer: list = field(default_factory=list)
    skip_near_by_layer: list = field(default_factory=list)
    prefill_topk_ids_by_layer: dict = field(default_factory=dict)
    first_decode_seeded: bool = False
    frozen: bool = False
    active: bool = True
    # Per-layer decode-incremental resident-page cache.  A +1 token step only
    # diffs interval edges for the affected layers.
    resident_page_ranges_by_layer: list = field(default_factory=list)
    last_sync_length: int = 0
    skip_policy_groups: tuple = ()


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
        max_prefill_seed_len=128,
    ):
        device = torch.device(device)
        self.geometry = geometry
        self.n_layers = n_layers
        self.max_requests = max_requests
        self.max_blocks = max_blocks
        self.max_visible_len = max_visible_len
        self.max_query_tokens = max_query_tokens
        self.max_prefill_seed_len = max_prefill_seed_len
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
        # Arena 地址生命周期与 cache 相同；selector capture 时只读该持久
        # pointer buffer，禁止从 Python data_ptr list 做 H2D 拷贝。
        self.layer_kv_ptrs_buf = torch.tensor(
            [layer.kv_pool_k.data_ptr() for layer in self.layers],
            dtype=torch.int64, device=device)

        self.states: dict[int, RequestState] = {}
        # 终态在物理请求槽释放后仍保留，供请求边界生成审计回执。
        self.request_phases: dict[int, RequestPhase] = {}
        self.pending_lengths: dict[int, int] = {}
        self.active_request_id = None
        self.active_req_ids = []
        self._pending_free_pages = []
        self._reclaimed_pages_by_layer = [0] * n_layers
        self.common_dirty = True
        # 仅由 GraphRunner 在完整 decode step 后递增；请求结束回执据此
        # fail-closed，避免把只完成 prefill 的 Cache 误认为已跑过 NFA decode。
        self.nfa_decode_steps = 0

        # 每层独立持有 logical->physical 映射和写入 slot；固定地址可被 Graph
        # capture，物理页号不再要求跨层对齐。
        self.block_table_buf = torch.full(
            (n_layers, max_requests, max_blocks),
            self.layers[0].PAD_PAGE_ID,
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_buf = torch.full(
            (n_layers, max_query_tokens),
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
        # selector 的 Graph 元数据按 batch 行存储，不能把 request id 当作行号。
        # 每步由 plan_common 原地刷新，capture/replay 只读固定设备地址。
        self.active_req_ids_buf = torch.full(
            (max_query_tokens,), self.INVALID_REQ_ID,
            dtype=torch.int64, device=device)
        # selector 每层只读：PAD 行映射到安全的 request 0，Q gather 使用固定行号。
        self.valid_req_ids_buf = torch.zeros(
            (max_query_tokens,), dtype=torch.int64, device=device)
        self.query_req_ids_buf = torch.arange(
            max_query_tokens, dtype=torch.int64, device=device)
        self.planned_lengths_buf = torch.zeros(
            (max_query_tokens,), dtype=torch.int32, device=device)

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
            (n_layers, max_requests, max_blocks),
            self.layers[0].PAD_PAGE_ID,
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_slot_mapping_staging = torch.full(
            (n_layers, max_query_tokens),
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
        self._cpu_active_req_ids_staging = torch.full(
            (max_query_tokens,), self.INVALID_REQ_ID,
            dtype=torch.int64, device='cpu')
        self._cpu_valid_req_ids_staging = torch.zeros(
            (max_query_tokens,), dtype=torch.int64, device='cpu')
        self._cpu_planned_lengths_staging = torch.zeros(
            (max_query_tokens,), dtype=torch.int32, device='cpu')
        self._cpu_skip_far_staging = torch.zeros(
            (max_requests,), dtype=torch.int32, device='cpu')
        self._cpu_skip_near_staging = torch.zeros(
            (max_requests,), dtype=torch.int32, device='cpu')
        self.skip_far_buf = torch.zeros(
            (max_requests,), dtype=torch.int32, device=device)
        self.skip_near_buf = torch.zeros(
            (max_requests,), dtype=torch.int32, device=device)
        self.skip_far_layer_buf = torch.zeros(
            (n_layers, max_requests), dtype=torch.int32, device=device)
        self.skip_near_layer_buf = torch.zeros(
            (n_layers, max_requests), dtype=torch.int32, device=device)
        self._cpu_skip_far_layer_staging = torch.zeros(
            (n_layers, max_requests), dtype=torch.int32, device='cpu')
        self._cpu_skip_near_layer_staging = torch.zeros(
            (n_layers, max_requests), dtype=torch.int32, device='cpu')
        if device.type == 'cuda':
            self._cpu_block_table_staging = self._cpu_block_table_staging.pin_memory()
            self._cpu_slot_mapping_staging = self._cpu_slot_mapping_staging.pin_memory()
            self._cpu_query_start_loc_staging = self._cpu_query_start_loc_staging.pin_memory()
            self._cpu_query_to_req_staging = self._cpu_query_to_req_staging.pin_memory()
            self._cpu_active_req_ids_staging = self._cpu_active_req_ids_staging.pin_memory()
            self._cpu_planned_lengths_staging = self._cpu_planned_lengths_staging.pin_memory()
            self._cpu_skip_far_staging = self._cpu_skip_far_staging.pin_memory()
            self._cpu_skip_near_staging = self._cpu_skip_near_staging.pin_memory()
            self._cpu_skip_far_layer_staging = self._cpu_skip_far_layer_staging.pin_memory()
            self._cpu_skip_near_layer_staging = self._cpu_skip_near_layer_staging.pin_memory()
        # 固定 prefill seed buffer：freeze 时一次性写入，decode 热路径不再
        # 遍历 Python lists。seed 在 Top-K 完成后与 selection 做 canonical union。
        self.prefill_seed_ids_buf = torch.full(
            (n_layers, max_requests, max_prefill_seed_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.prefill_seed_counts_buf = torch.zeros(
            (n_layers, max_requests),
            dtype=torch.int32,
            device=device,
        )
        # 每个请求是否已完成首个 decode 步的 seed 合并。0=未合并（下次选择需
        # 并入 prefill seed），1=已合并。整批所有层成功后统一置 1。
        self.first_decode_seeded_buf = torch.zeros(
            (max_requests,),
            dtype=torch.int32,
            device=device,
        )
        self._cpu_first_decode_seeded_staging = torch.zeros(
            (max_requests,),
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_prefill_seed_ids_staging = torch.full(
            (n_layers, max_requests, max_prefill_seed_len),
            -1,
            dtype=torch.int32,
            device='cpu',
        )
        self._cpu_prefill_seed_counts_staging = torch.zeros(
            (n_layers, max_requests),
            dtype=torch.int32,
            device='cpu',
        )
        if device.type == 'cuda':
            self._cpu_prefill_seed_ids_staging = self._cpu_prefill_seed_ids_staging.pin_memory()
            self._cpu_prefill_seed_counts_staging = self._cpu_prefill_seed_counts_staging.pin_memory()
            self._cpu_first_decode_seeded_staging = self._cpu_first_decode_seeded_staging.pin_memory()

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

    def slot_mapping_for_layer(self, layer_idx):
        return self.slot_mapping_buf[layer_idx]

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
    # Physical page allocation (independent per layer)
    # ------------------------------------------------------------------
    def _alloc_pages(self, layer_idx, count):
        self._reclaim_pending_free()
        if count == 0:
            return []
        layer = self.layers[layer_idx]
        if len(layer.free_ids) < count:
            raise RuntimeError('graph-ready physical page pool exhausted')
        return [heapq.heappop(layer.free_ids) for _ in range(count)]

    def _free_pages(self, layer_idx, physical_ids):
        if physical_ids:
            self.layers[layer_idx].free_pages(list(physical_ids))

    def _batch_reclaim(self, request_id):
        """Reclaim each layer's non-resident pages without cross-layer pinning."""
        state = self.states[request_id]
        if state.logical_length <= 0:
            return [[] for _ in range(self.n_layers)]
        reclaimed = []
        for layer_idx in range(self.n_layers):
            resident_ranges = self._resident_page_ranges_for_layer(
                state, layer_idx, state.logical_length)
            resident = set(self._pages_in_ranges(resident_ranges))
            block_table = state.block_tables[layer_idx]
            stale = [lp for lp in block_table if lp not in resident]
            physical_ids = [block_table.pop(lp) for lp in stale]
            self._free_pages(layer_idx, physical_ids)
            self._reclaimed_pages_by_layer[layer_idx] += len(physical_ids)
            reclaimed.append(physical_ids)
        return reclaimed

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
        self.states[request_id] = RequestState(
            block_tables=[{} for _ in range(self.n_layers)],
            free_phys_slots_by_layer=[deque() for _ in range(self.n_layers)],
            resident_page_ranges_by_layer=[() for _ in range(self.n_layers)],
        )
        self.request_phases[request_id] = RequestPhase.PREFILL
        self._cpu_block_table_staging[:, request_id].fill_(
            self.layers[0].PAD_PAGE_ID)
        return request_id

    def transition_request(self, request_id, phase):
        """校验并记录请求阶段；槽位复用时由 add_request 重新初始化。"""
        phase = RequestPhase(phase)
        current = self.request_phases.get(request_id)
        if current is None:
            raise ValueError(f'unknown request phase slot {request_id}')
        if current == phase:
            return
        if phase not in _REQUEST_PHASE_TRANSITIONS[current]:
            raise ValueError(
                f'invalid request phase transition {current.value} -> {phase.value}')
        self.request_phases[request_id] = phase
        state = self.states.get(request_id)
        if state is not None:
            state.phase = phase

    def _next_free_slot(self):
        for request_id in range(self.max_requests):
            if request_id not in self.states:
                return request_id
        raise RuntimeError('no free request slot')

    def remove_request(self, request_id):
        state = self.states.pop(request_id, None)
        if state is None:
            return
        # 各层分别归还已绑定和预留的物理页；页号在层间无需一致。
        physical_ids_by_layer = []
        for layer_idx in range(self.n_layers):
            physical_ids = set(state.block_tables[layer_idx].values())
            physical_ids.update(state.free_phys_slots_by_layer[layer_idx])
            physical_ids_by_layer.append(physical_ids)
        self._cpu_block_table_staging[:, request_id].fill_(
            self.layers[0].PAD_PAGE_ID)
        if self.layers[0].kv_pool_k.is_cuda:
            event = torch.cuda.Event()
            event.record()
            self._pending_free_pages.append((event, physical_ids_by_layer))
        else:
            for layer_idx, physical_ids in enumerate(physical_ids_by_layer):
                self._free_pages(layer_idx, physical_ids)
        if self.active_request_id == request_id:
            self.active_request_id = None

    def _reclaim_pending_free(self):
        if not self._pending_free_pages:
            return
        ready = []
        for event, physical_ids_by_layer in self._pending_free_pages:
            if event.query():
                for layer_idx, physical_ids in enumerate(physical_ids_by_layer):
                    self._free_pages(layer_idx, physical_ids)
                ready.append((event, physical_ids_by_layer))
        for item in ready:
            self._pending_free_pages.remove(item)

    def set_active_request(self, request_id):
        if request_id not in self.states:
            raise ValueError(f'unknown request {request_id}')
        self.active_request_id = request_id

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
        block_table = state.block_tables[layer_idx]
        if not block_table:
            logical_pages = list(range(0, (n + page - 1) // page))
            physical_ids = self._alloc_pages(layer_idx, len(logical_pages))
            state.block_tables[layer_idx] = dict(zip(logical_pages, physical_ids))
            block_table = state.block_tables[layer_idx]
            state.prompt_length = n
            state.logical_length = n
        elif n != state.prompt_length:
            raise ValueError('prefill length diverged across layers')

        self._write_block_table_row(request_id, state)

        # 逐页写入该层自己的物理 Arena。
        for logical_page, physical_id in block_table.items():
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
        self._cpu_skip_far_layer_staging[:, request_id] = torch.as_tensor(
            state.skip_far_by_layer, dtype=torch.int32)
        self._cpu_skip_near_layer_staging[:, request_id] = torch.as_tensor(
            state.skip_near_by_layer, dtype=torch.int32)
        non_blocking = self.layers[0].kv_pool_k.is_cuda
        self.skip_far_layer_buf[:, request_id].copy_(
            self._cpu_skip_far_layer_staging[:, request_id],
            non_blocking=non_blocking)
        self.skip_near_layer_buf[:, request_id].copy_(
            self._cpu_skip_near_layer_staging[:, request_id],
            non_blocking=non_blocking)
        # request slot 复用时强制下一步全量重建 NFA 持久页统计。
        if hasattr(self, '_nfa_page_stats_lengths'):
            self._nfa_page_stats_lengths[:, request_id].zero_()
        if prefill_topk_by_layer is not None:
            state.prefill_topk_ids_by_layer = {
                layer_idx: [int(x) for x in prefill_topk_by_layer.get(layer_idx, [])]
                for layer_idx in range(self.n_layers)
            }
            # 一次性把 Python seed lists 固化到 GPU buffer，decode 热路径零遍历。
            for layer_idx in range(self.n_layers):
                seed_ids = state.prefill_topk_ids_by_layer[layer_idx]
                if len(seed_ids) > self.max_prefill_seed_len:
                    raise ValueError(
                        f'prefill seed length {len(seed_ids)} exceeds capacity '
                        f'{self.max_prefill_seed_len} at layer {layer_idx}')
                self._cpu_prefill_seed_counts_staging[layer_idx, request_id] = len(seed_ids)
                if seed_ids:
                    self._cpu_prefill_seed_ids_staging[layer_idx, request_id, :len(seed_ids)] = \
                        torch.as_tensor(seed_ids, dtype=torch.int32)
                    self._cpu_prefill_seed_ids_staging[layer_idx, request_id, len(seed_ids):] = -1
                else:
                    self._cpu_prefill_seed_ids_staging[layer_idx, request_id, :] = -1
            self.prefill_seed_ids_buf[:, request_id].copy_(
                self._cpu_prefill_seed_ids_staging[:, request_id],
                non_blocking=non_blocking)
            self.prefill_seed_counts_buf[:, request_id].copy_(
                self._cpu_prefill_seed_counts_staging[:, request_id],
                non_blocking=non_blocking)

        # 首步 selector 会把 prefill seed 并入支持集；对应物理页必须在 freeze
        # 回收和首步 plan 中暂时保持 resident，消费后再由 commit_step 释放。
        state.first_decode_seeded = False
        state.max_resident_pages_by_layer = [
            self._max_resident_pages_for_layer(state, layer_idx)
            for layer_idx in range(self.n_layers)
        ]

        resident_now_ranges = [
            self._resident_page_ranges_for_layer(
                state, layer_idx, state.logical_length)
            for layer_idx in range(self.n_layers)
        ]
        # 每层只回收自己的非 resident prefill 页。
        self._batch_reclaim(request_id)

        # 各层独立补足请求专属 slot 池，decode 不触碰全局池。
        for layer_idx in range(self.n_layers):
            deficit = (state.max_resident_pages_by_layer[layer_idx]
                       - len(state.block_tables[layer_idx]))
            if deficit < 0:
                raise RuntimeError('resident page accounting is inconsistent')
            if deficit:
                state.free_phys_slots_by_layer[layer_idx].extend(
                    self._alloc_pages(layer_idx, deficit))

        # 记录每层增量 diff 基线，后续 decode 仅比较区间边沿。
        state.resident_page_ranges_by_layer = resident_now_ranges
        state.last_sync_length = state.logical_length
        self._write_block_table_row(request_id, state)
        state.frozen = True
        self.transition_request(request_id, RequestPhase.FREEZE)
        state.first_decode_seeded = False
        self._cpu_first_decode_seeded_staging[request_id] = 0
        self.first_decode_seeded_buf[request_id].copy_(
            self._cpu_first_decode_seeded_staging[request_id],
            non_blocking=non_blocking)

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

    def _resident_page_ranges_for_layer(self, state, layer_idx, length):
        page = self.geometry.page
        page_ranges = [
            (a // page, (b - 1) // page)
            for a, b in self._intervals_for_layer(state, layer_idx, length)
            if a < b
        ]
        if not state.first_decode_seeded:
            page_ranges.extend(
                (token_id // page, token_id // page)
                for token_id in state.prefill_topk_ids_by_layer.get(layer_idx, ())
                if 0 <= token_id < length
            )
        return self._merge_page_ranges(page_ranges)

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
            max_pages = (resident_tokens + page - 1) // page + 1
        else:
            resident_tokens = (
                self.geometry.sink
                + max(self.geometry.near, 0)
                + self.geometry.window
            )
            # sink/near/SWA are three disjoint intervals and can straddle up to
            # two extra page boundaries beyond the token-count page ceiling.
            max_pages = (resident_tokens + page - 1) // page + 2
        if not state.first_decode_seeded:
            # 上界允许与常规 resident 页重复计数；首步提交时会归还这部分余量。
            # Seed IDs are absolute positions, not offsets in the compact pool.
            # Match the residency filter even when far history is skipped.
            seed_pages = {
                token_id // page
                for token_id in state.prefill_topk_ids_by_layer.get(layer_idx, ())
                if 0 <= token_id < state.logical_length
            }
            max_pages += len(seed_pages)
        return max_pages

    def _write_block_table_row(self, request_id, state):
        """Refresh all layers' CPU staging rows after block-table mutations."""
        pad = self.layers[0].PAD_PAGE_ID
        for layer_idx in range(self.n_layers):
            row = self._cpu_block_table_staging[layer_idx, request_id]
            row.fill_(pad)
            for logical_page, physical_id in state.block_tables[layer_idx].items():
                if 0 <= logical_page < self.max_blocks:
                    row[logical_page] = physical_id

    def _sync_block_table(self, state, length, request_id=None):
        """Incrementally reconcile every layer's logical->physical table."""
        if len(state.resident_page_ranges_by_layer) != self.n_layers:
            state.resident_page_ranges_by_layer = [
                () for _ in range(self.n_layers)]
        new_ranges_by_layer = []
        pad = self.layers[0].PAD_PAGE_ID
        for layer_idx in range(self.n_layers):
            new_ranges = self._resident_page_ranges_for_layer(
                state, layer_idx, length)
            old_ranges = state.resident_page_ranges_by_layer[layer_idx] or ()
            missing = self._range_diff(new_ranges, old_ranges)
            stale = self._range_diff(old_ranges, new_ranges)
            block_table = state.block_tables[layer_idx]
            free_slots = state.free_phys_slots_by_layer[layer_idx]

            if len(missing) > len(free_slots):
                raise RuntimeError('resident slot pool exhausted')
            for logical_page in missing:
                block_table[logical_page] = free_slots.popleft()
            for logical_page in stale:
                if logical_page in block_table:
                    free_slots.append(block_table.pop(logical_page))

            if request_id is not None and (missing or stale):
                row = self._cpu_block_table_staging[layer_idx, request_id]
                for logical_page in missing:
                    if 0 <= logical_page < self.max_blocks:
                        row[logical_page] = block_table[logical_page]
                for logical_page in stale:
                    if 0 <= logical_page < self.max_blocks:
                        row[logical_page] = pad
            new_ranges_by_layer.append(new_ranges)

        state.resident_page_ranges_by_layer = new_ranges_by_layer
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

    # ------------------------------------------------------------------
    # Metadata preparation (common page table + per-layer sparse gather)
    # ------------------------------------------------------------------
    def _store_gather_slots(self, layer_idx, batch_idx, logical_ids, width):
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
        phys_pages = self.block_table_buf[layer_idx, batch_idx].gather(
            0, safe_pages.to(torch.int64))
        slots = phys_pages.to(torch.int32) * page_size + offsets.to(torch.int32)
        slots = torch.where(ids >= 0, slots,
                            torch.full_like(slots, INVALID_GATHER_IDX))
        self.gather_indices_buf[layer_idx, batch_idx, :width].copy_(
            slots, non_blocking=True)

    def _store_gather_slots_batch(self, layer_idx, logical_ids, counts):
        """Batch version of ``_store_gather_slots`` for fixed-width varlen rows.

        ``logical_ids`` is ``[B, MAX_WIDTH]`` and ``counts`` is ``[B]``.  Rows
        beyond their count are already sentinel-padded by the caller; this
        helper converts all rows to absolute physical slots in one GPU pass.
        """
        if logical_ids.numel() == 0 or counts.numel() == 0:
            return
        batch, width = logical_ids.shape
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
        phys_pages = self.block_table_buf[layer_idx, :batch].gather(
            1, safe_pages.to(torch.int64))
        slots = phys_pages.to(torch.int32) * page_size + offsets.to(torch.int32)
        slots = torch.where(
            ids >= 0, slots,
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

        This gives the request-batched selector the planned decode lengths
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
        self._cpu_slot_mapping_staging[:, :batch].fill_(PAD_SLOT)
        self._cpu_active_req_ids_staging[:batch].fill_(self.INVALID_REQ_ID)
        self._cpu_valid_req_ids_staging[:batch].zero_()
        self._cpu_planned_lengths_staging[:batch].zero_()
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
            self._cpu_active_req_ids_staging[b] = request_id
            self._cpu_valid_req_ids_staging[b] = request_id
            self._cpu_planned_lengths_staging[b] = planned_length

            # 只预分配/更新各层 block_table，不推进 state.logical_length。
            self._sync_block_table(state, planned_length, request_id=request_id)
            self.pending_lengths[request_id] = planned_length
            self.block_table_buf[:, request_id].copy_(
                self._cpu_block_table_staging[:, request_id], non_blocking=True)

            for t in range(qlen):
                pos = state.logical_length + t
                logical_page, offset = divmod(pos, page)
                for layer_idx in range(self.n_layers):
                    physical_id = state.block_tables[layer_idx].get(logical_page)
                    if physical_id is None:
                        raise RuntimeError(
                            f'missing logical page {logical_page} for request '
                            f'{request_id} layer {layer_idx}')
                    self._cpu_slot_mapping_staging[layer_idx, flat] = (
                        physical_id * page + offset)
                self._cpu_query_to_req_staging[flat] = b
                flat += 1

            cumsum += qlen

        self._cpu_query_start_loc_staging[batch] = cumsum
        self.slot_mapping_buf[:, :flat].copy_(
            self._cpu_slot_mapping_staging[:, :flat], non_blocking=True)
        self.query_start_loc_buf.copy_(self._cpu_query_start_loc_staging, non_blocking=True)
        self.query_to_req_buf.copy_(self._cpu_query_to_req_staging, non_blocking=True)
        self.active_req_ids_buf.copy_(
            self._cpu_active_req_ids_staging, non_blocking=True)
        self.valid_req_ids_buf.copy_(
            self._cpu_valid_req_ids_staging, non_blocking=True)
        self.planned_lengths_buf.copy_(
            self._cpu_planned_lengths_staging, non_blocking=True)
        self.common_dirty = False

    def _pack_skip_flags(self, layer_idx, req_ids):
        """Copy the current layer's per-request skip flags to pinned/GPU buffers.

        No Python branch is used downstream; the Triton kernels read these two
        fixed int32 buffers as tensors.  During CUDA Graph capture the staging
        copy is the same persistent buffer, so no host-device synchronization is
        introduced.
        """
        batch = len(req_ids)
        self._cpu_skip_far_staging[:batch].zero_()
        self._cpu_skip_near_staging[:batch].zero_()
        for batch_idx, request_id in enumerate(req_ids):
            if request_id < 0 or request_id not in self.states:
                continue
            state = self.states[request_id]
            self._cpu_skip_far_staging[batch_idx] = int(
                state.skip_far_by_layer[layer_idx])
            self._cpu_skip_near_staging[batch_idx] = int(
                state.skip_near_by_layer[layer_idx])
        non_blocking = self.layers[0].kv_pool_k.is_cuda
        self.skip_far_buf[:batch].copy_(
            self._cpu_skip_far_staging[:batch], non_blocking=non_blocking)
        self.skip_near_buf[:batch].copy_(
            self._cpu_skip_near_staging[:batch], non_blocking=non_blocking)

    def _prepare_layer_gather(self, layer_idx, req_ids, selected_ids_by_req=None):
        self._pack_skip_flags(layer_idx, req_ids)
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
                                         self._gather_logical_tmp, total)
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
                        layer_idx, batch_idx, selected, width)

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
        release_by_layer = [[] for _ in range(self.n_layers)]
        for request_id, planned_length in self.pending_lengths.items():
            state = self.states[request_id]
            state.logical_length = planned_length
            self._sync_block_table(state, state.logical_length)
            if state.first_decode_seeded:
                # seed-only 页已被 _sync_block_table 移入请求本地空闲池。首步
                # Graph 可能仍在异步读取，归还全局池前必须用 event 建立时序。
                for layer_idx in range(self.n_layers):
                    steady_max = self._max_resident_pages_for_layer(
                        state, layer_idx)
                    previous_max = state.max_resident_pages_by_layer[layer_idx]
                    release_count = max(0, previous_max-steady_max)
                    free_slots = state.free_phys_slots_by_layer[layer_idx]
                    if release_count > len(free_slots):
                        raise RuntimeError('seed page release accounting is inconsistent')
                    for _ in range(release_count):
                        release_by_layer[layer_idx].append(free_slots.pop())
                    state.max_resident_pages_by_layer[layer_idx] = steady_max
        self.pending_lengths.clear()
        if any(release_by_layer):
            if self.layers[0].kv_pool_k.is_cuda:
                event = torch.cuda.Event()
                event.record()
                self._pending_free_pages.append((event, release_by_layer))
            else:
                for layer_idx, physical_ids in enumerate(release_by_layer):
                    self._free_pages(layer_idx, physical_ids)


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
        workspace = getattr(self, '_nfa_selector_workspace', None)
        return paged_write_decode(
            layer, key_states, value_states, slot_mapping,
            workspace=workspace)

    def block_table(self):
        return self.block_table_buf

    def slot_mapping(self):
        return self.slot_mapping_buf

    def query_start_loc(self):
        return self.query_start_loc_buf

    def get_max_cache_shape(self):
        return -1

    def get_seq_length(self, layer_idx=0):
        # 顺序 prefill 复用同一 Cache 时，HF 用该值构造 cache_position。
        # 必须返回当前请求的长度，否则后续请求会从其他请求的长度开始做
        # RoPE；decode 由 GraphRunner 显式传 position_ids，不依赖该回退值。
        if self.active_request_id is not None:
            return self.states[self.active_request_id].logical_length
        if self.states:
            return max(state.logical_length for state in self.states.values())
        return 0

    def get_mask_sizes(self, cache_position, layer_idx=0):
        """HF ``create_causal_mask`` compatibility for the fixed arena cache."""
        if isinstance(cache_position, int):
            query_len = cache_position
        else:
            query_len = cache_position.shape[0]
        if self.active_request_id is not None:
            kv_len = self.states[self.active_request_id].logical_length
        else:
            kv_len = max((state.logical_length for state in self.states.values()),
                         default=0)
        return kv_len + query_len, 0

    def report(self):
        """JSON-safe summary for frozen-region receipts."""
        return {
            "requests": len(self.states),
            "active_requests": list(self.states),
            "n_layers": self.n_layers,
            "reclaimed_pages_by_layer": list(self._reclaimed_pages_by_layer),
            "free_physical_pages_by_layer": [
                len(layer.free_ids) for layer in self.layers],
            "layers": [
                {
                    "num_kv_heads": layer.num_kv_heads,
                    "head_dim_k": layer.head_dim_k,
                    "head_dim_v": layer.head_dim_v,
                }
                for layer in self.layers
            ],
        }
