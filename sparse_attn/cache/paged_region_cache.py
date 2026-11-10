"""GraphReady 固定地址分页 KV Cache 与连续批请求生命周期。"""
import heapq
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

    ``block_tables[layer]`` maps logical page id -> physical page id.  Tables
    are append-only for the request lifetime: near/far skip decisions change
    only the selector-visible intervals, never physical page membership.
    """
    logical_length: int = 0
    prompt_length: int = 0
    phase: RequestPhase = RequestPhase.PREFILL
    block_tables: list = field(default_factory=list)
    skip_far_by_layer: list = field(default_factory=list)
    skip_near_by_layer: list = field(default_factory=list)
    frozen: bool = False
    active: bool = True
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
        self.num_kv_heads = num_kv_heads
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
        self.common_dirty = True
        self._common_active_req_ids = ()
        self._common_state_refs = ()
        self._common_layout_revision = -1
        # request/state 布局只在 step 边界做 canonical 比较；各层 selector
        # 复用该单调版本，不再重复构造 tuple 或逐元素比较 request state。
        self.request_layout_revision = 0
        self._layout_req_ids = ()
        self._layout_state_refs = ()
        self._layout_source = None
        self._visible_bucket_cache = {}
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

        # 层间与 KV group 间均允许不同选集；query head 消费所属 KV group 的行。
        self.gather_indices_buf = torch.full(
            (n_layers, max_requests, num_kv_heads, max_visible_len),
            INVALID_GATHER_IDX,
            dtype=torch.int32,
            device=device,
        )
        self.visible_lens_buf = torch.zeros(
            (n_layers, max_requests, num_kv_heads),
            dtype=torch.int32,
            device=device,
        )

        # CPU staging：capture 后每步只 copy_，不在热路径 torch.tensor()。
        self._cpu_gather_staging = torch.full(
            (max_requests, num_kv_heads, max_visible_len),
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
            self._cpu_skip_far_layer_staging = self._cpu_skip_far_layer_staging.pin_memory()
            self._cpu_skip_near_layer_staging = self._cpu_skip_near_layer_staging.pin_memory()
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
                (self.n_layers, self.max_requests, self.num_kv_heads, width),
                INVALID_GATHER_IDX,
                dtype=torch.int32,
                device=self.gather_indices_buf.device,
            )
            self._visible_lens_by_vis[width] = torch.zeros(
                (self.n_layers, self.max_requests, self.num_kv_heads),
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
        """按策略缓存 coarse 上界；连续 decode 仅在余量不足时重算。"""
        states = []
        lengths = []
        policies = []
        for request_id in req_ids:
            if request_id is None or request_id < 0 or request_id not in self.states:
                states.append(None)
                lengths.append(-1)
                policies.append(None)
                continue
            state = self.states[request_id]
            states.append(state)
            lengths.append(self.pending_lengths.get(
                request_id, state.logical_length))
            groups = (state.skip_policy_groups
                      or self._build_skip_policy_groups(state))
            policies.append(tuple(key for key, _layer_idx in groups))

        cache_key = (tuple(policies), len(req_ids), tuple(sorted(visible_buckets)))
        cached = self._visible_bucket_cache.get(cache_key)
        state_refs = tuple(states)
        current_lengths = tuple(lengths)
        if cached is not None:
            old_refs, old_lengths, old_max, old_width = cached
            if old_refs == state_refs and all(
                    new >= old for new, old in zip(current_lengths, old_lengths)):
                max_delta = max(
                    (new - old for new, old in zip(current_lengths, old_lengths)),
                    default=0)
                # 每个请求的 coarse 区间总长度随 logical length 的增长量
                # 不超过长度增量。
                if old_max + max_delta <= old_width:
                    return old_width

        max_total = 0
        for state, length in zip(states, lengths):
            if state is None:
                continue
            groups = (state.skip_policy_groups
                      or self._build_skip_policy_groups(state))
            for _key, layer_idx in groups:
                intervals = self._intervals_for_layer(state, layer_idx, length)
                total = sum(b - a for a, b in intervals)
                max_total = max(max_total, total)
        width = self.max_visible_len
        for candidate in sorted(visible_buckets):
            if candidate >= max_total:
                width = candidate
                break
        self._visible_bucket_cache[cache_key] = (
            state_refs, current_lengths, max_total, width)
        return width

    # ------------------------------------------------------------------
    # Physical page allocation (independent per layer)
    # ------------------------------------------------------------------
    def _alloc_pages(self, layer_idx, count):
        return self.layers[layer_idx].alloc_pages(count)

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
        # 请求存续期间不归还任何页。删除时先等待所有流上已提交的 KV 读写，
        # 再一次性归还各层页，保证后续请求复用不会发生异步 UAF。
        physical_ids_by_layer = [
            set(state.block_tables[layer_idx].values())
            for layer_idx in range(self.n_layers)
        ]
        self._cpu_block_table_staging[:, request_id].fill_(
            self.layers[0].PAD_PAGE_ID)
        if self.layers[0].kv_pool_k.is_cuda:
            torch.cuda.synchronize(self.layers[0].kv_pool_k.device)
        for layer_idx, physical_ids in enumerate(physical_ids_by_layer):
            layer = self.layers[layer_idx]
            for physical_id in sorted(physical_ids):
                if physical_id == layer.PAD_PAGE_ID:
                    raise RuntimeError('PAD page leaked into free_ids')
                heapq.heappush(layer.free_ids, physical_id)
        if self.active_request_id == request_id:
            self.active_request_id = None

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
        required_pages = (n + page - 1) // page
        if required_pages > self.max_blocks:
            raise ValueError('prefill length exceeds fixed block-table capacity')
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

        # 只刷新并同步当前层/request 行；其余层在各自 prefill 到达时更新。
        row = self._cpu_block_table_staging[layer_idx, request_id]
        row.fill_(self.layers[0].PAD_PAGE_ID)
        for logical_page, physical_id in block_table.items():
            if 0 <= logical_page < self.max_blocks:
                row[logical_page] = physical_id
        self.block_table_buf[layer_idx, request_id].copy_(
            row, non_blocking=keys.is_cuda)

        # 整段 KV 单次批量 scatter，消除逐逻辑页 Python 循环和小拷贝。
        from ..ops.paged_region_kernels import paged_write_prefill
        workspace = getattr(self, '_nfa_selector_workspace', None)
        return paged_write_prefill(
            layer, keys, values, self.block_table_buf[layer_idx, request_id],
            workspace=workspace)

    def freeze(self, request_id, skip_far_list, skip_near_list):
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
        self._nfa_stats_dirty = True
        # 所有 skip 策略共用 prompt + max_new_tokens 的物理容量上限；实际页
        # 已在 prefill 分配，后续 decode 仅在跨入新逻辑页时追加分配。
        page = self.geometry.page
        capacity_pages = (
            state.prompt_length + self.geometry.max_new_tokens + page - 1) // page
        if capacity_pages > self.max_blocks:
            raise ValueError(
                'prompt + max_new_tokens exceeds fixed block-table capacity')

        self._write_block_table_row(request_id, state)
        self.block_table_buf[:, request_id].copy_(
            self._cpu_block_table_staging[:, request_id],
            non_blocking=non_blocking)
        self.common_dirty = True
        # freeze 可改变 both-skip 计划；即使 request/state 引用未变，也必须
        # 使按 layout revision 缓存的 selector eager plan 失效。
        self.request_layout_revision += 1
        self._layout_source = None
        state.frozen = True
        self.transition_request(request_id, RequestPhase.FREEZE)

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

    def _build_skip_policy_groups(self, state):
        """Group layers by their frozen (skip_far, skip_near) policy.

        Decode visible metadata only depends on these two flags, so the coarse
        interval bound can be computed once per unique policy instead of once
        per layer.
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
        """Append physical pages for newly reached logical pages only."""
        if length < state.logical_length:
            raise ValueError('paged block tables cannot move backwards')
        capacity = state.prompt_length + self.geometry.max_new_tokens
        if length > capacity:
            raise RuntimeError(
                f'planned length {length} exceeds request capacity {capacity}')
        page = self.geometry.page
        required_pages = (length + page - 1) // page
        if required_pages > self.max_blocks:
            raise RuntimeError('planned length exceeds fixed block-table capacity')

        for layer_idx in range(self.n_layers):
            block_table = state.block_tables[layer_idx]
            first_missing = len(block_table)
            if list(block_table) != list(range(first_missing)):
                raise RuntimeError('physical block table is not append-only')
            missing = range(first_missing, required_pages)
            physical_ids = self._alloc_pages(layer_idx, len(missing))
            for logical_page, physical_id in zip(missing, physical_ids):
                block_table[logical_page] = physical_id
                if request_id is not None:
                    self._cpu_block_table_staging[
                        layer_idx, request_id, logical_page] = physical_id

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
        self.gather_indices_buf[layer_idx, batch_idx, :, :width].copy_(
            slots.unsqueeze(0), non_blocking=True)

    def _store_gather_slots_batch(self, layer_idx, logical_ids, counts):
        """Batch version of ``_store_gather_slots`` for fixed-width varlen rows.

        ``logical_ids`` is ``[B, MAX_WIDTH]`` and ``counts`` is ``[B]``.  Rows
        beyond their count are already sentinel-padded by the caller; this
        helper converts all rows to absolute physical slots in one GPU pass.
        """
        if logical_ids.numel() == 0 or counts.numel() == 0:
            return
        if logical_ids.ndim == 2:
            logical_ids = logical_ids.unsqueeze(1)
        if logical_ids.ndim != 3 or logical_ids.shape[1] != self.num_kv_heads:
            raise ValueError('logical gather ids must be [B,W] or [B,Hkv,W]')
        batch, _, width = logical_ids.shape
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
        phys_pages = self.block_table_buf[
            layer_idx, :batch, None, :].expand(-1, ids.shape[1], -1).gather(
                2, safe_pages.to(torch.int64))
        slots = phys_pages.to(torch.int32) * page_size + offsets.to(torch.int32)
        slots = torch.where(
            ids >= 0, slots,
            torch.full_like(slots, INVALID_GATHER_IDX))
        self.gather_indices_buf[layer_idx, :batch, :, :width].copy_(
            slots, non_blocking=True)
        if counts.ndim == 1:
            counts = counts.unsqueeze(1)
        self.visible_lens_buf[layer_idx, :batch].copy_(
            counts.to(torch.int32), non_blocking=True)

    def new_step(self, active_req_ids):
        req_ids = tuple(active_req_ids)
        state_refs = tuple(
            self.states.get(rid) if rid is not None and rid >= 0 else None
            for rid in req_ids)
        active_changed = (
            req_ids != self._layout_req_ids
            or len(state_refs) != len(self._layout_state_refs)
            or any(new is not old for new, old in zip(
                state_refs, self._layout_state_refs)))
        if active_changed:
            self.request_layout_revision += 1
            self._layout_req_ids = req_ids
            self._layout_state_refs = state_refs
        self._layout_source = active_req_ids
        page = self.geometry.page
        needs_full = active_changed
        for state in state_refs:
            if state is None:
                continue
            # 下一 token 跨入新逻辑页时才需要重建设备页表；普通连续 decode
            # 只推进固定 slot mapping 中的页内 offset。
            needs_full = needs_full or state.logical_length % page == 0
        self.active_req_ids = list(req_ids)
        self.common_dirty = self.common_dirty or needs_full

    def plan_common(self, active_req_ids, query_lengths=None):
        """准备公共元数据；稳态单 token decode 只推进计划长度。"""
        # Scheduler/GraphRunner 已在同一步调用 new_step 时，同一 list 对象
        # 直接复用其 layout revision；独立调用方仍走完整 canonical 检测。
        if active_req_ids is not self._layout_source:
            self.new_step(active_req_ids)
        if query_lengths is None:
            query_lengths = [
                1 if (rid >= 0 and rid in self.states) else 0
                for rid in active_req_ids
            ]
        self.active_req_ids = list(active_req_ids)
        if self.common_dirty:
            self._prepare_common(active_req_ids, query_lengths)
        elif not self._prepare_decode_common_fast(active_req_ids, query_lengths):
            self._prepare_common(active_req_ids, query_lengths)

    def _prepare_decode_common_fast(self, req_ids, query_lengths):
        """复用未变化的页表/query 映射，并推进计划长度和页内写入位置。"""
        if self.request_layout_revision != self._common_layout_revision:
            return False
        req_ids = self._layout_req_ids
        state_refs = self._layout_state_refs
        batch = len(req_ids)
        planned = []
        for row, (request_id, state) in enumerate(zip(req_ids, state_refs)):
            qlen = int(query_lengths[row]) if row < len(query_lengths) else 0
            expected = 1 if state is not None else 0
            if qlen != expected:
                return False
            if state is None:
                planned.append(0)
                continue
            planned_length = state.logical_length + qlen
            if (qlen and state.logical_length // self.geometry.page
                    != (planned_length - 1) // self.geometry.page):
                return False
            capacity = state.prompt_length + self.geometry.max_new_tokens
            if planned_length > capacity:
                return False
            planned.append(planned_length)

        # 上一步与当前 token 位于同一物理页；有效 decode slot 的页内 offset
        # 随 token 位置加一，PAD_SLOT 保持不变。
        slots = self.slot_mapping_buf[:, :batch]
        slots.add_(slots >= 0)
        self.pending_lengths.clear()
        for row, (request_id, state) in enumerate(zip(req_ids, state_refs)):
            self._cpu_planned_lengths_staging[row] = planned[row]
            if state is not None:
                self.pending_lengths[request_id] = planned[row]
        self.planned_lengths_buf[:batch].copy_(
            self._cpu_planned_lengths_staging[:batch], non_blocking=True)
        return True

    def _prepare_common(self, req_ids, query_lengths):
        page = self.geometry.page
        batch = len(req_ids)

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

            # 只追加本步新到达的逻辑页，不推进 state.logical_length。
            self._sync_block_table(state, planned_length, request_id=request_id)
            self.pending_lengths[request_id] = planned_length
            # Existing rows are initialized by prefill/freeze; _sync_block_table
            # writes each appended page into staging. Do not rebuild the prefix.
            self.block_table_buf[:, request_id].copy_(
                self._cpu_block_table_staging[:, request_id],
                non_blocking=True)

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
        self._common_active_req_ids = tuple(req_ids)
        self._common_state_refs = tuple(
            self.states.get(rid) if rid is not None and rid >= 0 else None
            for rid in req_ids)
        self._common_layout_revision = self.request_layout_revision
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
                        self.gather_indices_buf[layer_idx, :, :, :visible_bucket],
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
            self.states[request_id].logical_length = planned_length
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
            "layers": [
                {
                    "num_kv_heads": layer.num_kv_heads,
                    "head_dim_k": layer.head_dim_k,
                    "head_dim_v": layer.head_dim_v,
                }
                for layer in self.layers
            ],
        }

