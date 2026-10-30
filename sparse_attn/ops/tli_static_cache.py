"""【阶段 2a】TLI 固定容量 KV cache（有效在前 left-pad 布局，decode-only）。

背景：pred_ruler_batch 的 TLI batch decode（`_tl_decode_batched`）当前消费
DynamicCache 变长 KV（有效占尾、形状随步增长）——CUDA Graph 化需要
**固定形状**的 KV 输入（T=S_cap 固定、cu=[0,S_cap] 烘焙、mask 全宽）。

本模块提供固定容量缓冲：层 buffer[b, :, 0:L_b, :] 为行 b 的有效 KV，
[L_b, S_cap) 为未写 pad 区（保持零）。decode 每步新 token 写到
pos = ctx.seq_lens[b] - 1（pred_ruler_batch 在 forward 前已将 seq_lens
含本步 +1）。配合 indexer.prepare_mask(..., valid_length=L_b) 与
eager_decoding_attn（cu=[0, S_cap]、mask [1,1,Hkv,S_cap]）→ 固定形状
decode（CUDA Graph 前置）。

prefill 不直接在本 cache 上做：pred_ruler_batch 沿用 DynamicCache
（left-pad 输入整行进 cache、有效占尾），prefill 后由
fill_static_from_dynamic() 逐层把每行有效列搬进本 cache 的 [0, L_b)
（一次性，每 chunk 一次）——避免与 prefill 的 2D bool padding mask
（有效占尾语义）冲突。

仅支持 decode（q_len==1）；prefill（q_len>1）显式拒绝（fail-closed）。
"""
from __future__ import annotations

import os

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from ..baseline_runtime import batch_decode_ctx


class TLIStaticLayer(CacheLayerMixin):
    """固定容量 KV 层。buffer [B, Hkv, S_cap, D]，有效在前。

    对齐 StaticLayer 的 CacheLayerMixin 契约（is_compileable / max_batch_size
    / max_cache_len / get_seq_length / get_mask_sizes / get_max_cache_shape
    / lazy_initialization / update / reset / crop / reorder_cache）。
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, max_batch_size: int):
        super().__init__()
        if max_cache_len < 1 or max_batch_size < 1:
            raise ValueError("TLIStaticLayer 需要 max_cache_len>=1 且 max_batch_size>=1")
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(max_batch_size)
        self.keys = None
        self.values = None
        self.num_heads = None
        self.head_dim = None
        # 单调递增累计写长（供 get_seq_length；batch decode 下 cache_position
        # 不影响正确性——RoPE 用显式 position_ids、create_causal_mask 4D 早退、
        # TLI patch 全宽分支不切片）。
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states=None):
        """按首层首个 decode 步（或搬移时）的 [B, Hkv, S, D] 形状预分配全宽 buffer。"""
        if key_states.ndim != 4:
            raise ValueError(f"TLIStaticLayer 期望 4D key_states，got {key_states.ndim}D")
        if value_states is not None and (
                value_states.shape != key_states.shape or value_states.dtype != key_states.dtype
                or value_states.device != key_states.device):
            raise ValueError("static KV initialization requires matching key/value tensors")
        self.num_heads = key_states.shape[1]
        self.head_dim = key_states.shape[-1]
        self.dtype, self.device = key_states.dtype, key_states.device
        if key_states.shape[0] != self.max_batch_size:
            raise ValueError(
                f"TLIStaticLayer batch {key_states.shape[0]} != 构造 batch "
                f"{self.max_batch_size}")
        shape = (self.max_batch_size, self.num_heads, self.max_cache_len,
                 self.head_dim)
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        """decode-only：新 token [B, Hkv, 1, D] → buffer[b, :, seq_lens[b]-1]。

        alive 行写新位置；dead 行跳过（不覆盖其已冻结的有效 KV）。
        返回全宽 buffer [B, Hkv, S_cap, D]（固定形状）。
        """
        if key_states.ndim != 4 or key_states.shape[-2] != 1:
            raise RuntimeError(
                "TLIStaticCache 仅支持 decode（q_len==1）；prefill 须走 "
                "DynamicCache + fill_static_from_dynamic 搬移。当前 "
                f"key_states shape={tuple(key_states.shape)}")
        if self.keys is None:
            self.lazy_initialization(key_states)
        ctx = batch_decode_ctx()
        if ctx is None:
            raise RuntimeError(
                "TLIStaticCache.update 需要 batch decode ctx（batch_decode_ctx "
                "为 None）；本 cache 仅供 pred_ruler_batch batch 模式")
        if ctx.batch != self.max_batch_size:
            raise RuntimeError(
                f"TLIStaticCache ctx.batch={ctx.batch} != 构造 batch "
                f"{self.max_batch_size}")
        # 【阶段 2c CUDA Graph 前置】KV 写入位置张量化：scatter_ 单 kernel
        # 沿 dim2（S_cap 维）每行写 pos[b] 一列。pos 为 tensor 索引 →
        # replay 时 pos_buf 更新 → 写新列（CUDA Graph 兼容）。
        # dead 行 seq_lens 冻结（不递增）；用 alive 张量保留旧值。
        # 非 graph 模式（pos_buf=None）回退 Python
        # 构造，语义与旧逐行 copy_ 逐位一致（纯内存拷贝，无算术）。
        from .cuda_graph_state import graph_state  # noqa: E402
        _gs = graph_state()
        if _gs.enabled and _gs.pos_buf is not None:
            pos = _gs.pos_buf
        else:
            pos = torch.tensor(
                [ctx.seq_lens[b] - 1 for b in range(self.max_batch_size)],
                dtype=torch.long, device=self.keys.device)
        pos = torch.clamp(pos, 0, self.max_cache_len - 1)
        # scatter_ 沿 dim2（S_cap 维）每行写 pos[b] 一列：index 与 src
        # 同 shape [B,Hkv,1,D]、元素值为 pos[b] → self[b,h,pos[b],d] =
        # src[b,h,0,d]（graph 兼容）。dead 行保留被冻结的最后一列。
        idx = pos.view(-1, 1, 1, 1).expand(
            self.max_batch_size, self.num_heads, 1, self.head_dim)
        if _gs.enabled:
            alive = (_gs.alive_buf if _gs.alive_buf is not None else
                     torch.ones(self.max_batch_size, dtype=torch.bool, device=self.keys.device))
        else:
            alive = torch.tensor(ctx.alive, dtype=torch.bool, device=self.keys.device)
        from .operator_pipeline import operator_mode
        if operator_mode() != 'off':
            from .kv_row_update import append_kv_batch
            append_kv_batch(self.keys, self.values, key_states, value_states, pos, alive)
        else:
            active = alive.view(-1, 1, 1, 1)
            self.keys.scatter_(2, idx, torch.where(active, key_states, self.keys.gather(2, idx)))
            self.values.scatter_(2, idx, torch.where(active, value_states, self.values.gather(2, idx)))
        self.cumulative_length += 1
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, cache_position):
        if getattr(self, "_direct_prefill_mask", False):
            # One-shot direct prefill has no past or padding. Let HF use SDPA's
            # is_causal path instead of materializing a quadratic static mask.
            # HF 4.x passed cache positions; HF 5.x passes query length.
            return (cache_position if isinstance(cache_position, int)
                    else cache_position.shape[0]), 0
        return self.max_cache_len, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self):
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
        self.cumulative_length = 0

    def crop(self, max_length):
        raise RuntimeError("TLIStaticLayer 不支持 crop（固定容量 decode-only）")

    def reorder_cache(self, beam_idx):
        raise RuntimeError("TLIStaticLayer 不支持 reorder（greedy decode-only）")


class _DirectPrefillLayer:
    def __init__(self, linear, config=None, mode="LINEAR_PROVISIONAL", linear_target=None):
        if mode not in ("RING_FINAL", "LINEAR_FINAL", "LINEAR_PROVISIONAL"):
            raise ValueError(f"unknown direct prefill storage plan: {mode}")
        self.linear = linear
        self.config = config
        self.mode = mode
        self.linear_target = linear_target
        self.ring = None
        self.written = False


class DirectPrefillStaging:
    """One-shot B=1 dense-prefill writer; never used for steady decode."""

    def __init__(self, cache, length, states):
        self.cache = cache
        self.length = length
        self.states = states

    def update(self, layer_idx, key_states, value_states, cache_kwargs):
        if layer_idx < 0 or layer_idx >= len(self.states):
            raise ValueError("direct prefill layer index out of range")
        state = self.states[layer_idx]
        L = self.length
        position = (cache_kwargs or {}).get("cache_position")
        # New HF attention no longer forwards cache_position. This staging
        # capability permits exactly one full B1 prefill of the prepared host
        # length, so its storage positions are fixed independently of RoPE IDs.
        # Explicit positions still undergo the exact validation below.
        if position is None:
            position = torch.arange(L, device=key_states.device)
        if (state.written or key_states.ndim != 4 or value_states.shape != key_states.shape
                or key_states.shape[0] != 1 or key_states.shape[-2] != L
                or key_states.requires_grad or value_states.requires_grad
                or position is None or position.shape != (L,)
                or position.dtype != torch.long or position.device != key_states.device
                or not torch.equal(position, torch.arange(L, device=position.device))):
            raise ValueError("direct prefill requires one detached, unpadded B=1 one-shot update")
        if state.config is not None and key_states.shape[-1] != 128:
            raise ValueError("direct compact prefill requires head_dim=128")
        linear = state.linear
        if state.mode == "RING_FINAL":
            from .far_skip_cache import FarSkipLayer
            state.ring = FarSkipLayer.from_prefill(key_states, value_states, state.config)
        else:
            target = state.linear_target
            if target is None:
                linear.lazy_initialization(key_states)
            else:
                batch_layer, row = target
                if (type(batch_layer) is not TLIStaticLayer or type(row) is not int
                        or not 0 <= row < batch_layer.max_batch_size
                        or batch_layer.max_cache_len != linear.max_cache_len):
                    raise ValueError("direct prefill received an invalid native batch-row target")
                if batch_layer.keys is None:
                    batch_layer.lazy_initialization(key_states[..., :1, :].expand(
                        batch_layer.max_batch_size, -1, -1, -1))
                expected = (batch_layer.max_batch_size, key_states.shape[1],
                            linear.max_cache_len, key_states.shape[-1])
                if (batch_layer.keys.shape != expected
                        or batch_layer.values.shape != expected
                        or batch_layer.keys.dtype != key_states.dtype
                        or batch_layer.values.dtype != value_states.dtype
                        or batch_layer.keys.device != key_states.device
                        or batch_layer.values.device != value_states.device):
                    raise ValueError("direct prefill native batch-row target changed shape or dtype")
                linear.keys = batch_layer.keys[row:row+1]
                linear.values = batch_layer.values[row:row+1]
                linear.num_heads, linear.head_dim = key_states.shape[1], key_states.shape[-1]
                linear.dtype, linear.device = key_states.dtype, key_states.device
                linear.is_initialized = True
            linear.keys[..., :L, :].copy_(key_states)
            linear.values[..., :L, :].copy_(value_states)
        # HF 5.x may omit cache_position. Report the completed logical prefill
        # length so the attention bridge does not slice its returned KV to zero.
        linear.cumulative_length = L
        state.written = True
        # Dense prefill must consume the original full-history tensors, not a
        # static/ring view. Returning the incoming objects also preserves layout.
        return key_states, value_states


class TLIStaticCache(Cache):
    """decode 用固定容量 KV cache（每层一个 TLIStaticLayer 全宽缓冲）。"""

    def __init__(self, config, max_cache_len: int, batch_size: int):
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        layers = [TLIStaticLayer(max_cache_len, batch_size) for _ in layer_types]
        super().__init__(layers=layers)
        self.batch_size = int(batch_size)
        # max_cache_len 由 Cache 基类只读 property 从 layers 推导
        # （max(layer.max_cache_len)），get_max_cache_shape 走
        # layers[layer_idx]——TF 4.56.2 基类契约如此，此处不自行赋值
        # （旧行 `self.max_cache_len = int(max_cache_len)` 会触发只读
        # property setter → AttributeError: no setter）。

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        staging = getattr(self, "_direct_prefill", None)
        if staging is not None:
            return staging.update(layer_idx, key_states, value_states, cache_kwargs)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def prepare_direct_prefill(cache: TLIStaticCache, model, length: int, _linear_targets=None):
    """Prepare direct partitioned storage for one dense compact+evict prefill.

    ``_linear_targets`` is internal to HybridBatchCacheBuilder: native-dense
    layers bind their B=1 metadata proxy to the final batch row on first write.
    """
    if (type(cache) is not TLIStaticCache or cache.batch_size != 1
            or getattr(cache, "_hybrid_batch", False)
            or getattr(cache, "_direct_prefill", None) is not None
            or type(length) is not int or not 1 <= length < cache.max_cache_len
            or model.training or torch.is_grad_enabled()
            or os.environ.get("TLI_DECODE_DENSE", "0") == "1"
            or os.environ.get("TLI_SPARSE_PREFILL", "0") == "1"):
        raise ValueError("direct prefill requires fresh compact static cache, eval/no_grad and B=1 capacity")
    modules = {m.layer_idx: m for m in model.modules()
               if getattr(getattr(m, "indexer", None), "far_skip_backend", "legacy") == "compact"}
    if not modules:
        raise ValueError("direct prefill requires compact Qwen3 attention")
    targets = {} if _linear_targets is None else dict(_linear_targets)
    if any(type(li) is not int or not 0 <= li < len(cache.layers) for li in targets):
        raise ValueError("direct prefill linear-target layer index out of range")
    states = []
    for li, old in enumerate(cache.layers):
        module = modules.get(li)
        config = None
        if module is not None:
            idx = module.indexer
            if (idx.far_skip_cache != "evict" or module.sliding_window is not None
                    or module.head_dim != 128 or idx.far_skip_config is None):
                raise ValueError("direct prefill requires compact+evict, head_dim=128 and no native SWA")
            config = idx.far_skip_config
            if li in targets:
                raise ValueError("compact rows must transfer independent storage, not a batch-row target")
            if not idx.dyn_dgate and idx.enable_layer_skip and idx._skip_ids is not None and li in idx._skip_ids:
                mode = "RING_FINAL"
            elif not idx.dyn_dgate and (not idx.enable_layer_skip or idx._skip_ids is not None):
                mode = "LINEAR_FINAL"
            else:
                mode = "LINEAR_PROVISIONAL"
        else:
            mode = "LINEAR_FINAL"
        target = targets.get(li)
        if target is not None and mode != "LINEAR_FINAL":
            raise ValueError("only final linear storage can bind a native batch-row target")
        states.append(_DirectPrefillLayer(
            TLIStaticLayer(old.max_cache_len, 1), config, mode, target))
    staging = DirectPrefillStaging(cache, length, states)
    cache.layers = [state.linear for state in states]
    for layer in cache.layers:
        layer.is_compileable = False
        layer._direct_prefill_mask = True
    cache._direct_prefill = staging
    cache._far_skip_prepared = False
    cache._far_skip_attention_mask = None
    cache._tli_seq_len = 0
    return staging


def finalize_direct_prefill(cache: TLIStaticCache, model, length: int):
    """Install request-local storage after dense prefill and gate calibration."""
    staging = getattr(cache, "_direct_prefill", None)
    if (type(cache) is not TLIStaticCache or staging is None
            or type(length) is not int or length != staging.length
            or len(staging.states) != len(cache.layers)
            or model.training or torch.is_grad_enabled()):
        raise ValueError("finalize requires the matching completed direct prefill")
    modules = {m.layer_idx: m for m in model.modules()
               if getattr(getattr(m, "indexer", None), "far_skip_backend", "legacy") == "compact"}
    if not modules:
        raise ValueError("direct prefill compact modules disappeared before finalize")

    # Validate every layer before gate resolution or storage replacement.
    for li, state in enumerate(staging.states):
        linear = state.linear
        if (state.linear is not cache.layers[li] or not state.written
                or state.mode not in ("RING_FINAL", "LINEAR_FINAL", "LINEAR_PROVISIONAL")):
            raise ValueError("direct prefill has incomplete or replaced storage")
        if state.mode == "RING_FINAL":
            ring = state.ring
            if (linear.keys is not None or linear.values is not None or ring is None
                    or ring.config != state.config or ring.logical_length != length
                    or ring.keys.shape[-2] != ring.layout.capacity
                    or ring.values.shape != ring.keys.shape):
                raise ValueError("direct prefill has incomplete or replaced ring storage")
        else:
            if (linear.keys is None or linear.values is None or state.ring is not None
                    or linear.keys.shape[0] != 1 or linear.keys.shape[-2] < length
                    or linear.values.shape != linear.keys.shape):
                raise ValueError("direct prefill has incomplete or replaced linear storage")
        target = state.linear_target
        if target is not None:
            batch_layer, row = target
            if (type(batch_layer) is not TLIStaticLayer
                    or batch_layer.keys is None or batch_layer.values is None):
                raise ValueError("direct prefill native batch-row target is not initialized")
            for source, batch in ((linear.keys, batch_layer.keys),
                                  (linear.values, batch_layer.values)):
                expected = batch[row:row+1]
                if (source.shape != expected.shape or source.stride() != expected.stride()
                        or source.data_ptr() != expected.data_ptr()
                        or source.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()):
                    raise ValueError("direct prefill source no longer matches its native batch-row target")
        module = modules.get(li)
        if module is None:
            if state.config is not None or state.mode != "LINEAR_FINAL":
                raise ValueError("direct prefill compact geometry changed before finalize")
            continue
        idx = module.indexer
        if (state.config is None or idx.far_skip_config != state.config
                or idx.far_skip_cache != "evict" or module.sliding_window is not None):
            raise ValueError("direct prefill compact storage or geometry changed")

    # Resolve every gate before allocating a provisional-skip ring or workspace.
    for li, state in enumerate(staging.states):
        module = modules.get(li)
        if module is None:
            continue
        idx = module.indexer
        idx.freeze_dyn_dgate()
        idx._resolve_skip()
        if ((state.mode == "RING_FINAL" and not idx.skip_far)
                or (state.mode == "LINEAR_FINAL" and idx.skip_far)):
            raise ValueError("direct prefill gate changed after final storage was selected")

    from .far_skip_cache import FarSkipLayer
    from .far_skip_static import FarSkipStaticLayer
    replacements = {}
    for li, state in enumerate(staging.states):
        module = modules.get(li)
        if module is None or not module.indexer.skip_far:
            continue
        idx = module.indexer
        cfg = state.config
        if cfg.near_pages == 0 and idx.sink_tokens + idx.sliding_window_size == 0:
            raise ValueError("static far selection needs a nonempty protected or page budget")
        if state.mode == "RING_FINAL":
            ring = state.ring
        else:
            # Dynamic/unknown skip pays one disclosed linear->ring compaction;
            # this is the only finalize KV copy in the three storage plans.
            ring = FarSkipLayer.from_prefill(
                state.linear.keys[..., :length, :],
                state.linear.values[..., :length, :], cfg)
        full = torch.arange(128, device=ring.keys.device)
        coarse = idx._subspace_indices(ring.keys.device) if idx.enable_subspace else full
        fine = (full if idx.l2_subspace == "full" else
                coarse if idx.enable_subspace else idx._qat_indices(ring.keys.device))
        replacements[li] = FarSkipStaticLayer.from_ring(
            ring, state.linear.max_cache_len, coarse, fine)

    # Commit only after all gates and skip workspaces have been constructed.
    for li, state in enumerate(staging.states):
        linear = state.linear
        linear.cumulative_length = length
        linear.is_compileable = True
        linear._direct_prefill_mask = False
        cache.layers[li] = replacements.get(li, linear)
        # A caller may retain the staging handle; do not let it retain evicted storage
        # or a view that keeps an obsolete batch allocation alive.
        state.linear = None
        state.ring = None
        state.linear_target = None
    cache._direct_prefill = None
    cache._far_skip_prepared = True
    cache._far_skip_attention_mask = None
    cache._tli_seq_len = length


def fill_static_from_dynamic(cache: TLIStaticCache, dyn, seq_lens, model=None):
    """把 prefill 后的 DynamicCache（有效占尾）搬进 TLIStaticCache（有效在前）。

    dyn.layers[l].keys/values：[B, Hkv, S_max, D]，行 b 有效列 =
    [S_max-lens[b], S_max)（left-pad 输入整行进 cache）。搬到
    buffer[l][b, :, 0:lens[b]]。逐层逐行 copy_（一次性，每 chunk 一次）。

    铁律一致性：搬移是纯内存布局重排（同一 KV 值、同一 token 顺序，
    列索引 0..L_b-1 对应 token 0..L_b-1 绝对位置）——下游
    prepare_mask(valid_length=L_b) 的有效区 [0, L_b) 与变长路径
    （直接消费 [0, L_b) 前缀）语义逐位一致。

    【TF 4.56 适配】4.56 移除 DynamicCache.key_cache/value_cache 属性，
    改为 layers[l].keys/values（DynamicLayer 直接字段）；旧属性访问
    AttributeError（GPU 单测未覆盖 e2e 形态，4.56.2 跑批现形）。

    model: compact 模式先冻结 gate 并直接分配 ring；其他层才分配线性
    buffer。重填 compact cache 必须提供 model，并重新 capture 外层 graph。
    """
    if getattr(cache, "_direct_prefill", None) is not None:
        raise ValueError("finalize direct prefill before conversion or decode")
    n_layers = len(cache.layers)
    if len(dyn.layers) != n_layers:
        raise RuntimeError(
            f"DynamicCache 层数 {len(dyn.layers)} != TLIStaticCache 层数 "
            f"{n_layers}")
    # Validate all source lengths before replacing any request's storage.
    if len(seq_lens) != cache.batch_size:
        raise ValueError("seq_lens must match the static cache batch")
    for l in range(n_layers):
        k, v = dyn.layers[l].keys, dyn.layers[l].values
        if (k is None or v is None or k.ndim != 4 or v.shape != k.shape
                or k.shape[0] != cache.batch_size):
            raise ValueError("prefill KV must match the static cache batch and shape")
        if any(type(L) is not int or L < 0 or L > k.shape[-2]
               or L > cache.layers[l].max_cache_len for L in seq_lens):
            raise ValueError("prefill length exceeds source or static cache capacity")
    from .far_skip_static import FarSkipStaticLayer, prepare_far_skip_static
    if model is None and any(isinstance(layer, FarSkipStaticLayer) for layer in cache.layers):
        raise ValueError("refilling compact cache requires model and graph recapture")
    compact_layers = (prepare_far_skip_static(cache, dyn, seq_lens, model)
                      if model is not None else set())
    for l in range(n_layers):
        if l in compact_layers:
            continue  # Already seeded directly from prefill into bounded storage.
        k = dyn.layers[l].keys
        v = dyn.layers[l].values
        layer = cache.layers[l]
        if layer.keys is None:
            layer.lazy_initialization(k)
        S_max = k.shape[-2]
        for b in range(cache.batch_size):
            L = seq_lens[b]
            if L:
                layer.keys[b, :, :L].copy_(k[b, :, S_max - L:])
                layer.values[b, :, :L].copy_(v[b, :, S_max - L:])
        layer.cumulative_length = max(seq_lens, default=0)
