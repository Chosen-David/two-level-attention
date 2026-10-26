"""Per-layer adaptive KV cache: one class, two strategies frozen at prefill end.

Motivation:
    Non-skip-far layers need full-width KV (far tokens are topk candidates).
    Skip-far layers only need sink + near (far is physically evicted).
    Two storage strategies in one class—indexer/attention dispatch on the
    layer's strategy flag, not on the class type.

Strategies:
    linear  — [sink | near | far] full-width buffer, scatter_ write (graph-safe)
    ring    — [sink | ring_tail] cyclic buffer, slot() write (graph-safe)

Decision:  calibrate_dyn_dgate (prefill end) → set_strategy() frozen for decode.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache

from .far_skip import FarSkipConfig, FarSkipLayout, select_far_skip, attend_far_skip
from .far_skip_workspace import FarSkipWorkspace


class HybridLayer:
    """Single KV layer with adaptive storage strategy.

    Strategy is frozen before decode by set_strategy().  update() / get_seq_length()
    / get_max_cache_shape() are strategy-agnostic; get_kv_prefix_len() lets the
    dispatch code (qwen3_attn_forward) distinguish paths.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, max_batch_size: int,
                 sink_tokens: int = 4, *, near_length: int = 0,
                 window_tokens: int = 128, block_size: int = 64,
                 near_pages: int = 8, token_budget: int = 0):
        if max_cache_len < 1 or max_batch_size < 1:
            raise ValueError("HybridLayer 需要 max_cache_len>=1 且 max_batch_size>=1")
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(max_batch_size)
        self.sink_tokens = sink_tokens
        self.keys = None
        self.values = None
        self.num_heads = None
        self.head_dim = None
        self.cumulative_length = 0

        # Strategy config (used only when strategy='ring')
        cfg_kwargs = dict(block_size=block_size, near_length=max(near_length, block_size),
                          near_pages=near_pages,
                          token_budget=token_budget if token_budget > 0 else max(near_length, block_size) + window_tokens,
                          sink_tokens=sink_tokens, window_tokens=window_tokens)
        self._ring_cfg = FarSkipConfig(**cfg_kwargs)
        self._ring_layout: FarSkipLayout | None = None
        self._ring_logical_length = 0

        # Prefill-end dispatch
        self.strategy: str | None = None  # 'linear' | 'ring'

    # ── strategy freeze ──────────────────────────────────────────────────
    def set_strategy(self, strategy: str, *, force: bool = False):
        """Called once at prefill end (fill_static_from_dynamic time).

        set_strategy('ring') validates ring capacity against max_cache_len.
        Subclasses (FarSkipHybridLayer) pass force=True to bypass the
        already-frozen guard (they set strategy at construction time).
        """
        if strategy not in ("linear", "ring"):
            raise ValueError(f"unknown strategy: {strategy}")
        if not force and self.strategy is not None and self.strategy != strategy:
            raise RuntimeError(f"strategy already frozen as {self.strategy}")
        if strategy == "ring":
            # 校验 ring 物理容量不超过分配的全量 buffer
            tail_cap = max(self._ring_cfg.near_length, self._ring_cfg.block_size) \
                + self._ring_cfg.window_tokens + self._ring_cfg.block_size - 1
            ring_cap = self.sink_tokens + tail_cap
            if ring_cap > self.max_cache_len:
                raise ValueError(
                    f"ring layout capacity {ring_cap} (sink={self.sink_tokens} "
                    f"+ tail={tail_cap}) exceeds S_cap={self.max_cache_len}")
        self.strategy = strategy
        if strategy == "ring" and self._ring_layout is None and self.keys is not None:
            self._ring_layout = FarSkipLayout(
                self.sink_tokens,
                max(self._ring_cfg.near_length, self._ring_cfg.block_size)
                + self._ring_cfg.window_tokens + self._ring_cfg.block_size - 1)

    # ── lazy init ────────────────────────────────────────────────────────
    def lazy_initialization(self, key_states: torch.Tensor):
        if key_states.ndim != 4:
            raise ValueError(f"HybridLayer 期望 4D key_states, got {key_states.ndim}D")
        self.num_heads = key_states.shape[1]
        self.head_dim = key_states.shape[-1]
        self.dtype, self.device = key_states.dtype, key_states.device
        if key_states.shape[0] != self.max_batch_size:
            raise ValueError(f"batch {key_states.shape[0]} != {self.max_batch_size}")
        shape = (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)

    # ── update ───────────────────────────────────────────────────────────
    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.ndim != 4 or key_states.shape[-2] != 1:
            raise RuntimeError(
                f"decode-only; got shape={tuple(key_states.shape)}")
        if self.keys is None:
            self.lazy_initialization(key_states)
        if self.strategy == "ring":
            return self._update_ring(key_states, value_states)
        return self._update_linear(key_states, value_states)

    def _update_linear(self, key_states, value_states):
        """对齐 TLIStaticLayer.update：batch_decode_ctx + scatter_ 张量索引。"""
        from ..baseline_runtime import batch_decode_ctx  # noqa: E402
        ctx = batch_decode_ctx()
        if ctx is not None and ctx.batch != self.max_batch_size:
            raise RuntimeError(
                f"HybridLayer ctx.batch={ctx.batch} != 构造 batch "
                f"{self.max_batch_size}")
        from .cuda_graph_state import graph_state  # noqa: E402
        _gs = graph_state()
        if _gs.enabled and _gs.pos_buf is not None:
            pos = _gs.pos_buf
        elif ctx is not None:
            pos = torch.tensor(
                [ctx.seq_lens[b] - 1 for b in range(self.max_batch_size)],
                dtype=torch.long, device=self.keys.device)
        else:
            pos = torch.full((self.max_batch_size,), self.cumulative_length,
                             dtype=torch.long, device=self.keys.device)
        pos = torch.clamp(pos, 0, self.max_cache_len - 1)
        idx = pos.view(-1, 1, 1, 1).expand(
            self.max_batch_size, self.num_heads, 1, self.head_dim)
        self.keys.scatter_(2, idx, key_states)
        self.values.scatter_(2, idx, value_states)
        self.cumulative_length += 1
        return self.keys, self.values

    def _update_ring(self, key_states, value_states):
        if self._ring_layout is None:
            self._ring_layout = FarSkipLayout(
                self.sink_tokens,
                max(self._ring_cfg.near_length, self._ring_cfg.block_size)
                + self._ring_cfg.window_tokens + self._ring_cfg.block_size - 1)
        slot = self._ring_layout.slot(self._ring_logical_length)
        self.keys[..., slot:slot + 1, :].copy_(key_states)
        self.values[..., slot:slot + 1, :].copy_(value_states)
        self._ring_logical_length += 1
        self.cumulative_length += 1
        return self.keys, self.values

    # ── status queries ───────────────────────────────────────────────────
    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def get_kv_prefix_len(self) -> int:
        if self.strategy == "ring" and self._ring_layout is not None:
            return self._ring_layout.capacity
        return self.max_cache_len

    def reset(self):
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
        self.cumulative_length = 0
        self._ring_logical_length = 0

    # ── for decoder dispatch ─────────────────────────────────────────────
    def ring_logical_length(self) -> int:
        return self._ring_logical_length

    @property
    def ring_layout(self) -> FarSkipLayout | None:
        return self._ring_layout


class HybridCache(Cache):
    """Unified decode KV cache wrapping per-layer HybridLayers."""

    def __init__(self, config, max_cache_len: int, batch_size: int,
                 sink_tokens: int = 4, near_length: int = 0,
                 window_tokens: int = 128, block_size: int = 64,
                 near_pages: int = 8, token_budget: int = 0):
        layers = [HybridLayer(max_cache_len, batch_size, sink_tokens,
                              near_length=near_length, window_tokens=window_tokens,
                              block_size=block_size, near_pages=near_pages,
                              token_budget=token_budget)
                  for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)
        self.batch_size = int(batch_size)


# ── Production far-skip path ─────────────────────────────────────────────
class FarSkipHybridLayer(HybridLayer):
    """Ring-strategy HybridLayer with far-skip attention metadata.

    Replaces FarSkipStaticLayer in the production compact far-skip path.
    B=1 only.  Initialized from a DynamicLayer via from_layer().
    """

    def __init__(self, source, cfg, max_cache_len, coarse, fine):
        from .far_skip_cache import FarSkipLayer  # 防循环导入
        ring = FarSkipLayer.from_layer(source, cfg)
        # 先设 keys 再调 super().__init__（ring 容量校验需要 max_cache_len）
        self.keys, self.values = ring.keys, ring.values
        self.dtype, self.device = ring.dtype, ring.device
        self.num_heads, self.head_dim = self.keys.shape[1], self.keys.shape[-1]
        self.config, self.layout = cfg, ring.layout
        self.cumulative_length = ring.logical_length
        self.coarse = coarse.clone()
        self.fine = self.coarse if self._same_dims(coarse, fine) else fine.clone()
        self.workspace = None
        self.eager_workspace = None
        self.logical_length = ring.logical_length
        # 构造基类（不调用 lazy_initialization，ring 已分配）
        super().__init__(max_cache_len, 1,
                         sink_tokens=cfg.sink_tokens,
                         near_length=cfg.near_length,
                         window_tokens=cfg.window_tokens,
                         block_size=cfg.block_size,
                         near_pages=cfg.near_pages,
                         token_budget=cfg.token_budget)
        self.strategy = "ring"
        self._ring_layout = ring.layout
        self._ring_logical_length = ring.logical_length

    @staticmethod
    def _same_dims(a, b):
        return a.numel() == b.numel() and torch.equal(a, b)

    def lazy_initialization(self, key_states: torch.Tensor):
        """Ring 已分配，跳过基类的零初始化。"""
        if self.keys is not None:
            return
        super().lazy_initialization(key_states)

    def update(self, key_states, value_states, cache_kwargs=None):
        """cache_position 字典接口（与 FarSkipStaticLayer.update 一致）。"""
        for x, storage in ((key_states, self.keys), (value_states, self.values)):
            if (x.shape != (*storage.shape[:2], 1, storage.shape[-1])
                    or x.dtype != storage.dtype or x.device != storage.device
                    or x.requires_grad):
                raise ValueError('static far KV must match detached single-token storage')
        pos = (cache_kwargs or {}).get('cache_position')
        if pos is None or pos.shape != (1,) or pos.dtype != torch.long or pos.device != self.device:
            raise ValueError('static far KV needs one device int64 cache position')
        slot = self.layout.map(pos)
        self.keys.index_copy_(2, slot, key_states)
        self.values.index_copy_(2, slot, value_states)
        self.logical_length = int(pos) + 1
        self._ring_logical_length = self.logical_length
        self.cumulative_length = self.logical_length
        return self.keys, self.values

    def get_kv_prefix_len(self) -> int:
        return self.layout.capacity

    def reset(self):
        raise ValueError('refill far static cache from a fresh prefill and recapture')

    def select_and_attend(self, q, k, v, length, scale):
        """Eager far-skip attention (non-graph mode)."""
        if self.eager_workspace is None:
            self.eager_workspace = FarSkipWorkspace()
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        selection = select_far_skip(q_t, k_t, length, self.config,
                                    self.coarse, self.fine, scale=scale,
                                    layout=self.layout, workspace=self.eager_workspace)
        return attend_far_skip(q_t, k_t, v_t, selection, self.config,
                               scale=scale, layout=self.layout, workspace=self.eager_workspace)
