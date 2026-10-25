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

import torch
from transformers.cache_utils import Cache

from ..baseline_runtime import batch_decode_ctx


class TLIStaticLayer:
    """固定容量 KV 层。buffer [B, Hkv, S_cap, D]，有效在前。

    对齐 StaticLayer 的 CacheLayerMixin 契约（is_compileable / max_batch_size
    / max_cache_len / get_seq_length / get_mask_sizes / get_max_cache_shape
    / lazy_initialization / update / reset / crop / reorder_cache）。
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int, max_batch_size: int):
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

    def lazy_initialization(self, key_states: torch.Tensor):
        """按首层首个 decode 步（或搬移时）的 [B, Hkv, S, D] 形状预分配全宽 buffer。"""
        if key_states.ndim != 4:
            raise ValueError(f"TLIStaticLayer 期望 4D key_states，got {key_states.ndim}D")
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
        # dead 行 seq_lens 冻结（不递增）→ pos 不变 → 重复写同列（被
        # mask 排除，无污染）。非 graph 模式（pos_buf=None）回退 Python
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
        # src[b,h,0,d]（单 kernel，graph 兼容）。dead 行 pos 冻结 →
        # 重复写同列（被 mask 排除，无污染）。
        idx = pos.view(-1, 1, 1, 1).expand(
            self.max_batch_size, self.num_heads, 1, self.head_dim)
        self.keys.scatter_(2, idx, key_states)
        self.values.scatter_(2, idx, value_states)
        self.cumulative_length += 1
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, cache_position):
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


def fill_static_from_dynamic(cache: TLIStaticCache, dyn, seq_lens, *, model=None):
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
    """
    n_layers = len(cache.layers)
    if len(dyn.layers) != n_layers:
        raise RuntimeError(
            f"DynamicCache 层数 {len(dyn.layers)} != TLIStaticCache 层数 "
            f"{n_layers}")
    far_layers = set()
    if model is not None:
        from .far_skip_static import prepare_far_skip_static
        far_layers = prepare_far_skip_static(cache, dyn, seq_lens, model)
    for l in range(n_layers):
        if l in far_layers:
            continue
        k = dyn.layers[l].keys
        v = dyn.layers[l].values
        layer = cache.layers[l]
        if layer.keys is None:
            layer.lazy_initialization(k)
        S_max = k.shape[-2]
        for b in range(cache.batch_size):
            L = seq_lens[b]
            if L < 0 or L > S_max or L > layer.max_cache_len:
                raise RuntimeError(
                    f"搬移越界：b={b} L={L} S_max={S_max} "
                    f"cap={layer.max_cache_len}")
            if L:
                layer.keys[b, :, :L].copy_(k[b, :, S_max - L:])
                layer.values[b, :, :L].copy_(v[b, :, S_max - L:])

