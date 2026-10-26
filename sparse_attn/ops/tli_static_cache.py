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
from .hybrid_cache import HybridLayer


class TLIStaticLayer(HybridLayer):
    """固定容量 KV 层（向后兼容别名，strategy='linear'）。

    对齐 StaticLayer 的 CacheLayerMixin 契约（is_compileable / max_batch_size
    / max_cache_len / get_seq_length / get_mask_sizes / get_max_cache_shape
    / lazy_initialization / update / reset / crop / reorder_cache）。
    """

    def __init__(self, max_cache_len: int, max_batch_size: int):
        super().__init__(max_cache_len, max_batch_size)
        self.strategy = "linear"  # TLIStaticLayer 始终 linear

    # lazy_initialization / update / _update_linear / get_seq_length /
    # get_mask_sizes / get_max_cache_shape / reset 全部继承自 HybridLayer。

    def crop(self, max_length):
        raise RuntimeError("TLIStaticLayer 不支持 crop（固定容量 decode-only）")

    def reorder_cache(self, beam_idx):
        raise RuntimeError("TLIStaticLayer 不支持 reorder（greedy decode-only）")


class TLIStaticCache(Cache):
    """decode 用固定容量 KV cache（每层一个 HybridLayer 全宽缓冲）。"""

    def __init__(self, config, max_cache_len: int, batch_size: int):
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        layers = [HybridLayer(max_cache_len, batch_size) for _ in layer_types]
        super().__init__(layers=layers)
        self.batch_size = int(batch_size)
        # max_cache_len 由 Cache 基类只读 property 从 layers 推导
        # （max(layer.max_cache_len)），get_max_cache_shape 走
        # layers[layer_idx]——TF 4.56.2 基类契约如此，此处不自行赋值
        # （旧行 `self.max_cache_len = int(max_cache_len)` 会触发只读
        # property setter → AttributeError: no setter）。


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

    model: 若提供，搬移后调用 prepare_far_skip_static 对 far-skip 层做一次
    prefill-end 截断/重分配到 ring buffer 容量（物理节省 dead far KV）。
    """
    n_layers = len(cache.layers)
    if len(dyn.layers) != n_layers:
        raise RuntimeError(
            f"DynamicCache 层数 {len(dyn.layers)} != TLIStaticCache 层数 "
            f"{n_layers}")
    for l in range(n_layers):
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

    # 冻结线性策略（非 far-skip 层）
    for layer in cache.layers:
        layer.set_strategy("linear")

    # Far-skip: prefill 结束后一次性截断死 far KV → 物理节省
    if model is not None:
        from .far_skip_static import prepare_far_skip_static
        prepare_far_skip_static(cache, dyn, seq_lens, model)
