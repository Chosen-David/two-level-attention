import os
import sys

import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward
)

from typing import Optional, Callable
from ..indexer import Indexer
from ..ops.eager_decoding import eager_decoding_attn
# 【E159 阶段 2】graph/batch 数据通路复用 qwen3 实现（行为逐位一致）：
# _tl_decode_batched（staged：ctx.seq_lens Python 侧 / graph：valid_len
# tensor 侧，内部自分流），batch_decode_ctx（BatchCtx 线程局部）。
from ..baseline_runtime import batch_decode_ctx
from .qwen3_attn_patch import _tl_decode_batched, _can_drop_static_mask, _paper_route
from .utils import (
    prepare_seqlens,
    prepare_cu_seqlens,
    prepare_cu_seqlens_from_mask,
    unpad_tensor
)


def llama3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """E159：LlamaAttention TLI patch（qwen3_attn_forward 同构移植）。

    移植动机（E159 阶段 2，2026-10-23）：旧版 decode 入口只接受 [1,S]
    2D bool padding mask，无法消费 E152v4/E159v4 graph 机器的 4D
    additive mask + TLIStaticCache 固定容量布局（GraphRunner 注入
    m4_buf）→ Llama graph 口径 fail-fast。移植后 decode 具备 qwen3 的
    三层通路：
      ① batch_decode_ctx 非 None → _tl_decode_batched（staged/graph
         张量路径，复用 qwen3 实现）——E159v4 两臂（graph 模式 +
         eager staged 参照臂）都走这里；
      ② 单序列 2D bool/None mask 老路径（B=1 逐 token，与旧版
         llama3_attn_forward 逐位一致：cu_seqlens/prepare_mask/
         eager_decoding_attn 链不变）——E159 阶段 1（e152 口径，
         dynamic cache + mask=None）走这里；
      ③ B2 走廊外形态显式拒绝（同旧版语义）。

    与 qwen3 版的结构差异（Llama 无这些组件）：
      - 无 q_norm/k_norm（投影后直接 view）；
      - LlamaAttention（HF 4.56.2）无 self.sliding_window 属性 →
        _sw = getattr(..., None)（Llama 恒 None，bypass 条件不受影响）；
      - prefill dense attention_interface 调用不传 sliding_window
        kwarg（与旧 llama 版一致；qwen3 版多传，sdpa 下行为等价，
        但保持与阶段 1 已采数据的调用形态逐位一致优先）。

    等价性：champion 配置（无 static pair / 无 dyn_dgate / 无稀疏
    prefill）下 observe_prefill_q / calibrate_dyn_dgate / freeze_
    dyn_dgate / TLI_SPARSE_PREFILL 分支均为 no-op 或不触发 → 阶段 1
    口径（dynamic cache 单序列）与旧版逐位等价。
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cur_len = 0
    is_static_cache = False
    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # 【E134-pre】StaticCache 支持（与 qwen3 patch 同款判别）：
        # 正的 get_max_cache_shape()（DynamicCache 为 -1）。静态容量分支：
        #   - batch_decode_ctx 且 q_len==1 → 全宽透传（cur_len=S_cap ≠1
        #     → 正确落 decode 分支，进入 ① batch 通路）；
        #   - 其余 → 切到当前序列长（单序列静态口径，零扰动）。
        capacity = past_key_values.get_max_cache_shape()
        is_static_cache = capacity is not None and capacity > 0
        if is_static_cache:
            _ctx = batch_decode_ctx()
            if _ctx is not None and input_shape[1] == 1:
                if not key_states.is_contiguous():
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
                cur_len = key_states.shape[-2]
            else:
                if (self.layer_idx == 0
                        or getattr(past_key_values, "_tli_seq_len", None) is None):
                    if cache_position is not None:
                        past_key_values._tli_seq_len = int(cache_position[-1]) + 1
                    else:
                        past_key_values._tli_seq_len = past_key_values.get_seq_length(self.layer_idx)
                _S = past_key_values._tli_seq_len
                key_states = key_states[:, :, :_S, :]
                value_states = value_states[:, :, :_S, :]
                # prefill（q_len>1）规整化一次（strided view → SDPA
                # 接受性灰区买确定性）；decode 自家 Triton kernel 吃
                # strided 不拷贝（与 qwen3 同款零扰动口径）。
                if input_shape[1] > 1 and not key_states.is_contiguous():
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
                cur_len = _S
        else:
            cur_len = past_key_values.get_seq_length(self.layer_idx)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    _sw = getattr(self, "sliding_window", None)

    if (past_key_values is not None
            and cur_len == input_shape[1]):
        # prefill 分支：cache 更新后长 == q_len（首步全量），原生 dense。
        indexer: Indexer = self.indexer
        indexer.clear()
        # 【E134-pre】静态缓存 4D bool 容量 mask 旁路（dynamic cache /
        # float mask 不旁路，fail-closed 回原路径）——Llama _sw 恒 None。
        if (attention_mask is not None
                and _sw is None
                and is_static_cache
                and _can_drop_static_mask(past_key_values, attention_mask, cur_len, prefill=True)):
            attention_mask = None
        # E85f：prefill q 统计采集（champion 无 static pair → no-op）
        if hasattr(indexer, "observe_prefill_q"):
            indexer.observe_prefill_q(query_states)
        # E137：动态 D' gate 标定（champion 无 dyn_dgate → no-op）
        if hasattr(indexer, "calibrate_dyn_dgate"):
            indexer.calibrate_dyn_dgate(query_states, key_states)
        # 稀疏 prefill（TLI_SPARSE_PREFILL=1 门控；E159 协议恒 0 → 不触发）
        if (os.environ.get("TLI_SPARSE_PREFILL", "0") == "1"
                and attention_mask is None and input_shape[0] == 1):
            from ..ops.eager_prefill import sparse_prefill_attn
            q_t = query_states.transpose(1, 2)
            k_t = key_states.transpose(1, 2)
            v_t = value_states.transpose(1, 2)
            attn_output = sparse_prefill_attn(
                indexer, q_t, k_t, v_t, softmax_scale=self.scaling
            )
            _paper_route(self.layer_idx, "sparse-prefill:" + os.environ.get("TLI_PREFILL_RUNTIME", "reference"))
            attn_weights = None
        else:
            # 与旧 llama 版调用形态一致：不传 sliding_window kwarg
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            _paper_route(self.layer_idx, "dense-prefill")
    else:
        # decode 分支：① batch_decode_ctx（E159v4 graph/staged 两臂）→
        # _tl_decode_batched（内部 staged / graph 张量自分流）。
        _ctx = batch_decode_ctx()
        if _ctx is not None:
            attn_output = _tl_decode_batched(
                self, self.indexer, query_states, key_states,
                value_states, _ctx)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx, "sparse-decode-batch")
            return attn_output, None
        # E138 消融（TLI_DECODE_DENSE=1）：显式 dense decode（协议恒 0）
        if os.environ.get("TLI_DECODE_DENSE", "0") == "1":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx, "dense-decode")
            return attn_output, attn_weights
        # 【B2 守卫】（与旧版逐字一致）use_cache=False / chunked
        # prefill / batch>1 / 非 [1,S] 2D bool mask 显式拒绝。
        if past_key_values is None:
            raise RuntimeError(
                "TLI 稀疏 patch 需要 KV cache（use_cache=True）；当前 "
                "past_key_values=None（use_cache=False 形态，训练/PPL/"
                "logprob 打分不支持稀疏索引路径），拒绝执行。"
            )
        if input_shape[0] != 1 or input_shape[1] != 1:
            raise RuntimeError(
                f"TLI 稀疏 decode 路径仅支持 batch=1 且 q_len=1 的逐 token "
                f"解码；当前 batch={input_shape[0]}、q_len={input_shape[1]}"
                f"（chunked prefill / batch>1 会静默只算首 token 或形状"
                f"崩溃），拒绝执行。"
            )
        # 【E134-pre】StaticCache decode：HF 4D bool 容量 mask 旁路验证
        # （q_len=1 因果约束空转；Llama _sw 恒 None）
        if (attention_mask is not None and is_static_cache
                and _sw is None
                and _can_drop_static_mask(past_key_values, attention_mask, cur_len, prefill=False)):
            attention_mask = None
        if attention_mask is not None and (
            attention_mask.dim() != 2 or attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != (1, key_states.shape[-2])
            or not bool(attention_mask[0, -1])
        ):
            raise RuntimeError(
                f"TLI 稀疏 decode 路径的 attention_mask 须为 [1, S] 2D bool "
                f"padding mask，且当前 query（最后位置）须有效；当前 dim={attention_mask.dim()}、"
                f"dtype={attention_mask.dtype}（HF 实际传 4D float causal "
                f"mask，按 2D 假设处理不正确），拒绝执行。"
            )
        indexer: Indexer = self.indexer
        # A6：decode 首步冻结 E137 动态标定（champion 无 dyn_dgate → no-op）
        if hasattr(indexer, "freeze_dyn_dgate"):
            indexer.freeze_dyn_dgate()
        query_states, key_states, value_states = (x.transpose(1, 2) for x in (query_states, key_states, value_states))
        if attention_mask is not None:
            cu_seqlens_k = prepare_cu_seqlens_from_mask(attention_mask)
            # Decode has one live query but S cached keys. The KV-length mask
            # cannot index Q; the guard above guarantees this query is valid.
            key_states, value_states = (unpad_tensor(x, attention_mask) for x in (key_states, value_states))
        else:
            cu_seqlens_k = prepare_cu_seqlens(key_states)
            query_states, key_states, value_states = (x.view(1, -1, *x.shape[-2:]) for x in (query_states, key_states, value_states))
        query_position_ids = prepare_seqlens(cu_seqlens_k) - 1
        mask, block_size = indexer.prepare_mask(
            query_states,
            query_position_ids,
            key_states,
            cu_seqlens_k
        )
        attn_output = eager_decoding_attn(
            query_states,
            key_states,
            value_states,
            mask,
            block_size,
            cu_seqlens_k,
            softmax_scale=self.scaling,
        )
        _paper_route(self.layer_idx, "sparse-decode")
        attn_weights = None

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
