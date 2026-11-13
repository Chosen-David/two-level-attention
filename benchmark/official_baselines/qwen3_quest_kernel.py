"""Qwen3 到本地官方 Quest CUDA 运行时的纯接口适配。

选择、Top-K、paged KV 与注意力全部来自 ``third-party/Quest`` 的
``quest.utils`` / ``quest._kernels``。本文件不实现稀疏注意力，也不提供
Torch/dense decode 回退。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Tuple

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
)

QUEST_ROOT = Path(__file__).resolve().parents[2] / "third-party" / "Quest"
if not QUEST_ROOT.is_dir():
    raise RuntimeError(f"官方 Quest 克隆不存在: {QUEST_ROOT}")
if str(QUEST_ROOT) not in sys.path:
    sys.path.insert(0, str(QUEST_ROOT))

import quest._kernels as quest_kernels  # noqa: E402
import quest.utils as quest_utils  # noqa: E402


def _assert_official_runtime() -> None:
    for module in (quest_kernels, quest_utils):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(QUEST_ROOT.resolve()):
            raise RuntimeError(f"Quest 模块并非来自本地官方克隆: {path}")


_assert_official_runtime()


LONG_PREFILL_TOKENWISE_CHUNK = 65536


def _tokenwise_prefill_chunk(original, hidden_states, chunk_tokens):
    """逐 token 算子分块降低长 prefill 临时激活；decode 与短输入原样执行。"""
    if hidden_states.ndim < 2:
        return original(hidden_states)
    token_dim = 1 if hidden_states.ndim >= 3 else 0
    if hidden_states.shape[token_dim] <= chunk_tokens:
        return original(hidden_states)
    output = torch.empty_like(hidden_states)
    for start in range(0, hidden_states.shape[token_dim], chunk_tokens):
        stop = min(start + chunk_tokens, hidden_states.shape[token_dim])
        index = [slice(None)] * hidden_states.ndim
        index[token_dim] = slice(start, stop)
        index = tuple(index)
        output[index].copy_(original(hidden_states[index]))
    return output


def _enable_long_prefill_tokenwise_chunk(model):
    for module in model.modules():
        if not isinstance(module, (Qwen3MLP, Qwen3RMSNorm)):
            continue
        original = module.forward

        def forward(self, hidden_states, _original=original):
            return _tokenwise_prefill_chunk(
                _original, hidden_states, LONG_PREFILL_TOKENWISE_CHUNK
            )

        module.forward = types.MethodType(forward, module)


def _qwen3_quest_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    del attention_mask
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise RuntimeError("Quest 官方 CUDA runtime 仅支持 batch_size=1")
    controller = self._quest_controller

    if self.layer_idx == 0:
        controller.prepare_metadata(q_len)
        controller.set_page_budget(controller._quest_max_page_limit)
        controller.begin_forward(q_len)
    if self.layer_idx == controller._quest_skip_layer:
        controller.end_forward()
        controller.set_page_budget(controller._quest_page_budget)
        controller.begin_forward(q_len, updateTensor=False)

    query_states = self.q_norm(
        self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    )
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
    )
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    query_states = query_states.reshape(q_len, self.num_heads, self.head_dim)
    key_states = key_states.reshape(q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.reshape(q_len, self.num_key_value_heads, self.head_dim)

    past_len = controller.kv_cache.seqlen - q_len
    rope_theta = float(self._quest_rope_theta)
    quest_utils.apply_rope_in_place(
        query_states, key_states, past_len, rope_scale=1.0, rope_theta=rope_theta
    )
    quest_utils.append_kv(key_states, value_states, controller, self.layer_idx)

    stats = self._quest_stats
    if q_len > 1:
        attn_output = quest_utils.prefill_forward(
            query_states,
            controller,
            self.layer_idx,
            rope_scale=1.0,
            rope_theta=rope_theta,
        )
        stats["prefill_forward"] += 1
    else:
        if controller.need_estimate():
            estimated_attn_score = quest_utils.decode_estimate(
                query_states, controller, self.layer_idx
            )
            quest_utils.decode_topk(estimated_attn_score, controller)
            indices = controller.topk_dindices_buffer
            stats["decode_estimate"] += 1
            stats["decode_topk"] += 1
        else:
            indices = controller.kv_indices_without_last
        attn_output = quest_utils.decode_sparse_attn(
            query_states,
            controller,
            self.layer_idx,
            indices,
            rope_scale=1.0,
            rope_theta=rope_theta,
        )
        stats["decode_sparse_attn"] += 1

    attn_output = attn_output.unsqueeze(0).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if self.layer_idx == controller._quest_num_layers - 1:
        controller.end_forward()

    # HF generate 仅从第 0 层 DynamicCache 读取序列长度；其余层不再重复保存
    # 第二份 KV。权威注意力 KV 始终位于官方 Quest 页池。
    if past_key_values is not None and self.layer_idx == 0:
        cache_kwargs = {}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        if position_embeddings is not None:
            cache_kwargs["sin"] = position_embeddings[0]
            cache_kwargs["cos"] = position_embeddings[1]
        past_key_values.update(
            key_states.unsqueeze(0).transpose(1, 2),
            value_states.unsqueeze(0).transpose(1, 2),
            self.layer_idx,
            cache_kwargs,
        )
    return attn_output, None


def enable_quest(
    model,
    token_budget: int = 1024,
    chunk_size: int = 16,
    max_seq_len: int = 40960,
    page_size: int = 16,
) -> int:
    """将全部 Qwen3Attention 绑定到官方 Quest kernel，禁止回退。"""
    if min(token_budget, chunk_size, max_seq_len, page_size) <= 0:
        raise ValueError("Quest budget/chunk/page/max_seq_len 必须为正整数")
    if token_budget % page_size != 0 or chunk_size != page_size:
        raise ValueError("对齐口径要求 token_budget 可整除 page_size 且 chunk_size=page_size")
    if next(model.parameters()).dtype != torch.float16:
        raise TypeError("Quest 官方 _kernels 要求 FP16 模型")
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("Quest 官方 _kernels 要求 CUDA 模型")

    _assert_official_runtime()
    _enable_long_prefill_tokenwise_chunk(model)
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // num_heads))
    device = next(model.parameters()).device
    page_budget = token_budget // page_size
    controller = quest_utils.InferenceController(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=max_seq_len,
        dtype=torch.float16,
        device=device,
        num_kv_heads=num_kv_heads,
    )
    controller._quest_num_layers = num_layers
    controller._quest_skip_layer = 2
    controller._quest_page_budget = page_budget
    controller._quest_max_page_limit = 1024 * 1024
    model._quest_controller = controller
    model.quest_runtime_stats = {
        "prefill_forward": 0,
        "decode_estimate": 0,
        "decode_topk": 0,
        "decode_sparse_attn": 0,
        "dense_or_torch_fallback": 0,
    }
    model.reset_quest_controller = controller.clean_states

    n_patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue
        module.num_heads = int(module.config.num_attention_heads)
        module.num_key_value_heads = int(module.config.num_key_value_heads)
        module.num_key_value_groups = module.num_heads // module.num_key_value_heads
        module.hidden_size = module.num_heads * module.head_dim
        module._quest_controller = controller
        module._quest_rope_theta = float(getattr(model.config, "rope_theta", 10000.0))
        module._quest_stats = model.quest_runtime_stats
        module.forward = types.MethodType(_qwen3_quest_forward, module)
        n_patched += 1
    if n_patched != num_layers:
        raise RuntimeError(f"Quest patch 层数 {n_patched} != {num_layers}")
    return n_patched
