"""Quest official high-performance kernel adapter for Qwen3.

This module keeps the official Quest kernel runtime (`quest.utils` plus the
compiled `quest._kernels` extension) unmodified and only performs interface
glue for HuggingFace Qwen3Attention:

* load the model in fp16 because Quest's PyTorch extension dispatches Half;
* repeat GQA K/V heads to the full Qo head count before the official paged KV
  cache so the upstream non-GQA kernels see a rectangular cache;
* apply Qwen3 q_norm/k_norm before the official RoPE kernel;
* pass Qwen3's rope_theta into the official rope kernels;
* manage one shared `quest.utils.InferenceController` on the model, mirroring
  the lifecycle of the official `quest.models.llama.LlamaModel`.

It is separate from `qwen3_quest_patch.py`, which uses the official PyTorch
reference (`upstream/evaluation/quest_attention.py`) and remains available.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Tuple

import torch

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

_VENDOR_ROOT = Path(__file__).resolve().parent
_UPSTREAM_ROOT = _VENDOR_ROOT / "upstream"
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

import quest.utils as quest_utils  # noqa: E402


def _repeat_kv_nhd(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Repeat KV heads along the head dimension for GQA.

    Input and output use the official kernel NHD layout `[q_len, heads, dim]`.
    """
    if n_groups == 1:
        return x
    return x.repeat_interleave(n_groups, dim=1).contiguous()


def _controller_begin_if_needed(self, q_len: int) -> None:
    controller = self._quest_controller
    num_layers = controller._quest_num_layers
    if self.layer_idx == 0:
        controller.prepare_metadata(q_len)
        controller.set_page_budget(controller._quest_max_page_limit)
        controller.begin_forward(q_len)
    if self.layer_idx == controller._quest_skip_layer:
        controller.end_forward()
        controller.set_page_budget(controller._quest_page_budget)
        controller.begin_forward(q_len, updateTensor=False)
    if self.layer_idx == num_layers - 1:
        # The official LlamaModel calls this after the layer loop. Calling it
        # here keeps the per-attention adapter self-contained.
        pass


def _qwen3_quest_kernel_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise RuntimeError("Quest official kernels only support batch size 1")

    controller = getattr(self, "_quest_controller", None)
    if controller is None:
        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    if self.layer_idx == 0:
        controller.prepare_metadata(q_len)
        controller.set_page_budget(controller._quest_max_page_limit)
        controller.begin_forward(q_len)
    if self.layer_idx == controller._quest_skip_layer:
        controller.end_forward()
        controller.set_page_budget(controller._quest_page_budget)
        controller.begin_forward(q_len, updateTensor=False)

    # Qwen3 q/k RMS norm + projection. Qwen3 stores unnormalized q_proj/k_proj,
    # while the official kernel expects already-normalized Q/K in NHD layout.
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    )
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    )
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    query_states = query_states.reshape(q_len, self.num_heads, self.head_dim)
    key_states = key_states.reshape(q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.reshape(q_len, self.num_key_value_heads, self.head_dim)

    if self.num_key_value_groups != 1:
        key_states = _repeat_kv_nhd(key_states, self.num_key_value_groups)
        value_states = _repeat_kv_nhd(value_states, self.num_key_value_groups)

    # `prepare_metadata` 已把本步 q_len 计入 kv_cache.seqlen；官方
    # QuestAttention 的 RoPE offset 是 `seqlen - q_len`，即本步之前的
    # 已缓存长度。直接用 seqlen 会让 prefill 整体偏移 q_len、decode 偏移
    # 1，RoPE 位置错位（曾导致生成退化）。
    past_len = controller.kv_cache.seqlen - q_len
    rope_theta = float(getattr(self, "_quest_rope_theta", 1e4))

    quest_utils.apply_rope_in_place(
        query_states, key_states, past_len, rope_scale=1.0, rope_theta=rope_theta
    )
    quest_utils.append_kv(key_states, value_states, controller, self.layer_idx)

    if q_len > 1:
        attn_output = quest_utils.prefill_forward(
            query_states,
            controller,
            self.layer_idx,
            rope_scale=1.0,
            rope_theta=rope_theta,
        )
    else:
        if not controller.need_estimate():
            attn_output = quest_utils.decode_sparse_attn(
                query_states,
                controller,
                self.layer_idx,
                controller.kv_indices_without_last,
                rope_scale=1.0,
                rope_theta=rope_theta,
            )
        else:
            estimated_attn_score = quest_utils.decode_estimate(
                query_states,
                controller,
                self.layer_idx,
            )
            quest_utils.decode_topk(estimated_attn_score, controller)
            attn_output = quest_utils.decode_sparse_attn(
                query_states,
                controller,
                self.layer_idx,
                controller.topk_dindices_buffer,
                rope_scale=1.0,
                rope_theta=rope_theta,
            )

    attn_output = attn_output.unsqueeze(0).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if self.layer_idx == controller._quest_num_layers - 1:
        controller.end_forward()

    # Keep the HF DynamicCache length/metadata coherent for `generate`. The
    # authoritative KV for Quest attention lives in the official paged cache.
    if past_key_values is not None:
        cache_kwargs = {}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        if position_embeddings is not None:
            cache_kwargs["sin"] = position_embeddings[0]
            cache_kwargs["cos"] = position_embeddings[1]
        cache_key = key_states.unsqueeze(0).transpose(1, 2)
        cache_value = value_states.unsqueeze(0).transpose(1, 2)
        past_key_values.update(cache_key, cache_value, self.layer_idx, cache_kwargs)

    return attn_output, None


def enable_quest(
    model,
    token_budget: int = 1024,
    chunk_size: int = 16,
    max_seq_len: int = 40960,
    page_size: int = 16,
) -> int:
    """Bind the official Quest kernel path to all Qwen3Attention modules."""
    if type(token_budget) is not int or type(chunk_size) is not int:
        raise ValueError("Quest token_budget/chunk_size must be integers")
    if token_budget <= 0 or chunk_size <= 0:
        raise ValueError("Quest token_budget/chunk_size must be positive")
    if page_size <= 0 or max_seq_len <= 0:
        raise ValueError("Quest page_size/max_seq_len must be positive")

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // num_heads)
    dtype = torch.float16
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Quest official kernels require a CUDA model")

    controller = quest_utils.InferenceController(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        page_budget=max(1, token_budget // page_size),
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
    controller._quest_num_layers = num_layers
    controller._quest_skip_layer = 2
    controller._quest_page_budget = max(1, token_budget // page_size)
    controller._quest_max_page_limit = 1024 * 1024

    model._quest_controller = controller

    def _reset_quest_controller():
        # 每个 LongBench 样本结束后释放官方 paged KV/metadata pool，避免多个
        # 样本共用一个 InferenceController 时池耗尽（`pop from an empty set`）。
        # 官方 clean_states 只 release 两个 KvCache；下一样本 layer0 的
        # prepare_metadata/begin_forward 会重新按需分配页。
        model._quest_controller.clean_states()

    model.reset_quest_controller = _reset_quest_controller

    rope_theta = float(getattr(model.config, "rope_theta", 10000.0))

    n_patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue
        module.layer_id = int(module.layer_idx)
        module.num_heads = int(module.config.num_attention_heads)
        module.num_key_value_heads = int(module.config.num_key_value_heads)
        module.num_key_value_groups = module.num_heads // module.num_key_value_heads
        module.hidden_size = int(module.num_heads * module.head_dim)
        module._quest_controller = controller
        module._quest_rope_theta = rope_theta
        module.flash_forward = module.forward
        module.forward = types.MethodType(_qwen3_quest_kernel_forward, module)
        n_patched += 1

    if n_patched != num_layers:
        raise RuntimeError(
            f"Quest kernel Qwen3 patch count {n_patched} != config layers {num_layers}"
        )
    return n_patched
