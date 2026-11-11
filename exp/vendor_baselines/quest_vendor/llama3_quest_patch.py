"""Quest 官方 forward 的 Llama 适配层（B 轨 vendor，复用官方 forward 不改算法）。

本文件只做「接口胶水」：把 Quest 官方 forward（quest_attention_official.py，
与官方仓库逐字节一致）绑定到 LlamaAttention。核心算法（sign 符号匹配 +
chunk max-pool score + topk chunk + 因果 tril）完全来自官方源码。

口径（对齐 Quest 论文）：token_budget=1024、chunk_size=16、前 2 层 dense。
"""
import inspect
import types
from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaAttention

from _upstream_loader import load_official_module


quest_official = load_official_module()
quest_official_forward = quest_official.forward


def _install_rotary_signature_compat() -> None:
    """适配 Transformers 5.x 删除 apply_rotary_pos_emb 的 position_ids 参数。"""
    rotary = quest_official.apply_rotary_pos_emb
    if (getattr(rotary, "_quest_position_ids_compat", False)
            or "position_ids" in inspect.signature(rotary).parameters):
        return

    def _compat(query_states, key_states, cos, sin, position_ids=None, **kwargs):
        return rotary(query_states, key_states, cos, sin, **kwargs)

    _compat._quest_position_ids_compat = True
    quest_official.apply_rotary_pos_emb = _compat


def _llama_quest_wrapper(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Llama 接口 → 官方 Quest forward。prefill/浅层走 dense（flash_forward）。"""
    q_len = hidden_states.shape[1]

    if q_len > 1 or self.layer_id < 2:
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    if cache_position is None and past_key_values is not None:
        previous_length = int(past_key_values.get_seq_length(self.layer_idx))
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(
        hidden_states, attention_mask, past_key_values, cache_position)

    if cache_position is not None:
        position_ids = cache_position.unsqueeze(0)
    else:
        position_ids = torch.arange(
            q_len, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)

    kwargs.pop("position_ids", None)
    kwargs.pop("use_cache", None)
    kwargs.pop("output_attentions", None)
    attn_output, _, new_past_key_value = self._official_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=False,
        use_cache=True,
        **kwargs,
    )
    return attn_output, new_past_key_value


def enable_quest(model, token_budget: int = 1024, chunk_size: int = 16) -> int:
    """把 model 内全部 LlamaAttention 绑定官方 Quest forward。返回 patch 层数。"""
    if (type(token_budget) is not int or type(chunk_size) is not int
            or token_budget <= 0 or chunk_size <= 0):
        raise ValueError("Quest budget/chunk_size must be positive integers")

    _install_rotary_signature_compat()

    model_rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if model_rotary is None:
        raise RuntimeError("Llama 模型顶层无 rotary_emb，无法运行官方 Quest")

    n_patched = 0
    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue
        module.layer_id = int(module.layer_idx)
        module.num_heads = module.config.num_attention_heads
        module.num_key_value_heads = module.config.num_key_value_heads
        module.hidden_size = module.num_heads * module.head_dim
        module.rotary_emb = model_rotary
        module.token_budget = token_budget
        module.chunk_size = chunk_size
        module.flash_forward = module.forward
        module._official_forward = types.MethodType(
            quest_official_forward, module)
        module.forward = types.MethodType(_llama_quest_wrapper, module)
        n_patched += 1

    expected = int(model.config.num_hidden_layers)
    if n_patched != expected:
        raise RuntimeError(
            f"Quest Llama patch 层数 {n_patched} != config 层数 {expected}")
    return n_patched
