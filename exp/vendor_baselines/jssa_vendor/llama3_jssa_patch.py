"""JSSA 投影子空间选择的 Llama 实现；算法口径与 Qwen3 适配一致。

JSSA = Joint Subspace Sparse Attention（arXiv:2610.19026v1）：
  - 校准基 M = TopEig_r(G_Qm/tr + G_K/tr)（r=32，[H_kv, r, d]）
  - decode 单级 token 稀疏：GQA 组平均 q → proj_q = q_kv·M^T → 对全 K
    投影子空间打分 top-B/head → 精确注意力（budget = 每 kv head top-B）
  - prefill dense

Llama 与 Qwen3 的差异：Llama 无 q_norm/k_norm，投影前不额外归一化。
"""
import types

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb, repeat_kv,
)

from qwen3_jssa_patch import _load_basis, _validate_basis_for_modules


def _llama_jssa_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    q_len = hidden_states.shape[1]
    if q_len > 1:
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs)

    if cache_position is None and past_key_values is not None:
        previous_length = int(past_key_values.get_seq_length(self.layer_idx))
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(
        hidden_states, attention_mask, past_key_values, cache_position)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    length = key_states.shape[2]
    kv_heads = key_states.shape[1]
    query_heads = query_states.shape[1]
    head_dim = key_states.shape[-1]
    groups = query_heads // kv_heads
    n_selected = min(self.jssa_budget, length)

    query_grouped = query_states[0, :, 0, :].reshape(
        kv_heads, groups, head_dim)
    query_kv = query_grouped.mean(dim=1)
    basis = self.jssa_M.to(device=query_states.device, dtype=query_states.dtype)
    projected_query = torch.einsum("hd,hcd->hc", query_kv, basis)
    projected_key = torch.einsum("hld,hcd->hlc", key_states[0], basis)
    scores = torch.einsum("hc,hlc->hl", projected_query, projected_key)
    top_idx = torch.topk(scores, k=n_selected, dim=-1).indices

    selected_key = key_states[0].gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, head_dim))
    selected_value = value_states[0].gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, head_dim))
    selected_key = repeat_kv(selected_key.unsqueeze(0), groups)
    selected_value = repeat_kv(selected_value.unsqueeze(0), groups)
    attn_weights = torch.matmul(
        query_states, selected_key.transpose(-2, -1)) * self.scaling
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, selected_value)
    attn_output = attn_output.transpose(1, 2).reshape(
        *input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def enable_jssa(model, basis_path: str, budget: int = 1024, rank: int = 32) -> int:
    if type(budget) is not int or budget <= 0:
        raise ValueError("JSSA budget must be a positive integer")
    if type(rank) is not int or rank <= 0:
        raise ValueError("JSSA rank must be a positive integer")

    probe = next(
        (module for module in model.modules()
         if isinstance(module, LlamaAttention)), None)
    if probe is None:
        raise ValueError("model 中未找到 LlamaAttention")
    basis_dict = _load_basis(
        basis_path, probe.q_proj.weight.device, probe.q_proj.weight.dtype)

    modules = [module for module in model.modules()
               if isinstance(module, LlamaAttention)]
    _validate_basis_for_modules(modules, basis_dict, rank,
                                int(model.config.num_hidden_layers))
    for module in modules:
        module.jssa_budget = budget
        module.jssa_rank = rank
        module.jssa_M = basis_dict[int(module.layer_idx)]
        module.flash_forward = module.forward
        module.forward = types.MethodType(_llama_jssa_forward, module)
    return len(modules)
