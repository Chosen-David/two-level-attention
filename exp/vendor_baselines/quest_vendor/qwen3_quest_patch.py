"""Quest (ICML'24) 稀疏 attention 的 Qwen3 移植适配层（B 轨 vendor 双证，clean-room）。

算法来源：Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
  (Tang et al., ICML 2024)。本项目未直接复用官方代码，而是按论文算法独立实现
  （sign 符号匹配 + chunk 级 max-pool score + topk chunk 选择 + 因果 tril），
  以对齐 Qwen3 (transformers 4.5x) 的 attention 接口。

口径（与 Quest 论文对齐）：
  - token_budget = 1024（等效 token 预算，与 TLI 的 K2=1024 同口径）
  - chunk_size = 16
  - prefill（q_len > 1）与浅层（layer_id < 2）走原生 dense attention
  - decode 单 token 走稀疏选择

实现要点：
  - 保留 Qwen3 原生 q_norm/k_norm（模型自身前处理，公平继承）
  - 复用 Qwen3 原生 rotary（position_embeddings）与 cache update
  - 稀疏选择只替换"哪些 chunk 参与 softmax"，其余逻辑与 dense 一致

用法：
  from exp.vendor_baselines.quest_vendor.qwen3_quest_patch import enable_quest
  model = AutoModelForCausalLM.from_pretrained(...)
  enable_quest(model, token_budget=1024, chunk_size=16)
"""

import math
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb


def _local_heavy_hitter_mask(attn_weights: torch.Tensor, token_budget: int, chunk_size: int) -> torch.Tensor:
    """Quest 论文的 chunk 级 topk 掩码。

    attn_weights: (bsz, n_head, q_len, kv_len) —— 块级 score（量化 key 与 query 的内积）。
    返回 mask: (bsz, n_head, q_len, kv_len)，选中 chunk 内 token 为 True。
    """
    seq_length = attn_weights.shape[-1]
    # pad 到 chunk_size 整数倍（mask 值 -inf，保证不入选）
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    if padding_length > 0:
        attn_weights = torch.cat(
            [
                attn_weights,
                torch.full(
                    (attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], padding_length),
                    torch.finfo(attn_weights.dtype).min,
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                ),
            ],
            dim=-1,
        )

    n_chunks = attn_weights.shape[3] // chunk_size
    # chunk 级 max-pool
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], n_chunks, chunk_size
    ).amax(dim=-1)

    # topk 选 chunk
    k = min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1))
    _, topk = chunk_attn_weights.topk(k=k, dim=-1)

    # 展开到 token 级
    topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size + torch.arange(
        chunk_size, device=topk.device
    )
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask.scatter_(-1, topk, True)
    mask = mask[:, :, :, :seq_length]
    return mask


def _quest_sparse_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3Attention 的稀疏 forward（decode 单 token 路径）。"""
    input_shape = hidden_states.shape[:-1]
    bsz, q_len = hidden_states.shape[:2]

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    kv_seq_len = key_states.shape[-2]

    # repeat_kv：与 Quest 原版一致（选择发生在 GQA 展开后）
    key_states_r = repeat_kv(key_states, self.num_key_value_groups)
    value_states_r = repeat_kv(value_states, self.num_key_value_groups)

    # --- Quest 核心：符号感知的 chunk max-pool score ---
    attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states_r * sign
    postive_query = query_states * sign

    # 按 chunk 对 max_key 做 max-pool（沿 token 维，逐 head_dim 取极值）
    n_chunks = math.ceil(kv_seq_len / self.chunk_size)
    pad_len = n_chunks * self.chunk_size - kv_seq_len
    if pad_len > 0:
        max_key = torch.cat(
            [
                max_key,
                torch.full(
                    (bsz, max_key.shape[1], pad_len, max_key.shape[3]),
                    torch.finfo(max_key.dtype).min,
                    device=max_key.device,
                    dtype=max_key.dtype,
                ),
            ],
            dim=-2,
        )
    chunk_max_key = max_key.reshape(bsz, max_key.shape[1], n_chunks, self.chunk_size, max_key.shape[3]).amax(dim=-2)
    # 展开回 token 级（每个 chunk 内所有 token 共享 chunk 极值）
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    chunk_max_key = chunk_max_key.reshape(bsz, max_key.shape[1], -1, max_key.shape[3])[:, :, :kv_seq_len, :]

    quantized_weight = torch.matmul(postive_query.float(), chunk_max_key.transpose(2, 3))

    # attention mask（若提供）
    if attention_mask is not None and attention_mask.shape[-1] == kv_seq_len:
        attn_weights = attn_weights + attention_mask
        quantized_weight = quantized_weight + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        quantized_weight = torch.max(quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min))

    token_budget = min(kv_seq_len, self.token_budget)
    mask = _local_heavy_hitter_mask(quantized_weight, token_budget, self.chunk_size)

    # 因果：tril 对角线 = 当前 decode 位置
    cur_pos = int(cache_position[-1].item()) if cache_position is not None else kv_seq_len - 1
    mask = torch.tril(mask, diagonal=cur_pos)

    attn_weights[~mask] = torch.tensor(torch.finfo(attn_weights.dtype).min)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_r)

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def enable_quest(model, token_budget: int = 1024, chunk_size: int = 16, skip_first_layers: int = 2) -> int:
    """把 model 内全部 Qwen3Attention 替换为 Quest 稀疏版本。返回 patch 的层数。

    skip_first_layers: 前 N 层保持 dense（对齐 Quest 论文 layer_id < 2 策略）。
    """
    patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal layer_id, patched
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.layer_id = layer_id
                layer_id += 1
                # 补 Quest 依赖的字段
                module.num_heads = module.config.num_attention_heads
                module.num_key_value_heads = module.config.num_key_value_heads
                module.hidden_size = module.config.hidden_size
                module.flash_forward = module.forward
                module.forward = types.MethodType(_quest_sparse_forward, module)
                module.token_budget = token_budget
                module.chunk_size = chunk_size
                patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    _patch_recursive(model)
    return patched
