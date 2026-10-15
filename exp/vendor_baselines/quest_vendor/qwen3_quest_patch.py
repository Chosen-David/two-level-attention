"""Quest 官方源码的 Qwen3 适配层（B 轨 vendor，复用官方 forward 不改算法）。

本文件只做「接口胶水」：把 Quest 官方 forward（quest_attention_official.py，
与官方仓库逐字节一致）绑定到 Qwen3Attention，并补齐 Qwen3 缺失的属性
（num_heads / num_key_value_heads / hidden_size / rotary_emb / layer_id）。
**核心算法（sign 符号匹配 + chunk max-pool score + topk chunk + 因果 tril）
完全来自官方源码，本层不重写任何选择逻辑。**

签名转换（wrapper）：
  Qwen3 调用: (hidden_states, position_embeddings, attention_mask,
               past_key_values, cache_position, **kwargs)
  官方 forward: (hidden_states, attention_mask, position_ids,
               past_key_value, output_attentions, use_cache, **kwargs)
  - position_ids 由 Qwen3 cache_position 提供
  - past_key_value = Qwen3 past_key_values（DynamicCache，官方支持）

口径（对齐 Quest 论文）：token_budget=1024、chunk_size=16、前 2 层 dense。
"""
import types
from typing import Tuple

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from quest_attention_official import forward as quest_official_forward


def _qwen3_quest_wrapper(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Qwen3 接口 → 官方 Quest forward。prefill/浅层走 dense（flash_forward）。"""
    q_len = hidden_states.shape[1]

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    from sparse_attn.baseline_runtime import require_unmasked_decode
    require_unmasked_decode(hidden_states, attention_mask, past_key_values, cache_position)

    # decode 单 token：官方 Quest 稀疏路径
    if cache_position is not None:
        position_ids = cache_position.unsqueeze(0)  # (1, q_len)，官方 forward 期望 (bsz, q_len)
    else:
        position_ids = torch.arange(
            q_len, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)

    # Qwen3DecoderLayer 会把 position_ids / use_cache / past_key_values /
    # cache_position / position_embeddings 都显式传给 self_attn —— 其中
    # position_ids、use_cache 与官方 forward 显式参数同名冲突（multiple
    # values），此处剔除，用上方由 cache_position 构造的 position_ids 与
    # 固定的 use_cache=True（decode 稀疏路径本就要求开 cache）。
    kwargs.pop("position_ids", None)
    kwargs.pop("use_cache", None)
    # 官方 forward 返回 3 元组 (attn_output, attn_weights, past_key_value)；
    # Qwen3DecoderLayer 期望 self_attn 返回 2 元组 (hidden_states, _)。
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
    """把 model 内全部 Qwen3Attention 绑定官方 Quest forward。返回 patch 层数。"""
    if type(token_budget) is not int or type(chunk_size) is not int or token_budget <= 0 or chunk_size <= 0:
        raise ValueError('Quest budget/chunk_size must be positive integers')
    n_patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                # 补齐官方 forward 依赖的属性
                module.layer_id = layer_id
                layer_id += 1
                module.num_heads = module.config.num_attention_heads
                module.num_key_value_heads = module.config.num_key_value_heads
                # 【32B 适配 bug 修复，2026-10-12】官方 forward 末尾
                # `attn_output.reshape(bsz, q_len, self.hidden_size)` 要求
                # hidden_size == num_heads*head_dim（Llama/Mistral 成立）。
                # Qwen3-32B config: num_heads=64, head_dim=128 → 8192，而
                # config.hidden_size=5120 不相等 → decode reshape 崩溃
                # (shape [1,1,5120] invalid for size 8192)。Qwen3-8B 恰好
                # 4096==4096 未暴露。此处注入 num_heads*head_dim（对
                # Llama/Mistral/Qwen3-8B 与 config.hidden_size 相同，零影响；
                # 对 32B 正确）。官方算法文件逐字节不动。
                module.hidden_size = module.num_heads * module.head_dim
                # 官方 forward 内部用 self.rotary_emb(value_states, position_ids)
                # （rotary_emb 实例在 _inject_rotary 阶段统一注入）
                module.token_budget = token_budget
                module.chunk_size = chunk_size
                module.flash_forward = module.forward
                module._official_forward = types.MethodType(quest_official_forward, module)
                module.forward = types.MethodType(_qwen3_quest_wrapper, module)
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    # 注入 model 顶层 rotary_emb（Qwen3 的 rotary 在 model 层，attention 无此属性）
    model_rotary = getattr(model, "rotary_emb", None)
    if model_rotary is None:
        for name, module in model.named_modules():
            if hasattr(module, "rotary_emb") and isinstance(module.rotary_emb, torch.nn.Module):
                model_rotary = module.rotary_emb
                break
    if model_rotary is None:
        raise RuntimeError("Qwen3 模型顶层无 rotary_emb，无法注入官方 Quest forward")

    def _inject_rotary(container):
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.rotary_emb = model_rotary
            elif len(list(module._modules.items())) > 0:
                _inject_rotary(module)

    _inject_rotary(model)
    _patch_recursive(model)
    return n_patched
