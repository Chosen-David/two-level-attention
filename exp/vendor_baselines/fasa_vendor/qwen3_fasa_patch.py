"""FASA 官方源码的 Qwen3 适配层（B 轨 vendor，复用官方核心算法不改）。

本文件只做「接口胶水 + 剪枝接线」：
  - fasa_utils.py = 官方 utils.py 逐字节一致（constructe_adaptive_selection /
    core_module_with_padding_old，完全模型无关，本层不重写任何选择逻辑）。
  - 官方 qwen_df.py 面向 Qwen2 Flash（transformers 4.43，Qwen2FlashAttention2
    forward 重写，decode 时在 repeat_kv 后调 core_module_with_padding_old）。
  - Qwen3（transformers 4.56）差异：
      * 单类 Qwen3Attention 统一（eager/sdpa/flash），kv 不预展开
        （attention_interface 内部才 repeat_kv）；
      * 故 decode 剪枝前须手动 repeat_kv 展开；展开后的 kv 再喂
        attention_interface 会被二次展开 —— 因此 decode 剪枝分支
        自己算 eager attention（q_len==1、kv==budget，计算量小且等价）。
      * prefill（q_len>1）保持原 forward 原样（不剪枝，dense）。
  - 剪枝口径：layer_idx>0（前 1 层 dense，对齐官方 qwen_df.py 的
    `query_states.shape[2] == 1 and self.layer_idx > 0`）。
"""
import os
import types
from typing import Tuple

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, eager_attention_forward, repeat_kv,
)

from fasa_utils import constructe_adaptive_selection, core_module_with_padding_old


def _qwen3_decode_eager(query, key, value, attention_mask, scaling, sel_idx=None):
    """decode 剪枝后的 eager attention（避免 attention_interface 二次 repeat_kv）。

    sel_idx（batch 模式特有）：[B, num_heads, k] 每行每 head 选中 token 的稠密
    KV 列位置。非 None 时 attention_mask 按选中列 gather（topk 选中 token 乱序、
    pad 列需按实际选中位置取 mask）；None 时走官方 slice 前 k 列（legacy
    batch=1 全 1 mask 恒等）。
    """
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        if sel_idx is not None:
            B, H = query.shape[0], query.shape[1]
            mask_b = attention_mask.expand(B, H, 1, -1)  # [B, H, 1, S]
            causal_mask = mask_b.gather(
                dim=-1,
                index=sel_idx.unsqueeze(2).expand(B, H, 1, -1))  # [B, H, 1, k]
        else:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _fasa_batch_core(query_states, key_states, value_states, layer_idx,
                     budget, records, valid_mask):
    """语义复现官方 core_module_with_padding_old，支持 batch（left-pad）。

    官方函数（fasa_utils.py:101-138）bs 维贯穿本身 batch 通用，唯一 batch 差异
    是稠密 KV 含 left-pad 的 pad 列：本函数在 attn_weights 对 pad 列置 -inf
    再 topk，天然排除 pad（官方 batch=1 无 pad，此步恒等）。valid_mask:
    [B, 1, S] bool（True=真实 token）。额外返回 sel_idx [B, H, k]（每行每
    head 选中 token 的稠密 KV 列位置，供 _qwen3_decode_eager 按选中列 gather
    attention_mask）。bs=1 且无 pad 时与官方输出逐位一致（-inf 加法零影响）。
    """
    bs, num_heads, query_len, head_dim = query_states.shape
    _, _, seq_len, _ = key_states.shape
    preceding_token_num = int(budget) if budget >= 1 else int(seq_len * budget)
    preceding_token_num = min(preceding_token_num, seq_len)
    head_selections = records[layer_idx]["head_selection"].to(query_states.device)
    selection_masks = records[layer_idx]["selection_masks"].to(query_states.device)
    selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(
        bs, -1, query_len, -1)
    query_selected = torch.gather(query_states, dim=-1, index=selection_expanded)
    key_selection_expanded = head_selections.unsqueeze(0).unsqueeze(2).expand(
        bs, -1, seq_len, -1)
    key_selected = torch.gather(key_states, dim=-1, index=key_selection_expanded)
    selection_masks_expanded_query = selection_masks.unsqueeze(0).unsqueeze(2).expand(
        bs, -1, query_len, -1)
    selection_masks_expanded_key = selection_masks.unsqueeze(0).unsqueeze(2).expand(
        bs, -1, seq_len, -1)
    query_selected = torch.where(selection_masks_expanded_query, query_selected, 0)
    key_selected = torch.where(selection_masks_expanded_key, key_selected, 0)
    attn_weights = torch.matmul(query_selected, key_selected.transpose(2, 3))
    if valid_mask is not None:
        # batch：pad 列 -inf 排除（官方 batch=1 无 pad，此步恒等）
        neg_inf = torch.finfo(attn_weights.dtype).min
        pad_mask = ~valid_mask  # [B, 1, S]
        attn_weights = attn_weights + (
            pad_mask[:, None, :, :].to(attn_weights.dtype) * neg_inf)
    top_indices = torch.topk(
        attn_weights, k=preceding_token_num, dim=-1).indices  # [B,H,q_len,k]
    sel_idx = top_indices[..., 0, :]  # [B, H, k]（query_len==1）
    top_indices = top_indices.transpose(2, 3).expand(-1, -1, -1, head_dim)
    key_output = torch.gather(key_states, dim=2, index=top_indices)
    value_output = torch.gather(value_states, dim=2, index=top_indices)
    return key_output, value_output, sel_idx


def _qwen3_fasa_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Qwen3Attention forward：prefill 走原逻辑；decode 剪枝 + 自算 attention。"""
    q_len = hidden_states.shape[1]
    if q_len > 1 or self.layer_idx == 0:
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, cache_position, **kwargs,
        )

    from sparse_attn.baseline_runtime import (
        require_unmasked_decode, batch_decode_ctx, batch_valid_mask)
    require_unmasked_decode(hidden_states, attention_mask, past_key_values, cache_position)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # FASA 剪枝：展开 GQA kv → budget 个 token（batch 模式 pad 列 -inf 排除）
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    _ctx = batch_decode_ctx()
    if _ctx is None:
        # legacy 单序列：官方 core_module_with_padding_old 逐字（零改动）
        key_states, value_states = core_module_with_padding_old(
            query_states, key_states, value_states,
            self.layer_idx, self.fasa_budget, self.fasa_records)
        sel_idx = None
    else:
        # batch：语义复现官方 + left-pad 排除 + 按选中列 mask
        valid_mask = batch_valid_mask(key_states, _ctx)
        key_states, value_states, sel_idx = _fasa_batch_core(
            query_states, key_states, value_states,
            self.layer_idx, self.fasa_budget, self.fasa_records, valid_mask)

    attn_output, attn_weights = _qwen3_decode_eager(
        query_states, key_states, value_states, attention_mask, self.scaling,
        sel_idx=sel_idx)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def enable_fasa(
    model,
    agreement_dir: str,
    model_name: str,
    dataset_name: str = "qasper",
    budget: int = 1024,
    threshold_ratio: float = 0.8,
    min_k: int = 16,
    max_k: int = 16,
    mode: str = "topk",
) -> int:
    """把 model 内全部 Qwen3Attention 绑定 FASA 剪枝 forward。返回 patch 层数。

    agreement_dir = 内容寻址 staged 根（sparse_attn/info.py 的
    stage_fasa_agreement 返回值，其下有 dimension_rankings/<model>/
    <dataset>_agreement.pkl）。官方 constructe_adaptive_selection 硬编码
    相对路径 dimension_rankings/（相对 CWD），故此处把 CWD 切到
    agreement_dir 后再调官方函数（try/finally 恢复，不污染进程后续
    相对路径操作）——官方相对路径即解析到内容寻址副本。

    142 并发残余修复（C+ 批）：不同校准内容 stage 到不同
    dimension_rankings_staged/<sha8>/ 目录，跨进程「A 先 replace、B 后
    replace 覆盖、A 的官方 loader 读到 B」结构性不可能；旧仓库根
    dimension_rankings/ 共享固定路径不再被本路径读到（历史产物原地
    保留，与本函数无关）。"""
    cwd_bak = os.getcwd()
    os.chdir(agreement_dir)
    try:
        records = constructe_adaptive_selection(
            model_name, budget, threshold_ratio, min_k, max_k,
            dataset_name=dataset_name,
            mode=mode,
        )
    finally:
        os.chdir(cwd_bak)
    n_patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.layer_id = layer_id
                layer_id += 1
                module.fasa_budget = budget
                module.fasa_records = records
                module.flash_forward = module.forward
                module.forward = types.MethodType(_qwen3_fasa_forward, module)
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    _patch_recursive(model)
    return n_patched
