# -*- coding: utf-8 -*-
"""JSSA 官方 vLLM backend 语义的 Qwen3 适配层（B 轨官方源码语义，torch 等价）。

边界（2026-10-18，deepseek 轨）：
  JSSA = Joint Subspace Sparse Attention（arXiv:2610.19026v1）。官方无独立
  开源仓库，实现以 vLLM patch 形式提供（third_party/JSSA/vllm_patch/
  jssa_vllm.patch，vLLM v1 attention backend jssa_attn.py + 三个 Triton ops）。
  语义已在 research/docs/JSSA_子空间对比分析_20261017_by_deepseek.md §2/§4
  精读确认可提取：
    - 校准基 M = TopEig_r(G_Qm/tr + G_K/tr)（r=32，jssa_calibrate.py 产出
      {'M': {layer_idx: [H_kv, r, d]}} .pt）
    - decode 单级 token 稀疏：GQA 组平均 q → proj_q = q_kv·M^T → 对全 K 投影
      子空间打分 top-B/head → gather-free 精确注意力（budget = 每 kv head
      top-B token）
    - prefill dense（backend 仅 route DECODER attn，prefill 走 dense）
  本层按该官方语义用纯 torch 实现 Qwen3 decode 等价选择器
  （统一后端机制轨，标签 jssa_vendor，不冒称 vLLM 原生 kernel）。

Qwen3 适配要点（差异披露）：
  - M 为 bf16（官方 vLLM 路径 bf16 basis；FP8 仅投影值存储，bf16 等价忽略
    fp8 量化误差，identity 标注）。
  - proj_K 每次 decode 对全 K 现算（官方在 KV 存储时预投影成 FP8 proj_k 缓存，
    语义等价；存储优化留作后续）。
  - radix top-k 用 torch.topk 精确替代（近似→精确，identity 标注）。
  - 无 sink/swa 强制区（官方 JSSA 无强制保留，同 Quest 口径，全预算自由竞争）。
  - 全层稀疏（basis 缺失层回退 dense，同官方 backend 行为）。
"""
import os
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, repeat_kv,
)


def _load_basis(basis_path: str, device, dtype) -> dict:
    """加载校准基 {'M': {layer_idx: [H_kv, r, d]}}；无 basis_path 时抛错。"""
    if not basis_path or not os.path.exists(basis_path):
        raise ValueError(
            f"JSSA basis 文件不存在: {basis_path!r}（先跑 third_party/JSSA/calibration/"
            f"jssa_calibrate.py 产出，或传 --jssa_basis_path）")
    ckpt = torch.load(basis_path, map_location="cpu", weights_only=False)
    M = ckpt["M"] if isinstance(ckpt, dict) and "M" in ckpt else ckpt
    if not isinstance(M, dict):
        raise ValueError(f"JSSA basis {basis_path} 不含 {'M': {layer_idx: tensor}} dict")
    return {int(k): v.to(device=device, dtype=dtype).contiguous() for k, v in M.items()}


def _qwen3_jssa_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Qwen3Attention forward：prefill（q_len>1）走原逻辑；decode（q_len=1）
    走 JSSA 单级投影子空间 top-k 稀疏。"""
    q_len = hidden_states.shape[1]
    if q_len > 1:
        # Transformers 5.12 删除了 cache_position 的显式位置参数；以关键字
        # 传递可同时兼容新旧签名。
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, **kwargs,
        )

    if cache_position is None and past_key_values is not None:
        # Transformers 5.x 不向 attention 透传 cache_position；RoPE 仍使用
        # 上层 position_embeddings，这里补齐 cache 更新与契约校验的位置。
        previous_length = int(past_key_values.get_seq_length(self.layer_idx))
        cache_position = torch.arange(
            previous_length, previous_length + q_len,
            device=hidden_states.device)

    from sparse_attn.baseline_runtime import require_unmasked_decode
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

    L = key_states.shape[2]
    H_kv = key_states.shape[1]
    n_qh = query_states.shape[1]
    d = key_states.shape[-1]
    G = n_qh // H_kv
    n_fac = min(self.jssa_budget, L)

    # ---- 1. GQA 组平均（官方 q_kv = mean over G of query heads）----
    q_g = query_states[0, :, 0, :].reshape(H_kv, G, d)     # [H_kv, G, d]
    q_kv = q_g.mean(dim=1)                                  # [H_kv, d]

    # ---- 2-3. 投影子空间打分（官方 proj_q·proj_K，全 K）----
    M = self.jssa_M.to(device=query_states.device,
                        dtype=query_states.dtype)          # [H_kv, r, d]
    proj_q = torch.einsum("hd,hcd->hc", q_kv, M)            # [H_kv, r]
    proj_K = torch.einsum("hld,hcd->hlc", key_states[0], M)  # [H_kv, L, r]
    scores = torch.einsum("hc,hlc->hl", proj_q, proj_K)     # [H_kv, L]

    # ---- 4. per-kv-head top-B token（官方 radix top-k 精确等价）----
    top_idx = torch.topk(scores, k=n_fac, dim=-1).indices   # [H_kv, n_fac]

    # ---- 5. 精确注意力（组内 query head 共享 kv head 的 top_idx）----
    k_sel = key_states[0].gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, d))         # [H_kv, n_fac, d]
    v_sel = value_states[0].gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, d))
    k_sel = repeat_kv(k_sel.unsqueeze(0), G)                # [1, n_qh, n_fac, d]
    v_sel = repeat_kv(v_sel.unsqueeze(0), G)
    attn_weights = torch.matmul(query_states, k_sel.transpose(-2, -1)) * self.scaling
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, v_sel)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def enable_jssa(
    model,
    basis_path: str,
    budget: int = 1024,
    rank: int = 32,
) -> int:
    """把 model 内全部 Qwen3Attention 绑定 JSSA 单级稀疏 forward。返回 patch 层数。

    basis_path: jssa_calibrate.py 产出 .pt（{'M': {layer_idx: [H_kv, r, d]}}）。
    budget: 每 kv head top-B token 预算（主表口径 K2=1024）。
    rank: 投影子空间维数（官方默认 head_size//4 = 32 for d=128）。
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("JSSA budget must be a positive integer")
    if type(rank) is not int or rank <= 0:
        raise ValueError("JSSA rank must be a positive integer")
    # 取任一 Qwen3Attention 的 device/dtype 预加载 basis（延迟到首个 forward 亦可）
    probe = None
    for mod in model.modules():
        if isinstance(mod, Qwen3Attention):
            probe = mod
            break
    if probe is None:
        raise ValueError("model 中未找到 Qwen3Attention")
    M_dict = _load_basis(basis_path, probe.q_proj.weight.device, probe.q_proj.weight.dtype)

    n_patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.layer_id = layer_id
                layer_id += 1
                module.jssa_budget = budget
                module.jssa_rank = rank
                M = M_dict.get(module.layer_id)
                if M is None:
                    module.jssa_M = None
                else:
                    # HF Qwen3Attention 无 num_heads/num_key_value_heads 属性；
                    # H_kv 从 k_proj 权重行数推断（[H_kv*head_dim, hidden]）。
                    H_kv = module.k_proj.weight.shape[0] // module.head_dim
                    expected = (H_kv, rank, module.head_dim)
                    if tuple(M.shape) != expected:
                        raise ValueError(
                            f"JSSA basis M[{module.layer_id}] shape {tuple(M.shape)} "
                            f"!= expected {expected}")
                    module.jssa_M = M
                module.flash_forward = module.forward
                module.forward = types.MethodType(_qwen3_jssa_forward, module)
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    _patch_recursive(model)
    return n_patched
