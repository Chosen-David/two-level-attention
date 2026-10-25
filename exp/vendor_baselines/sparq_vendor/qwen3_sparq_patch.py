"""SparQ 官方源码的 Qwen3 适配层（B 轨 vendor，复用官方算法不改）。

本文件只做「接口胶水 + 形状展开」，核心选择算法完全来自官方
sparq_attention_official.py（与 Graphcore sparq-gpt-fast 仓库的 sparq.py
逐字节一致，md5=522a085c...）：
  - `sparq_attn`：Stage1 用 |Q| 的 r 个最大坐标近似注意力分数 →
    top-k（k 个 token）→ Stage2 全维 QK 精确注意力 + V_mean lerp 插值。
  - RunningVMean：decode 时对 V 维护 running mean，用于 Stage3 插值
    （SparQ 论文核心：top-k 覆盖不到的 token 用 V 均值近似）。

与官方 gpt-fast 的差异（Qwen3 / transformers 4.56）：
  * 官方模型自管 DoubleKVCache/SingleKVCache + input_pos；Qwen3 用
    transformers DynamicCache，KV 存 kv_heads 粒度（GQA 不展开）。
  * 官方 `_generate` 里 V_mean 是 kv_heads 粒度，sparq_attn 的 lerp 期望
    与 y_ 同 H 粒度 —— 官方只在无 GQA 模型（Llama-2-7B）验证过；GQA
    模型 V_mean 直接传会 shape 不匹配。适配层在调官方 sparq_attn 前把
    V_mean repeat_kv 展开到 H（组内 G 头共享同一均值，语义不变）。
  * RunningVMean.init 官方接收展开后 V，shape 与 kv_heads 粒度 buffer
    不匹配 —— 适配层传 kv 粒度全量 V（mean(-2) 后形状恰好匹配）。

主表口径（用户 2026-10-14 拍板）：**k 固定 1024**（token budget 与
Quest/FASA 同口径）。r 由官方公式反推：
  ratio = context_length / (2*k)；r = round(head_dim / ratio)
Qwen3-8B head_dim=128 → 32K: r=8；128K: r=2（官方语义随序列变）。
"""
import types
import weakref
from typing import Tuple

import torch

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, repeat_kv,
)

from sparq_attention_official import (
    SparQArgs, RunningVMean, sparq_attn,
)


def _qwen3_sparq_wrapper(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Qwen3Attention forward：prefill 走原生；decode 走官方 SparQ 稀疏路径。"""
    # 【契约恢复 2026-10-23】B 轨重写（44a9868 RunningVMean 生命周期修复）
    # 时丢了 0fd6cac 集成版的计数/守卫契约——baseline_request receipt
    # 静默空壳（无 routes/identity，sparse-only 拒绝臂失效）。恢复：
    # batch=1 守卫 + native/sparq 路由计数 + decode 无掩码守卫。
    _bsz, _qlen_chk, _ = hidden_states.shape
    if _bsz != 1 or _qlen_chk < 1:
        raise ValueError('SparQ candidate supports batch=1 only')
    q_len = hidden_states.shape[1]

    if q_len > 1:
        # Native prefill may replace/refill the same cache object, even with a
        # length that aliases the previous decode. Invalidate before delegation
        # so a failed prefill cannot leave an apparently reusable running mean.
        self._sparq_mean_cache = None
        self._sparq_mean_length = None
        # prefill：原生 dense（与 fullkv 锚点一致；官方 SparQ prefill 也是 SDPA dense）
        _stats = getattr(self, '_sparq_stats', None)
        if _stats is not None:
            _stats['native_prefill_calls'] = _stats.get('native_prefill_calls', 0) + 1
        return self.flash_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, cache_position, **kwargs,
        )

    # decode 侧守卫（0fd6cac 契约）：无掩码单序列 + use_cache 必真——
    # padding/自定义掩码下稀疏选择语义不成立，fail closed。仅 cache 非 None
    # 时生效：cache=None 是 B 轨文档化的 V-mean 重建路径（无历史可选，
    # 单 token 重建，见 test_sparq_request_lifecycle），不属误用。
    if past_key_values is not None:
        from sparse_attn.baseline_runtime import require_unmasked_decode
        require_unmasked_decode(
            hidden_states, attention_mask, past_key_values, cache_position)
        if kwargs.get('use_cache', True) is not True:
            raise ValueError('SparQ decode requires use_cache=True')

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 投影 + Qwen3 per-head RMSNorm + rotary（与原生 prefill 一致）
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    new_v = value_states
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # 4.56 DynamicCache.update 返回全量 KV slice；RunningVMean.update 期望
        # 单 token 的 v，故须在 update 缓存前捕获本步新 token（decode q_len==1
        # 时为 [B, kv_heads, 1, D]）
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # ---- 官方 SparQ 稀疏路径（仅 decode，q_len==1）----
    # DynamicCache.update 返回全量 KV slice（[B, kv_heads, S, D]，GQA 未展开）
    full_k, full_v = key_states, value_states
    kv_len = full_k.shape[-2]

    # V_mean：首 decode 用含新 token 的全量 V init（等价官方 prefill init +
    # decode1 update，因 (S*mean_prev+v1)/(S+1) 恰为含 v1 的全量均值）；
    # 后续 update 本步新 token。kv 粒度维护，sparq_attn 前展开到 H。
    mean_cache = getattr(self, '_sparq_mean_cache', None)
    if (past_key_values is not None and mean_cache is not None
            and mean_cache() is past_key_values
            and getattr(self, '_sparq_mean_length', None) == kv_len - 1):
        self.sparq_vmean.update(new_v)          # [B, kv_heads, 1, D]
    else:
        self.sparq_vmean.init(full_v)          # [B, kv_heads, S, D]
    # Host metadata avoids n.item() (a CUDA synchronization). A weak reference
    # must not keep a finished request's potentially large cache alive. Unknown
    # or non-weak-referenceable cache implementations conservatively rebuild.
    try:
        self._sparq_mean_cache = weakref.ref(past_key_values)
    except TypeError:
        self._sparq_mean_cache = None
    self._sparq_mean_length = kv_len

    K = repeat_kv(full_k, self.num_key_value_groups)          # [B, H, S, D]
    V = repeat_kv(full_v, self.num_key_value_groups)          # [B, H, S, D]
    # 官方 RunningVMean buffer 是 fp32（register_buffer 默认）；官方 update()
    # 返回前 .to(v.dtype) 转回模型 dtype。此处直接读 buffer 须手动转，否则
    # torch.lerp(input=fp32, end=bf16) dtype 不匹配崩溃。
    V_mean = repeat_kv(
        self.sparq_vmean.V_mean.to(query_states.dtype),
        self.num_key_value_groups)

    # decode 单 token：全历史可见
    mask = torch.ones(1, 1, 1, kv_len, dtype=torch.bool,
                      device=query_states.device)

    attn_weights = sparq_attn(
        query_states,            # [B, H, 1, D]
        K, K,                    # K1 == K2 == K（官方 DoubleKVCache 双存 K/K^T，数值相同）
        V,
        V_mean,                  # [B, H, 1, D]
        mask,
        self.sparq_r,
        self.sparq_k,
        self.sparq_config,
    )
    attn_output = attn_weights.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    _stats = getattr(self, '_sparq_stats', None)
    if _stats is not None:
        _stats['sparq_decode_calls'] = _stats.get('sparq_decode_calls', 0) + 1
    return attn_output, None


def enable_sparq(
    model,
    k: int = 1024,
    context_length: int = 32768,
) -> int:
    """把 model 内全部 Qwen3Attention 绑定官方 SparQ 稀疏 forward。返回 patch 层数。

    r/k 主表口径：k 固定（默认 1024）；r = round(head_dim/ratio)，
    ratio = context_length/(2*k)（官方 get_r_k_for_compression_ratio 反推）。

    【契约恢复 2026-10-23】入参守卫 + 路由计数 + treatment 身份
    （0fd6cac 集成版契约，B 轨重写时丢失，本文件恢复）。
    """
    # 入参守卫：k/context_length 须正整数（type() 严格判 → bool 拒绝，
    # float 截断风险拒绝——与 _quota_integer 同 fail-closed 口径）
    if type(k) is not int or k <= 0 or type(context_length) is not int \
            or context_length <= 0:
        raise ValueError('SparQ k/context_length must be positive integers')
    if any(hasattr(m, 'flash_forward') for m in model.modules()
           if isinstance(m, Qwen3Attention)):
        raise ValueError('SparQ already installed')
    n_patched = 0
    layer_id = 0
    _ratio = context_length / (2 * k)
    # 路由计数共享 dict（全部层同一对象；baseline_request diff 消费）
    stats = {}

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.layer_id = layer_id
                layer_id += 1
                module.sparq_r = max(1, round(module.head_dim / _ratio))
                module.sparq_k = k
                module.sparq_config = SparQArgs()
                module.sparq_vmean = RunningVMean(
                    max_batch_size=1,
                    n_local_heads=module.config.num_key_value_heads,
                    head_dim=module.head_dim,
                )
                # RunningVMean 的 V_mean/n buffer 默认建在 CPU（torch.full/tensor
                # 默认 CPU），须随模型移到 CUDA，否则 sparq_attn 内 lerp 时报
                # "Expected all tensors on same device, cuda:0 vs cpu"。
                module.sparq_vmean.to(module.q_proj.weight.device)
                module.flash_forward = module.forward
                module.forward = types.MethodType(_qwen3_sparq_wrapper, module)
                module._sparq_stats = stats
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    _patch_recursive(model)
    if n_patched == 0 or n_patched != model.config.num_hidden_layers:
        raise ValueError('SparQ adapter supports complete Qwen3 attention models only')
    # treatment 身份 + 路由计数登记（baseline_request receipt 契约）
    model._baseline_stats = stats
    import hashlib
    from pathlib import Path as _P
    model._baseline_identity = dict(
        method='sparq_vendor',
        adapter_sha256=hashlib.sha256(
            _P(__file__).read_bytes()).hexdigest(),
        upstream='official sparq_attention_official.py (md5 522a085c)',
        k=k, context_length=context_length,
        selection='per-query-head; running-V-mean compensation; full resident KV',
        gpu_quality_validated=False)
    return n_patched
