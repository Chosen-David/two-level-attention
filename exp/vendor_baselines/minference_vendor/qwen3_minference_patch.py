# -*- coding: utf-8 -*-
"""MInference 官方源码的 Qwen3 适配层（B 轨 vendor，复用官方 kernel 不改算法）。

本文件只做「接口胶水」：把 MInference 官方 prefill 稀疏前向
（minference_forward.py 的 `minference_prefill_forward` / `minference_prefill_kernel`，
与官方仓库逐字节一致）绑定到 Qwen3Attention，并显式区分 prefill / decode。
**核心算法（vertical+slash 动态稀疏、search_pattern、topk 索引、sparse attention）
完全来自官方源码，本层不重写任何选择逻辑。**

为什么不能直接用官方 `MInference(...).patch_model()`（new_patch）：
- new_patch 会替换 DecoderLayer/model.forward 为 **Llama 写死架构**（Qwen3 不兼容）；
- 会替换 `_prepare_cache_for_generation` 为 kvcompression 的
  `DynamicCacheWithRepeat`，Qwen3 generate 崩（`no attribute 'key_cache'`）；
- `attn_forward` 的 prefill/decode 判断 `q_len == past_key_value.get_seq_length(layer_idx)`
  对 Qwen3 首次 prefill（cache 空 get_seq_length=0、q_len>1）**误走 decode（dense flash）
  → MInference sparse 静默不生效 = 被误标 MInference**，违背 sanity 门禁。

本胶水只替换 Qwen3Attention.forward，wrapper 显式 `q_len > 1` 判 prefill（官方 sparse），
`q_len == 1` 判 decode（原生 dense，保存的 flash_forward），Qwen3 原生 cache/generate
流程保留。Qwen3 的 QK-Norm（q_norm/k_norm）在 prefill 分支原生复用（官方 kernel 直接
对已 norm 的 q/k 做点积，天然兼容）。

口径（对齐 MInference 论文）：prefill 动态稀疏 + decode dense KV；starting_layer 之前
的浅层 dense。is_search=True 时在线搜 Qwen3-8B 自己的 pattern（32 层×40 头）写 JSON，
is_search=False 时读 best_pattern 正式跑 sparse。

135/136（GPT 审计 B 批 fail-closed）：官方 kernel 按无 padding + Q_len==K_len
方阵 + 单段全量 prefill 协议假设——wrapper 在进入稀疏 prefill 分支前显式拒绝
非平凡 padding mask（TL-MINF-PADMASK-135）与 chunked prefill with prefix cache
（TL-MINF-CHUNKCACHE-136），不触碰官方 kernel 本体（宁拒绝不静默错输出）。
"""
import json
import os
from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from minference.modules.minference_forward import minference_prefill_forward


# ---- 135/136（TL-MINF-PADMASK-135 / TL-MINF-CHUNKCACHE-136，GPT 审计
# B 批）：vendor 适配层 fail-closed 守卫 ----
# 审计复现的两个静默错误输出路径（官方 kernel 均按「无 padding + Q_len==K_len
# 方阵 + 单段全量 prefill」协议假设，不校验也不报错）：
#   135：attention_mask（padding）进 prefill_kwargs 后稀疏层
#        （vertical_and_slash / block_sparse）与浅层 flash_attn_func 均不
#        消费 → padding 改变有效 token 输出（被 pad 行错读全 KV）；
#   136：仅按 q_len==1 分 decode，q_len>1 且 cache 已有 token（chunked
#        prefill with prefix cache）仍走稀疏 prefill，方阵协议错位。
# 本层只加显式拒绝（raise），不触碰官方 kernel 本体——静默错误输出比
# 硬失败更危险，宁拒绝不虚标。

def _has_nontrivial_padding(attention_mask):
    """135：检测 4D/2D attention mask 中超出固有因果结构的被屏蔽位置。

    4D mask（transformers 4.56 create_causal_mask 产物，[B, 1, q, kv]）
    自带因果上三角屏蔽，不能简单 `not mask.all()`（纯因果 eager prefill
    会误杀）。无 padding 的全量 prefill（q==kv）中最后一个 query 行全
    位置可见——只要最后一行存在被屏蔽位置即 padding（右 padding 屏蔽
    尾部列 / 左 padding 屏蔽前缀列，末行必见；左 padding 下前缀列对
    所有行均屏蔽）。mask 语义：float 加性（0=可见，非 0=屏蔽）或
    bool（True=可见，transformers 新 masking 标准 / torch SDPA 语义）。
    2D mask（[B, kv_len]，1/True=可见）罕见路径下按整张判定。
    """
    if attention_mask is None:
        return False
    m = attention_mask
    if not torch.is_tensor(m):
        return True  # 非张量 mask（罕见/未预期协议）——一律视为非平凡
    if m.dtype == torch.bool:
        masked = ~m          # 新 masking 标准 / SDPA：True=可见
    elif m.is_floating_point():
        masked = (m != 0)    # float 加性：0=可见，非 0（dtype min）=屏蔽
    else:
        masked = (m == 0)    # 老式 int 0/1 掩码（2D）：1=可见，0=屏蔽
    if m.ndim == 4:
        return bool(masked[:, :, -1, :].any().item())
    if m.ndim == 2:
        return bool(masked.any().item())
    # 3D/5D 等未预期形态——同样视为非平凡，交给调用方 raise
    return True


def _prefill_guard_violation(q_len, past_key_values, attention_mask,
                             layer_idx=0):
    """135/136 单一守卫入口：返回 None=放行，或 (审计编号, 中文报错)。

    136 检测点刻意在 past_key_values.update() 之前调用（此时
    get_seq_length() 是更新前的长度，等价于「进本层前 cache 已有
    token」——HF cache update 发生在 attention 内部，若更新后再判
    get_seq_length()>0 则对首段全量 prefill 也恒真，会误杀）。

    layer_idx（GPT advice 2026-10-13_1446 跨层误杀修复）：判定与报错
    读数都必须用「本层」的 get_seq_length(layer_idx)——transformers
    4.56 Cache API 无参调用 get_seq_length() 默认 layer_idx=0，而
    Qwen3 逐层解码循环共享同一个 DynamicCache：首段完整 prefill 时
    layer0 守卫放行、其 attention 内部 update() 写入后，layer>=1 再
    进守卫时无参调用会读到 layer0 的非零长度 → 首次 prefill 的
    layer>=1 全部被跨层误杀。报错消息中的长度读数同用本层值。
    """
    # 136：chunked prefill with prefix cache——本层更新前 cache 已有
    # token 且本次仍输入 q_len>1 → 下游 kernel 按 Q_len==K_len 方阵协议
    # 错位。（按本层 layer_idx 读更新前长度，共享 cache 下其他层已写入
    # 不构成对本层的 chunked 判定依据。）
    if q_len > 1 and past_key_values is not None \
            and past_key_values.get_seq_length(layer_idx) > 0:
        cached = past_key_values.get_seq_length(layer_idx)
        return ("TL-MINF-CHUNKCACHE-136",
                f"q_len={q_len} > 1 且 past_key_values 第 {layer_idx} 层"
                f"更新前已缓存 {cached} 个 token——chunked "
                f"prefill with prefix cache 不被 MInference vendor kernel "
                f"的 Q_len==K_len 方阵协议支持（稀疏层 vertical+slash 按方阵"
                f"取对角线，错位即静默错误输出），fail-closed 拒绝。请关闭 "
                f"prefix cache 或整段一次性 prefill。")
    # 135：非平凡 padding mask——稀疏层与浅层 flash 均不消费 mask。
    if q_len > 1 and _has_nontrivial_padding(attention_mask):
        return ("TL-MINF-PADMASK-135",
                "attention_mask 含超出固有因果结构的被屏蔽位置（padding）"
                "——非平凡 padding mask 不被 MInference 稀疏 prefill 支持"
                "（sparse kernel 与浅层 flash_attn_func 均不消费 mask，被 "
                "pad 行会错读全部 KV token 静默产出错误输出），fail-closed "
                "拒绝。请逐样本单条输入（无 padding）运行。")
    return None


def _qwen3_minference_wrapper(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Qwen3 接口 → MInference 官方 prefill sparse / 原生 decode dense。

    Qwen3DecoderLayer 期望 (hidden_states, attn_weights) 2 元组。
    prefill（q_len>1）复制官方 attn_forward 骨架但显式判定，调用
    minference_prefill_forward（内部 per-head sparse / flash）；decode 走原生。
    """
    q_len = hidden_states.shape[1]

    if q_len == 1:
        # decode 单 token：原生 dense（flash/sdpa，由 config._attn_implementation 决定）
        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    # ---- 135/136 fail-closed 守卫（详见 _prefill_guard_violation）----
    # 位置在 decode 分支之后、cache update 之前：136 须用「本层」更新前
    # seq_length 判定（layer_idx=self.layer_idx，跨层误杀修复），135 在
    # 进入稀疏 prefill 分支前拒绝 padding。
    violation = _prefill_guard_violation(
        q_len, past_key_values, attention_mask, layer_idx=self.layer_idx)
    if violation is not None:
        raise RuntimeError(
            f"[MINFERENCE-VENDOR-FAIL-CLOSED {violation[0]}] {violation[1]}")

    # ---- prefill：MInference 官方 sparse（复用 Qwen3 原生 qkv + QK-Norm + rope）----
    bsz, _, _ = hidden_states.size()
    hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    # GQA：Qwen3-8B 40 头 Q / 8 头 KV。官方 attn_forward 在调 prefill_forward 前把
    # KV repeat 到与 Q 同头数（minference_prefill_forward 逐 head 取 key_states[:, head]）
    if query_states.size(1) != key_states.size(1):
        key_states = repeat_kv(
            key_states, query_states.size(1) // key_states.size(1)
        )
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )

    prefill_kwargs = {
        "attention_mask": attention_mask,
        "layer_idx": self.layer_idx,
        "num_hidden_layers": self.config.num_hidden_layers,
        "attn_forward_config": self._minference_attn_config,
    }
    attn_output = minference_prefill_forward(  # [bsz, num_heads, q_len, head_dim]
        query_states, key_states, value_states, prefill_kwargs
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, heads, dim]
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def enable_minference(
    model,
    is_search: bool = False,
    config_path: str = None,
    starting_layer: int = -1,
) -> int:
    """把 model 内全部 Qwen3Attention 绑定 MInference 官方 prefill forward。

    - is_search=True：在线搜 Qwen3 自己的 pattern 写 config_path（需要先跑一次长
      prefill，官方 minference_prefill_forward 逐层逐 head search_pattern）。
    - is_search=False：从 config_path 读 best_pattern 正式跑 sparse。
    - starting_layer：之前浅层走 dense（官方默认 -1 = 全层 sparse）。
    返回 patch 层数。
    """
    n_patched = 0
    layer_id = 0

    def _patch_recursive(container):
        nonlocal n_patched, layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                # 官方 kernel 依赖的属性（Qwen3Attention 原生缺 num_heads 等顶层计数）
                module.layer_id = layer_id
                layer_id += 1
                module.num_heads = module.config.num_attention_heads
                module.num_key_value_heads = module.config.num_key_value_heads
                # 只在首次保存原生 forward（重复 enable 时 module.forward 已是
                # wrapper，覆盖会形成 decode 自递归）
                if not hasattr(module, "flash_forward"):
                    module.flash_forward = module.forward
                module.forward = _qwen3_minference_wrapper.__get__(module, Qwen3Attention)
                n_patched += 1
            elif len(list(module._modules.items())) > 0:
                _patch_recursive(module)

    # 注入 attn_forward_config：is_search=False 时读 best_pattern（list[层][head_id]
    # → (ty, vertical_size, slash_size, _)），minference_prefill_kernel 按
    # config["best_pattern"][layer_idx].get(str(head_id), default) 取用。
    if is_search:
        if config_path and os.path.isfile(config_path):
            os.remove(config_path)  # 重新搜索：清空旧 pattern
        attn_config = {
            "is_search": True,
            "starting_layer": starting_layer,
            "config_path": config_path,
        }
    else:
        assert config_path and os.path.isfile(config_path), (
            f"is_search=False 需要现有 best_pattern 文件: {config_path}"
        )
        with open(config_path, "r") as f:
            attn_config = {
                "is_search": False,
                "starting_layer": starting_layer,
                "config_path": config_path,
                "best_pattern": json.load(f),
            }

    _patch_recursive(model)
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention):
            module._minference_attn_config = attn_config
    return n_patched
