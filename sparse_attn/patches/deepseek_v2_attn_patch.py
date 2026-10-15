"""【E142a】DeepSeek-V2 MLA 稀疏 attention patch（HF 路径 e2e 最小落地）。

背景（E141 trace 判决 e141_mla_nope_trace_verdict.json）：
  - nope-only 128d 在深层选择力崩塌（L20 .009-.229）——GQA tail 子空间
    结论不能按 nope-only 形式迁移；
  - rope-only 64d 与 full 192d 在全部可用层持平（≤.02）——选择子空间
    甜点 = MLA 布局尾段 rope 64 维（--tli_subspace mla_rope），且 k_pe 是
    MQA 共享 → 天然单索引（16 head 同选，indexer H=Hkv=16、G=1）；
  - MLA 质量 sink 主导（78-93%），L12/16/24 mid 质量精确为 0（天然层跳过
    ——E137 dyn gate 在 MLA 上的后续课题，本 patch 不启用）。

设计（镜像 qwen3_attn_patch 骨架 + vendored modeling_deepseek eager 投影
复刻，逐位同源）：
  - prefill（q_len == cache 长度，即首步完整 prompt）→ dense eager 原版
    复刻（4D float mask 相加语义保持）+ indexer.clear()；
  - decode（B=1、q_len=1、use_cache=True）→ 稀疏两级选择
    （prepare_mask）+ eager_decoding_attn(allow_kv_hetero=True)；
  - MLA 异维契约：k [1,S,H,192]（nope 128+rope 64）/ v [1,S,H,128]——
    选择只在 rope 64d 上做，注意力消费端 q·k(192) 与 p·v(128) 独立；
  - softmax_scale 读 self.softmax_scale（192^-0.5 × mscale²，模型属性）；
  - yarn RoPE 走模型自带 self.rotary_emb + 模型模块级
    apply_rotary_pos_emb（sys.modules[type(self).__module__] 同源取用，
    与 vendored 实现严格逐位一致——E141 probe 已验证口径）。

decode 期 attention_mask 契约：DSV2 走 model.generate，HF 传 4D float
causal mask（[1,1,1,S]，B=1 无 padding 下全 0 可加）。稀疏路径在选中
token 子集上做 causal-free softmax（选中的全是 ≤ 当前位置的 token——
sink/swa/near/far 分区几何保证），忽略该 mask（与 qwen3 mask=None 分支
同构）。B>1 / padding 形态显式拒绝（B2 守卫口径）。
"""

import sys

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from typing import Optional

from ..indexer import Indexer
from ..ops.eager_decoding import eager_decoding_attn
from .utils import (
    prepare_seqlens,
    prepare_cu_seqlens,
)


def _module_apply_rope(self):
    """从模块自身的模型实现里取 apply_rotary_pos_emb（trust_remote_code
    动态类不可静态 import——按 type(self).__module__ 同源取用，保证与
    vendored modeling_deepseek 的 rope 应用逐位一致）。"""
    mod = sys.modules.get(type(self).__module__)
    fn = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        raise RuntimeError(
            f"E142a: 模块 {type(self).__module__} 无 apply_rotary_pos_emb——"
            f"DeepSeek patch 需要模型自带 rope 实现（vendored "
            f"modeling_deepseek 同源），拒绝自造口径"
        )
    return fn


def deepseek_v2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    # ---- MLA 投影复刻（vendored modeling_deepseek eager forward 同构）----
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )

    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"E142a: {self.__class__.__name__} 无 layer_idx——cache "
                f"语义需要层索引，拒绝执行"
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    apply_rotary_pos_emb = _module_apply_rope(self)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ---- prefill/decode 分支（B2 守卫口径，镜像 qwen3_attn_patch）----
    if (past_key_value is not None
            and past_key_value.get_seq_length(self.layer_idx) == q_len
            and q_len > 1):
        # prefill：dense eager 原版复刻（4D float mask 相加语义保持）。
        # observe_prefill_q / calibrate_dyn_dgate 不调用——两者均为 Qwen3
        # 128 维 rotate_half 布局口径（E85f/E137），MLA 静态 pair 与动态
        # gate 的重校准是后续课题，本 patch 保持零 128 假设代码路径。
        indexer: Indexer = self.indexer
        indexer.clear()
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3))
            * self.softmax_scale
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, (attn_weights if output_attentions else None), past_key_value

    # decode 稀疏分支入口硬守卫（B2 口径）
    if past_key_value is None:
        raise RuntimeError(
            "E142a TLI 稀疏 patch 需要 KV cache（use_cache=True）；当前 "
            "past_key_value=None（训练/PPL/logprob 形态不支持稀疏索引"
            "路径），拒绝执行。"
        )
    if bsz != 1 or q_len != 1:
        raise RuntimeError(
            f"E142a TLI 稀疏 decode 路径仅支持 batch=1 且 q_len=1 的逐 "
            f"token 解码；当前 batch={bsz}、q_len={q_len}（chunked "
            f"prefill / batch>1 未实现），拒绝执行。"
        )
    # attention_mask：model.generate 传 4D float causal（B=1 无 padding
    # 下全 0 可加）——稀疏选择分区几何保证只选中 ≤ 当前位置的 token，
    # 该 mask 在稀疏路径为恒等，忽略（模块 docstring 契约）。

    indexer: Indexer = self.indexer
    if hasattr(indexer, "freeze_dyn_dgate"):
        indexer.freeze_dyn_dgate()

    # [B,H,S,D] → [1,S,H,D]（indexer 契约布局；MLA 下 H=Hkv=16、G=1）
    q_t = query_states.transpose(1, 2)   # [1, 1, H, 192]
    k_t = key_states.transpose(1, 2)     # [1, S, H, 192]
    v_t = value_states.transpose(1, 2)   # [1, S, H, 128]
    cu_seqlens_k = prepare_cu_seqlens(k_t)
    q_t, k_t, v_t = (x.view(1, -1, *x.shape[-2:]) for x in (q_t, k_t, v_t))
    query_position_ids = prepare_seqlens(cu_seqlens_k) - 1
    mask, block_size = indexer.prepare_mask(
        q_t,
        query_position_ids,
        k_t,
        cu_seqlens_k,
        softmax_scale=self.softmax_scale,
    )
    attn_output = eager_decoding_attn(
        q_t,
        k_t,
        v_t,
        mask,
        block_size,
        cu_seqlens_k,
        softmax_scale=self.softmax_scale,
        allow_kv_hetero=True,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = attn_output.contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value
