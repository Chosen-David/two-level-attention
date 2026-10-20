import os
import sys
import weakref

import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward
)

from typing import Optional, Callable
from ..indexer import Indexer
from ..ops.eager_decoding import eager_decoding_attn
# 【阶段 1（deepseek 轨）】batch>1 TLI 稀疏 decode：pred_ruler_batch 批量
# 入口在 decode 循环外层 set/clear BatchCtx（sparse_attn/baseline_runtime.py）。
from ..baseline_runtime import batch_decode_ctx
from .utils import (
    prepare_seqlens,
    prepare_cu_seqlens, 
    prepare_cu_seqlens_from_mask, 
    unpad_tensor
)


def _paper_route(layer,route):
    observer=sys.modules.get('exp.paper.receipt')
    if observer is not None:observer.record_qwen_route(layer,route)


def _can_drop_static_mask(cache, mask, length, *, prefill):
    """Only an exact, unpadded bool causal mask can be replaced by None.

    Cross-layer reuse is tied to the same live tensor and mutation version.
    Inference tensors without a version counter are validated on every call.
    A weak reference avoids retaining the quadratic prefill mask in the cache.
    """
    if (mask.dim() != 4 or mask.dtype != torch.bool or length < 1
            or mask.shape[:2] != (1, 1)
            or mask.shape[2] != (length if prefill else 1)
            or mask.shape[-1] < length):
        return False
    try:
        version = mask._version
    except RuntimeError:
        version = None
    signature = (version, length, prefill, tuple(mask.shape), tuple(mask.stride()), mask.dtype, mask.device)
    previous = getattr(cache, '_tli_mask_validation', None)
    if (version is not None and previous is not None
            and previous[0]() is mask and previous[1] == signature):
        return previous[2]
    prefix = mask[..., :length]
    if prefill:
        valid = torch.equal(prefix, torch.ones_like(prefix).tril_())
    else:
        valid = bool(prefix.all())
    valid = valid and not bool(mask[..., length:].any())
    cache._tli_mask_validation = ((weakref.ref(mask), signature, valid)
                                  if version is not None else None)
    return valid


def _tl_decode_batched(self, indexer, query_states, key_states,
                       value_states, ctx):
    """【阶段 1】TLI batch>1 稀疏 decode（deepseek 轨，正确性优先版）。

    right-pad KV 布局（pred_ruler_batch left-pad 协议）：cache 行 b 的
    有效 KV 占尾部 ctx.seq_lens[b] 列、pad 前缀在前。逐行切有效列后
    完全复用单序列 TLI 核心（prepare_mask + eager_decoding_attn），
    共享 indexer clear + 全量重建——无跨行增量状态，行间互不污染。
    全量 vs 单序列增量逐位一致（E133c：块统计/量化确定性归纳），故
    B=1 batch 模式与旧单序列入口逐位一致（gate_bs1_equiv 依据）。

    形状流：query_states [B,H,1,D]（cache.update 后 head-first）→
    q [B,1,H,D]；key/value_states [B,Hkv,S_total,D] → k/v
    [B,S_total,Hkv,D]；逐行 b：q_b [1,1,H,D]、k_b/v_b [1,L_b,Hkv,D]、
    cu=[0,L_b]、pos=[L_b-1]（该 query 行内绝对位置）。返回 [B,1,H,D]。
    """
    B = query_states.shape[0]
    q = query_states.transpose(1, 2)   # [B, 1, H, D]
    k = key_states.transpose(1, 2)     # [B, S_total, Hkv, D]
    v = value_states.transpose(1, 2)   # [B, S_total, Hkv, D]
    dev = q.device
    outs = []
    for b in range(B):
        L_b = ctx.seq_lens[b]
        q_b = q[b:b + 1]                    # [1, 1, H, D]
        k_b = k[b:b + 1, -L_b:]             # [1, L_b, Hkv, D]
        v_b = v[b:b + 1, -L_b:]             # [1, L_b, Hkv, D]
        cu = torch.tensor([0, L_b], device=dev, dtype=torch.int32)
        pos = torch.tensor([L_b - 1], device=dev, dtype=torch.long)
        indexer.clear()
        mask, block_size = indexer.prepare_mask(
            q_b, pos, k_b, cu, softmax_scale=self.scaling)
        outs.append(eager_decoding_attn(
            q_b, k_b, v_b, mask, block_size, cu,
            softmax_scale=self.scaling))
    return torch.cat(outs, dim=0)            # [B, 1, H, D]


def qwen3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cur_len = 0
    is_static_cache = False
    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # 【E134-pre】StaticCache 支持（opt-in）：update 返回全容量缓冲
        # （index_copy_ 就地写、零拷贝 append），下游统一切到当前序列长。
        # 判别用正的 get_max_cache_shape()（HF 4.56.2 DynamicCache 返回
        # -1；兼容无上限返回 None 的接口）。长度提示在 layer 0 每步刷新
        # 一次（cache_position[-1]+1，一次 .item 同步；StaticLayer.
        # get_seq_length 是 GPU any() 归约，逐层调用会引入 36 次隐式同步）。
        # HF 4.56.2 DynamicCache uses -1, not None, for unbounded capacity.
        capacity = past_key_values.get_max_cache_shape()
        is_static_cache = capacity is not None and capacity > 0
        if is_static_cache:
            if (self.layer_idx == 0
                    or getattr(past_key_values, "_tli_seq_len", None) is None):
                if cache_position is not None:
                    past_key_values._tli_seq_len = int(cache_position[-1]) + 1
                else:
                    past_key_values._tli_seq_len = past_key_values.get_seq_length(self.layer_idx)
            _S = past_key_values._tli_seq_len
            key_states = key_states[:, :, :_S, :]
            value_states = value_states[:, :, :_S, :]
            # 【E134-pre】prefill（q_len>1）期 contiguous 一次拷贝（每层
            # ~69MB×2，~7ms 总量）：切片是静态缓冲的 strided view
            # （head stride=cap·D≠S·D），pre 走 flash+enable_gqa 时 strided
            # 布局属后端接受性灰区，规整化一次买确定性（真慢源是 mask 旁
            # 路，见下方 _tli_pmask_ok 注释）。decode（q_len=1）不走 SDPA
            # （eager_decoding_attn 自家 Triton kernel 吃 strided），
            # 不拷贝零扰动。
            if input_shape[1] > 1 and not key_states.is_contiguous():
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            cur_len = _S
        else:
            cur_len = past_key_values.get_seq_length(self.layer_idx)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # 【B2 修复（kimi3 清单 F2，2026-10-08）】prefill/decode 分支判定守卫。
    # 原条件在三种走廊外形态下错误落 decode 分支或直接崩溃：
    #   ① use_cache=False（past_key_values=None，训练/PPL/logprob 形态）：
    #      None.get_seq_length → AttributeError；
    #   ② chunked prefill（带非空 past 的多 token 前向）：误判进 decode
    #      分支，eager_decoding_attn 只算 q[0] → token 1..tq-1 的输出
    #      恒 0，经 o_proj 后被残差掩盖，静默产出错误结果；batch>1 崩溃；
    #   ③ attention_mask 4D float（HF 实际形态）：prepare_cu_seqlens_
    #      from_mask 按 2D bool padding 假设实现，任何真实调用下不正确。
    # decode 分支入口见下方显式 raise（E109 生产路径 B=1、逐 token、
    # mask=None、use_cache=True 逐位不变）。
    if (past_key_values is not None
            and cur_len == input_shape[1]):
        indexer: Indexer = self.indexer
        indexer.clear()
        # 【E134-pre】prefill 期静态缓存 4D 容量 mask 旁路（profiler 归因
        # 2026-10-16：显式 mask 使 SDPA 脱离 flash/is_causal 快路径落
        # mem-efficient 慢路径 2.23ms/层 + 35.6ms/步 mask 拷贝链，L=32K
        # prefill +10s）。本分支 q_len==S==K/V 长度（分支条件本身），
        # 验证 [:S,:S] 下三角全真/上三角全假 + [S:] 全假后置 None，
        # SDPA 走 is_causal=True flash 路径，因果语义逐位等价。仅 bool
        # mask（4D float additive 不旁路，fail-closed 回原路径）；
        # sliding_window 激活时不旁路（SWA 语义编码在 mask 里）。
        # 同一 live mask + version 可跨层复用验证；不同/已修改 mask 重验。
        # 版本计数不可用的 inference tensor 不复用，避免静默吞掉新限制。
        if os.environ.get("TLI_E134_DEBUG") and self.layer_idx == 0 and attention_mask is not None:
            print(f"[E134dbg] L{self.layer_idx} mask dim={attention_mask.dim()} "
                  f"dtype={attention_mask.dtype} shape={tuple(attention_mask.shape)} "
                  f"sw={self.sliding_window} _S={cur_len}", flush=True)
        if (attention_mask is not None
                and self.sliding_window is None
                and is_static_cache
                and _can_drop_static_mask(past_key_values, attention_mask, cur_len, prefill=True)):
            attention_mask = None
        # E85f：prefill 走 dense，但旁路采集 post-RoPE q 统计供 per-layer 静态
        # pair 选取（decode 侧 _subspace_indices 消费；非静态 pair 模式为 no-op）
        if hasattr(indexer, "observe_prefill_q"):
            indexer.observe_prefill_q(query_states)
        # E137：动态 D' gate prefill 期标定（dense prefill 逐层一次）——
        # 尾行 far 概率质量 r_l < eps 的层 decode 期跳过 far（E60 已验证
        # prefill↔decode 层间 far mass corr 0.966~0.991）。默认关零扰动。
        if hasattr(indexer, "calibrate_dyn_dgate"):
            indexer.calibrate_dyn_dgate(query_states, key_states)
        # 稀疏 prefill（用户 2026-10-08 指令，TLI_SPARSE_PREFILL=1 门控）：
        # 默认未设置时走原 dense 分支逐位不变（E109 在跑链零扰动铁律）。
        # chunk 共享选择语义（MoBA 口径）见 ops/eager_prefill.py 模块注释。
        # 仅支持 B=1 且 attention_mask 为 None（无 padding）的质量路径口径，
        # 其余情形（含所有现状链）回退原 dense 分支。
        if (os.environ.get("TLI_SPARSE_PREFILL", "0") == "1"
                and attention_mask is None and input_shape[0] == 1):
            # 惰性 import：默认路径零依赖新增模块
            from ..ops.eager_prefill import sparse_prefill_attn
            q_t = query_states.transpose(1, 2)   # [1, L, H, D]
            k_t = key_states.transpose(1, 2)     # [1, S, Hkv, D]
            v_t = value_states.transpose(1, 2)
            attn_output = sparse_prefill_attn(
                indexer, q_t, k_t, v_t, softmax_scale=self.scaling
            )
            _paper_route(self.layer_idx,"sparse-prefill:"+os.environ.get("TLI_PREFILL_RUNTIME","reference"))
            attn_weights = None
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            _paper_route(self.layer_idx,"dense-prefill")
    else:
        # 【阶段 1（deepseek 轨）】TLI batch>1 稀疏 decode：pred_ruler_batch
        # 批量入口在 decode 循环外层 set BatchCtx。ctx 非 None 时走
        # _tl_decode_batched（逐行单序列核心 + 共享 indexer clear 全量
        # 重建，正确性最稳）。B=1 batch 模式与旧单序列入口逐位一致
        # （gate_bs1_equiv 依据）；非 batch 模式（ctx None）走既有路径
        # 逐位不变（铁律）。batch 分支独立于 TLI_DECODE_DENSE 消融
        # （消融仅单序列口径，batch 模式无意义）与 B2 守卫（B2 的
        # batch 拒绝在此分支之前已绕开；batch 分支自身要求
        # past_key_values 非 None，而 decode 分支进入条件已保证）。
        _ctx = batch_decode_ctx()
        if _ctx is not None:
            attn_output = _tl_decode_batched(
                self, self.indexer, query_states, key_states,
                value_states, _ctx)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx, "sparse-decode-batch")
            return attn_output, None
        # E138 消融①（prefill-only 稀疏，2026-10-14 用户指令）：TLI_DECODE_DENSE=1
        # 时 decode 显式回退 dense 原生前向——patch 保持挂载（prefill 稀疏
        # 选择消费 indexer），decode 不做稀疏选择。用于分离「prefill 稀疏
        # 的 KV 污染」与「decode 稀疏选择」两因素的贡献。默认不设 = 0 =
        # 现状稀疏 decode 逐位不变（B2 守卫/所有下游路径零扰动）。
        if os.environ.get("TLI_DECODE_DENSE", "0") == "1":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx,"dense-decode")
            return attn_output, attn_weights
        # 【B2 修复（kimi3 清单 F2，2026-10-08）】decode 分支入口硬守卫
        # （audit 建议口径）：use_cache=False / chunked prefill（q_len>1）/
        # batch>1 / 4D float mask 一律显式拒绝，替代静默零输出与形状崩溃。
        if past_key_values is None:
            raise RuntimeError(
                "TLI 稀疏 patch 需要 KV cache（use_cache=True）；当前 "
                "past_key_values=None（use_cache=False 形态，训练/PPL/"
                "logprob 打分不支持稀疏索引路径），拒绝执行。"
            )
        if input_shape[0] != 1 or input_shape[1] != 1:
            raise RuntimeError(
                f"TLI 稀疏 decode 路径仅支持 batch=1 且 q_len=1 的逐 token "
                f"解码；当前 batch={input_shape[0]}、q_len={input_shape[1]}"
                f"（chunked prefill / batch>1 会静默只算首 token 或形状"
                f"崩溃），拒绝执行。"
            )
        # 【E134-pre】StaticCache decode：HF 为静态缓存生成 4D bool 容量尺寸
        # causal mask（掩盖未写区）。K/V 上方已切到当前序列长 S=cur_len，
        # q_len=1 因果约束空转（单 query 只看 ≤ 自身位置的全部有效缓存）。
        # 验证「bool、[:S] 全真、[S:] 全假」，只对同一 live tensor/version
        # 复用；验证不过不吞掉，仍走下方原 B2 拒绝路径。
        # DynamicCache（mask=None）路径零扰动。
        if (attention_mask is not None and is_static_cache
                and self.sliding_window is None
                and _can_drop_static_mask(past_key_values, attention_mask, cur_len, prefill=False)):
            attention_mask = None
        if attention_mask is not None and (
            attention_mask.dim() != 2 or attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != (1, key_states.shape[-2])
            or not bool(attention_mask[0, -1])
        ):
            raise RuntimeError(
                f"TLI 稀疏 decode 路径的 attention_mask 须为 [1, S] 2D bool "
                f"padding mask，且当前 query（最后位置）须有效；当前 dim={attention_mask.dim()}、"
                f"dtype={attention_mask.dtype}（HF 实际传 4D float causal "
                f"mask，按 2D 假设处理不正确），拒绝执行。"
            )
        indexer: Indexer = self.indexer
        # A6（TL-E137-PREFILL-GATE-145）：decode 首步冻结 E137 动态标定决策。
        # prefill 期 calibrate_dyn_dgate 只写 _dyn_skip_pending（决策挂起），
        # 此处一次性落 skip_far（_dyn_frozen 哨兵防重复，clear() 复位两状态）
        # ——修复「完整 prompt 标定结果被稀疏 prefill chunk 消费」的非因果
        # 缺陷；dense prefill（默认路径）逐位不变。非 dyn_dgate 实例 no-op。
        if hasattr(indexer, "freeze_dyn_dgate"):
            indexer.freeze_dyn_dgate()
        query_states, key_states, value_states = (x.transpose(1, 2) for x in (query_states, key_states, value_states))
        if attention_mask is not None:
            cu_seqlens_k = prepare_cu_seqlens_from_mask(attention_mask)
            # Decode has one live query but S cached keys. The KV-length mask
            # cannot index Q; the guard above guarantees this query is valid.
            key_states, value_states = (unpad_tensor(x, attention_mask) for x in (key_states, value_states))
        else:
            cu_seqlens_k = prepare_cu_seqlens(key_states)
            query_states, key_states, value_states = (x.view(1, -1, *x.shape[-2:]) for x in (query_states, key_states, value_states))
        query_position_ids = prepare_seqlens(cu_seqlens_k) - 1
        mask, block_size = indexer.prepare_mask(
            query_states, 
            query_position_ids, 
            key_states, 
            cu_seqlens_k
        )
        attn_output = eager_decoding_attn(
            query_states,
            key_states,
            value_states,
            mask,
            block_size,
            cu_seqlens_k,
            softmax_scale=self.scaling,
        )
        _paper_route(self.layer_idx,"sparse-decode")
        attn_weights = None

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
