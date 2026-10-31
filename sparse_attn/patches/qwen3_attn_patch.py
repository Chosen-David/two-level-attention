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
from ..ops.operator_pipeline import decode_compact
from ..ops.partial_attention import partial_attention, merge_partials
# 【阶段 1（deepseek 轨）】batch>1 TLI 稀疏 decode：pred_ruler_batch 批量
# 入口在 decode 循环外层 set/clear BatchCtx（sparse_attn/baseline_runtime.py）。
from ..baseline_runtime import batch_decode_ctx


def _paper_route(layer,route):
    observer=sys.modules.get('exp.paper.receipt')
    if observer is not None:observer.record_qwen_route(layer,route)


def _can_drop_static_mask(cache, mask, length, *, prefill,
                          _chunk_elements=1 << 26):
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
        # torch.tril_ is incorrect on some CUDA builds once the matrix exceeds
        # 2**32 elements. Bounded chunks also avoid a second quadratic mask.
        rows = max(1, min(length, _chunk_elements // length))
        valid = True
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            chunk = prefix[..., start:stop, :]
            if not torch.equal(chunk, torch.ones_like(chunk).tril_(diagonal=start)):
                valid = False
                break
    else:
        valid = bool(prefix.all())
    valid = valid and not bool(mask[..., length:].any())
    cache._tli_mask_validation = ((weakref.ref(mask), signature, valid)
                                  if version is not None else None)
    return valid


# 【阶段 2c-5】capture 期常量张量缓存：torch.tensor([...], device=) 是
# host→device 拷贝，CUDA Graph capture 内禁止（"operation not permitted
# when stream is capturing"）→ 首调（warmup，capture 之外）构建后复用。
# key 只用 Python 值（device 字符串/S_cap），避免任何 tensor 读回同步。
_GRAPH_CONST_CACHE = {}


def _graph_const(key, build):
    t = _GRAPH_CONST_CACHE.get(key)
    if t is None:
        t = build()
        _GRAPH_CONST_CACHE[key] = t
    return t


def _batch_device_lengths(ctx, dev):
    """缓存 BatchCtx 的 seq_lens 为 device tensor（避免每步 host→device）。"""
    cached = getattr(ctx, '_tli_lengths_cache', None)
    if cached is not None and cached[0] is ctx:
        return cached[1]
    cached = (ctx, torch.tensor(ctx.seq_lens, dtype=torch.long, device=dev))
    ctx._tli_lengths_cache = cached
    return cached[1]


def _tl_decode_batched_graph(self, indexer, q, k, v, gs):
    """Graph decode single path: B1 serial compact, B>1 exact compact batch."""
    B = q.shape[0]
    dev = q.device
    S_cap = k.shape[1]
    vl = gs.valid_len_buf
    if vl is None or vl.numel() != B or vl.dtype != torch.long:
        raise ValueError("graph decode requires live int64 valid_len_buf[B]")
    if B == 1:
        cu = _graph_const(("cu_cap", str(dev), S_cap),
                          lambda: torch.tensor([0, S_cap], device=dev, dtype=torch.int32))
        output = decode_compact(indexer, indexer, q, k, v, cu, vl, self.scaling)
        indexer._graph_attention_route = 'graph_bs1_compact_tli'
        return output
    from ..ops.exact_linear_batch import exact_linear_batch_decode
    output = exact_linear_batch_decode(indexer, q, k, v, gs, self.scaling)
    indexer._graph_attention_route = 'graph_bsn_exact_compact_tli'
    return output


def _decode_batch_compact(self, indexer, q, k, v, lengths):
    """Eager batch compact path: selection and attention are both tensorized."""
    from ..indexer.tli_batch import prepare_selection_batched
    selection = prepare_selection_batched(indexer, q, k, lengths, self.scaling)
    packed = (selection.indices, selection.counts)
    kw = dict(sink=indexer.sink_tokens, swa=indexer.sliding_window_size,
              scale=self.scaling, splits=4, backend='simt')
    protected = partial_attention(q, k, v, packed, lengths,
                                  protected=True, **kw)
    selected = partial_attention(q, k, v, packed, lengths,
                                 protected=False, **kw)
    return merge_partials(protected, selected, q.dtype, backend='simt')


def _tl_decode_batched(self, indexer, query_states, key_states,
                       value_states, ctx):
    """Batch decode: graph uses the graph path, eager uses compact SIMT."""
    B = query_states.shape[0]
    if getattr(indexer, "dyn_dgate", False):
        indexer.freeze_dyn_dgate()
    q = query_states.transpose(1, 2)   # [B, 1, H, D]
    k = key_states.transpose(1, 2)     # [B, S_cap, Hkv, D]
    v = value_states.transpose(1, 2)   # [B, S_cap, Hkv, D]
    from ..ops.cuda_graph_state import graph_state
    gs = graph_state()
    if gs.enabled:
        return _tl_decode_batched_graph(self, indexer, q, k, v, gs)
    lengths = _batch_device_lengths(ctx, q.device)
    output = _decode_batch_compact(self, indexer, q, k, v, lengths)
    indexer._graph_attention_route = 'batch_eager_compact_tli'
    return output


def qwen3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    # Validate experimental selector routing before projections or any KV write.
    from ..ops.selector_options import validate_selector_options
    capacity = past_key_values.get_max_cache_shape() if past_key_values is not None else None
    validate_selector_options(capacity if isinstance(capacity, int) and capacity > 0 else None)
    if hasattr(self.indexer, "far_skip_backend") and not getattr(past_key_values, "paged_regions", False):
        raise ValueError(
            "TLI cache must be PagedRegionCache; use prefill_frozen_request")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Request-local segmented cache owns logical/physical position mapping.
    if getattr(past_key_values, "tli_regions", False):
        from ..ops.cuda_graph_state import graph_state
        ctx = batch_decode_ctx()
        if ((ctx is not None and ctx.batch != 1) or graph_state().enabled
                or os.environ.get("TLI_OPERATOR_MODE", "off") != "off"):
            raise ValueError("Qwen frozen regions require eager B1")
        if input_shape[0] != 1 or self.indexer.dyn_dgate:
            raise ValueError("Qwen frozen regions require B1 and independent region gates")
        layer = past_key_values.layers[self.layer_idx]
        if layer.logical_length:
            owned_mask = (attention_mask is None or (
                isinstance(attention_mask, dict)
                and set(attention_mask) == {"full_attention"}
                and attention_mask["full_attention"] is None))
            if input_shape[1] != 1 or not layer.frozen or not owned_mask:
                raise ValueError("Qwen frozen regions require frozen unpadded single-token decode")
            from ..ops.paged_region_tli import decode as region_decode
            result = region_decode(self, query_states, key_states, value_states,
                                   past_key_values, cache_position)
            return self.o_proj(result.reshape(*input_shape, -1).contiguous()), None

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
            # 【阶段 2a】batch decode（TLIStaticCache 固定容量、有效在前）：
            # 全宽 [B, Hkv, S_cap, D] 直接透传不切片——_tl_decode_batched 用
            # valid_length 区分有效区。cur_len 取 S_cap（恒 ≠1 → 正确落 decode
            # 分支；L177 的 prefill 判定 `cur_len == input_shape[1]` 不误触发）。
            # 单序列/非 batch 走下方原 E134-pre 变长切片（铁律零扰动）。
            _ctx = batch_decode_ctx()
            if _ctx is not None and input_shape[1] == 1:
                if not key_states.is_contiguous():
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()
                cur_len = key_states.shape[-2]
            else:
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
                # 路，见 _can_drop_static_mask 注释）。decode（q_len=1）不走
                # SDPA（eager_decoding_attn 自家 Triton kernel 吃 strided），
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
        elif (attention_mask is not None and is_static_cache
              and attention_mask.shape[-1] > key_states.shape[-2]):
            # KV above was trimmed to the live prefix. HF 5 SDPA no longer
            # trims a capacity-wide mask for us; retain every live mask value.
            attention_mask = attention_mask[..., :key_states.shape[-2]]
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
        _ctx = batch_decode_ctx()
        if _ctx is not None:
            attn_output = _tl_decode_batched(
                self, self.indexer, query_states, key_states,
                value_states, _ctx)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            _paper_route(self.layer_idx, "sparse-decode-batch")
            return attn_output, None
        if past_key_values is None:
            raise RuntimeError(
                "TLI sparse decode requires a KV cache (use_cache=True); "
                "past_key_values=None is unsupported")
        if input_shape[0] != 1 or input_shape[1] != 1:
            raise RuntimeError(
                f"TLI sparse decode requires batch=1 and q_len=1; got "
                f"batch={input_shape[0]}, q_len={input_shape[1]}")
        indexer: Indexer = self.indexer
        if getattr(indexer, "dyn_dgate", False):
            indexer.freeze_dyn_dgate()
        qs_t = query_states.transpose(1, 2)   # [1,1,H,D]
        ks_t = key_states.transpose(1, 2)     # [1,S,Hkv,D]
        vs_t = value_states.transpose(1, 2)
        dev = qs_t.device
        S_cap = ks_t.shape[1]
        vl = torch.tensor([cur_len], dtype=torch.int64, device=dev)
        cu = _graph_const(("cu_cap", str(dev), S_cap),
                          lambda: torch.tensor([0, S_cap], device=dev,
                                               dtype=torch.int32))
        attn_output = decode_compact(indexer, indexer, qs_t, ks_t, vs_t,
                                     cu, vl, self.scaling)
        _paper_route(self.layer_idx, "sparse-decode-compact")
        attn_weights = None
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
