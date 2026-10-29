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


def _parallel_batch_enabled(batch):
    """TLI_BATCH_INDEXER=parallel 开关（legacy=默认逐行路径）。"""
    mode = os.environ.get("TLI_BATCH_INDEXER", "legacy")
    if mode not in ("legacy", "parallel", "exact"):
        raise ValueError("TLI_BATCH_INDEXER must be legacy, parallel or exact")
    return batch >= 1 and mode == "parallel"


def _tl_decode_batched_graph(self, indexer, q, k, v, gs):
    """【阶段 2c-4】graph 数据通路版 batch decode（全 tensor、无 host 分支）。

    与 staged 路径（`_tl_decode_batched`）的唯一区别在「动态几何如何取」：
      - staged：mask 由 prepare_mask(valid_length=L_b) 产生，L_b 是 Python
        int（capture 时会被烘焙 → replay 陈旧，不可图化）；
      - graph：mask 由 compute_mask_graph 从 valid_len tensor 派生（每行
        prepare_index 容量宽一次 → 全部块统计/k_qat 就绪），attn gather
        不传 valid_length（只消费 mask）。

    于是整步 forward 里所有步间变化量（KV 内容 / valid_len）均由 tensor
    内容承载 → 可 capture 一次 + replay 循环 copy_ 注入（2c-5）。

    逐位一致性：mask ≡ staged prepare_mask 的有效区（G8 门禁 tail/full×
    8 几何逐位）；attn 归约集合 ≡（gather 路径 valid_length 不参与，
    T≥16K 走 Triton / T<16K 走 dense，pad 列由 mask False 排除）。
    """
    if os.environ.get('TLI_BATCH_INDEXER', 'legacy') == 'exact':
        from ..ops.exact_linear_batch import exact_linear_batch_decode
        return exact_linear_batch_decode(indexer,q,k,v,gs,self.scaling)
    B = q.shape[0]
    dev = q.device
    S_cap = k.shape[1]
    vl = gs.valid_len_buf
    if vl is None or vl.numel() != B:
        if _parallel_batch_enabled(B):
            raise ValueError("parallel graph requires valid_len_buf[B]")
        vl = torch.zeros(B, dtype=torch.long, device=dev)
        gs.valid_len_buf = vl
    from ..ops.operator_pipeline import operator_mode, decode_indexed
    if operator_mode() != 'off':
        if _parallel_batch_enabled(B):
            raise ValueError('operator pipeline currently requires TLI_BATCH_INDEXER=legacy')
        if B != 1:
            raise ValueError('use HybridBatchRunner for multi-request operator decode')
        cu = _graph_const(("cu_cap", str(dev), S_cap),
                          lambda: torch.tensor([0, S_cap], device=dev, dtype=torch.int32))
        return decode_indexed(indexer, indexer, q, k, v, cu, vl, self.scaling)
    if _parallel_batch_enabled(B):
        from ..indexer.tli_batch import prepare_mask_batched
        if gs.valid_len_buf is None:
            raise ValueError("parallel graph requires live valid_len_buf")
        mask, bs = prepare_mask_batched(indexer, q, k, vl, self.scaling)
        _paper_route(self, "tli_batch_indexer_parallel_graph")
        return eager_decoding_attn(
            q, k, v, mask, bs, indexer._batch_tli.cu,
            softmax_scale=self.scaling, valid_length=None)
    cu = _graph_const(("cu_cap", str(dev), S_cap),
                      lambda: torch.tensor([0, S_cap], device=dev,
                                           dtype=torch.int32))
    masks = []
    for b in range(B):
        # 【E152v4】不再每步 clear()：clear 是请求边界标记（prefill 前每层
        # 调用，失效 _inc/_ginc → 新请求首步全量重建）。graph 步进下每步
        # clear 会把 _ginc 持久容量宽 buffer 一起失效 → 退化回每步
        # _prepare_index_base 容量宽全量重建（with_stack 归因 ~15ms/步
        # 36 层 index 税）。prepare_index 内部在 cuda_graph_state 模式下
        # 分发到 pos_buf 驱动的增量更新（单块统计 + 单行 k_qat，固定形状
        # 纯张量 op，capture 安全）；不兼容配置自动回退全量路径（fail-
        # closed，行为逐位不变）。
        idx_dict = indexer.prepare_index(k[b:b + 1], cu)
        masks.append(indexer.compute_mask_graph(
            q[b:b + 1], idx_dict, self.scaling, vl[b:b + 1]))
    m_b = torch.cat(masks, dim=0)              # [B,1,Hkv,S_cap] bool
    cu_b = torch.arange(B + 1, device=dev, dtype=torch.int32) * S_cap
    # valid_length=None：mask 已排除 pad 列（prepare_mask 末尾清 False 的
    # graph 等价物是 compute_mask_graph 的 `sel & (ar_tok < S_real)`）。
    return eager_decoding_attn(
        q, k, v, m_b, 1, cu_b, softmax_scale=self.scaling, valid_length=None)


def _tl_decode_batched(self, indexer, query_states, key_states,
                       value_states, ctx):
    """【阶段 2a】TLI batch>1 固定宽度稀疏 decode（有效在前 left-pad KV 布局）。

    阶段 1 消费 DynamicCache 变长 KV（有效占尾、`-L_b:` 尾部切片）；阶段
    2a 切换 TLIStaticCache 固定容量 S_cap（pred_ruler_batch 构建）：行 b
    有效 KV 占 [0, L_b)、[L_b, S_cap) 为未写 pad 区（零）。逐行 b：
      - k_b/v_b = k[b:b+1] 全宽 [1, S_cap, Hkv, D]（不再变长尾部切片）；
      - cu = [0, S_cap]（固定）；pos = [L_b-1]（该 query 行内绝对位置）；
      - prepare_mask(..., valid_length=L_b) → mask [1,1,Hkv,S_cap]，有效区
        [0, L_b) 与变长路径逐位一致、pad 区全 False（E155e fixed_width
        探针已证）；
      - eager_decoding_attn：gather 路径只选中有效区（pad False 不选）、
        cu 契约 [0,S_cap] 成立 → 输出与变长路径逐位一致（同 token 集、
        同 q·k，归约顺序同）。固定 S_cap 形状 = CUDA Graph 前置。
    全量 vs 单序列增量逐位一致（E133c 归纳）保持，B=1 batch 模式与旧
    单序列入口逐位一致（gate_bs1_equiv 依据）。

    形状流：query_states [B,H,1,D] → q [B,1,H,D]；key/value_states
    [B,Hkv,S_cap,D] → k/v [B,S_cap,Hkv,D]。返回 [B,1,H,D]。

    【阶段 2b 批量张量化】B>1 且 block_size==1（TLI 生产 token 级 mask）时
    走批量路径：逐行 prepare_mask（indexer 状态逐行，mask 内容与逐行路径
    逐位相同）→ cat 成 [B,1,Hkv,S_cap] → 一次批量 eager_decoding_attn
    （Triton 批量 kernel A/B 逐 (b,r) program 独立归约、仅 batch 维寻址，
    与逐行单 query kernel 逐位一致——dbg_batch_kernel.py nd=0 硬证）。批量
    eager 不传 valid_length：mask 已在 prepare_mask 统一清 pad 区
    （base.prepare_mask L67 `mask[..., valid_length:]=False`），gather 只选
    有效区，与逐行口径一致。cu_b = arange(B+1)*S_cap（固定全宽契约，kernel
    A 逐行校验 cu[b]==b*T）。B==1 / block_size>1 保持逐行路径零扰动
    （gate_bs1_equiv 铁律：batch=1 与旧单序列入口逐位一致）。
    """
    B = query_states.shape[0]
    dyn_state = None
    if getattr(indexer, "dyn_dgate", False):
        indexer.freeze_dyn_dgate()
        dyn_state = (indexer.skip_far, indexer._dyn_skip_pending,
                     indexer._dyn_frozen)
    q = query_states.transpose(1, 2)   # [B, 1, H, D]
    k = key_states.transpose(1, 2)     # [B, S_cap, Hkv, D]
    v = value_states.transpose(1, 2)   # [B, S_cap, Hkv, D]
    dev = q.device
    S_cap = k.shape[1]
    # 【batch indexer 并行化】TLI_BATCH_INDEXER=parallel 时走批量 kernel
    # 路径（Triton batch scoring + topk），替代逐行循环。
    if _parallel_batch_enabled(B):
        from ..indexer.tli_batch import prepare_mask_batched
        lengths = _batch_device_lengths(ctx, dev)
        mask, bs = prepare_mask_batched(indexer, q, k, lengths, self.scaling)
        _paper_route(self, "tli_batch_indexer_parallel_eager")
        return eager_decoding_attn(
            q, k, v, mask, bs, indexer._batch_tli.cu,
            softmax_scale=self.scaling,
            valid_length=ctx.seq_lens if B > 1 else ctx.seq_lens[0])
    # 【阶段 2c-4】graph 模式（pred_ruler_batch capture 窗口内）：动态几何
    # 全走 tensor（valid_len_buf），无 Python int 烘焙。配置不支持
    # compute_mask_graph（非生产主配置）时 fail-closed 关 graph + 回 staged
    # 路径（capture 前的 warmup 阶段触发，之后两臂一致）。
    from ..ops.cuda_graph_state import graph_state  # noqa: E402
    _gs = graph_state()
    if os.environ.get('TLI_BATCH_INDEXER', 'legacy') == 'exact':
        if not _gs.enabled:
            raise ValueError('exact batch requires the fixed-address graph-state route')
        return _tl_decode_batched_graph(self,indexer,q,k,v,_gs)
    if _gs.enabled:
        try:
            return _tl_decode_batched_graph(self, indexer, q, k, v, _gs)
        except NotImplementedError:
            _gs.enabled = False
            print("[2c] compute_mask_graph 不支持当前配置 → 关闭 CUDA Graph，"
                  "回退 staged 逐行路径", flush=True)
    # 逐行 mask 准备（indexer 状态逐行，B>1 与 B==1 共用同一循环）——
    # 阶段 2b 把 mask 收集起来，B>1 且 token 级时一次批量 eager。
    masks = []
    bs0 = None
    for b in range(B):
        L_b = ctx.seq_lens[b]
        q_b = q[b:b + 1]                    # [1, 1, H, D]
        k_b = k[b:b + 1]                    # [1, S_cap, Hkv, D]（全宽，有效在前）
        v_b = v[b:b + 1]                    # [1, S_cap, Hkv, D]
        cu = torch.tensor([0, S_cap], device=dev, dtype=torch.int32)
        pos = torch.tensor([L_b - 1], device=dev, dtype=torch.long)
        indexer.clear()
        if dyn_state is not None:
            (indexer.skip_far, indexer._dyn_skip_pending,
             indexer._dyn_frozen) = dyn_state
        mask, block_size = indexer.prepare_mask(
            q_b, pos, k_b, cu, softmax_scale=self.scaling, valid_length=L_b)
        masks.append(mask)
        bs0 = block_size
    if B > 1 and bs0 == 1:
        m_b = torch.cat(masks, dim=0)       # [B, 1, Hkv, S_cap]
        cu_b = torch.arange(B + 1, device=dev, dtype=torch.int32) * S_cap
        # valid_length 传逐行真实长 list（Triton 批量只消费 mask 忽略之；
        # torch 批量回退的 dense 段逐行裁剪归约域 [0,L_b) → 与变长路径
        # 逐位一致，防 pad 0 项参与归约树致 ULP 翻硬币）
        return eager_decoding_attn(
            q, k, v, m_b, bs0, cu_b, softmax_scale=self.scaling,
            valid_length=ctx.seq_lens)
    # B==1（gate_bs1_equiv 铁律）/ block_size>1 兜底：逐行单 query 路径
    outs = []
    for b in range(B):
        cu = torch.tensor([0, S_cap], device=dev, dtype=torch.int32)
        outs.append(eager_decoding_attn(
            q[b:b + 1], k[b:b + 1], v[b:b + 1], masks[b], bs0, cu,
            softmax_scale=self.scaling, valid_length=ctx.seq_lens[b]))
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

    # Compact consumers own both the KV write and absolute-to-ring addressing.
    # Dispatch before the generic update, which would otherwise append twice.
    if (past_key_values is not None
            and getattr(self.indexer, "far_skip_backend", "legacy") == "compact"):
        from ..ops.far_skip_runtime import try_far_skip_decode
        compact = try_far_skip_decode(
            self, query_states, key_states, value_states, past_key_values,
            attention_mask, cache_position, batch_decode_ctx())
        if compact is not None:
            attn_output = self.o_proj(compact.reshape(*input_shape, -1).contiguous())
            _paper_route(self.layer_idx, "hybrid-batch-decode"
                         if getattr(past_key_values, '_hybrid_batch', False) else "far-skip-decode")
            return attn_output, None

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
        # 【Phase 4】Serial graph 路由：TLI_OPERATOR_MODE=serial 时走
        # decode_indexed（compact selection → partial attention），跳过
        # legacy prepare_mask + eager_decoding_attn full-width mask 路径。
        from ..ops.operator_pipeline import operator_mode, decode_indexed
        _op_mode = operator_mode()
        if _op_mode in ('serial', 'overlap'):
            # 单序列口径：B=1 post-RoPE → transpose → [1,1,Hkv,S_cap]
            qs_t = query_states.transpose(1, 2)   # [1,1,Hkv,D]
            ks_t = key_states.transpose(1, 2)     # [1,S_cap,Hkv,D]
            vs_t = value_states.transpose(1, 2)
            dev = qs_t.device
            S_cap = ks_t.shape[1]
            vl = torch.tensor([ks_t.shape[1]], dtype=torch.int64, device=dev)
            # 实际有效长度从 cache 获取；单序列 decode 全宽 S_cap=cur_len
            if is_static_cache:
                vl = torch.tensor([cur_len], dtype=torch.int64, device=dev)
            cu = _graph_const(("cu_cap", str(dev), S_cap),
                              lambda: torch.tensor([0, S_cap], device=dev,
                                                   dtype=torch.int32))
            attn_output = decode_indexed(indexer, indexer,
                                         qs_t, ks_t, vs_t, cu, vl,
                                         self.scaling)
            _paper_route(self.layer_idx, f"{_op_mode}-decode")
            attn_weights = None
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        # A6（TL-E137-PREFILL-GATE-145）：decode 首步冻结 E137 动态标定决策。
        # prefill 期 calibrate_dyn_dgate 只写 _dyn_skip_pending（决策挂起），
        # 此处一次性落 skip_far（_dyn_frozen 哨兵防重复，clear() 复位两状态）
        # ——修复「完整 prompt 标定结果被稀疏 prefill chunk 消费」的非因果
        # 缺陷；dense prefill（默认路径）逐位不变。非 dyn_dgate 实例 no-op。
        if hasattr(indexer, "freeze_dyn_dgate"):
            indexer.freeze_dyn_dgate()
        # Legacy far skip retains full storage. Compact decode returned before
        # the generic cache update above; never feed ring KV to this selector.
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
