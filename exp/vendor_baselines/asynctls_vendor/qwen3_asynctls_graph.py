"""AsyncTLS 论文语义的 Qwen3 固定缓存 CUDA Graph 复现。

这不是官方 TileLang 实现，也不包含真实异步 offload。复现 block64/top128、
per-KV-group 32 channel、INT4 量化-反量化、上一步 coarse blocks、RULER
budget512 和原始全维 selected attention。INT4 只复现数值行为，不压缩存储。
所有 decode host 标量均保存在固定 GPU buffer，Graph replay 不执行 .item()、
.tolist() 或逐 token D2H。
"""
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

from exp.vendor_baselines.asynctls_vendor.qwen3_asynctls_patch import _load_channels
from sparse_attn.patches.qwen3_flashinfer_patch import _FlashInferMaskMetadata


class AsyncTLSQwen3GraphRuntime:
    """固定物理 KV、block metadata 和一步延迟 coarse-block 状态。"""

    def __init__(
        self,
        model,
        batch_size,
        prefill_length,
        max_new_tokens,
        device,
        channel_file,
        token_budget=512,
        block_size=64,
        top_kb=128,
    ):
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_qo_heads)
        self.num_kv_groups = self.num_qo_heads // self.num_kv_heads
        if any(layer.self_attn.sliding_window is not None
               for layer in model.model.layers):
            raise ValueError("AsyncTLS benchmark requires full-attention Qwen3")
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.max_new_tokens = max_new_tokens
        self.token_budget = token_budget
        self.block_size = block_size
        self.top_kb = top_kb
        self.device = device
        self.dtype = model.dtype
        self.max_seq_len = prefill_length + max_new_tokens - 1
        self.max_blocks = (self.max_seq_len + block_size - 1) // block_size
        self.prefill_blocks = (prefill_length + block_size - 1) // block_size
        if self.prefill_blocks < top_kb:
            raise ValueError("AsyncTLS fixed Top-K requires prefill_blocks >= top_kb")
        self.candidate_capacity = top_kb * block_size + 1
        if self.prefill_length + 1 < token_budget:
            raise ValueError("AsyncTLS fixed token Top-K requires enough valid tokens")
        self.mask_metadata = _FlashInferMaskMetadata(
            self.num_layers, self.max_seq_len)

        # 每层独立 BF16 K/V；只读有效候选，未初始化尾部不会被 gather。
        cache_shape = (
            self.num_layers, batch_size, self.max_seq_len,
            self.num_kv_heads, self.head_dim)
        self.k_cache = torch.empty(cache_shape, dtype=self.dtype, device=device)
        self.v_cache = torch.empty(cache_shape, dtype=self.dtype, device=device)
        metadata_shape = (
            self.num_layers, batch_size, self.num_kv_heads,
            self.max_blocks, self.head_dim)
        self.k_min = torch.empty(
            metadata_shape, dtype=self.dtype, device=device)
        self.k_max = torch.empty(
            metadata_shape, dtype=self.dtype, device=device)
        self.previous_blocks = torch.empty(
            (self.num_layers, batch_size, self.num_kv_heads, top_kb),
            dtype=torch.long, device=device)
        self.initial_blocks = None

        channels = _load_channels(channel_file)
        if len(channels) != self.num_layers:
            raise ValueError("AsyncTLS channel layer count mismatch")
        self.channels = []
        for layer_channels in channels:
            if len(layer_channels) != self.num_kv_heads:
                raise ValueError("AsyncTLS channel KV-group count mismatch")
            value = torch.tensor(
                layer_channels, dtype=torch.long, device=device)
            if value.shape != (self.num_kv_heads, 32):
                raise ValueError("AsyncTLS reproduction requires dc=32")
            self.channels.append(value.contiguous())

        self.batch_indices = torch.arange(
            batch_size, dtype=torch.long, device=device)
        self.positions = torch.empty(
            batch_size, dtype=torch.long, device=device)
        self.decode_blocks = torch.empty(
            batch_size, dtype=torch.long, device=device)
        self.new_block = torch.empty(
            batch_size, dtype=torch.bool, device=device)
        self.block_offsets = torch.arange(
            block_size, dtype=torch.long, device=device)
        self.block_arange = torch.arange(
            self.max_blocks, dtype=torch.long, device=device)
        self.prefill_valid = (
            torch.arange(self.prefill_blocks * block_size, device=device)
            < prefill_length)
        self.decode_output = torch.empty(
            (self.num_layers, batch_size, self.num_qo_heads, self.head_dim),
            dtype=self.dtype, device=device)
        self.prefill_calls = 0
        self.decode_calls = 0
        self.execution = None

    def _block_scores(self, layer_idx, query):
        # query: [B,H,D]；对 GQA 组内 query head 与 channel 求和，对齐语义适配器。
        grouped = query.reshape(
            self.batch_size, self.num_kv_heads, self.num_kv_groups,
            self.head_dim)
        qmax = grouped.clamp(min=0)
        qmin = grouped.clamp(max=0)
        scores = torch.einsum(
            "bhgd,bhmd->bhm", qmax, self.k_max[layer_idx])
        scores = scores + torch.einsum(
            "bhgd,bhmd->bhm", qmin, self.k_min[layer_idx])
        return scores

    def append_prefill(self, layer_idx, query_states, key_states, value_states):
        if key_states.shape[0] != 1:
            raise ValueError("AsyncTLS benchmark prefill is serial per request")
        request_id = self.prefill_calls // self.num_layers
        key = key_states[0].transpose(0, 1)
        value = value_states[0].transpose(0, 1)
        self.k_cache[layer_idx, request_id, :self.prefill_length].copy_(key)
        self.v_cache[layer_idx, request_id, :self.prefill_length].copy_(value)

        padded_length = self.prefill_blocks * self.block_size
        key_padded = F.pad(
            key, (0, 0, 0, 0, 0, padded_length - self.prefill_length))
        blocks = key_padded.reshape(
            self.prefill_blocks, self.block_size,
            self.num_kv_heads, self.head_dim).transpose(1, 2)
        valid = self.prefill_valid.reshape(
            self.prefill_blocks, 1, self.block_size, 1)
        k_min = blocks.masked_fill(~valid, torch.inf).amin(dim=2).transpose(0, 1)
        k_max = blocks.masked_fill(~valid, -torch.inf).amax(dim=2).transpose(0, 1)
        self.k_min[layer_idx, request_id, :, :self.prefill_blocks].copy_(k_min)
        self.k_max[layer_idx, request_id, :, :self.prefill_blocks].copy_(k_max)

        query = query_states[0, :, -1].reshape(
            self.num_kv_heads, self.num_kv_groups, self.head_dim)
        scores = torch.einsum(
            "hgd,hmd->hm", query.clamp(min=0), k_max)
        scores = scores + torch.einsum(
            "hgd,hmd->hm", query.clamp(max=0), k_min)
        self.previous_blocks[layer_idx, request_id].copy_(
            torch.topk(scores, k=self.top_kb, dim=-1).indices)
        self.prefill_calls += 1

    def prepare_decode(self, execution, decode_steps):
        self.execution = execution
        if self.prefill_calls != self.batch_size * self.num_layers:
            raise RuntimeError("AsyncTLS prefill call count mismatch")
        self.initial_blocks = self.previous_blocks.clone()
        if decode_steps:
            self.prepare_decode_step(0, reset_pipeline=True)

    def prepare_decode_step(self, step, reset_pipeline=False):
        position = self.prefill_length + step
        self.positions.fill_(position)
        self.decode_blocks.fill_(position // self.block_size)
        self.new_block.fill_(position % self.block_size == 0)
        if reset_pipeline:
            if self.initial_blocks is None:
                raise RuntimeError("AsyncTLS initial coarse blocks are not ready")
            self.previous_blocks.copy_(self.initial_blocks)

    def decode_attention(self, layer_idx, query_states, key_states, value_states):
        key = key_states[:, :, 0]
        value = value_states[:, :, 0]
        query = query_states[:, :, 0]
        self.k_cache[layer_idx][self.batch_indices, self.positions] = key
        self.v_cache[layer_idx][self.batch_indices, self.positions] = value

        old_min = self.k_min[layer_idx][self.batch_indices, :, self.decode_blocks]
        old_max = self.k_max[layer_idx][self.batch_indices, :, self.decode_blocks]
        new_block = self.new_block[:, None, None]
        self.k_min[layer_idx][self.batch_indices, :, self.decode_blocks] = (
            torch.where(new_block, key, torch.minimum(old_min, key)))
        self.k_max[layer_idx][self.batch_indices, :, self.decode_blocks] = (
            torch.where(new_block, key, torch.maximum(old_max, key)))

        scores = self._block_scores(layer_idx, query)
        valid_blocks = self.block_arange <= self.decode_blocks[:, None, None]
        scores = scores.masked_fill(~valid_blocks, -torch.inf)
        current_blocks = torch.topk(scores, k=self.top_kb, dim=-1).indices

        candidates = (
            self.previous_blocks[layer_idx, ..., None] * self.block_size
            + self.block_offsets
        ).reshape(self.batch_size, self.num_kv_heads, -1)
        current = self.positions[:, None, None].expand(
            self.batch_size, self.num_kv_heads, 1)
        candidate_idx = torch.cat((candidates, current), dim=-1)
        candidate_valid = candidate_idx <= self.positions[:, None, None]
        # previous_blocks 对应未写入当前 token 的 M_{t-1}；当前位置单独追加。
        candidate_valid[..., :-1] &= (
            candidate_idx[..., :-1] < self.positions[:, None, None])
        # 部分尾块的 padding 即使被 mask，也必须先夹到合法 gather 域。
        candidate_idx = torch.minimum(
            candidate_idx, self.positions[:, None, None])

        key_bhld = self.k_cache[layer_idx].transpose(1, 2)
        value_bhld = self.v_cache[layer_idx].transpose(1, 2)
        candidate_key = torch.gather(
            key_bhld, 2,
            candidate_idx[..., None].expand(
                self.batch_size, self.num_kv_heads,
                self.candidate_capacity, self.head_dim))
        channels = self.channels[layer_idx]
        query_grouped = query.reshape(
            self.batch_size, self.num_kv_heads,
            self.num_kv_groups, self.head_dim)
        query_projected = torch.gather(
            query_grouped, 3,
            channels[None, :, None, :].expand(
                self.batch_size, self.num_kv_heads,
                self.num_kv_groups, channels.shape[1]))
        candidate_key = torch.gather(
            candidate_key, 3,
            channels[None, :, None, :].expand(
                self.batch_size, self.num_kv_heads,
                self.candidate_capacity, channels.shape[1]))

        # per-token INT4 量化-反量化；不宣称压缩存储或真实 offload。
        mn = candidate_key.amin(dim=-1, keepdim=True)
        mx = candidate_key.amax(dim=-1, keepdim=True)
        scale = (mx - mn) / 15
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = torch.round((candidate_key - mn) / scale - 8).clamp(-8, 7)
        candidate_key = (quantized + 8) * scale + mn
        logits = torch.einsum(
            "bhgc,bhjc->bhgj", query_projected, candidate_key)
        logits = logits * self._layer_scalings[layer_idx]
        logits = logits.masked_fill(
            ~candidate_valid[:, :, None, :], -torch.inf)
        candidate_score = F.softmax(logits.float(), dim=-1).mean(dim=2)
        candidate_score[..., -1] = torch.inf
        selected_local = torch.topk(
            candidate_score, k=self.token_budget, dim=-1).indices
        selected_idx = torch.gather(candidate_idx, 2, selected_local)
        selected_idx = torch.sort(selected_idx, dim=-1).values

        selected_key = torch.gather(
            key_bhld, 2,
            selected_idx[..., None].expand(
                self.batch_size, self.num_kv_heads,
                self.token_budget, self.head_dim))
        selected_value = torch.gather(
            value_bhld, 2,
            selected_idx[..., None].expand(
                self.batch_size, self.num_kv_heads,
                self.token_budget, self.head_dim))
        attn_weights = torch.matmul(
            query_grouped, selected_key.transpose(-2, -1))
        attn_weights = attn_weights * self._layer_scalings[layer_idx]
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(self.dtype)
        attn_output = torch.matmul(attn_weights, selected_value)
        self.previous_blocks[layer_idx].copy_(current_blocks)
        self.decode_output[layer_idx].copy_(attn_output.reshape(
            self.batch_size, self.num_qo_heads, self.head_dim))
        self.decode_calls += 1
        return self.decode_output[layer_idx]

    def receipt(self):
        return {
            "implementation": "independent PyTorch semantic reimplementation; not official TileLang",
            "paper_pdf_verified": True,
            "source_verified_official": False,
            "block_size": self.block_size,
            "top_kb": self.top_kb,
            "candidate_tokens_including_current": self.candidate_capacity,
            "channels_per_kv_group": 32,
            "token_budget": self.token_budget,
            "pipeline": "Stage2 uses previous-step coarse blocks M_(t-1)",
            "int4": "per-token 16-level quantize-dequantize; storage remains BF16",
            "offload": "none; no claim of asynchronous host offload",
            "stage3": "exact full-dimensional Q/K/V attention on selected tokens",
            "kv_cache_layout": "[layer,batch,position,H_kv,D] BF16 K and V",
            "block_metadata": "incremental per-block BF16 K min/max",
            "prefill_attention": "HF SDPA with original Qwen3 RoPE",
            "prefill_calls": self.prefill_calls,
            "decode_calls_warmup_and_capture": self.decode_calls,
        }


def qwen3_asynctls_graph_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    runtime = getattr(self, "_asynctls_graph_runtime", None)
    if runtime is None:
        raise RuntimeError("AsyncTLS Graph runtime has not been bound")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    if input_shape[1] != 1:
        runtime.append_prefill(
            self.layer_idx, query_states, key_states, value_states)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window,
            **kwargs)
        return self.o_proj(
            attn_output.reshape(*input_shape, -1).contiguous()), attn_weights

    attn_output = runtime.decode_attention(
        self.layer_idx, query_states, key_states, value_states)
    return self.o_proj(
        attn_output.reshape(*input_shape, -1).contiguous()), None


def register_asynctls_graph_patch(model):
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._asynctls_graph_original_forward = module.forward
            module.forward = MethodType(qwen3_asynctls_graph_forward, module)
            patched += 1
    return patched


def bind_asynctls_graph_runtime(model, runtime):
    patched = 0
    runtime._layer_scalings = []
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._asynctls_graph_runtime = runtime
            runtime._layer_scalings.append(module.scaling)
            patched += 1
    if patched != runtime.num_layers:
        raise RuntimeError(
            f"AsyncTLS bound {patched}/{runtime.num_layers} Qwen3 layers")
