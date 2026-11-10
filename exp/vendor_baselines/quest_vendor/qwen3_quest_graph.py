"""Quest 论文语义的 Qwen3 固定缓存 CUDA Graph 复现。

这不是官方 FP16/B1 CUDA runtime。实现 page16、每 query head 独立选页、
2048-token 总页预算（127 个历史 Top-K 页 + 当前页）以及前两层完整 KV。
原生 GQA 适配共享物理 K/V 和每 KV head 的 page min/max，不复制四倍 KV；
每个 GQA 组内的 query head 仍独立打分和选页。稀疏层只读取选中页，不计算
full-QK 后 mask。所有 decode 位置保存在固定 GPU buffer，Graph replay 不执行
.item()、.tolist() 或逐 token D2H。
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

from sparse_attn.patches.qwen3_flashinfer_patch import _FlashInferMaskMetadata


class QuestQwen3GraphRuntime:
    """固定物理 K/V、page metadata 和每 query head 的选中页。"""

    def __init__(
        self,
        model,
        batch_size,
        prefill_length,
        max_new_tokens,
        device,
        token_budget=2048,
        page_size=16,
        dense_layers=2,
        record_selection=False,
    ):
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_qo_heads)
        if self.num_qo_heads % self.num_kv_heads:
            raise ValueError("Quest GQA requires divisible query/KV heads")
        self.num_kv_groups = self.num_qo_heads // self.num_kv_heads
        if any(layer.self_attn.sliding_window is not None
               for layer in model.model.layers):
            raise ValueError("Quest benchmark requires full-attention Qwen3")
        if token_budget % page_size:
            raise ValueError("Quest token budget must contain whole pages")
        if not 0 <= dense_layers <= self.num_layers:
            raise ValueError("invalid Quest dense layer count")
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.max_new_tokens = max_new_tokens
        self.token_budget = token_budget
        self.page_size = page_size
        self.page_budget = token_budget // page_size
        self.history_topk = self.page_budget - 1
        self.dense_layers = dense_layers
        self.record_selection = record_selection
        self.device = device
        self.dtype = model.dtype
        self.max_seq_len = prefill_length + max_new_tokens - 1
        self.max_pages = (self.max_seq_len + page_size - 1) // page_size
        self.prefill_pages = (prefill_length + page_size - 1) // page_size
        if self.history_topk < 1 or self.prefill_pages <= self.page_budget:
            raise ValueError(
                "Quest fixed Graph benchmark requires prefill pages > page budget")
        self.mask_metadata = _FlashInferMaskMetadata(
            self.num_layers, self.max_seq_len)

        cache_shape = (
            self.num_layers, batch_size, self.max_seq_len,
            self.num_kv_heads, self.head_dim)
        # 前两层对固定容量做 masked dense attention，零初始化避免未用尾部产生 NaN。
        self.k_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        metadata_shape = (
            self.num_layers, batch_size, self.num_kv_heads,
            self.max_pages, self.head_dim)
        self.k_min = torch.empty(metadata_shape, dtype=self.dtype, device=device)
        self.k_max = torch.empty(metadata_shape, dtype=self.dtype, device=device)
        self.decode_output = torch.empty(
            (self.num_layers, batch_size, self.num_qo_heads, self.head_dim),
            dtype=self.dtype, device=device)

        self.batch_indices = torch.arange(
            batch_size, dtype=torch.long, device=device)
        self.positions = torch.empty(batch_size, dtype=torch.long, device=device)
        self.decode_pages = torch.empty(
            batch_size, dtype=torch.long, device=device)
        self.new_page = torch.empty(batch_size, dtype=torch.bool, device=device)
        self.page_offsets = torch.arange(
            page_size, dtype=torch.long, device=device)
        self.page_arange = torch.arange(
            self.max_pages, dtype=torch.long, device=device)
        self.token_arange = torch.arange(
            self.max_seq_len, dtype=torch.long, device=device)
        self.prefill_valid = (
            torch.arange(self.prefill_pages * page_size, device=device)
            < prefill_length)
        if record_selection:
            self.selected_pages = torch.empty(
                (self.num_layers, batch_size, self.num_qo_heads, self.page_budget),
                dtype=torch.long, device=device)
            self.selected_indices = torch.empty(
                (self.num_layers, batch_size, self.num_qo_heads, token_budget),
                dtype=torch.long, device=device)
        self.prefill_calls = 0
        self.decode_calls = 0
        self.execution = None

    def append_prefill(self, layer_idx, query_states, key_states, value_states):
        if key_states.shape[0] != 1:
            raise ValueError("Quest benchmark prefill is serial per request")
        request_id = self.prefill_calls // self.num_layers
        key = key_states[0].transpose(0, 1)
        value = value_states[0].transpose(0, 1)
        self.k_cache[layer_idx, request_id, :self.prefill_length].copy_(key)
        self.v_cache[layer_idx, request_id, :self.prefill_length].copy_(value)

        padded_length = self.prefill_pages * self.page_size
        key_padded = F.pad(
            key, (0, 0, 0, 0, 0, padded_length - self.prefill_length))
        pages = key_padded.reshape(
            self.prefill_pages, self.page_size,
            self.num_kv_heads, self.head_dim).transpose(1, 2)
        valid = self.prefill_valid.reshape(
            self.prefill_pages, 1, self.page_size, 1)
        k_min = pages.masked_fill(~valid, torch.inf).amin(dim=2).transpose(0, 1)
        k_max = pages.masked_fill(~valid, -torch.inf).amax(dim=2).transpose(0, 1)
        self.k_min[layer_idx, request_id, :, :self.prefill_pages].copy_(k_min)
        self.k_max[layer_idx, request_id, :, :self.prefill_pages].copy_(k_max)
        self.prefill_calls += 1

    def prepare_decode(self, execution, decode_steps):
        self.execution = execution
        if self.prefill_calls != self.batch_size * self.num_layers:
            raise RuntimeError("Quest prefill call count mismatch")
        if decode_steps:
            self.prepare_decode_step(0)

    def prepare_decode_step(self, step):
        position = self.prefill_length + step
        self.positions.fill_(position)
        self.decode_pages.fill_(position // self.page_size)
        self.new_page.fill_(position % self.page_size == 0)

    def _dense_attention(self, layer_idx, query):
        key = self.k_cache[layer_idx].transpose(1, 2)
        value = self.v_cache[layer_idx].transpose(1, 2)
        query_grouped = query.reshape(
            self.batch_size, self.num_kv_heads, self.num_kv_groups,
            1, self.head_dim)
        weights = torch.matmul(
            query_grouped, key.unsqueeze(2).transpose(-2, -1))
        weights = weights * self._layer_scalings[layer_idx]
        valid = self.token_arange <= self.positions[:, None]
        weights = weights.masked_fill(
            ~valid[:, None, None, None, :], -torch.inf)
        weights = F.softmax(weights.float(), dim=-1).to(self.dtype)
        output = torch.matmul(weights, value.unsqueeze(2))
        return output.reshape(
            self.batch_size, self.num_qo_heads, self.head_dim)

    def decode_attention(self, layer_idx, query_states, key_states, value_states):
        key = key_states[:, :, 0]
        value = value_states[:, :, 0]
        query = query_states[:, :, 0]
        self.k_cache[layer_idx][self.batch_indices, self.positions] = key
        self.v_cache[layer_idx][self.batch_indices, self.positions] = value

        old_min = self.k_min[layer_idx][self.batch_indices, :, self.decode_pages]
        old_max = self.k_max[layer_idx][self.batch_indices, :, self.decode_pages]
        new_page = self.new_page[:, None, None]
        self.k_min[layer_idx][self.batch_indices, :, self.decode_pages] = (
            torch.where(new_page, key, torch.minimum(old_min, key)))
        self.k_max[layer_idx][self.batch_indices, :, self.decode_pages] = (
            torch.where(new_page, key, torch.maximum(old_max, key)))

        if layer_idx < self.dense_layers:
            output = self._dense_attention(layer_idx, query)
            self.decode_output[layer_idx].copy_(output)
            self.decode_calls += 1
            return self.decode_output[layer_idx]

        query_grouped = query.reshape(
            self.batch_size, self.num_kv_heads,
            self.num_kv_groups, self.head_dim)
        scores = torch.einsum(
            "bhgd,bhpd->bhgp", query_grouped.clamp(min=0),
            self.k_max[layer_idx])
        scores = scores + torch.einsum(
            "bhgd,bhpd->bhgp", query_grouped.clamp(max=0),
            self.k_min[layer_idx])
        scores = scores.reshape(
            self.batch_size, self.num_qo_heads, self.max_pages)
        history_valid = self.page_arange < self.decode_pages[:, None, None]
        scores = scores.masked_fill(~history_valid, -torch.inf)
        history_pages = torch.topk(
            scores, k=self.history_topk, dim=-1).indices
        current_page = self.decode_pages[:, None, None].expand(
            self.batch_size, self.num_qo_heads, 1)
        selected_pages = torch.cat((history_pages, current_page), dim=-1)
        selected_idx = (
            selected_pages[..., None] * self.page_size + self.page_offsets
        ).reshape(self.batch_size, self.num_qo_heads, self.token_budget)
        selected_idx = torch.sort(selected_idx, dim=-1).values
        selected_valid = selected_idx <= self.positions[:, None, None]
        gather_idx = torch.minimum(
            selected_idx, self.positions[:, None, None])
        if self.record_selection:
            self.selected_pages[layer_idx].copy_(selected_pages)
            self.selected_indices[layer_idx].copy_(selected_idx)

        # 物理 K/V 只在 KV head 上存储；各 query head 的独立页选集在组内 gather。
        grouped_idx = gather_idx.reshape(
            self.batch_size, self.num_kv_heads, -1)
        gather_shape = grouped_idx[..., None].expand(
            self.batch_size, self.num_kv_heads,
            grouped_idx.shape[-1], self.head_dim)
        key_bhld = self.k_cache[layer_idx].transpose(1, 2)
        value_bhld = self.v_cache[layer_idx].transpose(1, 2)
        selected_key = torch.gather(key_bhld, 2, gather_shape).reshape(
            self.batch_size, self.num_kv_heads, self.num_kv_groups,
            self.token_budget, self.head_dim)
        selected_value = torch.gather(value_bhld, 2, gather_shape).reshape(
            self.batch_size, self.num_kv_heads, self.num_kv_groups,
            self.token_budget, self.head_dim)
        weights = torch.matmul(
            query_grouped.unsqueeze(3), selected_key.transpose(-2, -1))
        weights = weights * self._layer_scalings[layer_idx]
        weights = weights.masked_fill(
            ~selected_valid.reshape(
                self.batch_size, self.num_kv_heads, self.num_kv_groups,
                1, self.token_budget), -torch.inf)
        weights = F.softmax(weights.float(), dim=-1).to(self.dtype)
        output = torch.matmul(weights, selected_value)
        self.decode_output[layer_idx].copy_(output.reshape(
            self.batch_size, self.num_qo_heads, self.head_dim))
        self.decode_calls += 1
        return self.decode_output[layer_idx]

    def receipt(self):
        return {
            "implementation": (
                "independent PyTorch semantic reimplementation with native-GQA "
                "adaptation; not official Quest CUDA runtime"),
            "paper_pdf_verified": True,
            "official_source_checked": True,
            "official_cuda": False,
            "page_size": self.page_size,
            "token_budget": self.token_budget,
            "page_budget_including_current": self.page_budget,
            "history_topk_pages": self.history_topk,
            "current_page_always_retained": True,
            "selection_granularity": "independent Top-K pages per query head",
            "gqa_adaptation": (
                "shared physical K/V and per-KV-head page min/max; independent "
                "query-head scores and selections within each GQA group"),
            "dense_layers": self.dense_layers,
            "sparse_decode_reads_selected_pages_only": True,
            "full_qk_mask_semantics": False,
            "kv_cache_layout": "[layer,batch,position,H_kv,D] BF16 K and V",
            "page_metadata": "incremental per-KV-head BF16 K channel min/max",
            "prefill_attention": "HF SDPA with original Qwen3 RoPE",
            "prefill_calls": self.prefill_calls,
            "decode_calls_warmup_and_capture": self.decode_calls,
        }


def qwen3_quest_graph_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    runtime = getattr(self, "_quest_graph_runtime", None)
    if runtime is None:
        raise RuntimeError("Quest Graph runtime has not been bound")
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


def register_quest_graph_patch(model):
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._quest_graph_original_forward = module.forward
            module.forward = MethodType(qwen3_quest_graph_forward, module)
            patched += 1
    return patched


def bind_quest_graph_runtime(model, runtime):
    patched = 0
    runtime._layer_scalings = []
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._quest_graph_runtime = runtime
            runtime._layer_scalings.append(module.scaling)
            patched += 1
    if patched != runtime.num_layers:
        raise RuntimeError(
            f"Quest bound {patched}/{runtime.num_layers} Qwen3 layers")
