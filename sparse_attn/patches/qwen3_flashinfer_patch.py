"""Qwen3 全模型 decode 的独立 FlashInfer paged-attention 路径。

prefill 保留 HF 的 SDPA 与 RoPE；decode 每层直接追加 post-RoPE K/V 到
NHD paged cache，再调用 FlashInfer BatchDecode wrapper。该模块不经过 NFA
cache/selector，也不修改 HF StaticCache。
"""
from types import MethodType
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack


class _FlashInferMaskMetadata:
    """只向 HF mask builder 提供尺寸，不保存或更新任何 K/V。"""

    def __init__(self, num_layers, max_seq_len):
        self.is_sliding = [False] * num_layers
        self.is_compileable = False
        self.max_seq_len = max_seq_len

    def get_mask_sizes(self, cache_position, layer_idx):
        query_length = cache_position.shape[0]
        # prefill 必须得到严格 [L, L] causal mask；单 token decode 使用
        # 固定最大长度使 SDPA 走 None-mask 快路径，且避免 HF packed 误判。
        kv_length = self.max_seq_len if query_length == 1 else query_length
        return kv_length, 0


class FlashInferQwen3Runtime:
    """固定物理页、动态逻辑长度的 Qwen3 FlashInfer decode runtime。"""

    def __init__(self, model, batch_size, prefill_length, max_new_tokens,
                 device, page_size=16):
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, 'head_dim', config.hidden_size // self.num_qo_heads)
        if any(layer.self_attn.sliding_window is not None
               for layer in model.model.layers):
            raise ValueError(
                'FlashInfer canonical full-model path requires full-attention '
                'Qwen3 layers; sliding-window layers are not silently approximated')
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.max_new_tokens = max_new_tokens
        self.page_size = page_size
        self.device = device
        self.dtype = model.dtype
        # 第一个生成 token 在 prefill 后产生，最后一次 decode 只写入第
        # max_new_tokens-1 个生成 token，因此实际最大 KV 长度如下。
        self.max_seq_len = prefill_length + max_new_tokens - 1
        self.pages_per_request = (
            (self.max_seq_len + page_size - 1) // page_size)
        self.max_pages = batch_size * self.pages_per_request
        self.mask_metadata = _FlashInferMaskMetadata(
            self.num_layers, self.max_seq_len)

        # 每层独立 cache，形状严格为 [max_pages, 2, page_size, HKV, D]。
        # 所有层共享页表、plan 与 workspace，但 run 接收本层 cache view。
        self.paged_kv_cache = torch.empty(
            (self.num_layers, self.max_pages, 2, page_size,
             self.num_kv_heads, self.head_dim),
            dtype=self.dtype, device=device)
        self.decode_output = torch.empty(
            (self.num_layers, batch_size, self.num_qo_heads, self.head_dim),
            dtype=self.dtype, device=device)
        self.workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # 物理页在初始化时一次性固定；逻辑 CSR 由 indptr/indices 随长度增长。
        self.kv_indices = torch.arange(
            self.max_pages, dtype=torch.int32, device=device)
        self.kv_indptr = torch.empty(
            batch_size + 1, dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.empty(
            batch_size, dtype=torch.int32, device=device)
        self.decode_batch_indices = torch.arange(
            batch_size, dtype=torch.int32, device=device)
        self.decode_positions = torch.empty(
            batch_size, dtype=torch.int32, device=device)

        self.prefill_batch_indices = torch.arange(
            batch_size, dtype=torch.int32, device=device
        ).repeat_interleave(prefill_length)
        self.prefill_positions = torch.arange(
            prefill_length, dtype=torch.int32, device=device
        ).repeat(batch_size)
        self.wrapper = None
        self.execution = None
        self.indptr_history = None
        self.last_page_len_history = None
        self.page_table_updates = {}
        self.active_pages = None
        self.fixed_split_size = None
        self.set_sequence_length(prefill_length)

    def _compact_page_indices(self, pages):
        # 物理空间按 request 以 pages_per_request 为 stride 分段，但 FlashInfer
        # CSR 要求当前活跃页紧凑排列。B>1 时仅更新 indptr 会错指到上一行的
        # 预留尾页，因此跨页边界必须同步更新 indices 活跃前缀。
        return (torch.arange(
            self.batch_size, dtype=torch.int32, device=self.device
        )[:, None] * self.pages_per_request + torch.arange(
            pages, dtype=torch.int32, device=self.device)[None, :]).flatten()

    def set_sequence_length(self, seq_len):
        """所有 B 行同长；元数据包含即将 append 的当前 token。"""
        pages = (seq_len + self.page_size - 1) // self.page_size
        last_page_len = (seq_len - 1) % self.page_size + 1
        self.kv_indptr.copy_(
            torch.arange(self.batch_size + 1, dtype=torch.int32,
                         device=self.device) * pages)
        self.kv_indices[:self.batch_size * pages].copy_(
            self._compact_page_indices(pages))
        self.kv_last_page_len.fill_(last_page_len)
        self.active_pages = pages

    def _stable_fixed_split_size(self):
        """返回首步至末步 chunk 数恒定且不超 Graph CTA 预算的页数。"""
        properties = torch.cuda.get_device_properties(self.device)
        # FlashInfer FA2 Graph 的 padded batch 预算为 2 * SM / HKV；
        # B 行等长，按行均分后选择满足该预算的最大并发 chunk 数。
        total_chunk_budget = max(
            1, 2 * properties.multi_processor_count // self.num_kv_heads)
        chunk_budget = max(1, total_chunk_budget // self.batch_size)
        first_seq_len = self.prefill_length + 1
        max_pages = (
            (self.max_seq_len + self.page_size - 1) // self.page_size)
        initial = max(1, (max_pages + chunk_budget - 1) // chunk_budget)
        for split_pages in range(initial, max_pages + 1):
            split_tokens = split_pages * self.page_size
            first_chunks = (
                (first_seq_len + split_tokens - 1) // split_tokens)
            max_chunks = (self.max_seq_len + split_tokens - 1) // split_tokens
            if first_chunks == max_chunks:
                return split_pages
        # 单 chunk 必然覆盖所有步；仅在极短 KV 或异常几何下到达。
        return max_pages

    def prepare_decode(self, execution, decode_steps):
        """在计时与 Graph capture 前按最大长度 plan 一次。"""
        import flashinfer

        self.execution = execution
        seq_lens = self.prefill_length + torch.arange(
            1, decode_steps + 1, dtype=torch.int32, device=self.device)
        pages = (seq_lens + self.page_size - 1) // self.page_size
        self.indptr_history = (
            pages[:, None] * torch.arange(
                self.batch_size + 1, dtype=torch.int32, device=self.device
            )[None, :]).contiguous()
        self.last_page_len_history = (
            (seq_lens - 1) % self.page_size + 1).unsqueeze(1).expand(
                -1, self.batch_size).contiguous()
        self.page_table_updates = {}
        previous_pages = -1
        for step in range(decode_steps):
            pages = (self.prefill_length + step + self.page_size) // self.page_size
            if pages != previous_pages:
                self.page_table_updates[step] = self._compact_page_indices(pages)
                previous_pages = pages

        self.fixed_split_size = self._stable_fixed_split_size()
        if execution == 'graph':
            self.wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                self.workspace, self.kv_indptr, self.kv_indices,
                self.kv_last_page_len, kv_layout='NHD', use_tensor_cores=True)
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, kv_layout='NHD', use_tensor_cores=True)

        # plan 读取 host 长度以预留足够的调度资源；实际 run 从固定 device
        # buffer 读取每步真实长度。固定 split 使首末步 chunk 数一致：否则按
        # 最大长度生成的尾部 CTA 在短首步会出现 chunk_start > kv_len，无符号
        # chunk_size 下溢并令 kernel 停滞。plan 后恢复首步，绝不 attend 未来页。
        self.set_sequence_length(self.max_seq_len)
        self.wrapper.plan(
            self.kv_indptr, self.kv_indices, self.kv_last_page_len,
            self.num_qo_heads, self.num_kv_heads, self.head_dim,
            self.page_size, pos_encoding_mode='NONE',
            q_data_type=self.dtype, kv_data_type=self.dtype,
            fixed_split_size=self.fixed_split_size)
        if decode_steps:
            self.prepare_decode_step(0)
        else:
            self.set_sequence_length(self.prefill_length)

    def prepare_decode_step(self, step):
        """仅更新固定地址的小页表与 slot mapping，不复制任何 K/V。"""
        pages = (self.prefill_length + step + self.page_size) // self.page_size
        if pages != self.active_pages:
            update = self.page_table_updates[step]
            self.kv_indices[:self.batch_size * pages].copy_(update)
            self.active_pages = pages
        self.kv_indptr.copy_(self.indptr_history[step])
        self.kv_last_page_len.copy_(self.last_page_len_history[step])
        self.decode_positions.fill_(self.prefill_length + step)

    def append_prefill(self, layer_idx, key_states, value_states):
        import flashinfer

        batch, _, length, _ = key_states.shape
        append_key = key_states.transpose(1, 2).reshape(
            batch * length, self.num_kv_heads, self.head_dim)
        append_value = value_states.transpose(1, 2).reshape(
            batch * length, self.num_kv_heads, self.head_dim)
        flashinfer.append_paged_kv_cache(
            append_key, append_value, self.prefill_batch_indices,
            self.prefill_positions, self.paged_kv_cache[layer_idx],
            self.kv_indices, self.kv_indptr, self.kv_last_page_len,
            kv_layout='NHD')

    def decode_attention(self, layer_idx, query_states, key_states, value_states):
        import flashinfer

        # Qwen3 decode 原始布局 [B, H, 1, D] 的切片即为连续 [B, H, D]，
        # 无需 StaticCache 式 HND->NHD 全量复制。
        key = key_states[:, :, 0]
        value = value_states[:, :, 0]
        flashinfer.append_paged_kv_cache(
            key, value, self.decode_batch_indices, self.decode_positions,
            self.paged_kv_cache[layer_idx], self.kv_indices,
            self.kv_indptr, self.kv_last_page_len, kv_layout='NHD')
        return self.wrapper.run(
            query_states[:, :, 0], self.paged_kv_cache[layer_idx],
            out=self.decode_output[layer_idx])


def qwen3_flashinfer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    runtime = getattr(self, '_flashinfer_runtime', None)
    if runtime is None:
        raise RuntimeError('FlashInfer Qwen3 runtime has not been bound')
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
        # prefill 的 Q/K/V、RoPE、mask 和 SDPA 均保持 HF 数学语义；paged
        # append 只是为后续 decode 物化同一份 post-RoPE K/V。
        runtime.append_prefill(self.layer_idx, key_states, value_states)
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


def register_flashinfer_patch(model):
    """仅 patch Qwen3Attention.forward，不触碰 NFA 注册表或其他模块。"""
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            if not hasattr(module, '_flashinfer_original_forward'):
                module._flashinfer_original_forward = module.forward
            module.forward = MethodType(qwen3_flashinfer_forward, module)
            patched += 1
    return patched


def bind_flashinfer_runtime(model, runtime):
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._flashinfer_runtime = runtime
            patched += 1
    if patched != runtime.num_layers:
        raise RuntimeError(
            f'FlashInfer bound {patched}/{runtime.num_layers} Qwen3 layers')
