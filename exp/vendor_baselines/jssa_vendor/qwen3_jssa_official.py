"""Qwen3 的 JSSA 官方 Triton decode runtime。

直接加载 upstream/patched_src 中的官方 project/store、score/Top-K 与
indexed-attend kernel；不经过当前 HF torch 语义适配器。prefill 保持 HF SDPA，
同时在写入 post-RoPE K/V 时生成官方 FP8 projected-K cache。decode 使用固定
paged metadata 和预分配 scratch，支持 CUDA Graph。

边界：没有 vLLM radix binding 时使用官方文件自带的 torch.topk fallback；本
runtime 选择 int64 预分配索引，避免该 fallback 在 Graph capture 内分配临时
Top-K 输出。该回退会在测量回执中显式披露，不能称为 radix Top-K 性能。
"""
from importlib import util as importlib_util
from pathlib import Path
import sys
import types
from types import MethodType, SimpleNamespace
from typing import Optional

import torch
import triton
import triton.language as tl
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

from sparse_attn.patches.qwen3_flashinfer_patch import _FlashInferMaskMetadata


_OFFICIAL_OPS = None


def _load_module(name, path):
    spec = importlib_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official JSSA op from {path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_jssa_ops():
    """隔离加载官方 op；仅在缺少 vLLM 时临时注入 triton_utils import shim。"""
    global _OFFICIAL_OPS
    if _OFFICIAL_OPS is not None:
        return _OFFICIAL_OPS
    root = Path(__file__).resolve().parent / "upstream" / "patched_src" / "vllm" / "v1" / "attention" / "ops"
    saved = {}
    try:
        from vllm.triton_utils import tl as _tl, triton as _triton  # noqa: F401
    except ImportError:
        vllm_module = types.ModuleType("vllm")
        vllm_module.__path__ = []
        triton_utils = types.ModuleType("vllm.triton_utils")
        triton_utils.tl = tl
        triton_utils.triton = triton
        for name, module in (
            ("vllm", vllm_module),
            ("vllm.triton_utils", triton_utils),
        ):
            saved[name] = sys.modules.get(name)
            sys.modules[name] = module
    try:
        store = _load_module("_tli_official_jssa_store", root / "triton_jssa_store.py")
        score = _load_module("_tli_official_jssa_score_topk", root / "triton_jssa_score_topk.py")
        attend = _load_module("_tli_official_jssa_indexed_attend", root / "triton_jssa_indexed_attend.py")
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    _OFFICIAL_OPS = SimpleNamespace(store=store, score=score, attend=attend)
    return _OFFICIAL_OPS


def _load_basis(basis_path, num_layers, num_kv_heads, rank, head_dim, device, dtype):
    if not basis_path:
        raise ValueError("JSSA official runtime requires --jssa-basis-path")
    checkpoint = torch.load(basis_path, map_location="cpu", weights_only=False)
    basis = checkpoint.get("M") if isinstance(checkpoint, dict) else None
    if not isinstance(basis, dict):
        raise ValueError("JSSA basis must contain an M layer dictionary")
    result = []
    for layer_idx in range(num_layers):
        value = basis.get(layer_idx, basis.get(str(layer_idx)))
        if value is None:
            raise ValueError(f"JSSA basis misses layer {layer_idx}")
        expected = (num_kv_heads, rank, head_dim)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"JSSA M[{layer_idx}] shape {tuple(value.shape)} != {expected}")
        result.append(value.to(device=device, dtype=dtype).contiguous())
    return result


class JSSAQwen3Runtime:
    """固定物理页、动态逻辑长度的 JSSA 官方 kernel Qwen3 runtime。"""

    def __init__(
        self,
        model,
        batch_size,
        prefill_length,
        max_new_tokens,
        device,
        basis_path,
        budget=2048,
        rank=32,
        page_size=64,
    ):
        config = model.config
        self.ops = load_official_jssa_ops()
        self.num_layers = config.num_hidden_layers
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_qo_heads)
        self.num_kv_groups = self.num_qo_heads // self.num_kv_heads
        if self.num_kv_groups & (self.num_kv_groups - 1):
            raise ValueError("official JSSA indexed-attend requires power-of-two GQA")
        if any(layer.self_attn.sliding_window is not None
               for layer in model.model.layers):
            raise ValueError("JSSA official benchmark requires full-attention Qwen3")
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.max_new_tokens = max_new_tokens
        self.page_size = page_size
        self.budget = budget
        self.rank = rank
        self.device = device
        self.dtype = model.dtype
        self.max_seq_len = prefill_length + max_new_tokens - 1
        self.pages_per_request = (self.max_seq_len + page_size - 1) // page_size
        self.max_pages = batch_size * self.pages_per_request
        self.max_kv_len = self.pages_per_request * page_size
        self.mask_metadata = _FlashInferMaskMetadata(self.num_layers, self.max_seq_len)

        self.kv_cache = torch.empty(
            (self.num_layers, self.max_pages, page_size,
             self.num_kv_heads, 2 * self.head_dim),
            dtype=self.dtype, device=device)
        self.projk_cache = torch.empty(
            (self.num_layers, self.max_pages, page_size,
             self.num_kv_heads, rank),
            dtype=torch.float8_e4m3fn, device=device)
        self.decode_output = torch.empty(
            (self.num_layers, batch_size, self.num_qo_heads, self.head_dim),
            dtype=self.dtype, device=device)
        self.basis = _load_basis(
            basis_path, self.num_layers, self.num_kv_heads, rank,
            self.head_dim, device, self.dtype)
        self.projk_scale = [
            value.float().norm(dim=2).clamp_min(1e-6).to(self.dtype).contiguous()
            for value in self.basis
        ]

        self.block_table = (
            torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
            * self.pages_per_request
            + torch.arange(
                self.pages_per_request, dtype=torch.int32, device=device)[None, :]
        ).contiguous()
        self.seq_lens = torch.empty(batch_size, dtype=torch.int32, device=device)
        self.request_slot_bases = (
            torch.arange(batch_size, dtype=torch.int32, device=device)
            * self.pages_per_request * page_size)
        self.prefill_positions = torch.arange(
            prefill_length, dtype=torch.int32, device=device)
        self.prefill_slots = torch.empty(
            prefill_length, dtype=torch.int32, device=device)
        self.decode_slots = torch.empty(
            batch_size, dtype=torch.int32, device=device)

        self.scores = torch.empty(
            (batch_size, self.num_kv_heads, self.max_kv_len),
            dtype=torch.float32, device=device)
        # 无 radix binding 时，int64 out= 路径避免 Graph capture 内临时分配。
        self.top_idx = torch.empty(
            (batch_size, self.num_kv_heads, budget),
            dtype=torch.int64, device=device)
        self.top_scores = torch.empty(
            (batch_size, self.num_kv_heads, budget),
            dtype=torch.float32, device=device)
        self.q_kv = torch.empty(
            (batch_size, self.num_kv_heads, self.head_dim),
            dtype=self.dtype, device=device)
        self.projq_hrc = torch.empty(
            (self.num_kv_heads, batch_size, rank),
            dtype=self.dtype, device=device)
        self._scratch_buffers = {}
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.radix_topk_available = self.ops.score._radix_topk_available()
        self.execution = None
        self.prefill_store_calls = 0
        self.decode_store_calls = 0
        self.decode_attention_calls = 0

    def _scratch(self, name, shape, dtype, device):
        key = (name, tuple(shape), dtype, device)
        value = self._scratch_buffers.get(key)
        if value is None:
            value = torch.empty(shape, dtype=dtype, device=device)
            self._scratch_buffers[key] = value
        return value

    def begin_prefill_request(self, request_id):
        self.prefill_slots.copy_(self.prefill_positions)
        self.prefill_slots.add_(
            request_id * self.pages_per_request * self.page_size)

    def append_prefill(self, layer_idx, key_states, value_states):
        if key_states.shape[0] != 1:
            raise ValueError("JSSA benchmark prefill is serial per request")
        _, _, length, _ = key_states.shape
        key = key_states.transpose(1, 2).reshape(
            length, self.num_kv_heads, self.head_dim)
        value = value_states.transpose(1, 2).reshape(
            length, self.num_kv_heads, self.head_dim)
        self.ops.store.jssa_project_and_store_contig_fp8(
            key, value, self.kv_cache[layer_idx], self.projk_cache[layer_idx],
            self.projk_scale[layer_idx], self.prefill_slots, self.basis[layer_idx])
        self.prefill_store_calls += 1

    def prepare_decode(self, execution, decode_steps):
        self.execution = execution
        if self.prefill_length + 1 < min(self.budget, self.max_kv_len):
            raise ValueError("official indexed decode benchmark requires seq_len >= budget")
        if decode_steps:
            self.prepare_decode_step(0)
        else:
            self.seq_lens.fill_(self.prefill_length)

    def prepare_decode_step(self, step):
        position = self.prefill_length + step
        self.seq_lens.fill_(position + 1)
        self.decode_slots.copy_(self.request_slot_bases)
        self.decode_slots.add_(position)

    def decode_attention(self, layer_idx, query_states, key_states, value_states):
        key = key_states[:, :, 0]
        value = value_states[:, :, 0]
        self.ops.store.jssa_project_and_store_contig_fp8(
            key, value, self.kv_cache[layer_idx], self.projk_cache[layer_idx],
            self.projk_scale[layer_idx], self.decode_slots, self.basis[layer_idx])
        self.decode_store_calls += 1

        query = query_states[:, :, 0]
        torch.mean(
            query.view(self.batch_size, self.num_kv_heads,
                       self.num_kv_groups, self.head_dim),
            dim=2, out=self.q_kv)
        basis = self.basis[layer_idx]
        torch.bmm(
            self.q_kv.transpose(0, 1), basis.transpose(1, 2),
            out=self.projq_hrc)
        self.projq_hrc.mul_(self.projk_scale[layer_idx].unsqueeze(1))
        top_idx, _ = self.ops.score.jssa_score_topk(
            self.projq_hrc.transpose(0, 1), self.kv_cache[layer_idx],
            self.block_table, self.seq_lens, n_fac=self.budget,
            head_size=self.head_dim, cs_h=self.rank,
            scores_out=self.scores, top_idx_out=self.top_idx,
            top_scores_out=self.top_scores, use_radix=True,
            projk_cache=self.projk_cache[layer_idx])
        output = self.ops.attend.jssa_indexed_attend(
            query, self.kv_cache[layer_idx], self.block_table, top_idx,
            head_size=self.head_dim, scale=self.scaling(layer_idx),
            out=self.decode_output[layer_idx], scratch=self._scratch,
            sm_count=self._sm_count)
        self.decode_attention_calls += 1
        return output

    def scaling(self, layer_idx):
        # 由 bind 时从真实 Qwen3Attention 写入，避免重新推导 RoPE/softmax 配置。
        return self._layer_scalings[layer_idx]

    def receipt(self):
        return {
            "implementation": "official vendored Triton store/score/indexed-attend",
            "basis_path_independent_artifact": True,
            "budget": self.budget,
            "rank": self.rank,
            "page_size": self.page_size,
            "kv_cache_layout": "[layer, block, position, H_kv, K|V] BF16",
            "projected_key_cache": "separate contiguous FP8 E4M3 [layer, block, position, H_kv, rank]",
            "projected_key_scale": "per-(KV-head,channel) basis row norm folded into proj_q",
            "score_kernel": self.ops.score._jssa_score_contig_kernel.fn.__name__,
            "attention_kernel": "official gather-free online-softmax indexed-attend",
            "local_compatibility_patch": (
                "pad GQA rows to at least 16 in registers for Triton tl.dot; "
                "all padded loads/stores are masked"),
            "topk": (
                "vLLM radix" if self.radix_topk_available
                else "official torch.topk fallback with preallocated int64 Graph-safe output"),
            "radix_topk_available": self.radix_topk_available,
            "prefill_attention": "HF SDPA with original Qwen3 RoPE",
            "prefill_store": "official fused project+store for every post-RoPE K/V token",
            "decode_store_calls_warmup_and_capture": self.decode_store_calls,
            "decode_attention_calls_warmup_and_capture": self.decode_attention_calls,
        }


def qwen3_jssa_official_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    runtime = getattr(self, "_jssa_official_runtime", None)
    if runtime is None:
        raise RuntimeError("JSSA official Qwen3 runtime has not been bound")
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


def register_jssa_official_patch(model):
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            if not hasattr(module, "_jssa_official_original_forward"):
                module._jssa_official_original_forward = module.forward
            module.forward = MethodType(qwen3_jssa_official_forward, module)
            patched += 1
    return patched


def bind_jssa_official_runtime(model, runtime):
    patched = 0
    runtime._layer_scalings = []
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module._jssa_official_runtime = runtime
            runtime._layer_scalings.append(module.scaling)
            patched += 1
    if patched != runtime.num_layers:
        raise RuntimeError(
            f"JSSA official bound {patched}/{runtime.num_layers} Qwen3 layers")
