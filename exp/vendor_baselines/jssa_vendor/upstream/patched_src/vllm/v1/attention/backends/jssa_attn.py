# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSSA (Joint Subspace Sparse Attention) backend for GQA decoder layers.

Decode step for one layer (q_len == 1 per request):

  1. q_bar = mean of the G query heads that share a KV head      [H_kv, d]
  2. proj_q = M q_bar, with M the calibrated [H_kv, r, d] basis   [H_kv, r]
  3. score[t] = proj_q . proj_K[t] for every cached token t, where
     proj_K = M k is written once per token at KV-cache-update time
  4. top-B tokens per (request, KV head)   (radix top-k)
  5. exact full-dimensional attention of every query head over the B tokens
     selected for its KV head (gather-free indexed kernel, or gather + flash)

Prefill and mixed prefill/decode batches run dense attention. Layers without a
basis entry (e.g. the sliding-window layers of hybrid models) and sliding-window
layers run dense attention over their own window.

KV cache layout: one combined slot per (token, KV head),
  [ K (d) | V (d) | proj_K (r) | pad ]            bf16 projected-key cache
  [ K (d) | V (d) ]  + separate [.., H_kv, r] fp8  fp8 projected-key cache
proj_K lives in the slot unless ``jssa_fp8_projk`` is set, in which case it is
stored in a separate contiguous FP8 cache (coalesced score scan). For
head_size > 256 the slot keeps its (unused) proj_K region in FP8 mode as well,
so that the KV spec in ``attention.py`` and the backend agree on the slot size.

Naming: ``n_fac`` is the token budget B and ``cs_h`` the subspace rank r.

The all-decode branch is the one captured under CUDA graphs
(``UNIFORM_SINGLE_TOKEN_DECODE``); every buffer it touches is pre-allocated
and ``mark_static_address``-ed so replays see stable pointers.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_jssa_gather import jssa_gather
from vllm.v1.attention.ops.triton_jssa_score_topk import jssa_score_topk
from vllm.v1.attention.ops.triton_jssa_store import (
    jssa_project_and_store,
    jssa_project_and_store_contig_fp8,
    jssa_store,
)

# vLLM's Triton flash attention: head_dim-agnostic (Gemma 4 full-attention
# layers use head_dim=512, above FlashAttention's 256 cap) and O(L) memory.
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()


def default_rank(head_size: int) -> int:
    """Default subspace rank r = head_size // 4 (r = 32 for d = 128)."""
    return head_size // 4


_DECODE_SCRATCH: dict = {}


def _decode_scratch(name, shape, dtype, device):
    """Per-shape persistent scratch for sparse-decode intermediates. Fresh
    allocations inside the captured decode path come from the CUDA-graph pool
    and can alias captured memory at batch > 1; reusing a stable per-shape
    buffer with out=/copy_ keeps the decode path allocation-free."""
    key = (name, tuple(shape), dtype, device)
    t = _DECODE_SCRATCH.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _DECODE_SCRATCH[key] = t
    return t


def _resolve_rank(head_size: int) -> int:
    """Projection rank r: the ``jssa_rank`` override (e.g. 64 for Gemma 4) or
    head_size // 4."""
    cfg = get_current_vllm_config()
    override = getattr(cfg.attention_config, "jssa_rank", None)
    return override if override else default_rank(head_size)


def fp8_projk_in_separate_cache(fp8_projk: bool, head_size: int) -> bool:
    """True iff proj_K is NOT stored in the combined slot (FP8 mode, head<=256)."""
    return fp8_projk and head_size <= 256


def _resolve_slot_rank(head_size: int) -> int:
    """Width of the proj_K region inside the combined slot (0 when proj_K lives
    in the separate FP8 cache)."""
    cfg = get_current_vllm_config()
    if fp8_projk_in_separate_cache(
        getattr(cfg.attention_config, "jssa_fp8_projk", False), head_size
    ):
        return 0
    return _resolve_rank(head_size)


def align_slot_elems(slot_elems: int) -> int:
    """Pad the combined slot to a 16-byte boundary (8 bf16 elements):
    flash_attn_varlen_func reads K/V with row stride = slot size and requires
    16-byte-aligned rows."""
    return ((slot_elems + 7) // 8) * 8


class JSSAAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # "jssa" routes through JSSAFullAttentionSpec (attention.py) so vLLM's
    # memory budgeter sizes the cache with the combined slot size.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["jssa"]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "jssa"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "JSSA"

    @staticmethod
    def get_impl_cls() -> type["JSSAImpl"]:
        return JSSAImpl

    @staticmethod
    def get_builder_cls() -> type["JSSAMetadataBuilder"]:
        return JSSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("JSSA: block_size must be a multiple of 16.")
        slot_size = align_slot_elems(2 * head_size + _resolve_slot_rank(head_size))
        return (num_blocks, block_size, num_kv_heads, slot_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        from vllm.v1.attention.backends.fa_utils import flash_attn_supports_sinks

        return flash_attn_supports_sinks()

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 64/128: Llama / Qwen / Mistral family. 256/512: Gemma 4 26B-A4B
        # (sliding layers head_dim=256, full-attention layers head_dim=512).
        return [64, 128, 256, 512]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Gemma 4 flags its full-attention layers with use_mm_prefix. For text
        # decoding this reduces to ordinary combined-slot full attention.
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class JSSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # With reorder_batch_threshold=1 the scheduler places decode requests
    # first; the sparse path operates on [:num_decodes].
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    causal: bool = True
    # Views into the builder's static buffers, sized to this step's decode
    # batch (None for all-prefill batches).
    scores_buf: torch.Tensor | None = None
    top_idx_buf: torch.Tensor | None = None
    top_scores_buf: torch.Tensor | None = None
    cu_seqlens_q_dec: torch.Tensor | None = None
    cu_seqlens_k_dec: torch.Tensor | None = None
    # Pre-allocated gather output buffers (gather + flash path).
    K_sel_buf: torch.Tensor | None = None
    V_sel_buf: torch.Tensor | None = None


class JSSAMetadataBuilder(AttentionMetadataBuilder[JSSAMetadata]):
    # Decode-only CUDA-graph capture. Mixed prefill/decode batches take a
    # Python-side dense branch and are never captured.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Threshold=1: the scheduler reorders so all q_len==1 requests come first.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = kv_cache_spec.head_size
        self.block_size = kv_cache_spec.block_size
        self._device = device
        self.n_fac = vllm_config.attention_config.jssa_budget

        # Radix top-k writes int32 indices (torch.topk fallback is copied in).
        self._top_idx_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, self.n_fac),
            dtype=torch.int32,
            device=device,
        )
        self._top_scores_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, self.n_fac),
            dtype=torch.float32,
            device=device,
        )
        # Decode cu_seqlens: q = [0, 1, ..., N]; the gathered K/V holds n_fac
        # rows per request, so k = that arange scaled by n_fac.
        cu_q = torch.arange(self.max_num_reqs + 1, dtype=torch.int32, device=device)
        self._cu_seqlens_q_dec = cu_q
        self._cu_seqlens_k_dec = (cu_q * self.n_fac).to(torch.int32)

        # The scores buffer depends on the runtime block-table width, so it is
        # sized lazily on the first build() (before CUDA-graph capture).
        self._scores_buf: torch.Tensor | None = None
        self._scores_max_kv_len = 0

        # Gather output buffers sized for the worst-case decode batch. These
        # bypass the shared WorkspaceManager: the post-capture warmup grows that
        # workspace to a new base address, which would leave captured graphs
        # pointing at freed memory.
        kv_dtype = vllm_config.model_config.dtype
        sel_shape = (
            self.max_num_reqs * self.n_fac,
            self.num_kv_heads,
            self.head_size,
        )
        self._K_sel_buf = torch.empty(sel_shape, dtype=kv_dtype, device=device)
        self._V_sel_buf = torch.empty(sel_shape, dtype=kv_dtype, device=device)

        for buf in (
            self._top_idx_buf,
            self._top_scores_buf,
            self._cu_seqlens_q_dec,
            self._cu_seqlens_k_dec,
            self._K_sel_buf,
            self._V_sel_buf,
        ):
            torch._dynamo.mark_static_address(buf)

    def _ensure_scores_buf(self, max_kv_len: int) -> torch.Tensor:
        """Allocate (or grow) the static scores buffer. The first decode batch is
        seen during warmup, before capture; afterwards the buffer never moves."""
        if self._scores_buf is not None and self._scores_max_kv_len >= max_kv_len:
            return self._scores_buf
        aligned = ((max_kv_len + 127) // 128) * 128
        self._scores_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, aligned),
            dtype=torch.float32,
            device=self._device,
        )
        self._scores_max_kv_len = aligned
        torch._dynamo.mark_static_address(self._scores_buf)
        return self._scores_buf

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> JSSAMetadata:
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _num_prefills, num_decode_tokens, _num_prefill_tokens = (
            split_decodes_and_prefills(
                cam, decode_threshold=self.reorder_batch_threshold
            )
        )

        if num_decodes > 0:
            runtime_max_kv_len = cam.block_table_tensor.shape[1] * self.block_size
            scores_buf = self._ensure_scores_buf(runtime_max_kv_len)[:num_decodes]
            top_idx_buf = self._top_idx_buf[:num_decodes]
            top_scores_buf = self._top_scores_buf[:num_decodes]
            cu_seqlens_q_dec = self._cu_seqlens_q_dec[: num_decodes + 1]
            cu_seqlens_k_dec = self._cu_seqlens_k_dec[: num_decodes + 1]
            # jssa_gather slices these to num_decodes * n_fac rows.
            K_sel_buf = self._K_sel_buf
            V_sel_buf = self._V_sel_buf
        else:
            scores_buf = None
            top_idx_buf = None
            top_scores_buf = None
            cu_seqlens_q_dec = None
            cu_seqlens_k_dec = None
            K_sel_buf = None
            V_sel_buf = None

        return JSSAMetadata(
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            query_start_loc=cam.query_start_loc,
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            causal=cam.causal,
            scores_buf=scores_buf,
            top_idx_buf=top_idx_buf,
            top_scores_buf=top_scores_buf,
            cu_seqlens_q_dec=cu_seqlens_q_dec,
            cu_seqlens_k_dec=cu_seqlens_k_dec,
            K_sel_buf=K_sel_buf,
            V_sel_buf=V_sel_buf,
        )


class JSSAImpl(AttentionImpl[JSSAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        attn_cfg = get_current_vllm_config().attention_config
        self._basis_path = attn_cfg.jssa_basis_path
        self.n_fac = attn_cfg.jssa_budget
        self.cs_h = _resolve_rank(head_size)
        self.slot_size = align_slot_elems(2 * head_size + _resolve_slot_rank(head_size))
        # FP8 projected-key cache: stored in a separate contiguous
        # [num_blocks, block_size, H_kv, r] e4m3 cache (also for head 512,
        # where the slot still carries an unused bf16 proj_K region).
        self.fp8_projk = bool(getattr(attn_cfg, "jssa_fp8_projk", False)) and (
            head_size <= 512
        )
        # Gather-free indexed attention (fused selection-gather + attend). The
        # decode-time gate below falls back to gather + flash for sinks,
        # softcap, alibi, a non-power-of-2 GQA group, or the eager
        # seq_len < B tail.
        self.indexed_attend = bool(getattr(attn_cfg, "jssa_indexed_attend", True))
        self.indexed_min_batch = int(getattr(attn_cfg, "jssa_indexed_min_batch", 1) or 1)
        self._sm_count = (
            torch.cuda.get_device_properties(0).multi_processor_count
            if torch.cuda.is_available()
            else 132
        )
        self._projk_cache: torch.Tensor | None = None  # lazy: needs num_blocks
        self._projk_scale: torch.Tensor | None = None  # [H_kv, r] fp8 scale
        self._M_dict_cpu: dict[int, torch.Tensor] | None = None  # lazy load
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # Learned attention sinks (softmax([qk; sink])); None for most models.
        self.sinks = kwargs.get("sinks")
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn.flash_attn_interface import DEFAULT_FA_VERSION

        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                f"sinks {tuple(self.sinks.shape)} vs num_heads {num_heads}"
            )
            self._fa_version = get_flash_attn_version(has_sinks=True)
        else:
            self._fa_version = DEFAULT_FA_VERSION

    def _ensure_on_device(
        self, layer: torch.nn.Module, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Lazily load this layer's basis M [H_kv, r, head_size] (cached on the
        layer). Returns None for layers absent from the basis file, which then
        run dense attention (e.g. the sliding-window layers of Gemma 4)."""
        cached = getattr(layer, "_jssa_M", None)
        if cached is not None:
            return cached
        if getattr(layer, "_jssa_dense", False):
            return None

        if self._basis_path is None:
            raise RuntimeError(
                "JSSA backend requires attention_config.jssa_basis_path "
                "(a calibration .pt file {'M': {layer_idx: Tensor[H_kv, r, d]}})."
            )

        if self._M_dict_cpu is None:
            ckpt = torch.load(self._basis_path, map_location="cpu", weights_only=False)
            self._M_dict_cpu = ckpt["M"] if isinstance(ckpt, dict) and "M" in ckpt else ckpt
            if not isinstance(self._M_dict_cpu, dict):
                raise RuntimeError(
                    f"JSSA basis at {self._basis_path} does not contain a "
                    "{layer_idx: tensor} dict at key 'M'."
                )

        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(layer.layer_name)
        M_cpu = self._M_dict_cpu.get(layer_idx)
        if M_cpu is None:
            layer._jssa_dense = True
            return None

        # Tensor parallel: the basis covers ALL kv heads; vLLM shards kv heads
        # as contiguous slices per rank, so take this rank's slice.
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and M_cpu.shape[0] == self.num_kv_heads * tp_size:
            tp_rank = get_tensor_model_parallel_rank()
            M_cpu = M_cpu[tp_rank * self.num_kv_heads : (tp_rank + 1) * self.num_kv_heads]
        expected_shape = (self.num_kv_heads, self.cs_h, self.head_size)
        if tuple(M_cpu.shape) != expected_shape:
            raise RuntimeError(
                f"JSSA basis M[{layer_idx}] shape {tuple(M_cpu.shape)} != "
                f"expected {expected_shape} (H_kv={self.num_kv_heads}, "
                f"r={self.cs_h}, d={self.head_size})."
            )

        M = M_cpu.to(device=device, dtype=dtype).contiguous()
        # Layer-static: a stable address lets the captured decode graph keep it.
        torch._dynamo.mark_static_address(M)
        layer._jssa_M = M
        if self.fp8_projk and self._projk_scale is None:
            # Per-(head, channel) fp8 scale = row norm of M, folded into proj_q
            # at decode so the score recovers proj_q . proj_K up to fp8 rounding.
            scale = M.float().norm(dim=2).clamp_min(1e-6).to(dtype).contiguous()
            torch._dynamo.mark_static_address(scale)
            self._projk_scale = scale  # [H_kv, r]
        return M

    def _dense_varlen(self, query, key_cache, value_cache, output, attn_metadata,
                      q_slice, req_slice, cu_seqlens_q):
        """Dense paged attention for prefill rows (unified Triton attention for
        head_size > 256, FlashAttention otherwise)."""
        if self.head_size > 256:
            unified_attention(
                q=query[q_slice],
                k=key_cache,
                v=value_cache,
                out=output[q_slice],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens[req_slice],
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                block_table=attn_metadata.block_table[req_slice],
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=self.alibi_slopes,
            )
            return
        flash_attn_varlen_func(
            q=query[q_slice],
            k=key_cache,
            v=value_cache,
            out=output[q_slice],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens[req_slice],
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            block_table=attn_metadata.block_table[req_slice],
            softcap=self.logits_soft_cap,
            s_aux=self.sinks,
            fa_version=self._fa_version,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: JSSAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("JSSA: fused output quantization is not supported.")
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        head_size = self.head_size
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # K = slot[:d], V = slot[d:2d] (strided views; the attention kernels
        # are stride-aware).
        key_cache = kv_cache[..., :head_size]
        value_cache = kv_cache[..., head_size : 2 * head_size]

        M = self._ensure_on_device(layer, query.device, query.dtype)
        if M is None:
            # Layer without a basis: plain paged attention with its own window.
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window),
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                s_aux=self.sinks,
                fa_version=self._fa_version,
            )
            return output

        if num_decodes == 0:
            # All-prefill batch: dense attention.
            self._dense_varlen(
                query, key_cache, value_cache, output, attn_metadata,
                q_slice=slice(0, num_actual_tokens), req_slice=slice(None),
                cu_seqlens_q=attn_metadata.query_start_loc,
            )
            return output

        # Sparse decode for the first num_decodes requests (the all-decode case
        # is the CUDA-graph-captured path).
        self._sparse_decode_forward(query, kv_cache, attn_metadata, output, M)

        if num_decode_tokens < num_actual_tokens:
            # Mixed batch: dense attention for the trailing prefill rows.
            self._dense_varlen(
                query, key_cache, value_cache, output, attn_metadata,
                q_slice=slice(num_decode_tokens, num_actual_tokens),
                req_slice=slice(num_decodes, None),
                cu_seqlens_q=attn_metadata.query_start_loc[num_decodes:]
                - num_decode_tokens,
            )
        return output

    def _sparse_decode_forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: JSSAMetadata,
        output: torch.Tensor,
        M: torch.Tensor,
    ) -> None:
        """Group-mean q -> proj_q -> score + top-B -> exact attention on B tokens."""
        head_size = self.head_size
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        # supports_spec_as_decode=False guarantees one token per decode request.
        assert num_decode_tokens == num_decodes, (
            "JSSA sparse decode assumes q_len=1 per decode request "
            f"(num_decode_tokens={num_decode_tokens}, num_decodes={num_decodes})"
        )
        q_decode = query[:num_decode_tokens]  # (num_decodes, H_q, head_size)

        if self.sliding_window != (-1, -1):
            # Sliding-window layer that carries a basis: no selection, dense
            # windowed attention (the selected set would ignore the window).
            flash_attn_varlen_func(
                q=q_decode,
                k=kv_cache[..., :head_size],
                v=kv_cache[..., head_size : 2 * head_size],
                out=output[:num_decode_tokens],
                cu_seqlens_q=attn_metadata.cu_seqlens_q_dec,
                max_seqlen_q=1,
                seqused_k=attn_metadata.seq_lens[:num_decodes],
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=list(self.sliding_window),
                block_table=attn_metadata.block_table[:num_decodes],
                softcap=self.logits_soft_cap,
                s_aux=self.sinks,
                fa_version=self._fa_version,
            )
            return

        # 1. GQA group mean: H_q -> H_kv (query head h belongs to kv head h // G).
        q_kv = _decode_scratch(
            "qkv", (num_decodes, self.num_kv_heads, head_size),
            q_decode.dtype, q_decode.device,
        )
        torch.mean(
            q_decode.view(num_decodes, self.num_kv_heads, self.num_kv_groups, head_size),
            dim=2,
            out=q_kv,
        )

        # 2. proj_q[r, h, c] = sum_d q_kv[r, h, d] * M[h, c, d]  (per-kv-head bmm).
        projq_hrc = _decode_scratch(
            "projq", (self.num_kv_heads, num_decodes, self.cs_h),
            q_kv.dtype, q_kv.device,
        )
        torch.bmm(q_kv.transpose(0, 1), M.transpose(1, 2), out=projq_hrc)
        if self.fp8_projk:
            # Fold the fp8 scale into proj_q (in place, allocation-free).
            projq_hrc.mul_(self._projk_scale.unsqueeze(1))  # [H_kv, 1, r]
        proj_q = projq_hrc.transpose(0, 1)  # (num_decodes, H_kv, r) view

        # 3-4. Score every cached token, radix top-B per (request, kv head).
        block_table_dec = attn_metadata.block_table[:num_decodes]
        seq_lens_dec = attn_metadata.seq_lens[:num_decodes]
        top_idx, _ = jssa_score_topk(
            proj_q,
            kv_cache,
            block_table_dec,
            seq_lens_dec,
            n_fac=self.n_fac,
            head_size=head_size,
            cs_h=self.cs_h,
            scores_out=attn_metadata.scores_buf,
            top_idx_out=attn_metadata.top_idx_buf,
            top_scores_out=attn_metadata.top_scores_buf,
            use_radix=True,
            projk_cache=self._projk_cache,
        )  # (num_decodes, H_kv, n_fac_eff) int32

        n_fac_eff = top_idx.shape[-1]
        # seq_len < B: the radix pad cycles real tokens into the unused slots.
        # In eager mode, compact valid selections to the front and attend only
        # min(seq_len, B) rows, which is exact dense attention over the real
        # tokens. (Under CUDA-graph capture the host check is skipped; the
        # cycled duplicates are near-exact since equal multiplicity cancels.)
        partial = False
        if not torch.cuda.is_current_stream_capturing():
            partial = bool((seq_lens_dec < n_fac_eff).any().item())
        if partial:
            invalid = top_idx.long() >= seq_lens_dec.view(num_decodes, 1, 1)
            order = torch.argsort(invalid.to(torch.uint8), dim=-1, stable=True)
            top_idx = torch.gather(top_idx, -1, order)

        # 5a. Gather-free indexed attention.
        if (
            self.indexed_attend
            and num_decodes >= self.indexed_min_batch
            and head_size <= 512
            and not partial
            and self.sinks is None
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            # the kernel does tl.arange over the GQA group (power of 2 only;
            # Qwen3-14B has G = 5 and falls back to gather + flash).
            and (self.num_kv_groups & (self.num_kv_groups - 1)) == 0
        ):
            from vllm.v1.attention.ops.triton_jssa_indexed_attend import (
                jssa_indexed_attend,
            )

            jssa_indexed_attend(
                q_decode, kv_cache, block_table_dec, top_idx,
                head_size=head_size, scale=self.scale,
                out=output[:num_decode_tokens],
                scratch=_decode_scratch, sm_count=self._sm_count,
            )
            return

        # 5b. Gather + attend.
        K_sel, V_sel = jssa_gather(
            kv_cache,
            block_table_dec,
            top_idx,
            head_size=head_size,
            dtype=q_decode.dtype,
            K_sel_out=attn_metadata.K_sel_buf,
            V_sel_out=attn_metadata.V_sel_buf,
        )  # (num_decodes * n_fac_eff, H_kv, head_size)

        cu_seqlens_q = attn_metadata.cu_seqlens_q_dec
        if n_fac_eff == self.n_fac:
            cu_seqlens_k = attn_metadata.cu_seqlens_k_dec
        else:
            cu_seqlens_k = cu_seqlens_q * n_fac_eff

        if head_size > 256:
            # FlashAttention caps head_dim at 256 (Gemma 4 full layers are 512):
            # the selected set is small, so a manual GQA attention is cheap.
            nd = num_decodes
            nf = n_fac_eff
            Hkv = self.num_kv_heads
            G = self.num_kv_groups
            d = head_size
            K = K_sel.view(nd, nf, Hkv, d).permute(0, 2, 1, 3)  # (nd, Hkv, nf, d)
            V = V_sel.view(nd, nf, Hkv, d).permute(0, 2, 1, 3)
            q = q_decode.view(nd, Hkv, G, d).to(torch.float32)
            scores = torch.einsum("nhgd,nhfd->nhgf", q, K.to(torch.float32))
            scores = scores * self.scale
            if partial:
                tail = (
                    torch.arange(nf, device=scores.device).view(1, 1, 1, nf)
                    >= seq_lens_dec.view(nd, 1, 1, 1)
                )
                scores = scores.masked_fill(tail, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("nhgf,nhfd->nhgd", attn, V.to(torch.float32))
            output[:num_decode_tokens] = out.reshape(nd, Hkv * G, d).to(output.dtype)
            return

        if partial:
            # Attend only the first min(seq_len, B) gathered rows per request.
            # cu_seqlens_k and seqused_k are mutually exclusive, so present
            # K_sel as a paged cache with one n_fac_eff-sized page per request.
            flash_attn_varlen_func(
                q=q_decode,
                k=K_sel.view(num_decodes, n_fac_eff, self.num_kv_heads, head_size),
                v=V_sel.view(num_decodes, n_fac_eff, self.num_kv_heads, head_size),
                out=output[:num_decode_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seq_lens_dec.to(torch.int32).clamp(max=n_fac_eff),
                max_seqlen_k=n_fac_eff,
                block_table=torch.arange(
                    num_decodes, dtype=torch.int32, device=q_decode.device
                ).view(num_decodes, 1),
                softmax_scale=self.scale,
                causal=False,  # the selected tokens are a set, not a sequence
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                s_aux=self.sinks,
                fa_version=self._fa_version,
            )
            return
        flash_attn_varlen_func(
            q=q_decode,
            k=K_sel,
            v=V_sel,
            out=output[:num_decode_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=n_fac_eff,
            softmax_scale=self.scale,
            causal=False,  # the selected tokens are a set, not a sequence
            alibi_slopes=self.alibi_slopes,
            softcap=self.logits_soft_cap,
            s_aux=self.sinks,
            fa_version=self._fa_version,
        )

    def _ensure_projk_cache(
        self, kv_cache: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Lazily allocate the FP8 projected-key cache with this layer's block
        count. Re-sized if the block count changes: warmup passes may run on a
        small placeholder cache before the real KV cache exists. The final
        allocation happens in the eager warmup, before CUDA-graph capture.

        NOTE: this cache (r bytes per token and KV head) is allocated outside
        vLLM's KV-cache memory budget."""
        num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
        if self._projk_cache is None or self._projk_cache.shape[0] != num_blocks:
            self._projk_cache = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.cs_h,
                dtype=torch.float8_e4m3fn, device=device,
            )
            try:
                torch._dynamo.mark_static_address(self._projk_cache)
            except Exception:
                pass
        return self._projk_cache

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Fused write of K, V, and proj_K = M @ K.
        M = self._ensure_on_device(layer, key.device, key.dtype)
        if M is None:
            jssa_store(key, value, kv_cache, slot_mapping)
        elif self.fp8_projk:
            pk = self._ensure_projk_cache(kv_cache, key.device)
            jssa_project_and_store_contig_fp8(
                key, value, kv_cache, pk, self._projk_scale, slot_mapping, M
            )
        else:
            jssa_project_and_store(key, value, kv_cache, slot_mapping, M)
