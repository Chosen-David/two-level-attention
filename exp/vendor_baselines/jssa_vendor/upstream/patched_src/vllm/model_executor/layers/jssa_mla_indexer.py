# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSSA token selection for Multi-head Latent Attention (MLA).

Drop-in replacement for the DeepSeek-V3.2 lightning indexer: it writes the
top-B token indices of every query into the shared ``topk_indices_buffer``
through the same FP8 paged-MQA-logits + radix top-k op
(``SparseAttnIndexer``), which the FlashMLA sparse attention consumes unchanged.
Only the scoring representation differs:

    index_q[h] = [ M W_UK[h]^T q_nope[h] | q_pe[h] ]
    index_k    = [ M c_KV                | k_pe    ]

so ``sum_h index_q[h] . index_k`` = (M sum_h q_latent[h]) . (M c_KV) + the
decoupled-RoPE term, with q_latent[h] = W_UK[h]^T q_nope[h] the absorbed query.
M [r, kv_lora_rank] is the calibrated per-layer basis (r = 64 latent dims plus
64 RoPE dims = 128, the DSA index head dim). The selection is one token set per
layer (MLA has a single shared latent KV head).

NOTE: the DeepGEMM MQA-logits epilogue applies a per-head ReLU (DSA's indexer
is trained with it). JSSA's score is the plain head sum, so the ReLU must be
compiled out (see the DeepGEMM header patch shipped with this code).

``JSSAMLAGramCollector`` is the calibration counterpart: attached to each MLA
layer in dense-attention mode, it accumulates the second moments of the
group-mean absorbed query and of c_KV, from which the basis is computed offline.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.utils import extract_layer_index


def _w_uk_bf16(owner, kv_b_proj, n_head, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    """bf16 W_UK (the qk_nope slice of kv_b_proj), [H_local, qk_nope, kv_lora].
    Computed once and cached on ``owner``.

    For FP8 checkpoints, vLLM's process_weights_after_loading rewrites the
    block scales into a kernel-specific format, so a manual
    ``weight * weight_scale_inv`` dequant is wrong. Instead pass the identity
    through the module's own linear: kv_b_proj(I) = W^T. H is TP-local
    (kv_b_proj is column-parallel)."""
    cached = getattr(owner, "_wuk_bf16", None)
    if cached is not None:
        return cached
    W = kv_b_proj.weight
    if W.dtype == torch.float8_e4m3fn:
        eye = torch.eye(kv_lora_rank, device=W.device, dtype=torch.bfloat16)
        Wt = kv_b_proj(eye)[0]  # [kv_lora, H_local * (nope + v)] = W^T
        Wbf = Wt.t().contiguous().to(torch.bfloat16)
    else:
        Wbf = W.to(torch.bfloat16)
    W_UK = Wbf.view(n_head, qk_nope_head_dim + v_head_dim, kv_lora_rank)[
        :, :qk_nope_head_dim, :
    ].contiguous()
    owner._wuk_bf16 = W_UK
    return W_UK


class JSSAMLAIndexer(nn.Module):
    # Tells the MLA wrapper to pass its already-computed q / c_KV / k_pe.
    is_jssa = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig,
        kv_b_proj: nn.Module,
        basis_path: str,
        cs_h: int,
        n_fac: int,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        # TP-local heads (q_b_proj / kv_b_proj are column-parallel).
        self.n_head = (
            config.num_attention_heads // vllm_config.parallel_config.tensor_parallel_size
        )
        # The DeepGEMM logits kernel supports 32 or 64 (padded) query heads.
        assert self.n_head <= 64, f"JSSA-MLA: {self.n_head} local heads > 64"
        self.cs_h = cs_h
        self.head_dim = cs_h + self.qk_rope_head_dim  # index dim (128 at r=64)
        self.softmax_scale = self.head_dim**-0.5
        self.topk_tokens = n_fac
        self.quant_block_size = 128
        self.scale_fmt = "ue8m0"
        self.kv_b_proj = kv_b_proj
        self.layer_idx = extract_layer_index(prefix)
        self.topk_indices_buffer = topk_indices_buffer  # read by the MLA wrapper

        # Load M at construction (torch.load is not traceable) straight onto the
        # worker's device: a non-persistent CPU buffer would not be moved by
        # weight loading and would force a copy inside the captured graph.
        if basis_path is None:
            raise RuntimeError("JSSA-MLA requires attention_config.jssa_basis_path.")
        ckpt = torch.load(basis_path, map_location="cpu", weights_only=False)
        M0 = ckpt["M"][self.layer_idx][0]  # [r, kv_lora_rank]
        if tuple(M0.shape) != (cs_h, self.kv_lora_rank):
            raise RuntimeError(
                f"JSSA-MLA basis M[{self.layer_idx}] shape {tuple(M0.shape)} != "
                f"expected {(cs_h, self.kv_lora_rank)}."
            )
        _dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.register_buffer(
            "M_basis", M0.to(torch.bfloat16).to(_dev).contiguous(), persistent=False
        )
        torch._dynamo.mark_static_address(self.M_basis)

        # Reuse the DSA FP8 index cache and the FP8 paged-logits / radix top-k op.
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
        )

    def forward(self, hidden_states, q, kv_c, k_pe):
        """q [T, H, nope + rope] (RoPE slice rotated), kv_c [T, kv_lora] (normed
        latent), k_pe [T, 1, rope] (rotated)."""
        T = q.shape[0]
        H = self.n_head
        q_nope = q[..., : self.qk_nope_head_dim]  # [T, H, nope]
        q_pe = q[..., self.qk_nope_head_dim :]  # [T, H, rope]
        k_pe = k_pe.reshape(T, self.qk_rope_head_dim)
        Mb = self.M_basis
        # Fused query projection: proj_q = q_nope @ (W_UK M^T), with
        # WM = W_UK @ M^T [H, nope, r] cached (identical by associativity to
        # projecting the [T, H, kv_lora] absorbed query).
        WM = getattr(self, "_wm", None)
        if WM is None:
            W_UK = _w_uk_bf16(
                self, self.kv_b_proj, H, self.qk_nope_head_dim, self.v_head_dim,
                self.kv_lora_rank,
            )
            WM = torch.einsum("hnl,cl->hnc", W_UK, Mb).contiguous()
            self._wm = WM
        proj_q = torch.einsum("thn,hnc->thc", q_nope, WM)  # [T, H, r]
        proj_K = kv_c @ Mb.T  # [T, r]
        index_q = torch.cat([proj_q, q_pe], dim=-1)  # [T, H, head_dim]
        index_k = torch.cat([proj_K, k_pe], dim=-1)  # [T, head_dim]
        # Zero-pad the query heads to 32/64 (kernel constraint); padded heads
        # get weight 0, so the score is exactly the sum over the H real heads.
        H_pad = 32 if H <= 32 else 64
        if H_pad != H:
            index_q = torch.nn.functional.pad(index_q, (0, 0, 0, H_pad - H))
        qf = index_q.reshape(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            qf, self.quant_block_size, column_major_scales=False, use_ue8m0=True
        )
        q_fp8 = q_fp8.view(T, H_pad, self.head_dim)
        q_scale = q_scale.view(T, H_pad, 1)
        # Uniform head weights (the fp8 q scale is folded in).
        weights = (q_scale * self.softmax_scale * (H**-0.5)).squeeze(-1)  # [T, H_pad]
        if H_pad != H:
            weights[:, H:] = 0
        return self.indexer_op(hidden_states, q_fp8, index_k, weights)


class JSSAMLAGramCollector(nn.Module):
    """Calibration-time accumulator for one MLA layer (dense attention, eager).

    Accumulates, over all tokens passed through the layer,
        G_Qm = sum_t q_bar_t q_bar_t^T,  q_bar_t = mean_h W_UK[h]^T q_nope[h, t]
        G_K  = sum_t c_t c_t^T
    in float64. q_bar is the mean over ALL attention heads: with tensor
    parallelism the per-rank head sums are all-reduced first, so every rank
    ends up with the same G_Qm. c_KV is replicated across ranks.
    """

    def __init__(self, config, kv_b_proj: nn.Module, layer_idx: int):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        self.tp_size = get_tensor_model_parallel_world_size()
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.n_head_total = config.num_attention_heads
        self.n_head = config.num_attention_heads // self.tp_size
        self.layer_idx = layer_idx
        # Plain reference (not a submodule): kv_b_proj is owned by the attention.
        self._kv_b_proj = (kv_b_proj,)
        self.g_q: torch.Tensor | None = None
        self.g_k: torch.Tensor | None = None
        self.num_tokens = 0

    def reset(self) -> None:
        self.g_q = None
        self.g_k = None
        self.num_tokens = 0

    @torch.no_grad()
    def forward(self, q: torch.Tensor, kv_c: torch.Tensor) -> None:
        T = q.shape[0]
        if T == 0:
            return
        W_UK = _w_uk_bf16(
            self, self._kv_b_proj[0], self.n_head, self.qk_nope_head_dim,
            self.v_head_dim, self.kv_lora_rank,
        )
        q_nope = q[..., : self.qk_nope_head_dim].float()
        head_sum = torch.einsum("thn,hnl->tl", q_nope, W_UK.float())  # [T, kv_lora]
        if self.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            head_sum = tensor_model_parallel_all_reduce(head_sum)
        q_bar = head_sum / self.n_head_total
        c = kv_c.float()
        if self.g_q is None:
            d = self.kv_lora_rank
            self.g_q = torch.zeros(d, d, dtype=torch.float64, device=q.device)
            self.g_k = torch.zeros(d, d, dtype=torch.float64, device=q.device)
        self.g_q += (q_bar.T @ q_bar).double()
        self.g_k += (c.T @ c).double()
        self.num_tokens += T
