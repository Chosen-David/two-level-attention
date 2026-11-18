"""JSSA 单级论文性能语义：NFA 全量 32 维 score + Top-K。

复用 JSSA 官方 project/store、FP8 projected-K 与 indexed-attend，只把选择器
替换为 NFA 的 fine-score 和 filtered-Top-K kernel。候选始终是全部逻辑 token；
不经过 NFA L1、页统计、region gate 或候选块粗筛。
"""
from types import SimpleNamespace

import torch

from exp.vendor_baselines.jssa_vendor.qwen3_jssa_official import JSSAQwen3Runtime
from sparse_attn.indexer.nfa_layer_batch_selector import (
    _fused_fine_scores_paged,
    _nfa_filtered_topk,
)


class JSSANFASemanticRuntime(JSSAQwen3Runtime):
    """固定页布局下可直接 CUDA Graph capture 的 JSSA/NFA 选择器。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rank != 32:
            raise ValueError("JSSA/NFA semantic arm requires the paper rank=32")
        if self.budget <= 0 or self.budget & (self.budget - 1):
            raise ValueError("NFA filtered Top-K requires a power-of-two budget")
        device = self.device
        batch = self.batch_size
        width = self.max_kv_len
        blocks = (width + self.page_size - 1) // self.page_size

        layers = [
            SimpleNamespace(
                num_kv_heads=self.num_kv_heads,
                head_dim_k=self.rank,
                head_dim_v=self.rank,
                kv_pool_k=self.projk_cache[layer],
            )
            for layer in range(self.num_layers)
        ]
        self._nfa_cache = SimpleNamespace(
            layers=layers,
            layer_kv_ptrs_buf=torch.tensor(
                [self.projk_cache[layer].data_ptr()
                 for layer in range(self.num_layers)],
                dtype=torch.int64, device=device),
            block_table_buf=self.block_table.unsqueeze(0).repeat(
                self.num_layers, 1, 1).contiguous(),
            geometry=SimpleNamespace(page=self.page_size),
            max_blocks=blocks,
            max_requests=batch,
        )
        self._nfa_req_ids = torch.arange(batch, dtype=torch.int64, device=device)
        self._nfa_dims = torch.arange(self.rank, dtype=torch.int64, device=device)
        self._nfa_candidates = torch.arange(
            width, dtype=torch.int32, device=device).view(1, 1, width).repeat(
                batch, self.num_kv_heads, 1)
        self._nfa_keep = torch.ones(
            (batch, self.num_kv_heads, width), dtype=torch.bool, device=device)
        owner_blocks = torch.arange(
            (width + self.page_size - 1) // self.page_size,
            dtype=torch.int32, device=device)
        # fine-score 的 owner 标签只用于去重；全量候选本来唯一，预先赋予每页
        # 与 generation=1、pack_width=max_kv_len 一致的固定标签。
        owner_values = width + owner_blocks * self.page_size + self.page_size - 1
        self._nfa_owners = owner_values.unsqueeze(0).repeat(
            batch * self.num_kv_heads, 1)
        self._nfa_generations = torch.ones(
            (batch, self.num_kv_heads), dtype=torch.int32, device=device)
        self._nfa_lower = torch.zeros(batch, dtype=torch.int32, device=device)
        self._nfa_quota = torch.full(
            (batch,), self.budget, dtype=torch.int32, device=device)
        self._nfa_workspace = {}

    def _score_topk(self, layer_idx):
        logits = _fused_fine_scores_paged(
            self._nfa_cache, self._nfa_req_ids, 1, self._nfa_dims,
            self.projq_hrc.transpose(0, 1), self._nfa_candidates,
            self._nfa_keep, self._nfa_owners, self._nfa_generations,
            self.max_kv_len, self._nfa_lower, self.seq_lens, self._nfa_quota,
            tile=256, num_warps=4, layer_offset=layer_idx,
            workspace=self._nfa_workspace, bs=self.page_size,
            kv_position_major=True)
        self._nfa_logits = logits
        selection = _nfa_filtered_topk(
            logits, self._nfa_candidates, self._nfa_lower, self.seq_lens,
            self._nfa_quota, self.max_kv_len, self.budget, sink=0,
            layer_offset=layer_idx, workspace=self._nfa_workspace,
            merge_dynamic=True)
        self._nfa_top_idx = selection.indices
        return selection.indices

    def receipt(self):
        result = super().receipt()
        result.update(
            implementation="JSSA paper performance semantics with NFA selector kernels",
            selection_implementation="nfa-full-candidate-score-topk",
            score_kernel="_nfa_fine_scores_paged_kernel",
            score_candidate_scope="every logical token in [0, seq_len), no L1 pruning",
            score_dtype="FP8 projected-K and BF16 projected-Q, FP32 dot, BF16 logits",
            topk="NFA exact two-stage filtered Top-K, sorted logical-token output",
            topk_budget_per_kv_head=self.budget,
            nfa_l1_or_region_gate=False,
            attention_kernel="official gather-free online-softmax indexed-attend",
        )
        return result
