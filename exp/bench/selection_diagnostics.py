"""Opt-in eager decode diagnostics; never import from production attention.

Full-dimensional FP32 attention is a diagnostic reference, not an exact
replica of FlashAttention or a quality metric. Instrumented runs cannot be
used for latency claims. Records contain no Q/K tensors or prompt text.
"""
import json
from contextlib import contextmanager

import torch


@torch.no_grad()
def selection_mass(q, k, mask, scale, valid_length=None):
    """B1 single-token Q/K in [B,S,H,D], token mask [1,1,Hkv,S]."""
    if q.ndim != 4 or k.ndim != 4 or q.shape[:2] != (1, 1) or k.shape[0] != 1:
        raise ValueError("diagnostic supports B1 single-token decode only")
    length = k.shape[1] if valid_length is None else int(valid_length)
    h, d = q.shape[-2:]
    hk = k.shape[2]
    if not 0 < length <= k.shape[1] or h % hk or k.shape[-1] != d:
        raise ValueError("invalid length or GQA geometry")
    if mask.dtype != torch.bool or mask.shape != (1, 1, hk, k.shape[1]):
        raise ValueError("expected a boolean token mask matching KV capacity")
    if not torch.isfinite(q).all() or not torch.isfinite(k[:, :length]).all():
        raise ValueError("nonfinite Q/K")
    # Group consecutive query heads by their shared KV head. No repeated K.
    logits = torch.einsum("hgd,shd->hgs", q[0, 0].float().reshape(hk, h // hk, d),
                          k[0, :length].float()) * scale
    probs = logits.softmax(-1)
    selected = mask[0, 0, :, :length]
    counts = selected.sum(-1)
    mass = (probs * selected[:, None]).sum(-1)
    # Same per-KV-head token count oracle maximizes mean query-head mass.
    grouped = probs.mean(1)
    oracle = []
    for row, count in zip(grouped, counts.tolist()):
        oracle.append(row.topk(count).values.sum().item() if count else 0.0)
    return dict(valid_length=length, capacity=k.shape[1], selected_counts=counts.tolist(),
                retained_mass_by_query_head=mass.flatten().tolist(),
                oracle_mass_by_kv_head=oracle,
                peak_token_by_query_head=probs.argmax(-1).flatten().tolist(),
                query_heads=h, kv_heads=hk,
                diagnostic="full_dim_fp32_softmax_not_production_bitwise")


@contextmanager
def record_selection_mass(path, steps_per_request=1):
    """Temporarily observe TLI eager prepare_mask, restoring methods on failure.

    clear() defines request boundaries per layer. Graph paths are intentionally
    unsupported: this probe performs host readbacks and must run with Graph off.
    """
    import os
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    if os.environ.get("TLI_CUDA_GRAPH") not in ("0", "off"):
        raise ValueError("selection diagnostic requires explicit Graph off")
    if steps_per_request < 1:
        raise ValueError("steps_per_request must be positive")
    original_prepare = TLIIndexer.prepare_mask
    original_clear = TLIIndexer.clear
    had_prepare = "prepare_mask" in TLIIndexer.__dict__
    state = {}
    with open(path, "x", encoding="utf-8") as output:
        def clear(self, *args, **kwargs):
            result = original_clear(self, *args, **kwargs)
            previous = state.get(self, (-1, 0))
            state[self] = (previous[0] + 1, 0)
            return result

        def prepare(self, q, q_ids, k, cu_seqlens_k, softmax_scale=None, valid_length=None):
            result = original_prepare(self, q, q_ids, k, cu_seqlens_k,
                                      softmax_scale=softmax_scale, valid_length=valid_length)
            request, step = state.get(self, (0, 0))
            if step < steps_per_request:
                mask, block_size = result
                if block_size != 1:
                    raise ValueError("diagnostic requires token-level selection")
                row = selection_mass(q, k, mask, q.shape[-1] ** -0.5
                                     if softmax_scale is None else softmax_scale, valid_length)
                row.update(layer=self.layer_idx, request=request, step=step)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
            state[self] = (request, step + 1)
            return result

        TLIIndexer.clear = clear
        TLIIndexer.prepare_mask = prepare
        try:
            yield
        finally:
            TLIIndexer.clear = original_clear
            if had_prepare:
                TLIIndexer.prepare_mask = original_prepare
            else:
                del TLIIndexer.prepare_mask
