"""Fixed-capacity selected-token interface; no full-history mask on the hot path."""
from typing import NamedTuple
import torch


class TokenSelection(NamedTuple):
    indices: torch.Tensor  # [B,Hkv,K], ascending valid prefix, zero padding
    counts: torch.Tensor   # [B,Hkv], int32

    def valid(self):
        return torch.arange(self.indices.shape[-1], device=self.indices.device) < self.counts[..., None]

    def to_mask(self, capacity):
        # Diagnostics/oracles only; append a sentinel so invalid writes cannot
        # overwrite a real selected token at position zero.
        out = torch.zeros((*self.indices.shape[:-1], capacity+1),
                          dtype=torch.bool, device=self.indices.device)
        out.scatter_(-1, torch.where(self.valid(), self.indices.long(), capacity), True)
        return out[..., :capacity].unsqueeze(1)


def canonical_selection(indices, valid, capacity, width=None):
    """Union/deduplicate/sort a bounded slot list, preserving mask-scan order."""
    width = min(indices.shape[-1], capacity) if width is None else min(width, capacity)
    valid = valid & (indices >= 0) & (indices < capacity)
    ordered = torch.where(valid, indices, capacity).sort(-1).values
    previous = torch.cat((torch.full_like(ordered[..., :1], -1), ordered[..., :-1]), -1)
    unique = (ordered < capacity) & (ordered != previous)
    packed = torch.where(unique, ordered, capacity).sort(-1).values[..., :width]
    counts = (packed < capacity).sum(-1, dtype=torch.int32)
    return TokenSelection(torch.where(packed < capacity, packed, 0).to(torch.int32).contiguous(),
                          counts.contiguous())


def graph_selection(top, k2_mid, mid_mask, length, sink, swa, budget, capacity):
    h, n = top.shape
    ranks = torch.arange(n, device=top.device)
    keep = (ranks < k2_mid) & mid_mask[top] & (top < length)
    sink_ids = torch.arange(min(sink, capacity), device=top.device).expand(h, -1)
    window = ((length-swa).clamp(min=0) +
              torch.arange(min(swa, capacity), device=top.device)).expand(h, -1)
    ids = torch.cat((top, sink_ids, window), -1)
    valid = torch.cat((keep, sink_ids < length, window < length), -1)
    return canonical_selection(ids[None], valid[None], capacity,
                               max(budget, sink+swa))


def selected_attention(q, k, v, selection, scale):
    """Q[B,1,Hq,D], KV[B,S,Hkv,D]; direct existing Triton attention or SDPA."""
    from . import eager_decoding as eager
    b, _, hq, d = q.shape
    hkv, width = k.shape[2], selection.indices.shape[-1]
    if (q.is_cuda and eager._HAS_TRITON and d > 0 and d & (d-1) == 0
            and k.shape == v.shape and k.stride() == v.stride()
            and k.stride(-1) == 1 and v.stride(-1) == 1):
        qs = (q[:, 0]*scale).float().contiguous()
        out = torch.empty((b, hq, d), device=q.device, dtype=q.dtype)
        eager._eager_attn_kernel_b[(b*hq,)](
            selection.indices, selection.counts, qs, k, v, out, hkv,
            qs.stride(0), k.stride(0), k.stride(1), k.stride(2),
            qs.stride(1), out.stride(0),
            G=hq//hkv, D=d, PER_QH=False, BW=256, W=width, num_warps=4)
        return out[:, None]
    rows = torch.arange(b, device=q.device)[:, None, None]
    heads = torch.arange(hkv, device=q.device)[None, :, None]
    valid = selection.valid()
    kk = k[rows, selection.indices.long(), heads].masked_fill(~valid[..., None], 0)
    vv = v[rows, selection.indices.long(), heads].masked_fill(~valid[..., None], 0)
    out = torch.nn.functional.scaled_dot_product_attention(
        q[:, 0].reshape(b, hkv, hq//hkv, d), kk, vv,
        attn_mask=valid[:, :, None], scale=scale, dropout_p=0.0)
    return out.reshape(b, 1, hq, v.shape[-1])
