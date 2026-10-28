"""Bounded token IDs for decode. IDs are logical positions, never ring slots."""
from typing import NamedTuple
import torch


class TokenSelection(NamedTuple):
    indices: torch.Tensor  # [B,Hkv,K], sorted unique valid prefix, int32
    counts: torch.Tensor   # [B,Hkv], int32, device resident

    def valid(self):
        return torch.arange(self.indices.shape[-1], device=self.indices.device) < self.counts[..., None]

    def to_mask(self, capacity):
        """Diagnostic/compatibility adapter, never called by the new consumer."""
        result = torch.zeros((*self.indices.shape[:-1], capacity + 1),
                             dtype=torch.bool, device=self.indices.device)
        ids = torch.where(self.valid(), self.indices.long(), capacity)
        result.scatter_(-1, ids, True)
        return result[..., :capacity].unsqueeze(1)


def canonical_selection(ids, valid, capacity, width=None):
    """Preserve the old mask-scan order and deduplication, on a bounded domain.

    Two sorts are a correctness-first adapter, not a claimed optimal compactor.
    The caller must not truncate more unique valid IDs than its budget allows.
    """
    if ids.ndim != 3 or valid.shape != ids.shape or capacity < 1:
        raise ValueError('selection requires [B,Hkv,K] IDs/valid and positive capacity')
    width = ids.shape[-1] if width is None else min(width, ids.shape[-1])
    ordered = torch.where(valid & (ids >= 0) & (ids < capacity), ids, capacity).sort(-1).values
    previous = torch.cat((torch.full_like(ordered[..., :1], -1), ordered[..., :-1]), -1)
    unique = (ordered < capacity) & (ordered != previous)
    packed = torch.where(unique, ordered, capacity).sort(-1).values[..., :width]
    count = (packed < capacity).sum(-1, dtype=torch.int32)
    return TokenSelection(torch.where(packed < capacity, packed, 0).to(torch.int32).contiguous(),
                          count.contiguous())


def middle_selection(ids, valid, lengths, sink, swa, capacity, width=None):
    """Remove protected tokens from consumption, without changing L2 scoring."""
    lengths = lengths.reshape(-1, 1, 1)
    lo = lengths.clamp(max=sink)
    hi = torch.maximum(lo, lengths - swa)
    keep = valid & (ids >= lo) & (ids < hi) & (ids < lengths)
    return canonical_selection(ids, keep, capacity, width)


def graph_middle_selection(top, k2_mid, mid_mask, length, capacity, sink, swa):
    # Match the old topk -> rank gate -> scatter -> mid/length mask exactly.
    ranks = torch.arange(top.shape[-1], device=top.device)
    in_range = (top >= 0) & (top < capacity)
    safe = top.clamp(0, capacity - 1)
    keep = (ranks < k2_mid) & in_range & mid_mask[safe] & (top < length)
    return middle_selection(top[None], keep[None], length.reshape(1), sink, swa, capacity)
