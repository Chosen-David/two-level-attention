"""Strict far-off decode: bounded key reads and compact original-width attention.

This opt-in path implements near-page selection followed by pooled token selection.
Only protected sink/window tokens and the page-aligned near region are readable.
It deliberately excludes unprotected tokens on a far sink-boundary page. It is
not bitwise equivalent to the historical gate that allowed that page through.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FarSkipConfig:
    block_size: int
    near_length: int
    near_pages: int
    token_budget: int
    sink_tokens: int
    window_tokens: int
    near_method: str = "avg"
    quantize: bool = False

    def __post_init__(self):
        for name in ("block_size", "near_length", "near_pages", "token_budget",
                     "sink_tokens", "window_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in
                    ("block_size", "near_length", "token_budget") else 0):
                raise ValueError(f"invalid {name}: {value!r}")
        if self.token_budget < self.sink_tokens + self.window_tokens:
            raise ValueError("token budget must cover sink and window protection")
        if self.near_method not in ("avg", "minmax"):
            raise ValueError("compact far skip supports avg/minmax near scoring")

    def near_start(self, length):
        if type(length) is not int or length < 1:
            raise ValueError("length must be a positive host integer")
        bs = self.block_size
        return max(self.sink_tokens // bs,
                   (length - self.window_tokens - max(bs, self.near_length)) // bs) * bs

    def tail_start(self, length):
        return min(length, max(self.sink_tokens, self.near_start(length)))


@dataclass
class FarSkipSelection:
    # Absolute positions, equal-width rows per KV head; no padding or duplicates.
    positions: torch.Tensor
    near_start: int
    length: int


def _physical_positions(positions, *, length, physical_length, sink, tail_start):
    if tail_start is None:
        if physical_length != length:
            raise ValueError("retained KV must contain exactly the logical sequence")
        return positions
    prefix = min(sink, length)
    if physical_length != prefix + length - tail_start:
        raise ValueError("packed KV length does not match prefix and tail metadata")
    return torch.where(positions < prefix, positions, positions - tail_start + prefix)


def _keys_at(k, positions, *, length, cfg, tail_start):
    physical = _physical_positions(positions, length=length,
                                   physical_length=k.shape[1],
                                   sink=cfg.sink_tokens, tail_start=tail_start)
    # Positions are constructed from permitted intervals before indexing K.
    if positions.dim() == 1:
        return k[0].index_select(0, physical)
    heads = torch.arange(k.shape[2], device=k.device)[:, None]
    return k[0][physical, heads]


def _quantize(x):
    lo = x.amin(dim=-1, keepdim=True)
    scale = (x.amax(dim=-1, keepdim=True) - lo).clamp(min=1e-9) / 15
    return ((x - lo) / scale).round().clamp(0, 15) * scale + lo


def select_far_skip(q, k, length, cfg, coarse_dims, fine_dims, *,
                    scale=None, tail_start=None):
    """Select absolute IDs without building full-history summaries or masks.

    q=[1,1,H,D], k=[1,Sphysical,Hkv,D]. `tail_start` describes packed
    [0,sink) + [tail_start,length) storage; None means an uncompressed cache.
    Protected tokens participate in per-head normalization, but do not compete
    for the middle budget. Ties use ascending absolute token/page positions.
    """
    if (q.ndim != 4 or k.ndim != 4 or q.shape[:2] != (1, 1)
            or k.shape[0] != 1 or q.shape[-1] != k.shape[-1]
            or q.shape[2] % k.shape[2] or q.device != k.device
            or q.dtype != k.dtype):
        raise ValueError("far skip requires one decode query and compatible GQA KV")
    start = cfg.near_start(length)
    if tail_start is not None and tail_start != cfg.tail_start(length):
        raise ValueError("packed cache geometry differs from selector geometry")
    Hkv, G = k.shape[2], q.shape[2] // k.shape[2]
    bs, dev = cfg.block_size, k.device
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    swa_lo = max(0, length - cfg.window_tokens)
    prefix = min(cfg.sink_tokens, length)
    middle_lo = max(prefix, start)

    # Full near pages compete. Partial protection-boundary pages bypass L1;
    # their unprotected near tokens still compete in L2.
    first = (middle_lo + bs - 1) // bs
    stop = swa_lo // bs
    pages = torch.arange(first, max(first, stop), device=dev)
    n_pages = min(cfg.near_pages, pages.numel())
    q_coarse = q[0, 0].index_select(-1, coarse_dims).float() * scale
    if n_pages:
        positions = (pages[:, None] * bs + torch.arange(bs, device=dev)).flatten()
        keys = _keys_at(k, positions, length=length, cfg=cfg,
                        tail_start=tail_start).index_select(-1, coarse_dims)
        keys = keys.reshape(-1, bs, Hkv, coarse_dims.numel())
        if cfg.near_method == "avg":
            summary = keys.mean(1).float()
            scores = torch.einsum("hgd,phd->hgp", q_coarse.reshape(Hkv, G, -1), summary)
        else:
            qg = q_coarse.reshape(Hkv, G, -1)
            scores = (torch.einsum("hgd,phd->hgp", qg.clamp(min=0), keys.amax(1).float())
                      + torch.einsum("hgd,phd->hgp", qg.clamp(max=0), keys.amin(1).float()))
        scores = scores.to(q.dtype).mean(1)
        picked = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :n_pages]
        chosen = pages[picked].sort(-1).values
        candidates = (chosen[..., None] * bs + torch.arange(bs, device=dev)).reshape(Hkv, -1)
    else:
        candidates = torch.empty((Hkv, 0), dtype=torch.long, device=dev)

    edges = []
    if middle_lo < swa_lo:
        left_end = middle_lo
        if middle_lo % bs:
            left_end = min(swa_lo, (middle_lo // bs + 1) * bs)
            edges.append(torch.arange(middle_lo, left_end, device=dev))
        right_lo = max(left_end, (swa_lo // bs) * bs)
        if swa_lo % bs and right_lo < swa_lo:
            edges.append(torch.arange(right_lo, swa_lo, device=dev))
    if edges:
        candidates = torch.cat((candidates, torch.cat(edges).expand(Hkv, -1)), -1)
    candidates = candidates.sort(-1).values
    protected = torch.cat((torch.arange(prefix, device=dev),
                           torch.arange(max(prefix, swa_lo), length, device=dev)))
    remaining = cfg.token_budget - protected.numel()
    keep = min(remaining, candidates.shape[-1])
    if keep < candidates.shape[-1]:
        pool = torch.cat((candidates, protected.expand(Hkv, -1)), -1)
        keys = _keys_at(k, pool, length=length, cfg=cfg,
                        tail_start=tail_start).index_select(-1, fine_dims)
        if cfg.quantize:
            keys = _quantize(keys)
        qfine = (q[0, 0].index_select(-1, fine_dims).float() * scale).reshape(Hkv, G, -1)
        logits = torch.einsum("hgd,hnd->hgn", qfine, keys.float()).to(q.dtype)
        probabilities = logits.softmax(-1).mean(1)[:, :candidates.shape[-1]]
        rank = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :keep]
        candidates = candidates.gather(-1, rank)
    positions = torch.cat((protected.expand(Hkv, -1), candidates), -1).sort(-1).values
    return FarSkipSelection(positions, start, length)


def attend_far_skip(q, k, v, selection, cfg, *, scale=None, tail_start=None):
    """Original-width attention reads only selected protected/near KV."""
    if (v.shape[:3] != k.shape[:3] or v.device != k.device or v.dtype != k.dtype):
        raise ValueError("incompatible value cache")
    positions = selection.positions
    if positions.shape[-1] == 0:
        raise ValueError("far-off selection is empty")
    kk = _keys_at(k, positions, length=selection.length, cfg=cfg, tail_start=tail_start)
    vv = _keys_at(v, positions, length=selection.length, cfg=cfg, tail_start=tail_start)
    Hkv, G = k.shape[2], q.shape[2] // k.shape[2]
    qg = q[0, 0].reshape(Hkv, G, 1, q.shape[-1])
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    # Treat each KV group as a batch: eligible CUDA inputs can use fused SDPA.
    # Even a math-backend fallback sees only gathered KV, never far history.
    out = F.scaled_dot_product_attention(
        qg, kk[:, None], vv[:, None], dropout_p=0.0,
        is_causal=False, scale=scale, enable_gqa=True)
    return out.reshape(1, 1, q.shape[2], v.shape[-1])
