"""Native post-RoPE profiling, independent near/far signals, no decode updates."""
from dataclasses import dataclass
import math
import torch
from sparse_attn.policies.region_geometry import RegionGeometry


@dataclass(frozen=True)
class GateConfig:
    epsilon: float = .05
    rows: int = 256
    statistic: str = 'mean'
    mode: str = 'dual'

    def __post_init__(self):
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('epsilon must be in [0,1]')
        if self.rows < 1 or self.statistic not in ('mean', 'p95') or self.mode not in ('full', 'far', 'near', 'dual'):
            raise ValueError('invalid gate configuration')


def _tail_probabilities(q, k, rows, scale):
    """最后一组 query 的完整因果行概率，GQA K 只广播而不物化展开。"""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != k.shape[2]:
        raise ValueError('one-shot unpadded B=1 prefill required')
    scale = q.shape[-1] ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be finite and positive")
    length = q.shape[2]
    n = min(rows, length)
    heads, dim = q.shape[1], q.shape[-1]
    kv_heads = k.shape[1]
    group = heads // kv_heads
    qs = q.detach()[:, :, length - n:].float().reshape(
        1, kv_heads, group, n, dim)
    kk = k.detach().float().unsqueeze(2).transpose(-1, -2)
    scores = torch.matmul(qs, kk).reshape(1, heads, n, length) * scale
    pos = torch.arange(length, device=q.device)[None, None, :]
    qlen = torch.arange(length - n + 1, length + 1, device=q.device)[None, :, None]
    scores.masked_fill_(pos >= qlen, -torch.inf)
    return scores.softmax(-1)


def _region_stats_from_probabilities(probs, geometry, chunk):
    """复用完整行 softmax，分块仅限制 region reduction 的临时张量。"""
    length = probs.shape[-1]
    n = probs.shape[-2]
    query_start = length - n
    pos = torch.arange(length, device=probs.device)[None, :]
    signals = {'near': [], 'far': []}
    for offset in range(0, n, chunk):
        end_offset = min(offset + chunk, n)
        start = query_start + offset
        end = query_start + end_offset
        qlen = torch.arange(start + 1, end + 1, device=probs.device)[:, None]
        sink_hi = torch.minimum(qlen, torch.tensor(geometry.sink, device=probs.device))
        near_hi = torch.maximum(sink_hi, qlen - geometry.window)
        far_hi = torch.minimum(near_hi, torch.maximum(
            sink_hi, ((qlen - geometry.window - geometry.near) // geometry.page) * geometry.page))
        masks = {'far': (pos >= sink_hi) & (pos < far_hi),
                 'near': (pos >= far_hi) & (pos < near_hi)}
        chunk_probs = probs[:, :, offset:end_offset]
        for name, mask in masks.items():
            signals[name].append((chunk_probs * mask).sum(-1).mean((0, 1)))
    # 所有 chunk 先在 GPU 拼接，只在请求边界做一次 D2H。
    values = torch.stack([torch.cat(signals[name]) for name in ('near', 'far')]).cpu()
    return {name: {'mean': float(values[index].mean()),
                   'p95': float(torch.quantile(values[index], .95)),
                   'max': float(values[index].max())}
            for index, name in enumerate(('near', 'far'))}


def _core_tokens_from_probabilities(probs, geometry, topk):
    """先完整归一化每个查询行，再沿头和查询位置聚合，禁止局部 Top-K 拼接。"""
    length = probs.shape[-1]
    token_scores = probs.mean(dim=(0, 1)).mean(dim=0)
    sink_hi, far_hi, near_hi = geometry.bounds(length)
    selected = {}
    for name, lo, hi in (('far', sink_hi, far_hi), ('near', far_hi, near_hi)):
        # Region bounds are host integers and each region is contiguous.
        # Slice the same scores without a full-length boolean compaction.
        count = min(topk, hi - lo)
        selected[name] = torch.topk(token_scores[lo:hi], count).indices + lo
    counts = {name: int(value.numel()) for name, value in selected.items()}
    values = torch.cat([selected['far'], selected['near']]).cpu().tolist()
    return {'far': values[:counts['far']], 'near': values[counts['far']:]}


def profile_regions(q, k, geometry=RegionGeometry(), rows=256, chunk=16, scale=None):
    probs = _tail_probabilities(q, k, rows, scale)
    return _region_stats_from_probabilities(probs, geometry, chunk)


def profile_core_tokens(q, k, geometry=RegionGeometry(), rows=256, topk=64, scale=None):
    """Return the highest-attention prefill tokens per region.

    Uses the last ``rows`` prefill queries, averages probabilities across query
    positions and heads, then returns ``topk`` token positions from the far and
    near regions.  These IDs seed the first decode step when a region is frozen
    out, avoiding a first-token distribution cliff.
    """
    probs = _tail_probabilities(q, k, rows, scale)
    return _core_tokens_from_probabilities(probs, geometry, topk)


def profile_regions_and_core_tokens(q, k, geometry=RegionGeometry(), rows=256,
                                    chunk=16, topk=64, scale=None):
    """方案 B：一次尾部 QK/softmax 同时产出区域质量和 core-token IDs。"""
    probs = _tail_probabilities(q, k, rows, scale)
    profile = _region_stats_from_probabilities(probs, geometry, chunk)
    profile['core'] = _core_tokens_from_probabilities(probs, geometry, topk)
    return profile


def decide_regions(profile, config, present_near=True, present_far=True):
    n, f = (profile[name][config.statistic] for name in ('near', 'far'))
    near = present_near and config.mode in ('near', 'dual') and n < config.epsilon
    far = present_far and config.mode in ('far', 'dual') and f < config.epsilon
    return dict(skip_near=bool(near), skip_far=bool(far))
