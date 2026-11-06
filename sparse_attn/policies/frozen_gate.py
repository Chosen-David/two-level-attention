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


def _stream_tail_profile(q, k, geometry, rows, chunk, scale):
    """逐 query chunk 完整归一化并立即归约，避免物化所有头的尾部概率。"""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != k.shape[2]:
        raise ValueError('one-shot unpadded B=1 prefill required')
    scale = q.shape[-1] ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('attention scale must be finite and positive')
    if chunk < 1:
        raise ValueError('profile chunk must be positive')

    length = q.shape[2]
    n = min(rows, length)
    heads, dim = q.shape[1], q.shape[-1]
    kv_heads = k.shape[1]
    group = heads // kv_heads
    # GQA K 只广播而不物化展开；每次仅保留当前 chunk 的完整因果概率。
    qs = q.detach()[:, :, length - n:].float().reshape(
        1, kv_heads, group, n, dim)
    kk = k.detach().float().unsqueeze(2).transpose(-1, -2)
    pos = torch.arange(length, device=q.device)[None, :]
    signals = {'near': [], 'far': []}

    for offset in range(0, n, chunk):
        end_offset = min(offset + chunk, n)
        start = length - n + offset
        end = length - n + end_offset
        scores = torch.matmul(
            qs[:, :, :, offset:end_offset], kk).reshape(
                1, heads, end_offset - offset, length) * scale
        qlen = torch.arange(start + 1, end + 1, device=q.device)[:, None]
        scores.masked_fill_(pos >= qlen, -torch.inf)
        probs = scores.softmax(-1)

        sink_hi = torch.minimum(
            qlen, torch.tensor(geometry.sink, device=q.device))
        near_hi = torch.maximum(sink_hi, qlen - geometry.window)
        far_hi = torch.minimum(near_hi, torch.maximum(
            sink_hi,
            ((qlen - geometry.window - geometry.near) // geometry.page)
            * geometry.page))
        masks = {'far': (pos >= sink_hi) & (pos < far_hi),
                 'near': (pos >= far_hi) & (pos < near_hi)}
        for name, mask in masks.items():
            signals[name].append((probs * mask).sum(-1).mean((0, 1)))

    # 所有 chunk 先在 GPU 拼接，只在请求边界做一次 D2H。
    values = torch.stack([
        torch.cat(signals[name]) for name in ('near', 'far')]).cpu()
    return {
        name: {'mean': float(values[index].mean()),
               'p95': float(torch.quantile(values[index], .95)),
               'max': float(values[index].max())}
        for index, name in enumerate(('near', 'far'))}


def profile_regions(q, k, geometry=RegionGeometry(), rows=256, chunk=16, scale=None):
    return _stream_tail_profile(q, k, geometry, rows, chunk, scale)


def decide_regions(profile, config, present_near=True, present_far=True):
    n, f = (profile[name][config.statistic] for name in ('near', 'far'))
    near = present_near and config.mode in ('near', 'dual') and n < config.epsilon
    far = present_far and config.mode in ('far', 'dual') and f < config.epsilon
    return dict(skip_near=bool(near), skip_far=bool(far))
