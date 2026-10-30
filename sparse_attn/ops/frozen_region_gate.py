"""Native post-RoPE profiling, independent near/far signals, no decode updates."""
from dataclasses import dataclass
import math
import torch
from .frozen_region_cache import RegionGeometry


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


def profile_regions(q, k, geometry=RegionGeometry(), rows=256, chunk=16, scale=None):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != k.shape[2]:
        raise ValueError('one-shot unpadded B=1 prefill required')
    scale = q.shape[-1] ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be finite and positive")
    length = q.shape[2]
    n = min(rows, length)
    heads, dim = q.shape[1], q.shape[-1]
    kk = k.detach().float().repeat_interleave(heads//k.shape[1], 1)
    pos = torch.arange(length, device=q.device)[None, :]
    signals = {'near': [], 'far': []}
    for start in range(length-n, length, chunk):
        end = min(start+chunk, length)
        qe = q.detach()[:, :, start:end].float()
        scores = qe @ kk.transpose(-1, -2) * scale
        qlen = torch.arange(start+1, end+1, device=q.device)[:, None]
        scores.masked_fill_(pos >= qlen, -torch.inf)
        probs = scores.softmax(-1)
        sink_hi = torch.minimum(qlen, torch.tensor(geometry.sink, device=q.device))
        near_hi = torch.maximum(sink_hi, qlen-geometry.window)
        far_hi = torch.minimum(near_hi, torch.maximum(sink_hi, ((qlen-geometry.window-geometry.near)//geometry.page)*geometry.page))
        masks = {'far': (pos >= sink_hi) & (pos < far_hi),
                 'near': (pos >= far_hi) & (pos < near_hi)}
        for name, mask in masks.items():
            signals[name].append((probs*mask).sum(-1).mean((0, 1)).cpu())
    return {name: {'mean': float(torch.cat(values).mean()),
                   'p95': float(torch.quantile(torch.cat(values), .95)),
                   'max': float(torch.cat(values).max())} for name, values in signals.items()}


def decide_regions(profile, config, present_near=True, present_far=True):
    n, f = (profile[name][config.statistic] for name in ('near', 'far'))
    near = present_near and config.mode in ('near', 'dual') and n < config.epsilon
    far = present_far and config.mode in ('far', 'dual') and f < config.epsilon
    return dict(skip_near=bool(near), skip_far=bool(far))
