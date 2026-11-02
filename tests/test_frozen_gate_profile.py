import pytest
import torch

from sparse_attn.policies.frozen_gate import (
    profile_core_tokens, profile_regions_and_core_tokens)
from sparse_attn.policies.region_geometry import RegionGeometry


def _legacy_profile(q, k, geometry, rows, chunk, topk, scale):
    """修前实现的独立 region/core 统计，作为语义对拍参考。"""
    length = q.shape[2]
    n = min(rows, length)
    heads = q.shape[1]
    kk = k.detach().float().repeat_interleave(heads // k.shape[1], 1)
    pos = torch.arange(length, device=q.device)[None, :]
    signals = {'near': [], 'far': []}
    for start in range(length - n, length, chunk):
        end = min(start + chunk, length)
        qe = q.detach()[:, :, start:end].float()
        scores = qe @ kk.transpose(-1, -2) * scale
        qlen = torch.arange(start + 1, end + 1, device=q.device)[:, None]
        scores.masked_fill_(pos >= qlen, -torch.inf)
        probs = scores.softmax(-1)
        sink_hi = torch.minimum(qlen, torch.tensor(geometry.sink, device=q.device))
        near_hi = torch.maximum(sink_hi, qlen - geometry.window)
        far_hi = torch.minimum(near_hi, torch.maximum(
            sink_hi, ((qlen - geometry.window - geometry.near) // geometry.page) * geometry.page))
        masks = {'far': (pos >= sink_hi) & (pos < far_hi),
                 'near': (pos >= far_hi) & (pos < near_hi)}
        for name, mask in masks.items():
            signals[name].append((probs * mask).sum(-1).mean((0, 1)).cpu())
    profile = {name: {'mean': float(torch.cat(values).mean()),
                      'p95': float(torch.quantile(torch.cat(values), .95)),
                      'max': float(torch.cat(values).max())}
               for name, values in signals.items()}

    qs = q.detach()[:, :, length - n:].float()
    scores = qs @ kk.transpose(-1, -2) * scale
    pos = torch.arange(length, device=q.device)[None, None, :]
    qlen = torch.arange(length - n + 1, length + 1, device=q.device)[None, :, None]
    scores.masked_fill_(pos >= qlen, -torch.inf)
    token_scores = scores.softmax(-1).mean(dim=(0, 1)).mean(dim=0)
    sink_hi, far_hi, near_hi = geometry.bounds(length)
    ids = torch.arange(length, device=q.device)
    core = {}
    for name, lo, hi in (('far', sink_hi, far_hi), ('near', far_hi, near_hi)):
        positions = ids[(ids >= lo) & (ids < hi)]
        count = min(topk, positions.numel())
        core[name] = positions[torch.topk(token_scores[positions], count).indices].tolist()
    profile['core'] = core
    return profile


@pytest.mark.parametrize('length', [1, 7, 64, 257])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_shared_tail_profile_matches_legacy_region_and_core(length, kv_heads):
    torch.manual_seed(20261102 + length + kv_heads)
    geometry = RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3)
    q = torch.randn(1, 4, length, 8)
    k = torch.randn(1, kv_heads, length, 8)
    rows, chunk, topk, scale = 8, 3, 3, .7
    expected = _legacy_profile(q, k, geometry, rows, chunk, topk, scale)
    actual = profile_regions_and_core_tokens(
        q, k, geometry, rows=rows, chunk=chunk, topk=topk, scale=scale)
    for name in ('near', 'far'):
        for statistic in ('mean', 'p95', 'max'):
            assert actual[name][statistic] == pytest.approx(
                expected[name][statistic], abs=1e-6, rel=0)
    assert actual['core'] == expected['core']


@pytest.mark.parametrize('length', [1, 7, 64, 257])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_static_gate_core_only_matches_legacy(length, kv_heads):
    torch.manual_seed(20261202 + length + kv_heads)
    geometry = RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3)
    q = torch.randn(1, 4, length, 8)
    k = torch.randn(1, kv_heads, length, 8)
    rows, chunk, topk, scale = 8, 3, 3, .7
    expected = _legacy_profile(q, k, geometry, rows, chunk, topk, scale)['core']
    actual = profile_core_tokens(
        q, k, geometry, rows=rows, topk=topk, scale=scale)
    assert actual == expected
