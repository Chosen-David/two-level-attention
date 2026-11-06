import pytest
import torch

from sparse_attn.policies.frozen_gate import profile_regions
from sparse_attn.policies.region_geometry import RegionGeometry


def _legacy_region_profile(q, k, geometry, rows, chunk, scale):
    """修前 region 统计的独立实现对拍，确保删除 core 不改变 gate。"""
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
            signals[name].append((probs * mask).sum(-1).mean((0, 1)).cpu())
    return {
        name: {'mean': float(torch.cat(values).mean()),
               'p95': float(torch.quantile(torch.cat(values), .95)),
               'max': float(torch.cat(values).max())}
        for name, values in signals.items()}


def _assert_region_profile_matches_legacy(q, k, geometry, rows, chunk, scale):
    expected = _legacy_region_profile(q, k, geometry, rows, chunk, scale)
    actual = profile_regions(
        q, k, geometry, rows=rows, chunk=chunk, scale=scale)
    assert set(actual) == {'near', 'far'}
    for name in ('near', 'far'):
        for statistic in ('mean', 'p95', 'max'):
            assert actual[name][statistic] == pytest.approx(
                expected[name][statistic], abs=1e-6, rel=0)


@pytest.mark.parametrize('length', [1, 7, 64, 257])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_streaming_region_profile_matches_legacy(length, kv_heads):
    torch.manual_seed(20261102 + length + kv_heads)
    geometry = RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3)
    q = torch.randn(1, 4, length, 8)
    k = torch.randn(1, kv_heads, length, 8)
    _assert_region_profile_matches_legacy(q, k, geometry, 8, 3, .7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('length', [64, 257])
def test_streaming_region_profile_matches_legacy_on_cuda(length):
    torch.manual_seed(20261103 + length)
    geometry = RegionGeometry(
        sink=4, near=16, window=8, page=4, max_new_tokens=3)
    q = torch.randn(1, 4, length, 32, device='cuda')
    k = torch.randn(1, 2, length, 32, device='cuda')
    _assert_region_profile_matches_legacy(q, k, geometry, 17, 5, .7)
