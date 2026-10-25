"""Stateful and operation-count regressions for compact eager decode."""
import pytest
import torch
from transformers.cache_utils import DynamicCache

from sparse_attn.ops.far_skip import FarSkipConfig, select_far_skip, attend_far_skip, _keys_at
from sparse_attn.ops.far_skip_cache import evict_far
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace, stable_topk_positions


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('n', [1, 17, 513])
def test_stable_topk_including_ties_negative_zero_infinities_nan(dtype, n):
    torch.manual_seed(871)
    scores = torch.randn(3, n, dtype=dtype)
    scores[0] = 0
    special = [0., -0., float('inf'), -float('inf'), float('nan'), -float('nan'), 1., 1.]
    for i, value in enumerate(special[:n]):
        scores[1, i] = value
    # Adjacent representable values must not be displaced by index tie-breaking.
    if n >= 3:
        one = torch.tensor(1., dtype=dtype)
        scores[2, :3] = torch.stack((one, torch.nextafter(one, one+1), torch.nextafter(one, one-1)))
    ws = FarSkipWorkspace()
    for k in set([0, 1, n//2, n]):
        expected = scores.argsort(dim=-1, descending=True, stable=True)[:, :k].sort(-1).values
        actual = stable_topk_positions(scores, k, ws)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('method', ['avg', 'minmax'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_incremental_page_summaries_across_ring_wraps_and_new_queries(method, dtype):
    torch.manual_seed(819)
    cfg = FarSkipConfig(4, 24, 2, 13, 3, 5, method)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 73, 12, dtype=dtype)
    cache.update(k, v, 0)
    ring = evict_far(cache, 0, cfg)
    ws = FarSkipWorkspace()
    dims = torch.tensor([11, 3, 8, 0])
    first = (max(cfg.sink_tokens, cfg.near_start(74)) + 3)//4
    summary_addresses = None
    for length in range(74, 331):
        nk, nv = torch.randn(2, 1, 2, 1, 12, dtype=dtype)
        k, v = torch.cat((k, nk), -2), torch.cat((v, nv), -2)
        rk, rv = cache.update(nk, nv, 0)
        q = torch.randn(1, 1, 6, 12, dtype=dtype)
        reference = select_far_skip(q, k.transpose(1, 2), length, cfg, dims, dims)
        previous = ws.pages_rebuilt
        selection = select_far_skip(q, rk.transpose(1, 2), length, cfg, dims, dims,
                                    layout=ring.layout, workspace=ws)
        assert torch.equal(selection.positions, reference.positions)
        expected = attend_far_skip(q, k.transpose(1, 2), v.transpose(1, 2), reference, cfg)
        actual = attend_far_skip(q, rk.transpose(1, 2), rv.transpose(1, 2), selection, cfg,
                                 layout=ring.layout, workspace=ws)
        torch.testing.assert_close(actual, expected)
        if length != 74:
            assert ws.pages_rebuilt - previous <= 1
        addresses = tuple(t.data_ptr() for t in ws.summaries)
        if summary_addresses is None:
            summary_addresses = addresses
        assert addresses == summary_addresses
    assert ws.pages_rebuilt == (330-cfg.window_tokens)//4 - first
    # Reusing the workspace with a restarted logical sequence must rebuild.
    k2 = torch.randn(1, 73, 2, 12, dtype=dtype)
    reference = select_far_skip(q, k2, 73, cfg, dims, dims)
    actual = select_far_skip(q, k2, 73, cfg, dims, dims, workspace=ws)
    assert torch.equal(reference.positions, actual.positions)


@torch.inference_mode()
def test_decode_update_writes_two_single_token_slices_without_old_kv_copy():
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 101, 12)
    cache.update(k, v, 0)
    layer = evict_far(cache, 0, FarSkipConfig(4, 24, 2, 13, 3, 5))
    nk, nv = torch.randn(2, 1, 2, 1, 12)
    pointer = layer.keys.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        cache.update(nk, nv, 0)
    events = profile.events()
    assert not any(e.name in ('aten::cat', 'aten::empty', 'aten::index_select') for e in events)
    copies = [e for e in events if e.name == 'aten::copy_']
    assert len(copies) == 2
    assert all(e.input_shapes[:2] == [[1, 2, 1, 12], [1, 2, 1, 12]] for e in copies)
    assert layer.keys.data_ptr() == pointer


@pytest.mark.parametrize('invalid', ['heads', 'value_width', 'dtype'])
def test_ring_rejects_broadcast_or_conversion_before_mutating_kv(invalid):
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 101, 12)
    cache.update(k, v, 0)
    layer = evict_far(cache, 0, FarSkipConfig(4, 24, 2, 13, 3, 5))
    before_k, before_v = layer.keys.clone(), layer.values.clone()
    nk, nv = torch.randn(2, 1, 2, 1, 12)
    if invalid == 'heads':
        nk = nk[:, :1]
    elif invalid == 'value_width':
        nv = nv[..., :1]
    else:
        nv = nv.double()
    with pytest.raises(ValueError, match='must match ring'):
        cache.update(nk, nv, 0)
    assert cache.get_seq_length() == 101
    torch.testing.assert_close(layer.keys, before_k)
    torch.testing.assert_close(layer.values, before_v)


def test_subspace_gather_has_no_full_width_intermediate():
    k = torch.randn(1, 51, 2, 128)
    cfg = FarSkipConfig(4, 16, 2, 12, 3, 4)
    ids = torch.tensor([[40, 43, 46], [41, 42, 47]])
    dims = torch.tensor([5, 12, 34, 97])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        selected = _keys_at(k, ids, length=51, cfg=cfg, dims=dims)
    for head in range(2):
        torch.testing.assert_close(selected[head], k[0, ids[head], head][:, dims])
    events = profile.events()
    assert sum(e.name == 'aten::index' for e in events) == 1
    assert not any(e.name == 'aten::index_select' for e in events)


@torch.inference_mode()
def test_poison_unused_ring_slots_and_workspace_is_bounded():
    cfg = FarSkipConfig(4, 16, 1, 12, 3, 4)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 1, 2, 8)
    cache.update(k, v, 0)
    layer = evict_far(cache, 0, cfg)
    layer.keys[..., 2:, :] = float('nan')
    layer.values[..., 2:, :] = float('nan')
    ws, dims = FarSkipWorkspace(), torch.arange(8)
    for length in range(3, 303):
        nk, nv = torch.randn(2, 1, 1, 1, 8)
        k, v = torch.cat((k, nk), -2), torch.cat((v, nv), -2)
        rk, rv = cache.update(nk, nv, 0)
        q = torch.randn(1, 1, 2, 8)
        selection = select_far_skip(q, rk.transpose(1, 2), length, cfg, dims, dims,
                                    layout=layer.layout, workspace=ws)
        expected_sel = select_far_skip(q, k.transpose(1, 2), length, cfg, dims, dims)
        assert torch.equal(selection.positions, expected_sel.positions)
        actual = attend_far_skip(q, rk.transpose(1, 2), rv.transpose(1, 2), selection, cfg,
                                 layout=layer.layout, workspace=ws)
        expected = attend_far_skip(q, k.transpose(1, 2), v.transpose(1, 2), expected_sel, cfg)
        torch.testing.assert_close(actual, expected)
    assert max(t.numel() for t in ws._ranges.values()) <= 32
    assert sum(t.numel() for t in ws._buffers.values()) < 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA hardware')
@torch.inference_mode()
def test_cuda_stable_keys_and_stateful_ring_summary():
    torch.manual_seed(29)
    ws = FarSkipWorkspace()
    scores = torch.randn(4, 513, device='cuda', dtype=torch.bfloat16)
    scores[:, ::17] = 0
    scores[:, :4] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.], device='cuda')
    for count in (1, 25, 300):
        expected = scores.argsort(dim=-1, descending=True, stable=True)[:, :count].sort(-1).values
        assert torch.equal(stable_topk_positions(scores, count, ws), expected)
    cfg = FarSkipConfig(64, 512, 2, 160, 4, 64)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 1027, 128, device='cuda', dtype=torch.bfloat16)
    cache.update(k, v, 0)
    ring = evict_far(cache, 0, cfg)
    dims = torch.arange(32, device='cuda')
    ws = FarSkipWorkspace()
    for length in range(1028, 1157):
        nk, nv = torch.randn(2, 1, 2, 1, 128, device='cuda', dtype=torch.bfloat16)
        k, v = torch.cat((k, nk), -2), torch.cat((v, nv), -2)
        rk, rv = cache.update(nk, nv, 0)
        q = torch.randn(1, 1, 4, 128, device='cuda', dtype=torch.bfloat16)
        reference = select_far_skip(q, k.transpose(1, 2), length, cfg, dims, dims)
        selection = select_far_skip(q, rk.transpose(1, 2), length, cfg, dims, dims,
                                    layout=ring.layout, workspace=ws)
        assert torch.equal(reference.positions, selection.positions)
        actual = attend_far_skip(q, rk.transpose(1, 2), rv.transpose(1, 2), selection, cfg,
                                 layout=ring.layout, workspace=ws)
        expected = attend_far_skip(q, k.transpose(1, 2), v.transpose(1, 2), reference, cfg)
        torch.testing.assert_close(actual, expected)
