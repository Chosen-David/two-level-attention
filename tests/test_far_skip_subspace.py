"""Projection reuse preserves the original selected dimensions and QDQ."""
from dataclasses import replace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from sparse_attn.ops.far_skip import FarSkipConfig, _quantize, select_far_skip
from sparse_attn.ops.far_skip_cache import evict_far
from sparse_attn.ops.far_skip_subspace import FarSkipFineCache, fine_cache_for
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize('quant', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cached_projection_equals_full_history_reference_and_is_bounded(quant, dtype):
    torch.manual_seed(93)
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4, quantize=quant)
    dims = torch.tensor([10, 1, 7, 0])  # retain arbitrary dimension order
    full = torch.randn(1, 2, 39, 12, dtype=dtype)
    cache = DynamicCache(); cache.update(full, full.clone(), 0)
    ring = evict_far(cache, 0, cfg)
    fine = FarSkipFineCache(cfg, ring.layout, dims)
    fine.initialize(ring.keys.transpose(1, 2), 39)
    ptr = fine.keys.data_ptr()
    for length in range(40, 123):
        new = torch.randn(1, 2, 1, 12, dtype=dtype)
        full = torch.cat([full, new], -2)
        ring.update(new, new.clone())
        fine.append(ring.keys.transpose(1, 2), length)
        ids = torch.cat([torch.arange(3), torch.arange(cfg.tail_start(length), length)]).expand(2, -1)
        expected = full[0].index_select(1, ids[0]).index_select(-1, dims)
        if quant:
            expected = _quantize(expected)
        torch.testing.assert_close(fine.gather(ids), expected, atol=0, rtol=0)
        assert fine.keys.data_ptr() == ptr
        assert fine.keys.untyped_storage().nbytes() == 2*ring.layout.capacity*len(dims)*full.element_size()


@pytest.mark.parametrize('quant', [False, True])
@torch.inference_mode()
def test_fine_cache_initialization_reads_only_live_keys_and_gap_rebuilds(quant):
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4, quantize=quant)
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 5, 12), torch.randn(1, 2, 5, 12), 0)
    ring = evict_far(cache, 0, cfg)
    ring.keys[:, :, 5:] = float('nan')
    ws, dims = FarSkipWorkspace(), torch.tensor([11, 0, 9, 1])
    fine = fine_cache_for(ws, ring.keys.transpose(1, 2), 5, cfg, dims, ring.layout)
    assert torch.isfinite(fine.keys).all()
    for _ in range(8):
        ring.update(torch.randn(1, 2, 1, 12), torch.randn(1, 2, 1, 12))
    fine = fine_cache_for(ws, ring.keys.transpose(1, 2), 13, cfg, dims, ring.layout)
    raw = ring.keys[0, :, :13].index_select(-1, dims)
    torch.testing.assert_close(fine.keys[:, :13], _quantize(raw) if quant else raw, atol=0, rtol=0)
    # A new dimension tensor/configuration must not reuse the previous projection.
    other = fine_cache_for(ws, ring.keys.transpose(1, 2), 13, cfg, dims.flip(0), ring.layout)
    assert other is not fine


@torch.inference_mode()
def test_quantization_reduces_one_new_token_not_repeated_candidate_pool():
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4, quantize=True)
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 49, 12), torch.randn(1, 2, 49, 12), 0)
    ring = evict_far(cache, 0, cfg)
    dims = torch.arange(8)
    fine = FarSkipFineCache(cfg, ring.layout, dims)
    fine.initialize(ring.keys.transpose(1, 2), 49)
    ring.update(torch.randn(1, 2, 1, 12), torch.randn(1, 2, 1, 12))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
        fine.append(ring.keys.transpose(1, 2), 50)
        fine.gather(torch.arange(35, 50).expand(2, -1))
    shapes = [e.input_shapes[0] for e in p.events() if e.name in ['aten::amin', 'aten::amax']]
    assert shapes == [[2, 8], [2, 8]]


@torch.inference_mode()
def test_full_width_and_retain_do_not_allocate_fine_side_cache():
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4)
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 49, 12), torch.randn(1, 2, 49, 12), 0)
    q, full, small = torch.randn(1, 1, 6, 12), torch.arange(12), torch.arange(8)
    for dims in (full, small):
        ws = FarSkipWorkspace()
        select_far_skip(q, cache.layers[0].keys.transpose(1, 2), 49, cfg, dims, dims, workspace=ws)
        assert not hasattr(ws, '_fine_cache')
    ring = evict_far(cache, 0, cfg)
    ws = FarSkipWorkspace()
    select_far_skip(q, ring.keys.transpose(1, 2), 49, cfg, full, full, layout=ring.layout, workspace=ws)
    assert not hasattr(ws, '_fine_cache')
    fixed = FarSkipGraphWorkspace(cfg, ring.layout, 2, full, full)
    fixed.initialize(ring.keys.transpose(1, 2), 49)
    assert fixed.fine_cache is None
