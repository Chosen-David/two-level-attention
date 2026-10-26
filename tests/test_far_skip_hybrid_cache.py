"""Real HF caches and tiny random Qwen3: layout/dispatch, not model quality.

CPU tests require torch 2.8 and transformers 4.56.2. The CUDA test captures the
whole model when a GPU is available; it must not be reported as passed on CPU.
"""
import argparse
import copy
import gc
from types import MethodType
from unittest.mock import patch
import weakref

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp.bench.e152v4_graph_latency import GraphRunner
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.cuda_graph_state import disable_graph, enable_graph, graph_state
from sparse_attn.ops.far_skip_cache import FarSkipLayer
from sparse_attn.ops.far_skip import FarSkipConfig
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.tli_static_cache import TLIStaticCache, TLIStaticLayer, fill_static_from_dynamic
from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for name in ('TLI_DECODE_DENSE', 'TLI_SPARSE_PREFILL'):
        monkeypatch.setenv(name, '0')
    disable_graph()
    with torch.no_grad():
        yield
    disable_graph()
    torch.set_num_threads(threads)


def make_model(cache_mode='evict', device='cpu', budget=10):
    torch.manual_seed(261026)
    cfg = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=96,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=128, max_position_embeddings=512)
    cfg._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(cfg).eval().to(device)
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    args = parser.parse_args([
        '--method', 'tli', '--tia_block_size', '4', '--tli_near_len', '8',
        '--tli_near_pages', '1', '--tli_sink_tokens', '2', '--tli_swa_size', '4',
        '--tia_level1_topk', '4', '--tia_level2_topk', str(budget),
        '--tli_subspace', 'tail', '--tia_level2_cmp_ratio', '4',
        '--tli_far_skip_backend', 'compact', '--tli_far_skip_cache', cache_mode])
    attn = model.model.layers[0].self_attn
    attn.indexer = TLIIndexer(args)
    attn.indexer.layer_idx = 0
    attn.indexer._skip_ids = {0}
    attn.forward = MethodType(qwen3_attn_forward, attn)
    return model


def prefill(model, length=23, offset=0):
    ids = (torch.arange(length, device=next(model.parameters()).device) + offset) % 32
    return model(input_ids=ids[None], use_cache=True).past_key_values


def context(length):
    return BatchCtx(batch=1, seq_lens=[length], gen_lens=[0], alive=[True], pad_id=0)


def make_runner(model, capacity=128):
    device = next(model.parameters()).device
    cache = TLIStaticCache(model.config, capacity, 1)
    return GraphRunner(model, cache, torch.finfo(torch.float32).min,
                       capacity, 0, device)


def test_direct_allocation_never_creates_full_buffer_for_skip_layer():
    model = make_model()
    dyn = prefill(model)
    source = dyn.layers[0].keys
    old_ref = weakref.ref(source)
    cache = TLIStaticCache(model.config, 128, 1)
    skipped = cache.layers[0]
    with patch.object(skipped, 'lazy_initialization', side_effect=AssertionError('full allocation')):
        fill_static_from_dynamic(cache, dyn, [23], model=model)
    ring, linear = cache.layers
    assert isinstance(ring, FarSkipStaticLayer)
    assert type(linear) is TLIStaticLayer
    assert ring.keys.shape[-2] == 17  # sink2 + near8 + swa4 + page guard3
    assert linear.keys.shape[-2] == 128
    assert ring.keys.untyped_storage().nbytes() == ring.keys.numel() * ring.keys.element_size()
    assert ring.keys.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(linear.keys[..., :23, :], dyn.layers[1].keys, rtol=0, atol=0)
    assert linear.get_seq_length() == 23
    del source, dyn
    gc.collect()
    assert old_ref() is None


def test_refill_gate_changes_restore_linear_and_rebuild_ring():
    model = make_model()
    cache = TLIStaticCache(model.config, 128, 1)
    idx = model.model.layers[0].self_attn.indexer
    for skip, length in [(True, 33), (True, 27), (False, 41), (True, 25)]:
        idx._skip_ids = {0} if skip else set()
        dyn = prefill(model, length, offset=length)
        fill_static_from_dynamic(cache, dyn, [length], model=model)
        layer = cache.layers[0]
        assert isinstance(layer, FarSkipStaticLayer) == skip
        if not skip:
            torch.testing.assert_close(layer.keys[..., :length, :], dyn.layers[0].keys, rtol=0, atol=0)
        else:
            live = torch.cat((torch.arange(2), torch.arange(layer.config.tail_start(length), length)))
            torch.testing.assert_close(layer.keys.index_select(2, layer.layout.map(live)),
                                       dyn.layers[0].keys.index_select(2, live), rtol=0, atol=0)
    idx.far_skip_backend = 'legacy'
    dyn = prefill(model, 35)
    fill_static_from_dynamic(cache, dyn, [35], model=model)
    assert type(cache.layers[0]) is TLIStaticLayer
    assert not cache._far_skip_prepared
    torch.testing.assert_close(cache.layers[0].keys[..., :35, :], dyn.layers[0].keys, rtol=0, atol=0)


def test_refill_preflight_and_model_requirement():
    model = make_model()
    cache = TLIStaticCache(model.config, 128, 1)
    dyn = prefill(model)
    fill_static_from_dynamic(cache, dyn, [23], model=model)
    original = cache.layers[0]
    with pytest.raises(ValueError, match='requires model'):
        fill_static_from_dynamic(cache, dyn, [23])
    for lengths in ([129], [], [23, 23]):
        with pytest.raises(ValueError):
            fill_static_from_dynamic(cache, dyn, lengths, model=model)
        assert cache.layers[0] is original


@pytest.mark.parametrize('budget', [10, 32])
def test_eager_eviction_matches_retained_history_and_writes_once(budget):
    model = make_model(budget=budget)
    reference = copy.deepcopy(model)
    reference.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
    dyn, full = prefill(model), prefill(reference)
    with patch('sparse_attn.patches.qwen3_attn_patch.eager_decoding_attn',
               side_effect=AssertionError('ring reached legacy reader')):
        for pos in range(23, 65):
            inputs = dict(input_ids=torch.tensor([[pos % 32]]),
                          position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                          use_cache=True)
            expected = reference(**inputs, past_key_values=full).logits
            actual = model(**inputs, past_key_values=dyn).logits
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            layer = dyn.layers[0]
            assert isinstance(layer, FarSkipLayer)
            assert layer.get_seq_length() == pos + 1
            assert layer.keys.shape[-2] == 17
            live = torch.cat((torch.arange(2), torch.arange(layer.config.tail_start(pos + 1), pos + 1)))
            for name in ('keys', 'values'):
                torch.testing.assert_close(getattr(layer, name).index_select(2, layer.layout.map(live)),
                                           getattr(full.layers[0], name).index_select(2, live), rtol=0, atol=0)


@pytest.mark.parametrize('budget', [10, 32])
@pytest.mark.parametrize('length', [1, 7, 23])
def test_static_model_matches_retained_history_across_wraps(budget, length):
    model = make_model(budget=budget)
    reference = copy.deepcopy(model)
    reference.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
    dyn, full = prefill(model, length), prefill(reference, length)
    runner = make_runner(model)
    runner.fill(dyn, length)
    ctx = context(length)
    pointers = [t.data_ptr() for l in runner.cache.layers for t in (l.keys, l.values)]
    with patch('sparse_attn.patches.qwen3_attn_patch.eager_decoding_attn',
               side_effect=AssertionError('ring reached legacy reader')):
        for pos in range(length, length + 42):
            token = pos % 32
            expected = reference(input_ids=torch.tensor([[token]]),
                                 position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                                 past_key_values=full, use_cache=True).logits
            ctx.seq_lens[0] = pos + 1
            with batch_decode_scope(ctx):
                runner.inject(token, pos + 1)
                actual = runner.forward().logits
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert pointers == [t.data_ptr() for l in runner.cache.layers for t in (l.keys, l.values)]


def test_runner_discards_old_graph_and_registers_mask_on_every_refill():
    model = make_model()
    runner = make_runner(model)
    for length in (23, 31, 19):
        runner.graph, runner.out = object(), object()
        runner.fill(prefill(model, length, offset=length), length)
        assert runner.graph is None and runner.out is None
        assert runner.cache._far_skip_attention_mask is runner.m4_buf


def test_production_geometry_includes_page_boundary_guard():
    cfg = FarSkipConfig(64, 4096, 32, 1024, 4, 128)
    capacity = 4 + 4096 + 128 + 63
    live_counts = [min(4, length) + length - cfg.tail_start(length)
                   for length in range(1, 20001)]
    assert max(live_counts) == capacity
    assert 4 + 32831 - cfg.tail_start(32831) == capacity


def test_compact_rejects_unimplemented_model_consumer_before_patching():
    from transformers import LlamaConfig, LlamaForCausalLM
    from sparse_attn.patches import register_patch
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    model = LlamaForCausalLM(cfg).eval()
    args = make_model().model.layers[0].self_attn.indexer.args
    with pytest.raises(ValueError, match='Qwen3 consumer'):
        register_patch(model, args)
    assert not hasattr(model.model.layers[0].self_attn, 'indexer')


def test_unregistered_mask_and_changed_gate_reject_before_write():
    model = make_model()
    runner = make_runner(model)
    runner.fill(prefill(model), 23)
    layer = runner.cache.layers[0]
    before = layer.keys.clone()
    runner.inject(3, 24)
    with batch_decode_scope(context(24)):
        runner.cache._far_skip_attention_mask = None
        with pytest.raises(ValueError, match='registered'):
            runner.forward()
        torch.testing.assert_close(layer.keys, before, equal_nan=True, rtol=0, atol=0)
        runner.cache._far_skip_attention_mask = runner.m4_buf
        model.model.layers[0].self_attn.indexer.skip_far = False
        with pytest.raises(ValueError, match='gate/geometry'):
            runner.forward()
        torch.testing.assert_close(layer.keys, before, equal_nan=True, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA capture/replay')
def test_cuda_whole_model_replay_and_second_request():
    model = make_model(device='cuda')
    reference = copy.deepcopy(model)
    reference.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
    runner = make_runner(model)
    for length, offset in ((23, 0), (29, 7)):
        # Prefill stays eager; capture/replay is decode-only.
        disable_graph()
        dyn, full = prefill(model, length, offset), prefill(reference, length, offset)
        runner.fill(dyn, length)
        del dyn
        assert runner.graph is None
        ctx = context(length + 1)
        enable_graph(1)
        graph_state().pos_buf = runner.pos_buf
        graph_state().valid_len_buf = runner.vl_buf
        with batch_decode_scope(ctx):
            runner.warmup_and_capture(length % 32, length + 1)
        for pos in range(length, length + 37):
            inputs = dict(input_ids=torch.tensor([[pos % 32]], device='cuda'),
                          position_ids=torch.tensor([[pos]], device='cuda'),
                          cache_position=torch.tensor([pos], device='cuda'), use_cache=True)
            expected = reference(**inputs, past_key_values=full).logits
            runner.inject(pos % 32, pos + 1)
            runner.graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(runner.out.logits, expected, rtol=2e-4, atol=2e-5)
