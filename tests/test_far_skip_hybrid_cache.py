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
from sparse_attn.ops.tli_static_cache import (
    TLIStaticCache, TLIStaticLayer, fill_static_from_dynamic,
    finalize_direct_prefill, prepare_direct_prefill,
)
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
        '--tli_near_pages', '1', '--tli_near_tokens', 'off',
        '--tli_near_len', '8', '--tli_sink_tokens', '2', '--tli_swa_size', '4',
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


@pytest.mark.parametrize('length', [1, 7, 23, 33])
def test_direct_prefill_dense_semantics_layout_and_skip_finalize(length):
    model = make_model()
    reference = copy.deepcopy(model)
    ids = torch.arange(length)[None] % 32
    expected = reference(input_ids=ids, use_cache=True)
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    assert not cache.is_compileable
    assert cache.get_mask_sizes(torch.arange(length), 0) == (length, 0)
    actual = model(input_ids=ids, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)

    state = staging.states[0]
    assert state.mode == 'RING_FINAL'
    assert state.linear.keys is None and state.linear.values is None
    source = expected.past_key_values.layers[0]
    cfg = state.config
    start = cfg.tail_start(length)
    live = torch.cat((torch.arange(min(cfg.sink_tokens, length)),
                      torch.arange(start, length)))
    for name in ('keys', 'values'):
        torch.testing.assert_close(
            getattr(state.ring, name).index_select(2, state.ring.layout.map(live)),
            getattr(source, name).index_select(2, live), rtol=0, atol=0)
    torch.testing.assert_close(staging.states[1].linear.keys[..., :length, :],
                               expected.past_key_values.layers[1].keys, rtol=0, atol=0)
    abandoned = weakref.ref(state.linear)
    ring_ptrs = (state.ring.keys.data_ptr(), state.ring.values.data_ptr())
    linear_ptrs = (staging.states[1].linear.keys.data_ptr(),
                   staging.states[1].linear.values.data_ptr())

    finalize_direct_prefill(cache, model, length)
    assert isinstance(cache.layers[0], FarSkipStaticLayer)
    assert type(cache.layers[1]) is TLIStaticLayer
    assert cache._direct_prefill is None and cache._far_skip_prepared
    assert cache.is_compileable
    assert cache.get_mask_sizes(torch.tensor([length]), 0) == (128, 0)
    assert cache.layers[0].cumulative_length == length
    assert cache.layers[1].cumulative_length == length
    assert state.linear is None and state.ring is None
    assert ring_ptrs == (cache.layers[0].keys.data_ptr(), cache.layers[0].values.data_ptr())
    assert linear_ptrs == (cache.layers[1].keys.data_ptr(), cache.layers[1].values.data_ptr())
    gc.collect()
    assert abandoned() is None
    for name in ('keys', 'values'):
        torch.testing.assert_close(
            getattr(cache.layers[0], name).index_select(2, cache.layers[0].layout.map(live)),
            getattr(source, name).index_select(2, live), rtol=0, atol=0)
    with pytest.raises(ValueError, match='matching completed direct prefill'):
        finalize_direct_prefill(cache, model, length)


def test_direct_prefill_hf5_missing_cache_position_preserves_storage_plans():
    model = make_model()
    length = 7
    source = prefill(model, length)
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    for li, source_layer in enumerate(source.layers):
        keys, values = source_layer.keys, source_layer.values
        returned = staging.update(li, keys, values, None)
        assert returned[0] is keys and returned[1] is values
        state = staging.states[li]
        assert state.linear.cumulative_length == length
    assert staging.states[0].mode == 'RING_FINAL'
    assert staging.states[0].linear.keys is None
    assert staging.states[1].linear.is_initialized
    torch.testing.assert_close(staging.states[1].linear.keys[..., :length, :],
                               source.layers[1].keys, rtol=0, atol=0)


def test_direct_prefill_dynamic_non_skip_keeps_full_linear_without_ring():
    model = make_model()
    idx = model.model.layers[0].self_attn.indexer
    idx.enable_layer_skip = False
    idx.dyn_dgate = True
    idx._skip_ids = None
    idx._dyn_skip_pending = False
    reference = copy.deepcopy(model)
    length = 65
    ids = torch.arange(length)[None] % 32
    expected = reference(input_ids=ids, use_cache=True)
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    assert not cache.is_compileable
    assert cache.get_mask_sizes(torch.arange(length), 0) == (length, 0)
    actual = model(input_ids=ids, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
    assert staging.states[0].mode == 'LINEAR_PROVISIONAL'
    assert staging.states[0].ring is None
    ptrs = (staging.states[0].linear.keys.data_ptr(),
            staging.states[0].linear.values.data_ptr())
    idx._dyn_skip_pending = False
    finalize_direct_prefill(cache, model, length)
    assert type(cache.layers[0]) is TLIStaticLayer
    assert ptrs == (cache.layers[0].keys.data_ptr(), cache.layers[0].values.data_ptr())
    for li in range(2):
        for name in ('keys', 'values'):
            torch.testing.assert_close(
                getattr(cache.layers[li], name)[..., :length, :],
                getattr(expected.past_key_values.layers[li], name), rtol=0, atol=0)


def test_direct_prefill_dynamic_skip_compacts_once_and_releases_provisional_linear():
    model = make_model()
    idx = model.model.layers[0].self_attn.indexer
    idx.enable_layer_skip = False
    idx.dyn_dgate = True
    idx._skip_ids = None
    reference = copy.deepcopy(model)
    length = 65
    ids = torch.arange(length)[None] % 32
    expected = reference(input_ids=ids, use_cache=True)
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    actual = model(input_ids=ids, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
    state = staging.states[0]
    assert state.mode == 'LINEAR_PROVISIONAL' and state.ring is None
    abandoned = weakref.ref(state.linear.keys)
    idx._dyn_skip_pending = True
    finalize_direct_prefill(cache, model, length)
    layer = cache.layers[0]
    assert isinstance(layer, FarSkipStaticLayer)
    live = torch.cat((torch.arange(layer.config.sink_tokens),
                      torch.arange(layer.config.tail_start(length), length)))
    for name in ('keys', 'values'):
        torch.testing.assert_close(
            getattr(layer, name).index_select(2, layer.layout.map(live)),
            getattr(expected.past_key_values.layers[0], name).index_select(2, live),
            rtol=0, atol=0)
    gc.collect()
    assert abandoned() is None


@pytest.mark.parametrize('gate_mode', ['disabled', 'static_non_skip'])
def test_direct_prefill_known_non_skip_elides_redundant_staging_ring(gate_mode):
    model = make_model()
    idx = model.model.layers[0].self_attn.indexer
    idx.enable_layer_skip = gate_mode == 'static_non_skip'
    idx.dyn_dgate = False
    idx._skip_ids = set() if gate_mode == 'static_non_skip' else None
    reference = copy.deepcopy(model)
    length = 23
    ids = torch.arange(length)[None] % 32
    expected = reference(input_ids=ids, use_cache=True)
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    assert staging.states[0].mode == 'LINEAR_FINAL'
    actual = model(input_ids=ids, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
    assert staging.states[0].ring is None
    torch.testing.assert_close(staging.states[0].linear.keys[..., :length, :],
                               expected.past_key_values.layers[0].keys, rtol=0, atol=0)
    finalize_direct_prefill(cache, model, length)
    assert type(cache.layers[0]) is TLIStaticLayer
    torch.testing.assert_close(cache.layers[0].keys[..., :length, :],
                               expected.past_key_values.layers[0].keys, rtol=0, atol=0)


def test_direct_prefill_static_skip_gate_flip_fails_without_linear_fallback():
    model = make_model()
    length = 23
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, length)
    model(input_ids=torch.arange(length)[None] % 32,
          past_key_values=cache, use_cache=True)
    model.model.layers[0].self_attn.indexer._skip_ids = set()
    with pytest.raises(ValueError, match='gate changed after final storage'):
        finalize_direct_prefill(cache, model, length)
    assert staging.states[0].linear.keys is None
    assert cache.layers[0] is staging.states[0].linear


def test_direct_prefill_rejects_explicit_mask_before_writing():
    model = make_model()
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, 7)
    ids = torch.arange(7)[None] % 32
    mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match='unpadded mask-free'):
        model(input_ids=ids, past_key_values=cache, use_cache=True,
              attention_mask=mask)
    assert not staging.states[0].written
    assert cache.layers[0].keys is None


def test_direct_prefill_requires_finalize_before_decode():
    model = make_model()
    cache = TLIStaticCache(model.config, 128, 1)
    staging = prepare_direct_prefill(cache, model, 23)
    model(input_ids=torch.arange(23)[None] % 32, past_key_values=cache, use_cache=True)
    before = staging.states[0].ring.keys.clone()
    with pytest.raises(ValueError, match='finalize direct prefill before decode'):
        model(input_ids=torch.zeros(1, 1, dtype=torch.long),
              position_ids=torch.tensor([[23]]), cache_position=torch.tensor([23]),
              past_key_values=cache, use_cache=True)
    torch.testing.assert_close(staging.states[0].ring.keys, before, rtol=0, atol=0)
    assert cache.layers[0] is staging.states[0].linear


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


def test_direct_graph_runner_prefill_decode_and_second_request():
    model = make_model()
    reference = copy.deepcopy(model)
    reference.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
    runner = make_runner(model)
    with patch('sparse_attn.patches.qwen3_attn_patch.eager_decoding_attn',
               side_effect=AssertionError('ring reached legacy reader')):
        for length, offset in ((23, 0), (31, 7)):
            ids = (torch.arange(length) + offset)[None] % 32
            expected = reference(input_ids=ids, use_cache=True)
            full = expected.past_key_values
            runner.graph, runner.out = object(), object()
            runner.prepare_prefill(length)
            assert runner.graph is None and runner.out is None
            actual = model(input_ids=ids, past_key_values=runner.cache, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
            runner.finalize_prefill(length)
            assert runner.cache._far_skip_attention_mask is runner.m4_buf
            ctx = context(length)
            for pos in range(length, length + 18):
                token = (pos + offset) % 32
                expected_token = reference(input_ids=torch.tensor([[token]]),
                                           position_ids=torch.tensor([[pos]]),
                                           cache_position=torch.tensor([pos]),
                                           past_key_values=full, use_cache=True).logits
                ctx.seq_lens[0] = pos + 1
                with batch_decode_scope(ctx):
                    runner.inject(token, pos + 1)
                    actual_token = runner.forward().logits
                torch.testing.assert_close(actual_token, expected_token, rtol=2e-5, atol=2e-6)


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
    with pytest.raises(ValueError, match='Qwen3|compact far skip currently supports only'):
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
