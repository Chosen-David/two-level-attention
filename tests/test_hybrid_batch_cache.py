"""Real Qwen3 mixed-gate batches; CUDA tests exercise capture, never CPU mocks."""
import copy
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from test_far_skip_hybrid_cache import make_model, prefill, isolated_runtime  # noqa: F401
from sparse_attn.ops.cuda_graph_state import graph_state
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.hybrid_batch_cache import HybridBatchCacheBuilder
from sparse_attn.ops.hybrid_batch_runner import HybridBatchRunner


@pytest.fixture(autouse=True)
def cpu_reference_kernels(request, monkeypatch):
    if not request.node.name.startswith('test_cuda_'):
        # The existing eager TLI oracle has GPU-only fused dispatches. Exercise
        # its torch reference kernels on CPU, rather than mock their results.
        monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON', False)


def set_gate(model, skip):
    idx = model.model.layers[0].self_attn.indexer
    # Exercise the actual prefill calibration/freeze lifecycle. eps outside
    # probability range makes the two decisions deterministic for random Q/K.
    idx._skip_ids = None
    idx.dyn_dgate = True
    idx.dyn_dgate_eps = 2.0 if skip else -1.0


def build(model, lengths, skips, capacity=128):
    builder = HybridBatchCacheBuilder(model, capacity, len(lengths))
    for b, (length, skip) in enumerate(zip(lengths, skips)):
        set_gate(model, skip)
        dyn = prefill(model, length, offset=b)
        builder.add(dyn, length)
    return HybridBatchRunner(model, builder.finish())


def buffers(runner):
    return [t for layer in runner.cache.layers for row in getattr(layer, 'rows', [layer])
            for t in (row.keys, row.values)]


@pytest.mark.parametrize('batch', [1, 2, 8, 32])
def test_ragged_batch_matches_independent_retained_requests(batch):
    model = make_model()
    lengths = [[1, 7, 23, 31][b % 4] for b in range(batch)]
    skips = [b % 2 == 0 for b in range(batch)]
    refs, histories = [], []
    for b, (length, skip) in enumerate(zip(lengths, skips)):
        ref = copy.deepcopy(model)
        set_gate(ref, skip)
        ref.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
        histories.append(prefill(ref, length, offset=b))
        refs.append(ref)
    runner = build(model, lengths, skips)
    pointers = [x.data_ptr() for x in buffers(runner)]
    # Long enough to cross the 15-slot cyclic suffix more than twice.
    for step in range(35):
        tokens = [(b+step+3) % 32 for b in range(batch)]
        runner.inject(tokens, [length+step+1 for length in lengths])
        actual = runner.run().logits
        for b, ref in enumerate(refs):
            pos = lengths[b]+step
            expected = ref(input_ids=torch.tensor([[tokens[b]]]),
                           position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                           past_key_values=histories[b], use_cache=True).logits
            torch.testing.assert_close(actual[b:b+1], expected, rtol=3e-5, atol=3e-6, msg=f'batch={batch}, row={b}, step={step}')
    assert pointers == [x.data_ptr() for x in buffers(runner)]
    assert not graph_state().enabled


def test_short_non_page_aligned_capacity_with_large_l1_quota():
    model = make_model()
    model.model.layers[0].self_attn.indexer.args.tia_level1_topk = 128
    reference = copy.deepcopy(model)
    set_gate(reference, False)
    reference.model.layers[0].self_attn.indexer.far_skip_cache = 'retain'
    full = prefill(reference, 3)
    runner = build(model, [3], [False], capacity=17)
    for pos in range(3, 17):
        runner.inject([pos % 32], [pos+1])
        actual = runner.run().logits
        expected = reference(input_ids=torch.tensor([[pos % 32]]),
                             position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                             past_key_values=full, use_cache=True).logits
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_oversized_l2_fills_quota_without_duplicate_or_padding_slots():
    # Legacy eager and graph topk can break equal-zero-score ties differently.
    # Test quota semantics directly, then batch vs independent fixed-width rows.
    model = make_model(budget=32)
    runner = build(model, [7, 7], [False, False], capacity=65)
    references = []
    for b in range(2):
        reference = copy.deepcopy(model)
        set_gate(reference, False)
        builder = HybridBatchCacheBuilder(reference, 65, 1)
        builder.add(prefill(reference, 7, offset=b), 7)
        references.append(HybridBatchRunner(reference, builder.finish()))
    idx = runner.cache.layers[0].indexers[0]
    original = idx.compute_mask_graph
    masks = []
    def observe(*args, **kwargs):
        mask = original(*args, **kwargs)
        masks.append(mask.to_mask(65) if hasattr(mask, "to_mask") else mask.clone())
        return mask
    idx.compute_mask_graph = observe
    for length in range(8, 43):
        runner.inject([3, 4], [length, length])
        actual = runner.run().logits
        mask = masks[-1]
        assert not mask[..., length:].any()
        # Legacy sink-page floor semantics can spend up to sink_tokens quota
        # twice. Inactive candidate slots must not lose any further tokens.
        assert min(length, 32)-2 <= int(mask.sum()) <= min(length, 32)
        for b, ref in enumerate(references):
            ref.inject([3+b], [length])
            torch.testing.assert_close(actual[b:b+1], ref.run().logits, rtol=3e-5, atol=3e-6)


def test_all_patched_layers_keep_request_local_state():
    from types import MethodType
    from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward
    model = make_model()
    second = model.model.layers[1].self_attn
    second.indexer = copy.deepcopy(model.model.layers[0].self_attn.indexer)
    second.indexer.layer_idx, second.indexer._skip_ids = 1, {1}
    second.forward = MethodType(qwen3_attn_forward, second)
    lengths = [23, 29, 31, 35]
    references, histories = [], []
    for b, length in enumerate(lengths):
        ref = copy.deepcopy(model)
        set_gate(ref, b % 2 == 0)
        for layer in ref.model.layers:
            layer.self_attn.indexer.far_skip_cache = 'retain'
        histories.append(prefill(ref, length, offset=b))
        references.append(ref)
    runner = build(model, lengths, [True, False, True, False])
    for step in range(21):
        runner.inject([step % 32]*4, [length+step+1 for length in lengths])
        actual = runner.run().logits
        for b, ref in enumerate(references):
            pos = lengths[b]+step
            expected = ref(input_ids=torch.tensor([[step % 32]]),
                           position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                           past_key_values=histories[b], use_cache=True).logits
            torch.testing.assert_close(actual[b:b+1], expected, rtol=3e-5, atol=3e-6)


def test_storage_is_row_specific_and_prefill_history_is_released():
    model = make_model()
    builder = HybridBatchCacheBuilder(model, 128, 3)
    sources = []
    for length, skip in ((23, True), (31, False), (17, True)):
        set_gate(model, skip)
        dyn = prefill(model, length)
        sources.extend(weakref.ref(layer.keys) for layer in dyn.layers)
        builder.add(dyn, length)
        # add() itself consumes the source: each layer's prefill KV must be
        # dropped as soon as it is copied, before the caller releases dyn.
        gc.collect()
        assert all(source() is None for source in sources), (
            'add() must release consumed prefill KV layer by layer')
        del dyn
    runner = HybridBatchRunner(model, builder.finish())
    gc.collect()
    assert all(source() is None for source in sources)
    layer = runner.cache.layers[0]
    assert layer.keys is None and layer.values is None
    assert [isinstance(row, FarSkipStaticLayer) for row in layer.rows] == [True, False, True]
    assert [row.keys.shape[-2] for row in layer.rows] == [17, 128, 17]
    assert [idx.skip_far for idx in layer.indexers] == [True, False, True]
    assert len({id(idx) for idx in layer.indexers}) == 3
    # One patched layer with 2 rings + 1 linear row, one native dense layer.
    assert runner.kv_storage_bytes() == (2*17+128+3*128)*128*2*4
    with pytest.raises(ValueError, match='exactly once'):
        builder.finish()


def test_eos_freezes_both_ring_and_linear_and_native_dense_rows():
    model = make_model()
    runner = build(model, [23, 29, 7], [True, False, True])
    runner.inject([3, 4, 5], [24, 30, 8])
    runner.run()
    saved = [(row.keys.clone(), row.values.clone()) for row in runner.cache.layers[0].rows[:2]]
    dense_saved = [t[:2].clone() for t in (runner.cache.layers[1].keys, runner.cache.layers[1].values)]
    for step in range(34):
        runner.inject([31, 30, step % 32], [24, 30, 9+step], [False, False, True])
        runner.run()
    for row, pair in zip(runner.cache.layers[0].rows, saved):
        for actual, expected in zip((row.keys, row.values), pair):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    for actual, expected in zip((runner.cache.layers[1].keys[:2], runner.cache.layers[1].values[:2]), dense_saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='frozen'):
        runner.inject([1, 2, 3], [25, 30, 43], [True, False, True])


def test_runner_rejects_invalid_step_before_mutating_inputs_or_cache():
    runner = build(make_model(), [7, 23], [True, False], capacity=32)
    with pytest.raises(ValueError, match='first decode'):
        runner.forward()
    original = [x.clone() for x in buffers(runner)]
    for lengths in ([7, 24], [9, 24], [8, 33]):
        with pytest.raises(ValueError, match='append exactly'):
            runner.inject([2, 3], lengths)
    for actual, expected in zip(buffers(runner), original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert runner.ctx.seq_lens == [7, 23]


def test_inference_mode_geometry_updates_and_scope_restores_on_error():
    with torch.inference_mode():
        runner = build(make_model(), [7, 23], [True, False])
        for length in range(8, 20):
            runner.inject([3, 4], [length, length+16])
            assert torch.isfinite(runner.run().logits).all()
        # An exception must not leave row-local positions installed globally.
        runner.cache._far_skip_attention_mask = None
        with pytest.raises(ValueError, match='registered'):
            runner.forward()
        assert not graph_state().enabled
        assert graph_state().pos_buf is None


def test_production_batch_entry_and_repeated_batches(monkeypatch):
    import benchmark.RULER.pred_ruler_batch as production
    monkeypatch.setattr(production, 'MAX_GEN', 5)
    monkeypatch.setenv('TLI_CUDA_GRAPH', '0')
    model = make_model()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=31)
    for lengths, skip in (([7, 23, 13], True), ([11, 3], False), ([17]*8, True)):
        set_gate(model, skip)
        batch, width = len(lengths), max(lengths)
        ids, pos, mask = (torch.zeros(batch, width, dtype=torch.long) for _ in range(3))
        for b, length in enumerate(lengths):
            ids[b, -length:] = (torch.arange(length)+b) % 32
            pos[b, -length:] = torch.arange(length)
            mask[b, -length:] = 1
        result = production._decode_batch_chunk(model, tokenizer, [{}]*batch, ids, pos,
                                                mask, lengths, set(), False, fixed_cache=True)
        assert len(result) == batch and all(len(row) == 5 for row in result)
        assert not graph_state().enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA capture/replay')
@pytest.mark.parametrize('batch', [2, 8, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_cuda_mixed_batch_capture_replay_eos_and_new_request(batch, dtype):
    model = make_model(device='cuda').to(dtype)
    eager_model = copy.deepcopy(model)
    for repeat in range(2):
        lengths = [17+(b*7+repeat*3) % 20 for b in range(batch)]
        skips = [(b+repeat) % 2 == 0 for b in range(batch)]
        captured = build(model, lengths, skips)
        eager = build(eager_model, lengths, skips)
        pointers = [x.data_ptr() for x in buffers(captured)]
        for step in range(38):
            alive = [not (step > 4 and b % 3 == 0) for b in range(batch)]
            tokens = [(b+step) % 32 for b in range(batch)]
            lengths = [length+int(active) for length, active in zip(lengths, alive)]
            for runner in (captured, eager):
                runner.inject(tokens, lengths, alive)
            if step == 0:
                captured.capture()
            actual, expected = captured.run().logits, eager.run().logits
            torch.testing.assert_close(actual, expected, rtol=3e-3 if dtype == torch.float16 else 3e-4,
                                       atol=3e-4 if dtype == torch.float16 else 3e-5)
        assert captured.graph is not None
        assert pointers == [x.data_ptr() for x in buffers(captured)]
        for actual, expected in zip(buffers(captured), buffers(eager)):
            torch.testing.assert_close(actual, expected, rtol=3e-3 if dtype == torch.float16 else 3e-4,
                                       atol=3e-4 if dtype == torch.float16 else 3e-5, equal_nan=True)
        assert not graph_state().enabled
