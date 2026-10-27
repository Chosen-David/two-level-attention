"""Cache-policy equivalence, row capacity and owner lifetime."""
import gc
import weakref
import pytest
import torch
import sparse_attn.indexer.tli_indexer as tli
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
from exp.bench.geometry_cache_compare import legacy_functions, policy, tiny_model
from test_hybrid_batch_cache import build
from test_far_skip_hybrid_cache import isolated_runtime  # noqa: F401


@pytest.mark.parametrize('batch', [1, 2, 8, 9, 10, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_eager_pair_counts_and_request_lifetime(batch, dtype, monkeypatch):
    monkeypatch.setattr(tli, '_HAS_TRITON', False)
    candidate = tli._graph_geo_lookup, tli._graph_geo_store
    models = [tiny_model(dtype=dtype) for _ in range(2)]
    original = tli.TLIIndexer._build_graph_geometry
    calls = []
    def count(self, *args, **kwargs):
        calls.append(self.layer_idx)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(tli.TLIIndexer, '_build_graph_geometry', count)
    for request in range(2):
        lengths = [23+(b+request) % 5 for b in range(batch)]
        skips = [False]*batch if request == 0 else [b % 2 == 0 for b in range(batch)]
        runners = [build(model, lengths, skips) for model in models]
        policies = [legacy_functions(), candidate]
        for step in range(35):
            alive = [not (step > 8 and b % 3 == 0) for b in range(batch)]
            lengths = [n+int(a) for n, a in zip(lengths, alive)]
            outputs, counts = [], []
            for runner, funcs in zip(runners, policies):
                with policy(funcs):
                    runner.inject([(b+step)%32 for b in range(batch)], lengths, alive)
                    calls.clear()
                    outputs.append(runner.run().logits.clone())
                    counts.append(len(calls))
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            assert counts[1] == batch
            if request == 0:
                assert counts[0] == batch*(2 if batch > 9 else 1)
            cache = runners[1].ctx._tli_graph_geo_cache
            assert len(cache) == batch
        # Captured geometry may hold graph-pool tensors: no global owner should
        # retain the candidate batch after every request owner is dropped.
        refs = [weakref.ref(entry[0]) for entry in cache.values()]
        refs += [weakref.ref(t) for entry in cache.values() for t in entry[3].values()
                 if isinstance(t, torch.Tensor)]
        del runners, runner, cache, outputs
        gc.collect()
        assert all(ref() is None for ref in refs)


def test_scope_version_geometry_key_bound_and_inference():
    ctxs = [BatchCtx(32, [3]*32, [0]*32, [True]*32, 0) for _ in range(2)]
    length = torch.tensor([3])
    with batch_decode_scope(ctxs[0]):
        marker = object()
        tli._graph_geo_store(length, ('shape1',), marker)
        assert tli._graph_geo_lookup(length, ('shape1',)) is marker
        assert tli._graph_geo_lookup(length, ('shape2',)) is None
        with batch_decode_scope(ctxs[1]):
            assert tli._graph_geo_lookup(length, ('shape1',)) is None
        assert tli._graph_geo_lookup(length, ('shape1',)) is marker
        length.add_(1)
        assert tli._graph_geo_lookup(length, ('shape1',)) is None
        for _ in range(80):
            tli._graph_geo_store(torch.tensor([4]), ('shape1',), object())
            assert len(ctxs[0]._tli_graph_geo_cache) <= 32
        with torch.inference_mode():
            inference = torch.tensor([4])
            assert tli._graph_geo_store(inference, (), marker) is marker
            assert tli._graph_geo_lookup(inference, ()) is None
    with batch_decode_scope(None):
        tli._GRAPH_GEO_CACHE.clear()
        for _ in range(20):
            tli._graph_geo_store(torch.tensor([4]), (), marker)
            assert len(tli._GRAPH_GEO_CACHE) <= 9
        tli._GRAPH_GEO_CACHE.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
@pytest.mark.parametrize('batch', [2, 10, 32])
def test_cuda_capture_replay_and_new_batch(batch):
    candidate = tli._graph_geo_lookup, tli._graph_geo_store
    models = [tiny_model('cuda', torch.float16) for _ in range(2)]
    for request in range(2):
        lengths = [23+(b+request) % 7 for b in range(batch)]
        runners = [build(model, lengths, [(b+request)%2 == 0 for b in range(batch)]) for model in models]
        policies = [legacy_functions(), candidate]
        for runner, funcs in zip(runners, policies):
            with policy(funcs):
                runner.inject([3]*batch, [n+1 for n in lengths])
                runner.capture()
        lengths = [n+1 for n in lengths]
        for step in range(36):
            outputs = []
            for runner in runners:
                if step:
                    alive = [not (step > 8 and b % 3 == 0) for b in range(batch)]
                    runner.inject([(b+step)%32 for b in range(batch)],
                                  [n+int(a) for n,a in zip(runner.ctx.seq_lens, alive)], alive)
                outputs.append(runner.run().logits.clone())
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
        del runners, runner, outputs, policies
