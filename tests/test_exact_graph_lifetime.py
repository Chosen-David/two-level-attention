"""CPU GC ownership contracts; native CUDA lifetime remains a separate gate."""
import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops import cuda_graph_state as cgs
from sparse_attn.ops import exact_linear_batch as exact


@pytest.fixture
def fake_capture(monkeypatch):
    @contextmanager
    def capture(graph,**kwargs):
        yield graph
    monkeypatch.setattr(torch.cuda,'graph',capture)


def test_capture_owner_nested_exception_restore_and_geometry_disabled(fake_capture,monkeypatch):
    monkeypatch.setenv('TLI_GRAPH_GEOMETRY_SHARED','0')
    outer=SimpleNamespace();inner=SimpleNamespace();state=SimpleNamespace()
    assert cgs.current_capture_graph() is None
    with cgs.capture_graph(outer):
        assert cgs.current_capture_graph() is outer
        cgs.retain_capture_state(state);cgs.retain_capture_state(state)
        assert outer._tli_address_owners==[state]
        with pytest.raises(ValueError,match='failure'):
            with cgs.capture_graph(inner):
                assert cgs.current_capture_graph() is inner
                raise ValueError('failure')
        assert cgs.current_capture_graph() is outer
    assert cgs.current_capture_graph() is None
    with pytest.raises(RuntimeError,match='owner scope'):cgs.retain_capture_state(state)


def test_replacement_state_and_kv_die_with_old_graph_while_indexer_lives(fake_capture,monkeypatch):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    # Preserve actual state constructor/storage ownership, clear invalidation,
    # and replacement entry. Only tensor compute/unsupported CPU guards are mocked.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('lifetime_args',Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    args=importlib.util.module_from_spec(spec);spec.loader.exec_module(args)
    idx=TLIIndexer(args._make_prod_args());idx.layer_idx=2;idx.clear()
    monkeypatch.setattr(exact,'_validate_metadata',lambda *a:None)
    monkeypatch.setattr(exact,'_validate_supported',lambda *a:None)
    monkeypatch.setattr(exact,'BatchedCandidateProducer',lambda *a:None)
    monkeypatch.setattr(exact,'_consume_batched',lambda *a:None)
    monkeypatch.setattr(exact.ExactLinearBatch,'produce',lambda *a:None)
    def warmup(state,*args):
        state.ready.fill_(True);state.needs_warmup=False
        state.feature_arena=torch.empty(2,8,1,32)
        state.summary_arenas={'k_min':torch.empty(2,1,1,32)}
    monkeypatch.setattr(exact.ExactLinearBatch,'warmup',warmup)
    capturing=False
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:capturing)
    q=torch.zeros(2,1,4,128);k=torch.zeros(2,8,1,128);v=torch.zeros_like(k)
    gs=SimpleNamespace(pos_buf=torch.tensor([7,7]),valid_len_buf=torch.tensor([8,8]),alive_buf=torch.ones(2,dtype=torch.bool))
    exact.exact_linear_batch_decode(idx,q,k,v,gs,1.)
    old=idx._exact_linear_batch;old_ref=weakref.ref(old);kv_ref=weakref.ref(k)
    fine_ref=weakref.ref(old.feature_arena);summary_ref=weakref.ref(old.summary_arenas['k_min'])
    class Graph:pass
    graph=Graph();capturing=True
    with cgs.capture_graph(graph):exact.exact_linear_batch_decode(idx,q,k,v,gs,1.)
    assert graph._tli_address_owners==[old]
    capturing=False;idx.clear()
    assert not bool(old.ready)
    new_k=torch.ones_like(k);new_v=torch.ones_like(v)
    exact.exact_linear_batch_decode(idx,q,new_k,new_v,gs,1.)
    assert idx._exact_linear_batch is not old
    assert not hasattr(idx,'_retired_exact_linear_batches')
    del old,k,v
    gc.collect()
    assert old_ref() is not None and kv_ref() is not None
    assert fine_ref() is not None and summary_ref() is not None
    assert not bool(old_ref().ready)
    del graph;gc.collect()
    assert old_ref() is None and kv_ref() is None
    assert fine_ref() is None and summary_ref() is None
    assert idx._exact_linear_batch.storage[0] is new_k


def test_raw_exact_capture_rejected_before_allocating_state(monkeypatch):
    monkeypatch.setattr(exact,'_validate_metadata',lambda *a:None)
    monkeypatch.setattr(exact,'_validate_supported',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    owner=SimpleNamespace(_resolve_skip=lambda:None)
    with pytest.raises(RuntimeError,match=r'capture_graph\(\) owner scope'):
        exact.exact_linear_batch_decode(owner,None,None,None,None,1.)
    assert not hasattr(owner,'_exact_linear_batch')
