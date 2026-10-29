"""Native exact-linear integration gates; CPU collection is not GPU acceptance.

Run only when the host has queued these tests on its reserved idle GPU. Negative
ready assertions run in an isolated child; run that test alone if the host also
requires a single NVML process during correctness testing.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch

from sparse_attn.ops import exact_linear_batch as exact
from sparse_attn.ops import batch_candidate_producer as front
from sparse_attn.ops.cuda_graph_state import graph_state, capture_graph
from sparse_attn.ops.kv_row_update import append_kv_batch
from sparse_attn.ops.operator_pipeline import decode_indexed
from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched_graph

NATIVE = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires queued native CUDA')
CONFIG = {
    'TLI_BATCH_INDEXER':'exact', 'TLI_OPERATOR_MODE':'serial',
    'TLI_SPARSE_SELECTION':'1', 'TLI_GRAPH_INC':'1',
    'TLI_OPERATOR_BACKEND':'simt', 'TLI_OPERATOR_SELECTED_STAGES':'1',
    'TLI_L2_BYPASS':'0', 'TLI_COMPACT_NORMALIZATION':'0',
    'TLI_GRAPH_GEOMETRY_SHARED':'1',
}


def _indexer():
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    spec = importlib.util.spec_from_file_location('native_batch_args',
        Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    args = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args)
    result = TLIIndexer(args._make_prod_args())
    result.layer_idx = 2
    result.clear()
    result._resolve_skip()
    assert not result.skip_far and not result.dyn_dgate
    return result


class _Case:
    """Fixed shared storage; reference indexers never share feature state."""
    capacity, hkv, group, dim = 32768, 2, 4, 128
    scale = 128 ** -.5

    def __init__(self, batch):
        assert Path(exact.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1], 'wrong tested source tree'
        device = torch.device('cuda', torch.cuda.current_device())
        if not exact.supported_geometry(torch.bfloat16,device,self.capacity,
                                         self.group,batch*self.hkv):
            raise RuntimeError('native integration test requires pinned supported geometry')
        self.batch = batch
        self.source = _indexer()
        self.refs = [_indexer() for _ in range(batch)]
        self.module = SimpleNamespace(scaling=self.scale)
        self.q = torch.randn(batch,1,self.hkv*self.group,self.dim,
                             device=device,dtype=torch.bfloat16)
        self.k = torch.empty(batch,self.capacity,self.hkv,self.dim,
                             device=device,dtype=self.q.dtype)
        self.v = torch.empty_like(self.k)
        self.new_k = torch.randn(batch,self.hkv,1,self.dim,device=device,dtype=self.q.dtype)
        self.new_v = torch.randn_like(self.new_k)
        self.gs = graph_state()
        self.gs.enabled, self.gs.bs = True, batch
        self.gs.pos_buf = torch.zeros(batch,device=device,dtype=torch.int64)
        self.gs.valid_len_buf = torch.zeros_like(self.gs.pos_buf)
        self.gs.alive_buf = torch.ones(batch,device=device,dtype=torch.bool)
        self.cu = torch.tensor([0,self.capacity],device=device,dtype=torch.int32)
        # Different live prefixes; several rows cross a 64-token page shortly.
        self.lengths = [16383+(b%3)*63-b*128 for b in range(batch)]
        self.alive = [True]*batch
        self.refill(self.lengths)

    def refill(self, lengths):
        self.lengths = list(lengths)
        self.alive = [True]*self.batch
        self.k.normal_(); self.v.normal_(); self.q.normal_()
        for b, length in enumerate(self.lengths):
            self.k[b,length:].zero_();self.v[b,length:].zero_()
        self.inject_metadata()
        self.new_k.normal_();self.new_v.normal_()

    def inject_metadata(self):
        self.gs.valid_len_buf.copy_(torch.tensor(self.lengths,device=self.q.device))
        self.gs.pos_buf.copy_(self.gs.valid_len_buf-1)
        self.gs.alive_buf.copy_(torch.tensor(self.alive,device=self.q.device))

    def run(self):
        append_kv_batch(self.k.transpose(1,2),self.v.transpose(1,2),
                        self.new_k,self.new_v,self.gs.pos_buf,self.gs.alive_buf)
        return _tl_decode_batched_graph(self.module,self.source,self.q,self.k,self.v,self.gs)

    def reference(self):
        rows=[]
        for b, idx in enumerate(self.refs):
            with exact._row_scope(self.gs,b):
                rows.append(decode_indexed(idx,None,self.q[b:b+1],
                    self.k[b:b+1],self.v[b:b+1],self.cu,self.gs.valid_len_buf,self.scale))
        return torch.cat(rows,dim=0)

    def capture(self):
        graph=torch.cuda.CUDAGraph()
        with capture_graph(graph):out=self.run()
        return graph,out


@contextmanager
def _case(batch):
    gs=graph_state();saved=vars(gs).copy()
    try:
        yield _Case(batch)
    finally:
        gs.__dict__.clear();gs.__dict__.update(saved)


def _install_stage_checks(monkeypatch, batch, capacity):
    """Count actual batch Python entry calls, not internal GPU launch counts."""
    calls=[]
    mapping = {
        'score_candidates_batched':'score', 'candidate_probabilities_batched':'norm',
        'rank_unique_batched':'rank', 'partial_attention':'attention', 'merge_partials':'merge',
    }
    for name,label in mapping.items():
        original=getattr(exact,name)
        def wrap(*args,_fn=original,_label=label,**kw):
            if _label=='merge':observed=args[0].m.shape[0]
            else:observed=args[0].shape[0]
            assert observed==batch
            stage=('protected' if kw.get('protected') else 'selected') if _label=='attention' else _label
            calls.append(stage)
            return _fn(*args,**kw)
        monkeypatch.setattr(exact,name,wrap)
    stack=torch.stack
    def checked_stack(tensors,*args,**kw):
        values=tuple(tensors)
        for value in values:
            assert not (value.ndim==4 and value.shape[1]==capacity), 'full-history feature stack'
            assert not (value.ndim==3 and value.shape[0]==capacity), 'squeezed full-history stack'
            if torch.cuda.is_current_stream_capturing():
                assert not (value.ndim==4 and value.shape[1]==(capacity+63)//64), 'summary stack during capture'
        return stack(values,*args,**kw)
    monkeypatch.setattr(torch,'stack',checked_stack)
    return calls


def _install_front_checks(monkeypatch,batch):
    calls=[]
    maintenance=exact.maintain_batched
    def maintain(*args,**kw):
        assert args[0].shape[0]==batch and kw.get('check_values') is False
        calls.append('maintenance')
        return maintenance(*args,**kw)
    monkeypatch.setattr(exact,'maintain_batched',maintain)
    for name,label in [('score_coarse_batched','coarse'),('score_average_batched','average'),
                       ('compact_candidates_exact','compact')]:
        original=getattr(front,name)
        def wrap(*args,_fn=original,_label=label,**kw):
            assert args[0].shape[0]==batch
            calls.append(_label)
            return _fn(*args,**kw)
        monkeypatch.setattr(front,name,wrap)
    return calls


@NATIVE
@pytest.mark.parametrize('batch',[2,4,8])
def test_exact_batch_native_replay_append_and_eos(monkeypatch,batch):
    for key,value in CONFIG.items():monkeypatch.setenv(key,value)
    torch.manual_seed(320301)
    with torch.no_grad(), _case(batch) as case:
        calls=_install_stage_checks(monkeypatch,batch,case.capacity)
        front_calls=_install_front_checks(monkeypatch,batch)
        expected_stages=['score','norm','rank','protected','selected','merge']
        for _ in range(3):
            start=len(calls);front_start=len(front_calls);out=case.run()
            assert calls[start:]==expected_stages
            assert front_calls[front_start:]==['maintenance','coarse','average','compact']
        torch.cuda.synchronize()
        torch.testing.assert_close(out,case.reference(),rtol=0,atol=0)
        state=case.source._exact_linear_batch
        assert len({id(row) for row in state.rows})==batch
        assert state.maintenance_route=='batched_tail32'
        assert case.source._exact_batch_maintenance_route=='batched_tail32'
        arena=state.feature_arena
        for b,row in enumerate(state.rows):
            assert row._ginc['k_qat'].data_ptr()==arena[b:b+1].data_ptr()
        start=len(calls);front_start=len(front_calls);graph,out=case.capture()
        assert calls[start:]==expected_stages
        assert front_calls[front_start:]==['maintenance','coarse','average','compact']
        first_page=(case.lengths[0]-1)//64
        dead_length=None
        for step in range(5):
            if step==1:
                case.alive[-1]=False
                dead_length=case.lengths[-1]
            case.lengths=[length+int(live) for length,live in zip(case.lengths,case.alive)]
            case.inject_metadata();case.q.normal_();case.new_k.normal_();case.new_v.normal_()
            dead_before={b:(case.k[b].clone(),case.v[b].clone())
                         for b,live in enumerate(case.alive) if not live}
            start=len(calls);front_start=len(front_calls);graph.replay()
            torch.cuda.synchronize()
            assert len(calls)==start  # Replay must execute GPU graph, not Python row consumers.
            assert len(front_calls)==front_start
            for b,(key,value) in dead_before.items():
                torch.testing.assert_close(case.k[b],key,rtol=0,atol=0)
                torch.testing.assert_close(case.v[b],value,rtol=0,atol=0)
            torch.testing.assert_close(out,case.reference(),rtol=0,atol=0)
            assert state.feature_arena is arena
            assert all(row._ginc['k_qat'].data_ptr()==arena[b:b+1].data_ptr()
                       for b,row in enumerate(state.rows))
        assert case.lengths[-1] == dead_length
        assert (case.lengths[0]-1)//64 > first_page
        assert case.gs.valid_len_buf.cpu().tolist() == case.lengths


@NATIVE
def test_same_storage_clear_warmup_old_and_new_graphs(monkeypatch):
    for key,value in CONFIG.items():monkeypatch.setenv(key,value)
    torch.manual_seed(320302)
    with torch.no_grad(), _case(2) as case:
        for _ in range(3):case.run()
        torch.cuda.synchronize();old_graph,old_out=case.capture()
        state=case.source._exact_linear_batch;arena=state.feature_arena
        addresses=[tuple(row._ginc[name].data_ptr() for name in ('k_qat','k_min','k_max'))
                   for row in state.rows]
        case.source.clear()
        assert not bool(state.ready.cpu()) and state.needs_warmup
        for ref in case.refs:ref.clear()
        case.refill([8191,12287])  # Same pointers, different request prefixes.
        for _ in range(3):out=case.run()
        torch.cuda.synchronize()
        assert case.source._exact_linear_batch is state and state.feature_arena is arena
        assert addresses==[tuple(row._ginc[name].data_ptr() for name in ('k_qat','k_min','k_max'))
                           for row in state.rows]
        old_graph.replay();torch.cuda.synchronize()
        expected=case.reference()
        torch.testing.assert_close(old_out,expected,rtol=0,atol=0)
        new_graph,new_out=case.capture()
        new_graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(new_out,expected,rtol=0,atol=0)
        old_graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(old_out,expected,rtol=0,atol=0)


@NATIVE
def test_changed_storage_old_graph_ready_failure_isolated():
    env=dict(os.environ,**CONFIG)
    child=Path(__file__).with_name('exact_linear_batch_failure_worker.py')
    proc=subprocess.run([sys.executable,str(child)],env=env,text=True,
                        capture_output=True,timeout=300)
    combined=proc.stdout+proc.stderr
    assert 'EXPECT_OLD_READY_FAILURE' in combined, combined
    assert proc.returncode != 0, combined
    assert ('exact batch was cleared' in combined or 'device-side assert' in combined),combined
