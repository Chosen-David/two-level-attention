"""CPU state/dispatch contracts; no native exactness or speed claim."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest
import torch
from sparse_attn.ops import exact_linear_batch as batch
from sparse_attn.ops.cuda_graph_state import graph_state
from sparse_attn.ops.candidate_inputs import candidate_inputs
from sparse_attn.ops.token_selection import TokenSelection


@pytest.fixture
def setup(monkeypatch):
    spec=importlib.util.spec_from_file_location('batch_args',Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    args=importlib.util.module_from_spec(spec);spec.loader.exec_module(args)
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    idx=TLIIndexer(args._make_prod_args());idx.layer_idx=2;idx.clear();idx.skip_far=False
    gs=graph_state();saved=vars(gs).copy()
    gs.enabled=True;gs.bs=2
    gs.pos_buf=torch.tensor([32767,32766]);gs.valid_len_buf=gs.pos_buf+1;gs.alive_buf=torch.tensor([True,False])
    q=torch.zeros(2,1,4,128,dtype=torch.bfloat16)
    k=torch.zeros(2,32768,1,128,dtype=q.dtype);v=torch.zeros_like(k)
    monkeypatch.setenv('TLI_OPERATOR_MODE','serial')
    monkeypatch.setenv('TLI_OPERATOR_BACKEND','simt')
    monkeypatch.setenv('TLI_OPERATOR_SELECTED_STAGES','1')
    monkeypatch.setenv('TLI_L2_BYPASS','0')
    monkeypatch.setenv('TLI_GRAPH_INC','1')
    monkeypatch.setenv('TLI_SPARSE_SELECTION','1')
    yield idx,q,k,v,gs
    gs.__dict__.clear();gs.__dict__.update(saved)


def test_first_route_rejects_cpu_before_allocating_owner(setup):
    idx,q,k,v,gs=setup
    batch._validate_metadata(idx,q,k,v,gs)
    with pytest.raises(ValueError,match='pinned CUDA'):
        batch._validate_supported(q,k)
    assert getattr(idx,'_exact_linear_batch',None) is None


@pytest.mark.parametrize('change', ['dtype','batch','skip','dynamic','mode','metadata','basis','sparse'])
def test_unsupported_scope_is_explicit(setup,monkeypatch,change):
    idx,q,k,v,gs=setup
    if change=='dtype':q=q.float()
    elif change=='batch':q=q[:1];k=k[:1];v=v[:1]
    elif change=='skip':idx.skip_far=True
    elif change=='dynamic':idx.dyn_dgate=True
    elif change=='mode':monkeypatch.setenv('TLI_OPERATOR_MODE','overlap')
    elif change=='metadata':gs.alive_buf=torch.ones(2,dtype=torch.int32)
    elif change=='basis':idx._basis=torch.zeros(1)
    elif change=='sparse':monkeypatch.setenv('TLI_SPARSE_SELECTION','0')
    with pytest.raises(ValueError):batch._validate_metadata(idx,q,k,v,gs)


def test_row_scope_restores_all_fields_on_failure(setup):
    *_,gs=setup
    saved=(gs.enabled,gs.bs,gs.pos_buf,gs.valid_len_buf,gs.alive_buf)
    with pytest.raises(RuntimeError,match='forced'):
        with batch._row_scope(gs,1):
            assert gs.bs==1 and gs.pos_buf.shape==(1,)
            assert gs.pos_buf.data_ptr()==saved[2][1:].data_ptr()
            raise RuntimeError('forced')
    assert gs.enabled==saved[0] and gs.bs==saved[1]
    assert all(x is y for x,y in zip((gs.pos_buf,gs.valid_len_buf,gs.alive_buf),saved[2:]))


def test_cloned_rows_do_not_share_mutable_index_state(setup):
    idx,*_=setup
    class Existing:
        def invalidate(self):raise AssertionError('source invalidated by clone.clear')
    old=Existing();idx._exact_linear_batch=old
    a=batch._clone_owner(idx);b=batch._clone_owner(idx)
    assert a is not b and a.args is not b.args and a.args is not idx.args
    assert a._ginc is None and b._ginc is None
    assert a._graph_const_entries is not b._graph_const_entries
    assert a._exact_linear_batch is None and idx._exact_linear_batch is old


def test_indexer_clear_poison_retains_ready_buffer(setup):
    idx,*_=setup
    calls=[]
    state=batch.ExactLinearBatch.__new__(batch.ExactLinearBatch)
    state.ready=torch.tensor(True);ready=state.ready
    state.rows=[SimpleNamespace(clear=lambda:calls.append('clear'))]
    idx._exact_linear_batch=state
    idx.clear()
    assert state.ready is ready and not bool(ready) and state.needs_warmup
    assert calls==['clear'] and idx._exact_linear_batch is state


def test_consumer_runs_one_batched_call_per_stage_and_never_stacks_k(setup,monkeypatch):
    idx,q,k,v,gs=setup
    arena=torch.zeros(2,32768,1,16,dtype=q.dtype)
    candidates=[]
    for b in range(2):
        candidates.append(candidate_inputs(arena[b:b+1],torch.zeros(4,16),
            torch.arange(1024)[None],torch.ones(1,1024,dtype=torch.bool),torch.tensor([1024]),
            gs.valid_len_buf[b],torch.tensor(0),torch.tensor(32000),torch.tensor(892),
            capacity=32768,group_size=4,output_dtype=q.dtype,rank_width=1024,sink=4,swa=128))
    calls=[];original_stack=torch.stack
    def safe_stack(tensors,*args,**kw):
        assert all(t.ndim!=4 for t in tensors),'full-history features must not be stacked'
        return original_stack(tensors,*args,**kw)
    monkeypatch.setattr(torch,'stack',safe_stack)
    def score(feature,query,ids,counts,**kw):
        assert feature is arena and ids.shape==(2,1,1024) and query.shape==(2,4,16)
        calls.append('score');return torch.zeros(2,4,1024,dtype=q.dtype)
    def norm(scores,ids,keep,*args,**kw):
        calls.append('norm');return torch.zeros_like(ids,dtype=q.dtype),keep
    def rank(ids,p,unique,lower,upper,quota,**kw):
        assert lower.shape==upper.shape==quota.shape==(2,)
        calls.append('rank');return TokenSelection(torch.zeros(2,1,1024,dtype=torch.int32),torch.zeros(2,1,dtype=torch.int32))
    def attend(query,key,value,packed,lengths,protected=False,**kw):
        assert query is q and key is k and value is v and lengths is gs.valid_len_buf
        calls.append('protected' if protected else 'selected');return ('state',)
    def merge(*args,**kw):calls.append('merge');return q
    for name,fn in [('score_candidates_batched',score),('candidate_probabilities_batched',norm),
                    ('rank_unique_batched',rank),('partial_attention',attend),('merge_partials',merge)]:
        monkeypatch.setattr(batch,name,fn)
    assert batch._consume(candidates,arena,q,k,v,gs.valid_len_buf,0.5) is q
    assert calls==['score','norm','rank','protected','selected','merge']


def test_exact_qwen_graph_route_cannot_fall_back(setup,monkeypatch):
    from sparse_attn.patches import qwen3_attn_patch as patch
    idx,q,k,v,gs=setup;monkeypatch.setenv('TLI_BATCH_INDEXER','exact')
    calls=[]
    def fail(*args):calls.append(args);raise NotImplementedError('producer unsupported')
    monkeypatch.setattr(batch,'exact_linear_batch_decode',fail)
    with pytest.raises(NotImplementedError,match='producer unsupported'):
        patch._tl_decode_batched_graph(SimpleNamespace(scaling=0.5),idx,q,k,v,gs)
    assert len(calls)==1 and not patch._parallel_batch_enabled(2)


def test_mixed_ring_rejected_before_mutation(monkeypatch):
    from sparse_attn.ops.hybrid_batch_cache import try_hybrid_batch_decode
    monkeypatch.setenv('TLI_BATCH_INDEXER','exact')
    with pytest.raises(ValueError,match='shared linear storage'):
        try_hybrid_batch_decode(None,None,None,None,None,None,None)


def test_actual_entry_warmup_binds_arena_and_reuses_row_state(setup,monkeypatch):
    idx,q,k,v,gs=setup
    gs.valid_len_buf.copy_(torch.tensor([32000,31000]));gs.pos_buf.copy_(gs.valid_len_buf-1)
    owners=[]
    class Row:
        def __init__(self):self._ginc=None;self.full=0;self.incremental=0
        def clear(self):
            if self._ginc is not None:self._ginc['stale']=True
        def prepare_index(self,key,cu):
            if self._ginc is None:
                self._ginc={'k_qat':key[...,:16].contiguous().clone()};self.full+=1
            elif self._ginc.pop('stale',False):
                self._ginc['k_qat'].copy_(key[...,:16]);self.full+=1
            else:
                self.incremental+=1
                pos=int(gs.pos_buf[0])
                self._ginc['k_qat'][:,pos].copy_(key[:,pos,...,:16])
            return self._ginc
        def compute_candidate_inputs_graph(self,query,index,scale,lengths):
            return candidate_inputs(index['k_qat'],(query[0,0,:,:16]*scale).float(),
                torch.arange(1024)[None],torch.ones(1,1024,dtype=torch.bool),torch.tensor([1024]),
                lengths[0],torch.tensor(0),lengths[0]-128,torch.tensor(892),
                capacity=32768,group_size=4,output_dtype=query.dtype,rank_width=1024,sink=4,swa=128)
    def clone(source):
        row=Row();owners.append(row);return row
    monkeypatch.setattr(batch,'_clone_owner',clone)
    monkeypatch.setattr(batch,'_validate_supported',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    consumed=[]
    def consume(candidates,arena,*args):
        assert len(candidates)==2
        assert all(c.k_fine.data_ptr()==arena[b:b+1].data_ptr() for b,c in enumerate(candidates))
        consumed.append(arena);return q
    monkeypatch.setattr(batch,'_consume',consume)
    assert batch.exact_linear_batch_decode(idx,q,k,v,gs,0.5) is q
    state=idx._exact_linear_batch;arena=state.feature_arena
    assert len(owners)==2 and owners[0] is not owners[1]
    assert all(row.full==1 for row in owners)
    gs.valid_len_buf.add_(gs.alive_buf.long());gs.pos_buf.copy_(gs.valid_len_buf-1)
    batch.exact_linear_batch_decode(idx,q,k,v,gs,0.5)
    assert idx._exact_linear_batch is state and consumed==[arena,arena]
    assert all(row.full==1 and row.incremental==2 for row in owners)
    idx.clear();assert not bool(state.ready) and state.needs_warmup
    batch.exact_linear_batch_decode(idx,q,k,v,gs,0.5)
    assert state.feature_arena is arena and all(row.full==2 for row in owners)
    assert bool(state.ready)
    # Metadata-compatible replacement storage must be explicit, not silent reuse.
    with pytest.raises(ValueError,match='storage/config changed'):
        batch.exact_linear_batch_decode(idx,q,k.clone(),v,gs,0.5)
    idx.clear()
    batch.exact_linear_batch_decode(idx,q,k.clone(),v,gs,0.5)
    assert idx._exact_linear_batch is not state
    assert state in idx._retired_exact_linear_batches and not bool(state.ready)


def test_uncaptured_state_cannot_initialize_inside_capture(setup,monkeypatch):
    idx,q,k,v,gs=setup
    monkeypatch.setattr(batch,'_validate_supported',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    with pytest.raises(RuntimeError,match='warm up exact'):
        batch.exact_linear_batch_decode(idx,q,k,v,gs,0.5)
    assert getattr(idx,'_exact_linear_batch',None) is None


def test_hybrid_snapshot_does_not_invalidate_source_exact_owner(setup):
    from sparse_attn.ops.hybrid_batch_cache import _snapshot_indexer
    idx,*_=setup
    class Existing:
        def invalidate(self):raise AssertionError('source exact owner invalidated')
    state=Existing();idx._exact_linear_batch=state
    other=_snapshot_indexer(idx)
    assert other._exact_linear_batch is None and idx._exact_linear_batch is state


@pytest.mark.parametrize('kind',['rewind','dead_advance','revival','wrong_slot'])
def test_device_lifecycle_guards_reject_invalid_step_on_cpu(setup,kind):
    _,q,k,_,gs=setup
    gs.valid_len_buf.copy_(torch.tensor([100,100]));gs.pos_buf.copy_(gs.valid_len_buf-1)
    state=batch.ExactLinearBatch.__new__(batch.ExactLinearBatch)
    state.ready=torch.tensor(True);state.rows=[]
    state.previous_lengths=torch.tensor([100,100]);state.previous_alive=torch.tensor([True,False])
    if kind=='rewind':gs.valid_len_buf[0]=99;gs.pos_buf.copy_(gs.valid_len_buf-1)
    elif kind=='dead_advance':gs.valid_len_buf[1]=101;gs.pos_buf.copy_(gs.valid_len_buf-1)
    elif kind=='revival':gs.alive_buf[1]=True
    elif kind=='wrong_slot':gs.pos_buf[0]=0
    with pytest.raises(RuntimeError,match='exact batch'):
        state.produce(q,k,gs,0.5)
