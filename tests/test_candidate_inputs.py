"""CPU producer contract/roundtrip; native exactness is a separate GPU gate."""
import importlib.util
from pathlib import Path
import pytest
import torch
from sparse_attn.ops.candidate_inputs import candidate_inputs


def inputs():
    return dict(k_fine=torch.randn(1, 32, 2, 16, dtype=torch.bfloat16),
                q_fine=torch.randn(8, 16), ids=torch.tensor([[2,3,31],[7,0,0]]),
                keep=torch.tensor([[True,True,False],[True,False,False]]),
                counts=torch.tensor([2,1]), length=torch.tensor(20),
                lower=torch.tensor(0), upper=torch.tensor(16), quota=torch.tensor(8),
                capacity=32, group_size=4, output_dtype=torch.bfloat16,
                rank_width=16, sink=4, swa=4)


def test_metadata_preserves_tensor_identity_and_tail():
    args=inputs();result=candidate_inputs(**args)
    for name in ('k_fine','q_fine','ids','keep','counts','length','lower','upper','quota'):
        assert getattr(result,name) is args[name]
    # Inactive tail is not rewritten or deduplicated by this boundary.
    assert result.ids[0,-1] == 31


@pytest.mark.parametrize('field,value', [
    ('group_size',0), ('rank_width',33), ('sink',-1),
    ('q_fine',torch.zeros(8,16,dtype=torch.bfloat16)),
    ('counts',torch.tensor([2,1],dtype=torch.int32)),
    ('quota',torch.tensor([8])), ('output_dtype',torch.float16),
    ('ids',torch.zeros(2,3,dtype=torch.int32)),
    ('keep',torch.zeros(2,4,dtype=torch.bool)),
])
def test_metadata_rejects_incompatible_shapes_and_dtypes(field,value):
    args=inputs();args[field]=value
    with pytest.raises(ValueError):candidate_inputs(**args)


@pytest.fixture
def real_indexer(monkeypatch):
    from sparse_attn.indexer import tli_indexer as mod
    path=Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py'
    spec=importlib.util.spec_from_file_location('candidate_test_args',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(mod,'_HAS_TRITON',False)
    monkeypatch.setenv('TLI_SPARSE_SELECTION','0')
    monkeypatch.setenv('TLI_COMPACT_NORMALIZATION','0')
    monkeypatch.setenv('TLI_L2_BYPASS','0')
    indexer=mod.TLIIndexer(module._make_prod_args());indexer.layer_idx=2;indexer.clear()
    # Reduce CPU scheduling noise in this semantic test, not a timing claim.
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield indexer
    torch.set_num_threads(before)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('capacity,length',[(128,33),(1024,132),(1024,1000)])
def test_real_cpu_producer_reconstructs_unchanged_selection(real_indexer,dtype,capacity,length):
    from sparse_attn.ops.token_selection import graph_middle_selection
    torch.manual_seed(19)
    indexer=real_indexer
    q=torch.randn(1,1,8,128,dtype=dtype)
    k=torch.randn(1,capacity,2,128,dtype=dtype)
    vl=torch.tensor([length]);cu=torch.tensor([0,capacity],dtype=torch.int32)
    index=indexer.prepare_index(k,cu)
    c=indexer.compute_candidate_inputs_graph(q,index,128**-.5,vl)
    assert c.k_fine is index["k_qat"]
    dims=indexer._subspace_indices(q.device) if indexer.enable_subspace else indexer._qat_indices(q.device)
    expected_q=(q[0,0]*(128**-.5)).float()
    if indexer.l2_subspace != "full":
        expected_q=expected_q[...,dims]
    torch.testing.assert_close(c.q_fine,expected_q,rtol=0,atol=0)
    rank=torch.arange(c.ids.shape[-1])[None]
    assert torch.equal(c.keep,rank<c.counts[:,None])
    assert torch.all((~c.keep)|((c.ids>=0)&(c.ids<length)))
    keys=c.k_fine[0][c.ids,torch.arange(2)[:,None]].float()
    keys=keys.repeat_interleave(c.group_size,0)
    score=torch.einsum('hd,hrd->hr',c.q_fine,keys).to(c.output_dtype)
    ids=c.ids.repeat_interleave(c.group_size,0)
    keep=c.keep.repeat_interleave(c.group_size,0)
    full=torch.full((8,capacity+1),-torch.inf,dtype=dtype)
    full.scatter_(1,torch.where(keep,ids,capacity),score)
    p=full[:,:capacity].softmax(-1).nan_to_num(nan=0).reshape(2,4,capacity).mean(1)
    mid=(torch.arange(capacity)>=c.lower)&(torch.arange(capacity)<c.upper)
    top=p.masked_fill(~mid,-torch.inf).topk(c.rank_width,-1).indices
    rebuilt=graph_middle_selection(top,c.quota,mid,c.length,c.capacity,c.sink,c.swa)
    direct=indexer.compute_selection_graph(q,index,128**-.5,vl)
    torch.testing.assert_close(rebuilt.indices,direct.indices,rtol=0,atol=0)
    torch.testing.assert_close(rebuilt.counts,direct.counts,rtol=0,atol=0)
    # Existing full-mask default return stays a tensor and reconstructs protection.
    mask=indexer.compute_mask_graph(q,index,128**-.5,vl)
    pos=torch.arange(capacity)
    protected=((pos<min(length,c.sink))|((pos>=max(min(length,c.sink),length-c.swa))&(pos<length)))
    torch.testing.assert_close(mask[:,0],rebuilt.to_mask(capacity)|protected,rtol=0,atol=0)


def test_producer_excludes_bypass_and_multi_request(real_indexer,monkeypatch):
    q=torch.randn(1,1,8,128,dtype=torch.bfloat16)
    k=torch.randn(1,128,2,128,dtype=q.dtype)
    index=real_indexer.prepare_index(k,torch.tensor([0,128],dtype=torch.int32))
    monkeypatch.setenv('TLI_L2_BYPASS','1')
    with pytest.raises(NotImplementedError,match='L2 bypass'):
        real_indexer.compute_candidate_inputs_graph(q,index,1.,torch.tensor([128]))
    with pytest.raises(ValueError,match='one decode request'):
        real_indexer.compute_candidate_inputs_graph(q.expand(2,-1,-1,-1),index,1.,torch.tensor([128,128]))


def test_producer_stops_before_fine_normalization_and_ranking(real_indexer,monkeypatch):
    from sparse_attn.indexer import tli_indexer as mod
    q=torch.randn(1,1,8,128,dtype=torch.bfloat16)
    k=torch.randn(1,1024,2,128,dtype=q.dtype)
    index=real_indexer.prepare_index(k,torch.tensor([0,1024],dtype=torch.int32))
    original=mod.einsum
    def no_fine(*args,**kwargs):
        if any(isinstance(a,str) and 'h r d' in a for a in args):
            raise AssertionError('fine score was consumed')
        return original(*args,**kwargs)
    def no_softmax(*args,**kwargs):
        raise AssertionError('normalization was consumed')
    monkeypatch.setattr(mod,'einsum',no_fine)
    monkeypatch.setattr(mod.F,'softmax',no_softmax)
    result=real_indexer.compute_candidate_inputs_graph(q,index,128**-.5,torch.tensor([1000]))
    assert result.ids.shape[0] == 2
