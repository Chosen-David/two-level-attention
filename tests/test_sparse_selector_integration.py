"""Production selector equivalence with no dense post-scoring normalization."""
import importlib.util
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')

@pytest.mark.parametrize('capacity',[32768,32840])
@pytest.mark.parametrize('same_instance',[False,True])
def test_real_sparse_selector_graph(monkeypatch,capacity,same_instance):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    spec=importlib.util.spec_from_file_location('prod_args',Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(887)
    q=torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,capacity,8,128,device='cuda',dtype=q.dtype)
    vl=torch.tensor([capacity],device='cuda');cu=torch.tensor([0,capacity],device='cuda',dtype=torch.int32)
    idx=TLIIndexer(mod._make_prod_args());idx.layer_idx=2;idx.clear()
    index=idx.prepare_index(k,cu)
    reference=idx if same_instance else TLIIndexer(mod._make_prod_args())
    reference.layer_idx=2
    run=lambda:idx.compute_selection_graph(q,index,128**-.5,vl)
    monkeypatch.setenv('TLI_SPARSE_SELECTION','1')
    run()
    assert idx._graph_const_cache[1]['ar_tok'] is None,'sparse path fell back to dense geometry'
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    for length in (132,4097,32768,capacity,512,capacity):
        vl.fill_(length);q.normal_();graph.replay()
        monkeypatch.setenv('TLI_SPARSE_SELECTION','0')
        expected=reference.compute_selection_graph(q,index,128**-.5,vl)
        assert torch.equal(actual.indices,expected.indices),(capacity,length,'IDs')
        assert torch.equal(actual.counts,expected.counts),(capacity,length,'counts')
