"""Production selector equivalence with no dense post-scoring normalization."""
import importlib.util
import os
from pathlib import Path
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')

@pytest.mark.parametrize('capacity',[16400,16401,32768,32840])
@pytest.mark.parametrize('same_instance',[False,True])
@pytest.mark.parametrize('domain_pack',['fused','torch'])
@pytest.mark.parametrize('skip_far',[False,True])
@pytest.mark.parametrize('output_kind',['selection','mask'])
def test_real_sparse_selector_graph(monkeypatch,capacity,same_instance,domain_pack,skip_far,output_kind):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    if domain_pack == "torch":
        from sparse_attn.indexer import candidate_domain_triton
        monkeypatch.setattr(candidate_domain_triton,"complete_and_pack",lambda *args:None)
    spec=importlib.util.spec_from_file_location('prod_args',Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.manual_seed(887)
    q=torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,capacity,8,128,device='cuda',dtype=q.dtype)
    vl=torch.tensor([capacity],device='cuda');cu=torch.tensor([0,capacity],device='cuda',dtype=torch.int32)
    idx=TLIIndexer(mod._make_prod_args());idx.layer_idx=2;idx.clear()
    idx.skip_far=skip_far;idx._dyn_frozen=True
    index=idx.prepare_index(k,cu)
    reference=idx if same_instance else TLIIndexer(mod._make_prod_args())
    reference.layer_idx=2
    reference.skip_far=skip_far;reference._dyn_frozen=True
    method='compute_selection_graph' if output_kind=='selection' else 'compute_mask_graph'
    run=lambda:getattr(idx,method)(q,index,128**-.5,vl)
    monkeypatch.setenv('TLI_SPARSE_SELECTION','1')
    run()
    assert idx._graph_candidate_domain_route==domain_pack
    if capacity==16401 and os.environ.get('TLI_SPARSE_STATS','sorted') != 'owner':
        assert idx._graph_const_cache[1]['ar_tok'] is not None
        assert idx._graph_sparse_selection_fallback=='unsupported_static_geometry'
        assert idx._graph_sparse_selection_fallback_detail=='unaligned_requires_owner'
        assert idx._graph_ranking_route=='full_width_native'
    else:
        assert idx._graph_const_cache[1]['ar_tok'] is None,'sparse path fell back to dense geometry'
        assert idx._graph_sparse_selection_fallback is None
        assert idx._graph_sparse_selection_fallback_detail is None
        assert idx._graph_normalization_route=='sparse_candidate_domain'
        assert idx._graph_ranking_route=='candidate_domain_native_1024'
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    for length in (132,4097,min(32768,capacity),capacity,512,capacity):
        vl.fill_(length);q.normal_();graph.replay()
        monkeypatch.setenv('TLI_SPARSE_SELECTION','0')
        expected=getattr(reference,method)(q,index,128**-.5,vl)
        assert reference._graph_normalization_stats_route is None
        if output_kind=='mask':
            assert torch.equal(actual,expected),(capacity,length,'mask')
        else:
            assert torch.equal(actual.indices,expected.indices),(capacity,length,'IDs')
            assert torch.equal(actual.counts,expected.counts),(capacity,length,'counts')
