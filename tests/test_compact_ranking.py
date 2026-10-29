import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')

@pytest.mark.parametrize('capacity',[4097,16400,32768,65536])
@pytest.mark.parametrize('k',[1,32,1024,4096])
@pytest.mark.parametrize('heads',[1,24])
def test_native_tie_prefix_replay(capacity,k,heads):
    from sparse_attn.ops.compact_ranking import compact_topk
    torch.manual_seed(951)
    r=min(1536,capacity//2)
    ids=torch.stack([torch.randperm(capacity,device='cuda')[:r] for _ in range(heads)]).contiguous()
    keep=torch.ones_like(ids,dtype=torch.bool)
    p=torch.zeros((heads,capacity),device='cuda',dtype=torch.bfloat16)
    mid=torch.ones(capacity,device='cuda',dtype=torch.bool)
    compact_topk(p,ids,keep,mid,k)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=compact_topk(p,ids,keep,mid,k)
    for kind in ('random','ties','zero','underflow','partial','short','empty'):
        ids.copy_(torch.stack([torch.randperm(capacity,device="cuda")[:r] for _ in range(heads)]))
        p.zero_();keep.fill_(True);mid.fill_(True)
        scores=torch.rand((heads,r),device='cuda',dtype=p.dtype)
        if kind=='ties':scores.fill_(0.5)
        if kind=='zero':scores.zero_()
        if kind=='underflow':scores.mul_(1e-40)
        if kind=='partial':
            keep.random_(2);scores.masked_fill_(~keep,0.)
        if kind=='short':mid[k//2:]=False
        if kind=='empty':mid.fill_(False)
        p.scatter_(1,ids,scores);p.masked_fill_(~mid,-torch.inf)
        graph.replay()
        expected=p.topk(k,dim=-1).indices
        assert torch.equal(actual,expected),(capacity,k,heads,kind)


def test_zero_candidate_domain():
    from sparse_attn.ops.compact_ranking import compact_topk
    p=torch.zeros((2,16400),device='cuda',dtype=torch.bfloat16)
    ids=torch.empty((2,0),device='cuda',dtype=torch.int64)
    keep=torch.empty_like(ids,dtype=torch.bool)
    mid=torch.ones(16400,device='cuda',dtype=torch.bool)
    for k in (1,32,4096):
        assert torch.equal(compact_topk(p,ids,keep,mid,k),p.topk(k).indices)

@pytest.mark.parametrize('capacity',[16400,32832])
def test_production_rank_replay(capacity,monkeypatch):
    import runpy
    from pathlib import Path
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    torch.manual_seed(952)
    helpers=runpy.run_path(str(Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py'))
    idx=TLIIndexer(helpers['_make_prod_args']());idx.layer_idx=2;idx.clear()
    q=torch.randn((1,1,32,128),device='cuda',dtype=torch.bfloat16)
    key=torch.randn((1,capacity,8,128),device='cuda',dtype=q.dtype)
    length=torch.tensor([capacity],device='cuda')
    cu=torch.tensor([0,capacity],device='cuda',dtype=torch.int32)
    index=idx.prepare_index(key,cu)
    monkeypatch.setenv('TLI_COMPACT_NORMALIZATION','1')
    monkeypatch.setenv('TLI_COMPACT_RANKING','1')
    def run():return idx.compute_selection_graph(q,index,128**-.5,length)
    run();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    assert idx._graph_ranking_route=='compact_threshold_native_order'
    for n in (132,4097,capacity,512,capacity-1):
        length.fill_(n);q.normal_();graph.replay()
        monkeypatch.setenv('TLI_COMPACT_RANKING','0');expected=run()
        assert idx._graph_ranking_route=='full_width_native'
        assert torch.equal(actual.indices,expected.indices),n
        assert torch.equal(actual.counts,expected.counts),n
    from sparse_attn.ops import compact_ranking
    monkeypatch.setenv('TLI_COMPACT_RANKING','1')
    monkeypatch.setattr(compact_ranking,'compact_topk',lambda *args:None)
    fallback=run()
    assert idx._graph_ranking_route=='full_width_native'
    assert idx._graph_ranking_fallback=='unsupported_static_geometry'
    assert torch.equal(fallback.indices,expected.indices)
    assert torch.equal(fallback.counts,expected.counts)

def test_unsupported_static_geometry():
    from sparse_attn.ops.compact_ranking import compact_topk
    p=torch.zeros((1,16400),device='cuda',dtype=torch.bfloat16)
    ids=torch.zeros((1,1),device='cuda',dtype=torch.int64)
    keep=torch.ones_like(ids,dtype=torch.bool)
    mid=torch.ones(16400,device='cuda',dtype=torch.bool)
    assert compact_topk(p,ids,keep,mid,4097) is None
    assert compact_topk(p.float(),ids,keep,mid,32) is None
    assert compact_topk(p[:0],ids[:0],keep[:0],mid,32) is None
