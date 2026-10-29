import pytest
import torch
from sparse_attn.ops.ordered_selection import ordered_middle_selection
from sparse_attn.ops.token_selection import middle_selection

@pytest.mark.parametrize('width',[0,1,127,1156,4097,8192,8193])
def test_cpu_ordered_contract(width):
    ids=torch.arange(width)[None,None].expand(2,3,-1).contiguous()
    valid=(ids%3!=1);lengths=torch.tensor([width//2,width+17])
    a=ordered_middle_selection(ids,valid,lengths,4,9,2**31-1)
    b=middle_selection(ids,valid,lengths,4,9,2**31-1)
    assert torch.equal(a.indices,b.indices) and torch.equal(a.counts,b.counts)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('width',[1,127,1156,4097,8192,8193])
@pytest.mark.parametrize('base',[0,2**30])
def test_cuda_graph_live_compaction(width,base):
    ids=(torch.arange(width,device='cuda',dtype=torch.int64)+base)[None,None].expand(2,3,-1).contiguous()
    valid=torch.ones_like(ids,dtype=torch.bool);lengths=torch.tensor([base+width,base+width+17],device='cuda')
    run=lambda:ordered_middle_selection(ids,valid,lengths,4,9,2**31-1)
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):a=run()
    for pattern in (0,1,2,0):
        valid.copy_((torch.arange(width,device='cuda')%3!=pattern).expand_as(valid))
        if pattern==0:valid.zero_()
        lengths.copy_(torch.tensor([base+width//2,base+width+17],device='cuda'))
        # Invalid slots intentionally break raw storage ordering; only the
        # producer's valid subsequence must be sorted and unique.
        ids.copy_(torch.where(valid,(torch.arange(width,device='cuda')+base).expand_as(ids),-999))
        graph.replay();b=middle_selection(ids,valid,lengths,4,9,2**31-1)
        assert torch.equal(a.indices,b.indices) and torch.equal(a.counts,b.counts)

@pytest.mark.parametrize('pages,budget',[(0,7),(2,7),(2,12),(4,40)])
def test_real_ring_producer_order(pages,budget):
    from sparse_attn.ops.far_skip import FarSkipConfig,FarSkipLayout
    from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
    cfg=FarSkipConfig(4,16,pages,budget,3,4);layout=FarSkipLayout(3,23)
    torch.manual_seed(883)
    for length in (1,2,3,4,7,8,15,16,31,97):
        k=torch.randn(1,layout.capacity,2,32);q=torch.randn(1,1,8,32);dims=torch.arange(32)
        ws=FarSkipGraphWorkspace(cfg,layout,2,dims,dims);ws.initialize(k,length)
        ids,valid=ws.select(q,k,torch.tensor([length]),32**-.5)
        for head in range(2):
            values=ids[head,valid[head]]
            assert bool((values[1:]>values[:-1]).all())
