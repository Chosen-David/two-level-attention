import pytest
import torch
from sparse_attn.ops.tiled_compact import compact_tiled


def test_metadata_rejects_without_launch():
    with pytest.raises(ValueError,match='int64 IDs'):
        compact_tiled(torch.zeros(2,3),torch.ones(2,3,dtype=torch.bool),None,100)
    with pytest.raises(ValueError,match='mid must'):
        compact_tiled(torch.zeros(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.bool),torch.zeros(99,dtype=torch.bool),100)
    with pytest.raises(ValueError,match='CUDA'):
        compact_tiled(torch.zeros(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.bool),None,100)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='queued GPU gate')
@pytest.mark.parametrize('width',[0,1,1023,1024,1025,8448,33024])
@pytest.mark.parametrize('with_mid',[False,True])
def test_native_graph_stable_prefix_and_inactive_sentinel(monkeypatch,width,with_mid):
    from sparse_attn.indexer.tli_indexer import _iter3_compact
    torch.manual_seed(1030);h=8 if width>=8448 else 3;s=32840
    ids=torch.randint(s,(h,width),device='cuda');keep=torch.ones_like(ids,dtype=torch.bool)
    mid=torch.randint(2,(s,),device='cuda',dtype=torch.bool) if with_mid else None
    def run():
        monkeypatch.setenv('TLI_COMPACT_SCAN_TILED','1')
        return _iter3_compact(ids,keep,mid,width,h,s,ids.device)
    run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run()
    for pattern in ['all','none','alternating','random']:
        ids.random_(s)
        if pattern=='all':keep.fill_(True)
        if pattern=='none':keep.fill_(False)
        if pattern=='alternating':keep.copy_((torch.arange(width,device='cuda')%2==0)[None].expand(h,-1))
        if pattern=='random':keep.copy_(torch.rand(h,width,device='cuda')>.35)
        # The inactive input must never drive MID indirect reads.
        ids.masked_fill_(~keep,-(2**60))
        if mid is not None:mid.copy_(torch.randint(2,(s,),device='cuda',dtype=torch.bool))
        graph.replay()
        perm=(~keep).to(torch.int32).argsort(dim=-1,stable=True)
        expected_ids=torch.where(keep,ids,s-1).gather(-1,perm)
        expected_keep=keep.gather(-1,perm);expected_count=keep.sum(-1)
        assert torch.equal(actual[0],expected_ids)
        assert torch.equal(actual[1],expected_keep)
        assert torch.equal(actual[3],expected_count)
        if mid is not None:assert torch.equal(actual[2],mid[expected_ids])
        else:assert actual[2] is None
        monkeypatch.setenv('TLI_COMPACT_SCAN_TILED','0')
        old=_iter3_compact(ids,keep,mid,width,h,s,ids.device)
        for a,b in zip(actual,old):
            if a is not None:assert torch.equal(a,b),(width,with_mid,pattern)
