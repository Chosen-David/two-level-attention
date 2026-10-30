import pytest
import torch
from sparse_attn.indexer.candidate_domain import complete_protected_candidates


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('sink,swa', [(0,0), (4,128), (64,64), (512,512)])
@pytest.mark.parametrize('width', [0, 33, 1089, 8448])
def test_complete_pack_replay(sink, swa, width):
    from sparse_attn.indexer.candidate_domain_triton import complete_and_pack
    torch.manual_seed(730)
    capacity = 257
    ids = torch.randint(-2, capacity+2, (3,width), device='cuda')
    keep = (ids >= 0) & (ids < capacity) & (ids % 3 == 0)
    length = torch.tensor([capacity], device='cuda')
    args = (ids, keep, length, sink, swa, capacity)
    complete_and_pack(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_ids, actual_valid, actual_counts = complete_and_pack(*args)
    for size in (0,1,4,65,129,capacity):
        length.fill_(size)
        graph.replay()
        raw, valid = complete_protected_candidates(*args)
        raw = torch.where(valid, raw, capacity-1)
        order = torch.argsort(~valid, dim=-1, stable=True)
        assert torch.equal(actual_ids, raw.gather(-1,order)), size
        assert torch.equal(actual_valid, valid.gather(-1,order)), size
        assert torch.equal(actual_counts, valid.sum(-1)), size
