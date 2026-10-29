"""Fused packing preserves rank gating and dynamic-length Graph semantics."""
import pytest
import torch
from sparse_attn.ops.token_selection import graph_middle_selection


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('width', [1, 33, 1024, 4096])
@pytest.mark.parametrize('per_head', [False, True])
def test_pack_replay_matches_generic(width, per_head):
    from sparse_attn.ops.selection_pack import pack_unique_top
    torch.manual_seed(729)
    capacity, heads = 8192, 3
    top = torch.stack([torch.randperm(capacity, device='cuda')[:width]
                       for _ in range(heads)])
    quota = torch.full((heads if per_head else 1,), width, device='cuda')
    length = torch.tensor([capacity], device='cuda')
    mid = torch.ones(capacity, device='cuda', dtype=torch.bool)
    args = (top, quota, mid, length, capacity, 4, 128)
    assert pack_unique_top(*args) is not None
    graph_middle_selection(*args, unique_top=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = graph_middle_selection(*args, unique_top=True)
    for size in (0, 1, 4, 129, 1025, capacity):
        for budget in (0, 1, width, width + 7):
            length.fill_(size)
            quota.fill_(budget)
            if per_head:
                quota[0] = -1
            mid[::7] = False
            # Invalid positions must not cause indirect out-of-range reads.
            top[:, 0] = -1
            if width > 1:
                top[:, -1] = capacity + 9
            graph.replay()
            expected = graph_middle_selection(*args)
            assert torch.equal(actual.indices, expected.indices), (size, budget)
            assert torch.equal(actual.counts, expected.counts), (size, budget)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_pack_valid_singleton_and_layout_fallback(dtype):
    from sparse_attn.ops.selection_pack import pack_unique_top
    top = torch.tensor([[0], [7]], device='cuda', dtype=dtype)
    quota = torch.tensor([1], device='cuda', dtype=dtype)
    length = torch.tensor([8], device='cuda', dtype=dtype)
    mid = torch.ones(8, device='cuda', dtype=torch.bool)
    args = (top, quota, mid, length, 8, 0, 0)
    ids, counts = pack_unique_top(*args)
    assert ids.tolist() == [[[0], [7]]]
    assert counts.tolist() == [[1, 1]]
    assert pack_unique_top(top, 1, mid, length, 8, 0, 0) is None
    assert pack_unique_top(top, quota, mid, length.expand(2), 8, 0, 0) is None
    assert pack_unique_top(top, quota, mid[::2], length, 8, 0, 0) is None
    actual = graph_middle_selection(top, 1, mid, length, 8, 0, 0, unique_top=True)
    assert torch.equal(actual.indices, ids)
    assert torch.equal(actual.counts, counts)
