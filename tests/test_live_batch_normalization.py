"""Batched native live-width reduction must equal each row's eager softmax."""
import pytest
import torch
from sparse_attn.ops.sparse_normalization import candidate_probabilities_batched
from test_sparse_selection_cuda import dense_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@pytest.mark.parametrize('group', [1, 2, 4, 8])
def test_live_batch_ragged_capture_exact(batch, group):
    torch.manual_seed(906)
    heads, width, capacity = 2, 2048, 32768
    ids = torch.randint(capacity, (batch, heads, width), device='cuda')
    ids[:, :, -128:] = ids[:, :, :128]  # repeated owners must count only once
    raw = torch.randn(batch, heads*group, capacity, device='cuda', dtype=torch.bfloat16)
    scores = raw.gather(-1, ids.repeat_interleave(group, 1)).contiguous()
    lengths = torch.full((batch,), 16384, device='cuda', dtype=torch.int64)
    keep = (ids < lengths[:, None, None]).contiguous()
    def run():
        return candidate_probabilities_batched(scores, ids, keep, capacity, group,
                                               return_unique=True, valid_lengths=lengths)
    run(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual, unique = run()
    for step in range(3):
        logical = [[9217, 16384, 16385, 32612][(b+step) % 4] for b in range(batch)]
        lengths.copy_(torch.tensor(logical, device='cuda'))
        raw.normal_()
        if step == 1: raw.zero_()
        scores.copy_(raw.gather(-1, ids.repeat_interleave(group, 1)))
        keep.copy_(ids < lengths[:, None, None])
        graph.replay()
        for b, length in enumerate(logical):
            full = dense_reference(scores[b], ids[b], keep[b], length, group)
            expected = full.gather(1, ids[b].clamp(max=length-1)).masked_fill(~keep[b], 0)
            assert torch.equal(actual[b], expected), (batch, group, b, length)
            for head in range(heads):
                assert ids[b, head, unique[b, head]].numel() == ids[b, head, keep[b, head]].unique().numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_live_extension_rejects_lengths_splitting_gqa_groups():
    from sparse_attn.ops.compact_normalization import build
    scores = torch.zeros(8, 16, device='cuda', dtype=torch.bfloat16)
    owners = torch.full((4, 16384), -1, device='cuda', dtype=torch.int32)
    lengths = torch.full((3,), 16384, device='cuda', dtype=torch.int64)
    with pytest.raises(RuntimeError, match='dividing KV heads'):
        build().virtual_compact_softmax_live(scores, owners, 16384, 2, False,
                                            torch.empty(0, device='cuda', dtype=torch.bool), lengths)
    assert (torch.ones(1, device='cuda') + 1).item() == 2
