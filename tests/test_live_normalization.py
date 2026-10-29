"""Dynamic logical width must preserve native per-row softmax rounding."""
import pytest
import torch
from sparse_attn.ops.compact_normalization import build

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('capacity,lengths',[(16448,[16384,16385,16400,16401]),(32768,[32617,32626,32640,32676])])
def test_live_width_capture_replay_exact_probabilities(capacity,lengths):
    torch.manual_seed(419)
    scores=torch.randn(8,capacity,device='cuda',dtype=torch.bfloat16)*3
    # Leave substantial sparse gaps without changing normalization domain.
    scores[:,::3]=-torch.inf
    owner=torch.arange(capacity,device='cuda',dtype=torch.int32)[None].repeat(2,1)
    live=torch.tensor([lengths[0]],device='cuda',dtype=torch.int64)
    empty=torch.empty(0,device='cuda',dtype=torch.bool)
    ext=build()
    for _ in range(2):ext.virtual_compact_softmax_live(scores,owner,capacity,4,False,empty,live)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output=ext.virtual_compact_softmax_live(scores,owner,capacity,4,False,empty,live)
    for length in lengths:
        live.fill_(length);graph.replay()
        expected=scores[:,:length].contiguous().softmax(-1)
        assert torch.equal(output[:,:length],expected), (length,int((output[:,:length]!=expected).sum()))


def test_bypass_does_not_claim_normalization(monkeypatch):
    import runpy
    from pathlib import Path
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    args=runpy.run_path(str(Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py'))['_make_prod_args']
    idx=TLIIndexer(args());idx.layer_idx=14;idx.clear()
    k=torch.randn(1,16384,2,128,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(1,1,8,128,device='cuda',dtype=k.dtype)
    index=idx.prepare_index(k,torch.tensor([0,16384],device='cuda',dtype=torch.int32))
    monkeypatch.setenv('TLI_LIVE_NORMALIZATION','1')
    monkeypatch.setenv('TLI_L2_BYPASS','1')
    idx._graph_normalization_width_route='live_length'
    idx.compute_mask_graph(q,index,128**-.5,torch.tensor([16384],device='cuda'))
    assert idx._graph_normalization_width_route is None


@pytest.mark.parametrize('group', [1, 2, 4, 8])
@pytest.mark.parametrize('length', [9217, 16384, 16385])
def test_live_candidate_dedup_gqa_matches_native(group, length):
    from sparse_attn.ops.compact_normalization import compact_probabilities
    torch.manual_seed(904)
    capacity = 16448
    # Repeated protected IDs and sparse holes exercise owner reduction rather
    # than a dense identity map. Duplicate fine logits are identical by contract.
    ids = torch.cat((torch.arange(0, capacity, 3), torch.arange(4),
                     torch.arange(length - 128, length))).cuda().int()[None]
    keep = ids < length
    dense = torch.randn(group, capacity, device='cuda', dtype=torch.bfloat16) * 3
    scores = dense[:, ids[0].long()].contiguous()
    excluded = torch.arange(capacity, device='cuda') >= length - 128
    excluded[:4] = True
    live = torch.tensor([length], device='cuda', dtype=torch.int64)
    actual = compact_probabilities(scores, ids, keep, capacity, group, excluded, live)
    domain = torch.zeros(capacity, device='cuda', dtype=torch.bool)
    domain[ids[keep].long()] = True
    probs = dense[:, :length].masked_fill(~domain[:length], -torch.inf).contiguous().softmax(-1)
    # Eager BF16 GQA rounds each probability before sequential FP32 sum.
    expected = torch.zeros(length, device='cuda', dtype=torch.float32)
    for row in probs:
        expected += row.float()
    expected = (expected / group).bfloat16().masked_fill(excluded[:length], -torch.inf)
    assert torch.equal(actual[0, :length], expected)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_live_wrapper_rejects_non_bf16_before_launch(dtype):
    from sparse_attn.ops.compact_normalization import compact_probabilities, supported
    scores = torch.ones(2, 4, device='cuda', dtype=dtype)
    ids = torch.arange(4, device='cuda', dtype=torch.int32)[None]
    keep = torch.ones_like(ids, dtype=torch.bool)
    assert not supported(scores, ids, keep, 16448, 2)
    with pytest.raises(ValueError, match='unsupported exact compact'):
        compact_probabilities(scores, ids, keep, 16448, 2,
                              valid_length=torch.tensor([16384], device='cuda'))
