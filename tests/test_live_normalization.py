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
