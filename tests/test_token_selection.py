import argparse
import os
import pytest
import torch
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.token_selection import canonical_selection, selected_attention


@pytest.mark.parametrize('length', [1, 7, 17, 65])
def test_union_matches_boolean_oracle(length):
    torch.manual_seed(length)
    ids = torch.randint(-3, length+3, (3, 2, 80))
    valid = torch.rand(ids.shape) > .2
    selected = canonical_selection(ids, valid, length)
    expected = torch.zeros(3, 2, length, dtype=torch.bool)
    for b in range(3):
        for h in range(2):
            for i, ok in zip(ids[b, h].tolist(), valid[b, h].tolist()):
                if ok and 0 <= i < length:
                    expected[b, h, i] = True
    assert torch.equal(selected.to_mask(length)[:, 0], expected)
    for b in range(3):
        for h in range(2):
            assert selected.indices[b,h,:selected.counts[b,h]].tolist() == expected[b,h].nonzero().flatten().tolist()


@pytest.mark.parametrize('capacity,budget', [(17,32), (65,10), (257,32)])
def test_real_graph_selector_same_set_as_existing_mask(capacity, budget, monkeypatch):
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON', False)
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    args = parser.parse_args(['--method','tli','--tia_block_size','4',
        '--tli_near_len','8','--tli_near_pages','1','--tli_near_tokens','off',
        '--tli_sink_tokens','2','--tli_swa_size','4', '--tli_subspace','tail',
        '--tia_level1_topk','4','--tia_level2_topk',str(budget)])
    idx = TLIIndexer(args)
    idx._skip_ids = set()
    q, k = torch.randn(1,1,4,128), torch.randn(1,capacity,2,128)
    index = idx.prepare_index(k, torch.tensor([0,capacity]))
    for n in [1,2,3,4,7,min(16,capacity),capacity]:
        length = torch.tensor([n])
        a = idx.compute_mask_graph(q,index,.1,length)
        b = idx.compute_mask_graph(q,index,.1,length,return_selection=True)
        assert torch.equal(a,b.to_mask(capacity)), (capacity,budget,n)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_attention_matches_independent_dense_mask(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(29)
    q = torch.randn(2,1,4,32,device=device)
    k, v = [torch.randn(2,63,2,32,device=device) for _ in range(2)]
    ids = torch.randint(0,63,(2,2,23),device=device)
    selection = canonical_selection(ids,torch.ones_like(ids,dtype=torch.bool),63)
    actual = selected_attention(q,k,v,selection,32**-.5)
    dense_k = k.transpose(1,2).repeat_interleave(2,dim=1)
    dense_v = v.transpose(1,2).repeat_interleave(2,dim=1)
    mask = selection.to_mask(63)[:,0].repeat_interleave(2,dim=1)[:,:,None]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2),dense_k,dense_v,attn_mask=mask).transpose(1,2)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('dim', [16, 128])
def test_direct_existing_kernel_matches_dense_oracle(dim):
    interpreter = os.environ.get('TRITON_INTERPRET') == '1'
    if not interpreter and not torch.cuda.is_available():
        pytest.skip('CUDA required (or explicit Triton CPU interpreter)')
    pytest.importorskip('triton')
    from sparse_attn.ops.eager_decoding import _eager_attn_kernel_b
    from sparse_attn.ops.token_selection import TokenSelection
    dev = 'cpu' if interpreter else 'cuda'
    torch.manual_seed(319)
    q = torch.randn(2, 4, dim, device=dev)
    k, v = [torch.randn(2, 301, 2, dim, device=dev) for _ in range(2)]
    ids = torch.arange(277, dtype=torch.int32, device=dev).expand(2,2,-1).contiguous()
    counts = torch.tensor([[0,277], [17,265]], device=dev, dtype=torch.int32)
    selection = TokenSelection(ids, counts)
    out = torch.empty_like(q)
    qs = (q*dim**-.5).float().contiguous()
    _eager_attn_kernel_b[(8,)](ids,counts,qs,k,v,out,2,
        qs.stride(0),k.stride(0),k.stride(1),k.stride(2),qs.stride(1),out.stride(0),
        G=2,D=dim,PER_QH=False,BW=256,W=277,num_warps=4)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(2,2,2,dim), k.transpose(1,2), v.transpose(1,2),
        attn_mask=selection.to_mask(301)[:,0,:,None]).reshape_as(q)
    torch.testing.assert_close(out,expected,rtol=3e-5,atol=3e-6)
