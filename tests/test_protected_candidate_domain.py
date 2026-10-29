"""Full protected denominator regression; independent dense-domain oracle."""
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest
import torch
from sparse_attn.indexer.candidate_domain import complete_protected_candidates
from sparse_attn.indexer import tli_indexer as mod


@pytest.mark.parametrize('length,sink,swa', [(1,4,8),(5,4,8),(12,4,8),(19,4,8),(64,0,0),(65,4,0),(129,64,64)])
def test_protected_union_counts_once(length,sink,swa):
    ids=torch.arange(160).reshape(1,-1)
    keep=(ids % 3 == 0) & (ids < length)
    out,valid=complete_protected_candidates(ids,keep,torch.tensor(length),sink,swa,160)
    actual=out[valid].tolist()
    expected={i for i in range(length) if i%3==0 or i<sink or i>=max(0,length-swa)}
    assert len(actual)==len(set(actual))
    assert set(actual)==expected


def make_indexer():
    helpers=runpy.run_path(str(Path(__file__).parents[1]/'exp/bench/test_glm47_d256_gates.py'))
    args=helpers['make_prod_args'](128)
    args.tia_level1_topk=4;args.tli_near_pages=2;args.tli_near_len=128
    args.tia_level2_topk=192;args.tli_sink_tokens=4;args.tli_swa_size=128
    return helpers['make_indexer'](128,args)


@pytest.mark.parametrize('backend',['full','compact_torch','compact_triton','graph'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_indexer_softmax_contains_full_protected_domain(backend,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    if backend=='compact_triton' and device=='cpu':
        pytest.skip('Triton requires CUDA')
    torch.manual_seed(731)
    idx=make_indexer()
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    length=1025
    k=torch.randn(1,length,2,128,device=device,dtype=dtype)
    q=torch.randn(1,1,4,128,device=device,dtype=dtype)
    cu=torch.tensor([0,length],device=device,dtype=torch.int32)
    pos=torch.tensor([length-1],device=device)
    captured=[]
    softmax=mod.F.softmax
    def record(x,*a,**kw):
        captured.append(x.detach().clone())
        return softmax(x,*a,**kw)
    # Disable Triton only for the CPU and deliberate Torch fallback cases.
    use_triton=device=='cuda' and backend in ('compact_triton','graph')
    with patch.object(mod,'_HAS_TRITON',use_triton), patch.object(mod.F,'softmax',record):
        if backend=='graph':
            index=idx.prepare_index(k,cu)
            idx.compute_mask_graph(q,index,128**-.5,torch.tensor([length],device=device))
        else:
            idx._fine_gather=backend!='full'
            index=idx.prepare_index(k,cu)
            scores=idx.compute_score(q,pos,index,128**-.5)
            if backend.startswith('compact'):
                assert scores.get('cand_idx') is not None
            idx.compute_mask(pos,scores)
    scores=captured[-1].reshape(4,-1)
    expected_ids=list(range(4))+list(range(length-128,length))
    assert torch.isfinite(scores[:,expected_ids]).all()
    # Independent score oracle includes query dtype rounding before FP32 dot,
    # and casts logits back to query dtype before the full-domain softmax.
    dims=idx._subspace_indices(k.device)
    qf=(q[0,0]*128**-.5).float().index_select(-1,dims)
    kf=k[0].float().index_select(-1,dims).repeat_interleave(2,dim=1)
    expected=torch.einsum('hd,thd->ht',qf,kf).to(dtype)
    torch.testing.assert_close(scores[:,expected_ids],expected[:,expected_ids],rtol=0.02 if device=='cuda' else 1e-5,atol=0.02 if device=='cuda' else 1e-5)


def test_protected_gqa_denominator_can_change_winner():
    logits=torch.tensor([[20.,4.,0.],[0.,0.,3.]])
    complete=logits.softmax(-1).mean(0)[1:]
    filtered=logits[:,1:].softmax(-1).mean(0)
    assert filtered.argmax().item()==0
    assert complete.argmax().item()==1
