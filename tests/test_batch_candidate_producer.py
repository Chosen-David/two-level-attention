"""CPU integer/order contract vs actual primary single-request producer."""
import importlib.util
from pathlib import Path
import pytest
import torch
from sparse_attn.ops import batch_candidate_producer as producer


@pytest.fixture
def primary(monkeypatch):
    from sparse_attn.indexer import tli_indexer as mod
    spec=importlib.util.spec_from_file_location('producer_args',Path(__file__).parents[1]/'exp/bench/test_tli_graph_gates.py')
    args=importlib.util.module_from_spec(spec);spec.loader.exec_module(args)
    monkeypatch.setattr(mod,'_HAS_TRITON',False)
    monkeypatch.setenv('TLI_L2_BYPASS','0');monkeypatch.setenv('TLI_SPARSE_SELECTION','0')
    monkeypatch.setenv('TLI_COMPACT_NORMALIZATION','0')
    before=torch.get_num_threads();torch.set_num_threads(1)
    def make(sink,swa,near,k1,nn):
        i=mod.TLIIndexer(args._make_prod_args());i.layer_idx=2;i.clear()
        i.sink_tokens=sink;i.sliding_window_size=swa;i.nt_near_len=near
        i.nt_near_pages=nn;i.args.tia_level1_topk=k1
        return i
    yield mod,make
    torch.set_num_threads(before)


@pytest.mark.parametrize('batch,capacity',[(2,128),(4,1024)])
@pytest.mark.parametrize('settings',[(4,128,4096,128,32),(4,0,64,5,2),
                                    (64,1,128,0,0),(0,128,64,5,0)])
@pytest.mark.parametrize('kind',['random','ties','nonfinite'])
@pytest.mark.parametrize('methods',[('minmax','avg'),('avg','minmax'),('avg','avg'),('minmax','minmax')])
def test_batch_producer_matches_row_candidates(primary,monkeypatch,batch,capacity,settings,kind,methods):
    mod,make=primary;sink,swa,near,k1,nn=settings
    torch.manual_seed(320303);p=(capacity+63)//64;h,g,d=2,4,128
    lengths=torch.tensor(([1,33,65,capacity-7]*2)[:batch])
    q=torch.randn(batch,1,h*g,d,dtype=torch.bfloat16)
    keys=torch.randn(batch,capacity,h,d,dtype=q.dtype)
    raw=torch.randn(batch,h*g,p,dtype=q.dtype)
    avg=torch.randn_like(raw)
    if kind=='ties':raw.zero_();avg.zero_()
    if kind=='nonfinite':
        raw[...,0]=torch.nan;raw[...,-1]=torch.inf
        avg[...,0]=-torch.inf;avg[...,-1]=torch.nan
    indices=[];expected=[];seen_coarse=[];seen_fine=[];owners=[]
    for b in range(batch):
        idx=make(*settings);idx.far_method,idx.near_method=methods;owners.append(idx)
        def coarse(kmin,kmax,query,out_dtype,out=None,_b=b,**kw):
            seen_coarse.append(query.clone());out.copy_(raw[_b]);return out
        def average(kavg,query,out_dtype,out=None,_b=b,**kw):
            out.copy_(avg[_b]);return out
        monkeypatch.setattr(idx,'_fast_score_coarse',coarse)
        monkeypatch.setattr(idx,'_fast_score_fine',average)
        index=idx.prepare_index(keys[b:b+1],torch.tensor([0,capacity],dtype=torch.int32))
        indices.append(index)
        expected.append(idx.compute_candidate_inputs_graph(q[b:b+1],index,128**-.5,lengths[b:b+1]))
        seen_fine.append(expected[-1].q_fine)
    # Test-only full stacks establish the arena inputs; production binds them once.
    summaries={name:(None if indices[0].get(name) is None else
                     torch.cat([x[name] for x in indices],dim=0))
               for name in ('k_min','k_max','k_avg')}
    arena=torch.cat([x['k_qat'] for x in indices],dim=0)
    calls=[]
    def batch_coarse(kmin,kmax,query,**kw):
        assert torch.equal(query,torch.stack(seen_coarse))
        assert not kw.get('return_mean',False)
        calls.append('coarse');return raw
    def batch_average(kavg,query,**kw):
        calls.append('average');return avg
    monkeypatch.setattr(producer,'score_coarse_batched',batch_coarse)
    monkeypatch.setattr(producer,'score_average_batched',batch_average)
    front=producer.BatchedCandidateProducer(owners[0],keys)
    actual=front(q,arena,summaries,lengths,128**-.5)
    assert calls==(['coarse','average'] if summaries['k_avg'] is not None else ['coarse'])
    assert actual.k_fine is arena
    for b,reference in enumerate(expected):
        for name in ('ids','keep','counts','q_fine','lower','upper','quota'):
            torch.testing.assert_close(getattr(actual,name)[b],getattr(reference,name),rtol=0,atol=0)
        active=actual.ids[b][actual.keep[b]]
        assert bool(((active>=0)&(active<lengths[b])).all())
        for head in range(h):
            active=actual.ids[b,head,actual.keep[b,head]]
            assert active.numel()==active.unique().numel()
        assert torch.equal(actual.keep[b],torch.arange(actual.ids.shape[-1])[None]<actual.counts[b,:,None])
    assert torch.equal(actual.q_fine,torch.stack(seen_fine))


@pytest.mark.parametrize('sink,swa,near',[(4,128,4096),(4,0,64),(64,1,128),(0,128,64)])
def test_batch_geometry_matches_primary_scalars(primary,sink,swa,near):
    mod,make=primary;i=make(sink,swa,near,128,32)
    lengths=torch.tensor([0,1,33,63,64,65,132,4097,32767])
    pages=torch.arange(512);offsets=torch.arange(64)
    actual=producer.candidate_geometry(lengths,pages,offsets,block_size=64,sink=sink,swa=swa,near_len=near,k2=1024)
    cst={'ar_blk':pages,'ar_tok':None,'ar_bs':offsets,'zero_l':torch.tensor(0),
         'sink_cross':torch.tensor(sink%64!=0),'far_lo':torch.tensor(sink//64)}
    for b in range(len(lengths)):
        ref=i._build_graph_geometry(lengths[b:b+1],32768,512,cst,compact_only=True)
        assert torch.equal(actual.far_pool[b],ref['far_pool_mask'])
        assert torch.equal(actual.near_pool[b],ref['near_pool_mask'])
        assert torch.equal(actual.bypass_ids[b],ref['byp_tok'].reshape(2,64))
        assert torch.equal(actual.bypass_keep[b],ref['byp_keep'].reshape(2,64))
        assert actual.lower[b]==ref['far_tok_lo'] and actual.upper[b]==max(ref['far_tok_lo'],int(ref['swa_lo_tok']))
        assert actual.quota[b]==ref['k2_mid']


def test_geometry_reuse_obeys_mutation_version_and_namespaces(primary):
    _,make=primary;i=make(4,128,4096,128,32)
    front=producer.BatchedCandidateProducer(i,torch.empty(2,32768,2,128))
    lengths=torch.tensor([32000,31000]);a=front.geometry(lengths)
    assert front.geometry(lengths) is a
    lengths.add_(1);b=front.geometry(lengths)
    assert b is not a
    with torch.inference_mode():
        other=torch.tensor([30000,29000])
        assert front.geometry(other) is not front.geometry(other)
