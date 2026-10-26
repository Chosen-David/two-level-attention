"""GPU fusion gates. Interpreter covers logic only, never CUDA performance."""
import os
from types import SimpleNamespace
import pytest
import torch

INTERPRETER = os.environ.get('TRITON_INTERPRET') == '1'
gpu = pytest.mark.skipif(not INTERPRETER and not torch.cuda.is_available(),
                         reason='needs CUDA or explicitly enabled Triton interpreter')


def device(dtype=torch.float32):
    if INTERPRETER and dtype == torch.bfloat16:
        pytest.skip('BF16 requires CUDA execution')
    return 'cpu' if INTERPRETER else 'cuda'


@gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('layout', ['token', 'head', 'strided'])
@pytest.mark.parametrize('dc,df,bs', [(32,32,64), (32,128,64), (128,32,17)])
@torch.no_grad()
def test_direct_cache_update(dtype, layout, dc, df, bs):
    from sparse_attn.indexer import tli_batch_pipeline as kernels
    dev = device(dtype)
    torch.manual_seed(104)
    B,S,H,D = 5,137,2,128
    k = torch.randn(B,S,H,D,device=dev,dtype=dtype)
    if layout == 'head':
        k = k.transpose(1,2).contiguous().transpose(1,2)
    if layout == 'strided':
        k = torch.stack((k,k),-1)[...,0]
    lengths = torch.tensor([1,bs-1,bs,bs+1,S],device=dev)
    for b,l in enumerate(lengths.tolist()):
        k[b,l:] = float('nan')
    cd = torch.arange(96,128,device=dev) if dc == 32 else None
    fd = cd if dc == df else torch.arange(96,128,device=dev) if df == 32 else None
    P = (S+bs-1)//bs
    st = SimpleNamespace(B=B,H=H,P=P,bs=bs,coarse_dims=cd,fine_dims=fd,
        need_minmax=True,need_avg=True,previous=lengths.clone(),
        kmin=torch.full((B,P,H,dc),71.,device=dev,dtype=dtype),
        kmax=torch.full((B,P,H,dc),72.,device=dev,dtype=dtype),
        kavg=torch.full((B,P,H,dc),73.,device=dev,dtype=dtype),
        kfine=torch.full((B,H,S,df),74.,device=dev,dtype=dtype).transpose(1,2))
    # Independent scalar/request reference, includes untouched page/cache guards.
    expected = {n:getattr(st,n).clone() for n in ('kmin','kmax','kavg','kfine')}
    for b,l in enumerate(lengths.tolist()):
        p = (l-1)//bs
        x = torch.zeros(bs,H,dc,device=dev,dtype=dtype)
        live = k[b,p*bs:l]
        if cd is not None: live = live.index_select(-1,cd)
        x[:live.shape[0]] = live
        expected['kmin'][b,p] = x.amin(0)
        expected['kmax'][b,p] = x.amax(0)
        expected['kavg'][b,p] = x.mean(0)
        fine = k[b,l-1] if fd is None else k[b,l-1].index_select(-1,fd)
        expected['kfine'][b,l-1] = fine
    st.previous.sub_(1)
    kernels.update_cache(st,k,lengths)
    for name,value in expected.items():
        if name == 'kavg':
            torch.testing.assert_close(getattr(st,name),value,rtol=2e-3 if dtype!=torch.float32 else 2e-6,atol=2e-3 if dtype!=torch.float32 else 2e-7)
        else:
            assert torch.equal(getattr(st,name),value), name
    assert torch.equal(st.previous,lengths)


@gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['same', 'full', 'mixed'])
def test_query_projection(dtype, mode):
    from sparse_attn.indexer.tli_batch_pipeline import queries
    dev = device(dtype)
    q = torch.randn(3,1,8,256,device=dev,dtype=dtype)[...,::2]
    cd = None if mode=='full' else torch.arange(96,128,device=dev)
    fd = None if mode=='mixed' else cd
    qc,qf = queries(q,128**-.5,cd,fd,2)
    ref = (q[:,0]*128**-.5).float().reshape(3,2,4,128)
    assert torch.equal(qc,ref if cd is None else ref.index_select(-1,cd))
    assert torch.equal(qf,ref if fd is None else ref.index_select(-1,fd))
    assert (qc is qf) == (cd is fd)


@gpu
@pytest.mark.parametrize('nf,nn,bs', [(0,0,16),(0,4,16),(4,4,16),(32,32,64)])
@pytest.mark.parametrize('sink', [0,4,64])
def test_fused_candidates_and_top_mask(nf, nn, bs, sink):
    from sparse_attn.indexer.tli_batch_pipeline import candidates, mask_from_top
    from sparse_attn.indexer.tli_batch_kernels import assemble_mask
    dev = device()
    B,H,P,S = 5,2,160,10240
    swa,near_len,lo = 128,4096,sink//bs
    lengths = torch.tensor([1,bs,129,4227,S],device=dev)
    torch.manual_seed(89)
    fi = torch.stack([torch.randperm(P,device=dev)[:nf] for _ in range(B*H)]).view(B,H,nf)
    ni = torch.stack([torch.randperm(P,device=dev)[:nn] for _ in range(B*H)]).view(B,H,nn)
    ids,keep = candidates(fi,ni,lengths,bs,lo,sink,swa,near_len)
    # Scalar independent construction, preserving BOTH partition halves.
    for b,length in enumerate(lengths.tolist()):
        swalo = max(length-swa,0)
        near = max((length-swa-max(bs,near_len))//bs,lo)
        swablock = max(near,(length+bs-1)//bs-max(1,swa//bs))
        bypage = swalo//bs
        byok = swalo%bs!=0 and bypage>=lo
        for h in range(H):
            slots=[]
            for p in fi[b,h].tolist():
                ok = lo<=p<near and not (sink%bs and p==lo)
                slots.extend((p*bs+t,ok and p*bs+t<length) for t in range(bs))
            for p in ni[b,h].tolist():
                ok = near<=p<swablock and not (sink%bs and p==lo) and not(p==bypage and byok)
                slots.extend((p*bs+t,ok and p*bs+t<length) for t in range(bs))
            for p,ok in [(lo if sink%bs else 0,bool(sink%bs)),(bypage,byok and not(bypage==lo and sink%bs))]:
                slots.extend((p*bs+t,ok and p*bs+t<length) for t in range(bs))
            slots=sorted(slots,key=lambda x:not x[1])
            assert ids[b,h].tolist()==[x[0] for x in slots]
            assert keep[b,h].tolist()==[bool(x[1]) for x in slots]
    R=ids.shape[-1];ntop=min(1024,R);k2=1024
    top=torch.stack([torch.randperm(R,device=dev)[:ntop] for _ in range(B*H)]).view(B,H,ntop)
    selected=ids.gather(-1,top)
    swalo=(lengths-swa).clamp(min=0)
    valid=(keep & (ids>=lo*bs) & (ids<swalo[:,None,None])).gather(-1,top)
    valid &= torch.arange(ntop,device=dev)[None,None]<(k2-sink-(lengths-swalo)).clamp(min=0)[:,None,None]
    ref=assemble_mask(selected,valid,lengths,S,sink,swa)
    actual=mask_from_top(top,ids,keep,lengths,S,lo*bs,sink,swa,k2)
    assert torch.equal(actual,ref)


@gpu
@pytest.mark.parametrize('skip', [False,True])
@pytest.mark.parametrize('batch', [1,2])
@pytest.mark.parametrize('method', ['avg', 'minmax', 'mixed'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16])
@torch.no_grad()
def test_pipeline_route_and_steps(monkeypatch,method,dtype,skip,batch):
    from test_tli_batch_parallel import make_indexer
    from sparse_attn.indexer.tli_batch import prepare_mask_batched
    from sparse_attn.indexer import tli_batch_pipeline, tli_batch_kernels
    dev=device(dtype)
    monkeypatch.setenv('TLI_BATCH_SCORE','torch' if INTERPRETER else 'triton')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','0' if INTERPRETER else '1')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','0')
    torch.manual_seed(317)
    kw=dict(tli_dyn_dgate=batch==1,tli_far_method='minmax' if method=='mixed' else method,
            tli_near_method='avg' if method=='mixed' else method)
    ref, cand=make_indexer(**kw),make_indexer(**kw)
    ref.skip_far=cand.skip_far=skip
    if batch == 1:
        for idx in (ref,cand):
            idx._dyn_skip_pending=skip
            idx.freeze_dyn_dgate()
    k=torch.randn(batch,2,512,128,device=dev,dtype=dtype).transpose(1,2)
    q=torch.randn(batch,1,8,128,device=dev,dtype=dtype)
    lengths=torch.tensor([287,399][:batch],device=dev)
    def run(idx):return prepare_mask_batched(idx,q,k,lengths,128**-.5)[0]
    run(ref)
    if INTERPRETER:
        run(cand)
        cand._batch_tli.gpu_pipeline=tli_batch_pipeline
        cand._batch_tli.triton_ops=tli_batch_kernels
        cand._batch_tli.triton_score=tli_batch_kernels.score
    else:
        monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','1')
        run(cand)
    assert cand._batch_tli.gpu_pipeline is tli_batch_pipeline
    funcs=[lambda:run(ref),lambda:run(cand)]
    if not INTERPRETER:
        # CUDA-only capture/replay with device lengths changing across a page.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for fn in funcs:fn();fn()
        torch.cuda.current_stream().wait_stream(stream)
        graphs=[];buffers=[]
        for fn in funcs:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            graphs.append(graph);buffers.append(out)
        funcs += [lambda j=j:(graphs[j].replay(),buffers[j])[1] for j in range(2)]
    for _ in range(4):
        outputs=[fn() for fn in funcs]
        for output in outputs[1:]: assert torch.equal(output,outputs[0])
        for name in ('kmin','kmax','kfine'):
            x,y=getattr(ref._batch_tli,name),getattr(cand._batch_tli,name)
            assert (x is None and y is None) or torch.equal(x,y)
        k[torch.arange(batch,device=dev),lengths]=torch.randn(batch,2,128,device=dev,dtype=dtype)
        lengths.add_(1);q.normal_()


def test_pipeline_rejects_wrong_config(monkeypatch):
    from test_tli_batch_parallel import make_indexer
    from sparse_attn.indexer.tli_batch import BatchedTLI
    monkeypatch.setenv('TLI_BATCH_SCORE','triton')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','1')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','0')
    k=torch.empty(2,64,2,128);lengths=torch.tensor([32,32])
    with pytest.raises(ValueError,match='requires TLI_BATCH_FUSIONS'):
        BatchedTLI(make_indexer(),k,lengths)
    monkeypatch.setenv('TLI_BATCH_FUSIONS','1')
    with pytest.raises(ValueError,match='l2_quant=False'):
        BatchedTLI(make_indexer(tli_l2_quant=True),k,lengths)


def test_skip_change_requires_clear(monkeypatch):
    from test_tli_batch_parallel import make_indexer
    from sparse_attn.indexer.tli_batch import prepare_mask_batched
    monkeypatch.setenv('TLI_BATCH_SCORE','torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','0')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','0')
    idx=make_indexer();q=torch.randn(1,1,8,128);k=torch.randn(1,512,2,128)
    lengths=torch.tensor([300])
    prepare_mask_batched(idx,q,k,lengths,128**-.5)
    idx.skip_far=True
    with pytest.raises(ValueError,match='skip mode changed'):
        prepare_mask_batched(idx,q,k,lengths,128**-.5)
    idx.clear()
    prepare_mask_batched(idx,q,k,lengths,128**-.5)
    assert idx._batch_tli.skip_far is True


@pytest.mark.parametrize('backend',['legacy','compact'])
def test_frozen_gate_full_kv_admission(monkeypatch,backend):
    from test_tli_batch_parallel import make_indexer
    from sparse_attn.indexer.tli_batch import prepare_mask_batched
    monkeypatch.setenv('TLI_BATCH_SCORE','torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS','0')
    monkeypatch.setenv('TLI_BATCH_GPU_PIPELINE','0')
    idx=make_indexer(tli_dyn_dgate=True,tli_far_skip_backend=backend,
                     tli_far_skip_cache='evict' if backend=='compact' else 'retain')
    q=torch.randn(1,1,8,128);k=torch.randn(1,512,2,128);lengths=torch.tensor([300])
    with pytest.raises(ValueError,match='frozen B=1'):
        prepare_mask_batched(idx,q,k,lengths,128**-.5)
    idx._dyn_skip_pending=False;idx.freeze_dyn_dgate()
    prepare_mask_batched(idx,q,k,lengths,128**-.5)
    idx.clear();idx._dyn_skip_pending=True;idx.freeze_dyn_dgate()
    if backend=='compact':
        with pytest.raises(ValueError,match='evicted KV'):
            prepare_mask_batched(idx,q,k,lengths,128**-.5)
    else:
        prepare_mask_batched(idx,q,k,lengths,128**-.5)
    with pytest.raises(ValueError,match='frozen B=1'):
        prepare_mask_batched(idx,q.expand(2,-1,-1,-1),k.expand(2,-1,-1,-1),lengths.expand(2),128**-.5)


@gpu
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('g',[1,4])
@pytest.mark.parametrize('method',['avg','minmax'])
@pytest.mark.parametrize('skip',[False,True])
def test_pool_score_regions_and_zero_far_launch(monkeypatch,dtype,g,method,skip):
    from sparse_attn.indexer import tli_batch_pipeline as kernels
    dev=device(dtype)
    torch.manual_seed(971)
    B,P,H,D,bs=3,128,2,32,64
    q=torch.randn(B,H,g,D,device=dev)
    low=torch.randn(B,P,H,D,device=dev,dtype=dtype)
    high=low+2
    avg=low+.5
    lengths=torch.tensor([1,4231,8001],device=dev)
    idx=SimpleNamespace(far_method=method,near_method=method,_sink_pool_lo_blk=0,
                        sink_tokens=4,sliding_window_size=128,nt_near_len=4096)
    st=SimpleNamespace(B=B,H=H,P=P,bs=bs,indexer=idx,kmin=low,kmax=high,kavg=avg)
    n=torch.arange(P,device=dev)
    near=((lengths-128-4096)//bs).clamp(min=0)
    end=torch.maximum(near,(lengths+bs-1)//bs-2)
    swalo=(lengths-128).clamp(min=0);by=swalo//bs;byok=swalo%bs!=0
    fp=(n[None]<near[:,None]) & (n[None]!=0)
    np=(n[None]>=near[:,None]) & (n[None]<end[:,None]) & (n[None]!=0)
    np &= ~((n[None]==by[:,None]) & byok[:,None])
    # Poison inactive pages: merely masking after reading/scoring is caught by
    # the output checks. Static source + branch establishes skipped dot tiles;
    # real memory transactions still require Nsight, not this poison test.
    live=np if skip else fp|np
    low.masked_fill_(~live[:,:,None,None],float('nan'))
    high.masked_fill_(~live[:,:,None,None],float('nan'))
    avg.masked_fill_(~live[:,:,None,None],float('nan'))
    ref=(torch.einsum('bhgd,bphd->bhgp',q.clamp(max=0),low.float())+
         torch.einsum('bhgd,bphd->bhgp',q.clamp(min=0),high.float()) if method=='minmax' else
         torch.einsum('bhgd,bphd->bhgp',q,avg.float())).to(dtype).mean(2)
    original=kernels._pool_score;calls=[];outputs=[]
    class Spy:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                original[grid](*args,**kwargs)
                calls.append(args[-3]);outputs.append(args[4].clone())
            return launch
    monkeypatch.setattr(kernels,'_pool_score',Spy())
    fi,ni=kernels.coarse_indices(st,q,lengths,dtype,0 if skip else 32,32)
    assert calls==([1] if skip else [0,1])
    assert fi.shape[-1]==(0 if skip else 32)
    for region,out in zip(calls,outputs):
        valid=(fp if region==0 else np)[:,None].expand(B,H,P)
        assert torch.isneginf(out[~valid]).all()
        torch.testing.assert_close(out[valid],ref[valid],rtol=3e-3 if dtype!=torch.float32 else 3e-5,
                                   atol=3e-3 if dtype!=torch.float32 else 3e-6)
    calls.clear()
    fi,ni=kernels.coarse_indices(st,q,lengths,dtype,0,0)
    assert calls==[] and fi.numel()==ni.numel()==0
