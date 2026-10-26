"""Batch candidate gates. CUDA execution is explicitly skipped without a GPU."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.indexer.tli_batch import prepare_mask_batched
from sparse_attn.metrics import Metrics


def make_indexer(**kwargs):
    args = dict(tia_block_size=16, tia_level1_topk=8, tia_level2_topk=64,
                tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
                tli_enable_layer_skip=False, tli_sink_tokens=4, tli_swa_size=32,
                tli_near_pages=4, tli_near_tokens='off', tli_near_len=128,
                tli_subspace='tail')
    args.update(kwargs)
    idx = TLIIndexer(SimpleNamespace(**args))
    idx.metrics = Metrics(threshold=0)
    return idx


def test_merged_full_statistics_uses_local_subspace_coordinates(monkeypatch):
    """The upstream full-stat kernel receives already-gathered tail features."""
    import sparse_attn.indexer.tli_indexer as module
    idx = make_indexer()
    assert idx._inc is None and idx._triton_ok
    assert idx._km_centroids is None and idx._km_dims is None
    k = torch.arange(128., dtype=torch.float32).expand(1,32,2,128).clone()
    seen = []
    def stats(features, dims, need_avg):
        seen.append((features.shape, dims))
        assert dims is None, 'original channel indices cannot index compact features'
        assert torch.equal(features[0,0,0], torch.cat((torch.arange(48.,64.), torch.arange(112.,128.))))
        blocks = features.reshape(1,2,16,2,32)
        return blocks.amin(2), blocks.amax(2), blocks.mean(2) if need_avg else None
    monkeypatch.setattr(module, '_HAS_TRITON', True)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(idx, '_fast_stat_full', stats)
    with patch('torch.cuda.is_current_stream_capturing', return_value=False):
        result = idx._prepare_index_base(k, torch.tensor([0,32], dtype=torch.int32))
    assert seen == [(torch.Size([1,32,2,32]), None)]
    assert torch.equal(result['k_min'], result['k_max'])


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv('TLI_BATCH_SCORE', 'torch')
    monkeypatch.setenv('TLI_BATCH_FUSIONS', '0')
    yield
    torch.set_num_threads(n)


def graph_oracle(idx, q, k, lengths):
    out = []
    with patch('torch.cuda.is_current_stream_capturing', return_value=False):
        for b, length in enumerate(lengths.tolist()):
            kk = k[b:b+1].clone()
            kk[:,length:] = 0
            # Independent original preparation and original graph selector.
            d = idx._prepare_index_base(kk, torch.tensor([0,k.shape[1]],device=k.device,dtype=torch.int32))
            out.append(idx.compute_mask_graph(q[b:b+1], d, 128**-.5, lengths[b:b+1]))
    return torch.cat(out)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('method', ['avg', 'minmax'])
@pytest.mark.parametrize('quant', [False, True])
@torch.no_grad()
def test_original_graph_mask_oracle(dtype, method, quant):
    torch.manual_seed(814)
    idx = make_indexer(tli_far_method=method, tli_near_method=method, tli_l2_quant=quant)
    assert idx.l2_quant == quant
    k = torch.randn(3,512,2,128,dtype=dtype)
    lengths = torch.tensor([289,319,400])
    for b,l in enumerate(lengths):
        k[b,l:] = float('nan')
    for step in range(4):
        q = torch.randn(3,1,8,128,dtype=dtype)
        actual,_ = prepare_mask_batched(idx,q,k,lengths,128**-.5)
        expected = graph_oracle(make_indexer(tli_far_method=method,tli_near_method=method,tli_l2_quant=quant),q,k,lengths)
        assert torch.equal(actual,expected), (actual != expected).sum().item()
        assert not (actual & (torch.arange(512)[None,None,None] >= lengths[:,None,None,None])).any()
        rows=torch.arange(3)
        k[rows,lengths]=torch.randn(3,2,128,dtype=dtype)
        lengths+=1


@pytest.mark.parametrize('skip', [False,True])
@pytest.mark.parametrize('lengths', [[1,3,15,16], [31,32,33,66], [290,317,388,400]])
@torch.no_grad()
def test_batch_independence_incremental_and_refill(skip,lengths):
    torch.manual_seed(803)
    idx=make_indexer(); idx.skip_far=skip
    k=torch.randn(4,512,2,128); lengths=torch.tensor(lengths)
    for step in range(4):
        q=torch.randn(4,1,8,128)
        m,_=prepare_mask_batched(idx,q,k,lengths,128**-.5)
        for b in range(4):
            single=make_indexer(); single.skip_far=skip
            ref,_=prepare_mask_batched(single,q[b:b+1],k[b:b+1],lengths[b:b+1],128**-.5)
            assert torch.equal(m[b:b+1],ref)
        assert not (m & (torch.arange(512)[None,None,None] >= lengths[:,None,None,None])).any()
        # Finished row remains at its length, other rows advance.
        k[torch.arange(1,4),lengths[1:]]=torch.randn(3,2,128)
        lengths[1:]+=1
    idx.clear()
    assert idx._batch_tli is None
    new=make_indexer(); new.skip_far=skip
    expected,_=prepare_mask_batched(new,q,k,lengths,128**-.5)
    actual,_=prepare_mask_batched(idx,q,k,lengths,128**-.5)
    assert torch.equal(actual,expected)


@torch.no_grad()
def test_metadata_rejections_and_metrics():
    idx=make_indexer(); k=torch.randn(2,512,2,128); q=torch.randn(2,1,8,128)
    lengths=torch.tensor([290,310])
    prepare_mask_batched(idx,q,k,lengths,128**-.5)
    for bad in (lengths-1,lengths+2,torch.tensor([0,310]),torch.tensor([290,513])):
        with pytest.raises(RuntimeError):
            prepare_mask_batched(idx,q,k,bad,128**-.5)
    with pytest.raises(ValueError,match='storage changed'):
        prepare_mask_batched(idx,q,k.clone(),lengths,128**-.5)
    idx.per_q_head=True
    with pytest.raises(ValueError,match='requires'):
        prepare_mask_batched(idx,q,k,lengths,128**-.5)
    metrics=Metrics(threshold=3); ref=Metrics(threshold=3)
    mask=torch.rand(4,1,2,20)>.5; lengths=torch.tensor([1,3,4,5])
    metrics.add_select_result_batch(lengths,mask,1)
    for b in range(4):
        ref.add_select_result(lengths[b:b+1]-1,mask[b:b+1],1)
    assert metrics.get_select_tokens()==ref.get_select_tokens()


@pytest.mark.parametrize('backend,fusions',[('torch',False),('triton',False),('triton',True)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA capture')
@torch.no_grad()
def test_real_capture_replay(monkeypatch,backend,dtype,fusions):
    monkeypatch.setenv('TLI_BATCH_SCORE',backend)
    monkeypatch.setenv('TLI_BATCH_FUSIONS','1' if fusions else '0')
    idx=make_indexer(); k=torch.randn(4,512,2,128,device='cuda',dtype=dtype)
    q=torch.randn(4,1,8,128,device='cuda',dtype=dtype); lengths=torch.tensor([290,315,380,400],device='cuda')
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            prepare_mask_batched(idx,q,k,lengths,128**-.5)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out,_=prepare_mask_batched(idx,q,k,lengths,128**-.5)
    for _ in range(20):
        k[torch.arange(4,device='cuda'),lengths]=torch.randn(4,2,128,device='cuda',dtype=dtype)
        lengths.add_(1); q.normal_(); graph.replay()
        ref,_=prepare_mask_batched(make_indexer(),q,k,lengths,128**-.5)
        assert torch.equal(out,ref)

@pytest.mark.parametrize('graph',[False,True])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('skip',[False,True])
@torch.no_grad()
def test_qwen_routes_whole_batch_once(monkeypatch,graph,batch,skip):
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched
    from sparse_attn.ops.cuda_graph_state import graph_state,disable_graph
    from sparse_attn.baseline_runtime import BatchCtx
    monkeypatch.setenv('TLI_BATCH_INDEXER','parallel')
    idx=make_indexer(); idx.skip_far=skip; q=torch.randn(batch,8,1,128)
    k=torch.randn(batch,2,512,128); v=torch.randn_like(k)
    ctx=BatchCtx(batch=batch,seq_lens=[289,319,400][:batch],gen_lens=[0]*batch,alive=[True]*batch,pad_id=0)
    gs=graph_state(); gs.enabled=graph; gs.valid_len_buf=torch.tensor(ctx.seq_lens)
    calls=[]
    def attention(q,k,v,mask,bs,cu,**kwargs):
        calls.append(mask.clone())
        return q
    try:
        with patch.object(idx,'prepare_mask',side_effect=AssertionError('serial indexer called')), \
             patch('sparse_attn.patches.qwen3_attn_patch.eager_decoding_attn',side_effect=attention):
            _tl_decode_batched(SimpleNamespace(scaling=128**-.5),idx,q,k,v,ctx)
        reference=make_indexer();reference.skip_far=skip
        expected,_=prepare_mask_batched(reference,q.transpose(1,2),k.transpose(1,2),torch.tensor(ctx.seq_lens),128**-.5)
        assert len(calls)==1 and torch.equal(calls[0],expected)
    finally:
        disable_graph()

@torch.no_grad()
def test_actual_eager_attention_consumer(monkeypatch):
    # CPU oracle must use the real Torch path even if Triton is installed.
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON', False)
    from sparse_attn.patches.qwen3_attn_patch import _tl_decode_batched
    from sparse_attn.baseline_runtime import BatchCtx
    torch.manual_seed(881)
    q=torch.randn(3,8,1,128); k=torch.randn(3,2,512,128); v=torch.randn_like(k)
    ctx=BatchCtx(batch=3,seq_lens=[289,319,400],gen_lens=[0]*3,alive=[True]*3,pad_id=0)
    outputs=[]
    with patch('torch.cuda.is_current_stream_capturing',return_value=False):
        for mode in ('legacy','parallel'):
            monkeypatch.setenv('TLI_BATCH_INDEXER',mode)
            outputs.append(_tl_decode_batched(SimpleNamespace(scaling=128**-.5),make_indexer(),q,k,v,ctx))
    torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-5,atol=1e-6)

@torch.no_grad()
def test_short_sequences_forced_dense_and_strided_subspace():
    idx=make_indexer(); k=torch.randn(4,2,77,128).transpose(1,2)
    lengths=torch.tensor([1,3,16,32]);q=torch.randn(4,1,8,128)
    m,_=prepare_mask_batched(idx,q,k,lengths,128**-.5)
    expected=(torch.arange(77)[None,None,None] < lengths[:,None,None,None]).expand(4,1,2,77)
    assert torch.equal(m,expected)


def test_length_upload_shared_across_layers_and_steps():
    from sparse_attn.patches.qwen3_attn_patch import _batch_device_lengths
    ctx=SimpleNamespace(seq_lens=[17,31])
    old=_batch_device_lengths(ctx,torch.device('cpu'))
    assert _batch_device_lengths(ctx,torch.device('cpu')) is old
    ctx.seq_lens[0]+=1
    new=_batch_device_lengths(ctx,torch.device('cpu'))
    assert new is not old and new.tolist()==[18,31] and old.tolist()==[17,31]

@pytest.mark.parametrize('subspace,l2',[('full','same'),('tail','full'),('rope','same'),('nope','full')])
@torch.no_grad()
def test_subspace_modes_against_original_graph(subspace,l2):
    torch.manual_seed(889)
    opts=dict(tli_subspace=subspace,tli_l2_subspace=l2,tli_far_method='minmax',tli_near_method='avg')
    k=torch.randn(2,512,2,128);q=torch.randn(2,1,8,128);lengths=torch.tensor([290,400])
    actual,_=make_indexer(**opts).prepare_mask_batched(q,k,lengths)
    expected=graph_oracle(make_indexer(**opts),q,k,lengths)
    assert torch.equal(actual,expected)
