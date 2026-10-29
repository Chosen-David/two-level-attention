import pytest
import torch
from sparse_attn.ops.kv_row_update import append_kv_batch

@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_ragged_writes(batch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(9)
    # Noncontiguous batch/head/time strides, contiguous feature dimension.
    k=torch.randn(batch,3,34,40,device=device)[:,:,::2,:32]
    v=torch.randn_like(k);nk=torch.randn(batch,3,1,32,device=device);nv=torch.randn_like(nk)
    slots=torch.tensor([(b*7)%17 for b in range(batch)],device=device)
    alive=torch.tensor([b%3!=0 for b in range(batch)],device=device)
    expected_k,expected_v=k.clone(),v.clone()
    for b in range(batch):
        if alive[b]:
            expected_k[b,:,slots[b]:slots[b]+1]=nk[b]
            expected_v[b,:,slots[b]:slots[b]+1]=nv[b]
    append_kv_batch(k,v,nk,nv,slots,alive)
    torch.testing.assert_close(k,expected_k,rtol=0,atol=0)
    torch.testing.assert_close(v,expected_v,rtol=0,atol=0)

def test_invalid_metadata_does_not_mutate():
    k=torch.zeros(2,1,8,16);v=k.clone();nk=torch.ones(2,1,1,16)
    with pytest.raises(ValueError):append_kv_batch(k,v,nk,nk,torch.tensor([1]),torch.tensor([True,False]))
    assert not k.any() and not v.any()


def test_static_layer_dispatch_preserves_legacy_and_dead_rows(monkeypatch):
    from sparse_attn.baseline_runtime import BatchCtx,batch_decode_scope
    from sparse_attn.ops.cuda_graph_state import disable_graph
    from sparse_attn.ops.tli_static_cache import TLIStaticLayer
    import sparse_attn.ops.kv_row_update as ops
    disable_graph()
    fused,legacy=TLIStaticLayer(17,4),TLIStaticLayer(17,4)
    calls=[];native=ops.append_kv_batch
    def record(*args):calls.append(1);return native(*args)
    monkeypatch.setattr(ops,'append_kv_batch',record)
    for step in range(8):
        nk=torch.randn(4,3,1,32);nv=torch.randn_like(nk)
        ctx=BatchCtx(batch=4,seq_lens=[-2,step+1,30,5],gen_lens=[0]*4,
                     alive=[True,True,step%2==0,False],pad_id=0)
        with batch_decode_scope(ctx):
            monkeypatch.setenv('TLI_OPERATOR_MODE','off');legacy.update(nk,nv)
            monkeypatch.setenv('TLI_OPERATOR_MODE','serial');fused.update(nk,nv)
        torch.testing.assert_close(fused.keys,legacy.keys,rtol=0,atol=0)
        torch.testing.assert_close(fused.values,legacy.values,rtol=0,atol=0)
    assert len(calls)==8 and fused.cumulative_length==legacy.cumulative_length==8


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA Graph required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_batch_append_graph_mutable_metadata(dtype):
    b,h,c,d=8,3,17,128
    k=torch.zeros(b,h,c,d,device='cuda',dtype=dtype);v=torch.zeros_like(k)
    nk=torch.zeros(b,h,1,d,device='cuda',dtype=dtype);nv=torch.zeros_like(nk)
    slots=torch.zeros(b,device='cuda',dtype=torch.int64)
    alive=torch.zeros(b,device='cuda',dtype=torch.bool)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):append_kv_batch(k,v,nk,nv,slots,alive)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):append_kv_batch(k,v,nk,nv,slots,alive)
    expected_k,expected_v=k.clone(),v.clone()
    pointers=(k.data_ptr(),v.data_ptr())
    for step in range(40):
        # Empty work, revived rows, ring-like wrap, invalid metadata masks.
        host_slots=[(step+i)%(c+2)-1 for i in range(b)]
        host_alive=[step%7!=0 and (i+step)%3!=0 for i in range(b)]
        slots.copy_(torch.tensor(host_slots,device='cuda'))
        alive.copy_(torch.tensor(host_alive,device='cuda'))
        nk.normal_();nv.normal_()
        for i,(slot,live) in enumerate(zip(host_slots,host_alive)):
            if live and 0<=slot<c:
                expected_k[i,:,slot:slot+1].copy_(nk[i])
                expected_v[i,:,slot:slot+1].copy_(nv[i])
        graph.replay()
        torch.testing.assert_close(k,expected_k,rtol=0,atol=0)
        torch.testing.assert_close(v,expected_v,rtol=0,atol=0)
        assert pointers==(k.data_ptr(),v.data_ptr())
