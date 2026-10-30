import pytest
import torch
from sparse_attn.ops.ordered_attention import ordered_attention
from sparse_attn.ops.eager_decoding import eager_decoding_attn,eager_decoding_check_errors,_raise_for_code

NATIVE=pytest.mark.skipif(not torch.cuda.is_available(),reason='queued native GPU required')


def _case(batch,group,dtype):
    cap,width,h=32768,1024,2
    q=torch.randn(batch,1,h*group,128,device='cuda',dtype=dtype)
    k=torch.randn(batch,cap,h,128,device='cuda',dtype=dtype);v=torch.randn_like(k)
    values=[(1,9217,16385,32613)[i%4] for i in range(batch)]
    lengths=torch.tensor(values,device='cuda',dtype=torch.int64)
    ids=torch.stack([torch.randint(0,n,(h,width),device='cuda',dtype=torch.int32) for n in values])
    counts=torch.tensor([[0 if b==0 else width,97] for b in range(batch)],device='cuda',dtype=torch.int32)
    return q,k,v,ids,counts,lengths


def _reference(q,k,v,ids,counts,lengths,sink=4,swa=128):
    mask=torch.zeros(k.shape[0],1,k.shape[2],k.shape[1],device='cuda',dtype=torch.bool)
    for b,n in enumerate(lengths.tolist()):
        mask[b,0,:,:min(sink,n)]=True
        mask[b,0,:,max(0,n-swa):n]=True
        for h,c in enumerate(counts[b].tolist()):mask[b,0,h,ids[b,h,:c].long()]=True
    cu=torch.arange(k.shape[0]+1,device='cuda',dtype=torch.int32)*k.shape[1]
    return eager_decoding_attn(q,k,v,mask,1,cu,128**-.5,valid_length=lengths.tolist())


@NATIVE
@pytest.mark.parametrize('batch',[1,2,4,8])
@pytest.mark.parametrize('group',[1,2,4,8])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_original_ordered_reduction_exact_and_replay(batch,group,dtype):
    torch.manual_seed(771)
    q,k,v,ids,counts,lengths=_case(batch,group,dtype)
    with torch.no_grad():
        expected=_reference(q,k,v,ids,counts,lengths)
        for _ in range(2):actual=ordered_attention(q,k,v,(ids,counts),lengths,sink=4,swa=128,scale=128**-.5)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out=ordered_attention(q,k,v,(ids,counts),lengths,sink=4,swa=128,scale=128**-.5)
        lengths.add_(1);q.mul_(.9)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(out,_reference(q,k,v,ids,counts,lengths),rtol=0,atol=0)
        eager_decoding_check_errors()


@NATIVE
@pytest.mark.parametrize('bad',['count','id','length','empty','overflow'])
def test_deferred_contract_failure_and_recovery(monkeypatch,bad):
    q,k,v,ids,counts,lengths=_case(2,2,torch.bfloat16)
    sink,swa=4,128;message='invalid selected'
    with monkeypatch.context() as env:
        if bad=='count':counts[1,0]=1025
        if bad=='id':ids[1,0,0]=-1
        if bad=='length':lengths[1]=32769
        if bad=='empty':counts.zero_();sink=swa=0;message='全 False'
        if bad=='overflow':env.setenv('TLI_E127_W_UPPER','1');message='上界'
        ordered_attention(q,k,v,(ids,counts),lengths,sink=sink,swa=swa,scale=128**-.5)
        with pytest.raises(RuntimeError,match=message):eager_decoding_check_errors()
    q,k,v,ids,counts,lengths=_case(2,2,torch.bfloat16)
    actual=ordered_attention(q,k,v,(ids,counts),lengths,sink=4,swa=128,scale=128**-.5)
    torch.testing.assert_close(actual,_reference(q,k,v,ids,counts,lengths),rtol=0,atol=0)
    eager_decoding_check_errors()


def test_invalid_selection_code_is_not_silently_accepted():
    with pytest.raises(RuntimeError,match='invalid selected'):_raise_for_code(4)


@pytest.mark.parametrize('strides',[(2**31,128,128,1),(16384*128,2**31,128,1)])
def test_storage_holes_are_included_in_address_bound(strides):
    from sparse_attn.ops.ordered_attention import _int32_addressable
    value=torch.empty_strided((2,16384,1,128),strides,device='meta',dtype=torch.bfloat16)
    assert value.numel()<torch.iinfo(torch.int32).max
    assert not _int32_addressable(value)


def test_head_major_static_view_fits_address_bound():
    from sparse_attn.ops.ordered_attention import _int32_addressable
    value=torch.empty(8,8,32768,128,device='meta',dtype=torch.bfloat16).transpose(1,2)
    assert _int32_addressable(value)


@pytest.mark.parametrize('selected_width,packed_width',[(1024,1),(1,2048)])
def test_broadcast_kv_cannot_hide_oversized_index_tables(selected_width,packed_width):
    from sparse_attn.ops.ordered_attention import _int32_addressable,_int32_index_extents
    batch,heads=8,262144
    k=torch.empty_strided((batch,16384,heads,128),(0,128,0,1),device='meta')
    assert _int32_addressable(k)
    assert batch*heads*128<torch.iinfo(torch.int32).max
    assert not _int32_index_extents(batch,heads,selected_width,packed_width)
