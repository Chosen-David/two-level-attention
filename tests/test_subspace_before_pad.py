"""Frozen-main pairs including packed boundaries, full-D consumers and capture."""
import copy
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from exp.bench.subspace_pad_compare import make_indexer, equal_dict
from exp.bench.subspace_pad_reference import _prepare_index_base as reference


@pytest.mark.parametrize('mode', ['tail','full','mla_rope','moba','quant','l2full',
                                  'projection','lazy_projection','static_pair'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('layout', ['aligned','ragged','packed','valid','strided_dim'])
@torch.no_grad()
def test_full_prepare_matches_main(mode, dtype, layout):
    torch.manual_seed(271027)
    lengths = [33,31,7] if layout == 'packed' else [128 if layout == 'aligned' else 129]
    dim = 192 if mode == 'mla_rope' else 128
    k = torch.randn(1,2,sum(lengths),dim*(2 if layout=='strided_dim' else 1),dtype=dtype).transpose(1,2)
    if layout == 'strided_dim': k=k[...,::2]
    cu = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0)),dtype=torch.int32)
    valid = 65 if layout == 'valid' else None
    ix = make_indexer(mode)
    other = copy.deepcopy(ix)
    expected = reference(other,k,cu,valid_length=valid)
    actual = ix._prepare_index_base(k,cu,valid_length=valid)
    equal_dict(actual,expected)
    assert actual['k_qat'].shape == expected['k_qat'].shape
    if mode == 'l2full' and not ix.l2_quant:
        assert actual['k_qat'] is k


class Zeros(TorchDispatchMode):
    def __init__(self): self.allocations=[]
    def __torch_dispatch__(self,fn,types,args=(),kwargs=None):
        out=fn(*args,**(kwargs or {}))
        if fn == torch.ops.aten.new_zeros.default and out.ndim == 4:
            self.allocations.append((tuple(out.shape),out.untyped_storage().nbytes()))
        return out


def test_padding_storage_reduction_and_moba_full_width():
    k=torch.ones(1,1025,8,128,dtype=torch.bfloat16)
    cu=torch.tensor([0,1025],dtype=torch.int32)
    for mode,width in [('tail',32),('full',128),('moba',128)]:
        ix=make_indexer(mode)
        with Zeros() as old: reference(copy.deepcopy(ix),k,cu)
        with Zeros() as new: ix._prepare_index_base(k,cu)
        assert old.allocations == [((1,1088,8,128),1088*8*128*2)]
        assert new.allocations == [((1,1088,8,width),1088*8*width*2)]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mode',['tail','mla_rope','full','moba','quant','l2full'])
@torch.no_grad()
def test_cuda_eager_capture_replay_and_new_shape(mode,dtype):
    for length in (129,257):
        dim=192 if mode=='mla_rope' else 128
        k=torch.randn(1,2,length,dim,device='cuda',dtype=dtype).transpose(1,2)
        cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
        a,b=make_indexer(mode,'cuda'),make_indexer(mode,'cuda')
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reference(a,k,cu); b._prepare_index_base(k,cu)
        torch.cuda.current_stream().wait_stream(stream)
        expected=reference(a,k,cu)
        equal_dict(expected,b._prepare_index_base(k,cu))
        assert a._stat_ok and b._stat_ok
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out=b._prepare_index_base(k,cu)
        for _ in range(4):
            k.add_(0.125)
            graph.replay()
            equal_dict(reference(a,k,cu),out)
        del graph,out,expected,a,b


@pytest.mark.parametrize('mode',['tail','full','mla_rope','moba','quant','l2full'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@torch.no_grad()
def test_public_incremental_and_new_request(mode,dtype):
    from types import MethodType
    torch.manual_seed(271028)
    a,b=make_indexer(mode),make_indexer(mode)
    a._prepare_index_base=MethodType(reference,a)
    dim=192 if mode=='mla_rope' else 128
    for request in range(2):
        a.clear(); b.clear()
        k=torch.randn(1,2,193,dim,dtype=dtype).transpose(1,2)
        for length in range(119+request,136+request):
            cu=torch.tensor([0,length],dtype=torch.int32)
            equal_dict(a.prepare_index(k[:,:length],cu), b.prepare_index(k[:,:length],cu))
