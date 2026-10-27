"""Logical storage measurements on CPU, not CUDA allocator measurements."""
import pytest
import torch
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace

@pytest.mark.parametrize('fixed',[False,True])
def test_pair_gather_storage_and_no_shrink(fixed):
    ws=FarSkipWorkspace(fixed_shapes=fixed)
    like=torch.empty((),dtype=torch.bfloat16)
    shape=(8,1156,128)
    bufs=[ws.scratch(name,shape,like) for name in ('cuda_gather_x','cuda_gather_y')]
    size=sum(x.untyped_storage().nbytes() for x in bufs)
    assert size==(4734976 if fixed else 8388608)
    for name,out in zip(('cuda_gather_x','cuda_gather_y'),bufs):
        assert ws.scratch(name,(8,12,128),like).data_ptr()==out.data_ptr()
        assert ws.scratch(name,shape,like).data_ptr()==out.data_ptr()

@pytest.mark.parametrize('method',['avg','minmax'])
@torch.inference_mode()
def test_graph_selection_matches_growth_workspace_after_wrap(method):
    torch.manual_seed(78)
    cfg=FarSkipConfig(4,20,2,12,3,4,near_method=method,quantize=False)
    layout=FarSkipLayout(3,27)
    dims=torch.arange(8)
    k=torch.randn(1,layout.capacity,2,16)
    q=torch.randn(1,1,6,16)
    exact=FarSkipGraphWorkspace(cfg,layout,2,dims,dims)
    growth=FarSkipGraphWorkspace(cfg,layout,2,dims,dims)
    growth.ws.fixed_shapes=False
    for w in (exact,growth):w.initialize(k,39)
    previous=None
    for length in range(40,125):
        k[0,layout.slot(length-1)].normal_()
        p=torch.tensor([length])
        a=exact.select(q,k,p,0.25);b=growth.select(q,k,p,0.25)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        ptrs={key:v.data_ptr() for key,v in exact.ws._buffers.items()}
        if previous is not None:assert ptrs==previous
        previous=ptrs
    exact_bytes=sum(v.untyped_storage().nbytes() for v in exact.ws._buffers.values())
    growth_bytes=sum(v.untyped_storage().nbytes() for v in growth.ws._buffers.values())
    assert exact_bytes<growth_bytes
