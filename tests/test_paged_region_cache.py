import pytest
import torch
from sparse_attn.ops.paged_region_cache import PagedRegionLayer,PagedRegionCache
from sparse_attn.ops.frozen_region_cache import RegionGeometry,RegionCacheLayer
from sparse_attn.ops.paged_region_kernels import paged_attention

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('sink',[0,4])
@torch.inference_mode()
def test_aging_no_copy_and_real_page_reclaim(near,far,sink):
    g=RegionGeometry(sink=sink,near=32,window=17,page=16,max_new_tokens=180)
    cache=PagedRegionLayer(g);torch.manual_seed(31)
    k=torch.randn(1,2,133,8);v=torch.randn(1,2,133,6)
    cache.update(k,v);cache.freeze(near,far)
    historyk=k.clone();historyv=v.clone()
    for step in range(150):
        snapshots={lp:(x[1].data_ptr(),x[2].data_ptr(),x[1].clone(),x[2].clone()) for lp,x in cache.pages.items()}
        nk=torch.randn(1,2,1,8);nv=torch.randn(1,2,1,6)
        oldlength=cache.logical_length;cache.update(nk,nv)
        historyk=torch.cat([historyk,nk],2);historyv=torch.cat([historyv,nv],2)
        s,f,n=g.bounds(cache.logical_length)
        expected=[p for p in range(cache.logical_length) if p<s or p>=n or (p<f and not far) or (f<=p<n and not near)]
        assert cache.visible_positions()==expected
        for lp,(_,pk,pv) in cache.pages.items():
            lo=lp*g.page;hi=min(lo+g.page,cache.logical_length)
            torch.testing.assert_close(pk[:,:,:hi-lo],historyk[:,:,lo:hi],rtol=0,atol=0)
            torch.testing.assert_close(pv[:,:,:hi-lo],historyv[:,:,lo:hi],rtol=0,atol=0)
            if lp in snapshots:
                ak,av,bk,bv=snapshots[lp];assert pk.data_ptr()==ak and pv.data_ptr()==av
                end=min(g.page,max(0,oldlength-lo))
                torch.testing.assert_close(pk[:,:,:end],bk[:,:,:end],rtol=0,atol=0)
        report=cache.report();assert report['aging_copy_bytes']==0
        if far:assert len(cache.pages)<=((g.near+g.window)//g.page+4 if not near else g.window//g.page+4)
        else:assert report['resident_tokens']==cache.logical_length
    if far:assert cache.reclaimed_pages>0 and cache.reused_page_ids>0

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@torch.inference_mode()
def test_pagewise_attention_matches_dense_visible_oracle(near,far):
    torch.manual_seed(72);g=RegionGeometry(near=32,window=17,page=16)
    layer=PagedRegionLayer(g);k=torch.randn(1,2,155,8);v=torch.randn(1,2,155,6)
    layer.update(k,v);layer.freeze(near,far);q=torch.randn(1,6,1,8)
    mask=torch.zeros(1,1,2,155,dtype=torch.bool);mask[...,layer.visible_positions()]=True
    got=paged_attention(layer,q,mask,.17)
    ids=layer.visible_positions();ks=k[:,:,ids].repeat_interleave(3,1);vs=v[:,:,ids].repeat_interleave(3,1)
    expected=torch.nn.functional.scaled_dot_product_attention(q,ks,vs,scale=.17).transpose(1,2)
    torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)

@torch.inference_mode()
def test_page_tables_reused_until_page_lifecycle_changes():
    layer=PagedRegionLayer(RegionGeometry(near=32,window=17,page=16));x=torch.randn(1,1,130,8)
    layer.update(x,x);layer.freeze(True,False);a=layer.pointer_tables()
    layer.update(x[:,:,:1],x[:,:,:1]);b=layer.pointer_tables()
    assert a is b
    with pytest.raises(ValueError):layer.freeze(False,False)
    with pytest.raises(ValueError):layer.update(x[:,:,:2],x[:,:,:2])

@torch.inference_mode()
def test_cross_request_dtype_rejected_before_aging_or_new_page():
    layer=PagedRegionLayer(RegionGeometry(near=4,window=1,page=4))
    x=torch.randn(1,1,4,8);layer.update(x,x);layer.freeze(False,False)
    before=layer.report()
    with pytest.raises(ValueError,match='dtype/device changed'):
        layer.update(x[:,:,:1].bfloat16(),x[:,:,:1].bfloat16())
    assert layer.report()==before

@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@torch.inference_mode()
def test_ring_arithmetic_bfloat16_rounding(near,far):
    torch.manual_seed(43)
    layer=PagedRegionLayer(RegionGeometry(near=32,window=17,page=16))
    k=torch.randn(1,2,155,8).bfloat16();v=torch.randn(1,2,155,6).bfloat16()
    layer.update(k,v);layer.freeze(near,far)
    q=torch.randn(1,6,1,8).bfloat16()
    mask=torch.zeros(1,1,2,155,dtype=torch.bool);mask[...,layer.visible_positions()]=True
    mask[:,:,0,::3]=False;mask[:,:,1,::5]=False
    actual=paged_attention(layer,q,mask,.17,ring_arithmetic=True)
    scores=(q*.17).float()@k.repeat_interleave(3,1).float().transpose(-1,-2)
    fullmask=mask.reshape(1,2,1,-1).repeat_interleave(3,1)
    probs=scores.masked_fill(~fullmask,float('-inf')).softmax(-1).bfloat16()
    expected=(probs@v.repeat_interleave(3,1)).transpose(1,2)
    torch.testing.assert_close(actual,expected,atol=.002,rtol=.002)

@pytest.mark.parametrize('length,width,vwidth,fast,upper,healthy,cuda,expected',[
    (16383,128,128,'1','2048',True,True,True),
    (16384,128,128,'1','2048',True,True,False),
    (16384,128,128,'0','2048',True,True,True),
    (16384,128,128,'1','4096',True,True,True),
    (32768,192,128,'1','2048',True,True,True),
    (32768,128,128,'1','2048',False,True,True),
    (32768,128,128,'1','2048',True,False,True),
])
def test_ring_probability_rounding_tracks_fallbacks(monkeypatch,length,width,vwidth,fast,upper,healthy,cuda,expected):
    from sparse_attn.ops import eager_decoding as eager
    from sparse_attn.ops.paged_region_kernels import ring_rounds_probabilities
    monkeypatch.setenv('TLI_E127_FAST',fast);monkeypatch.setenv('TLI_E127_W_UPPER',upper)
    monkeypatch.setattr(eager,'_HAS_TRITON',True)
    monkeypatch.setitem(eager._EAGER_TRITON_OK,'ok',healthy)
    assert ring_rounds_probabilities(length,width,vwidth,cuda=cuda)==expected
