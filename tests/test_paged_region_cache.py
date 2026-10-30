import gc
import weakref

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


@pytest.mark.parametrize('near,far',[(False,False),(True,False),(False,True),(True,True)])
@torch.inference_mode()
def test_freeze_reclaims_exact_whole_pages_and_releases_tensor_references(near,far):
    g=RegionGeometry(sink=2,near=32,window=17,page=16,max_new_tokens=32)
    layer=PagedRegionLayer(g);k=torch.randn(1,2,133,8);v=torch.randn(1,2,133,6)
    layer.update(k,v)
    refs={lp:(weakref.ref(page[1]),weakref.ref(page[2])) for lp,page in layer.pages.items()}
    before_bytes=layer.report()['kv_bytes'];before_pages=set(layer.pages)
    sample=next(iter(layer.pages.values()))
    page_bytes=(sample[1].numel()+sample[2].numel())*k.element_size()
    s,f,n=g.bounds(layer.logical_length)
    resident=set(range(s))|set(range(n,layer.logical_length))
    if not far:resident|=set(range(s,f))
    if not near or not far:resident|=set(range(f,n))
    expected={lp for lp in before_pages if any(p in resident for p in range(lp*g.page,min((lp+1)*g.page,layer.logical_length)))}
    layer.freeze(near,far);gc.collect()
    report=layer.report()
    assert set(layer.pages)==expected
    assert report['reclaimed_pages']==len(before_pages-expected)
    assert report['kv_bytes']==before_bytes-len(before_pages-expected)*page_bytes
    assert report['resident_tokens']==len(resident)
    assert report['inactive_retained_slots']==len(expected)*g.page-len(resident)
    for lp in before_pages-expected:
        assert refs[lp][0]() is None and refs[lp][1]() is None
    for lp in expected:
        assert refs[lp][0]() is not None and refs[lp][1]() is not None


@torch.inference_mode()
def test_near_only_staging_is_invisible_then_ages_into_visible_far_without_initial_savings():
    g=RegionGeometry(sink=2,near=32,window=17,page=16,max_new_tokens=40)
    layer=PagedRegionLayer(g);x=torch.randn(1,2,133,8)
    layer.update(x,x);before=layer.report();_,f,n=g.bounds(layer.logical_length)
    layer.freeze(True,False);after=layer.report()
    assert after['kv_bytes']==before['kv_bytes']
    assert after['live_pages']==before['live_pages']
    assert after['resident_tokens']==layer.logical_length
    assert after['attention_tokens']==layer.logical_length-(n-f)
    assert f not in layer.visible_positions()
    for step in range(g.page):
        layer.update(x[:,:,:1],x[:,:,:1])
    assert g.bounds(layer.logical_length)[1]>f
    assert f in layer.visible_positions()
    assert layer.report()['resident_tokens']==layer.logical_length
    assert layer.aging_copy_bytes==0


@torch.inference_mode()
def test_interleaved_requests_keep_independent_pages_positions_and_free_lists():
    g=RegionGeometry(sink=2,near=32,window=17,page=16,max_new_tokens=80)
    near_only=PagedRegionLayer(g);both=PagedRegionLayer(g)
    a=torch.randn(1,2,133,8);b=torch.randn(1,2,133,8)+1000
    near_only.update(a,a);both.update(b,b)
    near_only.freeze(True,False);both.freeze(True,True)
    for step in range(48):
        na=torch.full((1,2,1,8),float(step));nb=na+1000
        near_only.update(na,na);both.update(nb,nb)
        assert near_only.visible_positions()[0]==0
        assert both.visible_positions()==list(range(2))+list(range(both.logical_length-17,both.logical_length))
        for lp,(_,pk,_) in both.pages.items():
            lo=lp*g.page;hi=min(lo+g.page,both.logical_length)
            expected=torch.cat([b,torch.arange(step+1,dtype=b.dtype).reshape(1,1,-1,1).expand(1,2,-1,8)+1000],2)
            torch.testing.assert_close(pk[:,:,:hi-lo],expected[:,:,lo:hi],rtol=0,atol=0)
        for pa,pb in zip(near_only.pages.values(),both.pages.values()):
            assert pa[1].untyped_storage().data_ptr()!=pb[1].untyped_storage().data_ptr()
    assert near_only.reclaimed_pages==0
    assert both.reclaimed_pages>0 and both.reused_page_ids>0


@torch.inference_mode()
def test_failed_page_copy_rolls_back_and_can_retry(monkeypatch):
    layer=PagedRegionLayer(RegionGeometry(sink=2,near=8,window=4,page=8,max_new_tokens=8))
    x=torch.randn(1,2,13,8);layer.update(x,x);layer.freeze(False,False)
    before=layer.report();offset=layer.logical_length%layer.geometry.page
    target=layer.pages[layer.logical_length//layer.geometry.page][2]
    snapshots={lp:(page[1].clone(),page[2].clone()) for lp,page in layer.pages.items()}
    original=torch.Tensor.copy_;failed=[False]
    def fail_once_after_value_copy(self,src,*args,**kwargs):
        result=original(self,src,*args,**kwargs)
        if not failed[0] and self.data_ptr()==target[...,offset:offset+1,:].data_ptr():
            failed[0]=True;raise RuntimeError('injected copy failure')
        return result
    monkeypatch.setattr(torch.Tensor,'copy_',fail_once_after_value_copy)
    incoming=torch.randn(1,2,1,8)
    with pytest.raises(RuntimeError,match='injected copy failure'):
        layer.update(incoming,incoming)
    assert failed[0] and layer.report()==before
    for lp,page in layer.pages.items():
        torch.testing.assert_close(page[1],snapshots[lp][0],rtol=0,atol=0)
        torch.testing.assert_close(page[2],snapshots[lp][1],rtol=0,atol=0)
    layer.update(incoming,incoming)
    assert layer.logical_length==before['logical_length']+1


@torch.inference_mode()
def test_native_decode_cannot_silently_attend_only_incoming_kv():
    cache=PagedRegionCache(1,RegionGeometry(near=4,window=2,page=4))
    x=torch.randn(1,1,12,8);cache.update(x,x,0)
    cache.freeze({0:dict(skip_near=False,skip_far=False)})
    before=cache.report()
    with pytest.raises(ValueError,match='patched direct-page consumer'):
        cache.update(x[:,:,:1],x[:,:,:1],0)
    assert cache.report()==before
    cache.update_for_paged_decode(x[:,:,:1],x[:,:,:1],0)
    assert cache.layers[0].logical_length==13
