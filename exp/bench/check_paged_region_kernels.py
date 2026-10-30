"""Direct paged CUDA correctness. Synthetic checks, not task-quality evidence."""
import argparse,json,os,time
from pathlib import Path
import torch
from sparse_attn.ops.paged_region_cache import PagedRegionLayer
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.paged_region_kernels import paged_attention

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.manual_seed(81);torch.set_num_threads(4);rows=[]
    for dtype in [torch.float32,torch.bfloat16]:
      for d,dv in [(128,128),(192,128)]:
       for near,far in [(False,False),(True,False),(False,True),(True,True)]:
        g=RegionGeometry(near=256,window=128,page=64,max_new_tokens=256)
        k=torch.randn(1,2,777,d,device='cuda',dtype=dtype);v=torch.randn(1,2,777,dv,device='cuda',dtype=dtype)
        layer=PagedRegionLayer(g);layer.update(k,v);layer.freeze(near,far)
        for step in range(130):
            nk=torch.randn(1,2,1,d,device='cuda',dtype=dtype);nv=torch.randn(1,2,1,dv,device='cuda',dtype=dtype)
            layer.update(nk,nv);k=torch.cat([k,nk],2);v=torch.cat([v,nv],2)
        before=layer.report()
        with torch.cuda.stream(torch.cuda.Stream()):
            try:layer.update(nk,nv)
            except ValueError as exc:assert 'owning CUDA stream' in str(exc)
            else:raise AssertionError('cross-stream update accepted')
        assert layer.report()==before
        if near and not far:
            _,f,n=g.bounds(layer.logical_length)
            for pos in range(f,n):
                _,pk,pv=layer.pages[pos//g.page];pk[:,:,pos%g.page].fill_(float('nan'));pv[:,:,pos%g.page].fill_(float('nan'))
        dims=torch.tensor([0,3,d-1],device='cuda');projection=layer.project_visible(dims,64)
        expected=projection.new_zeros(projection.shape);ids=layer.visible_positions()
        expected[:,ids]=k[:,:,ids].index_select(-1,dims).transpose(1,2)
        torch.testing.assert_close(projection,expected,rtol=0,atol=0)
        q=torch.randn(1,8,1,d,device='cuda',dtype=dtype);mask=torch.zeros(1,1,2,layer.logical_length,device='cuda',dtype=torch.bool)
        mask[...,ids]=True
        mask[:,:,0,::3]=False;mask[:,:,1,::5]=False;mask[...,-1]=True
        actual=paged_attention(layer,q,mask,d**-.5)
        fullmask=mask.reshape(1,2,1,-1).repeat_interleave(4,1)
        oracle=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(4,1),v.repeat_interleave(4,1),attn_mask=fullmask,scale=d**-.5).transpose(1,2)
        error=float((actual.float()-oracle.float()).abs().max())
        tol=.003 if dtype==torch.bfloat16 else 2e-6
        torch.testing.assert_close(actual,oracle,atol=tol,rtol=tol)
        rows.append(dict(dtype=str(dtype),d=d,dv=dv,near=near,far=far,max_abs_error=error,report=layer.report()))
        print(rows[-1]|{'report':'omitted'},flush=True)
    Path(args.out).write_text(json.dumps(dict(status='pending independent verification',device=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
