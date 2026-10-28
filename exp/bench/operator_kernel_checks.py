"""Offline compilation or CPU interpretation. Neither is native GPU acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def geometry(dq, dv, protected, ring, width=37):
    return dict(QSB=8*dq,QSH=dq,KSB=(35 if ring else 93)*2*dq,KST=2*dq,KSH=dq,
                VSB=(35 if ring else 93)*2*dv,VST=2*dv,VSH=dv,
                HQ=8,HK=2,DQ=dq,DV=dv,DQB=1<<(dq-1).bit_length(),DVB=1<<(dv-1).bit_length(),
                CAP=35 if ring else 93,WIDTH=13 if protected else width,
                SPLITS=4,SPAN=max(32,((13 if protected else width)+127)//128*32),
                PROTECTED=protected,SINK=4,SWA=9,RING_SINK=4 if ring else 0,
                RING_TAIL=31 if ring else 0,TILE=32)


def compile_kernels():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from sparse_attn.ops.partial_attention import _partial_simt, _partial_tc, _merge
    records=[]
    for sm in (80,90):
        for backend,fn in (('simt',_partial_simt),('tc',_partial_tc)):
            for dq,dv in ((128,128),(192,128)):
                for protected in (False,True):
                    for ring in (False,True):
                        const=geometry(dq,dv,protected,ring,1024)
                        if protected:
                            const.update(SWA=128,WIDTH=132,SPLITS=1,SPAN=160)
                        if ring:
                            const.update(RING_TAIL=4224,CAP=4228,KSB=4228*2*dq,VSB=4228*2*dv)
                        else:
                            const.update(CAP=32768,KSB=32768*2*dq,VSB=32768*2*dv)
                        if backend=='tc': const['GROUP_BLOCK']=16
                        sig={name:('*bf16' if name in ('Q','K','V') else
                                   '*i32' if name in ('IDS','COUNT') else
                                   '*i64' if name=='LEN' else '*fp32') for name in fn.arg_names[:9]}
                        result=triton.compile(ASTSource(fn,sig,constexprs=const),
                                              target=GPUTarget('cuda',sm,32),
                                              options={'num_warps':4,'num_stages':2})
                        ptx=result.asm['ptx']
                        if backend=='tc': assert 'mma.sync' in ptx or 'wgmma' in ptx
                        records.append(dict(sm=sm,backend=backend,dq=dq,dv=dv,protected=protected,ring=ring,
                                            shared_bytes=result.metadata.shared))
        sig={name:'*bf16' if name=='OUT' else '*fp32' for name in _merge.arg_names[:7]}
        triton.compile(ASTSource(_merge,sig,constexprs=dict(P=1,S=4,DV=128,DB=128,NB=8)),
                       target=GPUTarget('cuda',sm,32),options={'num_warps':4})
    print(json.dumps(dict(scope='OFFLINE compile only; no native GPU execution',partial_kernels=len(records),
                          merge_kernels=2,records=records),indent=2))


def interpret_kernels():
    import torch
    from triton.runtime import interpreter
    from sparse_attn.ops.partial_attention import _partial_simt, _partial_tc, _merge
    # Triton 3.4's loop conversion predates NumPy's stricter scalar conversion.
    original=interpreter._patch_lang_tensor
    def patch(tensor):
        original(tensor)
        tensor.__index__=lambda self:int(self.handle.data.item())
    interpreter._patch_lang_tensor=patch
    import importlib.util
    spec=importlib.util.spec_from_file_location('operator_test_inputs',ROOT/'tests/test_operator_pipeline.py')
    test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    torch.set_num_threads(1)
    count=0
    for backend,fn in (('simt',_partial_simt),('tc',_partial_tc)):
        for dq,dv in ((128,128),(192,128)):
            for ring in (False,True):
                for length in (0,3,17,93):
                    q,k,v,sel,lens,layout,full=test.fixture(dtype=torch.float32,length=length,dq=dq,dv=dv,ring=ring)
                    qs=(q[:,0]*.17).contiguous()
                    states=[]
                    for protected in (True,False):
                        const=geometry(dq,dv,protected,ring,sel.indices.shape[-1])
                        const.update(KSB=k.stride(0),KST=k.stride(1),KSH=k.stride(2),
                                     VSB=v.stride(0),VST=v.stride(1),VSH=v.stride(2),CAP=k.shape[1])
                        if backend=='tc':const['GROUP_BLOCK']=16
                        m=torch.empty((2,8,4));l=torch.empty_like(m);a=torch.empty((2,8,4,dv))
                        fn[(4 if backend=='tc' else 16,4)](qs,k,v,sel.indices,sel.counts,lens,m,l,a,**const)
                        states.append((m,l,a))
                    out=torch.empty((2,1,8,dv))
                    _merge[(16,)](*states[0],*states[1],out,P=4,S=4,DV=dv,DB=dv,NB=8)
                    torch.testing.assert_close(out,test.oracle(q,full,sel,lens),atol=1e-5,rtol=5e-5)
                    count+=1
    print(json.dumps(dict(scope='CPU Triton interpreter, FP32 math; TC hardware/low-precision NOT validated',cases=count)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['compile','interpret']);args=parser.parse_args()
    if args.mode=='interpret':
        os.environ['TRITON_INTERPRET']='1'
        interpret_kernels()
    else:
        compile_kernels()
