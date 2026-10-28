"""Offline SM80/SM90 compilation plus ptxas resources, not GPU execution."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sparse_attn.indexer import tli_batch_pipeline as k


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    rows=[]
    def compile_one(fn,arch,sig,const):
        obj=triton.compile(ASTSource(fn,sig,constexprs=const),
            target=GPUTarget('cuda',arch,32),options=dict(num_warps=4,enable_fp_fusion=False))
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'kernel.ptx';src.write_text(obj.asm['ptx'])
            proc=subprocess.run([str(ptxas),'-v','--gpu-name='+re.search(r'\.target\s+(sm_\w+)',obj.asm['ptx'])[1],
                                 str(src),'-o',str(Path(tmp)/'kernel.cubin')],capture_output=True,text=True,check=True)
        rows.append(dict(kernel=fn.__name__,arch=arch,signature=sig,constants=const,
                         shared_bytes=obj.metadata.shared,ptxas=proc.stderr,
                         ptx_sha256=hashlib.sha256(obj.asm['ptx'].encode()).hexdigest(),
                         cubin_bytes=len(obj.asm['cubin'])))
    for arch in (80,90):
        for dtype in ('fp32','fp16','bf16'):
            for dc,df in ((32,32),(32,128),(128,32),(128,128)):
                for layout in ('head','token'):
                    for minmax,avg in ((True,False),(False,True),(True,True)):
                        sig={n:'*'+dtype for n in ('K','KMIN','KMAX','KAVG','KFINE')}
                        sig.update(LENGTHS='*i64',CDIMS='*i64' if dc!=128 else '*'+dtype,
                                   FDIMS='*i64' if df!=128 else '*'+dtype,PREV='*i64')
                        const=dict(KB=32768*8*128,KT=128 if layout=='head' else 8*128,
                            KH=32768*128 if layout=='head' else 128,KD=1,
                            FB=32768*8*df,FT=8*df,FH=df,H=8,P=512,BS=64,DC=dc,DF=df,
                            HAS_C=dc!=128,HAS_F=df!=128,MINMAX=minmax,AVG=avg,TB=64,TD=32)
                        compile_one(k._update_cache,arch,sig,const)
                sig=dict(Q='*'+dtype,CDIMS='*i64' if dc!=128 else '*'+dtype,
                         FDIMS='*i64' if df!=128 else '*'+dtype,QC='*fp32',QF='*fp32',SCALE='fp32')
                const=dict(QB=32*128,QH=128,QD=1,HQ=32,DC=dc,DF=df,HAS_C=dc!=128,
                           HAS_F=df!=128,SAME=dc==df,BLOCK=max(dc,df))
                compile_one(k._queries,arch,sig,const)
        for dtype in ('fp32','fp16','bf16'):
            for d in (32,128):
                for g in (1,4):
                    for region in (0,1):
                        for minmax in (False,True):
                            compile_one(k._pool_score,arch,
                                dict(K='*'+dtype,KMAX='*'+dtype,Q='*fp32',LENGTHS='*i64',OUT='*'+dtype,P='i32'),
                                dict(H=8,G=g,D=d,BS=64,LO=0,SINK=4,SWA=128,NEAR=4096,
                                     REGION=region,MINMAX=minmax,TILE=16 if d>64 else 32))
        for nf,nn,bs in ((0,0,16),(0,32,64),(32,32,64)):
            for sink in (0,4,64):
                compile_one(k._candidates,arch,
                    dict(FI='*i64',NI='*i64',LENGTHS='*i64',IDS='*i64',KEEP='*i1'),
                    dict(H=8,NF=nf,NN=nn,BS=bs,LO=sink//bs,SINK=sink,SWA=128,NEAR=4096,
                         BLOCK=256,PAGE_BLOCK=triton.next_power_of_2(nf+nn+2)))
        compile_one(k._mask_top,arch,
            dict(TOP='*i64',IDS='*i64',KEEP='*i1',LENGTHS='*i64',OUT='*i1',S='i32',R='i32',NTOP='i32'),
            dict(H=8,LO_TOKEN=0,SINK=4,SWA=128,K2=1024,BLOCK=256))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='offline compilation only; runtime pending',
        triton=triton.__version__,source_sha256=hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows),indent=2)+'\n')
    print(f'{len(rows)} specializations compiled; resource logs saved; GPU unverified')


if __name__=='__main__':main()
