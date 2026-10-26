"""Offline compiler gate only: no GPU execution or timing."""
import argparse
import hashlib
import json
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sparse_attn.indexer import tli_batch_kernels as kernels


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rows=[]
    def compile_case(fn,arch,sig,const):
        obj=triton.compile(ASTSource(fn,sig,constexprs=const),target=GPUTarget('cuda',arch,32),options=dict(num_warps=4,enable_fp_fusion=False))
        rows.append(dict(kernel=fn.__name__,arch=arch,signature=sig,constants=const,
                         shared_bytes=obj.metadata.shared,
                         ptx_sha256=hashlib.sha256(obj.asm['ptx'].encode()).hexdigest(),cubin_bytes=len(obj.asm['cubin'])))
    for arch in (80,90):
        for d in (32,128):
            for mode in (0,1,2):
                for dtype in ('fp32','bf16'):
                    sig=dict(K='*'+dtype,KMAX='*'+dtype,Q='*fp32',IDS='*i64',KEEP='*i1',OUT='*'+dtype,N='i32')
                    const=dict(KB=32768*8*d,KT=8*d,KH=d,QB=32*d,QH=4*d,QG=d,H=8,G=4,D=d,MODE=mode,TILE=32)
                    compile_case(kernels._score,arch,sig,const)
        compile_case(kernels._compact,arch,dict(IDS='*i64',KEEP='*i1',OIDS='*i64',OKEEP='*i1',R='i32'),dict(BLOCK=256))
        for sink in (0,4):
            compile_case(kernels._mask_base,arch,dict(LENGTHS='*i64',OUT='*i1',S='i32'),dict(B=4,H=8,SINK=sink,SWA=128,BLOCK=256))
        compile_case(kernels._mask_selected,arch,dict(IDS='*i64',VALID='*i1',OUT='*i1',S='i32',R='i32'),dict(BLOCK=256))
        for layout in ('head','token'):
            for dc,df in ((32,32),(32,128),(128,32),(128,128)):
                for dtype in ('fp32','bf16'):
                    sig=dict(K='*'+dtype,LENGTHS='*i64',CDIMS='*i64' if dc!=128 else '*'+dtype,
                             FDIMS='*i64' if df!=128 else '*'+dtype,PAGE='*'+dtype,FINE='*'+dtype)
                    const=dict(KB=32768*8*128,KT=128 if layout=='head' else 8*128,
                               KH=32768*128 if layout=='head' else 128,KD=1,H=8,BS=64,
                               DC=dc,DF=df,HAS_C=dc!=128,HAS_F=df!=128,BLOCK=256)
                    compile_case(kernels._gather_update,arch,sig,const)
    Path(a.out).write_text(json.dumps(dict(triton=triton.__version__,status='offline compile only',source_sha256=hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),cases=rows),indent=2)+'\n')
    print(f'{len(rows)} CUDA specializations compiled; GPU runtime unverified')
if __name__=='__main__':main()
