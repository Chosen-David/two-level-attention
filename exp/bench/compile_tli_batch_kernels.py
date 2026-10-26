"""Offline compiler gate only: no GPU execution or timing."""
import argparse, hashlib, json
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sparse_attn.indexer import tli_batch_kernels as kernels


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rows=[]
    for arch in (80,90):
        for d in (32,128):
            for mode in (0,1,2):
                for dtype in ('fp32','bf16'):
                    sig=dict(K='*'+dtype,KMAX='*'+dtype,Q='*fp32',IDS='*i64',KEEP='*i1',OUT='*'+dtype,N='i32')
                    const=dict(KB=32768*8*d,KT=8*d,KH=d,QB=32*d,QH=4*d,QG=d,H=8,G=4,D=d,MODE=mode,TILE=32)
                    obj=triton.compile(ASTSource(kernels._score,sig,constexprs=const),target=GPUTarget('cuda',arch,32),options=dict(num_warps=4,enable_fp_fusion=False))
                    rows.append(dict(arch=arch,d=d,mode=mode,dtype=dtype,signature=sig,constants=const,ptx_sha256=hashlib.sha256(obj.asm['ptx'].encode()).hexdigest(),cubin_bytes=len(obj.asm['cubin'])))
    Path(a.out).write_text(json.dumps(dict(triton=triton.__version__,status='offline compile only',source_sha256=hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),cases=rows),indent=2)+'\n')
    print(f'{len(rows)} CUDA specializations compiled; GPU runtime unverified')
if __name__=='__main__':main()
