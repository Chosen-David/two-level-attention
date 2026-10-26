"""Batched GQA scoring. Grid is (token tile, batch*KV head), not B launches.

Reduction follows the repository's scalar Triton scoring kernels: FP32 sums,
then rounding to the query dtype before the separate GQA mean. GPU validation
is still required; the interpreter/offline compiler do not establish speed.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _score(K, KMAX, Q, IDS, KEEP, OUT, N,
           KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
           QB: tl.constexpr, QH: tl.constexpr, QG: tl.constexpr,
           H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
           MODE: tl.constexpr, TILE: tl.constexpr):
    n = tl.program_id(0)*TILE+tl.arange(0,TILE)
    bh = tl.program_id(1)
    b, h = bh//H, bh%H
    d = tl.arange(0,D)
    live = n < N
    if MODE == 2:
        pos = tl.load(IDS+bh*N+n,live,other=0)
        live &= tl.load(KEEP+bh*N+n,n < N,other=0)
    else:
        pos = n
    x = tl.load(K+b*KB+pos[:,None]*KT+h*KH+d[None,:],
                live[:,None],other=0).to(tl.float32)
    if MODE == 1:
        hi = tl.load(KMAX+b*KB+pos[:,None]*KT+h*KH+d[None,:],
                     live[:,None],other=0).to(tl.float32)
    for g in tl.static_range(G):
        q = tl.load(Q+b*QB+h*QH+g*QG+d)
        if MODE == 1:
            s = tl.sum(x*tl.minimum(q[None,:],0),1)+tl.sum(hi*tl.maximum(q[None,:],0),1)
        else:
            s = tl.sum(x*q[None,:],1)
        s = tl.where(live,s,float('-inf'))
        tl.store(OUT+(bh*G+g)*N+n,s,n < N)


def score(k, q, dtype, *, kmax=None, ids=None, keep=None):
    # Private checked caller: cached features contiguous, q compact and FP32.
    B,T,H,D=k.shape
    q=q.contiguous()
    G=q.shape[2]
    n=T if ids is None else ids.shape[-1]
    out=torch.empty((B,H,G,n),dtype=dtype,device=k.device)
    mode=2 if ids is not None else 1 if kmax is not None else 0
    _score[(triton.cdiv(n,32), B*H)](
        k, k if kmax is None else kmax, q,
        k if ids is None else ids, k if keep is None else keep, out,n,
        *k.stride()[:3],*q.stride()[:3],H,G,D,mode,32,
        num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=['R'])
def _compact(IDS, KEEP, OIDS, OKEEP, R, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ar = tl.arange(0,BLOCK)
    total = tl.full((),0,tl.int32)
    for chunk in range(tl.cdiv(R,BLOCK)):
        n = chunk*BLOCK+ar
        flag = tl.load(KEEP+row*R+n,n<R,other=0).to(tl.int32)
        total += tl.sum(flag,0)
    before = tl.full((),0,tl.int32)
    for chunk in range(tl.cdiv(R,BLOCK)):
        n = chunk*BLOCK+ar
        flag = tl.load(KEEP+row*R+n,n<R,other=0).to(tl.int32)
        prefix = before+tl.cumsum(flag,0)
        dest = tl.where(flag != 0,prefix-1,total+n-prefix)
        value = tl.load(IDS+row*R+n,n<R,other=0)
        tl.store(OIDS+row*R+dest,value,n<R)
        tl.store(OKEEP+row*R+dest,flag != 0,n<R)
        before += tl.sum(flag,0)


def compact(ids, keep):
    ids,keep=ids.contiguous(),keep.contiguous()
    out=torch.empty_like(ids);valid=torch.empty_like(keep)
    r=ids.shape[-1]
    _compact[(ids.numel()//r,)](ids,keep,out,valid,r,256,num_warps=4)
    return out,valid
