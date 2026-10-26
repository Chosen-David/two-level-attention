"""Opt-in GPU producer/consumer fusion; no page/candidate staging tensors.

Private callers validate the append-only, finite-K, no-QDQ batch contract.
All live positions come from device lengths. Compile/warm before capture.
This module is a GPU candidate, not a measured performance claim.
"""
import torch
import triton
import triton.language as tl
from .tli_batch_kernels import _mask_base


@triton.jit(do_not_specialize=['P'])
def _pool_score(K, KMAX, Q, LENGTHS, OUT, P,
                H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                BS: tl.constexpr, LO: tl.constexpr, SINK: tl.constexpr,
                SWA: tl.constexpr, NEAR: tl.constexpr,
                REGION: tl.constexpr, MINMAX: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(1)
    b,h = row//H,row%H
    n = tl.program_id(0)*TILE+tl.arange(0,TILE)
    d = tl.arange(0,D)
    length = tl.load(LENGTHS+b)
    swalo = tl.maximum(length-SWA,0)
    near = tl.maximum((length-SWA-NEAR)//BS,LO)
    if REGION == 0:
        live = (n>=LO) & (n<near)
    else:
        end = tl.maximum(near,(length+BS-1)//BS-max(1,SWA//BS))
        bypage = swalo//BS
        byok = (swalo%BS != 0) & (bypage>=LO)
        live = (n>=near) & (n<end) & ~((n==bypage) & byok)
    if SINK%BS:
        live &= n != LO
    live &= n<P
    result = tl.full((TILE,),float('-inf'),tl.float32)
    # Entire out-of-region tiles do no Q/K loads or dot products. Partial
    # tiles mask feature loads. Full-P output preserves the old top-k domain.
    if tl.sum(live.to(tl.int32),0)>0:
        offsets = ((b*P+n[:,None])*H+h)*D+d[None,:]
        x = tl.load(K+offsets,live[:,None],other=0).to(tl.float32)
        if MINMAX:
            high = tl.load(KMAX+offsets,live[:,None],other=0).to(tl.float32)
        acc = tl.full((TILE,),0,tl.float32)
        for g in range(G):
            q = tl.load(Q+(row*G+g)*D+d)
            if MINMAX:
                s = tl.sum(x*tl.minimum(q[None,:],0),1)+tl.sum(high*tl.maximum(q[None,:],0),1)
            else:
                s = tl.sum(x*q[None,:],1)
            # The old score buffer rounded each Q-head score before mean.
            acc += s.to(OUT.dtype.element_ty).to(tl.float32)
        result = tl.where(live,acc/G,float('-inf'))
    tl.store(OUT+row*P+n,result,n<P)


def coarse_indices(state, q, lengths, dtype, nf, nn):
    """Pool-restricted score + dtype-rounding + GQA mean + geometry mask.

    A zero quota has no score allocation, no kernel launch, and no top-k.
    Keep full-P top-k inputs so compaction/tie domains do not shrink.
    """
    idx=state.indexer
    def select(method, region, quota):
        if quota == 0:
            return torch.empty((state.B,state.H,0),device=q.device,dtype=torch.long)
        minmax = method == 'minmax'
        k = state.kmin if minmax else state.kavg
        out = torch.empty((state.B,state.H,state.P),device=q.device,dtype=dtype)
        tile = 8 if q.shape[-1]>64 else 32
        _pool_score[(triton.cdiv(state.P,tile),state.B*state.H)](
            k,state.kmax if minmax else k,q,lengths,out,state.P,
            state.H,q.shape[2],q.shape[-1],state.bs,idx._sink_pool_lo_blk,
            idx.sink_tokens,idx.sliding_window_size,max(state.bs,int(idx.nt_near_len)),
            region,minmax,tile,num_warps=4,enable_fp_fusion=False)
        return out.topk(min(quota,state.P),-1).indices
    return select(idx.far_method,0,nf),select(idx.near_method,1,nn)


@triton.jit
def _update_cache(K, LENGTHS, CDIMS, FDIMS, KMIN, KMAX, KAVG, KFINE, PREV,
                  KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
                  KD: tl.constexpr, FB: tl.constexpr, FT: tl.constexpr,
                  FH: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
                  BS: tl.constexpr, DC: tl.constexpr, DF: tl.constexpr,
                  HAS_C: tl.constexpr, HAS_F: tl.constexpr,
                  MINMAX: tl.constexpr, AVG: tl.constexpr,
                  TB: tl.constexpr, TD: tl.constexpr):
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.program_id(0)*TD + tl.arange(0, TD)
    t = tl.arange(0, TB)
    length = tl.load(LENGTHS+b)
    last = length-1
    page = last//BS
    cd = d
    if HAS_C:
        cd = tl.load(CDIMS+d, d<DC, other=0)
    token = page*BS+t
    # Zero padding is part of the reference page statistics (also min/max).
    x = tl.load(K+b*KB+token[:, None]*KT+h*KH+cd[None, :]*KD,
                (t[:, None]<BS) & (token[:, None]<length) & (d[None, :]<DC),
                other=0).to(tl.float32)
    dst = ((b*P+page)*H+h)*DC+d
    if MINMAX:
        low = tl.min(tl.where(t[:, None]<BS, x, float('inf')), 0)
        high = tl.max(tl.where(t[:, None]<BS, x, float('-inf')), 0)
        tl.store(KMIN+dst, low, d<DC)
        tl.store(KMAX+dst, high, d<DC)
    if AVG:
        tl.store(KAVG+dst, tl.sum(x, 0)/BS, d<DC)
    fd = d
    if HAS_F:
        fd = tl.load(FDIMS+d, d<DF, other=0)
    fine = tl.load(K+b*KB+last*KT+h*KH+fd*KD, d<DF, other=0)
    tl.store(KFINE+b*FB+last*FT+h*FH+d, fine, d<DF)
    if (h == 0) & (tl.program_id(0) == 0):
        tl.store(PREV+b, length)


def update_cache(state, k, lengths):
    """One launch: project page, reduce stats, direct-store fine and lengths."""
    cd, fd = state.coarse_dims, state.fine_dims
    dc = k.shape[-1] if cd is None else cd.numel()
    df = state.kfine.shape[-1]
    _update_cache[(triton.cdiv(max(dc, df), 32), state.B*state.H)](
        k, lengths, k if cd is None else cd, k if fd is None else fd,
        k if state.kmin is None else state.kmin,
        k if state.kmax is None else state.kmax,
        k if state.kavg is None else state.kavg, state.kfine, state.previous,
        *k.stride(), *state.kfine.stride()[:3], state.H, state.P, state.bs,
        dc, df, cd is not None, fd is not None, state.need_minmax,
        state.need_avg, triton.next_power_of_2(state.bs), 32,
        num_warps=4, enable_fp_fusion=False)


@triton.jit(do_not_specialize=['SCALE'])
def _queries(Q, CDIMS, FDIMS, QC, QF, SCALE,
             QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
             HQ: tl.constexpr, DC: tl.constexpr, DF: tl.constexpr,
             HAS_C: tl.constexpr, HAS_F: tl.constexpr,
             SAME: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b, h = row//HQ, row%HQ
    d = tl.arange(0, BLOCK)
    cd = d
    if HAS_C:
        cd = tl.load(CDIMS+d, d<DC, other=0)
    q = tl.load(Q+b*QB+h*QH+cd*QD, d<DC, other=0)
    # Preserve Torch's input-dtype rounding before the FP32 score input.
    qc = (q.to(tl.float32)*SCALE).to(q.dtype).to(tl.float32)
    tl.store(QC+row*DC+d, qc, d<DC)
    if not SAME:
        fd = d
        if HAS_F:
            fd = tl.load(FDIMS+d, d<DF, other=0)
        f = tl.load(Q+b*QB+h*QH+fd*QD, d<DF, other=0)
        qf = (f.to(tl.float32)*SCALE).to(f.dtype).to(tl.float32)
        tl.store(QF+row*DF+d, qf, d<DF)


def queries(q, scale, coarse_dims, fine_dims, hkv):
    B, _, HQ, D = q.shape
    dc = D if coarse_dims is None else coarse_dims.numel()
    df = D if fine_dims is None else fine_dims.numel()
    qc = torch.empty((B, hkv, HQ//hkv, dc), device=q.device, dtype=torch.float32)
    same = coarse_dims is fine_dims
    qf = qc if same else torch.empty((B, hkv, HQ//hkv, df), device=q.device, dtype=torch.float32)
    _queries[(B*HQ,)](q, q if coarse_dims is None else coarse_dims,
        q if fine_dims is None else fine_dims, qc, qf, scale,
        q.stride(0), q.stride(2), q.stride(3), HQ, dc, df,
        coarse_dims is not None, fine_dims is not None, same,
        triton.next_power_of_2(max(dc, df)), num_warps=4, enable_fp_fusion=False)
    return qc, qf


@triton.jit
def _candidate_pages(FI, NI, row, length, slot,
                     NF: tl.constexpr, NN: tl.constexpr, BS: tl.constexpr,
                     LO: tl.constexpr, SINK: tl.constexpr, SWA: tl.constexpr,
                     NEAR: tl.constexpr, SLOTS: tl.constexpr):
    swalo = tl.maximum(length-SWA, 0)
    near = tl.maximum((length-SWA-NEAR)//BS, LO)
    swablock = tl.maximum(near, (length+BS-1)//BS-max(1, SWA//BS))
    bypage = swalo//BS
    byok = (swalo%BS != 0) & (bypage >= LO)
    page = tl.full((SLOTS,), 0, tl.int64)
    valid = tl.full((SLOTS,), False, tl.int1)
    if NF:
        fp = tl.load(FI+row*NF+slot, slot<NF, other=0)
        fvalid = (fp>=LO) & (fp<near)
        if SINK%BS:
            fvalid &= fp != LO
        page = tl.where(slot<NF, fp, page)
        valid = (slot<NF) & fvalid
    if NN:
        np = tl.load(NI+row*NN+slot-NF, (slot>=NF) & (slot<NF+NN), other=0)
        nvalid = (np>=near) & (np<swablock) & ~((np==bypage) & byok)
        if SINK%BS:
            nvalid &= np != LO
        page = tl.where((slot>=NF) & (slot<NF+NN), np, page)
        valid |= (slot>=NF) & (slot<NF+NN) & nvalid
    page = tl.where(slot==NF+NN, LO if SINK%BS else 0, page)
    page = tl.where(slot==NF+NN+1, bypage, page)
    valid |= (slot==NF+NN) & (SINK%BS != 0)
    valid |= (slot==NF+NN+1) & byok & ~((bypage==LO) & (SINK%BS != 0))
    return page, valid


@triton.jit
def _candidates(FI, NI, LENGTHS, IDS, KEEP,
                H: tl.constexpr, NF: tl.constexpr, NN: tl.constexpr,
                BS: tl.constexpr, LO: tl.constexpr, SINK: tl.constexpr,
                SWA: tl.constexpr, NEAR: tl.constexpr, BLOCK: tl.constexpr,
                PAGE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(LENGTHS+row//H)
    slots = tl.arange(0, PAGE_BLOCK)
    pages, page_valid = _candidate_pages(FI,NI,row,length,slots,NF,NN,BS,LO,SINK,SWA,NEAR,PAGE_BLOCK)
    counts = tl.minimum(tl.maximum(length-pages*BS,0),BS)
    total = tl.sum(tl.where(page_valid & (slots<NF+NN+2),counts,0),0).to(tl.int32)
    R: tl.constexpr = (NF+NN+2)*BS
    before = tl.full((),0,tl.int32)
    ar = tl.arange(0,BLOCK)
    # Only one token chunk is live. Avoid the 255-register full-row scan at R=4224.
    for chunk in range(tl.cdiv(R,BLOCK)):
        n = chunk*BLOCK+ar
        page, valid = _candidate_pages(FI,NI,row,length,n//BS,NF,NN,BS,LO,SINK,SWA,NEAR,BLOCK)
        ids = page*BS+n%BS
        valid &= (n<R) & (ids<length)
        flags = valid.to(tl.int32)
        prefix = before+tl.cumsum(flags,0)
        dest = tl.where(valid,prefix-1,total+n-prefix)
        tl.store(IDS+row*R+dest,ids,n<R)
        tl.store(KEEP+row*R+dest,valid,n<R)
        before += tl.sum(flags,0)


def candidates(fi, ni, lengths, block_size, lo, sink, swa, near_len):
    """Expand, validate, bypass-deduplicate and stable-partition in one CTA/row."""
    B, H, nf = fi.shape
    nn = ni.shape[-1]
    r = (nf+nn+2)*block_size
    ids = torch.empty((B,H,r), device=fi.device, dtype=torch.long)
    keep = torch.empty_like(ids, dtype=torch.bool)
    _candidates[(B*H,)](fi,ni,lengths,ids,keep,H,nf,nn,block_size,lo,sink,swa,
                        max(block_size,int(near_len)),256,triton.next_power_of_2(nf+nn+2),num_warps=4)
    return ids, keep


@triton.jit(do_not_specialize=['S', 'R', 'NTOP'])
def _mask_top(TOP, IDS, KEEP, LENGTHS, OUT, S, R, NTOP,
              H: tl.constexpr, LO_TOKEN: tl.constexpr, SINK: tl.constexpr,
              SWA: tl.constexpr, K2: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    length = tl.load(LENGTHS+row//H)
    swalo = tl.maximum(length-SWA,0)
    budget = tl.maximum(K2-SINK-(length-swalo),0)
    slot = tl.load(TOP+row*NTOP+n, n<NTOP, other=0)
    token = tl.load(IDS+row*R+slot, n<NTOP, other=0)
    keep = tl.load(KEEP+row*R+slot, n<NTOP, other=0)
    valid = (n<NTOP) & (n<budget) & keep & (token>=LO_TOKEN) & (token<swalo)
    tl.store(OUT+row*S+token, True, valid)


def mask_from_top(top, ids, keep, lengths, capacity, lo_token, sink, swa, k2):
    B,H,R = ids.shape
    ntop = top.shape[-1]
    out = torch.empty((B,H,capacity),device=ids.device,dtype=torch.bool)
    # Separate launches guarantee base initialization before sparse writes.
    _mask_base[(triton.cdiv(B*H*capacity,256),)](
        lengths,out,capacity,B,H,sink,swa,256,num_warps=4)
    if ntop:
        _mask_top[(triton.cdiv(ntop,256),B*H)](
            top,ids,keep,lengths,out,capacity,R,ntop,H,lo_token,sink,swa,k2,256,num_warps=4)
    return out
