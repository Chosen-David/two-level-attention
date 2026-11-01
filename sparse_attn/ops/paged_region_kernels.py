"""Page-table projection and selected-token attention; no contiguous KV history.

CPU uses a pagewise two-pass softmax oracle. CUDA reads full K/V directly through
page pointer tables. Only scalar selection indices and low-dimensional index
scratch are materialized. This eager implementation does not support Graphs.
"""
import math
import os
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None

if triton is not None:
    @triton.jit
    def _project(KPTR,DIMS,OUT,L,PAD:tl.constexpr,H:tl.constexpr,D:tl.constexpr,R:tl.constexpr,P:tl.constexpr,
                 S,F,N,SN:tl.constexpr,SF:tl.constexpr,B:tl.constexpr):
        i=tl.program_id(0)*B+tl.arange(0,B)
        rid=i%R;head=(i//R)%H;pos=i//(R*H)
        valid=(i<PAD*H*R)&(pos<L)
        visible=(pos<S)|(pos>=N)|((pos>=S)&(pos<F)&(not SF))|((pos>=F)&(pos<N)&(not SN))
        address=tl.load(KPTR+pos//P,mask=valid,other=0)
        dim=tl.load(DIMS+rid)
        ptr=address.to(tl.pointer_type(OUT.dtype.element_ty))
        val=tl.load(ptr+head*P*D+(pos%P)*D+dim,mask=valid&visible&(address!=0),other=0)
        tl.store(OUT+i,val,mask=i<PAD*H*R)

    @triton.jit
    def _attend(Q,KPTR,VPTR,IDS,OUT,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,DV:tl.constexpr,
                P:tl.constexpr,K:tl.constexpr,SCALE:tl.constexpr,ROUND_PROB:tl.constexpr,BD:tl.constexpr,BV:tl.constexpr,BT:tl.constexpr):
        h=tl.program_id(0);kh=h//(HQ//HK)
        d=tl.arange(0,BD);vd=tl.arange(0,BV);t=tl.arange(0,BT)
        q=tl.load(Q+h*D+d,mask=d<D,other=0).to(tl.float32)
        m=tl.full((),float('-inf'),tl.float32);z=tl.full((),0,tl.float32)
        acc=tl.full((BV,),0,tl.float32)
        for start in range(tl.cdiv(K,BT)):
            ix=start*BT+t
            pos=tl.load(IDS+kh*K+ix,mask=ix<K,other=-1)
            valid=pos>=0
            kp=tl.load(KPTR+pos//P,mask=valid,other=0)
            vp=tl.load(VPTR+pos//P,mask=valid,other=0)
            kptr=kp.to(tl.pointer_type(Q.dtype.element_ty))
            vptr=vp.to(tl.pointer_type(Q.dtype.element_ty))
            keys=tl.load(kptr[:,None]+kh*P*D+(pos%P)[:,None]*D+d[None,:],mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
            score=tl.sum(keys*q[None,:],1)*SCALE
            score=tl.where(valid,score,float('-inf'))
            nm=tl.maximum(m,tl.max(score,0));safe=tl.where(nm==float('-inf'),0.,nm)
            alpha=tl.exp(m-safe);prob=tl.exp(score-safe)
            vals=tl.load(vptr[:,None]+kh*P*DV+(pos%P)[:,None]*DV+vd[None,:],mask=valid[:,None]&(vd[None,:]<DV),other=0).to(tl.float32)
            acc=acc*alpha+tl.sum(vals*prob[:,None],0)
            z=z*alpha+tl.sum(prob,0);m=nm
        if ROUND_PROB:
            # Match eager dense reference: normalized probabilities are rounded
            # to the activation dtype before PV. Direct second pass rereads pages.
            acc=tl.full((BV,),0,tl.float32)
            for start in range(tl.cdiv(K,BT)):
                ix=start*BT+t
                pos=tl.load(IDS+kh*K+ix,mask=ix<K,other=-1)
                valid=pos>=0
                kp=tl.load(KPTR+pos//P,mask=valid,other=0)
                vp=tl.load(VPTR+pos//P,mask=valid,other=0)
                kptr=kp.to(tl.pointer_type(Q.dtype.element_ty))
                vptr=vp.to(tl.pointer_type(Q.dtype.element_ty))
                keys=tl.load(kptr[:,None]+kh*P*D+(pos%P)[:,None]*D+d[None,:],mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
                score=tl.where(valid,tl.sum(keys*q[None,:],1)*SCALE,float('-inf'))
                prob=(tl.exp(score-m)/z).to(Q.dtype.element_ty).to(tl.float32)
                vals=tl.load(vptr[:,None]+kh*P*DV+(pos%P)[:,None]*DV+vd[None,:],mask=valid[:,None]&(vd[None,:]<DV),other=0).to(tl.float32)
                acc+=tl.sum(vals*prob[:,None],0)
            z=1.
        tl.store(OUT+h*DV+vd,acc/z,mask=vd<DV)


def project_visible(layer,dims,block_size):
    _,key,_=next(iter(layer.pages.values()))
    length=layer.logical_length;pad=(length+block_size-1)//block_size*block_size
    out=key.new_zeros((1,pad,key.shape[1],dims.numel()))
    if key.is_cuda:
        if triton is None:raise RuntimeError('Triton required for direct CUDA page reads')
        s,f,n=layer.geometry.bounds(length)
        _project[(triton.cdiv(out.numel(),256),)](layer.pointer_tables()[0],dims.contiguous(),out,
            length,pad,key.shape[1],key.shape[-1],dims.numel(),layer.geometry.page,s,f,n,layer.skip_near,layer.skip_far,256)
    else:
        for lo,hi in layer.intervals():
            for lp in range(lo//layer.geometry.page,(hi-1)//layer.geometry.page+1):
                _,k,_=layer.pages[lp];base=lp*layer.geometry.page
                a,b=max(lo,base),min(hi,base+layer.geometry.page)
                out[:,a:b]=k[...,a-base:b-base,:].index_select(-1,dims).transpose(1,2)
    return out


def ring_rounds_probabilities(visible_tokens,qk_width,v_width,*,cuda):
    """Arithmetic of the current compact-ring eager route, before kernel launch.

    The torch dense AND gather fallbacks round probabilities. Only the admitted
    Triton gather branch keeps them FP32; sequence length alone is insufficient.
    The reference's global failure flag is respected on subsequent calls.
    """
    from . import eager_decoding as eager
    upper=min(int(os.environ.get('NFA_E127_W_UPPER','2048')),visible_tokens)
    fused=(os.environ.get('NFA_E127_FAST','1')!='0' and cuda
           and visible_tokens>=16384 and upper*8<=visible_tokens
           and eager._HAS_TRITON and eager._EAGER_TRITON_OK['ok']
           and qk_width==v_width and qk_width>0 and qk_width&(qk_width-1)==0)
    return not fused
def paged_attention(layer,q,mask,scale,*,ring_arithmetic=False):
    """q[B,Hq,1,D], mask[1,1,Hkv,L]; output[B,1,Hq,Dv]."""
    _,k,v=next(iter(layer.pages.values()))
    hq=q.shape[1];hk=k.shape[1]
    if q.shape[0]!=1 or q.shape[2]!=1 or hq%hk or q.shape[-1]!=k.shape[-1]:
        raise ValueError('paged attention requires B1 GQA single-query matching QK width')
    if q.dtype!=k.dtype or q.device!=k.device or mask.device!=q.device or mask.dtype!=torch.bool:
        raise ValueError('paged attention dtype/device mismatch')
    if mask.shape!=(1,1,hk,layer.logical_length):raise ValueError('invalid logical selection mask')
    selected=mask.reshape(hk,layer.logical_length)
    visible=torch.zeros(layer.logical_length,device=q.device,dtype=torch.bool)
    for a,b in layer.intervals():visible[a:b]=True
    if torch.any(selected & ~visible):raise ValueError('selection references hidden/reclaimed KV')
    counts=selected.sum(-1)
    if torch.any(counts==0):raise ValueError('empty attention row')
    # Ring eager always rounds scaled Q to activation dtype. Its small-context
    # dense branch also rounds normalized probabilities; mirror both explicitly.
    round_prob=ring_arithmetic and ring_rounds_probabilities(
        sum(b-a for a,b in layer.intervals()),k.shape[-1],v.shape[-1],cuda=q.is_cuda)
    if ring_arithmetic:
        q=q*scale
        scale=1.
    if q.is_cuda:
        positions=torch.arange(layer.logical_length,device=q.device).expand(hk,-1)
        candidates=positions.masked_fill(~selected,-1)
        count=int(counts.max())
        # 这里仅压缩已选位置，不是 L2 top-k；用排序避免与选择器 top-k 混淆。
        indices=candidates.sort(dim=-1,descending=True).values[:,:count].flip(-1).contiguous()
        out=q.new_empty((1,1,hq,v.shape[-1]))
        kp,vp=layer.pointer_tables()
        _attend[(hq,)](q.contiguous(),kp,vp,indices,out,hq,hk,k.shape[-1],v.shape[-1],layer.geometry.page,
            count,scale,round_prob,triton.next_power_of_2(k.shape[-1]),triton.next_power_of_2(v.shape[-1]),32)
        return out
    # CPU independent pagewise oracle: full K/V are never concatenated.
    outputs=[]
    for h in range(hq):
        kh=h//(hq//hk);pieces=[];m=q.new_tensor(float('-inf'),dtype=torch.float32)
        for lp,(_,keys,vals) in sorted(layer.pages.items()):
            a=lp*layer.geometry.page;b=min(a+layer.geometry.page,layer.logical_length)
            ids=selected[kh,a:b].nonzero().flatten()
            if not ids.numel():continue
            score=(keys[0,kh,ids].float()*q[0,h,0].float()).sum(-1)*scale
            pieces.append((score,vals[0,kh,ids]));m=torch.maximum(m,score.max())
        z=q.new_zeros((),dtype=torch.float32);acc=q.new_zeros(v.shape[-1],dtype=torch.float32)
        for score,vals in pieces:
            weights=(score-m).exp();z+=weights.sum();acc+=(weights[:,None]*vals.float()).sum(0)
        if round_prob:
            acc.zero_()
            for score,vals in pieces:
                weights=((score-m).exp()/z).to(q.dtype).float()
                acc+=(weights[:,None]*vals.float()).sum(0)
            z=1.
        outputs.append((acc/z).to(q.dtype))
    return torch.stack(outputs).reshape(1,1,hq,v.shape[-1])


def paged_attention_selected(layer,q,selection,scale,*,ring_arithmetic=False):
    """Compact selected-token page-table attention; no full-width bool mask.

    selection is a TokenSelection-like tuple ``(indices [1,Hkv,K], counts
    [1,Hkv])``.  ``indices`` may use either 0 or -1 as padding; this wrapper
    rewrites out-of-count slots to -1 so the Triton ``_attend`` kernel never
    reads token 0 for shorter heads.
    """
    indices, counts = selection
    _,k,v=next(iter(layer.pages.values()))
    hq=q.shape[1];hk=k.shape[1]
    if indices.ndim!=3 or indices.shape[0]!=1 or counts.ndim!=2 or counts.shape[0]!=1:
        raise ValueError('compact paged attention expects TokenSelection [1,Hkv,K]')
    if indices.shape[1]!=hk or counts.shape[1]!=hk:
        raise ValueError('compact selection KV-head count does not match paged layer')
    if q.shape[0]!=1 or q.shape[2]!=1 or hq%hk or q.shape[-1]!=k.shape[-1]:
        raise ValueError('paged attention requires B1 GQA single-query matching QK width')
    if q.dtype!=k.dtype or q.device!=k.device or indices.device!=q.device or counts.device!=q.device:
        raise ValueError('paged attention dtype/device mismatch')
    if indices.dtype!=torch.int32 or counts.dtype!=torch.int32:
        raise ValueError('compact selection must use int32 IDs/counts')
    length=layer.logical_length
    visible=torch.zeros(length,device=q.device,dtype=torch.bool)
    for a,b in layer.intervals():visible[a:b]=True
    ids=indices[0]
    width=ids.shape[-1]
    count=counts[0]
    active=torch.arange(width,device=ids.device)[None] < count[:,None]
    safe_ids=ids.clamp(0,length-1)
    if torch.any(active & ~visible[safe_ids]):
        raise ValueError('selection references hidden/reclaimed KV')
    if torch.any(count==0):raise ValueError('empty attention row')
    K=int(count.max())
    attn_ids=torch.where(torch.arange(K,device=ids.device)[None] < count[:,None],
                         ids[:,:K], -1).contiguous()
    round_prob=ring_arithmetic and ring_rounds_probabilities(
        sum(b-a for a,b in layer.intervals()),k.shape[-1],v.shape[-1],cuda=q.is_cuda)
    if ring_arithmetic:
        q=q*scale
        scale=1.
    if q.is_cuda:
        out=q.new_empty((1,1,hq,v.shape[-1]))
        kp,vp=layer.pointer_tables()
        _attend[(hq,)](q.contiguous(),kp,vp,attn_ids,out,hq,hk,k.shape[-1],v.shape[-1],layer.geometry.page,
            K,scale,round_prob,triton.next_power_of_2(k.shape[-1]),triton.next_power_of_2(v.shape[-1]),32)
        return out
    outputs=[]
    for h in range(hq):
        kh=h//(hq//hk);row_ids=attn_ids[kh];row_ids=row_ids[row_ids>=0]
        pieces=[];m=q.new_tensor(float('-inf'),dtype=torch.float32)
        for lp,(_,keys,vals) in sorted(layer.pages.items()):
            a=lp*layer.geometry.page;b=min(a+layer.geometry.page,length)
            ids=row_ids[(row_ids>=a)&(row_ids<b)]-a
            if not ids.numel():continue
            score=(keys[0,kh,ids].float()*q[0,h,0].float()).sum(-1)*scale
            pieces.append((score,vals[0,kh,ids]));m=torch.maximum(m,score.max())
        z=q.new_zeros((),dtype=torch.float32);acc=q.new_zeros(v.shape[-1],dtype=torch.float32)
        for score,vals in pieces:
            weights=(score-m).exp();z+=weights.sum();acc+=(weights[:,None]*vals.float()).sum(0)
        if round_prob:
            acc.zero_()
            for score,vals in pieces:
                weights=((score-m).exp()/z).to(q.dtype).float()
                acc+=(weights[:,None]*vals.float()).sum(0)
            z=1.
        outputs.append((acc/z).to(q.dtype))
    return torch.stack(outputs).reshape(1,1,hq,v.shape[-1])


if triton is not None:
    @triton.jit
    def _paged_write_kernel(
        KEYS, VALUES, SLOTS, K_POOL, V_POOL,
        B,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        PAD_SLOT_SAFE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
    ):
        b = tl.program_id(0)
        if b >= B:
            return
        slot = tl.load(SLOTS + b)
        if slot < 0:
            return
        safe = tl.where(slot >= 0, slot, PAD_SLOT_SAFE)
        phys = (safe // PAGE).to(tl.int64)
        off = (safe % PAGE).to(tl.int64)

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        for h in tl.static_range(HK):
            key = tl.load(
                KEYS + b * HK * D + h * D + d,
                mask=d < D, other=0.0)
            k_ptr = (
                K_POOL
                + phys * (HK * PAGE * D)
                + h * (PAGE * D)
                + off * D
                + d
            )
            tl.store(k_ptr, key, mask=d < D)

            val = tl.load(
                VALUES + b * HK * DV + h * DV + vd,
                mask=vd < DV, other=0.0)
            v_ptr = (
                V_POOL
                + phys * (HK * PAGE * DV)
                + h * (PAGE * DV)
                + off * DV
                + vd
            )
            tl.store(v_ptr, val, mask=vd < DV)


    @triton.jit
    def _gather_attend_kernel(
        Q,
        K_POOL,
        V_POOL,
        GATHER,          # [B, MAX_VISIBLE] absolute physical slots
        VISIBLE_LENS,    # [B]
        QUERY_TO_REQ,    # [TOTAL_QUERIES]
        OUT,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        LOG2_PAGE: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
        USE_BF16: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        # 2D grid: one program per (flat query row, query head).  This avoids
        # the flattened pid division/modulo on the inner path and keeps page
        # loads private to each q/head program.
        q_idx = tl.program_id(0)
        head = tl.program_id(1)
        if q_idx >= TOTAL_QUERIES:
            return

        # Host precomputes q_idx -> req_idx, eliminating the per-program
        # linear scan over query_start_loc in the inner kernel.
        req_idx = tl.load(QUERY_TO_REQ + q_idx, cache_modifier='.cg')
        kv_head = head // (HQ // HK)

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)

        if req_idx < 0:
            # Padded batch rows have no request slot.  Their q rows are never
            # consumed by the scheduler, but write zeros to keep graph replay
            # deterministic and avoid an out-of-bounds VISIBLE_LENS load.
            tl.store(
                OUT + q_idx * HQ * DV + head * DV + vd,
                tl.zeros((BV,), tl.float32).to(OUT.dtype.element_ty),
                mask=vd < DV,
            )
            return
        vis_len = tl.load(VISIBLE_LENS + req_idx, cache_modifier='.cg')

        # Keep Q and K in the KV dtype for tl.dot.  bf16 and fp16 both map to
        # Tensor Core MMA; the online softmax accumulators stay in fp32.
        q_vec = tl.load(
            Q + q_idx * HQ * D + head * D + d,
            mask=d < D, other=0.0, cache_modifier='.ca')
        if USE_DOT:
            if USE_BF16:
                q_vec = q_vec.to(tl.bfloat16)
            else:
                q_vec = q_vec.to(tl.float16)
            # tl.dot requires N >= 16.  Replicate the single query vector to
            # a [BD, 16] tile and reduce the identical columns afterwards.
            q_tile = tl.broadcast_to(
                tl.reshape(q_vec, (BD, 1)), (BD, 16))
        else:
            q_vec = q_vec.to(tl.float32)

        m = tl.full((), float('-inf'), tl.float32)
        z = tl.zeros((), tl.float32)
        acc = tl.zeros((BV,), tl.float32)

        # Triton 3.4 does not accept ``break`` inside ``tl.range``.  The
        # equivalent dynamic upper bound below stops at the first block beyond
        # this request's actual visible length, so short requests do not ride
        # along with the fixed MAX_VISIBLE tail.
        num_blocks = tl.cdiv(tl.minimum(vis_len, MAX_VISIBLE), BT)
        for block_idx in range(0, num_blocks):
            start = block_idx * BT
            col = tl.max_contiguous(start + t, BT)
            col_valid = col < vis_len
            slot = tl.load(
                GATHER + req_idx * MAX_VISIBLE + col,
                mask=col_valid, other=-1, cache_modifier='.cg')
            valid = col_valid & (slot >= 0)

            phys_page = (slot >> LOG2_PAGE).to(tl.int64)
            offset = slot & (PAGE - 1)

            k_base = (
                K_POOL
                + phys_page * (HK * PAGE * D)
                + kv_head * (PAGE * D)
                + offset * D
            )
            v_base = (
                V_POOL
                + phys_page * (HK * PAGE * DV)
                + kv_head * (PAGE * DV)
                + offset * DV
            )

            k = tl.load(
                k_base[:, None] + d[None, :],
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0, cache_modifier='.ca')
            if USE_DOT:
                if USE_BF16:
                    k = k.to(tl.bfloat16)
                else:
                    k = k.to(tl.float16)
                score_tile = tl.dot(k, q_tile)
                score = tl.max(score_tile, 1) * SCALE
            else:
                k = k.to(tl.float32)
                score = tl.sum(k * q_vec[None, :], axis=1) * SCALE
            score = tl.where(valid, score, float('-inf'))

            new_m = tl.maximum(m, tl.max(score, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score - safe_m)

            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0, cache_modifier='.ca').to(tl.float32)

            acc = acc * alpha + tl.sum(v * prob[:, None], axis=0)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        out = tl.where(z == 0.0, 0.0, acc / z)
        tl.store(
            OUT + q_idx * HQ * DV + head * DV + vd,
            out.to(OUT.dtype.element_ty),
            mask=vd < DV,
        )



    @triton.jit
    def _gather_attend_grouped_kernel(
        Q,
        K_POOL,
        V_POOL,
        GATHER,          # [B, MAX_VISIBLE] absolute physical slots
        VISIBLE_LENS,    # [B]
        QUERY_TO_REQ,    # [TOTAL_QUERIES]
        OUT,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        LOG2_PAGE: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
        HEAD_GROUP: tl.constexpr,
        HEAD_TILE: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        """Grouped Tensor-Core gather attention.

        One program owns one ``(query row, KV head)`` pair and computes all
        ``HEAD_GROUP`` query heads that share that KV head.  The Q tile is
        therefore ``[BD, HEAD_TILE]``, where ``HEAD_TILE`` is padded to at
        least 16 so both QK and PV use MMA instructions without the previous
        per-head 16-column replication.
        """
        q_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        if q_idx >= TOTAL_QUERIES:
            return

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)
        h = tl.arange(0, HEAD_TILE)

        req_idx = tl.load(QUERY_TO_REQ + q_idx, cache_modifier='.cg')
        out_base = OUT + q_idx * HQ * DV + kv_head * HEAD_GROUP * DV
        out_ptrs = out_base + h[:, None] * DV + vd[None, :]
        out_mask = (h[:, None] < HEAD_GROUP) & (vd[None, :] < DV)

        if req_idx < 0:
            # Padded batch row: no request slot and no KV gather.  Zero the
            # output tile deterministically and return before touching
            # VISIBLE_LENS at a negative index.
            tl.store(
                out_ptrs,
                tl.zeros((HEAD_TILE, BV), tl.float32).to(OUT.dtype.element_ty),
                mask=out_mask,
            )
            return

        vis_len = tl.load(VISIBLE_LENS + req_idx, cache_modifier='.cg')

        q_ptrs = Q + q_idx * HQ * D + d[:, None] + (kv_head * HEAD_GROUP + h)[None, :] * D
        q_tile = tl.load(
            q_ptrs,
            mask=(d[:, None] < D) & (h[None, :] < HEAD_GROUP),
            other=0.0,
            cache_modifier='.ca',
        )
        if USE_BF16:
            q_tile = q_tile.to(tl.bfloat16)
        else:
            q_tile = q_tile.to(tl.float16)

        m = tl.full((HEAD_TILE,), float('-inf'), tl.float32)
        z = tl.zeros((HEAD_TILE,), tl.float32)
        acc = tl.zeros((HEAD_TILE, BV), tl.float32)

        # Dynamic upper bound is the functional replacement for ``break`` on
        # Triton 3.4: a short request never walks the fixed MAX_VISIBLE tail.
        num_blocks = tl.cdiv(tl.minimum(vis_len, MAX_VISIBLE), BT)
        for block_idx in range(0, num_blocks):
            start = block_idx * BT
            col = tl.max_contiguous(start + t, BT)
            col_valid = col < vis_len
            slot = tl.load(
                GATHER + req_idx * MAX_VISIBLE + col,
                mask=col_valid, other=-1, cache_modifier='.cg')
            valid = col_valid & (slot >= 0)

            phys_page = (slot >> LOG2_PAGE).to(tl.int64)
            offset = slot & (PAGE - 1)

            k_base = (
                K_POOL
                + phys_page * (HK * PAGE * D)
                + kv_head * (PAGE * D)
                + offset * D
            )
            v_base = (
                V_POOL
                + phys_page * (HK * PAGE * DV)
                + kv_head * (PAGE * DV)
                + offset * DV
            )

            k = tl.load(
                k_base[:, None] + d[None, :],
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0, cache_modifier='.ca')
            if USE_BF16:
                k = k.to(tl.bfloat16)
            else:
                k = k.to(tl.float16)

            score_tile = tl.dot(k, q_tile) * SCALE
            score_valid = valid[:, None] & (h[None, :] < HEAD_GROUP)
            score_tile = tl.where(score_valid, score_tile, float('-inf'))

            new_m = tl.maximum(m, tl.max(score_tile, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score_tile - safe_m[None, :])

            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0, cache_modifier='.ca').to(tl.float32)

            acc = acc * alpha[:, None] + tl.dot(tl.trans(prob), v)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        out = tl.where(z[:, None] == 0.0, 0.0, acc / z[:, None])
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=out_mask)


def paged_write_decode(cache_layer, key_states, value_states, slot_mapping):
    """Write one decode token per batch row into a fixed physical arena.

    ``key_states`` / ``value_states`` are [B, HK, 1, D] / [B, HK, 1, DV].
    ``slot_mapping`` is [B] int32.  Invalid slots (-1) are routed to the PAD
    page so writes are safe discards and the kernel has no data-dependent
    branch.
    """
    if key_states.ndim != 4 or key_states.shape[-2] != 1:
        raise ValueError('paged write decode expects [B,HK,1,D] keys')
    if value_states.ndim != 4 or value_states.shape[-2] != 1:
        raise ValueError('paged write decode expects [B,HK,1,DV] values')
    if key_states.shape[0] != slot_mapping.shape[0]:
        raise ValueError('slot mapping batch must match KV batch')
    if key_states.shape[1] != cache_layer.num_kv_heads:
        raise ValueError('KV head count does not match arena')
    if key_states.device != slot_mapping.device:
        slot_mapping = slot_mapping.to(key_states.device)

    B = key_states.shape[0]
    HK = cache_layer.num_kv_heads
    D = cache_layer.head_dim_k
    DV = cache_layer.head_dim_v
    PAGE = cache_layer.geometry.page
    PAD_SLOT_SAFE = cache_layer.PAD_PAGE_ID * PAGE

    if key_states.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for paged write decode')
        _paged_write_kernel[(B,)](
            key_states,
            value_states,
            slot_mapping.contiguous(),
            cache_layer.kv_pool_k,
            cache_layer.kv_pool_v,
            B,
            HK,
            D,
            DV,
            PAGE,
            PAD_SLOT_SAFE,
            triton.next_power_of_2(D),
            triton.next_power_of_2(DV),
        )
        return key_states, value_states

    # CPU reference: PAD writes are intentionally allowed and discarded.
    for b in range(B):
        slot = int(slot_mapping[b])
        if slot < 0:
            slot = PAD_SLOT_SAFE
        physical_id, offset = divmod(slot, PAGE)
        cache_layer.kv_pool_k[physical_id, 0, :, offset, :] = key_states[b, :, 0, :]
        cache_layer.kv_pool_v[physical_id, 0, :, offset, :] = value_states[b, :, 0, :]
    return key_states, value_states


def sparse_paged_attention_gather(
    cache_layer,
    q,
    gather_indices,
    visible_lens,
    query_to_req,
    *,
    scale,
):
    """Sparse gather paged attention over a GraphReadyPagedRegionLayer arena.

    ``q`` is flat [T, HQ, D].  ``gather_indices`` is [B, MAX_VISIBLE] absolute
    physical slots (``physical_page * PAGE + offset``), ``visible_lens`` is [B],
    and ``query_to_req`` is [T] mapping each flat query row to its request slot.
    The kernel performs no block-table lookup and no integer division.
    """
    if q.ndim != 3 or q.shape[-1] != cache_layer.head_dim_k:
        raise ValueError('sparse gather attention expects flat [T,HQ,D] query')

    hq = q.shape[1]
    hk = cache_layer.num_kv_heads
    if hq % hk != 0:
        raise ValueError('GQA requires HQ % HK == 0')
    total_q = q.shape[0]
    max_visible = gather_indices.shape[1]
    page = cache_layer.geometry.page
    log2_page = page.bit_length() - 1
    if (page & (page - 1)) != 0:
        raise ValueError('sparse gather attention requires power-of-two page size')
    use_bf16 = cache_layer.kv_pool_k.dtype == torch.bfloat16
    head_group = hq // hk
    head_tile = max(16, triton.next_power_of_2(head_group)) if triton is not None else 16
    use_grouped = (
        cache_layer.head_dim_k >= 16
        and cache_layer.head_dim_v >= 16
        and head_group <= head_tile
    )

    if q.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for sparse gather attention')
        out = q.new_empty((total_q, hq, cache_layer.head_dim_v))
        if use_grouped:
            # Small batches are latency-bound and prefer a wider token tile;
            # larger batches are occupancy-bound and prefer BT=64.
            block_tokens = 128 if total_q <= 8 else 64
            _gather_attend_grouped_kernel[(total_q, hk)](
                q.contiguous(),
                cache_layer.kv_pool_k,
                cache_layer.kv_pool_v,
                gather_indices,
                visible_lens,
                query_to_req,
                out,
                total_q,
                hq, hk,
                cache_layer.head_dim_k,
                cache_layer.head_dim_v,
                page,
                log2_page,
                max_visible,
                visible_lens.shape[0],
                block_tokens,
                scale,
                triton.next_power_of_2(cache_layer.head_dim_k),
                triton.next_power_of_2(cache_layer.head_dim_v),
                head_group,
                head_tile,
                use_bf16,
            )
        else:
            # Small-head fallback keeps the scalar-FMA path because tl.dot
            # requires N >= 16 and would otherwise fabricate dummy columns.
            _gather_attend_kernel[(total_q, hq)](
                q.contiguous(),
                cache_layer.kv_pool_k,
                cache_layer.kv_pool_v,
                gather_indices,
                visible_lens,
                query_to_req,
                out,
                total_q,
                hq, hk,
                cache_layer.head_dim_k,
                cache_layer.head_dim_v,
                page,
                log2_page,
                max_visible,
                visible_lens.shape[0],
                32,
                scale,
                triton.next_power_of_2(cache_layer.head_dim_k),
                triton.next_power_of_2(cache_layer.head_dim_v),
                use_bf16,
                False,
            )
        return out

    # CPU reference: physical-slot gather, loops only visible_lens slots.
    page = cache_layer.geometry.page
    out = q.new_zeros((total_q, hq, cache_layer.head_dim_v))
    for q_idx in range(total_q):
        req_idx = int(query_to_req[q_idx])
        if req_idx < 0:
            # Padded batch rows have no request slot; leave their output zero.
            continue
        vis_len = int(visible_lens[req_idx].item())
        for h in range(hq):
            kh = h // (hq // hk)
            m = float('-inf')
            z = 0.0
            acc = torch.zeros(cache_layer.head_dim_v, dtype=torch.float32,
                              device=q.device)
            for col in range(vis_len):
                slot = int(gather_indices[req_idx, col].item())
                if slot < 0:
                    continue
                phys, offset = divmod(slot, page)
                k = cache_layer.kv_pool_k[phys, 0, kh, offset].float()
                v = cache_layer.kv_pool_v[phys, 0, kh, offset].float()
                score = (q[q_idx, h].float() * k).sum() * scale
                new_m = max(m, score)
                if new_m == float('-inf'):
                    continue
                alpha = math.exp(m - new_m)
                prob = math.exp(score - new_m)
                acc = acc * alpha + prob * v
                z = z * alpha + prob
                m = new_m
            if z != 0:
                out[q_idx, h] = (acc / z).to(q.dtype)
    return out
