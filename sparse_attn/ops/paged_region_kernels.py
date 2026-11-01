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
    upper=min(int(os.environ.get('TLI_E127_W_UPPER','2048')),visible_tokens)
    fused=(os.environ.get('TLI_E127_FAST','1')!='0' and cuda
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
        GATHER,          # [B, MAX_VISIBLE]
        VISIBLE_LENS,    # [B]
        BLOCK_TABLE,     # [B, MAX_BLOCKS]
        QUERY_START_LOC, # [B+1]
        OUT,
        TOTAL_QUERIES,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        PAGE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        MAX_VISIBLE: tl.constexpr,
        MAX_BATCH: tl.constexpr,
        BT: tl.constexpr,
        SCALE: tl.constexpr,
        BD: tl.constexpr,
        BV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid // HQ
        head = pid % HQ
        if q_idx >= TOTAL_QUERIES:
            return

        # 定位 q_idx 属于哪个请求。B 很小，固定循环不会引入动态 shape。
        req_idx = -1
        for r in range(MAX_BATCH):
            lo = tl.load(QUERY_START_LOC + r)
            hi = tl.load(QUERY_START_LOC + r + 1)
            if (q_idx >= lo) and (q_idx < hi):
                req_idx = r

        vis_len = tl.load(VISIBLE_LENS + req_idx)
        kv_head = head // (HQ // HK)

        d = tl.arange(0, BD)
        vd = tl.arange(0, BV)
        t = tl.arange(0, BT)

        q = tl.load(
            Q + q_idx * HQ * D + head * D + d,
            mask=d < D, other=0.0).to(tl.float32)

        m = tl.full((), float('-inf'), tl.float32)
        z = tl.zeros((), tl.float32)
        acc = tl.zeros((BV,), tl.float32)

        # 循环上界是固定 MAX_VISIBLE；每列再用 vis_len 屏蔽 padding。
        for start in range(0, MAX_VISIBLE, BT):
            col = start + t
            col_valid = col < vis_len
            logical_pos = tl.load(
                GATHER + req_idx * MAX_VISIBLE + col,
                mask=col_valid, other=-1)
            valid = col_valid & (logical_pos >= 0)

            logical_page = logical_pos // PAGE
            offset = logical_pos % PAGE
            phys_page = tl.load(
                BLOCK_TABLE + req_idx * MAX_BLOCKS + logical_page,
                mask=valid, other=0)
            # 不分支判断 phys_page 是否 PAD。业务路径 phys_page 合法；
            # invalid 时 mask 已为 False，不会真正读 K/V。
            phys_page = phys_page.to(tl.int64)

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
                other=0.0).to(tl.float32)
            score = tl.sum(k * q[None, :], axis=1) * SCALE
            score = tl.where(valid, score, float('-inf'))

            new_m = tl.maximum(m, tl.max(score, 0))
            safe_m = tl.where(new_m == float('-inf'), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            prob = tl.exp(score - safe_m)

            v = tl.load(
                v_base[:, None] + vd[None, :],
                mask=valid[:, None] & (vd[None, :] < DV),
                other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(v * prob[:, None], axis=0)
            z = z * alpha + tl.sum(prob, 0)
            m = new_m

        out = tl.where(z == 0.0, 0.0, acc / z)
        tl.store(
            OUT + q_idx * HQ * DV + head * DV + vd,
            out.to(OUT.dtype.element_ty),
            mask=vd < DV,
        )


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
    block_table,
    query_start_loc,
    *,
    scale,
):
    """Sparse gather paged attention over a GraphReadyPagedRegionLayer arena.

    ``q`` is flat [T, HQ, D]. ``gather_indices`` is [B, MAX_VISIBLE] logical
    token positions and ``visible_lens`` is [B]. The kernel never loops over
    seq_len; it loops over MAX_VISIBLE and masks columns with visible_lens.
    """
    if q.ndim != 3 or q.shape[-1] != cache_layer.head_dim_k:
        raise ValueError('sparse gather attention expects flat [T,HQ,D] query')

    hq = q.shape[1]
    hk = cache_layer.num_kv_heads
    if hq % hk != 0:
        raise ValueError('GQA requires HQ % HK == 0')
    total_q = q.shape[0]
    max_visible = gather_indices.shape[1]

    if q.is_cuda:
        if triton is None:
            raise RuntimeError('Triton required for sparse gather attention')
        out = q.new_empty((total_q, hq, cache_layer.head_dim_v))
        _gather_attend_kernel[(total_q * hq,)](
            q.contiguous(),
            cache_layer.kv_pool_k,
            cache_layer.kv_pool_v,
            gather_indices,
            visible_lens,
            block_table,
            query_start_loc,
            out,
            total_q,
            hq, hk,
            cache_layer.head_dim_k,
            cache_layer.head_dim_v,
            cache_layer.geometry.page,
            block_table.shape[1],
            max_visible,
            visible_lens.shape[0],
            32,
            scale,
            triton.next_power_of_2(cache_layer.head_dim_k),
            triton.next_power_of_2(cache_layer.head_dim_v),
        )
        return out

    # CPU reference: still pagewise, still loops only visible_lens tokens.
    page = cache_layer.geometry.page
    out = q.new_zeros((total_q, hq, cache_layer.head_dim_v))
    for q_idx in range(total_q):
        req_idx = int(torch.searchsorted(
            query_start_loc, q_idx, right=True).item()) - 1
        vis_len = int(visible_lens[req_idx].item())
        for h in range(hq):
            kh = h // (hq // hk)
            m = float('-inf')
            z = 0.0
            acc = torch.zeros(cache_layer.head_dim_v, dtype=torch.float32,
                              device=q.device)
            for col in range(vis_len):
                logical_pos = int(gather_indices[req_idx, col].item())
                if logical_pos < 0:
                    continue
                lp, offset = divmod(logical_pos, page)
                phys = int(block_table[req_idx, lp].item())
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
