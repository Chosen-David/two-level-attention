"""Page-table projection and selected-token attention; no contiguous KV history.

CPU uses a pagewise two-pass softmax oracle. CUDA reads full K/V directly through
page pointer tables. Only scalar selection indices and low-dimensional index
scratch are materialized. This eager implementation does not support Graphs.
"""
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
