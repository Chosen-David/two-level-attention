"""Opt-in ragged-KV batch attention. Pointer tables are request-owned and pinned."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attend(Q, META, IDS, VALID, ALIVE, OUT,
            QB: tl.constexpr, QH: tl.constexpr,
            H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            W: tl.constexpr, SCALE: tl.constexpr, TILE: tl.constexpr):
    bh = tl.program_id(0)
    b, qh = bh // (H*G), bh % (H*G)
    h = qh // G
    d = tl.arange(0, D)
    acc = tl.full((D,), 0, tl.float32)
    denom = tl.full((), 0, tl.float32)
    if tl.load(ALIVE+b):
        kp = tl.load(META+b*8).to(tl.pointer_type(Q.dtype.element_ty))
        vp = tl.load(META+b*8+1).to(tl.pointer_type(Q.dtype.element_ty))
        kh, kt = tl.load(META+b*8+2), tl.load(META+b*8+3)
        vh, vt = tl.load(META+b*8+4), tl.load(META+b*8+5)
        sink, tail = tl.load(META+b*8+6), tl.load(META+b*8+7)
        q = tl.load(Q+b*QB+qh*QH+d).to(tl.float32)*SCALE
        maximum = tl.full((), float('-inf'), tl.float32)
        for start in range(tl.cdiv(W,TILE)):
            rank = start*TILE+tl.arange(0,TILE)
            ok = tl.load(VALID+(b*H+h)*W+rank, rank<W, other=0)
            ids = tl.load(IDS+(b*H+h)*W+rank, rank<W, other=0).to(tl.int64)
            physical = ids
            if tail > 0:
                physical = tl.where(ids<sink,ids,sink+(ids-sink)%tail)
            k = tl.load(kp+h*kh+physical[:,None]*kt+d[None,:], ok[:,None], other=0).to(tl.float32)
            v = tl.load(vp+h*vh+physical[:,None]*vt+d[None,:], ok[:,None], other=0).to(tl.float32)
            score = tl.where(ok,tl.sum(k*q[None,:],1),float('-inf'))
            next_max = tl.maximum(maximum,tl.max(score,0))
            safe_max = tl.where(next_max==float('-inf'),0.,next_max)
            alpha = tl.exp(tl.where(maximum==float('-inf'),0.,maximum)-safe_max)
            prob = tl.where(ok,tl.exp(score-safe_max),0.)
            acc = acc*alpha+tl.sum(prob[:,None]*v,0)
            denom = denom*alpha+tl.sum(prob,0)
            maximum = next_max
    tl.store(OUT+bh*D+d,acc/tl.where(denom==0,1.,denom))


class RaggedAttentionPlan:
    def __init__(self, rows):
        self.rows = tuple(rows)  # lifetime pin for raw device addresses
        self.signature = self._signature(rows)
        entries=[]
        for row in rows:
            if row.keys.stride(-1)!=1 or row.values.stride(-1)!=1:
                raise ValueError('ragged kernel requires unit feature strides')
            layout=getattr(row,'layout',None)
            entries.append([row.keys.data_ptr(),row.values.data_ptr(),
                row.keys.stride(1),row.keys.stride(2),row.values.stride(1),row.values.stride(2),
                layout.sink if layout else 0,layout.tail_capacity if layout else 0])
        self.meta=torch.tensor(entries,device=rows[0].keys.device,dtype=torch.int64)

    @staticmethod
    def _signature(rows):
        return tuple((r.keys.data_ptr(),r.values.data_ptr(),r.keys.shape,r.values.shape,
                      r.keys.stride(),r.values.stride(),r.keys.dtype,r.values.dtype,
                      r.keys.device,r.values.device,getattr(r,'layout',None)) for r in rows)

    def __call__(self,q,rows,selections,scale,alive):
        if self.signature!=self._signature(rows):
            raise ValueError('ragged KV storage changed; rebuild plan and recapture')
        b,_,hq,d=q.shape
        h=rows[0].keys.shape[1]
        if (not q.is_cuda or d<1 or d & (d-1) or q.stride(-1)!=1
                or any(r.keys.dtype!=q.dtype or r.values.dtype!=q.dtype for r in rows)):
            raise ValueError('ragged attention requires CUDA, matching dtype and power-of-two D')
        width=max(ids.shape[-1] for ids,valid in selections)
        ids=torch.stack([torch.nn.functional.pad(i,(0,width-i.shape[-1])) for i,v in selections]).contiguous()
        valid=torch.stack([torch.nn.functional.pad(v,(0,width-v.shape[-1])) for i,v in selections]).contiguous()
        out=torch.empty((b,1,hq,d),dtype=q.dtype,device=q.device)
        _attend[(b*hq,)](q,self.meta,ids,valid,alive,out,q.stride(0),q.stride(2),
            h,hq//h,d,width,scale,64,num_warps=4,enable_fp_fusion=False)
        return out
