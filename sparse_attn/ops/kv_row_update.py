"""One-row append without temporary gather buffers; dead rows do no writes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _append(K, V, NK, NV, SLOT, ALIVE, CAP: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr,
            VH: tl.constexpr, VT: tl.constexpr, NH: tl.constexpr,
            NVH: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    slot = tl.load(SLOT)
    live = tl.load(ALIVE) & (slot >= 0) & (slot < CAP)
    kval = tl.load(NK + h * NH + d, live & (d < D), other=0)
    vval = tl.load(NV + h * NVH + d, live & (d < D), other=0)
    tl.store(K + h * KH + slot * KT + d, kval, live & (d < D))
    tl.store(V + h * VH + slot * VT + d, vval, live & (d < D))


def append_kv_row(k, v, new_k, new_v, slot, alive):
    """K/V [1,H,capacity,D], incoming [1,H,1,D], slot/alive device scalars.

    The caller validates slot geometry. Bounds are also masked at the store;
    this routine never reads old KV, including for dead rows.
    """
    if (k.ndim != 4 or k.shape[0] != 1 or v.shape != k.shape
            or new_k.shape != (1, k.shape[1], 1, k.shape[3]) or new_v.shape != new_k.shape
            or any(t.device != k.device for t in (v, new_k, new_v, slot, alive))
            or any(t.dtype != k.dtype or t.stride(-1) != 1 for t in (v, new_k, new_v))
            or k.stride(-1) != 1 or slot.numel() != 1 or alive.numel() != 1
            or slot.dtype != torch.int64 or alive.dtype != torch.bool):
        raise ValueError('invalid one-row KV append metadata')
    if not k.is_cuda:
        for storage, incoming in ((k, new_k), (v, new_v)):
            old = storage.index_select(2, slot)
            storage.index_copy_(2, slot, torch.where(alive.view(1,1,1,1), incoming, old))
        return
    _append[(k.shape[1],)](k, v, new_k, new_v, slot, alive,
        k.shape[2], k.shape[1], k.shape[3], k.stride(1), k.stride(2),
        v.stride(1), v.stride(2), new_k.stride(1), new_v.stride(1),
        BLOCK=triton.next_power_of_2(k.shape[3]), num_warps=4)


@triton.jit
def _append_batch(K,V,NK,NV,SLOT,ALIVE,CAP:tl.constexpr,H:tl.constexpr,
                  D:tl.constexpr,KB:tl.constexpr,KH:tl.constexpr,KT:tl.constexpr,
                  VB:tl.constexpr,VH:tl.constexpr,VT:tl.constexpr,
                  NB:tl.constexpr,NH:tl.constexpr,NVB:tl.constexpr,NVH:tl.constexpr,
                  BLOCK:tl.constexpr):
    b,h=tl.program_id(0),tl.program_id(1)
    d=tl.arange(0,BLOCK)
    slot=tl.load(SLOT+b)
    live=tl.load(ALIVE+b)&(slot>=0)&(slot<CAP)
    kval=tl.load(NK+b*NB+h*NH+d,live&(d<D),other=0)
    vval=tl.load(NV+b*NVB+h*NVH+d,live&(d<D),other=0)
    tl.store(K+b*KB+h*KH+slot*KT+d,kval,live&(d<D))
    tl.store(V+b*VB+h*VH+slot*VT+d,vval,live&(d<D))


def append_kv_batch(k,v,new_k,new_v,slots,alive):
    """Append one token per row of [B,H,C,D]; no reads/writes for dead rows.

    Caller owns slot policy (the static cache clamps before dispatch). Invalid
    slots are masked for both backends; feature strides must be one.
    """
    if (k.ndim!=4 or v.shape!=k.shape or new_k.shape!=(k.shape[0],k.shape[1],1,k.shape[3])
            or new_v.shape!=new_k.shape or slots.shape!=(k.shape[0],) or alive.shape!=slots.shape
            or slots.dtype!=torch.int64 or alive.dtype!=torch.bool
            or not slots.is_contiguous() or not alive.is_contiguous()
            or any(t.device!=k.device for t in (v,new_k,new_v,slots,alive))
            or any(t.dtype!=k.dtype or t.stride(-1)!=1 for t in (k,v,new_k,new_v))):
        raise ValueError('invalid batched KV append metadata')
    if not k.is_cuda:
        for b in range(k.shape[0]):
            slot=int(slots[b])
            if bool(alive[b]) and 0<=slot<k.shape[2]:
                k[b,:,slot:slot+1].copy_(new_k[b])
                v[b,:,slot:slot+1].copy_(new_v[b])
        return
    if not k.numel():
        return
    _append_batch[(k.shape[0],k.shape[1])](k,v,new_k,new_v,slots,alive,
        k.shape[2],k.shape[1],k.shape[3],k.stride(0),k.stride(1),k.stride(2),
        v.stride(0),v.stride(1),v.stride(2),new_k.stride(0),new_k.stride(1),
        new_v.stride(0),new_v.stride(1),BLOCK=triton.next_power_of_2(k.shape[3]),num_warps=4)
