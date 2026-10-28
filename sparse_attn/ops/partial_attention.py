"""Split-KV decode with logical-ID gathers and disjoint protected attention.

SIMT is the portable GPU baseline. The optional TC kernel shares K/V across a
GQA group; TC padding and probability rounding require separate acceptance.
Both support Dqk != Dv. No selected K/V tensor is materialized on the GPU path.
"""
from typing import NamedTuple
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


class PartialState(NamedTuple):
    m: torch.Tensor       # [B,Hq,splits], FP32
    l: torch.Tensor       # [B,Hq,splits], FP32
    acc: torch.Tensor     # [B,Hq,splits,Dv], FP32


if triton is not None:
    @triton.jit
    def _positions(IDS, COUNT, LEN, b, kv, offsets,
                   HK: tl.constexpr, WIDTH: tl.constexpr, PROTECTED: tl.constexpr,
                   SINK: tl.constexpr, SWA: tl.constexpr,
                   RING_SINK: tl.constexpr, RING_TAIL: tl.constexpr,
                   CAP: tl.constexpr):
        length = tl.load(LEN + b)
        prefix = tl.minimum(length, SINK)
        start = tl.maximum(prefix, length - SWA)
        if PROTECTED:
            count = prefix + length - start
            logical = tl.where(offsets < prefix, offsets, start + offsets - prefix)
            valid = (offsets < count) & (offsets < WIDTH)
        else:
            count = tl.load(COUNT + b * HK + kv)
            valid = (offsets < count) & (offsets < WIDTH)
            logical = tl.load(IDS + (b * HK + kv) * WIDTH + offsets,
                              mask=valid, other=0)
            valid = valid & (logical >= prefix) & (logical < start)
        valid = valid & (logical >= 0) & (logical < length)
        if RING_TAIL > 0:
            # Reject evicted positions rather than aliasing stale ring slots.
            valid = valid & ((logical < RING_SINK) |
                             (logical >= tl.maximum(RING_SINK, length - RING_TAIL)))
            physical = tl.where(logical < RING_SINK, logical,
                                RING_SINK + (logical - RING_SINK) % RING_TAIL)
        else:
            physical = logical
        valid = valid & (physical >= 0) & (physical < CAP)
        return tl.where(valid, physical, 0), valid

    @triton.jit
    def _partial_simt(Q, K, V, IDS, COUNT, LEN, M, L, A,
                      QSB: tl.constexpr, QSH: tl.constexpr,
                      KSB: tl.constexpr, KST: tl.constexpr, KSH: tl.constexpr,
                      VSB: tl.constexpr, VST: tl.constexpr, VSH: tl.constexpr,
                      HQ: tl.constexpr, HK: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
                      DQB: tl.constexpr, DVB: tl.constexpr, CAP: tl.constexpr,
                      WIDTH: tl.constexpr, SPLITS: tl.constexpr, SPAN: tl.constexpr,
                      PROTECTED: tl.constexpr, SINK: tl.constexpr, SWA: tl.constexpr,
                      RING_SINK: tl.constexpr, RING_TAIL: tl.constexpr, TILE: tl.constexpr):
        row, split = tl.program_id(0), tl.program_id(1)
        b, qh = row // HQ, row % HQ
        kv = qh // (HQ // HK)
        dq, dv = tl.arange(0, DQB), tl.arange(0, DVB)
        q = tl.load(Q + b * QSB + qh * QSH + dq, dq < DQ, 0).to(tl.float32)
        m, l = float('-inf'), 0.0
        acc = tl.zeros((DVB,), tl.float32)
        for t in range(0, SPAN, TILE):
            offsets = split * SPAN + t + tl.arange(0, TILE)
            tok, valid = _positions(IDS, COUNT, LEN, b, kv, offsets, HK, WIDTH,
                                    PROTECTED, SINK, SWA, RING_SINK, RING_TAIL, CAP)
            kk = tl.load(K + b * KSB + tok[:, None] * KST + kv * KSH + dq[None, :],
                         valid[:, None] & (dq[None, :] < DQ), 0).to(tl.float32)
            vv = tl.load(V + b * VSB + tok[:, None] * VST + kv * VSH + dv[None, :],
                         valid[:, None] & (dv[None, :] < DV), 0).to(tl.float32)
            scores = tl.where(valid, tl.sum(kk * q[None, :], 1), float('-inf'))
            new_m = tl.maximum(m, tl.max(scores, 0))
            safe_m = tl.where(new_m == float('-inf'), 0., new_m)
            alpha = tl.exp(m - safe_m)
            p = tl.where(valid, tl.exp(scores - safe_m), 0.)
            acc = acc * alpha + tl.sum(p[:, None] * vv, 0)
            l = l * alpha + tl.sum(p, 0)
            m = new_m
        ix = row * SPLITS + split
        tl.store(M + ix, m)
        tl.store(L + ix, l)
        tl.store(A + ix * DV + dv, acc, dv < DV)

    @triton.jit
    def _partial_tc(Q, K, V, IDS, COUNT, LEN, M, L, A,
                    QSB: tl.constexpr, QSH: tl.constexpr,
                    KSB: tl.constexpr, KST: tl.constexpr, KSH: tl.constexpr,
                    VSB: tl.constexpr, VST: tl.constexpr, VSH: tl.constexpr,
                    HQ: tl.constexpr, HK: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
                    DQB: tl.constexpr, DVB: tl.constexpr, CAP: tl.constexpr,
                    WIDTH: tl.constexpr, SPLITS: tl.constexpr, SPAN: tl.constexpr,
                    PROTECTED: tl.constexpr, SINK: tl.constexpr, SWA: tl.constexpr,
                    RING_SINK: tl.constexpr, RING_TAIL: tl.constexpr,
                    TILE: tl.constexpr, GROUP_BLOCK: tl.constexpr):
        row, split = tl.program_id(0), tl.program_id(1)
        b, kv = row // HK, row % HK
        g = tl.arange(0, GROUP_BLOCK)
        dq, dv = tl.arange(0, DQB), tl.arange(0, DVB)
        qh = kv * (HQ // HK) + g
        q = tl.load(Q + b * QSB + qh[:, None] * QSH + dq[None, :],
                    (g[:, None] < HQ // HK) & (dq[None, :] < DQ), 0)
        m = tl.full((GROUP_BLOCK,), float('-inf'), tl.float32)
        l = tl.zeros((GROUP_BLOCK,), tl.float32)
        acc = tl.zeros((GROUP_BLOCK, DVB), tl.float32)
        for t in range(0, SPAN, TILE):
            offsets = split * SPAN + t + tl.arange(0, TILE)
            tok, valid = _positions(IDS, COUNT, LEN, b, kv, offsets, HK, WIDTH,
                                    PROTECTED, SINK, SWA, RING_SINK, RING_TAIL, CAP)
            kk = tl.load(K + b * KSB + tok[:, None] * KST + kv * KSH + dq[None, :],
                         valid[:, None] & (dq[None, :] < DQ), 0)
            vv = tl.load(V + b * VSB + tok[:, None] * VST + kv * VSH + dv[None, :],
                         valid[:, None] & (dv[None, :] < DV), 0)
            scores = tl.where(valid[None, :], tl.dot(q, tl.trans(kk)), float('-inf'))
            new_m = tl.maximum(m, tl.max(scores, 1))
            safe_m = tl.where(new_m == float('-inf'), 0., new_m)
            alpha = tl.exp(m - safe_m)
            p = tl.where(valid[None, :], tl.exp(scores - safe_m[:, None]), 0.)
            acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
            l = l * alpha + tl.sum(p, 1)
            m = new_m
        ix = (b * HQ + qh) * SPLITS + split
        tl.store(M + ix, m, g < HQ // HK)
        tl.store(L + ix, l, g < HQ // HK)
        tl.store(A + ix[:, None] * DV + dv[None, :], acc,
                 (g[:, None] < HQ // HK) & (dv[None, :] < DV))

    @triton.jit
    def _merge(MP, LP, AP, MS, LS, AS, OUT, P: tl.constexpr, S: tl.constexpr,
               DV: tl.constexpr, DB: tl.constexpr, NB: tl.constexpr):
        row = tl.program_id(0)
        n, d = tl.arange(0, NB), tl.arange(0, DB)
        mp = tl.load(MP + row * P + n, n < P, float('-inf'))
        ms = tl.load(MS + row * S + n - P, (n >= P) & (n < P + S), float('-inf'))
        m = tl.where(n < P, mp, ms)
        lp = tl.load(LP + row * P + n, n < P, 0.)
        ls = tl.load(LS + row * S + n - P, (n >= P) & (n < P + S), 0.)
        ap = tl.load(AP + (row * P + n[:, None]) * DV + d[None, :],
                     (n[:, None] < P) & (d[None, :] < DV), 0.)
        ass = tl.load(AS + (row * S + n[:, None] - P) * DV + d[None, :],
                      (n[:, None] >= P) & (n[:, None] < P + S) & (d[None, :] < DV), 0.)
        max_m = tl.max(m, 0)
        safe_m = tl.where(max_m == float('-inf'), 0., max_m)
        weight = tl.exp(m - safe_m)
        den = tl.sum(weight * tl.where(n < P, lp, ls), 0)
        num = tl.sum(weight[:, None] * tl.where(n[:, None] < P, ap, ass), 0)
        tl.store(OUT + row * DV + d, num / tl.where(den > 0., den, 1.), d < DV)


def _validate(q, k, v, ids, counts, lengths, sink, swa, splits, layout):
    if q.ndim != 4 or q.shape[1] != 1 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('decode requires Q[B,1,Hq,Dqk], K/V[B,T,Hkv,D]')
    if (k.shape[:3] != v.shape[:3] or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1] or k.shape[2] < 1
            or q.shape[2] % k.shape[2] or min(k.shape[1], k.shape[-1], v.shape[-1]) < 1):
        raise ValueError('incompatible decode dimensions')
    if any(t.device != q.device for t in (k, v, ids, counts, lengths)):
        raise ValueError('all decode inputs must share a device')
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('Q/K/V require matching floating dtypes')
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError('unit feature stride required')
    if (ids.ndim != 3 or ids.shape[:2] != (q.shape[0], k.shape[2])
            or counts.shape != ids.shape[:2] or ids.dtype != torch.int32
            or counts.dtype != torch.int32 or not ids.is_contiguous() or not counts.is_contiguous()
            or lengths.shape != (q.shape[0],) or lengths.dtype != torch.int64
            or not lengths.is_contiguous()):
        raise ValueError('invalid device selection/length metadata')
    if any(type(x) is not int or x < 0 for x in (sink, swa)) or type(splits) is not int or not 1 <= splits <= 64:
        raise ValueError('invalid protection/split geometry')
    if layout is not None and (layout.tail_capacity < 1 or layout.capacity != k.shape[1] or layout.sink != sink):
        raise ValueError('ring geometry does not match storage/protected prefix')


def partial_attention(q, k, v, selection, lengths, *, sink, swa, scale,
                      protected=False, splits=4, backend='simt', layout=None):
    ids, counts = selection
    _validate(q, k, v, ids, counts, lengths, sink, swa, splits, layout)
    if backend not in ('torch', 'simt', 'tc'):
        raise ValueError('backend must be torch, simt or tc')
    if backend == 'torch':
        return _torch_partial(q, k, v, selection, lengths, sink, swa, scale, protected, splits, layout)
    if triton is None or not q.is_cuda:
        raise RuntimeError('GPU partial attention requires CUDA and Triton; no implicit fallback')
    if backend == 'tc' and q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('TC backend requires fp16/bf16')
    b, _, h, dq = q.shape
    dv, hk = v.shape[-1], k.shape[2]
    width = sink + swa if protected else ids.shape[-1]
    tile = 32
    span = max(tile, triton.cdiv(width, splits * tile) * tile)
    # Match the existing consumer's model-dtype scale rounding before FP32.
    qs = (q[:, 0] * scale).contiguous()
    m = torch.empty((b, h, splits), device=q.device, dtype=torch.float32)
    l, a = torch.empty_like(m), torch.empty((b, h, splits, dv), device=q.device, dtype=torch.float32)
    kwargs = dict(HQ=h, HK=hk, DQ=dq, DV=dv, DQB=max(16, triton.next_power_of_2(dq)),
                  DVB=max(16, triton.next_power_of_2(dv)), CAP=k.shape[1], WIDTH=width,
                  SPLITS=splits, SPAN=span, PROTECTED=protected, SINK=sink, SWA=swa,
                  RING_SINK=0 if layout is None else layout.sink,
                  RING_TAIL=0 if layout is None else layout.tail_capacity, TILE=tile)
    kernel, grid = _partial_simt, (b * h, splits)
    if backend == 'tc':
        kernel, grid = _partial_tc, (b * hk, splits)
        kwargs['GROUP_BLOCK'] = max(16, triton.next_power_of_2(h // hk))
    kernel[grid](qs, k, v, ids, counts, lengths, m, l, a,
                 *qs.stride()[:2], *k.stride()[:3], *v.stride()[:3],
                 **kwargs, num_warps=4, num_stages=2)
    return PartialState(m, l, a)


def _torch_partial(q, k, v, selection, lengths, sink, swa, scale, protected, splits, layout):
    """Portable reference; materializes gathered tensors, never a fast fallback."""
    ids, counts = selection
    b, _, h, _ = q.shape
    hk, dv = k.shape[2], v.shape[-1]
    length = lengths[:, None, None]
    prefix = length.clamp(max=sink)
    start = torch.maximum(prefix, length - swa)
    if protected:
        ranks = torch.arange(sink + swa, device=q.device)[None, None]
        ids = torch.where(ranks < prefix, ranks, start + ranks - prefix).expand(b, hk, -1)
        valid = (ranks < prefix + length - start).expand_as(ids)
    else:
        valid = selection.valid() & (ids >= prefix) & (ids < start)
    valid = valid & (ids >= 0) & (ids < length)
    if layout is not None:
        valid = valid & ((ids < layout.sink) | (ids >= (length - layout.tail_capacity).clamp(min=layout.sink)))
        physical = layout.map(ids)
    else:
        physical = ids
    valid = valid & (physical >= 0) & (physical < k.shape[1])
    physical = torch.where(valid, physical, 0).long()
    rows = torch.arange(b, device=q.device)[:, None, None]
    heads = torch.arange(hk, device=q.device)[None, :, None]
    kk = k[rows, physical, heads].float().masked_fill(~valid[..., None], 0)
    vv = v[rows, physical, heads].float().masked_fill(~valid[..., None], 0)
    qg = (q[:, 0] * scale).float().reshape(b, hk, h // hk, -1)
    scores = torch.einsum('bhgd,bhnd->bhgn', qg, kk).masked_fill(~valid[:, :, None], -torch.inf)
    width = ids.shape[-1]
    span = max(1, (width + splits - 1) // splits)
    ms, ls, accs = [], [], []
    for i in range(splits):
        s, values = scores[..., i*span:(i+1)*span], vv[..., i*span:(i+1)*span, :]
        if s.shape[-1] == 0:
            m = torch.full((b, hk, h // hk), -torch.inf, device=q.device)
            l = torch.zeros_like(m)
            acc = torch.zeros((*m.shape, dv), device=q.device)
        else:
            m = s.amax(-1)
            safe = torch.where(torch.isfinite(m), m, 0.)
            p = torch.exp(s - safe[..., None])
            l = p.sum(-1)
            acc = torch.einsum('bhgn,bhnd->bhgd', p, values)
        ms.append(m.reshape(b, h)); ls.append(l.reshape(b, h)); accs.append(acc.reshape(b, h, dv))
    return PartialState(torch.stack(ms, 2), torch.stack(ls, 2), torch.stack(accs, 2))


def merge_partials(protected, selected, dtype, *, backend='simt'):
    mp, lp, ap = protected
    ms, ls, ass = selected
    if backend == 'torch':
        m, l, a = torch.cat((mp, ms), 2), torch.cat((lp, ls), 2), torch.cat((ap, ass), 2)
        top = m.amax(2, keepdim=True)
        weight = torch.exp(m - torch.where(torch.isfinite(top), top, 0.))
        den = (weight * l).sum(2)
        out = (weight[..., None] * a).sum(2) / torch.where(den > 0, den, 1.)[..., None]
        return out[:, None].to(dtype)
    if triton is None or not mp.is_cuda:
        raise RuntimeError('CUDA merge requires Triton')
    b, h, p = mp.shape
    s, dv = ms.shape[2], ap.shape[-1]
    out = torch.empty((b, 1, h, dv), device=mp.device, dtype=dtype)
    _merge[(b*h,)](mp, lp, ap, ms, ls, ass, out, P=p, S=s, DV=dv,
                    DB=triton.next_power_of_2(dv), NB=triton.next_power_of_2(p+s), num_warps=4)
    return out
