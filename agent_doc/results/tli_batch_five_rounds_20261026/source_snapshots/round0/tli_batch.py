"""Opt-in fixed-capacity batched TLI decode, without a Python request loop.

This is a numerical candidate, not a bitwise replacement for the sliced eager
selector: fixed-slot softmax/topk can change rounding and tie selection. Its
algorithm follows compute_mask_graph's compact candidate domain. All live
lengths are device tensors, so the same entry point supports eager and capture.
"""
import os
import torch
import torch.nn.functional as F


def capturing(device):
    return device.type == 'cuda' and torch.cuda.is_current_stream_capturing()


def validate_config(indexer):
    indexer._resolve_skip()
    if not (indexer.alpha > 0 and indexer.beta > 0
            and indexer.gamma is None and indexer.nt_near_pages is not None
            and indexer.nt_near_len is not None and indexer.boundary_bypass
            and not indexer.per_q_head and indexer._basis_all is None
            and indexer.far_select == indexer.near_select == '4bit'
            and not indexer.moba_gate and indexer.sigma_select == 'none'
            and not indexer.enable_async and not indexer.sinkswa_only
            and not indexer.dyn_dgate and not indexer.static_pair
            and getattr(indexer, 'far_skip_backend', 'legacy') == 'legacy'):
        raise ValueError('parallel batch indexer requires fixed near pages/length, '
                         'free K2, boundary bypass, static subspace, legacy KV; '
                         'projection, dynamic gate, compact eviction and ablations unsupported')


class BatchedTLI:
    """One append-only request batch, one attention layer. clear() at refill.

    Input K is [B,S,Hkv,D], Q [B,1,Hq,D], lengths [B] (valid prefix).
    Warm up before capture. During decode lengths may stay unchanged (finished
    rows or capture warmup) or advance by one. Refill/reorder/rewind requires a
    new instance and graph recapture, including if the underlying pointer stays.
    """
    def __init__(self, indexer, k, lengths):
        validate_config(indexer)
        if capturing(k.device):
            raise RuntimeError('warm up parallel batch indexer before capture')
        self.score_backend = os.environ.get('TLI_BATCH_SCORE', 'torch')
        if self.score_backend not in ('torch', 'triton'):
            raise ValueError('TLI_BATCH_SCORE must be torch or triton')
        self.triton_score = None
        if self.score_backend == 'triton':
            if not k.is_cuda:
                raise ValueError('Triton batch scoring requires CUDA')
            from .tli_batch_kernels import score
            self.triton_score = score
        self.indexer = indexer
        self.signature = (k.shape, k.stride(), k.dtype, k.device, k.data_ptr())
        self.B, self.S, self.H, self.D = k.shape
        self.bs = indexer.args.tia_block_size
        self.P = (self.S + self.bs - 1) // self.bs
        self.coarse_dims = (indexer._subspace_indices(k.device)
                            if indexer.enable_subspace else None)
        self.fine_dims = (None if indexer.l2_subspace == 'full' else
                          self.coarse_dims if self.coarse_dims is not None
                          else indexer._qat_indices(k.device))
        self.rows = torch.arange(self.B, device=k.device)
        self.heads = torch.arange(self.H, device=k.device)
        self.tokens = torch.arange(self.S, device=k.device)
        self.pages = torch.arange(self.P, device=k.device)
        self.offsets = torch.arange(self.bs, device=k.device)
        self.previous = lengths.clone()
        x = k if self.coarse_dims is None else k.index_select(-1, self.coarse_dims)
        # Poisoned/unwritten padding must not affect block stats. Legacy uses
        # zero padding inside the last block, including for min/max and mean.
        x = x.masked_fill((self.tokens[None] >= lengths[:, None])[..., None, None], 0)
        x = F.pad(x, (0, 0, 0, 0, 0, self.P*self.bs-self.S))
        pages = x.reshape(self.B, self.P, self.bs, self.H, x.shape[-1])
        self.kmin, self.kmax = pages.amin(2), pages.amax(2)
        self.kavg = pages.mean(2) if indexer._need_avg_score() else None
        self.kfine = self._fine(k).clone()
        self.kfine.masked_fill_((self.tokens[None] >= lengths[:, None])[..., None, None], 0)
        self.cu = torch.arange(self.B+1, dtype=torch.int32, device=k.device)*self.S

    def _fine(self, x):
        x = x if self.fine_dims is None else x.index_select(-1, self.fine_dims)
        return self.indexer.min_max_per_token_quant(x) if self.indexer.l2_quant else x

    def update(self, k, lengths):
        if self.signature != (k.shape, k.stride(), k.dtype, k.device, k.data_ptr()):
            raise ValueError('parallel indexer KV storage changed; clear and recapture')
        torch._assert_async(((lengths >= self.previous)
                             & (lengths <= self.previous+1)).all(),
                            'parallel indexer needs append-only lengths; clear at refill')
        pos = lengths - 1
        page = pos // self.bs
        ids = page[:, None]*self.bs + self.offsets[None]
        x = k[self.rows[:, None], ids.clamp(max=self.S-1)]
        x = x if self.coarse_dims is None else x.index_select(-1, self.coarse_dims)
        x = x.masked_fill((ids >= lengths[:, None])[..., None, None], 0)
        self.kmin[self.rows, page] = x.amin(1)
        self.kmax[self.rows, page] = x.amax(1)
        if self.kavg is not None:
            self.kavg[self.rows, page] = x.mean(1)
        self.kfine[self.rows, pos] = self._fine(k[self.rows, pos])
        self.previous.copy_(lengths)

    def mask(self, q, k, lengths, scale):
        self.update(k, lengths)
        idx, B, H, S, bs = self.indexer, self.B, self.H, self.S, self.bs
        G = q.shape[2] // H
        # Preserve original scale-in-input-dtype followed by FP32 conversion.
        q32 = (q[:, 0]*scale).float().reshape(B, H, G, self.D)
        qc = q32 if self.coarse_dims is None else q32.index_select(-1, self.coarse_dims)
        qf = q32 if self.fine_dims is None else q32.index_select(-1, self.fine_dims)
        def dot(x, qq):
            return torch.einsum('bhgd,bthd->bhgt', qq, x.float())
        if self.triton_score is not None:
            sc = self.triton_score(self.kmin,qc,q.dtype,kmax=self.kmax).mean(2)
            avg = (self.triton_score(self.kavg,qc,q.dtype).mean(2)
                   if self.kavg is not None else None)
        else:
            sc = (dot(self.kmin, qc.clamp(max=0))
                  + dot(self.kmax, qc.clamp(min=0))).to(q.dtype).mean(2)
            avg = dot(self.kavg, qc).to(q.dtype).mean(2) if self.kavg is not None else None
        sink, swa = idx.sink_tokens, idx.sliding_window_size
        lo = idx._sink_pool_lo_blk
        swalo = (lengths-swa).clamp(min=0)
        near = ((lengths-swa-max(bs, int(idx.nt_near_len)))//bs).clamp(min=lo)
        swablock = torch.maximum(near, (lengths+bs-1)//bs-max(1, swa//bs))
        bypage = swalo//bs
        byok = (swalo % bs != 0) & (bypage >= lo)
        fp = (self.pages[None] >= lo) & (self.pages[None] < near[:, None])
        np = (self.pages[None] >= near[:, None]) & (self.pages[None] < swablock[:, None])
        if sink % bs:
            fp &= self.pages[None] != lo
            np &= self.pages[None] != lo
        np &= ~((self.pages[None] == bypage[:, None]) & byok[:, None])
        nf = 0 if idx.skip_far else idx.args.tia_level1_topk-int(idx.nt_near_pages)
        nn = int(idx.nt_near_pages)
        far_src = avg if idx.far_method == 'avg' and avg is not None else sc
        near_src = avg if idx.near_method == 'avg' and avg is not None else sc
        fi = far_src.masked_fill(~fp[:, None], -torch.inf).topk(min(nf,self.P), -1).indices
        ni = near_src.masked_fill(~np[:, None], -torch.inf).topk(min(nn,self.P), -1).indices
        blocks = torch.cat((fi, ni), -1)
        ok = torch.cat((fp[:, None].expand(-1,H,-1).gather(-1,fi),
                        np[:, None].expand(-1,H,-1).gather(-1,ni)), -1)
        ids = (blocks[..., None]*bs+self.offsets).flatten(-2)
        keep = (ok[..., None].expand(-1,-1,-1,bs).flatten(-2)
                & (ids < lengths[:, None, None]))
        byblocks = torch.stack((torch.full_like(bypage,lo if sink%bs else 0), bypage), -1)
        byids = (byblocks[..., None]*bs+self.offsets).flatten(-2)
        bykeep = torch.stack((torch.full_like(byok,bool(sink%bs)),byok), -1)
        bykeep = bykeep[..., None].expand(-1,-1,bs).flatten(-2) & (byids < lengths[:,None])
        # When the sink and SWA cross the same page, include it only once.
        bykeep[:, bs:] &= ~((bypage == lo) & bool(sink%bs))[:,None]
        ids = torch.cat((ids, byids[:,None].expand(-1,H,-1)), -1)
        keep = torch.cat((keep, bykeep[:,None].expand(-1,H,-1)), -1)
        order = (~keep).argsort(dim=-1, stable=True)
        ids, keep = ids.gather(-1,order), keep.gather(-1,order)
        safe = torch.where(keep, ids, 0)
        # No [B,Hq,S,D] repeat; only [B,Hkv,R,d] candidate features.
        if self.triton_score is not None:
            sf = self.triton_score(self.kfine,qf,q.dtype,ids=safe,keep=keep)
        else:
            kc = self.kfine[self.rows[:,None,None], safe, self.heads[None,:,None]]
            sf = torch.einsum('bhgd,bhrd->bhgr', qf, kc.float()).to(q.dtype)
            sf.masked_fill_(~keep[:,:,None], -torch.inf)
        p = sf.softmax(-1).nan_to_num(nan=0).mean(2)
        mid = keep & (ids >= lo*bs) & (ids < swalo[:,None,None])
        ntop = min(idx.args.tia_level2_topk, ids.shape[-1])
        top = p.masked_fill(~mid, -torch.inf).topk(ntop, -1).indices
        selected = ids.gather(-1, top)
        midbudget = (idx.args.tia_level2_topk-sink-(lengths-swalo)).clamp(min=0)
        valid = mid.gather(-1,top) & (torch.arange(ntop,device=q.device)[None,None] < midbudget[:,None,None])
        # Extra sentinel column prevents invalid slots from overwriting token 0.
        selected = torch.where(valid, selected, S)
        mask = torch.zeros((B,H,S+1), dtype=torch.bool, device=q.device)
        mask.scatter_(-1, selected, True)
        force = (self.tokens[None] < sink) | (self.tokens[None] >= swalo[:,None])
        mask = (mask[...,:S] | force[:,None]) & (self.tokens[None,None] < lengths[:,None,None])
        return mask[:,None]


def prepare_mask_batched(indexer, q, k, lengths, scale):
    """Explicit batch API. All lengths are device-side; no scalar reads on CUDA."""
    validate_config(indexer)
    if (q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0]
            or q.shape[0] < 1 or q.shape[2] < 1
            or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or q.shape[1] != 1 or k.shape[1] < 1 or k.shape[2] < 1
            or q.shape[2] % k.shape[2] or q.shape[-1] != k.shape[-1]
            or k.shape[-1] != 128 or q.dtype != k.dtype or q.device != k.device
            or lengths.shape != (q.shape[0],) or lengths.device != k.device
            or lengths.dtype != torch.long or q.requires_grad or k.requires_grad):
        raise ValueError('parallel indexer expects decode Q[B,1,Hq,128], K[B,S,Hkv,128], int64 lengths[B]')
    torch._assert_async(((lengths >= 1) & (lengths <= k.shape[1])).all(), 'invalid batch lengths')
    state = getattr(indexer, '_batch_tli', None)
    if state is None:
        state = BatchedTLI(indexer, k, lengths)
        indexer._batch_tli = state
    mask = state.mask(q, k, lengths, scale)
    if not capturing(q.device):
        indexer.metrics.add_select_result_batch(lengths, mask, 1)
    return mask, 1
