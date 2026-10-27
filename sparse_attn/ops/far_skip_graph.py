"""Fixed-shape far-off selection for tensor-position CUDA Graph execution.

All changing geometry comes from a device int64 length, never Python counters.
The workspace is bound to one append-only request, one stream, immutable dims
and fixed geometry. Setup occurs outside capture. No full-history mask/index is
built. Padded lanes redirect their reads to token zero and are zeroed before
arithmetic, so stale/NaN far storage cannot contaminate valid results.
"""
import torch
import torch.nn.functional as F

from .far_skip import _quantize
from .far_skip_workspace import FarSkipWorkspace, stable_topk_positions
from .far_skip_subspace import FarSkipFineCache, same_dimensions


class FarSkipGraphWorkspace:
    def __init__(self, cfg, layout, heads, coarse_dims, fine_dims):
        self.cfg, self.layout, self.heads = cfg, layout, heads
        self.shared_dims = same_dimensions(coarse_dims, fine_dims)
        self.coarse, self.fine = coarse_dims.clone(), fine_dims.clone()
        self.ws = FarSkipWorkspace(fixed_shapes=True)
        self.device = coarse_dims.device
        self.page_capacity = (max(cfg.near_length, cfg.block_size) + cfg.block_size - 1) // cfg.block_size
        self.page_count = min(cfg.near_pages, self.page_capacity)
        self.candidate_capacity = self.page_count*cfg.block_size + 2*(cfg.block_size-1)
        self.needs_fine = (0 < cfg.token_budget-cfg.sink_tokens-cfg.window_tokens
                           < self.candidate_capacity)
        # Force all constant ranges to exist before graph capture.
        self.ws.ids(max(layout.capacity, cfg.token_budget, self.page_capacity,
                        self.candidate_capacity+cfg.sink_tokens+cfg.window_tokens), self.device)
        self.head_ids = self.ws.ids(heads, self.device)
        self.summaries = None
        self.fine_cache = None

    def initialize(self, k, length):
        """Seed only complete near pages from prefill, outside capture.

        Thereafter each step (including capture warmup at the same position)
        refreshes only the newest complete page. At most one can become eligible
        per token. Arbitrary forward jumps/rewinds require a new workspace.
        """
        cfg, bs = self.cfg, self.cfg.block_size
        if self.needs_fine and self.fine.numel() < k.shape[-1]:
            self.fine_cache = FarSkipFineCache(cfg, self.layout, self.fine)
            self.fine_cache.initialize(k, length)
        if not 0 < self.page_count < self.page_capacity:
            return
        slots = self.page_capacity+1
        shape = (slots, self.heads, self.coarse.numel())
        self.summaries = [k.new_zeros(shape) for _ in range(1 if cfg.near_method == 'avg' else 2)]
        middle = max(min(length, cfg.sink_tokens), cfg.near_start(length))
        first = (middle+bs-1)//bs
        stop = max(first, max(0, length-cfg.window_tokens)//bs)
        if stop > first:
            pages = self.ids(stop-first)+first
            pos = (pages[:, None]*bs+self.ids(bs)).flatten()
            values = self._reduce_pages(k, pos, torch.ones_like(pos, dtype=torch.bool), stop-first)
            for target, source in zip(self.summaries, values):
                target.index_copy_(0, pages.remainder(slots), source)

    def _reduce_pages(self, k, positions, valid, count):
        if self.shared_dims and not self.cfg.quantize and self.fine_cache is not None:
            keys = self.fine_cache.gather(positions.expand(self.heads, -1),
                                         valid.expand(self.heads, -1)).transpose(0, 1)
        else:
            keys = self.gather(k, positions, valid, self.coarse)
        keys = keys.reshape(count, self.cfg.block_size, self.heads, -1)
        return ([keys.mean(1)] if self.cfg.near_method == 'avg'
                else [keys.amin(1), keys.amax(1)])

    def _summaries(self, k, pages, valid, first, stop):
        if self.summaries is None:
            raise ValueError('initialize graph page summaries before decode/capture')
        last = (stop-1).clamp(min=0)
        pos = last*self.cfg.block_size+self.ids(self.cfg.block_size)
        values = self._reduce_pages(k, pos, (stop>first).expand_as(pos), 1)
        slots = self.page_capacity+1
        for target, source in zip(self.summaries, values):
            target.index_copy_(0, last.remainder(slots), source)
        return [t.index_select(0, pages.remainder(slots)).masked_fill(~valid[:, None, None], 0).float()
                for t in self.summaries]

    def ids(self, n):
        return self.ws.ids(n, self.device)

    def gather(self, data, positions, valid, dims=None):
        from .far_skip_accel import enabled, gather
        if enabled(data):
            rows = positions.expand(self.heads, -1) if positions.ndim == 1 else positions
            mask = valid.expand(self.heads, -1) if valid.ndim == 1 else valid
            out = gather(data, rows, self.layout, self.ws, dims=dims, valid=mask)
            return out.transpose(0, 1) if positions.ndim == 1 else out
        safe = torch.where(valid, positions, 0)
        physical = self.layout.map(safe)
        if dims is None:
            out = data[0][physical, self.head_ids[:, None]]
        elif positions.ndim == 1:
            out = data[0][physical[:, None, None], self.head_ids[None, :, None],
                          dims[None, None, :]]
        else:
            out = data[0][physical[..., None], self.head_ids[:, None, None], dims]
        mask = valid[:, None, None] if positions.ndim == 1 else valid[..., None]
        return out.masked_fill(~mask, 0)

    def select(self, q, k, length, scale):
        cfg, bs, h = self.cfg, self.cfg.block_size, self.heads
        if self.fine_cache is not None:
            self.fine_cache.update_tensor(k, length-1)
        g = q.shape[2] // h
        prefix = torch.clamp(length, max=cfg.sink_tokens)
        swa = torch.clamp(length - cfg.window_tokens, min=0)
        start = torch.clamp((length-cfg.window_tokens-max(bs, cfg.near_length)) // bs,
                            min=cfg.sink_tokens // bs) * bs
        middle = torch.maximum(prefix, start)
        first = (middle + bs - 1) // bs
        stop = torch.maximum(first, swa // bs)
        pages = first + self.ids(self.page_capacity)
        page_valid = pages < stop
        qg = None
        if self.page_count:
            if self.page_count == self.page_capacity:
                chosen_pages = pages.expand(h, -1)
                chosen_valid = page_valid.expand(h, -1)
            else:
                summaries = self._summaries(k, pages, page_valid, first, stop)
                qg = (q[0, 0].index_select(-1, self.coarse).float() * scale).reshape(h, g, -1)
                if cfg.near_method == 'avg':
                    scores = torch.einsum('hgd,phd->hgp', qg, summaries[0])
                else:
                    scores = (torch.einsum('hgd,phd->hgp', qg.clamp(min=0), summaries[1])
                              + torch.einsum('hgd,phd->hgp', qg.clamp(max=0), summaries[0]))
                scores = scores.to(q.dtype).mean(1)
                chosen = stable_topk_positions(scores, self.page_count, self.ws,
                                              valid=page_valid.expand(h, -1))
                chosen_pages = pages.expand(h, -1).gather(1, chosen)
                chosen_valid = page_valid.expand(h, -1).gather(1, chosen)
        else:
            chosen_pages = self.ids(0).expand(h, -1)
            chosen_valid = chosen_pages.bool()

        # Only selected full pages and the two partial boundary fragments enter
        # L2. Padding is bounded by 2*(bs-1), independent of history/near length.
        left = (middle+self.ids(bs-1)).expand(h, -1)
        full = (chosen_pages[..., None]*bs+self.ids(bs)).reshape(h, -1)
        right = (stop*bs+self.ids(bs-1)).expand(h, -1)
        candidates = self.ws.cat('graph_candidates', [left, full, right])
        valid = self.ws.cat('graph_candidate_valid', [
            (left < first*bs) & (left < swa),
            chosen_valid[..., None].expand(-1, -1, bs).reshape(h, -1), right < swa])
        sink = self.ids(cfg.sink_tokens).expand(h, -1)
        window = (torch.maximum(prefix, swa) + self.ids(cfg.window_tokens)).expand(h, -1)
        protected = self.ws.cat('graph_protected', [sink, window])
        protected_valid = self.ws.cat('graph_protected_valid', [sink < prefix, window < length])
        keep = min(cfg.token_budget - cfg.sink_tokens - cfg.window_tokens, candidates.shape[-1])
        if 0 < keep < candidates.shape[-1]:
            pool = self.ws.cat('graph_pool', [candidates, protected])
            pool_valid = self.ws.cat('graph_pool_valid', [valid, protected_valid])
            if self.fine_cache is not None:
                keys = self.fine_cache.gather(pool, pool_valid)
            else:
                keys = self.gather(k, pool, pool_valid, self.fine)
                if cfg.quantize:
                    keys = _quantize(keys)
            qfine = (qg if self.shared_dims and qg is not None else
                     (q[0, 0].index_select(-1, self.fine).float() * scale).reshape(h, g, -1))
            logits = torch.einsum('hgd,hnd->hgn', qfine, keys.float()).to(q.dtype)
            logits = logits.masked_fill(~pool_valid[:, None], -torch.inf)
            probabilities = logits.softmax(-1).mean(1)[:, :candidates.shape[-1]]
            chosen = stable_topk_positions(probabilities, keep, self.ws, valid=valid)
            candidates = candidates.gather(1, chosen)
            valid = valid.gather(1, chosen)
        elif keep == 0:
            candidates, valid = candidates[:, :0], valid[:, :0]
        positions = self.ws.cat('graph_positions', [sink, candidates, window])
        mask = self.ws.cat('graph_valid', [sink < prefix, valid, window < length])
        return positions, mask

    def attend(self, q, k, v, positions, valid, scale):
        from .far_skip_accel import enabled, gather
        if enabled(k):
            kk, vv = gather(k, positions, self.layout, self.ws, other=v, valid=valid)
        else:
            kk, vv = self.gather(k, positions, valid), self.gather(v, positions, valid)
        g = q.shape[2] // self.heads
        out = F.scaled_dot_product_attention(
            q[0, 0].reshape(self.heads, 1, g, q.shape[-1]),
            kk[:, None], vv[:, None], attn_mask=valid[:, None, None],
            dropout_p=0.0, is_causal=False, scale=scale)
        return out.reshape(1, 1, q.shape[2], v.shape[-1])

    def __call__(self, q, k, v, length, scale):
        positions, valid = self.select(q, k, length, scale)
        return self.attend(q, k, v, positions, valid, scale)
