"""Fixed-shape far-off selection for staged eager and whole-model CUDA Graph.

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


class FarSkipGraphWorkspace:
    def __init__(self, cfg, layout, heads, coarse_dims, fine_dims):
        self.cfg, self.layout, self.heads = cfg, layout, heads
        self.coarse, self.fine = coarse_dims.clone(), fine_dims.clone()
        self.ws = FarSkipWorkspace()
        self.device = coarse_dims.device
        self.page_capacity = (cfg.near_length + cfg.block_size - 1) // cfg.block_size + 1
        self.page_count = min(cfg.near_pages, self.page_capacity)
        # Force all constant ranges to exist before graph capture.
        self.ws.ids(max(layout.capacity, cfg.token_budget, self.page_capacity), self.device)
        self.head_ids = self.ws.ids(heads, self.device)

    def ids(self, n):
        return self.ws.ids(n, self.device)

    def gather(self, data, positions, valid, dims=None):
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
        if self.page_count:
            page_pos = (pages[:, None] * bs + self.ids(bs)).flatten()
            page_mask = page_valid[:, None].expand(-1, bs).flatten()
            page_keys = self.gather(k, page_pos, page_mask, self.coarse).reshape(
                self.page_capacity, bs, h, -1)
            qg = (q[0, 0].index_select(-1, self.coarse).float() * scale).reshape(h, g, -1)
            if cfg.near_method == 'avg':
                scores = torch.einsum('hgd,phd->hgp', qg, page_keys.mean(1).float())
            else:
                scores = (torch.einsum('hgd,phd->hgp', qg.clamp(min=0), page_keys.amax(1).float())
                          + torch.einsum('hgd,phd->hgp', qg.clamp(max=0), page_keys.amin(1).float()))
            scores = scores.to(q.dtype).mean(1)
            chosen = stable_topk_positions(scores, self.page_count, self.ws,
                                          valid=page_valid.expand(h, -1))
            chosen_pages = pages.expand(h, -1).gather(1, chosen)
            chosen_valid = page_valid.expand(h, -1).gather(1, chosen)
        else:
            chosen_pages = self.ids(0).expand(h, -1)
            chosen_valid = chosen_pages.bool()

        # Initial fixed-path baseline: score the bounded near tail, masking
        # unchosen full pages. Round 3 can compact this pool without changing
        # page or token decisions.
        candidates = (middle + self.ids(max(bs, cfg.near_length) + bs - 1)).expand(h, -1)
        candidate_page = candidates // bs
        full = (candidate_page >= first) & (candidate_page < stop)
        chosen_mask = ((candidate_page[..., None] == chosen_pages[:, None, :])
                       & chosen_valid[:, None, :]).any(-1)
        valid = (candidates < swa) & (~full | chosen_mask)
        sink = self.ids(cfg.sink_tokens).expand(h, -1)
        window = (torch.maximum(prefix, swa) + self.ids(cfg.window_tokens)).expand(h, -1)
        protected = torch.cat([sink, window], -1)
        protected_valid = torch.cat([sink < prefix, window < length], -1)
        keep = min(cfg.token_budget - cfg.sink_tokens - cfg.window_tokens, candidates.shape[-1])
        if keep:
            pool = torch.cat([candidates, protected], -1)
            pool_valid = torch.cat([valid, protected_valid], -1)
            keys = self.gather(k, pool, pool_valid, self.fine)
            if cfg.quantize:
                keys = _quantize(keys)
            qg = (q[0, 0].index_select(-1, self.fine).float() * scale).reshape(h, g, -1)
            logits = torch.einsum('hgd,hnd->hgn', qg, keys.float()).to(q.dtype)
            logits = logits.masked_fill(~pool_valid[:, None], -torch.inf)
            probabilities = logits.softmax(-1).mean(1)[:, :candidates.shape[-1]]
            chosen = stable_topk_positions(probabilities, keep, self.ws, valid=valid)
            candidates = candidates.gather(1, chosen)
            valid = valid.gather(1, chosen)
        else:
            candidates, valid = candidates[:, :0], valid[:, :0]
        positions = torch.cat([sink, candidates, window], -1)
        mask = torch.cat([sink < prefix, valid, window < length], -1)
        return positions, mask

    def attend(self, q, k, v, positions, valid, scale):
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
