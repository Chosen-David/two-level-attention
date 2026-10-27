"""Bound near-only AVG dot products without changing the full-pool top-k domain."""
import torch


def near_page_scores(indexer, averages, query, dtype, start, pool_mask,
                     near_length, block_size, kv_heads, groups):
    pages = averages.shape[0]
    # Both boundary rounding operations can add a page. Keep top-k's original
    # full-width domain/tie order; only the AVG dot products are regional.
    width = min(pages, (near_length+block_size-1)//block_size+2)
    ids = start + torch.arange(width, device=averages.device)
    safe = ids.clamp(max=pages-1)
    selected = averages.index_select(0, safe)
    try:
        scores = indexer._fast_score_fine(selected, query, dtype)
    except Exception:
        from einops import repeat, einsum
        keys = repeat(selected, 't h d -> t (h g) d', g=groups).float()
        scores = einsum(query, keys, 'h d, t h d -> h t').to(dtype)
    scores = scores.reshape(kv_heads, groups, width).mean(1)
    valid = (ids < pages) & pool_mask[safe]
    # Invalid slots go to a separate sentinel: clamping several slots to the
    # final page must never overwrite that page's valid score with -inf.
    targets = torch.where(valid, ids, pages).expand(kv_heads, -1)
    output = scores.new_full((kv_heads, pages+1), -torch.inf)
    output.scatter_(-1, targets, scores.masked_fill(~valid, -torch.inf))
    return output[:, :pages].contiguous().view(1, 1, kv_heads, pages)
