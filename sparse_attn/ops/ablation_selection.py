"""Explicit eager ablation selector with inclusive, unique K1/K2 budgets.

Production scoring is shared. This deliberately separate selector makes the
paper's budget convention explicit; it is not an optimized deployment path.
"""
import torch
from .frozen_region_cache import RegionGeometry


def select(idx, scores, visible, length, q, k, ids, scale):
    spec = idx.ablation_spec
    g = RegionGeometry(sink=spec['sink'], window=spec['swa'], near=spec['near'])
    sink_hi, far_hi, near_hi = g.bounds(length)
    bs = idx.args.tia_block_size
    coarse = scores['score_coarse'].float()
    hq, pages = coarse.shape[-2:]
    group = idx.group_size
    hk = hq // group
    coarse = coarse.reshape(hk, group, pages).mean(1)
    avg = scores.get('score_coarse_avg')
    if avg is not None:
        avg = avg.float().reshape(hk, group, pages).mean(1)
    pos = torch.arange(length, device=k.device)
    protected = ((pos < sink_hi) | (pos >= near_hi)) & visible
    dynamic = visible & ~protected
    page_id = pos // bs
    page_dynamic = torch.zeros(pages, dtype=torch.bool, device=k.device)
    page_dynamic.scatter_(0, page_id[dynamic], True)
    # Page-aligned far boundary; only mixed sink/SWA pages can require bypass.
    boundary = torch.zeros_like(page_dynamic)
    if spec['boundary']:
        for edge in (sink_hi, near_hi):
            if edge % bs and edge < length:
                boundary[edge // bs] = page_dynamic[edge // bs]
    near_page = torch.arange(pages, device=k.device) * bs >= far_hi
    selected = torch.zeros((hk, pages), dtype=torch.bool, device=k.device)
    k1 = spec['k1']
    partition = spec['partition']
    pools = [(page_dynamic, k1, 'minmax')] if not partition else [
        (page_dynamic & ~near_page, k1-spec['near_pages'], spec['far_score']),
        (page_dynamic & near_page, spec['near_pages'], spec['near_score'])]
    counts = []
    for pool, quota, method in pools:
        forced = pool & boundary
        forced_count = int(forced.sum())
        if forced_count > quota:
            raise ValueError('boundary pages exceed regional candidate quota')
        selected |= forced[None]
        candidates = (pool & ~forced).nonzero().flatten()
        take = min(quota-forced_count, candidates.numel())
        source = avg if method == 'avg' else coarse
        if source is None:
            raise ValueError('Avg selector requires actual mean-key scores')
        if take:
            choice = source[:, candidates].topk(take, -1).indices
            selected.scatter_(1, candidates[choice], True)
        counts.append(selected.sum(-1).tolist())
    candidates = selected[:, page_id] & dynamic[None]
    allowed = candidates | protected[None]
    fine = scores['score_fine'].float().reshape(hk, group, length)
    # Same shared-KV convention as TLI: average query-head probabilities.
    probabilities = fine.masked_fill(~allowed[:, None], -torch.inf).softmax(-1).mean(1)
    probabilities = probabilities.masked_fill(~candidates, -torch.inf)
    mask = protected[None].expand(hk, -1).clone()
    dynamic_budget = max(0, spec['k2']-int(protected.sum()))
    if int(protected.sum()) > spec['k2']:
        raise ValueError('protected tokens exceed total attention budget')
    take = min(dynamic_budget, length)
    if take:
        indices = probabilities.topk(take, -1).indices
        mask.scatter_(1, indices, torch.gather(candidates, 1, indices))
        mask |= protected[None]
    receipt = dict(candidate_pages_per_kv_head=selected.sum(-1).tolist(),
        protected_tokens=int(protected.sum()), total_budget=spec['k2'],
        candidate_budget=k1, boundary_pages=int(boundary.sum()),
        budget_convention='K1 includes mixed boundary pages; K2 includes unique protected tokens')
    if spec.get('measure_mass', False) and getattr(idx, 'ablation_step', 1) in (1,2,4,8,16,32,64,128,256,512):
        # Exact full-dimensional attention over all currently visible K. B has
        # skip disabled, so this is the complete historical attention reference.
        qq = q[0, 0].float().reshape(hk, group, -1)
        kk = k[0].float().permute(1, 0, 2)
        p = torch.einsum('hgd,htd->hgt', qq, kk).mul(scale).softmax(-1)
        cm = allowed[:, ids]
        receipt['candidate_full_attention_mass'] = float((p*cm[:,None]).sum(-1).mean())
        receipt['selected_full_attention_mass'] = float((p*mask[:,ids,None].transpose(1,2)).sum(-1).mean())
    return mask[None, None], receipt
