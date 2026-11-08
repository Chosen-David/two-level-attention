"""Experimental candidate-domain lexicographic ranking, not dispatched.

Native equivalence is scoped to pinned CUDA TopK's stable medium-radix sort;
callers must independently verify that scope. This routine itself implements
(probability descending, logical ID ascending), including implicit zero tokens.
"""
import torch
from .token_selection import TokenSelection


def rank_candidates(ids, probabilities, valid, lower, upper, quota, *, capacity, width=1024, sink=0, deduplicated=False):
    """No capacity-wide arrays. IDs/probabilities/valid are [Hkv,R].

    Repeated IDs must have identical probabilities. The ranking interval is
    [lower,upper); final compact output additionally removes IDs below sink.
    All query heads have already been normalized separately and GQA averaged.
    Lower/upper/quota are device scalar tensors, never extracted to the host.
    """
    if (ids.ndim != 2 or probabilities.shape != ids.shape or valid.shape != ids.shape
            or width < 1 or capacity < 1 or capacity >= 2**31
            or probabilities.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('invalid candidate ranking geometry')
    if (ids.dtype != torch.int64 or valid.dtype != torch.bool
            or any(t.device != ids.device for t in (probabilities,valid,lower,upper,quota))
            or any(t.ndim != 0 or t.dtype != torch.int64 for t in (lower,upper,quota))):
        raise ValueError('same-device int64 IDs/scalar bounds and bool valid required')
    torch._assert_async((lower >= 0) & (upper >= lower) & (upper <= capacity)
                        & (quota >= 0) & (quota <= width), 'invalid ranking interval/quota')
    hk, r = ids.shape
    if hk < 1:
        raise ValueError('at least one KV head required')
    if (deduplicated and ids.is_cuda and probabilities.dtype in (torch.float16,torch.bfloat16)
            and capacity < 65536 and width == 1024 and r >= width
            and hk*r < 2**31 and all(t.is_contiguous() for t in (ids,probabilities,valid))):
        from .unique_ranking import rank_unique
        return rank_unique(ids,probabilities,valid,lower,upper,quota,
                           capacity=capacity,width=width,sink=sink)
    if r == 0:
        positive_ids = ids.new_full((hk, 1), capacity)
        positive_p = probabilities.new_zeros((hk, 1))
    elif deduplicated:
        # Upstream normalization supplies one valid slot per logical token.
        active = valid & (ids >= lower) & (ids < upper) & (probabilities > 0)
        positive_ids = torch.where(active, ids, capacity).to(torch.int32)
        positive_p = torch.where(active, probabilities, 0)
    else:
        active = valid & (ids >= lower) & (ids < upper) & (probabilities > 0)
        # Sorting by token ID both deduplicates and builds the zero-rank map.
        ordered, perm = torch.where(active, ids, capacity).to(torch.int32).sort(dim=-1, stable=True)
        p = probabilities.gather(-1, perm)
        previous = torch.cat((torch.full_like(ordered[:, :1], -1), ordered[:, :-1]), -1)
        unique = (ordered < capacity) & (ordered != previous)
        # Stable partition already-sorted unique IDs without another radix
        # sort. Both sides receive distinct destinations, including padding.
        prefix = unique.cumsum(-1)
        total = prefix[:, -1:]
        slots_in = torch.arange(r, device=ids.device)[None]
        destination = torch.where(unique, prefix-1, total+slots_in-prefix)
        positive_ids = torch.empty_like(ordered).scatter_(
            -1, destination, torch.where(unique, ordered, capacity))
        positive_p = torch.empty_like(p).scatter_(
            -1, destination, torch.where(unique, p, 0))
    npositive = (positive_ids < capacity).sum(-1)
    slots = torch.arange(width, device=ids.device)[None]
    n = positive_ids.shape[-1]
    # Nonnegative IEEE float order is its int32 bit order. The low 31 bits
    # encode the opposite logical ID so ties prefer smaller token positions.
    if positive_p.dtype in (torch.bfloat16, torch.float16) and capacity < 65536:
        # Positive 16-bit IEEE values sort by their bits, including +inf.
        # Keep the sign bit clear and use all 16 low bits for inverse ID.
        bits = positive_p.contiguous().view(torch.int16).to(torch.int32)
        key = (bits << 16) + (capacity - positive_ids).to(torch.int32)
    else:
        bits = positive_p.float().contiguous().view(torch.int32).to(torch.int64)
        key = (bits << 31) + (capacity - positive_ids).to(torch.int64)
    take = min(width, n)
    chosen = positive_ids.gather(-1, key.topk(take, -1).indices)
    if take < width:
        chosen = torch.nn.functional.pad(chosen, (0, width-take), value=capacity)
    q = quota.clamp(min=0, max=width)
    pos_keep = slots < torch.minimum(npositive[:, None], q)
    chosen = torch.where(pos_keep, chosen, capacity)
    if deduplicated:
        # Zero fill is needed only when npositive < quota. In that case TopK
        # above contains EVERY positive token, so K-wide sorting suffices.
        positive_ids = chosen.sort(-1).values
        n = width
    # For sorted positive ID p_j, p_j-lower-j is the number of implicit
    # zeros before it. Inverting these nondecreasing boundaries selects zeros
    # from the entire interval, even if no candidate slot represents them.
    rank = torch.arange(n, device=ids.device)[None]
    zero_boundaries = positive_ids - lower - rank
    zero_boundaries = torch.where(positive_ids < capacity, zero_boundaries, capacity)
    zero_ranks = slots.expand(hk, -1).contiguous()
    skipped_positive = torch.searchsorted(zero_boundaries.contiguous(), zero_ranks, right=True)
    zeros = lower + zero_ranks + skipped_positive
    zero_keep = ((slots < (q - npositive[:, None]).clamp(min=0)) & (zeros < upper))
    zeros = torch.where(zero_keep, zeros, capacity)
    combined = torch.cat((chosen, zeros), -1)
    combined = torch.where((combined >= sink) & (combined < upper), combined, capacity)
    packed = combined.sort(-1).values[:, :width]
    counts = (packed < capacity).sum(-1, dtype=torch.int32)[None]
    return TokenSelection(torch.where(packed < capacity, packed, 0)[None].to(torch.int32), counts)
