"""Fixed-shape protected-domain completion for append-only decode."""
import torch


def complete_protected_candidates(ids, keep, length, sink, swa, capacity):
    """Append each protected ID once, removing its old L1/bypass occurrence.

    Input active non-protected IDs must already be unique (disjoint L1 pools
    and bypass pages). All bounds depend on device length; shapes depend only
    on configuration/capacity, so changing length does not require recapture.
    Invalid IDs are never used for a load before downstream compaction.
    """
    lo = (length - swa).clamp(min=0)
    keep = keep & (ids >= sink) & (ids < lo) & (ids >= 0)
    head = torch.arange(min(sink, capacity), device=ids.device)
    tail = lo + torch.arange(min(swa, capacity), device=ids.device)
    protected = torch.cat((head, tail))
    valid = torch.cat((head < length, (tail < length) & (tail >= sink)))
    shape = (*ids.shape[:-1], -1)
    return (torch.cat((ids, protected.expand(shape)), -1),
            torch.cat((keep, valid.expand(shape)), -1))
