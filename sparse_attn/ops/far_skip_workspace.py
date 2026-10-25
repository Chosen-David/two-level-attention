"""Request-local ranges, scratch buffers and immutable complete-page summaries.

Only append-only inference KV is supported. Reset at every request boundary.
All caches are bounded by geometry, not by absolute sequence length.
"""
import torch


class FarSkipWorkspace:
    def __init__(self):
        self._ranges = {}
        self._buffers = {}
        self._signature = None
        self.last_length = 0
        self.first = self.stop = 0
        self.summaries = None
        self.pages_rebuilt = 0

    def ids(self, count, device):
        key = torch.device(device)
        buf = self._ranges.get(key)
        if buf is None or buf.numel() < count:
            capacity = max(1, 1 << (count - 1).bit_length())
            buf = torch.arange(capacity, device=device)
            self._ranges[key] = buf
        return buf[:count]

    def interval(self, start, stop, device):
        ids = self.ids(max(0, stop-start), device)
        return ids if start == 0 else ids + start

    def cat(self, name, parts, dim=-1):
        if len(parts) == 1:
            return parts[0]
        shape = list(parts[0].shape)
        shape[dim] = sum(t.shape[dim] for t in parts)
        size = 1
        for n in shape:
            size *= n
        key = (name, parts[0].device, parts[0].dtype)
        buf = self._buffers.get(key)
        if buf is None or buf.numel() < size:
            buf = parts[0].new_empty(max(1, 1 << (size - 1).bit_length()))
            self._buffers[key] = buf
        out = buf[:size].view(shape)
        return torch.cat(parts, dim=dim, out=out)

    def page_summary(self, k, length, cfg, dims, first, stop, gather):
        # Dimension indices are immutable request configuration; the runtime
        # reuses the same tensor. Holding it also prevents data_ptr ABA reuse.
        signature = (cfg, k.device, k.dtype, k.shape[2:], dims.data_ptr())
        if signature != self._signature or length <= self.last_length:
            self._signature = signature
            self._dims = dims
            self.first = self.stop = 0
            self.summaries = None
        self.last_length = length
        slots = (cfg.near_length + cfg.block_size - 1) // cfg.block_size + 2
        if self.summaries is None:
            shape = (slots, k.shape[2], dims.numel())
            self.summaries = [k.new_empty(shape) for _ in range(1 if cfg.near_method == 'avg' else 2)]
        begin = first if first < self.first else max(first, self.stop)
        if begin < stop:
            pages = self.interval(begin, stop, k.device)
            positions = (pages[:, None] * cfg.block_size
                         + self.ids(cfg.block_size, k.device)).flatten()
            keys = gather(positions, dims).reshape(-1, cfg.block_size, k.shape[2], dims.numel())
            values = [keys.mean(1)] if cfg.near_method == 'avg' else [keys.amin(1), keys.amax(1)]
            indices = pages.remainder(slots)
            for target, value in zip(self.summaries, values):
                target.index_copy_(0, indices, value)
            self.pages_rebuilt += stop - begin
        self.first, self.stop = first, stop
        indices = self.interval(first, stop, k.device).remainder(slots)
        return [t.index_select(0, indices).float() for t in self.summaries]


def stable_topk_positions(scores, count, workspace):
    """Exact lexicographic (score descending, input position ascending) top-k.

    IEEE fp32 order plus a unique position suffix fits signed int64 for N<2^31.
    No epsilon perturbation or CPU synchronization. Only K selected positions
    are sorted; fp64 retains the general stable-sort reference path.
    """
    n = scores.shape[-1]
    ids = workspace.ids(n, scores.device)
    if count == n:
        return ids.expand(*scores.shape[:-1], n)
    if count == 0:
        return ids[:0].expand(*scores.shape[:-1], 0)
    if scores.dtype == torch.float64 or n >= 2**31:
        return scores.argsort(dim=-1, descending=True, stable=True)[..., :count].sort(-1).values
    # +0 canonicalizes -0, which must tie with +0 under numerical sorting.
    normalized = scores.float() + 0.0
    bits = normalized.contiguous().view(torch.int32).to(torch.int64)
    ordered = bits.bitwise_xor((bits >> 31).bitwise_and(0x7fffffff))
    # PyTorch descending argsort places all NaNs before +inf, stably.
    ordered = torch.where(normalized.isnan(), 0x7fffffff, ordered)
    keys = ordered * n + (n - 1 - ids)
    chosen = keys.topk(count, dim=-1, sorted=False).indices
    return chosen.sort(-1).values
