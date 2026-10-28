"""Bounded, head-major fine-key projection; never a full-history side cache.

Keys and dimensions are immutable within an append-only request. QDQ is token
local and query independent, so caching its original-dtype result preserves the
selector's rounding. V and final-attention K remain original-width/original dtype.
"""
import torch


def same_dimensions(a, b):
    """Metadata-only identity check, safe during capture (no device readback)."""
    return (a.device == b.device and a.dtype == b.dtype and a.shape == b.shape
            and a.stride() == b.stride() and a.data_ptr() == b.data_ptr())


def quantize_keys(x):
    lo = x.amin(dim=-1, keepdim=True)
    scale = (x.amax(dim=-1, keepdim=True) - lo).clamp(min=1e-9) / 15
    return ((x - lo) / scale).round().clamp(0, 15) * scale + lo


class FarSkipFineCache:
    def __init__(self, cfg, layout, dims):
        self.cfg, self.layout, self.dims = cfg, layout, dims
        self.keys = None
        self.last_length = None

    def _encode(self, values):
        return quantize_keys(values) if self.cfg.quantize else values

    def initialize(self, k, length):
        h = k.shape[2]
        self.keys = k.new_zeros(h, self.layout.capacity, self.dims.numel())
        # Only live retained positions: no reads/quantization of far or unused
        # slots even if the original ring's uninitialized slots contain NaNs.
        ids = torch.cat([torch.arange(min(self.cfg.sink_tokens, length), device=k.device),
                         torch.arange(self.cfg.tail_start(length), length, device=k.device)])
        physical = self.layout.map(ids)
        self.heads = torch.arange(h, device=k.device)
        projected = k[0][physical[None, :, None], self.heads[:, None, None], self.dims]
        self.keys.index_copy_(1, physical, self._encode(projected))
        self.last_length = length

    def append(self, k, length):
        if self.last_length == length:
            return
        if self.last_length != length-1:
            self.initialize(k, length)
            return
        from .far_skip_accel import enabled, project_token
        if not self.cfg.quantize and enabled(k):
            project_token(k, self.keys, self.dims, length-1, self.layout)
            self.last_length = length
            return
        slot = self.layout.slot(length-1)
        values = k[0, slot].index_select(-1, self.dims)
        self.keys[:, slot].copy_(self._encode(values))
        self.last_length = length

    def update_tensor(self, k, position):
        # Graph warmup/capture may overwrite the same current token. Refresh
        # that projection on every execution; no Python length state is used.
        from .far_skip_accel import enabled, project_token
        if not self.cfg.quantize and enabled(k):
            project_token(k, self.keys, self.dims, position, self.layout)
            return
        physical = self.layout.map(position)
        values = k[0][physical[None, :, None], self.heads[:, None, None], self.dims]
        self.keys.index_copy_(1, physical, self._encode(values))

    def gather(self, positions, valid=None):
        from .far_skip_accel import enabled, gather
        if enabled(self.keys):
            from .far_skip_workspace import FarSkipWorkspace
            if not hasattr(self, '_gather_workspace'):
                self._gather_workspace = FarSkipWorkspace()
            return gather(self.keys.transpose(0, 1)[None], positions, self.layout,
                          self._gather_workspace, valid=valid)
        if valid is not None:
            positions = torch.where(valid, positions, 0)
        physical = self.layout.map(positions)
        values = self.keys.gather(1, physical[..., None].expand(-1, -1, self.dims.numel()))
        return values if valid is None else values.masked_fill(~valid[..., None], 0)


def fine_cache_for(workspace, k, length, cfg, dims, layout):
    signature = (cfg, layout, k.device, k.dtype, k.shape, k.stride(), k.data_ptr(),
                 dims.shape, dims.stride(), dims.dtype, dims.device, dims.data_ptr())
    if getattr(workspace, '_fine_cache_signature', None) != signature:
        workspace._fine_cache_signature = signature
        workspace._fine_cache = FarSkipFineCache(cfg, layout, dims)
        workspace._fine_cache.initialize(k, length)
    else:
        workspace._fine_cache.append(k, length)
    return workspace._fine_cache
