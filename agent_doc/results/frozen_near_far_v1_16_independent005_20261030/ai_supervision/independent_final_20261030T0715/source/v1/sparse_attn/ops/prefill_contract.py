"""Metadata-only validation for the existing token-mask prefill interface."""
from numbers import Integral

import torch


def token_selection(mask, block_size, *, length, heads, kv_heads, device):
    """Return a view of a complete single-query token mask, without device reads.

    Block-level selectors need a separate prefill adapter; silently interpreting
    their block IDs as token IDs changes support. Padding and strided views are
    valid, but incomplete extents and implicit head/batch broadcasts are not.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size != 1:
        raise ValueError("prefill selection requires token-level block_size=1")
    if not isinstance(mask, torch.Tensor) or mask.layout != torch.strided:
        raise ValueError("prefill selection mask must be a strided tensor")
    if mask.dtype != torch.bool or mask.device != device:
        raise ValueError("prefill selection mask must be bool on the q/k/v device")
    if (mask.ndim != 4 or mask.shape[:2] != (1, 1)
            or mask.shape[2] not in (kv_heads, heads)):
        raise ValueError("prefill selection mask must have shape [1, 1, Hkv or H, T_pad]")
    if mask.shape[-1] < length:
        raise ValueError("prefill selection mask must cover the full query prefix")
    return mask[0, 0, :, :length]
