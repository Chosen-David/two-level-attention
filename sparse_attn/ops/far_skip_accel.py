"""Explicit GPU backend selection; never silently fall back after opting in."""
import os
from functools import lru_cache

import torch


def enabled(tensor):
    mode = os.environ.get('TLI_FAR_SKIP_KERNELS', 'torch')
    if mode not in ('torch', 'triton'):
        raise ValueError('TLI_FAR_SKIP_KERNELS must be torch or triton')
    return mode == 'triton' and tensor.is_cuda


@lru_cache(None)
def kernels():
    try:
        from . import far_skip_kernels
    except ImportError as exc:
        raise RuntimeError('TLI_FAR_SKIP_KERNELS=triton requires Triton') from exc
    return far_skip_kernels


def order_keys(scores, valid, workspace):
    """Same int64 total order as the Torch reference, in one GPU launch."""
    if scores.ndim != 2 or scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('fused order keys require 2D fp16/bf16/fp32 scores')
    rows, n = scores.shape
    if not 0 < n < 2**31:
        raise ValueError('fused order keys need 0 < N < 2**31')
    if valid is not None and (valid.shape != scores.shape or valid.dtype != torch.bool
                              or valid.device != scores.device):
        raise ValueError('validity must match scores')
    out = workspace.scratch('cuda_order_keys', scores.shape, scores, dtype=torch.int64)
    vs = valid.stride() if valid is not None else (0, 0)
    kernels().order_keys[(rows, (n+255)//256)](
        scores, valid if valid is not None else scores, out, n,
        *scores.stride(), *vs, valid is not None, 256, num_warps=4)
    return out


def write_pair(k, v, new_k, new_v, position, layout, *, alive=None):
    """HF [1,H,S,D] update; layout=None is linear, dead rows do not load/store KV.

    Callers validate storage/input geometry before capture. Alive is a one-row
    device bool view, so EOS can change during replay without a host branch.
    """
    tensor_position = isinstance(position, torch.Tensor)
    h, d = k.shape[1], k.shape[-1]
    kernels().write_pair[((h*d+255)//256,)](
        k, v, new_k, new_v, position, alive if alive is not None else k,
        h, d, *k.stride()[1:], *v.stride()[1:],
        new_k.stride(1), new_k.stride(3), new_v.stride(1), new_v.stride(3),
        layout.sink if layout is not None else 0,
        layout.tail_capacity if layout is not None else 0,
        tensor_position, alive is not None, 256, num_warps=4)


def project_token(k, out, dims, position, layout):
    """Unquantized [1,S,H,D] -> existing [H,S,d] ring; no temporary storage.

    Shapes/dimensions belong to a setup-validated FarSkipFineCache. Position may
    be a host integer (staged eager) or one device int64 (capture/replay).
    """
    h, _, d = out.shape
    if h and d:
        kernels().project_token[((h*d+255)//256,)](
            k, out, dims, position, h, d, *k.stride()[1:], *out.stride(),
            dims.stride(0), layout.sink, layout.tail_capacity,
            isinstance(position, torch.Tensor), 256, num_warps=4)


def gather(data, positions, layout, workspace, *, dims=None, valid=None, other=None):
    """Read [1,S,H,D] into [H,N,d], optionally K and V in one launch.

    Uses actual strides, including HF head-major transposes and strided dims.
    Invalid lanes never load a cache slot. Outputs borrow request-local scratch.
    """
    if data.ndim != 4 or data.shape[0] != 1 or positions.ndim != 2:
        raise ValueError('fused gather requires [1,S,H,D] and [H,N] positions')
    h, n = positions.shape
    if layout is not None and data.shape[1] != layout.capacity:
        raise ValueError('ring KV shape disagrees with layout')
    if h != data.shape[2] or positions.dtype != torch.long or positions.device != data.device:
        raise ValueError('fused gather positions must match KV heads/device')
    if valid is not None and (valid.shape != positions.shape or valid.dtype != torch.bool
                              or valid.device != data.device):
        raise ValueError('fused gather validity must match positions')
    if dims is not None and (dims.ndim != 1 or dims.dtype != torch.long or dims.device != data.device):
        raise ValueError('fused gather dimensions must be a device int64 vector')
    if other is not None and (other.shape != data.shape or other.dtype != data.dtype
                              or other.device != data.device):
        raise ValueError('fused KV pair must have matching shapes/dtypes/devices')
    d = data.shape[-1] if dims is None else dims.numel()
    shape = (h, n, d)
    out = workspace.scratch('cuda_gather_x', shape, data)
    out_v = workspace.scratch('cuda_gather_y', shape, data) if other is not None else out
    if n and d:
        vs = valid.stride() if valid is not None else (0, 0)
        ys = other.stride()[1:] if other is not None else data.stride()[1:]
        kernels().gather_rows[(h, (n+3)//4)](
            data, other if other is not None else data, positions,
            valid if valid is not None else positions, dims if dims is not None else positions,
            out, out_v, n, d, *data.stride()[1:], *ys, *positions.stride(), *vs,
            dims.stride(0) if dims is not None else 0,
            layout.sink if layout is not None else 0,
            layout.tail_capacity if layout is not None else 0,
            valid is not None, dims is not None, other is not None,
            4, 1 << (d-1).bit_length(), num_warps=4)
    return (out, out_v) if other is not None else out
