"""Qwen3 eager decode bridge. Default/other-layer paths stay unchanged."""
import os

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .far_skip import FarSkipConfig, select_far_skip, attend_far_skip
from .far_skip_cache import FarSkipLayer, evict_far
from .far_skip_workspace import FarSkipWorkspace


def configure_far_skip(indexer):
    args = indexer.args
    backend = getattr(args, "tli_far_skip_backend", "legacy")
    cache_mode = getattr(args, "tli_far_skip_cache", "retain")
    if backend not in ("legacy", "compact") or cache_mode not in ("retain", "evict"):
        raise ValueError("invalid far skip backend/cache mode")
    if cache_mode == "evict" and backend != "compact":
        raise ValueError("far KV eviction requires compact far skipping")
    indexer.far_skip_backend, indexer.far_skip_cache = backend, cache_mode
    indexer.far_skip_config = None
    if backend == "legacy":
        return
    if (indexer.nt_near_len is None or indexer.nt_near_pages is None
            or indexer.gamma is not None or not indexer.boundary_bypass
            or indexer.per_q_head or indexer.enable_async
            or indexer._basis_all is not None or indexer.moba_gate
            or indexer.sigma_select != "none" or indexer.sinkswa_only
            or indexer.near_select != "4bit" or indexer.far_select != "4bit"
            or indexer.subspace == "mla_rope"):
        raise ValueError("compact far skip requires fixed near_len/pages, near_tokens=off, "
                         "boundary bypass and ordinary shared-head two-level selection")
    if indexer.nt_near_len < args.tia_block_size:
        raise ValueError("compact far skip requires near_len >= block_size")
    indexer.far_skip_config = FarSkipConfig(
        args.tia_block_size, indexer.nt_near_len, indexer.nt_near_pages,
        args.tia_level2_topk, indexer.sink_tokens, indexer.sliding_window_size,
        indexer.near_method, indexer.l2_quant)


def try_far_skip_decode(module, q, k, v, cache, mask, cache_position, batch_ctx):
    """Return compact output or None. All rejection guards precede KV mutation.

    Inputs use HF [B,H,S,D]. Output uses attention [B,S,H,D]. A None return
    leaves cache update to the caller. Only a frozen far-off layer enters.
    """
    idx = module.indexer
    if getattr(idx, "far_skip_backend", "legacy") != "compact":
        return None
    from .tli_static_cache import TLIStaticCache
    if type(cache) is TLIStaticCache:
        if getattr(cache, '_hybrid_batch', False):
            from .hybrid_batch_cache import try_hybrid_batch_decode
            return try_hybrid_batch_decode(module, q, k, v, cache, mask, batch_ctx)
        from .far_skip_static import try_far_skip_static
        return try_far_skip_static(module, q, k, v, cache, mask, batch_ctx)
    prefill = (type(cache) is DynamicCache and cache.get_seq_length(module.layer_idx) == 0)
    if (module.training or torch.is_grad_enabled()
            or module.sliding_window is not None or (batch_ctx is not None and not (
                prefill and batch_ctx.batch == 1 and batch_ctx.seq_lens == [q.shape[-2]]))
            or q.shape[0] != 1 or type(cache) is not DynamicCache
            or getattr(cache, "offloading", False)
            or os.environ.get("TLI_DECODE_DENSE", "0") == "1"
            or os.environ.get("TLI_SPARSE_PREFILL", "0") == "1"
            or (q.is_cuda and torch.cuda.is_current_stream_capturing())):
        raise ValueError("compact far skip requires eval and no_grad/inference_mode, B=1, eager unpadded decode, "
                         "DynamicCache, dense prefill, and no native sliding window")
    li = module.layer_idx
    length = cache.get_seq_length(li)
    if not length:
        idx._far_skip_workspace = None
        return None  # full prefill and gate calibration still use the original path
    if q.shape[-2] != 1 or mask is not None:
        raise ValueError("compact far skip requires one decode token and attention_mask=None")
    layer = cache.layers[li]
    if type(layer) is not DynamicLayer and not isinstance(layer, FarSkipLayer):
        raise ValueError("compact far skip requires ordinary DynamicLayer storage")
    idx.freeze_dyn_dgate()
    idx._resolve_skip()
    packed = isinstance(layer, FarSkipLayer) and layer.config is not None
    if not idx.skip_far:
        if packed:
            raise ValueError("cannot re-enable far after KV eviction")
        return None
    cfg = idx.far_skip_config
    if packed and (idx.far_skip_cache != "evict" or layer.config != cfg):
        raise ValueError("cannot change cache mode/geometry after eviction")
    # Reject incompatible head dimensions before evicting anything.
    ws = getattr(idx, "_far_skip_workspace", None)
    if ws is None:
        ws = idx._far_skip_workspace = FarSkipWorkspace()
    full = getattr(ws, "full_dims", None)
    if full is None:
        full = ws.full_dims = torch.arange(q.shape[-1], device=q.device)
    coarse = idx._subspace_indices(q.device) if idx.enable_subspace else full
    fine = (full if idx.l2_subspace == "full" else
            coarse if idx.enable_subspace else idx._qat_indices(q.device))
    if q.shape[-1] != 128:
        raise ValueError("compact TLI Qwen3 bridge currently requires head_dim=128")
    if idx.far_skip_cache == "evict":
        layer = evict_far(cache, li, cfg)
    k, v = cache.update(k, v, li, {"cache_position": cache_position})
    layout = layer.layout if isinstance(layer, FarSkipLayer) else None
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    selection = select_far_skip(q, k, length + 1, cfg, coarse, fine,
                               scale=module.scaling, layout=layout, workspace=ws)
    return attend_far_skip(q, k, v, selection, cfg,
                           scale=module.scaling, layout=layout, workspace=ws)
