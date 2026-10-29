"""Qwen3 eager decode bridge. Default/other-layer paths stay unchanged."""
import os

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .far_skip import FarSkipConfig, select_far_skip, attend_far_skip
from .far_skip_cache import FarSkipLayer, evict_far
from .far_skip_workspace import FarSkipWorkspace


def _config_supports_compact(config):
    if config is None or getattr(config, "model_type", None) != "qwen3":
        return False
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        heads = getattr(config, "num_attention_heads", 0)
        head_dim = (getattr(config, "hidden_size", 0) // heads) if heads else None
    layer_types = getattr(config, "layer_types", None) or ()
    native_swa = (getattr(config, "use_sliding_window", False)
                  or getattr(config, "sliding_window", None) is not None
                  or "sliding_attention" in layer_types)
    return head_dim == 128 and not native_swa


def _model_supports_compact(model):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.llama.modeling_llama import LlamaAttention

    targets = []
    for module in model.modules():
        is_mla = all(hasattr(module, name) for name in (
            "kv_a_proj_with_mqa", "q_head_dim", "qk_rope_head_dim", "rotary_emb"))
        if isinstance(module, (Qwen3Attention, LlamaAttention)) or is_mla:
            targets.append(module)
    return (bool(targets) and _config_supports_compact(getattr(model, "config", None))
            and all(isinstance(module, Qwen3Attention)
                    and module.head_dim == 128 and module.sliding_window is None
                    for module in targets))


def _args_support_compact(args):
    near_tokens = getattr(args, "tli_near_tokens", None)
    near_len = getattr(args, "tli_near_len", None)
    try:
        near_len_ok = near_len is not None and int(near_len) >= int(
            getattr(args, "tia_block_size", 64))
    except (TypeError, ValueError, OverflowError):
        near_len_ok = False
    return (near_len_ok
            and getattr(args, "tli_near_pages", None) is not None
            and isinstance(near_tokens, str)
            and near_tokens.strip().lower() == "off"
            and getattr(args, "tli_boundary_bypass", True)
            and not getattr(args, "tli_per_q_head", False)
            and not getattr(args, "tia_enable_async_topk", False)
            and not getattr(args, "tli_proj_basis", None)
            and not getattr(args, "tli_moba", False)
            and getattr(args, "tli_sigma_select", "none") == "none"
            and not getattr(args, "tli_sinkswa_only", False)
            and getattr(args, "tli_near_select", "4bit") == "4bit"
            and getattr(args, "tli_far_select", "4bit") == "4bit"
            and getattr(args, "tli_subspace", "full") != "mla_rope")


def resolve_far_skip_modes(args, *, model=None, config=None):
    """Resolve ``auto`` once from the actual model/config and write it back.

    Missing fields retain the historical explicit legacy/retain semantics.
    Explicit compact/evict never silently degrades on an unsupported model.
    """
    backend = getattr(args, "tli_far_skip_backend", "legacy")
    cache_mode = getattr(args, "tli_far_skip_cache", "retain")
    if backend not in ("auto", "legacy", "compact") or cache_mode not in ("auto", "retain", "evict"):
        raise ValueError("invalid far skip backend/cache mode")
    if model is None and config is None and (backend == "auto" or cache_mode == "auto"):
        raise ValueError("auto far skip modes require model/config before snapshot")
    model_supported = (getattr(args, "method", None) == "tli"
                       and (_model_supports_compact(model) if model is not None
                            else _config_supports_compact(config)))
    supported = model_supported and _args_support_compact(args)
    requested_backend = backend
    if backend == "auto":
        backend = "compact" if supported else "legacy"
    if cache_mode == "auto":
        cache_mode = "evict" if backend == "compact" else "retain"
    if ((backend == "compact" or (requested_backend == "auto" and cache_mode == "evict"))
            and not supported):
        if not model_supported:
            raise ValueError("compact far skip currently supports only TLI Qwen3, "
                             "head_dim=128, without native sliding-window layers")
        raise ValueError("compact far skip requires fixed near_len/pages, near_tokens=off, "
                         "boundary bypass and ordinary shared-head two-level selection")
    if cache_mode == "evict" and backend != "compact":
        raise ValueError("far KV eviction requires compact far skipping")
    args.tli_far_skip_backend = backend
    args.tli_far_skip_cache = cache_mode
    return backend, cache_mode


def configure_far_skip(indexer):
    args = indexer.args
    backend = getattr(args, "tli_far_skip_backend", "legacy")
    cache_mode = getattr(args, "tli_far_skip_cache", "retain")
    if backend not in ("legacy", "compact") or cache_mode not in ("retain", "evict"):
        raise ValueError("unresolved or invalid far skip backend/cache mode")
    if cache_mode == "evict" and backend != "compact":
        raise ValueError("far KV eviction requires compact far skipping")
    indexer.far_skip_backend, indexer.far_skip_cache = backend, cache_mode
    indexer.far_skip_config = None
    if backend == "legacy":
        return
    if (not _args_support_compact(args) or indexer.gamma is not None
            or indexer._basis_all is not None):
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
        staging = getattr(cache, "_direct_prefill", None)
        if staging is not None:
            state = staging.states[module.layer_idx]
            if not state.written and q.shape[-2] == staging.length:
                if mask is not None:
                    raise ValueError("direct prefill requires an unpadded mask-free SDPA forward")
                return None  # Dense prefill continues into DirectPrefillStaging.update().
            raise ValueError("finalize direct prefill before decode or another forward")
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
    from .operator_pipeline import operator_mode, validate_operator_inputs
    mode = operator_mode()
    validate_operator_inputs(q, k, v)
    if mode != 'off' and not 0 < length + 1 < torch.iinfo(torch.int32).max:
        raise ValueError('operator logical lengths must fit positive int32 IDs')
    if idx.far_skip_cache == "evict":
        layer = evict_far(cache, li, cfg)
    k, v = cache.update(k, v, li, {"cache_position": cache_position})
    layout = layer.layout if isinstance(layer, FarSkipLayer) else None
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if mode != 'off':
        from .operator_pipeline import decode_selected
        lengths = torch.tensor([length + 1], device=q.device, dtype=torch.int64)
        def select(query, keys, lengths, scale):
            selection = select_far_skip(query, keys, length + 1, cfg, coarse, fine,
                                        scale=scale, layout=layout, workspace=ws)
            return selection.positions, torch.ones_like(selection.positions, dtype=torch.bool)
        return decode_selected(q, k, v, lengths, module.scaling,
                               cfg=cfg, layout=layout, select=select)
    selection = select_far_skip(q, k, length + 1, cfg, coarse, fine,
                               scale=module.scaling, layout=layout, workspace=ws)
    return attend_far_skip(q, k, v, selection, cfg,
                           scale=module.scaling, layout=layout, workspace=ws)
