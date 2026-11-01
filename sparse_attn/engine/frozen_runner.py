"""Shared prefill/freeze/receipt lifecycle for frozen-region requests."""
from contextlib import contextmanager

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attn.cache.paged_region_cache import GraphReadyPagedRegionCache
from sparse_attn.policies.frozen_gate import (
    decide_regions, profile_core_tokens, profile_regions)
from sparse_attn.policies.static_gate import StaticGateDecision


@contextmanager
def capture_profiles(cache, geometry, config):
    """Capture each layer exactly once while preserving the native SDPA call."""
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    profiles = {}

    def profiled(module, query, key, value, *args, **kwargs):
        layer_idx = module.layer_idx
        if layer_idx in profiles:
            raise ValueError('multiple prefill chunks unsupported')
        profile = profile_regions(
            query, key, geometry, rows=config.rows,
            scale=kwargs.get('scaling', module.scaling))
        profile['core'] = profile_core_tokens(
            query, key, geometry, rows=config.rows,
            scale=kwargs.get('scaling', module.scaling))
        profiles[layer_idx] = profile
        return original(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS['sdpa'] = profiled
    try:
        yield profiles
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


@contextmanager
def capture_core_tokens(cache, geometry, config):
    """Capture only first-decode seed tokens without running gate profiling."""
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    core = {}

    def profiled(module, query, key, value, *args, **kwargs):
        layer_idx = module.layer_idx
        if layer_idx in core:
            raise ValueError('multiple prefill chunks unsupported')
        core[layer_idx] = profile_core_tokens(
            query, key, geometry, rows=config.rows,
            scale=kwargs.get('scaling', module.scaling))
        return original(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS['sdpa'] = profiled
    try:
        yield core
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


def _validate_static_gate(static_gate, n_layers, model_type):
    if static_gate is None:
        return
    if static_gate.model_type != model_type:
        raise ValueError(
            f"static gate model_type={static_gate.model_type} 与当前模型 "
            f"{model_type} 不一致")
    if static_gate.num_layers != n_layers:
        raise ValueError(
            f"static gate num_layers={static_gate.num_layers} 与当前模型 "
            f"{n_layers} 不一致")


def prefill_frozen_request(model, input_ids, config, geometry, static_gate=None):
    """Run one-shot dense prefill directly inside GraphReadyPagedRegionCache.

    When ``static_gate`` is None the original request-time calibration runs
    (dynamic frozen gate).  When provided, only the first-decode seed tokens
    are profiled and the per-layer near/far skip decisions are read from the
    precomputed artifact.
    """
    n_layers = model.config.num_hidden_layers
    prompt_length = int(input_ids.shape[-1])
    page = geometry.page
    max_blocks = ((prompt_length + geometry.max_new_tokens + page - 1) // page) + 8
    max_visible_len = prompt_length + geometry.max_new_tokens + 16
    device = next(model.parameters()).device
    head_dim = getattr(model.config, 'head_dim', None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    cache = GraphReadyPagedRegionCache(
        n_layers, geometry,
        max_requests=1,
        max_physical_pages=max_blocks + 8,
        max_blocks=max_blocks,
        max_visible_len=max_visible_len,
        max_query_tokens=1,
        num_kv_heads=model.config.num_key_value_heads,
        num_query_heads=model.config.num_attention_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        dtype=model.dtype,
        device=device,
    )
    request_id = cache.add_request()
    cache.set_active_request(request_id)

    if static_gate is None:
        with capture_profiles(cache, geometry, config) as profiles:
            output = model(input_ids, past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
        if len(profiles) != model.config.num_hidden_layers:
            raise ValueError('incomplete native layer profiling')
        decisions = {
            layer_idx: decide_regions(
                profile, config,
                present_near=_region_present_from_length(prompt_length, geometry)[0],
                present_far=_region_present_from_length(prompt_length, geometry)[1])
            for layer_idx, profile in profiles.items()
        }
        skip_far = [decisions[layer_idx]['skip_far'] for layer_idx in range(n_layers)]
        skip_near = [decisions[layer_idx]['skip_near'] for layer_idx in range(n_layers)]
        prefill_topk_by_layer = {}
        for layer_idx, profile in profiles.items():
            ids = []
            if decisions[layer_idx]['skip_far']:
                ids.extend(profile['core'].get('far', []))
            if decisions[layer_idx]['skip_near']:
                ids.extend(profile['core'].get('near', []))
            prefill_topk_by_layer[layer_idx] = sorted(set(ids))
    else:
        _validate_static_gate(
            static_gate, n_layers, getattr(model.config, 'model_type', None))
        with capture_core_tokens(cache, geometry, config) as core:
            output = model(input_ids, past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
        if len(core) != model.config.num_hidden_layers:
            raise ValueError('incomplete native layer core-token profiling')
        skip_far = list(static_gate.skip_far)
        skip_near = list(static_gate.skip_near)
        decisions = {
            layer_idx: {'skip_near': skip_near[layer_idx],
                        'skip_far': skip_far[layer_idx]}
            for layer_idx in range(n_layers)
        }
        prefill_topk_by_layer = {}
        for layer_idx in range(n_layers):
            ids = []
            if skip_far[layer_idx]:
                ids.extend(core[layer_idx].get('far', []))
            if skip_near[layer_idx]:
                ids.extend(core[layer_idx].get('near', []))
            prefill_topk_by_layer[layer_idx] = sorted(set(ids))

    cache.freeze(request_id, skip_far, skip_near, prefill_topk_by_layer)
    return output, dict(profiles=None if static_gate is not None else profiles,
                        decisions=decisions,
                        request_id=request_id, cache=cache)


def _region_present(layer, geometry):
    """Return (near_region_present, far_region_present) for a filled layer."""
    return _region_present_from_length(layer.logical_length, geometry)


def _region_present_from_length(length, geometry):
    sink_hi, far_hi, near_hi = geometry.bounds(length)
    return near_hi > far_hi, far_hi > sink_hi


def _json_safe(value):
    """Convert device-resident receipts once at the request boundary."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def complete_frozen_request(model, cache, generated_count, audit):
    """Verify real decode receipts before a prediction can be committed."""
    final = cache.report()
    receipts = {
        layer_idx: getattr(layer, 'nfa_receipt', None)
        for layer_idx, layer in enumerate(cache.layers)
    }
    if getattr(model, '_frozen_nfa', False) and generated_count > 1:
        if not all(receipts.values()):
            raise ValueError('missing real NFA decode receipts')
    if getattr(cache, 'decode_trace_enabled', False):
        traces = {
            layer_idx: getattr(layer, 'nfa_decode_trace', [])
            for layer_idx, layer in enumerate(cache.layers)
        }
        if getattr(model, '_frozen_nfa', False):
            expected_steps = max(0, generated_count-1)
            for layer_idx, entries in traces.items():
                steps = [entry.get('decode_step') for entry in entries]
                if steps != list(range(expected_steps)):
                    raise ValueError(
                        f'incomplete NFA decode trace at layer {layer_idx}: {steps}')
        audit['nfa_decode_trace'] = traces
    counts = {name: 0 for name in ('keep', 'near', 'far', 'both')}
    for decision in audit['decisions'].values():
        state = ('both' if decision['skip_near'] and decision['skip_far'] else
                 'near' if decision['skip_near'] else
                 'far' if decision['skip_far'] else 'keep')
        counts[state] += 1
    audit.update(gate_counts=counts, nfa_receipts=receipts, final_kv=final)
    return _json_safe(audit)
