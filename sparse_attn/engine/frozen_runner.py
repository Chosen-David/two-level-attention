"""Shared prefill/freeze/receipt lifecycle for frozen-region requests."""
from contextlib import contextmanager

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attn.cache.paged_region_cache import PagedRegionCache
from sparse_attn.policies.frozen_gate import decide_regions, profile_regions


@contextmanager
def capture_profiles(cache, geometry, config):
    """Capture each layer exactly once while preserving the native SDPA call."""
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    profiles = {}

    def profiled(module, query, key, value, *args, **kwargs):
        layer_idx = module.layer_idx
        if not cache.layers[layer_idx].frozen:
            if layer_idx in profiles:
                raise ValueError('multiple prefill chunks unsupported')
            profiles[layer_idx] = profile_regions(
                query, key, geometry, rows=config.rows,
                scale=kwargs.get('scaling', module.scaling))
        return original(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS['sdpa'] = profiled
    try:
        yield profiles
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


def prefill_frozen_request(model, input_ids, config, geometry):
    """Run one-shot dense prefill, then freeze independent per-layer decisions."""
    cache = PagedRegionCache(model.config.num_hidden_layers, geometry)
    cache.nfa_regions = bool(getattr(model, '_frozen_nfa', False))
    if cache.nfa_regions:
        for module in model.model.layers:
            module.self_attn.indexer.clear()
            module.self_attn.indexer.skip_far = False
    with capture_profiles(cache, geometry, config) as profiles:
        output = model(input_ids, past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
    if len(profiles) != model.config.num_hidden_layers:
        raise ValueError('incomplete native layer profiling')
    before = cache.report()
    decisions = {
        layer_idx: decide_regions(
            profile, config,
            present_near=_region_present(cache.layers[layer_idx], geometry)[0],
            present_far=_region_present(cache.layers[layer_idx], geometry)[1])
        for layer_idx, profile in profiles.items()
    }
    cache.freeze(decisions)
    after = cache.report()
    for layer_idx, decision in decisions.items():
        if decision['skip_near'] and not decision['skip_far']:
            before_bytes = before['layers'][layer_idx]['kv_bytes']
            after_bytes = after['layers'][layer_idx]['kv_bytes']
            if before_bytes != after_bytes:
                raise ValueError('near-only must not claim a resident KV reduction')
    return output, dict(profiles=profiles, decisions=decisions,
                        prefill_full_kv=before, frozen_kv=after)


def _region_present(layer, geometry):
    """Return (near_region_present, far_region_present) for a filled layer."""
    sink_hi, far_hi, near_hi = geometry.bounds(layer.logical_length)
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
