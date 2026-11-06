"""Shared prefill/freeze/receipt lifecycle for frozen-region requests."""
from contextlib import contextmanager
from dataclasses import replace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attn.cache.paged_region_cache import (
    GraphReadyPagedRegionCache, RequestPhase)
from sparse_attn.policies.frozen_gate import decide_regions, profile_regions
from sparse_attn.policies.static_gate import StaticGateDecision
from sparse_attn.scheduler.scheduler import greedy_next_tokens


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
        profiles[layer_idx] = profile
        return original(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS['sdpa'] = profiled
    try:
        yield profiles
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


@contextmanager
def _request_failure_guard(cache, request_id):
    """任何请求边界异常都进入 FAILED，且保留原异常继续抛出。"""
    try:
        yield
    except Exception:
        if cache.request_phases[request_id] != RequestPhase.FAILED:
            cache.transition_request(request_id, RequestPhase.FAILED)
        raise


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


def build_frozen_cache(model, geometry, sequence_capacity, *, max_requests=1,
                       max_visible_len=None):
    """为同一批请求创建固定地址 Cache；visible 宽度只约束选择输出而非 prompt。"""
    page = geometry.page
    max_blocks = ((sequence_capacity + geometry.max_new_tokens + page - 1) // page) + 8
    if max_visible_len is None:
        max_visible_len = sequence_capacity + geometry.max_new_tokens + 16
    cfg = model.config
    if getattr(cfg, 'model_type', None) == 'glm4_moe_lite':
        head_dim = int(getattr(cfg, 'qk_nope_head_dim', 0)) + int(
            getattr(cfg, 'qk_rope_head_dim', 0))
    else:
        head_dim = getattr(cfg, 'head_dim', None) or (
            cfg.hidden_size // cfg.num_attention_heads)
    head_dim_v = int(getattr(cfg, 'v_head_dim', None) or head_dim)
    return GraphReadyPagedRegionCache(
        cfg.num_hidden_layers, geometry,
        max_requests=max_requests,
        max_physical_pages=max_requests * max_blocks + 8,
        max_blocks=max_blocks,
        max_visible_len=max_visible_len,
        max_query_tokens=max_requests,
        num_kv_heads=cfg.num_key_value_heads,
        num_query_heads=cfg.num_attention_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim_v,
        dtype=model.dtype,
        device=next(model.parameters()).device,
    )


def prefill_frozen_request(model, input_ids, config, geometry, static_gate=None,
                           suffix_length=0, *, cache=None, request_id=None):
    """Run one-shot dense prefill directly inside GraphReadyPagedRegionCache.

    When ``static_gate`` is None the original request-time calibration runs
    (dynamic frozen gate).  When provided, the per-layer near/far skip decisions
    are read directly from the precomputed artifact.  调用方可传入共享 Cache，
    逐请求 prefill 后统一批量 decode；缺省仍创建独立 B1 Cache。
    """
    n_layers = model.config.num_hidden_layers
    prompt_length = int(input_ids.shape[-1])
    if not isinstance(suffix_length, int) or suffix_length < 0:
        raise ValueError('suffix_length must be a non-negative integer')
    sequence_capacity = prompt_length + suffix_length
    # Decode 会先逐 token 消费 chat suffix，再生成 answer；Cache 的物理容量
    # 必须把 suffix 计入 max_new_tokens，否则最后一 token 会越过 capacity。
    effective_geometry = (
        replace(geometry, max_new_tokens=geometry.max_new_tokens + suffix_length)
        if suffix_length else geometry)
    if cache is None:
        cache = build_frozen_cache(model, effective_geometry, sequence_capacity)
    if cache.n_layers != n_layers or cache.geometry != effective_geometry:
        raise ValueError('shared cache does not match model layers or geometry')
    if sequence_capacity + geometry.max_new_tokens > cache.max_blocks * geometry.page:
        raise ValueError('request exceeds shared cache sequence capacity')
    request_id = cache.add_request(request_id)
    cache.set_active_request(request_id)

    with _request_failure_guard(cache, request_id):
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
        else:
            _validate_static_gate(
                static_gate, n_layers, getattr(model.config, 'model_type', None))
            output = model(input_ids, past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
            skip_far = list(static_gate.skip_far)
            skip_near = list(static_gate.skip_near)
            decisions = {
                layer_idx: {'skip_near': skip_near[layer_idx],
                            'skip_far': skip_far[layer_idx]}
                for layer_idx in range(n_layers)
            }

        cache.freeze(request_id, skip_far, skip_near)
        return output, dict(profiles=None if static_gate is not None else profiles,
                            decisions=decisions,
                            request_id=request_id,
                            material_length=prompt_length,
                            suffix_length=suffix_length)


def decode_frozen_request(model, output, *, max_new_tokens, eos_ids,
                          pad_token_id=0, capture_graph=True,
                          min_new_tokens=0, suffix_ids=None):
    """逐 token 消费完整 suffix，随后只把答案新增 token 交给调用方。"""
    from sparse_attn.engine.graph_runner import GraphRunner
    from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler

    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be non-negative')
    if min_new_tokens < 0:
        raise ValueError('min_new_tokens must be non-negative')
    cache = output.past_key_values
    request_id = cache.active_request_id
    if request_id is None:
        raise ValueError('frozen decode requires an active request')
    eos_ids = frozenset(int(token) for token in eos_ids)
    suffix_ids = () if suffix_ids is None else tuple(
        int(token) for token in suffix_ids)

    with _request_failure_guard(cache, request_id):
        generated = []
        if max_new_tokens == 0:
            cache.transition_request(request_id, RequestPhase.FINISHED)
            return generated, cache

        cache.transition_request(
            request_id, RequestPhase.SUFFIX if suffix_ids else RequestPhase.GENERATE)
        if not suffix_ids:
            first_token = greedy_next_tokens(
                output.logits, eos_ids, generated_counts=0,
                min_new_tokens=min_new_tokens)
            generated.append(int(first_token[0]))
            if (len(generated) >= max_new_tokens
                    or (len(generated) >= min_new_tokens and generated[-1] in eos_ids)):
                cache.transition_request(request_id, RequestPhase.FINISHED)
                return generated, cache

        scheduler = ContinuousBatchScheduler(
            cache=cache, max_batch=1, eos_ids=eos_ids,
            pad_token_id=pad_token_id)
        scheduler.add_request(request_id)
        runner = GraphRunner(
            scheduler=scheduler, buckets=(1,), device=next(model.parameters()).device)
        if capture_graph:
            runner.capture_all(model, cache)

        if suffix_ids:
            scheduler.next_tokens[request_id] = suffix_ids[0]
            for index in range(len(suffix_ids)):
                forced = ([suffix_ids[index + 1]]
                          if index + 1 < len(suffix_ids) else None)
                _, tokens = runner.step(
                    model, cache, generated_counts=0,
                    min_new_tokens=min_new_tokens, return_next_tokens=True,
                    forced_tokens=forced, stop_on_eos=forced is None)
            generated.append(int(tokens[0]))
            if (len(generated) >= max_new_tokens
                    or (len(generated) >= min_new_tokens and generated[-1] in eos_ids)):
                cache.transition_request(request_id, RequestPhase.FINISHED)
                return generated, cache
            cache.transition_request(request_id, RequestPhase.GENERATE)
        else:
            scheduler.next_tokens[request_id] = generated[0]

        while len(generated) < max_new_tokens:
            _, tokens = runner.step(
                model, cache, generated_counts=len(generated),
                min_new_tokens=min_new_tokens, return_next_tokens=True)
            generated.append(int(tokens[0]))
            if len(generated) >= min_new_tokens and generated[-1] in eos_ids:
                break
        cache.transition_request(request_id, RequestPhase.FINISHED)
        return generated, cache


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
    decode_steps = getattr(cache, 'nfa_decode_steps', 0)
    receipts = {
        layer_idx: ({
            'route': 'graph_ready_sparse',
            'visible_lens': cache.visible_lens_buf[
                layer_idx, :cache.max_requests].detach(),
        } if decode_steps > 0 else None)
        for layer_idx in range(cache.n_layers)
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
            expected_steps = (audit.get('suffix_length', 0)
                              + max(0, generated_count - 1))
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
    phase = cache.request_phases.get(audit['request_id'])
    audit.update(gate_counts=counts, nfa_receipts=receipts, final_kv=final,
                 request_phase=None if phase is None else phase.value)
    return _json_safe(audit)
