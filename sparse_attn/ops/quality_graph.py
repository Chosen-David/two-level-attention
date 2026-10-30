"""Graph-first quality generation, with explicit execution receipts.

0/off is an eager reference, auto prefers supported graphs, 1/on requires one.
Sampling and error checks stay outside capture. Every request owns its cache,
input buffers and frozen gate; a different request never reuses an old graph.
"""
import gc
import os
import time
import threading

import torch

from ..baseline_runtime import BatchCtx, batch_decode_scope
from .cuda_graph_state import graph_state, enable_graph, disable_graph, capture_graph
from .eager_decoding import eager_decoding_check_errors


_warmup_streams = threading.local()


def _request_warmup_stream(device):
    reuse = os.environ.get('TLI_REUSE_WARMUP_STREAM', '1')
    if reuse not in ('0', '1'):
        raise ValueError('TLI_REUSE_WARMUP_STREAM must be 0 or 1')
    if reuse == '0':
        return torch.cuda.Stream(device=device), False
    # cuBLAS workspaces are stream-associated. Bound the number of warmup
    # streams per calling thread/device without retaining request KV or Graphs.
    streams = getattr(_warmup_streams, 'by_device', None)
    if streams is None:
        streams = _warmup_streams.by_device = {}
    key = device.index if device.index is not None else torch.cuda.current_device()
    reused = key in streams
    if not reused:
        streams[key] = torch.cuda.Stream(device=device)
    return streams[key], reused


def graph_policy():
    value = os.environ.get("TLI_CUDA_GRAPH", "auto").lower()
    values = {"0": "off", "off": "off", "1": "on", "on": "on", "auto": "auto"}
    if value not in values:
        raise ValueError("TLI_CUDA_GRAPH must be auto, 0/off, or 1/on")
    return values[value]


def execution(model, actual, reason=None, **kwargs):
    record = dict(requested=graph_policy(), actual=actual, fallback_reason=reason,
                  captures=0, replay_steps=0,
                  operator_mode=os.environ.get("TLI_OPERATOR_MODE", "off"))
    record.update(kwargs)
    model._quality_graph_execution = record
    return record


def unsupported(model, reason):
    if graph_policy() == "on":
        raise RuntimeError("CUDA Graph required but unavailable: " + reason)
    return execution(model, "eager", reason)


def owned_decode_mask(cache, mask, ctx):
    """Only the runner's exact decode buffer may bypass padding rejection."""
    return (cache is not None and ctx is not None
            and getattr(cache, "_quality_decode_ctx", None) is ctx
            and getattr(cache, "_quality_decode_mask", None) is mask
            and mask is not None and mask.ndim == 4
            and tuple(mask.shape[:3]) == (ctx.batch, 1, 1)
            and mask.shape[-1] == cache.get_max_cache_shape())


def _reset_indexes(model):
    for module in model.modules():
        idx = getattr(module, "indexer", None)
        if idx is None:
            continue
        state = (idx.skip_far, idx._dyn_skip_pending, idx._dyn_frozen) \
            if getattr(idx, "dyn_dgate", False) else None
        idx.clear()
        if state is not None:
            idx.skip_far, idx._dyn_skip_pending, idx._dyn_frozen = state


_CAPTURE_ROUTE_FIELDS = (
    "attention_route", "ring_capacity", "batch_index_route",
    "candidate_domain_route", "normalization_route", "normalization_stats_route", "normalization_width_route",
    "ranking_route", "ranking_kth_route", "ranking_gather_route",
    "ranking_fallback", "sparse_selection_fallback", "sparse_selection_fallback_detail",
)


def _clear_capture_routes(model):
    # Clear after warmup too: a receipt must describe the captured forward,
    # rather than an earlier request or a warmup-only branch.
    for module in model.modules():
        idx = getattr(module, "indexer", None)
        if idx is not None:
            for field in _CAPTURE_ROUTE_FIELDS:
                setattr(idx, "_graph_" + field, None)


def _layer_routes(model):
    """Snapshot Python dispatch metadata; never read device tensors here."""
    rows = []
    for name, module in model.named_modules():
        idx = getattr(module, "indexer", None)
        if idx is None:
            continue
        row = {"module": name}
        for field in _CAPTURE_ROUTE_FIELDS:
            value = getattr(idx, "_graph_" + field, None)
            row[field] = value if isinstance(value, (str, bool, int, float)) else None
        for field, attr in (("dynamic_gate", "dyn_dgate"),
                            ("gate_frozen", "_dyn_frozen"), ("skip_far", "skip_far")):
            value = getattr(idx, attr, None)
            row[field] = value if isinstance(value, bool) else None
        rows.append(row)
    return rows


def _capture_routes(model):
    """Capture-specific receipt boundary; eager execution uses its own phase."""
    return _layer_routes(model)


def _live_width_fallback(prompt_lengths, dtype=None):
    # The first decode appends one token. Check host-known lengths before
    # allocation/capture; capacity alone does not imply a supported row width.
    from .sparse_policy import live_normalization_enabled
    if live_normalization_enabled(quality_request=True):
        if dtype is not None and dtype != torch.bfloat16:
            return "live_normalization_requires_bfloat16"
        if any(int(length) + 1 <= 9216 for length in prompt_lengths):
            return "live_normalization_length_below_9217"
    return None


def static_decode(model, cache, gen, lens, eos_ids, max_gen, ctx, *, min_new_tokens=2,
                  online_eos=True):
    """Finish generation from prefill's first tokens using fixed-address KV.

    online_eos=True 保留逐步 token 读回和提前停止；False 是离线固定步数
    口径，不在步间同步读回，也不应用 EOS，最后一次性回收采样 token。
    """
    if type(online_eos) is not bool:
        raise TypeError('online_eos must be bool')
    policy = graph_policy()
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    batch, capacity = len(gen), cache.get_max_cache_shape()
    measure = os.environ.get("TLI_QUALITY_TIMING", "0") == "1"
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if measure:
        synchronize()
        setup_start = time.perf_counter()
    step_ms, replay_events = [], []
    initial_tokens = sum(map(len, gen))
    devices = {p.device for p in model.parameters()}
    record = execution(model, "staged_static", batch_size=batch, capacity=capacity,
                       generation_mode='online_eos' if online_eos else 'fixed_steps',
                       token_readback='per_step' if online_eos else 'final')
    # static_decode 自身就是请求边界，不能依赖调用者一定先跑过 prefill。
    # 新池只保留本请求每层/每行摘要；clear 保留旧 Graph 地址但软失效。
    ctx._tli_row_index_pools = {}
    _reset_indexes(model)
    if max_gen <= 1 or not any(ctx.alive):
        record["actual"] = "no_decode"
        return gen
    reason = ("model_spans_devices" if len(devices) != 1 else
              "non_cuda_device" if device.type != "cuda" else
              "capacity_below_16384" if capacity < 16384 else
              _live_width_fallback(lens, dtype))
    if policy == "on" and reason:
        raise RuntimeError("CUDA Graph required but unavailable: " + reason)
    if reason and policy != "off":
        record["fallback_reason"] = reason
    tokens = torch.full((batch, 1), ctx.pad_id, device=device, dtype=torch.long)
    positions = torch.zeros_like(tokens)
    pos = torch.zeros(batch, device=device, dtype=torch.long)
    lengths = torch.zeros_like(pos)
    alive = torch.ones(batch, device=device, dtype=torch.bool)
    mask = torch.empty(batch, 1, 1, capacity, device=device, dtype=dtype)
    cache._quality_decode_ctx, cache._quality_decode_mask = ctx, mask
    if getattr(cache, "_far_skip_prepared", False):
        cache._far_skip_attention_mask = mask
    snapshots = (list(ctx.seq_lens), list(ctx.gen_lens), list(ctx.alive))
    device_inputs, previous_tokens = None, None
    fixed_samples = None if online_eos else torch.empty(
        (max_gen - 1, batch), device=device, dtype=torch.long)
    output_readback_ms = 0.0

    def restore():
        ctx.seq_lens[:], ctx.gen_lens[:], ctx.alive[:] = snapshots

    def inject(step):
        for b in range(batch):
            if ctx.alive[b]:
                ctx.seq_lens[b] += 1
                ctx.gen_lens[b] += 1
        if max(ctx.seq_lens) > capacity:
            raise ValueError("decode exceeds fixed KV capacity")
        if device_inputs is not None:
            device_inputs.inject(step, previous_tokens)
            return
        if not online_eos and step > 0:
            tokens.copy_(previous_tokens.view(batch, 1))
        else:
            tokens.copy_(torch.tensor([row[-1] if active else ctx.pad_id
                                     for row, active in zip(gen, ctx.alive)],
                                     device=device, dtype=torch.long).view(batch, 1))
        positions.copy_(torch.tensor([n + step for n in lens], device=device).view(batch, 1))
        lengths.copy_(torch.tensor(ctx.seq_lens, device=device))
        pos.copy_(lengths - 1)
        alive.copy_(torch.tensor(ctx.alive, device=device))
        mask.fill_(torch.finfo(dtype).min)
        for b, n in enumerate(ctx.seq_lens):
            mask[b, 0, 0, :n] = 0

    def forward():
        return model(input_ids=tokens, position_ids=positions, attention_mask=mask,
                     past_key_values=cache, use_cache=True)

    gs = graph_state()
    if gs.enabled:
        raise RuntimeError("nested quality graph requests are unsupported")
    graph, output = None, None
    try:
        eager_decoding_check_errors()
        with batch_decode_scope(ctx):
            # Graph 与高效 eager 共用固定地址元数据、动态长度、增量索引和
            # compute_mask_graph；policy 只决定是否 capture，不改变算法。
            if reason is None:
                enable_graph(batch, quality_defaults=True)
                gs.pos_buf, gs.valid_len_buf, gs.alive_buf = pos, lengths, alive
            record['decode_index_route'] = (
                'request_row_incremental' if gs.enabled else 'legacy_rebuild')
            if policy != "off" and reason is None:
                try:
                    inject(0)
                    stream, reused = _request_warmup_stream(device)
                    record.update(warmup_stream=int(stream.cuda_stream),
                                  warmup_stream_reused=reused)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            forward()
                    torch.cuda.current_stream(device).wait_stream(stream)
                    eager_decoding_check_errors()
                    if not gs.enabled:
                        raise NotImplementedError("selector disabled Graph during warmup")
                    graph = torch.cuda.CUDAGraph()
                    _clear_capture_routes(model)
                    with capture_graph(graph):
                        output = forward()
                    capture_layers = _capture_routes(model)
                    record.update(actual="graph", captures=1,
                                  budget_scope="not_measured_during_replay",
                                  capture_layers=capture_layers)
                except Exception as exc:
                    # OOM/capture 失败不应把 eager 静默换成另一种全量算法；
                    # selector 自己禁用 tensor 路径时则保留明确的 legacy 回退。
                    reuse_tensor_eager = gs.enabled
                    disable_graph()
                    graph, output = None, None
                    restore()
                    _reset_indexes(model)
                    ctx._tli_row_index_pools = {}
                    if reuse_tensor_eager:
                        enable_graph(batch, quality_defaults=True)
                        gs.pos_buf, gs.valid_len_buf, gs.alive_buf = pos, lengths, alive
                    record['decode_index_route'] = (
                        'request_row_incremental' if gs.enabled else 'legacy_rebuild')
                    record['eager_index_fallback'] = None if gs.enabled else 'selector_disabled_tensor_route'
                    # Deferred attention failures and poisoned CUDA contexts
                    # must never be hidden by an eager retry.
                    fatal = any(x in str(exc).lower() for x in (
                        "eager_decoding_attn:", "illegal memory access", "device-side assert"))
                    if policy == "on" or fatal:
                        raise
                    record["fallback_reason"] = f"capture_failed:{type(exc).__name__}:{exc}"
                    gc.collect()
                    torch.cuda.empty_cache()
                restore()
            input_mode = os.environ.get('TLI_DEVICE_DECODE_INPUTS', '1')
            if input_mode not in ('0', '1'):
                raise ValueError('TLI_DEVICE_DECODE_INPUTS must be 0 or 1')
            record.update(input_update_requested=input_mode,
                          input_update_route='host_reference', input_update_fallback=None)
            if input_mode == '1':
                if device.type != 'cuda':
                    record['input_update_fallback'] = 'non_cuda_device'
                else:
                    from .quality_decode_inputs import DeviceDecodeInputs
                    device_inputs = DeviceDecodeInputs(
                        tokens, positions, pos, lengths, alive, mask,
                        ctx.seq_lens, lens, [len(row) for row in gen], ctx.alive,
                        ctx.pad_id, eos_ids if online_eos else set(), min_new_tokens)
                    previous_tokens = torch.tensor(
                        [row[-1] if active else ctx.pad_id for row, active in zip(gen, ctx.alive)],
                        device=device, dtype=torch.long)
                    # Compile the two input-update variants in request setup.
                    # No model/KV write runs here; decode step zero resets all
                    # mutable row metadata from the immutable request seed.
                    device_inputs.inject(0, previous_tokens)
                    device_inputs.inject(1, previous_tokens)
                    record['input_update_route'] = 'device_row_fused'
            if measure:
                synchronize()
                setup_ms = (time.perf_counter() - setup_start) * 1000
                # Allocate/initialize events outside each measured decode step.
                if graph is not None:
                    replay_events = [(torch.cuda.Event(enable_timing=True),
                                      torch.cuda.Event(enable_timing=True))
                                     for _ in range(max_gen - 1)]
                    for begin, end in replay_events:
                        begin.record()
                        end.record()
                    synchronize()
                decode_start = time.perf_counter()
            steps_run = 0
            for step in range(max_gen - 1):
                if online_eos and not any(ctx.alive):
                    break
                if measure:
                    step_start = time.perf_counter()
                inject(step)
                if graph is not None:
                    if measure:
                        replay_events[step][0].record()
                    graph.replay()
                    if measure:
                        replay_events[step][1].record()
                    record["replay_steps"] += 1
                else:
                    if step == 0:
                        _clear_capture_routes(model)
                    output = forward()
                    if step == 0:
                        record["eager_layers"] = _layer_routes(model)
                        record['decode_index_route'] = (
                            'request_row_incremental' if gs.enabled else 'legacy_rebuild')
                        if not gs.enabled:
                            record['eager_index_fallback'] = 'selector_disabled_tensor_route'
                steps_run += 1
                previous_tokens = output.logits[:, -1].argmax(-1)
                if online_eos:
                    next_tokens = previous_tokens.tolist()
                    for b, token in enumerate(next_tokens):
                        if ctx.alive[b]:
                            gen[b].append(token)
                            if token in eos_ids and len(gen[b]) >= min_new_tokens:
                                ctx.alive[b] = False
                else:
                    # 离线固定步数只保留设备端采样链，避免每步 .tolist()
                    # 同步；最终统一读回，且明确不应用 EOS。
                    fixed_samples[step].copy_(previous_tokens)
                if measure:
                    step_ms.append((time.perf_counter() - step_start) * 1000)
            if not online_eos:
                if measure:
                    synchronize()
                    readback_start = time.perf_counter()
                sampled_rows = fixed_samples[:steps_run].t().tolist()
                for b, row in enumerate(sampled_rows):
                    if ctx.alive[b]:
                        gen[b].extend(row)
                if measure:
                    output_readback_ms = (time.perf_counter() - readback_start) * 1000
            eager_decoding_check_errors()
            if measure:
                synchronize()
                decode_ms = (time.perf_counter() - decode_start) * 1000
                replay_ms = [begin.elapsed_time(end) for begin, end in
                             replay_events[:record["replay_steps"]]]
                record["timing"] = dict(
                    scope="static decode; excludes prefill and cache conversion",
                    instrumentation=("CUDA events around replay; online host clock includes inject, argmax, per-step readback and EOS; "
                                     "fixed-step decode includes one final readback, reported separately; final error check included"),
                    step_timing_scope=("host_submit_only" if not online_eos else
                                       "inject_forward_argmax_readback_eos"),
                    setup_wall_ms=setup_ms, decode_wall_ms=decode_ms,
                    step_wall_ms=step_ms, replay_gpu_ms=replay_ms,
                    decode_steps=steps_run,
                    generated_decode_tokens=sum(map(len, gen)) - initial_tokens,
                    output_readback_wall_ms=output_readback_ms,
                    tpot_wall_ms=((sum(step_ms) / len(step_ms)) if online_eos and step_ms else
                                  (decode_ms / steps_run if steps_run else None)),
                    replay_gpu_mean_ms=sum(replay_ms) / len(replay_ms) if replay_ms else None)
            print(f"[quality-graph] {record}", flush=True)
            return gen
    finally:
        disable_graph()
        # This entry owns a one-shot Graph and never returns it. Low-level
        # exact owners deliberately retain KV for same-storage recapture, but
        # retaining them here pins the completed request through the next
        # prefill. Any still-live local Graph continues to own its state.
        for module in model.modules():
            idx = getattr(module, "indexer", None)
            if idx is not None and getattr(idx, "_exact_linear_batch", None) is not None:
                idx._exact_linear_batch = None
        cache._quality_decode_ctx = cache._quality_decode_mask = None
        if getattr(cache, "_far_skip_prepared", False):
            cache._far_skip_attention_mask = None


def from_dynamic(model, past, first_token, max_gen, eos_ids, *, min_new_tokens=2,
                 online_eos=True):
    """Own a B1 prefill cache and return tokens, or None before taking ownership."""
    policy = graph_policy()
    if policy == "off":
        execution(model, "eager")
        return None
    if max_gen <= 1 or (min_new_tokens == 1 and first_token in eos_ids):
        execution(model, "no_decode")
        return [first_token]
    modules = list(model.modules()) if hasattr(model, "modules") else []
    indexes = [m.indexer for m in modules if hasattr(m, "indexer")]
    reason = None
    if not indexes or any(type(idx).__name__ != "TLIIndexer" for idx in indexes):
        reason = "method_has_no_supported_tli_indexer"
    elif getattr(model.config, "model_type", None) not in ("qwen3", "glm4_moe_lite"):
        reason = "model_not_in_quality_graph_allowlist"
    elif any(getattr(idx, "far_skip_backend", "legacy") not in ("legacy", "compact")
             for idx in indexes):
        reason = "unsupported_far_skip_backend"
    elif any(getattr(idx, "far_skip_backend", "legacy") == "compact" and (
            model.config.model_type != "qwen3" or getattr(idx, "far_skip_cache", None) != "evict")
            for idx in indexes):
        reason = "single_request_compact_graph_requires_qwen_eviction"
    elif len({p.device for p in model.parameters()}) != 1:
        reason = "model_spans_devices"
    elif next(model.parameters()).device.type != "cuda":
        reason = "non_cuda_device"
    elif past.get_seq_length() + max_gen < 16384:
        reason = "capacity_below_16384"
    else:
        reason = _live_width_fallback([past.get_seq_length()], next(model.parameters()).dtype)
    if reason:
        unsupported(model, reason)
        return None
    from .tli_static_cache import TLIStaticCache, fill_static_from_dynamic
    length = int(past.get_seq_length())
    capacity = length + max_gen
    # Admission only covers the additional static KV allocation, not an
    # unmeasured promise about graph workspace. Capture OOM may fall back.
    needed = sum((layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
                 for layer in past.layers) * capacity // length
    free, _ = torch.cuda.mem_get_info(next(model.parameters()).device)
    if needed > free * 0.9:
        unsupported(model, "insufficient_memory_for_static_kv")
        return None
    eager_decoding_check_errors()
    measure = os.environ.get("TLI_QUALITY_TIMING", "0") == "1"
    if measure:
        torch.cuda.synchronize(next(model.parameters()).device)
        conversion_start = time.perf_counter()
    cache = TLIStaticCache(model.config, capacity, 1)
    # The caller's explicit compact configuration owns eviction; ordinary
    # quality requests continue to allocate linear storage. Freeze the gate
    # and seed request-local ring summaries before capture or source release.
    fill_static_from_dynamic(cache, past, [length], model=model)
    # Transfer is now complete. Release dynamic KV before capture; failures
    # after ownership transfer stay on this static cache or abort the request.
    past.layers.clear()
    ctx = BatchCtx(1, [length], [0], [True], 0)
    if measure:
        torch.cuda.synchronize(next(model.parameters()).device)
        conversion_ms = (time.perf_counter() - conversion_start) * 1000
    result = static_decode(model, cache, [[first_token]], [length], set(eos_ids),
                           max_gen, ctx, min_new_tokens=min_new_tokens,
                           online_eos=online_eos)[0]
    if measure and "timing" in model._quality_graph_execution:
        model._quality_graph_execution["timing"]["cache_conversion_wall_ms"] = conversion_ms
    return result
