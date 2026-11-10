"""真实 Qwen RULER decode：NFA / FlashAttention-2 / FlashInfer，支持 B1/2/4 与 Graph。

性能口径固定复制同一条 RULER prompt 到 B 行，并执行固定长度 decode；它不是
质量评测。输入来源由 --input-provenance 显式记录，实际 token 长度必须落在
声明档位的 95%--100%；64K/128K 循环体扩展数据不得冒充原生 RULER。
每个进程只跑一个配置，避免 CUDA Graph 私有池跨配置残留。
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from types import MethodType

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.rope_policy import configure_rope, verify_runtime_rope
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.engine.graph_runner import GraphRunner
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.patches.qwen3_flashinfer_patch import (
    FlashInferQwen3Runtime,
    bind_flashinfer_runtime,
    register_flashinfer_patch,
)
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler, greedy_next_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='niah_single_1')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072, 262144, 524288))
    parser.add_argument('--batch-size', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--method', required=True, choices=('nfa', 'fa2', 'flashinfer'))
    parser.add_argument('--execution', required=True, choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension'),
                        help='输入数据来源必须显式声明，禁止按长度猜测')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--fa2-prefill-cpu-offload', action='store_true',
                        help='仅在 prefill 逐层卸载 KV 以降低峰值；decode 前恢复固定 GPU Cache')
    return parser.parse_args()


def _load_input(args, tokenizer):
    path = os.path.join(args.data_root, str(args.context_length), f'{args.task}.jsonl')
    with open(path, 'rb') as handle:
        data_sha256 = hashlib.sha256(handle.read()).hexdigest()
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    file_row_index = (
        0 if args.input_provenance == 'repeated-needle-body-extension'
        else args.sample_index)
    row = rows[file_row_index]
    ids = tokenizer(row['input'], return_tensors='pt', truncation=False).input_ids
    prompt_tokens = ids.shape[-1]
    if not 0.95 * args.context_length <= prompt_tokens <= args.context_length:
        raise ValueError(
            f'RULER prompt length mismatch: requested {args.context_length}, '
            f'got {prompt_tokens} tokens from {path} sample {args.sample_index}')
    return ids[0].cuda(), {
        'path': os.path.abspath(path), 'sha256': data_sha256,
        'sample_index': args.sample_index, 'sample_id': row.get('index'),
        'file_row_index': file_row_index,
        'actual_prompt_tokens': prompt_tokens,
        'provenance': args.input_provenance,
        'performance_input_mode': 'replicate one prompt across all batch rows',
    }


LONG_PREFILL_MLP_CHUNK = 65536


def _mlp_forward_with_prefill_chunk(original, x, chunk_tokens):
    """MLP 逐 token 独立；仅长 prefill 分块降低激活峰值，decode 原样执行。"""
    if x.ndim < 2 or x.shape[-2] <= chunk_tokens:
        return original(x)
    output = torch.empty_like(x)
    for start in range(0, x.shape[-2], chunk_tokens):
        stop = min(start + chunk_tokens, x.shape[-2])
        output[..., start:stop, :].copy_(original(x[..., start:stop, :]))
    return output


def _enable_long_prefill_mlp_chunk(model):
    for module in model.modules():
        if not isinstance(module, Qwen3MLP):
            continue
        original = module.forward

        def forward(self, x, _original=original):
            return _mlp_forward_with_prefill_chunk(
                _original, x, LONG_PREFILL_MLP_CHUNK)

        module.forward = MethodType(forward, module)


def _load_model(args, max_new_tokens):
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'qwen3':
        raise ValueError('this production benchmark currently requires Qwen3')
    configure_rope(
        config, args.context_length, max_new_tokens,
        yarn=args.context_length > 32768,
        factor={65536: 2.0, 131072: 4.0,
                262144: 8.0, 524288: 16.0}.get(args.context_length))
    config._attn_implementation = (
        'flash_attention_2' if args.method == 'fa2' else 'sdpa')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    if args.method == 'nfa':
        patch_args = nfa_args()
        patched = register_patch(
            model, patch_args, resolve_treatment_snapshot(patch_args),
            rmsnorm_decode=True, mlp_silu_decode=True)
        if patched != config.num_hidden_layers:
            raise RuntimeError(f'NFA patched {patched}/{config.num_hidden_layers} layers')
    elif args.method == 'flashinfer':
        patched = register_flashinfer_patch(model)
        if patched != config.num_hidden_layers:
            raise RuntimeError(
                f'FlashInfer patched {patched}/{config.num_hidden_layers} layers')
    if args.context_length > LONG_PREFILL_MLP_CHUNK:
        _enable_long_prefill_mlp_chunk(model)
    return model


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@contextmanager
def _fa2_no_packed_position_ids():
    """固定无 padding 输入避免 packed D2H；可选在 prefill 注意力后逐层卸载 KV。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
    state = {'prefill_cache': None}

    def forward(module, query, key, value, *args, **kwargs):
        kwargs.pop('position_ids', None)
        result = original(module, query, key, value, *args, **kwargs)
        cache = state['prefill_cache']
        if cache is not None:
            layer = cache.layers[module.layer_idx]
            layer.keys = layer.keys.to('cpu')
            layer.values = layer.values.to('cpu')
        return result

    ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = forward
    try:
        yield state
    finally:
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = original


def _summary(step_ms, warmup):
    steady = step_ms[warmup:] or step_ms
    return {
        'decode_steps': len(step_ms),
        'tpot_ms_median': statistics.median(step_ms),
        'steady_tpot_ms_median': statistics.median(steady),
        'tpot_ms_all': step_ms,
        'warmup_steps_excluded_from_steady': min(warmup, len(step_ms)),
    }


def _memory_snapshot():
    return {
        'allocated_bytes': torch.cuda.memory_allocated(),
        'reserved_bytes': torch.cuda.memory_reserved(),
        'max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


def _source_receipt():
    paths = (
        'sparse_attn', 'benchmark/rope_policy.py',
        'benchmark/efficiency/benchmark_nfa_end_to_end.py',
        'benchmark/efficiency/measure_nfa_selector_e2e.py',
    )
    diff = subprocess.check_output(
        ['git', '-C', REPO, 'diff', 'HEAD', '--binary', '--', *paths])
    with open(__file__, 'rb') as handle:
        harness = handle.read()
    return {
        'head': subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
        'diff_sha256': hashlib.sha256(diff).hexdigest(),
        'harness_sha256': hashlib.sha256(harness).hexdigest(),
    }


def _run_nfa(args, model, prompt, max_new_tokens, eos_ids):
    batch = args.batch_size
    patch_args = nfa_args()
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
        max_new_tokens=max_new_tokens)
    cache = build_frozen_cache(
        model, geometry, prompt.numel(), max_requests=batch,
        max_visible_len=args.visible_len)
    gate = GateConfig()
    generated = {}
    req_ids = []
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    for request_id in range(batch):
        output, _ = prefill_frozen_request(
            model, prompt.unsqueeze(0), gate, geometry,
            cache=cache, request_id=request_id)
        first = greedy_next_tokens(
            output.logits, eos_ids, generated_counts=0, min_new_tokens=1)
        generated[request_id] = [int(first[0])]
        req_ids.append(request_id)
        cache.transition_request(request_id, 'GENERATE')
        del output
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    prefill_memory = _memory_snapshot()

    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=batch, eos_ids=eos_ids, pad_token_id=0)
    for request_id in req_ids:
        scheduler.add_request(request_id)
        scheduler.next_tokens[request_id] = generated[request_id][0]
    runner = GraphRunner(
        scheduler=scheduler, buckets=(batch,), visible_buckets=(args.visible_len,),
        device=prompt.device)
    if args.execution == 'graph':
        runner.capture_all(model, cache)
    decode_ready_memory = _memory_snapshot()

    decode_steps = max_new_tokens - 1
    event_pairs = [_events() for _ in range(decode_steps)]
    token_history = torch.empty(
        (decode_steps, batch), dtype=torch.long, device=prompt.device)
    steady_decode_start = None
    if args.warmup_steps == 0:
        torch.cuda.synchronize()
        steady_decode_start = _memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
    for step in range(decode_steps):
        start, end = event_pairs[step]
        start.record()
        counts = [step + 1] * batch
        _, tokens = runner.step(
            model, cache, generated_counts=counts, min_new_tokens=1,
            return_next_tokens=True, stop_on_eos=False)
        end.record()
        token_history[step].copy_(tokens)
        if step + 1 == args.warmup_steps:
            torch.cuda.synchronize()
            steady_decode_start = _memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
    if steady_decode_start is None:
        steady_decode_start = _memory_snapshot()
    scheduler.prepare_step()
    torch.cuda.synchronize()
    for rows in token_history.cpu().tolist():
        for row, request_id in enumerate(req_ids):
            generated[request_id].append(int(rows[row]))
    step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    result = _summary(step_ms, args.warmup_steps)
    result.update(prefill_seconds=prefill_seconds,
                  generated_token_ids=[generated[request_id] for request_id in req_ids],
                  skip_policy_counts={
                      str(request_id): {
                          'far': sum(cache.states[request_id].skip_far_by_layer),
                          'near': sum(cache.states[request_id].skip_near_by_layer),
                          'both': sum(far and near for far, near in zip(
                              cache.states[request_id].skip_far_by_layer,
                              cache.states[request_id].skip_near_by_layer)),
                      } for request_id in req_ids},
                  gate_mode='dynamic', graph_replays=(len(step_ms) if args.execution == 'graph' else 0),
                  memory={
                      'prefill_end': prefill_memory,
                      'decode_ready': decode_ready_memory,
                      'steady_decode_start': steady_decode_start,
                      'decode_end': _memory_snapshot(),
                  })
    return result


def _run_fa2(args, model, prompt, max_new_tokens, eos_ids, fa2_state):
    batch = args.batch_size
    length = prompt.numel()
    device = prompt.device
    cache = StaticCache(config=model.config, max_cache_len=length + max_new_tokens)
    if not args.fa2_prefill_cpu_offload:
        cache.early_initialization(
            batch, model.config.num_key_value_heads, model.config.head_dim,
            model.dtype, device)
    input_ids = prompt.unsqueeze(0).expand(batch, -1).contiguous()
    cache_position = torch.arange(length, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    fa2_state['prefill_cache'] = cache if args.fa2_prefill_cpu_offload else None
    try:
        output = model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=cache,
            use_cache=True, logits_to_keep=1)
    finally:
        fa2_state['prefill_cache'] = None
    if args.fa2_prefill_cpu_offload:
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
    next_tokens = greedy_next_tokens(
        output.logits, eos_ids, generated_counts=0, min_new_tokens=1)
    generated = [[int(token)] for token in next_tokens.cpu()]
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    del output
    prefill_memory = _memory_snapshot()

    input_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    position_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)
    graph = None
    static_logits = None

    def forward():
        return model(
            input_ids=input_buf, position_ids=position_buf,
            cache_position=cache_position_buf, past_key_values=cache,
            use_cache=True, logits_to_keep=1)

    if args.execution == 'graph':
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length)
        cache_position_buf.fill_(length)
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = forward()
    decode_ready_memory = _memory_snapshot()

    decode_steps = max_new_tokens - 1
    event_pairs = [_events() for _ in range(decode_steps)]
    token_history = torch.empty(
        (decode_steps, batch), dtype=torch.long, device=device)
    steady_decode_start = None
    if args.warmup_steps == 0:
        torch.cuda.synchronize()
        steady_decode_start = _memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
    for step in range(decode_steps):
        start, end = event_pairs[step]
        start.record()
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length + step)
        cache_position_buf.fill_(length + step)
        if graph is None:
            logits = forward().logits
        else:
            graph.replay()
            logits = static_logits.logits
        next_tokens = greedy_next_tokens(
            logits, eos_ids, generated_counts=step + 1, min_new_tokens=1)
        end.record()
        token_history[step].copy_(next_tokens)
        if step + 1 == args.warmup_steps:
            torch.cuda.synchronize()
            steady_decode_start = _memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
    if steady_decode_start is None:
        steady_decode_start = _memory_snapshot()
    torch.cuda.synchronize()
    for rows in token_history.cpu().tolist():
        for row in range(batch):
            generated[row].append(int(rows[row]))
    step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    result = _summary(step_ms, args.warmup_steps)
    result.update(prefill_seconds=prefill_seconds, generated_token_ids=generated,
                  prefill_cpu_offload=args.fa2_prefill_cpu_offload,
                  graph_replays=(len(step_ms) if graph is not None else 0),
                  memory={
                      'prefill_end': prefill_memory,
                      'decode_ready': decode_ready_memory,
                      'steady_decode_start': steady_decode_start,
                      'decode_end': _memory_snapshot(),
                  })
    return result


def _validate_flashinfer_runtime(runtime, execution, expected_page_size):
    """校验实际运行对象，而不是只相信命令行或配置字段。"""
    expected_wrapper = (
        'CUDAGraphBatchDecodeWithPagedKVCacheWrapper' if execution == 'graph'
        else 'BatchDecodeWithPagedKVCacheWrapper')
    actual_wrapper = type(runtime.wrapper).__name__
    cache_shape = tuple(runtime.paged_kv_cache.shape)
    errors = []
    if runtime.execution != execution:
        errors.append(f'execution={runtime.execution}, expected={execution}')
    if runtime.page_size != expected_page_size:
        errors.append(
            f'page_size={runtime.page_size}, expected={expected_page_size}')
    if len(cache_shape) != 6 or cache_shape[2] != 2:
        errors.append(f'invalid paged cache shape={cache_shape}')
    if len(cache_shape) == 6 and cache_shape[3] != expected_page_size:
        errors.append(f'cache page dimension={cache_shape[3]}')
    if actual_wrapper != expected_wrapper:
        errors.append(f'wrapper={actual_wrapper}, expected={expected_wrapper}')
    if runtime.fixed_split_size is None:
        errors.append('FlashInfer plan was not prepared')
    if errors:
        raise RuntimeError('FlashInfer runtime validation failed: ' + '; '.join(errors))


def _run_flashinfer(args, model, prompt, max_new_tokens, eos_ids):
    """完整 Qwen3 prefill/decode；decode 每层真实调用 FlashInfer paged wrapper。"""
    batch = args.batch_size
    length = prompt.numel()
    device = prompt.device
    decode_steps = max_new_tokens - 1
    page_size = nfa_args().tia_block_size
    if page_size != 64:
        raise RuntimeError(
            f'fair FlashInfer comparison requires NFA page64, got {page_size}')
    runtime = FlashInferQwen3Runtime(
        model, batch, length, max_new_tokens, device=device, page_size=page_size)
    bind_flashinfer_runtime(model, runtime)

    input_ids = prompt.unsqueeze(0)
    cache_position = torch.arange(length, device=device)
    position_ids = cache_position.unsqueeze(0)
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    # 与 NFA 一样逐请求 prefill；每层 append 的都是当前模型 post-RoPE K/V。
    all_prefill_batch_indices = runtime.prefill_batch_indices
    all_prefill_positions = runtime.prefill_positions
    next_token_rows = []
    for request_id in range(batch):
        runtime.prefill_batch_indices = all_prefill_batch_indices[
            request_id * length:(request_id + 1) * length]
        runtime.prefill_positions = all_prefill_positions[:length]
        output = model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=runtime.mask_metadata,
            use_cache=True, logits_to_keep=1)
        next_token_rows.append(greedy_next_tokens(
            output.logits, eos_ids, generated_counts=0, min_new_tokens=1))
        del output
    runtime.prefill_batch_indices = all_prefill_batch_indices
    runtime.prefill_positions = all_prefill_positions
    next_tokens = torch.cat(next_token_rows)
    generated = [[int(token)] for token in next_tokens.cpu()]
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    del input_ids, position_ids, cache_position
    prefill_memory = _memory_snapshot()

    runtime.prepare_decode(args.execution, decode_steps)
    _validate_flashinfer_runtime(runtime, args.execution, page_size)
    input_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    position_buf = torch.empty((batch, 1), dtype=torch.long, device=device)
    cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)
    graph = None
    static_logits = None

    def forward():
        return model(
            input_ids=input_buf, position_ids=position_buf,
            cache_position=cache_position_buf,
            past_key_values=runtime.mask_metadata, use_cache=True,
            logits_to_keep=1)

    if args.execution == 'graph':
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length)
        cache_position_buf.fill_(length)
        runtime.prepare_decode_step(0)
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        runtime.prepare_decode_step(0)
        with torch.cuda.graph(graph):
            static_logits = forward()
    decode_ready_memory = _memory_snapshot()

    event_pairs = [_events() for _ in range(decode_steps)]
    token_history = torch.empty(
        (decode_steps, batch), dtype=torch.long, device=device)
    steady_decode_start = None
    if args.warmup_steps == 0:
        torch.cuda.synchronize()
        steady_decode_start = _memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
    for step in range(decode_steps):
        start, end = event_pairs[step]
        start.record()
        input_buf.copy_(next_tokens[:, None])
        position_buf.fill_(length + step)
        cache_position_buf.fill_(length + step)
        runtime.prepare_decode_step(step)
        if graph is None:
            logits = forward().logits
        else:
            graph.replay()
            logits = static_logits.logits
        next_tokens = greedy_next_tokens(
            logits, eos_ids, generated_counts=step + 1, min_new_tokens=1)
        end.record()
        token_history[step].copy_(next_tokens)
        if step + 1 == args.warmup_steps:
            torch.cuda.synchronize()
            steady_decode_start = _memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
    if steady_decode_start is None:
        steady_decode_start = _memory_snapshot()
    torch.cuda.synchronize()
    for rows in token_history.cpu().tolist():
        for row in range(batch):
            generated[row].append(int(rows[row]))
    step_ms = [start.elapsed_time(end) for start, end in event_pairs]

    expected_prefill_calls = batch * runtime.num_layers
    # Graph 稳态不经过 Python；append/run 计数只含 warmup+capture。
    expected_decode_calls = runtime.num_layers * (
        2 if graph is not None else decode_steps)
    if runtime.prefill_append_calls != expected_prefill_calls:
        raise RuntimeError(
            f'FlashInfer prefill append calls={runtime.prefill_append_calls}, '
            f'expected={expected_prefill_calls}')
    if (runtime.decode_append_calls != expected_decode_calls
            or runtime.decode_wrapper_calls != expected_decode_calls):
        raise RuntimeError(
            'FlashInfer decode API call mismatch: '
            f'append={runtime.decode_append_calls}, '
            f'wrapper={runtime.decode_wrapper_calls}, '
            f'expected={expected_decode_calls}')

    result = _summary(step_ms, args.warmup_steps)
    result.update(
        prefill_seconds=prefill_seconds, generated_token_ids=generated,
        prefill_cpu_offload=False,
        graph_capture_count=int(graph is not None),
        graph_replays=(len(step_ms) if graph is not None else 0),
        graph_eager_fallbacks=0,
        flashinfer_runtime={
            'version': __import__('flashinfer').__version__,
            'decode_attention': (
                'flashinfer.append_paged_kv_cache + paged decode wrapper.run'),
            'wrapper': type(runtime.wrapper).__name__,
            'kv_layout': 'NHD',
            'paged_cache_shape_per_layer': list(
                runtime.paged_kv_cache.shape[1:]),
            'page_size': runtime.page_size,
            'nfa_page_size': page_size,
            'page_size_matches_nfa': runtime.page_size == page_size,
            'use_tensor_cores': True,
            'fixed_split_size_pages': runtime.fixed_split_size,
            'fixed_split_size_tokens': (
                runtime.fixed_split_size * runtime.page_size),
            'max_plan_seq_len': runtime.max_seq_len,
            'physical_pages_per_request': runtime.pages_per_request,
            'prefill_append_calls': runtime.prefill_append_calls,
            'python_decode_append_calls': runtime.decode_append_calls,
            'python_decode_wrapper_calls': runtime.decode_wrapper_calls,
            'decode_call_accounting': (
                'Graph: warmup+capture Python calls; measured decode uses '
                'graph.replay. Eager: one append+run per layer per step.'),
            'query_source': (
                'Qwen3 q_proj + q_norm + original apply_rotary_pos_emb'),
            'key_value_source': (
                'Qwen3 k/v_proj + k_norm + original RoPE; post-RoPE K/V appended'),
            'prefill_attention': 'HF SDPA with original Qwen3 RoPE',
            'prefill_kv_cache': 'flashinfer.append_paged_kv_cache',
        },
        memory={
            'prefill_end': prefill_memory,
            'decode_ready': decode_ready_memory,
            'steady_decode_start': steady_decode_start,
            'decode_end': _memory_snapshot(),
        })
    return result


def main():
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError('max-new-tokens must be at least 2 for decode timing')
    if args.fa2_prefill_cpu_offload and args.method != 'fa2':
        raise ValueError('fa2-prefill-cpu-offload is FA2-only')
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = _load_input(args, tokenizer)
    model = _load_model(args, args.max_new_tokens)
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)
    model_loaded_memory = _memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    fa2_scope = _fa2_no_packed_position_ids() if args.method == 'fa2' else nullcontext()
    with torch.inference_mode(), fa2_scope as fa2_state:
        if args.method == 'nfa':
            result = _run_nfa(args, model, prompt, args.max_new_tokens, eos_ids)
        elif args.method == 'fa2':
            result = _run_fa2(args, model, prompt, args.max_new_tokens, eos_ids, fa2_state)
        else:
            result = _run_flashinfer(args, model, prompt, args.max_new_tokens, eos_ids)
    memory = result.setdefault('memory', {})
    memory['model_loaded'] = model_loaded_memory
    # steady_decode_start 在 reset 前保留了 prefill + warmup 的历史峰值；
    # decode_end 则只反映 reset 后的稳态 decode 峰值。
    memory['peak_incremental_allocated_bytes'] = (
        memory['steady_decode_start']['max_allocated_bytes']
        - model_loaded_memory['allocated_bytes'])
    memory['peak_incremental_reserved_bytes'] = (
        memory['steady_decode_start']['max_reserved_bytes']
        - model_loaded_memory['reserved_bytes'])
    memory['steady_decode_resident_incremental_allocated_bytes'] = (
        memory['steady_decode_start']['allocated_bytes']
        - model_loaded_memory['allocated_bytes'])
    memory['steady_decode_resident_incremental_reserved_bytes'] = (
        memory['steady_decode_start']['reserved_bytes']
        - model_loaded_memory['reserved_bytes'])
    memory['steady_decode_peak_incremental_allocated_bytes'] = (
        memory['decode_end']['max_allocated_bytes']
        - memory['steady_decode_start']['allocated_bytes'])
    memory['steady_decode_peak_incremental_reserved_bytes'] = (
        memory['decode_end']['max_reserved_bytes']
        - memory['steady_decode_start']['reserved_bytes'])
    if args.method == 'nfa':
        selector_args = nfa_args()
        result['nfa_selector_config'] = {
            'subspace': selector_args.nfa_subspace,
            'subspace_cmp_ratio': selector_args.tia_level2_cmp_ratio,
            'level1_topk': selector_args.tia_level1_topk,
            'level2_topk': selector_args.tia_level2_topk,
            'page_size': selector_args.tia_block_size,
            'near_pages': selector_args.nfa_near_pages,
            'near_tokens': selector_args.nfa_near_tokens,
            'near_len': selector_args.nfa_near_len,
            'enable_layer_skip': selector_args.nfa_enable_layer_skip,
            'sink_tokens': selector_args.nfa_sink_tokens,
            'sliding_window_size': selector_args.nfa_swa_size,
        }
    result.update(
        schema='qwen-e2e-decode-v2', source=_source_receipt(),
        method=args.method, execution=args.execution, batch_size=args.batch_size,
        context_length=args.context_length, max_new_tokens=args.max_new_tokens,
        warmup_steps=args.warmup_steps, visible_len=args.visible_len,
        task=args.task, sample_index=args.sample_index,
        input_provenance=args.input_provenance,
        long_prefill_mlp_chunk_tokens=(
            LONG_PREFILL_MLP_CHUNK
            if args.context_length > LONG_PREFILL_MLP_CHUNK else None),
        model_path=os.path.abspath(args.model_path), input=input_receipt,
        decode_protocol='fixed length; EOS recorded but not used for early stop',
        prefill_batch_execution=(
            'serial one-request prefills'
            if args.method in ('nfa', 'flashinfer')
            else 'one batched prefill forward'),
        prefill_cross_method_comparable=(
            args.batch_size == 1 and not args.fa2_prefill_cpu_offload),
        prefill_comparison_scope=(
            'B>1 serial NFA/FlashInfer prefill must not be directly compared '
            'with batched FA2 prefill or normalized by dividing by B; compare '
            'decode TPOT instead. B1 compares complete method-specific prefill '
            'only when FA2 CPU offload is disabled.'),
        timing_scope=(
            'CUDA-event elapsed time around each complete decode step: '
            'input/position updates, method-specific host scheduling and '
            'metadata preparation, model/selector execution, and argmax'),
        timing_metric='end-to-end decode TPOT, not pure attention/operator latency',
        host_readback_in_decode_loop=False,
        token_history_copy_inside_timed_events=False,
        final_token_readback='one D2H after all decode steps',
        gpu_hygiene={'status': 'requires external process monitor'},
        transformers_version=__import__('transformers').__version__,
        torch_version=torch.__version__)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('generated_token_ids', 'tpot_ms_all')}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == '__main__':
    main()
