"""真实 Qwen 长文本 decode：NFA / FlashAttention-2 / FlashInfer，支持 B1/2/4 与 Graph。

性能口径固定复制同一条 prompt 到 B 行，并执行固定长度 decode；它不是质量评测。
输入来源由 --input-provenance 显式记录；原生/扩展 RULER 的实际 token 长度必须
落在声明档位的 95%--100%，LongBench-v2 保留真实长度且不得冒充原生 RULER。
每个进程只跑一个配置，避免 CUDA Graph 私有池跨配置残留。
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

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
    parser.add_argument('--task', default='niah_multikey_1')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--batch-size', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--method', required=True, choices=('nfa', 'fa2', 'flashinfer'))
    parser.add_argument('--execution', required=True, choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension',
                                 'longbench-v2'),
                        help='输入数据来源必须显式声明，禁止按长度猜测')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--gate-epsilon', type=float, default=0.05,
                        help='NFA prefill 区域 gate 阈值；0 用于同输入 no-skip 对照')
    parser.add_argument(
        '--measurement-mode', default='latency',
        choices=('latency', 'profile', 'traffic', 'correctness'),
        help='latency/correctness 在 decode 后统一同步读取；其他模式保留逐步回执')
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
    row = rows[args.sample_index]
    ids = tokenizer(row['input'], return_tensors='pt', truncation=False).input_ids
    prompt_tokens = ids.shape[-1]
    min_tokens = (8192 if args.input_provenance == 'longbench-v2'
                  else 0.95 * args.context_length)
    if not min_tokens <= prompt_tokens <= args.context_length:
        raise ValueError(
            f'prompt length mismatch: requested {args.context_length}, '
            f'minimum {min_tokens}, got {prompt_tokens} tokens from '
            f'{path} sample {args.sample_index}')
    return ids[0].cuda(), {
        'path': os.path.abspath(path), 'sha256': data_sha256,
        'sample_index': args.sample_index, 'sample_id': row.get('index'),
        'actual_prompt_tokens': prompt_tokens,
        'provenance': args.input_provenance,
        'performance_input_mode': 'replicate one prompt across all batch rows',
    }


def _load_model(args, max_new_tokens):
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'qwen3':
        raise ValueError('this production benchmark currently requires Qwen3')
    configure_rope(config, args.context_length, max_new_tokens,
                   yarn=args.context_length > 32768,
                   factor={65536: 2.0, 131072: 4.0}.get(args.context_length))
    config._attn_implementation = (
        'flash_attention_2' if args.method == 'fa2' else 'sdpa')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    if args.method == 'nfa':
        patch_args = nfa_args()
        # 两种执行必须使用同一逐位等价的 decode 融合链；仅 Graph 开启会在
        # 长生成中累积数值差异并改变 argmax，破坏 eager/Graph token 身份。
        patched = register_patch(
            model, patch_args, resolve_treatment_snapshot(patch_args),
            rmsnorm_decode=True,
            mlp_silu_decode=True)
        if patched != config.num_hidden_layers:
            raise RuntimeError(f'NFA patched {patched}/{config.num_hidden_layers} layers')
    elif args.method == 'flashinfer':
        patched = register_flashinfer_patch(model)
        if patched != config.num_hidden_layers:
            raise RuntimeError(
                f'FlashInfer patched {patched}/{config.num_hidden_layers} layers')
    return model


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@contextmanager
def _fa2_unpacked_position_ids():
    """已知无 padding/非 packed 输入走原生 FA2 regular kernel，避免 packed 检测 D2H。"""
    from transformers import modeling_flash_attention_utils as fa_utils
    original = fa_utils._is_packed_sequence
    # Qwen3 在调用 attention 前已用 position_ids 完成 RoPE；这里只关闭 HF
    # 对 packed 序列的检测，不丢弃模型 position_ids，也不改变 RoPE/FA2 kernel。
    fa_utils._is_packed_sequence = lambda position_ids, batch_size: False
    try:
        yield
    finally:
        fa_utils._is_packed_sequence = original


@contextmanager
def _fa2_prefill_cpu_offload(cache):
    """仅在 FA2 prefill 注意力后逐层卸载 KV；decode 不保留额外 Python wrapper。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['flash_attention_2']

    def forward(module, query, key, value, *args, **kwargs):
        result = original(module, query, key, value, *args, **kwargs)
        layer = cache.layers[module.layer_idx]
        layer.keys = layer.keys.to('cpu')
        layer.values = layer.values.to('cpu')
        return result

    ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = forward
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = original


def _summary(step_ms, warmup):
    steady = step_ms[warmup:] or step_ms
    steady_tpot_ms_median = statistics.median(steady)
    return {
        'decode_steps': len(step_ms),
        'tpot_ms_median': statistics.median(step_ms),
        'steady_tpot_ms_median': steady_tpot_ms_median,
        # 每行 token 生成速率；batch 聚合吞吐不得混入此字段。
        'throughput_tokens_per_second': 1000.0 / steady_tpot_ms_median,
        'tpot_ms_all': step_ms,
        'warmup_steps_excluded_from_steady': min(warmup, len(step_ms)),
    }


def _strict_region_sets(length, geometry):
    """独立计算首步 P/R/X；不得调用 cache 或 selector 的区间实现。"""
    sink_hi = min(geometry.sink, length)
    near_hi = max(sink_hi, length - geometry.window)
    far_hi = min(near_hi, max(
        sink_hi,
        ((length - geometry.window - geometry.near) // geometry.page)
        * geometry.page))
    protected = set(range(sink_hi)) | set(range(near_hi, length))
    far = set(range(sink_hi, far_hi))
    near = set(range(far_hi, near_hi))
    return protected, far, near


def _strict_skip_receipt(cache, geometry, req_ids, gather, visible, planned_length):
    """把首步物理 gather 反解为逻辑 ID，并独立检查 strict-skip 集合合同。"""
    gather = gather.detach().cpu()
    visible = visible.detach().cpu()
    block_tables = cache.block_table_buf.detach().cpu()
    page = geometry.page
    reports = []
    violations = []

    def violation(layer, request_id, kv_head, message):
        violations.append({
            'layer': layer, 'request_id': request_id,
            'kv_head': kv_head, 'message': message,
        })

    for layer in range(cache.n_layers):
        for row, request_id in enumerate(req_ids):
            state = cache.states[request_id]
            skip_far = bool(state.skip_far_by_layer[layer])
            skip_near = bool(state.skip_near_by_layer[layer])
            protected, far, near = _strict_region_sets(planned_length, geometry)
            excluded = (far if skip_far else set()) | (near if skip_near else set())
            allowed = protected | far | near
            table = block_tables[layer, request_id].tolist()
            pages = (planned_length + page - 1) // page
            inverse = {
                int(physical): logical
                for logical, physical in enumerate(table[:pages])
                if int(physical) != cache.layers[layer].PAD_PAGE_ID
            }
            logical_by_kv = []
            physical_by_kv = []
            for kv_head in range(gather.shape[2]):
                count = int(visible[layer, row, kv_head])
                slots = [int(slot) for slot in gather[layer, row, kv_head, :count]]
                logical_ids = []
                if len(slots) != len(set(slots)):
                    violation(layer, request_id, kv_head, 'duplicate physical gather slots')
                for slot in slots:
                    if slot < 0:
                        violation(layer, request_id, kv_head, f'invalid gather slot {slot}')
                        continue
                    physical_page, offset = divmod(slot, page)
                    logical_page = inverse.get(physical_page)
                    if logical_page is None:
                        violation(layer, request_id, kv_head,
                                  f'physical page {physical_page} is not owned by request')
                        continue
                    logical_id = logical_page * page + offset
                    logical_ids.append(logical_id)
                    if table[logical_page] * page + offset != slot:
                        violation(layer, request_id, kv_head,
                                  f'logical/physical mapping mismatch at {logical_id}')
                attended = set(logical_ids)
                if len(logical_ids) != len(attended):
                    violation(layer, request_id, kv_head, 'duplicate logical token IDs')
                if not protected <= attended:
                    violation(layer, request_id, kv_head, 'protected tokens are missing')
                if attended & excluded:
                    violation(layer, request_id, kv_head,
                              f'frozen tokens re-entered: {sorted(attended & excluded)}')
                if not attended <= allowed:
                    violation(layer, request_id, kv_head, 'token outside P/R or current length')
                if skip_far and skip_near and attended != protected:
                    violation(layer, request_id, kv_head, 'both-skip set is not exactly protected')
                logical_by_kv.append(logical_ids)
                physical_by_kv.append(slots)
            reports.append({
                'layer': layer, 'request_id': request_id,
                'skip_far': skip_far, 'skip_near': skip_near,
                'block_table': table[:pages],
                'logical_ids_by_kv_head': logical_by_kv,
                'physical_slots_by_kv_head': physical_by_kv,
            })
    return {
        'valid': not violations,
        'planned_length': planned_length,
        'oracle': 'independent P/R/X plus logical-to-physical address inversion',
        'layers': reports,
        'violations': violations,
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
    gate = GateConfig(epsilon=args.gate_epsilon)
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

    step_ms = []
    clean_decode = args.measurement_mode in ('latency', 'correctness')
    decode_steps = max_new_tokens - 1
    event_pairs = [_events() for _ in range(decode_steps)] if clean_decode else None
    token_history = (torch.empty(
        (decode_steps, batch), dtype=torch.long, device=prompt.device)
        if clean_decode else None)
    capture_strict = args.measurement_mode == 'correctness'
    first_gather = (torch.empty(
        (cache.n_layers, batch, cache.num_kv_heads, cache.active_visible_bucket),
        dtype=cache.gather_indices_buf.dtype, device=prompt.device)
        if capture_strict else None)
    first_visible = (torch.empty(
        (cache.n_layers, batch, cache.num_kv_heads), dtype=torch.int32,
        device=prompt.device) if capture_strict else None)
    for step in range(decode_steps):
        if clean_decode:
            counts = [step + 1] * batch
            start, end = event_pairs[step]
        else:
            counts = [len(generated[request_id]) for request_id in req_ids]
            start, end = _events()
        start.record()
        _, tokens = runner.step(
            model, cache, generated_counts=counts, min_new_tokens=1,
            return_next_tokens=True, stop_on_eos=False)
        end.record()
        if capture_strict and step == 0:
            for layer in range(cache.n_layers):
                first_gather[layer].copy_(
                    cache.gather_indices_for_layer(layer)[req_ids])
                first_visible[layer].copy_(
                    cache.visible_lens_for_layer(layer)[req_ids])
        if clean_decode:
            token_history[step].copy_(tokens)
        else:
            rows = tokens.detach().cpu().tolist()
            for row, request_id in enumerate(req_ids):
                generated[request_id].append(int(rows[row]))
            step_ms.append(start.elapsed_time(end))
    scheduler.prepare_step()
    if clean_decode:
        torch.cuda.synchronize()
        history = token_history.cpu().tolist()
        for rows in history:
            for row, request_id in enumerate(req_ids):
                generated[request_id].append(int(rows[row]))
        step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    result = _summary(step_ms, args.warmup_steps)
    skip_layers = {
        str(request_id): {
            'far': [layer for layer, skip in enumerate(
                cache.states[request_id].skip_far_by_layer) if skip],
            'near': [layer for layer, skip in enumerate(
                cache.states[request_id].skip_near_by_layer) if skip],
        } for request_id in req_ids}
    # 计时循环结束后一次 D2H：这些最终步 visible_lens 是 selector 写给
    # sparse attention 的实际消费缓冲，不只是 prefill gate flags。
    final_visible_lens = torch.stack([
        cache.visible_lens_for_layer(layer)
        for layer in range(cache.n_layers)
    ])[:, req_ids].detach().cpu().tolist()
    strict_receipt = (_strict_skip_receipt(
        cache, geometry, req_ids, first_gather, first_visible,
        planned_length=prompt.numel() + 1) if capture_strict else None)
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
                  skip_policy_layers=skip_layers,
                  skip_decode_receipt={
                      'nfa_decode_steps': cache.nfa_decode_steps,
                      'request_ids': req_ids,
                      'layer_major_final_visible_lens': final_visible_lens,
                  },
                  gate_mode='dynamic', gate_epsilon=args.gate_epsilon,
                  graph_replays=(len(step_ms) if args.execution == 'graph' else 0))
    if strict_receipt is not None:
        result['strict_skip_first_step'] = strict_receipt
    return result


def _run_fa2(args, model, prompt, max_new_tokens, eos_ids):
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
    offload_scope = (
        _fa2_prefill_cpu_offload(cache)
        if args.fa2_prefill_cpu_offload else nullcontext())
    with offload_scope:
        output = model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=cache,
            use_cache=True, logits_to_keep=1)
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

    step_ms = []
    clean_decode = args.measurement_mode in ('latency', 'correctness')
    decode_steps = max_new_tokens - 1
    event_pairs = [_events() for _ in range(decode_steps)] if clean_decode else None
    token_history = (torch.empty(
        (decode_steps, batch), dtype=torch.long, device=device)
        if clean_decode else None)
    for step in range(decode_steps):
        start, end = event_pairs[step] if clean_decode else _events()
        start.record()
        # 与 NFA GraphRunner.step 对齐：输入、位置更新也计入完整 decode step。
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
        if clean_decode:
            token_history[step].copy_(next_tokens)
        else:
            rows = next_tokens.cpu().tolist()
            for row in range(batch):
                generated[row].append(int(rows[row]))
            step_ms.append(start.elapsed_time(end))
    if clean_decode:
        torch.cuda.synchronize()
        history = token_history.cpu().tolist()
        for rows in history:
            for row in range(batch):
                generated[row].append(int(rows[row]))
        step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    result = _summary(step_ms, args.warmup_steps)
    result.update(prefill_seconds=prefill_seconds, generated_token_ids=generated,
                  prefill_cpu_offload=args.fa2_prefill_cpu_offload,
                  graph_replays=(len(step_ms) if graph is not None else 0))
    return result


def _run_flashinfer(args, model, prompt, max_new_tokens, eos_ids):
    """完整 Qwen3 decode：页表更新、KV append、FlashInfer、所有层与 argmax。"""
    batch = args.batch_size
    length = prompt.numel()
    device = prompt.device
    decode_steps = max_new_tokens - 1
    runtime = FlashInferQwen3Runtime(
        model, batch, length, max_new_tokens, device=device)
    bind_flashinfer_runtime(model, runtime)

    input_ids = prompt.unsqueeze(0)
    cache_position = torch.arange(length, device=device)
    position_ids = cache_position.unsqueeze(0)
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    # 与 NFA 一致逐请求 prefill，避免 128K B4 的 SDPA 激活同时驻留而 OOM。
    # runtime 的 append 索引切到当前请求；decode wrapper、页表和计时口径不变。
    all_prefill_batch_indices = runtime.prefill_batch_indices
    all_prefill_positions = runtime.prefill_positions
    next_token_rows = []
    for request_id in range(batch):
        runtime.prefill_batch_indices = all_prefill_batch_indices[
            request_id * length:(request_id + 1) * length]
        runtime.prefill_positions = all_prefill_positions[:length]
        # mask_metadata 只抑制 HF 的 packed/mask 误构，不保存 K/V；独立
        # runtime 在 patched attention 内完成真实 paged append。
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

    runtime.prepare_decode(args.execution, decode_steps)
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
        # FlashInfer plan 已在 prepare_decode 中完成；capture 仅包含 append/run。
        runtime.prepare_decode_step(0)
        with torch.cuda.graph(graph):
            static_logits = forward()

    step_ms = []
    clean_decode = args.measurement_mode in ('latency', 'correctness')
    event_pairs = [_events() for _ in range(decode_steps)] if clean_decode else None
    token_history = (torch.empty(
        (decode_steps, batch), dtype=torch.long, device=device)
        if clean_decode else None)
    for step in range(decode_steps):
        start, end = event_pairs[step] if clean_decode else _events()
        start.record()
        # 与 NFA/FA2 同口径：输入、位置、页表和 slot mapping 均在事件内。
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
        if clean_decode:
            token_history[step].copy_(next_tokens)
        else:
            rows = next_tokens.cpu().tolist()
            for row in range(batch):
                generated[row].append(int(rows[row]))
            step_ms.append(start.elapsed_time(end))
    if clean_decode:
        torch.cuda.synchronize()
        history = token_history.cpu().tolist()
        for rows in history:
            for row in range(batch):
                generated[row].append(int(rows[row]))
        step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    result = _summary(step_ms, args.warmup_steps)
    result.update(
        prefill_seconds=prefill_seconds, generated_token_ids=generated,
        prefill_cpu_offload=False,
        graph_replays=(len(step_ms) if graph is not None else 0),
        flashinfer_runtime={
            'version': __import__('flashinfer').__version__,
            'decode_attention': 'FlashInfer paged KV decode wrapper',
            'wrapper': type(runtime.wrapper).__name__,
            'kv_layout': 'NHD',
            'paged_cache_shape_per_layer': list(
                runtime.paged_kv_cache.shape[1:]),
            'page_size': runtime.page_size,
            'use_tensor_cores': True,
            'fixed_split_size_pages': runtime.fixed_split_size,
            'fixed_split_size_tokens': (
                runtime.fixed_split_size * runtime.page_size),
            'max_plan_seq_len': runtime.max_seq_len,
            'physical_pages_per_request': runtime.pages_per_request,
            'page_table_update': (
                'fixed physical pages; compact CSR indices updated at page '
                'boundaries; indptr/last_page_len and current-token slot '
                'mapping updated every step'),
            'prefill_attention': 'HF SDPA with original Qwen3 RoPE',
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
    fa2_scope = _fa2_unpacked_position_ids() if args.method == 'fa2' else nullcontext()
    with torch.inference_mode(), fa2_scope:
        if args.method == 'nfa':
            result = _run_nfa(args, model, prompt, args.max_new_tokens, eos_ids)
        elif args.method == 'fa2':
            result = _run_fa2(args, model, prompt, args.max_new_tokens, eos_ids)
        else:
            result = _run_flashinfer(
                args, model, prompt, args.max_new_tokens, eos_ids)
    if args.method == 'nfa':
        selector_args = nfa_args()
        result['nfa_selector_config'] = {
            'subspace': selector_args.nfa_subspace,
            'subspace_cmp_ratio': selector_args.tia_level2_cmp_ratio,
            'level1_topk': selector_args.tia_level1_topk,
            'level2_topk': selector_args.tia_level2_topk,
            'near_pages': selector_args.nfa_near_pages,
            'near_tokens': selector_args.nfa_near_tokens,
            'near_len': selector_args.nfa_near_len,
            'enable_layer_skip': selector_args.nfa_enable_layer_skip,
            'sink_tokens': selector_args.nfa_sink_tokens,
            'sliding_window_size': selector_args.nfa_swa_size,
        }
    result.update(
        method=args.method, execution=args.execution, batch_size=args.batch_size,
        context_length=args.context_length, max_new_tokens=args.max_new_tokens,
        model_path=os.path.abspath(args.model_path), input=input_receipt,
        measurement_mode=args.measurement_mode,
        decode_protocol='fixed length; EOS recorded but not used for early stop',
        prefill_batch_execution=(
            'serial one-request prefills'
            if args.method in ('nfa', 'flashinfer')
            else 'one batched prefill forward'),
        prefill_timing_scope=(
            'model prefill, first-token argmax/readback, and request/cache setup; '
            'FA2 optional KV restore included; FlashInfer paged KV append included'),
        timing_scope=(
            'CUDA events around complete decode step: input/position updates, '
            'scheduler/cache planning where applicable, model/selector, and argmax'),
        timing_quality=(
            'clean-device-history' if args.measurement_mode == 'latency'
            else 'host-readback-in-loop'),
        host_readback_in_decode_loop=args.measurement_mode not in (
            'latency', 'correctness'),
        gpu_hygiene={'status': 'requires external process monitor'},
        transformers_version=__import__('transformers').__version__,
        torch_version=torch.__version__)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('generated_token_ids', 'tpot_ms_all',
                                     'strict_skip_first_step')}, indent=2))
    print(f"saved -> {args.output}")
    strict_receipt = result.get('strict_skip_first_step')
    if strict_receipt is not None and not strict_receipt['valid']:
        raise RuntimeError(
            f"strict-skip first-step violations: {strict_receipt['violations']}")


if __name__ == '__main__':
    main()
