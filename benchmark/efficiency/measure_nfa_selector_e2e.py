"""NFA 算子 profiling 与干净 decode TPOT 的独立进程测量。

operator-profile 由外部 nsys 采集 eager 或 CUDA Graph kernel，存在观察者效应，
只能解释为 profiling 条件下的 kernel 时间；clean-e2e 不启用任何 profiler。
两者不得相减推导 Python/host 开销，也不得把 profiled operator total 当作
无扰动 device busy time。
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = Path(os.environ.get(
    'NFA_REPO', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO))
from benchmark.rope_policy import configure_rope, verify_runtime_rope
from benchmark.efficiency.benchmark_nfa_end_to_end import (
    LONG_PREFILL_MLP_CHUNK,
    _enable_long_prefill_mlp_chunk,
    _load_input as _load_ruler_input,
    _summary,
)
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.engine.graph_runner import GraphRunner
try:
    from sparse_attn.engine.graph_runner import session_graph_runner
except ImportError:  # 同 HEAD 基线没有 session pool 时使用原公共入口。
    session_graph_runner = None
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler, greedy_next_tokens


STAGE_TOKENS = {
    'l1_score_ms': (
        '_nfa_page_stats_kernel', '_nfa_page_stats_prepare_q_kernel',
        '_nfa_l1_scores_kernel', '_nfa_l1_near_scores_kernel',
        '_nfa_l1_far_scores_kernel', '_prepare_q_kernel',
    ),
    'l1_topk_ms': (
        '_l1_topk_partial_kernel', '_topk_candidate_prepare_kernel',
        '_topk_candidate_batched_kernel',
    ),
    'candidate_ms': (
        '_candidate_expand_batched_kernel', '_candidate_pack_batched_kernel',
        '_owner_generation_prepare',
    ),
    'l2_score_ms': (
        '_prepare_q_grouped_kernel', '_nfa_fine_scores_paged_kernel',
    ),
    'l2_topk_ms': (
        '_nfa_filtered_topk_kernel', '_nfa_filtered_topk_radix_kernel',
        '_nfa_filtered_topk_partial_kernel', '_nfa_filtered_topk_merge_kernel',
        '_nfa_finish_topk_kernel',
    ),
    'selection_union_ms': (
        '_final_union_physical_kernel', '_final_union_protected_sort_kernel',
        '_final_union_two_way_merge_kernel', '_middle_physical_kernel',
    ),
    'selected_attention_ms': (
        '_gather_attend_kernel', '_gather_attend_grouped_kernel',
        '_gather_attend_grouped_split_kernel', '_gather_attend_split_combine_kernel',
        '_gather_attend_regions_combine_kernel',
    ),
}
AUDIT_TOKENS = (
    'nfa', 'topk', 'candidate', 'union', 'gather_attend', 'prepare_q',
    'fine_scores', 'page_stats', 'l1_scores', 'owner_generation',
    'middle_physical',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--task', default='niah_multikey_2')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072, 262144, 524288))
    parser.add_argument('--batch-size', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--execution', required=True, choices=('eager', 'graph'))
    parser.add_argument('--measurement-mode', required=True,
                        choices=('operator-profile', 'clean-e2e'))
    parser.add_argument('--subspace', required=True, choices=('full', 'tail'))
    parser.add_argument('--l2-subspace', required=True, choices=('same', 'full'))
    parser.add_argument('--gate-epsilon', type=float, required=True)
    parser.add_argument(
        '--force-skip-mode', default='natural',
        choices=('natural', 'no-skip', 'near-only', 'far-only', 'both-skip'),
        help='验收用强制冻结区域；natural 保留 prefill 标定的逐层决策')
    parser.add_argument('--operator-mode', required=True,
                        choices=('off', 'serial', 'overlap'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--input-provenance', required=True,
                        choices=('native-ruler', 'repeated-needle-body-extension',
                                 'deterministic-longbench-context-repetition'))
    parser.add_argument(
        '--longbench-data',
        default='/home/wangyuanshuo02/datasets/LongBench/data')
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--profile-repeat', type=int, default=5)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _load_input(args, tokenizer):
    if args.input_provenance != 'deterministic-longbench-context-repetition':
        return _load_ruler_input(args, tokenizer)
    if args.task not in ('hotpotqa', 'gov_report'):
        raise ValueError('LongBench performance input requires hotpotqa or gov_report')
    if args.context_length <= 128:
        raise ValueError('LongBench performance context must exceed suffix reserve')
    # 与精度链共享同一 material 构造；生成 suffix 只记录身份，不进入性能 prompt。
    from benchmark.LongBench.prefill_frozen_skip import prepare_ids

    path = Path(args.longbench_data) / f'{args.task}.jsonl'
    data_bytes = path.read_bytes()
    rows = [json.loads(line) for line in data_bytes.splitlines()]
    if not 0 <= args.sample_index < len(rows):
        raise ValueError(f'LongBench row out of range: {args.sample_index}')
    row = rows[args.sample_index]
    prompts = json.loads(
        (REPO / 'benchmark/LongBench/config/dataset2prompt.json').read_text())
    material_ids, suffix_ids, info = prepare_ids(
        tokenizer, row, args.task, prompts[args.task], args.context_length - 128,
        repeat_context_to_cap=True)
    prompt_tokens = len(material_ids)
    if not 0.95 * args.context_length <= prompt_tokens <= args.context_length:
        raise ValueError(
            f'LongBench material length mismatch: requested {args.context_length}, '
            f'got {prompt_tokens} tokens from {path} row {args.sample_index}')
    receipt = {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(data_bytes).hexdigest(),
        'sample_index': args.sample_index,
        'sample_id': row.get('_id', row.get('index')),
        'file_row_index': args.sample_index,
        'actual_prompt_tokens': prompt_tokens,
        'provenance': args.input_provenance,
        'performance_input_mode': 'replicate one material prompt across all batch rows',
        'material_sha256': info['material_sha256'],
        'suffix_sha256': info['suffix_sha256'],
        'suffix_tokens': len(suffix_ids),
        'chat_generation_suffix_excluded_from_performance_prompt': True,
        'construction': info['construction'],
    }
    return torch.tensor(material_ids, dtype=torch.long, device='cuda'), receipt


def load_model(args, max_new_tokens):
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'qwen3':
        raise ValueError('this benchmark requires Qwen3')
    configure_rope(
        config, args.context_length, max_new_tokens,
        yarn=args.context_length > 32768,
        factor={65536: 2.0, 131072: 4.0,
                262144: 8.0, 524288: 16.0}.get(args.context_length))
    config._attn_implementation = 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16).cuda().eval()
    verify_runtime_rope(model)
    patch_args = nfa_args(args.subspace)
    patch_args.nfa_l2_subspace = args.l2_subspace
    patch_args.nfa_operator_mode = args.operator_mode
    patched = register_patch(
        model, patch_args, resolve_treatment_snapshot(patch_args),
        rmsnorm_decode=True, mlp_silu_decode=True)
    if patched != config.num_hidden_layers:
        raise RuntimeError(f'NFA patched {patched}/{config.num_hidden_layers} layers')
    if args.context_length > LONG_PREFILL_MLP_CHUNK:
        _enable_long_prefill_mlp_chunk(model)
    return model, patch_args


def score_dimensions(model):
    """记录生产 selector 实际采用的 L1/L2 维度，而非仅从臂名推断。"""
    indexer = model.model.layers[0].self_attn.indexer
    device = next(model.parameters()).device
    if indexer.enable_subspace:
        l1_dims = indexer._subspace_indices(device).numel()
    else:
        l1_dims = indexer.head_dim
    if indexer.l2_subspace == 'full':
        l2_dims = indexer.head_dim
    elif indexer.enable_subspace:
        l2_dims = l1_dims
    else:
        l2_dims = indexer._qat_indices(device).numel()
    return {
        'head_dim': indexer.head_dim,
        'l1_score_dims': l1_dims,
        'l2_score_dims': l2_dims,
    }


def classify_nfa(kernels, *, required_stages=None):
    stages = {name: 0.0 for name in STAGE_TOKENS}
    memberships = {}
    for name, row in kernels.items():
        matches = [
            stage for stage, tokens in STAGE_TOKENS.items()
            if any(token in name for token in tokens)
        ]
        if len(matches) > 1:
            raise ValueError(f'kernel 重复归属: {name} -> {matches}')
        if matches:
            stage = matches[0]
            stages[stage] += row['device_ms']
            memberships[name] = stage
        elif any(token in name.lower() for token in AUDIT_TOKENS):
            raise ValueError(f'未分类的 NFA 相关 kernel: {name}')
    required = tuple(STAGE_TOKENS) if required_stages is None else required_stages
    missing = [stage for stage in required if stage not in memberships.values()]
    if missing:
        raise ValueError(f'NFA 算子链缺少阶段: {missing}')
    operator_total = sum(stages.values())
    stages['l1_ms'] = (
        stages['l1_score_ms'] + stages['l1_topk_ms'] + stages['candidate_ms'])
    stages['l2_ms'] = (
        stages['l2_score_ms'] + stages['l2_topk_ms']
        + stages['selection_union_ms'])
    stages['topk_total_ms'] = stages['l1_topk_ms'] + stages['l2_topk_ms']
    stages['main_attention_ms'] = stages['selected_attention_ms']
    stages['nfa_operator_total_ms'] = operator_total
    return stages, memberships


def _region_kernel_call_audit(result, kernels):
    """按冻结逐层签名核对区域入口；零实例不能由空 CTA 或耗时近似。"""
    expected = {
        'generic_l1': 0, 'near_only_l1': 0, 'far_only_l1': 0,
        'active_page_stats': 0, 'active_candidate': 0, 'active_fine_score': 0,
    }
    requests = list(result['skip'].values())
    n_layers = requests[0]['n_layers']
    for request in requests:
        near = set(request['near_layers'])
        far = set(request['far_layers'])
        both = set(request['both_layers'])
        if near & far != both:
            raise ValueError('skip receipt 的 both 集合与 near/far 交集不一致')
        if request['n_layers'] != n_layers:
            raise ValueError('同一 batch 的层数不一致')
    # selector 按层对整个 batch 发射一次 kernel，而不是每请求发射一次。
    # 只有本层所有请求模式一致才走 compact；mixed batch 统一走 generic。
    for layer in range(n_layers):
        modes = [
            (layer in request['far_layers'], layer in request['near_layers'])
            for request in requests]
        uniform = all(mode == modes[0] for mode in modes[1:])
        if not uniform:
            region = 0
        else:
            skip_far, skip_near = modes[0]
            region = (3 if skip_far and skip_near else
                      1 if skip_far else 2 if skip_near else 0)
        if region == 0:
            expected['generic_l1'] += 1
        elif region == 1:
            expected['near_only_l1'] += 1
        elif region == 2:
            expected['far_only_l1'] += 1
        if region != 3:
            expected['active_page_stats'] += 1
            expected['active_candidate'] += 1
            expected['active_fine_score'] += 1
    tokens = {
        'generic_l1': ('_nfa_l1_scores_kernel',),
        'near_only_l1': ('_nfa_l1_near_scores_kernel',),
        'far_only_l1': ('_nfa_l1_far_scores_kernel',),
        'active_page_stats': (
            '_nfa_page_stats_kernel', '_nfa_page_stats_prepare_q_kernel'),
        'active_candidate': (
            '_topk_candidate_batched_kernel', '_topk_candidate_prepare_kernel'),
        'active_fine_score': ('_nfa_fine_scores_paged_kernel',),
    }

    def calls(names):
        return sum(row['calls'] for name, row in kernels.items()
                   if any(token in name for token in names))

    repeat = result['profile_repeat']
    observed = {name: calls(names) / repeat for name, names in tokens.items()}
    passed = observed == expected
    audit = {
        'expected_calls_per_step': expected,
        'observed_calls_per_step': observed,
        'passed': passed,
        'interpretation': (
            'calls are per layer for the whole batch, not per request; '
            'generic=non-skip or mixed-batch, near_only=skip-far, '
            'far_only=skip-near; uniform both-skip contributes zero to '
            'every active selector entry'),
    }
    if not passed:
        raise ValueError(f'区域 kernel 实例数与冻结签名不一致: {audit}')
    return audit


def finalize_nsys_profile(result, sqlite_path):
    """从 nsys SQLite 汇总 Graph node kernel；只含 cudaProfilerApi 采集区间。"""
    connection = sqlite3.connect(str(sqlite_path))
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        table = 'CUPTI_ACTIVITY_KIND_KERNEL'
        if table not in tables:
            raise RuntimeError(f'nsys SQLite 缺少 {table}: {sqlite_path}')
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info({table})')]
        name_columns = [name for name in ('demangledName', 'shortName')
                        if name in columns]
        if not name_columns or not {'start', 'end'} <= set(columns):
            raise RuntimeError(f'nsys kernel 表缺少时间或名称列: {columns}')
        grid_columns = [name for name in ('gridX', 'gridY', 'gridZ') if name in columns]
        selected = ['start', 'end'] + grid_columns + name_columns
        query = f"SELECT {', '.join(selected)} FROM {table}"
        records = connection.execute(query).fetchall()
        strings = dict(connection.execute('SELECT id, value FROM StringIds'))
    finally:
        connection.close()
    if not records:
        raise RuntimeError(f'nsys 采集区间没有 CUDA kernel: {sqlite_path}')

    instances = defaultdict(list)
    grid_instances = defaultdict(list)
    for record in records:
        start, end = record[:2]
        grid = record[2:2 + len(grid_columns)]
        name_values = record[2 + len(grid_columns):]
        name = next((strings.get(value) for value in name_values
                     if strings.get(value)), None)
        if name is None:
            raise RuntimeError(f'nsys kernel 名称无法解析: {record}')
        duration_ms = (end - start) / 1e6
        if duration_ms > 0:
            instances[name].append(duration_ms)
            grid_instances[name].append(grid)
    totals = {
        name: {'calls': len(values), 'device_ms': sum(values)}
        for name, values in instances.items()
    }
    selector_stages = {
        'l1_score_ms', 'l1_topk_ms', 'candidate_ms',
        'l2_score_ms', 'l2_topk_ms',
    }
    all_both_skip = all(
        request['n_layers'] == len(request['both_layers'])
        for request in result['skip'].values())
    required_stages = [
        stage for stage in STAGE_TOKENS
        if not (all_both_skip and stage in selector_stages)]
    # 单 CTA 快路把 L1 Top-K 与 candidate expansion 融在同一个 kernel；
    # 此时没有独立 candidate 实例，时间已完整计入 l1_topk_ms，不能误报缺阶段。
    fused_candidate = any(
        '_topk_candidate_batched_kernel' in name for name in totals)
    separate_candidate = any(
        token in name for name in totals
        for token in STAGE_TOKENS['candidate_ms'])
    if fused_candidate and not separate_candidate:
        required_stages.remove('candidate_ms')
    stages, memberships = classify_nfa(
        totals, required_stages=required_stages)
    result['region_kernel_audit'] = _region_kernel_call_audit(result, totals)
    repeat = result['profile_repeat']
    result['operator_summary'] = {
        name: {
            'total_ms': value,
            'mean_ms_per_step': value / repeat,
        } for name, value in stages.items()
    }
    def grid_values(name, axis):
        if axis not in grid_columns:
            return []
        position = grid_columns.index(axis)
        return sorted({item[position] for item in grid_instances.get(name, ())})

    result['kernel_summary'] = {
        name: {
            'calls_total': len(values),
            'mean_calls_per_step': len(values) / repeat,
            'total_ms': sum(values),
            'mean_ms_per_step': sum(values) / repeat,
            'mean_instance_ms': statistics.mean(values),
            'median_instance_ms': statistics.median(values),
            'min_instance_ms': min(values),
            'max_instance_ms': max(values),
            'grid_x_values': grid_values(name, 'gridX'),
            'grid_y_values': grid_values(name, 'gridY'),
            'grid_z_values': grid_values(name, 'gridZ'),
            'stage': memberships.get(name),
        } for name, values in sorted(instances.items())
    }
    result['nsys_profile_finalized'] = True
    result['profile_aggregation'] = (
        f"aggregate {repeat} {result['execution']} decode steps; total and "
        'arithmetic mean per step; kernel instance min/median/max are not '
        'per-step distributions')
    return result


def source_receipt():
    paths = (
        'sparse_attn', 'benchmark/rope_policy.py',
        'benchmark/efficiency/benchmark_nfa_end_to_end.py',
        'benchmark/efficiency/measure_nfa_selector_e2e.py',
        'benchmark/LongBench/prefill_frozen_skip.py',
        'benchmark/LongBench/config/dataset2prompt.json',
    )
    diff = subprocess.check_output(
        ['git', '-C', str(REPO), 'diff', 'HEAD', '--binary', '--', *paths])
    harness = Path(__file__).read_bytes()
    return {
        'head': subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'diff_sha256': hashlib.sha256(diff).hexdigest(),
        'harness_sha256': hashlib.sha256(harness).hexdigest(),
    }


def skip_receipt(cache, request_id):
    state = cache.states[request_id]
    return {
        'n_layers': len(state.skip_near_by_layer),
        'near_layers': [i for i, flag in enumerate(state.skip_near_by_layer) if flag],
        'far_layers': [i for i, flag in enumerate(state.skip_far_by_layer) if flag],
        'both_layers': [i for i, (near, far) in enumerate(zip(
            state.skip_near_by_layer, state.skip_far_by_layer)) if near and far],
    }


def run(args):
    if args.warmup_steps < 1 or args.profile_repeat < 1 or args.decode_steps < 2:
        raise ValueError('warmup/profile/decode steps must be positive')
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = _load_input(args, tokenizer)
    measured_steps = (
        args.profile_repeat if args.measurement_mode == 'operator-profile'
        else args.decode_steps)
    all_steps = args.warmup_steps + measured_steps
    total_capacity = all_steps + 2
    model, patch_args = load_model(args, total_capacity)
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
        max_new_tokens=total_capacity)
    cache = build_frozen_cache(
        model, geometry, prompt.numel(), max_requests=args.batch_size,
        max_visible_len=args.visible_len)
    gate = GateConfig(epsilon=args.gate_epsilon)
    first_tokens = {}
    prefill_decisions = {}
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        for request_id in range(args.batch_size):
            output, audit = prefill_frozen_request(
                model, prompt.unsqueeze(0), gate, geometry,
                cache=cache, request_id=request_id)
            first = greedy_next_tokens(
                output.logits, set(), generated_counts=0, min_new_tokens=1)
            first_tokens[request_id] = int(first[0])
            prefill_decisions[request_id] = audit['decisions']
            if args.force_skip_mode != 'natural':
                skip_far, skip_near = {
                    'no-skip': (False, False),
                    'near-only': (True, False),
                    'far-only': (False, True),
                    'both-skip': (True, True),
                }[args.force_skip_mode]
                cache.freeze(
                    request_id,
                    [skip_far] * model.config.num_hidden_layers,
                    [skip_near] * model.config.num_hidden_layers)
            cache.transition_request(request_id, 'GENERATE')
            del output
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=args.batch_size, eos_ids=set(), pad_token_id=0)
    for request_id in range(args.batch_size):
        scheduler.add_request(request_id)
        scheduler.next_tokens[request_id] = first_tokens[request_id]
    runner_args = dict(
        scheduler=scheduler, buckets=(args.batch_size,),
        visible_buckets=(args.visible_len,), device=prompt.device)
    if session_graph_runner is None:
        runner = GraphRunner(**runner_args)
        if args.execution == 'graph':
            runner.capture_all(model, cache)
    else:
        runner = session_graph_runner(
            model, cache, **runner_args, capture=args.execution == 'graph')

    token_history = torch.empty(
        (all_steps, args.batch_size), dtype=torch.long, device=prompt.device)
    event_pairs = []
    with torch.inference_mode():
        for step in range(all_steps):
            counts = [step + 1] * args.batch_size
            if (args.measurement_mode == 'operator-profile'
                    and step == args.warmup_steps):
                # Graph capture 和 warmup 均已结束；nsys 只采集后面的稳态 replay。
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
            if args.measurement_mode == 'clean-e2e' and step >= args.warmup_steps:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True)
                event_pairs.append((start, end))
                start.record()
            _, tokens = runner.step(
                model, cache, generated_counts=counts, min_new_tokens=1,
                return_next_tokens=True, stop_on_eos=False)
            if args.measurement_mode == 'clean-e2e' and step >= args.warmup_steps:
                end.record()
            token_history[step].copy_(tokens)
        if args.measurement_mode == 'operator-profile':
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        scheduler.prepare_step()
        torch.cuda.synchronize()
    history = token_history.cpu().tolist()
    step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    timing = (
        _summary(step_ms, warmup=0)
        if args.measurement_mode == 'clean-e2e' else None)
    generated = []
    for row in range(args.batch_size):
        generated.append([first_tokens[row]] + [step[row] for step in history])
    return {
        'schema': 'nfa-selector-and-e2e-v2',
        'source': source_receipt(),
        'method': 'nfa',
        'measurement_mode': args.measurement_mode,
        'model_path': str(Path(args.model_path).resolve()),
        'context_length': args.context_length,
        'batch_size': args.batch_size,
        'execution': args.execution,
        'subspace': args.subspace,
        'l2_subspace': args.l2_subspace,
        'score_dimensions': score_dimensions(model),
        'gate_epsilon': args.gate_epsilon,
        'force_skip_mode': args.force_skip_mode,
        'operator_mode': args.operator_mode,
        'task': args.task,
        'sample_index': args.sample_index,
        'input_provenance': args.input_provenance,
        'visible_len': args.visible_len,
        'input': input_receipt,
        'warmup_steps': args.warmup_steps,
        'profile_repeat': args.profile_repeat,
        'decode_steps': args.decode_steps,
        'measured_decode_steps': len(step_ms),
        'profiled_decode_steps': (
            args.profile_repeat
            if args.measurement_mode == 'operator-profile' else 0),
        'profiler_clean_e2e_isolated': True,
        'profile_backend': (
            f"nsys-{args.execution}-external"
            if args.measurement_mode == 'operator-profile' else None),
        'nsys_profile_finalized': (
            False if args.measurement_mode == 'operator-profile' else None),
        'operator_time_basis': (
            f'nsys/CUPTI {args.execution} kernel trace observed time; tracing '
            'may significantly perturb short kernels and runtime scheduling'),
        'host_overhead_derivable_by_subtraction': False,
        'cross_mode_comparison_rule': (
            'Do not subtract profiled nfa_operator_total_ms from clean decode '
            'TPOT and do not interpret their difference as Python/host overhead.'),
        'timing_scope': (
            'clean-e2e uses a separate process and CUDA events around complete '
            'decode steps, including host scheduling/metadata, model/selector '
            'execution and argmax; operator-profile has no clean TPOT'),
        'timing_metric': (
            'end-to-end decode TPOT, not pure attention/operator latency'
            if timing is not None else None),
        'prefill_seconds': prefill_seconds,
        'long_prefill_mlp_chunk_tokens': (
            LONG_PREFILL_MLP_CHUNK
            if args.context_length > LONG_PREFILL_MLP_CHUNK else None),
        'prefill_decisions': prefill_decisions,
        'skip': {str(request_id): skip_receipt(cache, request_id)
                 for request_id in range(args.batch_size)},
        'operator_summary': None,
        'kernel_summary': None,
        'region_kernel_audit': None,
        'timing': timing,
        'generated_token_ids': generated,
        'graph_capture_count': getattr(runner, 'capture_count', None),
        'graph_replays': getattr(runner, 'replay_count', None),
        'graph_eager_fallbacks': getattr(runner, 'eager_fallback_count', None),
        'max_memory_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'max_memory_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'torch_version': torch.__version__,
        'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
    }


def main():
    args = parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({
        'context_length': args.context_length,
        'batch_size': args.batch_size,
        'execution': args.execution,
        'measurement_mode': args.measurement_mode,
        'subspace': args.subspace,
        'l2_subspace': args.l2_subspace,
        'score_dimensions': result['score_dimensions'],
        'gate_epsilon': args.gate_epsilon,
        'force_skip_mode': args.force_skip_mode,
        'operator_mode': args.operator_mode,
        'operator_summary': result['operator_summary'],
        'steady_tpot_ms_median': (
            result['timing']['steady_tpot_ms_median']
            if result['timing'] is not None else None),
        'prefill_seconds': result['prefill_seconds'],
        'skip': result['skip'],
        'saved': str(output),
    }, indent=2))


if __name__ == '__main__':
    main()
