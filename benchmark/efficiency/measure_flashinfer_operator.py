"""FlashInfer 稳态 Graph decode 的独立 nsys kernel 观测入口。

复用 benchmark_nfa_end_to_end.py 的真实 Qwen3/FlashInfer 路径，只在稳态
decode 外围加入 cudaProfilerApi 控制。干净 E2E 回执不启用本入口，两种口径
不得相减推导 host 开销。
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency import benchmark_nfa_end_to_end as benchmark


def _argument(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _kernel_instances(sqlite_path):
    connection = sqlite3.connect(str(sqlite_path))
    try:
        table = 'CUPTI_ACTIVITY_KIND_KERNEL'
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise RuntimeError(f'nsys SQLite 缺少 {table}: {sqlite_path}')
        columns = [row[1] for row in connection.execute(
            f'PRAGMA table_info({table})')]
        name_columns = [name for name in ('demangledName', 'shortName')
                        if name in columns]
        if not name_columns or not {'start', 'end'} <= set(columns):
            raise RuntimeError(f'nsys kernel 表缺少时间或名称列: {columns}')
        records = connection.execute(
            f"SELECT start, end, {', '.join(name_columns)} FROM {table}").fetchall()
        strings = dict(connection.execute('SELECT id, value FROM StringIds'))
    finally:
        connection.close()
    if not records:
        raise RuntimeError(f'nsys 采集区间没有 CUDA kernel: {sqlite_path}')
    instances = defaultdict(list)
    for record in records:
        start, end = record[:2]
        name = next((strings.get(value) for value in record[2:]
                     if strings.get(value)), None)
        if name is None:
            raise RuntimeError(f'nsys kernel 名称无法解析: {record}')
        duration_ms = (end - start) / 1e6
        if duration_ms > 0:
            instances[name].append(duration_ms)
    return instances


def finalize_nsys_profile(result, sqlite_path):
    instances = _kernel_instances(sqlite_path)
    repeat = result['profiled_decode_steps']
    if repeat < 1:
        raise ValueError('profiled_decode_steps must be positive')
    categories = {
        'attention_main': [],
        'attention_merge': [],
        'kv_append': [],
    }
    unclassified = []
    for name, values in instances.items():
        if 'AppendPagedKVCacheKernel' in name:
            categories['kv_append'].extend(values)
        elif 'PersistentVariableLengthMergeStatesKernel' in name:
            categories['attention_merge'].extend(values)
        elif ('BatchPrefillWithPagedKVCacheKernel' in name
              or 'BatchDecodeWithPagedKVCacheKernel' in name):
            categories['attention_main'].extend(values)
        elif 'flashinfer::' in name:
            unclassified.append(name)
    if unclassified:
        raise RuntimeError(
            '存在未分类 FlashInfer kernel，拒绝静默漏项: '
            + json.dumps(unclassified, ensure_ascii=False))
    if not categories['attention_main'] or not categories['kv_append']:
        raise RuntimeError(
            'FlashInfer attention 主 kernel 和 KV append kernel 必须都存在: '
            + json.dumps({key: len(value) for key, value in categories.items()}))

    def timing(values):
        return {
            'calls_total': len(values),
            'total_ms': sum(values),
            'mean_ms_per_step': sum(values) / repeat,
        }

    attention_without_append = (
        categories['attention_main'] + categories['attention_merge'])
    attention_with_append = attention_without_append + categories['kv_append']
    result['operator_summary'] = {
        'all_observed_model_kernels_ms': timing(
            [value for values in instances.values() for value in values]),
        'attention_main_ms': timing(categories['attention_main']),
        'attention_merge_ms': timing(categories['attention_merge']),
        'kv_append_ms': timing(categories['kv_append']),
        'attention_without_append_ms': timing(attention_without_append),
        'attention_with_append_ms': timing(attention_with_append),
        'classification': {
            'attention_main': (
                'BatchPrefillWithPagedKVCacheKernel or '
                'BatchDecodeWithPagedKVCacheKernel'),
            'attention_merge': 'PersistentVariableLengthMergeStatesKernel',
            'kv_append': 'AppendPagedKVCacheKernel',
            'unclassified_flashinfer_kernels': unclassified,
        },
    }
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
        } for name, values in sorted(instances.items())
    }
    result['nsys_profile_finalized'] = True
    result['profile_aggregation'] = (
        f'aggregate {repeat} steady Graph replay steps; total and arithmetic '
        'mean per step; kernel instance statistics are not per-step distributions')
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _finalize_only():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sqlite', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    result = json.loads(output.read_text(encoding='utf-8'))
    finalize_nsys_profile(result, Path(args.sqlite))
    _write_json(output, result)
    print(json.dumps({
        'output': str(output),
        'operator_summary': result['operator_summary'],
        'kernel_count': len(result['kernel_summary']),
    }, indent=2))


def _measure():
    warmup = int(_argument('--warmup-steps', '3'))
    max_new_tokens = int(_argument('--max-new-tokens'))
    decode_steps = max_new_tokens - 1
    if warmup < 1 or decode_steps <= warmup:
        raise ValueError('需要 warmup>=1 且稳态 decode steps>=1')
    output_value = _argument('--output')
    if output_value is None:
        raise ValueError('--output is required')
    output = Path(output_value)
    state = {'started': False, 'stopped': False}
    original_prepare = benchmark.FlashInferQwen3Runtime.prepare_decode_step
    original_summary = benchmark._summary

    def prepare_decode_step(runtime, step):
        original_prepare(runtime, step)
        if step == warmup and not state['started']:
            # Graph capture 和 warmup 已结束，仅采集稳态 replay 的模型 kernel。
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            state['started'] = True

    def summary(step_ms, warmup_steps):
        if state['started'] and not state['stopped']:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            state['stopped'] = True
        return original_summary(step_ms, warmup_steps)

    benchmark.FlashInferQwen3Runtime.prepare_decode_step = prepare_decode_step
    benchmark._summary = summary
    benchmark.main()
    if not state['stopped']:
        raise RuntimeError('FlashInfer 稳态 profiling 未正确结束')
    result = json.loads(output.read_text(encoding='utf-8'))
    if result['method'] != 'flashinfer' or result['execution'] != 'graph':
        raise RuntimeError('本入口只接受 FlashInfer CUDA Graph')
    result.update(
        measurement_mode='operator-profile',
        profiled_decode_steps=decode_steps - warmup,
        nsys_profile_finalized=False,
        profile_backend='nsys-cuda-graph-node-external',
        operator_time_basis=(
            'nsys/CUPTI CUDA Graph node trace observed kernel time; node tracing '
            'may significantly perturb short kernels and runtime scheduling'),
        host_overhead_derivable_by_subtraction=False,
        cross_mode_comparison_rule=(
            'Do not subtract profiled kernel totals from clean decode TPOT and '
            'do not interpret their difference as Python/host overhead.'),
        operator_scope=(
            'all CUDA kernels observed during steady full-model Graph decode; '
            'FlashInfer attention and KV append kernels remain individually '
            'identifiable in kernel_summary'),
    )
    _write_json(output, result)


if __name__ == '__main__':
    if '--finalize-only' in sys.argv:
        _finalize_only()
    else:
        _measure()
