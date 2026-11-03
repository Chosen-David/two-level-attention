"""真实 Qwen3-8B + 原生 RULER 的 NFA / FlashAttention-2 对照驱动。

每个配置启动独立进程运行 ``benchmark_nfa_end_to_end.py``，避免 CUDA Graph
私有池和 NFA workspace 跨方法污染显存。默认测量 RULER 32K、B1/B4、
eager/Graph 的完整模型 decode TPOT，以及模型加载后的实际显存增量。
本脚本不再构造随机 Q/K/V 或模拟 attention shape。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, 'benchmark', 'efficiency',
                      'benchmark_nfa_end_to_end.py')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument(
        '--data-root',
        default='/home/wangyuanshuo02/sparse-bench/third_party/'
                'KVCache-Factory/data/RULER')
    parser.add_argument('--task', default='niah_single_1')
    parser.add_argument('--context-length', type=int, default=32768,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1, 4],
                        choices=(1, 2, 4))
    parser.add_argument('--execution', nargs='+', default=['eager', 'graph'],
                        choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--output-dir',
                        default='/tmp/nfa_vs_fa2_qwen3_8b_ruler32k')
    parser.add_argument('--output',
                        default='/tmp/nfa_vs_fa2_qwen3_8b_ruler32k.json')
    return parser.parse_args()


def _run_one(args, batch, execution, method):
    name = f'{method}_ruler{args.context_length // 1024}k_b{batch}_{execution}.json'
    output = os.path.join(args.output_dir, name)
    command = [
        sys.executable, RUNNER,
        '--model-path', args.model_path,
        '--data-root', args.data_root,
        '--output', output,
        '--task', args.task,
        '--context-length', str(args.context_length),
        '--batch-size', str(batch),
        '--method', method,
        '--execution', execution,
        '--sample-index', str(args.sample_index),
        '--input-provenance', 'native-ruler',
        '--max-new-tokens', str(args.max_new_tokens),
        '--warmup-steps', str(args.warmup_steps),
        '--visible-len', str(args.visible_len),
    ]
    print(f'\n===== {method.upper()} B{batch} {execution} =====', flush=True)
    env = os.environ.copy()
    python_paths = [REPO]
    wheel_dir = env.get('FLASH_ATTN_WHEEL_DIR')
    if wheel_dir:
        python_paths.insert(0, wheel_dir)
    if env.get('PYTHONPATH'):
        python_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(python_paths)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    with open(output, encoding='utf-8') as handle:
        return json.load(handle), output


def _gib(value):
    return value / (1024 ** 3)


def _compare(batch, execution, nfa, fa2):
    nfa_memory = nfa['memory']
    fa2_memory = fa2['memory']
    nfa_peak = nfa_memory['peak_incremental_allocated_bytes']
    fa2_peak = fa2_memory['peak_incremental_allocated_bytes']
    nfa_ready = nfa_memory['decode_ready']['allocated_bytes']
    fa2_ready = fa2_memory['decode_ready']['allocated_bytes']
    nfa_steady_resident = nfa_memory[
        'steady_decode_resident_incremental_allocated_bytes']
    fa2_steady_resident = fa2_memory[
        'steady_decode_resident_incremental_allocated_bytes']
    saved = fa2_peak - nfa_peak
    ready_saved = fa2_ready - nfa_ready
    steady_resident_saved = fa2_steady_resident - nfa_steady_resident
    if nfa['input']['sha256'] != fa2['input']['sha256']:
        raise RuntimeError('NFA/FA2 input SHA256 mismatch')
    if nfa['input']['actual_prompt_tokens'] != fa2['input']['actual_prompt_tokens']:
        raise RuntimeError('NFA/FA2 prompt length mismatch')
    return {
        'batch_size': batch,
        'execution': execution,
        'nfa_steady_tpot_ms': nfa['steady_tpot_ms_median'],
        'fa2_steady_tpot_ms': fa2['steady_tpot_ms_median'],
        'fa2_over_nfa_tpot': (
            fa2['steady_tpot_ms_median'] / nfa['steady_tpot_ms_median']),
        'nfa_prefill_seconds': nfa['prefill_seconds'],
        'fa2_prefill_seconds': fa2['prefill_seconds'],
        'fa2_over_nfa_prefill': nfa['prefill_seconds'] / fa2['prefill_seconds'],
        'generated_token_identity': (
            nfa['generated_token_ids'] == fa2['generated_token_ids']),
        'nfa_skip_policy_counts': nfa.get('skip_policy_counts'),
        'nfa_peak_incremental_allocated_gib': _gib(nfa_peak),
        'fa2_peak_incremental_allocated_gib': _gib(fa2_peak),
        'peak_incremental_allocated_saved_gib': _gib(saved),
        'peak_incremental_allocated_saving_percent': (
            100.0 * saved / fa2_peak if fa2_peak else None),
        'decode_ready_allocated_saved_gib': _gib(ready_saved),
        'nfa_steady_decode_resident_incremental_allocated_gib': _gib(
            nfa_steady_resident),
        'fa2_steady_decode_resident_incremental_allocated_gib': _gib(
            fa2_steady_resident),
        'steady_decode_resident_allocated_saved_gib': _gib(
            steady_resident_saved),
        'steady_decode_resident_allocated_saving_percent': (
            100.0 * steady_resident_saved / fa2_steady_resident
            if fa2_steady_resident else None),
        'nfa_steady_decode_peak_incremental_allocated_gib': _gib(
            nfa_memory['steady_decode_peak_incremental_allocated_bytes']),
        'fa2_steady_decode_peak_incremental_allocated_gib': _gib(
            fa2_memory['steady_decode_peak_incremental_allocated_bytes']),
        'nfa_decode_ready_reserved_gib': _gib(
            nfa_memory['decode_ready']['reserved_bytes']),
        'fa2_decode_ready_reserved_gib': _gib(
            fa2_memory['decode_ready']['reserved_bytes']),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = {}
    comparisons = []
    for batch in args.batch_size:
        for execution in args.execution:
            nfa, nfa_path = _run_one(args, batch, execution, 'nfa')
            fa2, fa2_path = _run_one(args, batch, execution, 'fa2')
            comparison = _compare(batch, execution, nfa, fa2)
            comparisons.append(comparison)
            runs[f'B{batch}/{execution}'] = {
                'nfa': nfa_path,
                'fa2': fa2_path,
            }
            print(
                f"SUMMARY B{batch} {execution}: "
                f"TPOT NFA={comparison['nfa_steady_tpot_ms']:.4f}ms "
                f"FA2={comparison['fa2_steady_tpot_ms']:.4f}ms "
                f"FA2/NFA={comparison['fa2_over_nfa_tpot']:.3f}x | "
                f"steady resident NFA={comparison['nfa_steady_decode_resident_incremental_allocated_gib']:.3f}GiB "
                f"FA2={comparison['fa2_steady_decode_resident_incremental_allocated_gib']:.3f}GiB "
                f"saved={comparison['steady_decode_resident_allocated_saved_gib']:.3f}GiB "
                f"({comparison['steady_decode_resident_allocated_saving_percent']:.2f}%) | "
                f"token_identity={comparison['generated_token_identity']}",
                flush=True)

    receipt = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'benchmark_scope': (
            'actual Qwen3-8B weights, native RULER prompt, full-model '
            'prefill and fixed-length decode; TPOT excludes prefill'),
        'model_path': os.path.abspath(args.model_path),
        'data_root': os.path.abspath(args.data_root),
        'task': args.task,
        'context_length': args.context_length,
        'max_new_tokens': args.max_new_tokens,
        'memory_scope': (
            'torch CUDA allocator; steady resident is post-warmup allocated '
            'minus post-model-load allocated; steady peak is measured after '
            'resetting peak stats at the same post-warmup boundary; legacy '
            'peak incremental includes prefill; separate process per arm'),
        'gpu_hygiene': {'status': 'requires external process monitor'},
        'runs': runs,
        'comparisons': comparisons,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    print(f'\nsaved comparison -> {args.output}')


if __name__ == '__main__':
    main()
