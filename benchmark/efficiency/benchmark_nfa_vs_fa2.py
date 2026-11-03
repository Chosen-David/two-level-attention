"""真实 Qwen3-8B + 原生 RULER 的 NFA / FlashAttention-2 ABBA 对照驱动。

每个 arm 启动独立进程，按 NFA-A、FA2-A、FA2-B、NFA-B 顺序测量。
latency 与 memory 使用不同 run；默认只报告完整生产 decode TPOT。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
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
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1],
                        choices=(1, 2, 4))
    parser.add_argument('--execution', nargs='+', default=['eager', 'graph'],
                        choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--gate-epsilon', type=float, default=0.05)
    parser.add_argument('--measurement-mode', choices=('latency', 'memory'),
                        default='latency')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--output-dir',
                        default='/tmp/nfa_vs_fa2_qwen3_8b_ruler32k')
    parser.add_argument('--output',
                        default='/tmp/nfa_vs_fa2_qwen3_8b_ruler32k.json')
    return parser.parse_args()


def _smi(gpu_id, query, apps=False):
    key = '--query-compute-apps' if apps else '--query-gpu'
    command = ['nvidia-smi', '-i', str(gpu_id), f'{key}={query}',
               '--format=csv,noheader,nounits']
    return subprocess.check_output(command, text=True).strip()


def _preflight(gpu_id):
    used = int(_smi(gpu_id, 'memory.used'))
    apps = _smi(gpu_id, 'pid,used_memory', apps=True)
    if used >= 500 or apps:
        raise RuntimeError(
            f'GPU{gpu_id} 不满足干净起跑条件: used={used} MiB, apps={apps!r}')


class GpuMonitor:
    """30 秒旁路监控；只允许本 driver 直接启动的 runner PID。"""

    def __init__(self, gpu_id, log_path):
        self.gpu_id = gpu_id
        self.log_path = log_path
        self.allowed_pids = set()
        self.pollution = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        _preflight(self.gpu_id)
        self._thread.start()

    def register(self, pid):
        with self._lock:
            self.allowed_pids.add(pid)

    def _sample(self):
        timestamp = int(time.time())
        gpu = _smi(self.gpu_id, 'index,memory.used,utilization.gpu')
        apps = _smi(self.gpu_id, 'pid,used_memory', apps=True)
        rows = [row.strip() for row in apps.splitlines() if row.strip()]
        with self._lock:
            allowed = set(self.allowed_pids)
        unknown = []
        for row in rows:
            pid = int(row.split(',')[0].strip())
            if pid not in allowed:
                unknown.append(row)
        record = {'timestamp': timestamp, 'gpu': gpu, 'apps': rows,
                  'allowed_pids': sorted(allowed), 'unknown_apps': unknown}
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        if unknown:
            self.pollution.append(record)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(30)

    def stop(self):
        self._stop.set()
        self._thread.join()


def _run_one(args, batch, execution, method, leg, attempt, monitor):
    tag = f'{method}_{leg}'
    stem = (f'{tag}_ruler{args.context_length // 1024}k_b{batch}_'
            f'{execution}_attempt{attempt}')
    output = os.path.join(args.output_dir, stem + '.json')
    log_path = os.path.join(args.output_dir, stem + '.log')
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
        '--gate-epsilon', str(args.gate_epsilon),
        '--measurement-mode', args.measurement_mode,
    ]
    print(f'\n===== {tag.upper()} B{batch} {execution} attempt{attempt} =====',
          flush=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    python_paths = [REPO]
    wheel_dir = env.get('FLASH_ATTN_WHEEL_DIR')
    if wheel_dir:
        python_paths.insert(0, wheel_dir)
    if env.get('PYTHONPATH'):
        python_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(python_paths)
    with open(log_path, 'w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        monitor.register(process.pid)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f'{tag} 失败，日志: {log_path}')
    with open(output, encoding='utf-8') as handle:
        result = json.load(handle)
    return result, {'json': output, 'log': log_path, 'pid': process.pid}


def _run_abba(args, batch, execution, attempt, monitor):
    arms = {}
    paths = {}
    for method, leg in (('nfa', 'a'), ('fa2', 'a'),
                        ('fa2', 'b'), ('nfa', 'b')):
        result, path = _run_one(
            args, batch, execution, method, leg, attempt, monitor)
        key = f'{method}_{leg}'
        arms[key] = result
        paths[key] = path
    return arms, paths


def _gib(value):
    return value / (1024 ** 3)


def _first_divergence(lhs, rhs):
    for index, (left, right) in enumerate(zip(lhs[0], rhs[0])):
        if left != right:
            return index
    return None if len(lhs[0]) == len(rhs[0]) else min(len(lhs[0]), len(rhs[0]))


def _memory_compare(nfa, fa2):
    nfa_memory = nfa['memory']
    fa2_memory = fa2['memory']
    nfa_peak = nfa_memory['peak_incremental_allocated_bytes']
    fa2_peak = fa2_memory['peak_incremental_allocated_bytes']
    nfa_ready = nfa_memory['decode_ready']['allocated_bytes']
    fa2_ready = fa2_memory['decode_ready']['allocated_bytes']
    nfa_resident = nfa_memory[
        'steady_decode_resident_incremental_allocated_bytes']
    fa2_resident = fa2_memory[
        'steady_decode_resident_incremental_allocated_bytes']
    return {
        'nfa_peak_incremental_allocated_gib': _gib(nfa_peak),
        'fa2_peak_incremental_allocated_gib': _gib(fa2_peak),
        'peak_incremental_allocated_saved_gib': _gib(fa2_peak - nfa_peak),
        'decode_ready_allocated_saved_gib': _gib(fa2_ready - nfa_ready),
        'nfa_steady_decode_resident_incremental_allocated_gib': _gib(nfa_resident),
        'fa2_steady_decode_resident_incremental_allocated_gib': _gib(fa2_resident),
        'steady_decode_resident_allocated_saved_gib': _gib(fa2_resident - nfa_resident),
        'steady_decode_resident_allocated_saving_percent': (
            100.0 * (fa2_resident - nfa_resident) / fa2_resident
            if fa2_resident else None),
        'nfa_steady_decode_peak_incremental_allocated_gib': _gib(
            nfa_memory['steady_decode_peak_incremental_allocated_bytes']),
        'fa2_steady_decode_peak_incremental_allocated_gib': _gib(
            fa2_memory['steady_decode_peak_incremental_allocated_bytes']),
        'nfa_decode_ready_reserved_gib': _gib(
            nfa_memory['decode_ready']['reserved_bytes']),
        'fa2_decode_ready_reserved_gib': _gib(
            fa2_memory['decode_ready']['reserved_bytes']),
    }


def _compare(args, batch, execution, arms):
    nfa_a, nfa_b = arms['nfa_a'], arms['nfa_b']
    fa2_a, fa2_b = arms['fa2_a'], arms['fa2_b']
    for method in ('nfa', 'fa2'):
        lhs, rhs = arms[f'{method}_a'], arms[f'{method}_b']
        if lhs['input']['sha256'] != rhs['input']['sha256']:
            raise RuntimeError(f'{method} ABBA 输入 SHA256 不一致')
    if nfa_a['input']['sha256'] != fa2_a['input']['sha256']:
        raise RuntimeError('NFA/FA2 输入 SHA256 不一致')
    if nfa_a['input']['actual_prompt_tokens'] != fa2_a['input']['actual_prompt_tokens']:
        raise RuntimeError('NFA/FA2 prompt 长度不一致')
    nfa_ms = [nfa_a['steady_tpot_ms_median'], nfa_b['steady_tpot_ms_median']]
    fa2_ms = [fa2_a['steady_tpot_ms_median'], fa2_b['steady_tpot_ms_median']]
    result = {
        'batch_size': batch,
        'execution': execution,
        'nfa_repeat_token_identity': (
            nfa_a['generated_token_ids'] == nfa_b['generated_token_ids']),
        'fa2_repeat_token_identity': (
            fa2_a['generated_token_ids'] == fa2_b['generated_token_ids']),
        'nfa_fa2_token_identity': (
            nfa_a['generated_token_ids'] == fa2_a['generated_token_ids']),
        'nfa_fa2_first_divergence': _first_divergence(
            nfa_a['generated_token_ids'], fa2_a['generated_token_ids']),
        'nfa_skip_policy_counts': nfa_a.get('skip_policy_counts'),
    }
    if args.measurement_mode == 'latency':
        result.update(
            nfa_steady_tpot_ms=nfa_ms,
            fa2_steady_tpot_ms=fa2_ms,
            nfa_steady_tpot_median_ms=statistics.median(nfa_ms),
            fa2_steady_tpot_median_ms=statistics.median(fa2_ms),
            fa2_over_nfa_tpot=(
                statistics.median(fa2_ms) / statistics.median(nfa_ms)),
            nfa_prefill_seconds=[nfa_a['prefill_seconds'], nfa_b['prefill_seconds']],
            fa2_prefill_seconds=[fa2_a['prefill_seconds'], fa2_b['prefill_seconds']],
        )
    else:
        result['memory'] = {'a': _memory_compare(nfa_a, fa2_a),
                            'b': _memory_compare(nfa_b, fa2_b)}
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_receipt(output_dir):
    head = subprocess.check_output(
        ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', REPO, 'diff', '--binary'])
    diff_path = os.path.join(output_dir, 'source.diff')
    with open(diff_path, 'wb') as handle:
        handle.write(diff)
    return {'head': head, 'diff_path': diff_path,
            'diff_sha256': hashlib.sha256(diff).hexdigest()}


def _model_receipt(model_path):
    result = {'path': os.path.abspath(model_path), 'metadata_sha256': {}}
    for name in ('config.json', 'model.safetensors.index.json'):
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            result['metadata_sha256'][name] = _sha256(path)
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = {}
    comparisons = []
    hygiene = {}
    input_receipt = None
    software_receipt = None
    for batch in args.batch_size:
        for execution in args.execution:
            key = f'B{batch}/{execution}'
            final_attempt = 0
            for attempt in range(2):
                final_attempt = attempt
                monitor_path = os.path.join(
                    args.output_dir,
                    f'gpu{args.gpu_id}_b{batch}_{execution}_attempt{attempt}_hygiene.jsonl')
                monitor = GpuMonitor(args.gpu_id, monitor_path)
                monitor.start()
                try:
                    arms, paths = _run_abba(args, batch, execution, attempt, monitor)
                finally:
                    monitor.stop()
                if not monitor.pollution:
                    break
                print(f'WARNING {key} attempt{attempt} 检测到其他 GPU 进程，作废重跑',
                      flush=True)
            comparison = _compare(args, batch, execution, arms)
            comparisons.append(comparison)
            if input_receipt is None:
                input_receipt = arms['nfa_a']['input']
                software_receipt = {
                    'torch': arms['nfa_a']['torch_version'],
                    'transformers': arms['nfa_a']['transformers_version'],
                }
            runs[key] = paths
            hygiene[key] = {
                'status': 'clean' if not monitor.pollution else 'polluted-rerun',
                'attempt': final_attempt,
                'log': monitor_path,
                'pollution_samples': len(monitor.pollution),
            }
            if args.measurement_mode == 'latency':
                print(
                    f"SUMMARY {key}: NFA={comparison['nfa_steady_tpot_median_ms']:.4f}ms "
                    f"FA2={comparison['fa2_steady_tpot_median_ms']:.4f}ms "
                    f"FA2/NFA={comparison['fa2_over_nfa_tpot']:.3f}x | "
                    f"NFA repeat identity={comparison['nfa_repeat_token_identity']} | "
                    f"NFA/FA2 identity={comparison['nfa_fa2_token_identity']} | "
                    f"hygiene={hygiene[key]['status']}", flush=True)
            else:
                memory_a = comparison['memory']['a']
                memory_b = comparison['memory']['b']
                print(
                    f"SUMMARY {key}: steady resident saved "
                    f"A={memory_a['steady_decode_resident_allocated_saved_gib']:.3f}GiB "
                    f"B={memory_b['steady_decode_resident_allocated_saved_gib']:.3f}GiB | "
                    f"hygiene={hygiene[key]['status']}（本模式不报告 TPOT）",
                    flush=True)

    receipt = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'benchmark_scope': (
            'actual Qwen3-8B weights, native RULER prompt, full GraphRunner.step; '
            'ABBA independent subprocesses; TPOT excludes prefill'),
        'arguments': vars(args),
        'source': _source_receipt(args.output_dir),
        'model': _model_receipt(args.model_path),
        'input': input_receipt,
        'gpu': {
            'id': args.gpu_id,
            'name': _smi(args.gpu_id, 'name'),
            'driver_version': _smi(args.gpu_id, 'driver_version'),
        },
        'software': {
            'python': sys.version,
            'executable': sys.executable,
            **software_receipt,
        },
        'memory_scope': (
            '独立 memory run；steady resident 是 post-warmup allocated 减 model-loaded '
            'allocated；legacy peak 含 prefill，仅作辅助'
            if args.measurement_mode == 'memory' else None),
        'gpu_hygiene': {
            'status': ('clean' if all(x['status'] == 'clean' for x in hygiene.values())
                       else 'polluted-rerun'),
            'configs': hygiene,
        },
        'runs': runs,
        'comparisons': comparisons,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    print(f'\nsaved comparison -> {args.output}')


if __name__ == '__main__':
    main()
