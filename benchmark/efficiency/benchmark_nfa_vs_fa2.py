"""真实 Qwen3-8B + 显式来源长文本的双方法完整 decode ABBA 对照驱动。

每个 arm 启动独立进程，按 A、B、B、A 顺序测量，只报告生产 decode
TPOT、吞吐量与正确性，不提供显存测量模式。
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
    parser.add_argument('--task', default='niah_multikey_1')
    parser.add_argument('--input-provenance', default='native-ruler',
                        choices=('native-ruler', 'repeated-needle-body-extension',
                                 'longbench-v2'),
                        help='输入数据来源必须显式声明并透传到每个测量臂')
    parser.add_argument('--context-length', type=int, default=32768,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1],
                        choices=(1, 2, 4))
    parser.add_argument('--execution', nargs='+', default=['eager', 'graph'],
                        choices=('eager', 'graph'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--gate-epsilon', type=float, default=0.05)
    parser.add_argument('--methods', nargs='+', default=['nfa', 'flashinfer'],
                        choices=('nfa', 'flashinfer'),
                        help='恰好两个方法：前者为候选，后者为 dense 对照')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--output-dir',
                        default='/tmp/nfa_vs_flashinfer_qwen3_8b_ruler32k')
    parser.add_argument('--output',
                        default='/tmp/nfa_vs_flashinfer_qwen3_8b_ruler32k.json')
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
    """30 秒旁路监控；活动 arm 只允许一个 CUDA 进程，空闲期必须为空。"""

    def __init__(self, gpu_id, log_path):
        self.gpu_id = gpu_id
        self.log_path = log_path
        self.allowed_pids = set()
        self.active_pid = None
        self.pollution = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        _preflight(self.gpu_id)
        self._thread.start()

    def register(self, pid):
        # nvidia-smi 返回的宿主 PID 在当前容器 /proc 中不可见，Ngid 也不是
        # 宿主 PID。各 arm 已串行独立起跑且 preflight 为空，因此活动期的
        # 唯一 CUDA 进程归属当前 runner；其他组合一律按污染处理。
        with self._lock:
            if self.active_pid is not None:
                raise RuntimeError('GpuMonitor 不允许并发注册多个活动 arm')
            self.active_pid = pid
            self.allowed_pids.add(pid)

    def unregister(self, pid):
        with self._lock:
            if self.active_pid != pid:
                raise RuntimeError(f'GpuMonitor 活动 arm 不匹配: {pid}')
            self.active_pid = None

    def _sample(self):
        timestamp = int(time.time())
        gpu = _smi(self.gpu_id, 'index,memory.used,utilization.gpu')
        apps = _smi(self.gpu_id, 'pid,used_memory', apps=True)
        rows = [row.strip() for row in apps.splitlines() if row.strip()]
        with self._lock:
            allowed = set(self.allowed_pids)
            active_pid = self.active_pid
        owned_apps = rows if active_pid is not None and len(rows) == 1 else []
        unknown = [] if not rows or owned_apps else rows
        record = {'timestamp': timestamp, 'gpu': gpu, 'apps': rows,
                  'allowed_pids': sorted(allowed), 'active_pid': active_pid,
                  'owned_apps': owned_apps, 'unknown_apps': unknown}
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
    input_family = ('ruler' if args.input_provenance == 'native-ruler'
                    else args.input_provenance)
    stem = (f'{tag}_{input_family}{args.context_length // 1024}k_b{batch}_'
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
        '--input-provenance', args.input_provenance,
        '--max-new-tokens', str(args.max_new_tokens),
        '--warmup-steps', str(args.warmup_steps),
        '--visible-len', str(args.visible_len),
        '--gate-epsilon', str(args.gate_epsilon),
        '--measurement-mode', 'latency',
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
        try:
            returncode = process.wait()
        finally:
            monitor.unregister(process.pid)
    if returncode:
        raise RuntimeError(f'{tag} 失败，日志: {log_path}')
    with open(output, encoding='utf-8') as handle:
        result = json.load(handle)
    return result, {'json': output, 'log': log_path, 'pid': process.pid}


def _run_abba(args, batch, execution, attempt, monitor):
    arms = {}
    paths = {}
    candidate, baseline = args.methods
    for method, leg in ((candidate, 'a'), (baseline, 'a'),
                        (baseline, 'b'), (candidate, 'b')):
        result, path = _run_one(
            args, batch, execution, method, leg, attempt, monitor)
        key = f'{method}_{leg}'
        arms[key] = result
        paths[key] = path
    return arms, paths


def _first_divergence(lhs, rhs):
    """返回任意 batch 行内的首个 token 分叉，而不是只检查第 0 行。"""
    for batch, (left_row, right_row) in enumerate(zip(lhs, rhs)):
        for index, (left, right) in enumerate(zip(left_row, right_row)):
            if left != right:
                return {'batch': batch, 'token': index}
        if len(left_row) != len(right_row):
            return {'batch': batch, 'token': min(len(left_row), len(right_row))}
    if len(lhs) != len(rhs):
        return {'batch': min(len(lhs), len(rhs)), 'token': 0}
    return None


def _compare(args, batch, execution, arms):
    candidate, baseline = args.methods
    candidate_a, candidate_b = arms[f'{candidate}_a'], arms[f'{candidate}_b']
    baseline_a, baseline_b = arms[f'{baseline}_a'], arms[f'{baseline}_b']
    for method in (candidate, baseline):
        lhs, rhs = arms[f'{method}_a'], arms[f'{method}_b']
        if lhs['input']['sha256'] != rhs['input']['sha256']:
            raise RuntimeError(f'{method} ABBA 输入 SHA256 不一致')
    if candidate_a['input']['sha256'] != baseline_a['input']['sha256']:
        raise RuntimeError('对照方法输入 SHA256 不一致')
    if (candidate_a['input']['actual_prompt_tokens']
            != baseline_a['input']['actual_prompt_tokens']):
        raise RuntimeError('对照方法 prompt 长度不一致')
    candidate_ms = [
        candidate_a['steady_tpot_ms_median'],
        candidate_b['steady_tpot_ms_median']]
    baseline_ms = [
        baseline_a['steady_tpot_ms_median'],
        baseline_b['steady_tpot_ms_median']]
    candidate_median = statistics.median(candidate_ms)
    baseline_median = statistics.median(baseline_ms)
    result = {
        'batch_size': batch,
        'execution': execution,
        'candidate_method': candidate,
        'baseline_method': baseline,
        f'{candidate}_repeat_token_identity': (
            candidate_a['generated_token_ids'] == candidate_b['generated_token_ids']),
        f'{baseline}_repeat_token_identity': (
            baseline_a['generated_token_ids'] == baseline_b['generated_token_ids']),
        f'{candidate}_{baseline}_token_identity': (
            candidate_a['generated_token_ids'] == baseline_a['generated_token_ids']),
        f'{candidate}_{baseline}_first_divergence': _first_divergence(
            candidate_a['generated_token_ids'], baseline_a['generated_token_ids']),
        f'{candidate}_steady_tpot_ms': candidate_ms,
        f'{baseline}_steady_tpot_ms': baseline_ms,
        f'{candidate}_steady_tpot_median_ms': candidate_median,
        f'{baseline}_steady_tpot_median_ms': baseline_median,
        f'{candidate}_throughput_tokens_per_second': 1000.0 / candidate_median,
        f'{baseline}_throughput_tokens_per_second': 1000.0 / baseline_median,
        f'{baseline}_over_{candidate}_tpot': baseline_median / candidate_median,
        f'{candidate}_prefill_seconds': [
            candidate_a['prefill_seconds'], candidate_b['prefill_seconds']],
        f'{baseline}_prefill_seconds': [
            baseline_a['prefill_seconds'], baseline_b['prefill_seconds']],
    }
    if candidate == 'nfa':
        result['nfa_skip_policy_counts'] = candidate_a.get('skip_policy_counts')
        result['nfa_skip_policy_layers'] = candidate_a.get('skip_policy_layers')
        result['nfa_skip_decode_receipt'] = candidate_a.get('skip_decode_receipt')
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
    # FlashInfer runtime 首次引入时可能尚未 tracked；正式 receipt 必须包含
    # 实际执行的源码，不能只记录一个看似完整但缺失新文件的 tracked diff。
    runtime_source = 'sparse_attn/patches/qwen3_flashinfer_patch.py'
    untracked = subprocess.check_output(
        ['git', '-C', REPO, 'ls-files', '--others', '--exclude-standard',
         '--', runtime_source], text=True).strip()
    if untracked:
        added = subprocess.run(
            ['git', '-C', REPO, 'diff', '--no-index', '--binary',
             '/dev/null', runtime_source], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        if added.returncode != 1:
            raise RuntimeError(added.stderr.decode(errors='replace'))
        diff += added.stdout
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
    if len(args.methods) != 2 or len(set(args.methods)) != 2:
        raise ValueError('--methods 必须指定两个不同方法，前者为候选、后者为对照')
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
            else:
                raise RuntimeError(
                    f'{key} 两次 ABBA 均受其他 GPU 进程污染，拒绝输出有效判决')
            comparison = _compare(args, batch, execution, arms)
            comparisons.append(comparison)
            if input_receipt is None:
                first_arm = arms[f'{args.methods[0]}_a']
                input_receipt = first_arm['input']
                software_receipt = {
                    'torch': first_arm['torch_version'],
                    'transformers': first_arm['transformers_version'],
                }
            runs[key] = paths
            hygiene[key] = {
                'status': 'clean' if not monitor.pollution else 'polluted-rerun',
                'attempt': final_attempt,
                'log': monitor_path,
                'pollution_samples': len(monitor.pollution),
            }
            candidate, baseline = args.methods
            candidate_ms = comparison[f'{candidate}_steady_tpot_median_ms']
            baseline_ms = comparison[f'{baseline}_steady_tpot_median_ms']
            speedup = comparison[f'{baseline}_over_{candidate}_tpot']
            repeat_identity = comparison[f'{candidate}_repeat_token_identity']
            cross_identity = comparison[f'{candidate}_{baseline}_token_identity']
            print(
                f'SUMMARY {key}: {candidate}={candidate_ms:.4f}ms '
                f'({1000.0 / candidate_ms:.2f} tok/s) | '
                f'{baseline}={baseline_ms:.4f}ms '
                f'({1000.0 / baseline_ms:.2f} tok/s) | '
                f'{baseline}/{candidate}={speedup:.3f}x | '
                f'{candidate} repeat identity={repeat_identity} | '
                f'{candidate}/{baseline} identity={cross_identity} | '
                f"hygiene={hygiene[key]['status']}", flush=True)

    receipt = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'benchmark_scope': (
            f'actual Qwen3-8B weights, {args.input_provenance} prompt, '
            'complete decode step; '
            'ABBA independent subprocesses (two measured repeats per method); '
            f'first {args.warmup_steps} decode steps excluded from steady TPOT; '
            'TPOT excludes prefill; throughput is per-request tokens/s'),
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
