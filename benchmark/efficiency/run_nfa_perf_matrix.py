"""双卡断点续跑：NFA Graph 算子、干净 decode E2E 与 FlashInfer 单次缓存。

长度固定从大到小。NFA 算子使用独立 nsys Graph-node 进程；干净 E2E 不使用
任何 profiler。不同观测口径不得相减推导 host 开销。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


REPO = Path(__file__).resolve().parents[2]
MEASURE = REPO / 'benchmark/efficiency/measure_nfa_selector_e2e.py'
E2E = REPO / 'benchmark/efficiency/benchmark_nfa_end_to_end.py'
FI_MEASURE = REPO / 'benchmark/efficiency/measure_flashinfer_operator.py'
ARMS = (
    'nfa-operator-profile', 'nfa-clean-e2e',
    'flashinfer-operator-profile', 'flashinfer-e2e',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--data-root', type=Path, default=Path('/tmp/nfa_kernel_inputs'))
    parser.add_argument('--task', default='niah_multikey_2')
    parser.add_argument('--sample-index', type=int, default=2)
    parser.add_argument('--input-provenance', default=None, choices=(
        'native-ruler', 'repeated-needle-body-extension',
        'deterministic-longbench-context-repetition'))
    parser.add_argument('--longbench-data', default='/home/wangyuanshuo02/datasets/LongBench/data')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[524288, 262144, 131072, 65536, 32768])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--profile-repeat', type=int, default=5)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--subspace', choices=('full', 'tail'), default='tail')
    parser.add_argument('--l2-subspace', choices=('same', 'full'), default='same')
    parser.add_argument('--gate-epsilon', type=float, default=0.05)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_receipt(repo, harness):
    paths = (
        'sparse_attn', 'benchmark/rope_policy.py',
        'benchmark/efficiency/benchmark_nfa_end_to_end.py',
        'benchmark/efficiency/measure_nfa_selector_e2e.py',
        'benchmark/LongBench/prefill_frozen_skip.py',
        'benchmark/LongBench/config/dataset2prompt.json',
    )
    diff = subprocess.check_output(
        ['git', '-C', str(repo), 'diff', 'HEAD', '--binary', '--', *paths])
    return {
        'head': subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'diff_sha256': hashlib.sha256(diff).hexdigest(),
        'harness_sha256': sha256(harness),
    }


def make_sources(args):
    nfa = source_receipt(args.repo, args.repo / MEASURE.relative_to(REPO))
    return {
        'nfa-operator-profile': nfa,
        'nfa-clean-e2e': nfa,
        'flashinfer-operator-profile': source_receipt(
            args.repo, args.repo / FI_MEASURE.relative_to(REPO)),
        'flashinfer-e2e': source_receipt(
            args.repo, args.repo / E2E.relative_to(REPO)),
    }


def prepare_extended_input(args, length):
    if length <= 131072:
        return
    output = args.data_root / str(length) / f'{args.task}.jsonl'
    receipt_path = args.data_root / f'extended_input_receipt_{args.task}_{length}.json'
    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('sample_index') == args.sample_index
                and receipt.get('sha256') == sha256(output)):
            return
    from transformers import AutoTokenizer

    source = args.data_root / '131072' / f'{args.task}.jsonl'
    source_bytes = source.read_bytes()
    row = json.loads(source_bytes.splitlines()[args.sample_index])
    text = row['input']
    prefix, remainder = text.split('\n', 1)
    body, question = remainder.rsplit('\n', 1)
    prefix += '\n'
    question = '\n' + question
    repetitions = length // 131072
    if repetitions < 2:
        raise ValueError(f'unsupported extension length {length}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def token_count(value):
        return tokenizer(value, return_tensors='pt', truncation=False).input_ids.shape[-1]

    low, high = 0, len(body)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = prefix + body * (repetitions - 1) + body[:middle] + question
        if token_count(candidate) <= length:
            low = middle
        else:
            high = middle - 1
    extended = prefix + body * (repetitions - 1) + body[:low] + question
    actual = token_count(extended)
    if not 0.95 * length <= actual <= length:
        raise RuntimeError(f'extended prompt length {actual} outside {length} contract')
    row['input'] = extended
    row['length'] = actual
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f'.tmp{os.getpid()}')
    temporary.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    write_json(receipt_path, {
        'schema': 'nfa-repeated-needle-body-extension-v1',
        'source_path': str(source.resolve()),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'task': args.task,
        'sample_index': args.sample_index,
        'construction': '保留首行指令和最终问题，顺序重复 needle 正文，二分截断最后一次重复',
        'performance_only': True,
        'context_length': length,
        'path': str(output),
        'sha256': sha256(output),
        'actual_prompt_tokens': actual,
        'body_repetitions': repetitions,
        'last_body_characters': low,
        'provenance': 'repeated-needle-body-extension',
    })


def input_provenance(args, length):
    return args.input_provenance or (
        'native-ruler' if length <= 131072
        else 'repeated-needle-body-extension')


def input_path(args, length):
    if input_provenance(args, length) == 'deterministic-longbench-context-repetition':
        return Path(args.longbench_data) / f'{args.task}.jsonl'
    return args.data_root / str(length) / f'{args.task}.jsonl'


def gpu_state(gpu):
    used = int(subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    apps = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True).strip()
    rows = [line.strip() for line in apps.splitlines() if line.strip()]
    return {'used_mib': used, 'compute_process_rows': rows}


def gpu_total_gib(gpu):
    mib = int(subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.total',
        '--format=csv,noheader,nounits'], text=True).strip())
    return mib / 1024.0


class GpuMonitor:
    def __init__(self, gpu, path):
        self.gpu = gpu
        self.path = path
        self.active = False
        self.pollution = []
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        state = gpu_state(self.gpu)
        if state['used_mib'] >= 500 or state['compute_process_rows']:
            raise RuntimeError(f'GPU{self.gpu} 起跑前不干净: {state}')
        self._thread.start()

    def set_active(self, active):
        self.active = active

    def _sample(self):
        state = gpu_state(self.gpu)
        rows = state['compute_process_rows']
        # 容器 PID 与宿主 PID 不可直接比较；活跃测量只允许一个 compute process。
        unknown = len(rows) > 1 if self.active else bool(rows)
        record = {'timestamp': int(time.time()), 'active': self.active,
                  'unknown': unknown, **state}
        self.samples.append(record)
        with open(self.path, 'a', encoding='utf-8') as handle:
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


def memory_lower_bound(args, length, batch):
    config = json.loads((Path(args.model_path) / 'config.json').read_text())
    total_tokens = length + args.warmup_steps + max(
        getattr(args, 'profile_repeat', args.decode_steps), args.decode_steps) + 2
    kv_bytes = (
        batch * total_tokens * config['num_hidden_layers']
        * config['num_key_value_heads'] * config['head_dim'] * 2 * 2)
    required_gib = kv_bytes / 2**30 + 20.0
    return {
        'batch': batch,
        'kv_gib': kv_bytes / 2**30,
        'weights_and_runtime_reserve_gib': 20.0,
        'required_lower_bound_gib': required_gib,
        'gpu_total_gib': gpu_total_gib(args.gpu),
        'feasible': required_gib < gpu_total_gib(args.gpu),
        'note': 'paged frozen skip 与 FlashInfer 都保留全部物理 KV；这是容量下界，不是实测显存',
    }


def cells(args):
    # 长上下文和大 batch 优先；infeasible 格只记录，不占用运行时间。
    return [(length, batch)
            for length in sorted(args.lengths, reverse=True)
            for batch in sorted(args.batches, reverse=True)]


def arm_path(args, length, batch, arm):
    return args.output_dir / arm / f'{arm}_l{length}_b{batch}_graph.json'


def expected_fields(args, length, batch, arm):
    common = {
        'context_length': length,
        'batch_size': batch,
        'execution': 'graph',
        'task': args.task,
        'sample_index': args.sample_index,
        'input_provenance': input_provenance(args, length),
        'visible_len': args.visible_len,
        'warmup_steps': args.warmup_steps,
    }
    if arm.startswith('flashinfer-'):
        fields = {**common, 'method': 'flashinfer',
                  'max_new_tokens': args.warmup_steps + args.decode_steps + 1}
        if arm == 'flashinfer-operator-profile':
            fields.update(
                measurement_mode='operator-profile',
                profiled_decode_steps=args.decode_steps)
        return fields
    return {
        **common,
        'method': 'nfa',
        'measurement_mode': (
            'operator-profile' if arm == 'nfa-operator-profile' else 'clean-e2e'),
        'subspace': args.subspace,
        'l2_subspace': args.l2_subspace,
        'gate_epsilon': args.gate_epsilon,
        'operator_mode': 'off',
        'profile_repeat': args.profile_repeat,
        'decode_steps': args.decode_steps,
    }


def valid_existing(args, path, length, batch, arm, source):
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    expected_schema = (
        'qwen-e2e-decode-v2' if arm.startswith('flashinfer-')
        else 'nfa-selector-and-e2e-v2')
    if (row.get('schema') != expected_schema or row.get('source') != source
            or row.get('gpu_hygiene') != 'clean'
            or any(row.get(key) != value for key, value in
                   expected_fields(args, length, batch, arm).items())):
        return False
    source_input_path = input_path(args, length)
    if row.get('input', {}).get('sha256') != sha256(source_input_path):
        return False
    if row.get('graph_eager_fallbacks') != 0:
        return False
    if arm.startswith('flashinfer-'):
        runtime = row.get('flashinfer_runtime', {})
        if not (
                row.get('graph_capture_count') == 1
                and runtime.get('page_size') == 64
                and runtime.get('page_size_matches_nfa') is True
                and runtime.get('wrapper')
                == 'CUDAGraphBatchDecodeWithPagedKVCacheWrapper'):
            return False
        if arm == 'flashinfer-e2e':
            return row.get('steady_tpot_ms_median', 0) > 0
    elif row.get('graph_capture_count') != 1 or 'skip_validation' not in row:
        return False
    if arm == 'nfa-clean-e2e':
        return row.get('timing') is not None
    if not row.get('nsys_profile_finalized'):
        return False
    receipt = row.get('nsys_receipt', {})
    for key in ('report', 'sqlite'):
        artifact = Path(receipt.get(f'{key}_path', ''))
        if (not artifact.is_file()
                or receipt.get(f'{key}_sha256') != sha256(artifact)):
            return False
    return bool(row.get('operator_summary') and row.get('kernel_summary'))


def measure_command(args, length, batch, arm, output):
    common = [
        '--model-path', args.model_path,
        '--data-root', str(args.data_root),
        '--task', args.task,
        '--context-length', str(length),
        '--batch-size', str(batch),
        '--execution', 'graph',
        '--sample-index', str(args.sample_index),
        '--input-provenance', input_provenance(args, length),
        '--visible-len', str(args.visible_len),
        '--warmup-steps', str(args.warmup_steps),
        '--output', str(output),
    ]
    if arm.startswith('flashinfer-'):
        script = FI_MEASURE if arm == 'flashinfer-operator-profile' else E2E
        command = [
            sys.executable, '-u', str(args.repo / script.relative_to(REPO)),
            *common, '--method', 'flashinfer',
            '--max-new-tokens', str(args.warmup_steps + args.decode_steps + 1),
        ]
        if arm == 'flashinfer-e2e':
            return command
    else:
        mode = 'operator-profile' if arm == 'nfa-operator-profile' else 'clean-e2e'
        command = [
            sys.executable, '-u', str(args.repo / MEASURE.relative_to(REPO)),
            *common,
            '--measurement-mode', mode,
            '--subspace', args.subspace,
            '--l2-subspace', args.l2_subspace,
            '--gate-epsilon', str(args.gate_epsilon), '--operator-mode', 'off',
            '--profile-repeat', str(args.profile_repeat),
            '--decode-steps', str(args.decode_steps),
            '--longbench-data', str(args.longbench_data),
        ]
        if arm != 'nfa-operator-profile':
            return command
    report_base = output.with_suffix('')
    return [
        'nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
        '--cuda-event-trace=false', '--cuda-graph-trace=node',
        '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
        '--export=sqlite', '--force-overwrite=true', f'--output={report_base}',
        *command,
    ]


def run_arm(args, length, batch, arm, source):
    output = arm_path(args, length, batch, arm)
    if valid_existing(args, output, length, batch, arm, source):
        print(json.dumps({'skip': str(output), 'reason': 'valid-existing'}), flush=True)
        return {'status': 'skipped', 'output': str(output)}
    log = output.with_suffix('.log')
    monitor_path = output.with_suffix('.hygiene.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = measure_command(args, length, batch, arm, output)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['NFA_REPO'] = str(args.repo.resolve())
    if length > 131072:
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    python_paths = [str(args.repo.resolve())]
    if env.get('PYTHONPATH'):
        python_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(python_paths)
    last = None
    for attempt in range(1, args.max_attempts + 1):
        script = {
            'flashinfer-e2e': E2E,
            'flashinfer-operator-profile': FI_MEASURE,
        }.get(arm, MEASURE)
        if source_receipt(
                args.repo, args.repo / script.relative_to(REPO)) != source:
            raise RuntimeError('测量期间源代码身份发生变化，停止复用旧回执')
        monitor_path.unlink(missing_ok=True)
        monitor = GpuMonitor(args.gpu, monitor_path)
        try:
            monitor.start()
        except RuntimeError as exc:
            print(f'WARNING {output.name} 起跑等待干净 GPU: {exc}', flush=True)
            if attempt == args.max_attempts:
                raise
            time.sleep(2)
            continue
        try:
            with open(log, 'w', encoding='utf-8') as log_handle:
                process = subprocess.Popen(
                    command, cwd=str(args.repo), env=env,
                    stdout=log_handle, stderr=subprocess.STDOUT)
                monitor.set_active(True)
                returncode = process.wait()
                monitor.set_active(False)
        finally:
            monitor.stop()
        last = {
            'status': 'ran', 'output': str(output), 'log': str(log),
            'attempt': attempt, 'returncode': returncode,
            'hygiene_samples': monitor.samples,
            'pollution_samples': len(monitor.pollution),
        }
        if returncode != 0:
            raise RuntimeError(f'{output.name} 失败，日志: {log}')
        row = json.loads(output.read_text(encoding='utf-8'))
        if arm == 'flashinfer-operator-profile':
            base_source = source_receipt(
                args.repo, args.repo / E2E.relative_to(REPO))
            if row.get('source') != base_source:
                raise RuntimeError(f'{output.name} FlashInfer 基础路径源码身份变化')
            row['base_source'] = row['source']
            row['source'] = source
        elif row.get('source') != source:
            raise RuntimeError(f'{output.name} 运行前后源代码身份变化')
        row['gpu_hygiene'] = 'polluted-rerun' if monitor.pollution else 'clean'
        row['gpu'] = args.gpu
        row['arm'] = arm
        if arm.startswith('nfa-'):
            active = []
            for request_id in range(batch):
                skip = row['skip'][str(request_id)]
                active.append(bool(skip['far_layers'] or skip['near_layers']))
            row['skip_validation'] = (
                'active' if all(active) else 'partial' if any(active) else 'inactive')
        if not monitor.pollution and arm in (
                'nfa-operator-profile', 'flashinfer-operator-profile'):
            if arm == 'nfa-operator-profile':
                from benchmark.efficiency.measure_nfa_selector_e2e import (
                    finalize_nsys_profile)
            else:
                from benchmark.efficiency.measure_flashinfer_operator import (
                    finalize_nsys_profile)
            report = output.with_suffix('.nsys-rep')
            sqlite = output.with_suffix('.sqlite')
            finalize_nsys_profile(row, sqlite)
            row['nsys_receipt'] = {
                'version': subprocess.check_output(
                    ['nsys', '--version'], text=True).strip(),
                'report_path': str(report), 'report_sha256': sha256(report),
                'sqlite_path': str(sqlite), 'sqlite_sha256': sha256(sqlite),
                'trace': 'cuda', 'cuda_graph_trace': 'node',
                'capture_range': 'cudaProfilerApi around steady decode only',
                'observer_effect': (
                    'Graph node tracing may significantly perturb runtime; values '
                    'are observed kernel times, not zero-overhead ground truth'),
            }
        write_json(output, row)
        if not monitor.pollution:
            if not valid_existing(args, output, length, batch, arm, source):
                raise RuntimeError(f'{output} 回执身份、配置或证据校验失败')
            summary = {'done': output.name, 'hygiene': row['gpu_hygiene']}
            if arm == 'nfa-operator-profile':
                summary.update({
                    key: row['operator_summary'][key]['mean_ms_per_step']
                    for key in ('l1_ms', 'l2_ms', 'topk_total_ms',
                                'main_attention_ms', 'nfa_operator_total_ms')})
            elif arm == 'flashinfer-operator-profile':
                summary.update({
                    key: row['operator_summary'][key]['mean_ms_per_step']
                    for key in ('attention_main_ms', 'attention_merge_ms',
                                'kv_append_ms', 'attention_without_append_ms',
                                'attention_with_append_ms')})
            else:
                summary['tpot_ms'] = row.get('timing', row)['steady_tpot_ms_median']
            print(json.dumps(summary), flush=True)
            return last
        print(f'WARNING {output.name} 检测到其他 GPU 进程，作废重跑', flush=True)
    raise RuntimeError(f'{output.name} 多次受 GPU 污染: {last}')


def first_divergence(left, right):
    for row, (left_ids, right_ids) in enumerate(zip(left, right)):
        for index, (a, b) in enumerate(zip(left_ids, right_ids)):
            if a != b:
                return {'row': row, 'token_index': index, 'left': a, 'right': b}
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        return {'length_mismatch': True}
    return None


def summarize(args):
    sources = make_sources(args)
    records = []
    missing = []
    infeasible = []
    for length, batch in cells(args):
        memory = memory_lower_bound(args, length, batch)
        if not memory['feasible']:
            infeasible.append({'length': length, 'batch': batch, 'memory': memory})
            continue
        rows = {}
        for arm in args.arms:
            path = arm_path(args, length, batch, arm)
            if not valid_existing(args, path, length, batch, arm, sources[arm]):
                missing.append(str(path))
                continue
            rows[arm] = json.loads(path.read_text(encoding='utf-8'))
        if len(rows) != len(args.arms):
            continue
        record = {'length': length, 'batch': batch, 'execution': 'graph', 'arms': {}}
        profile = rows.get('nfa-operator-profile')
        clean = rows.get('nfa-clean-e2e')
        flashinfer_profile = rows.get('flashinfer-operator-profile')
        flashinfer = rows.get('flashinfer-e2e')
        if profile is not None:
            record['arms']['nfa_operator_profile'] = {
                'operator_summary': profile['operator_summary'],
                'kernel_summary': profile['kernel_summary'],
                'profiled_decode_steps': profile['profiled_decode_steps'],
                'nsys_receipt': profile['nsys_receipt'],
                'path': str(arm_path(args, length, batch, 'nfa-operator-profile')),
            }
        if clean is not None:
            record['arms']['nfa_clean_e2e'] = {
                'tpot_ms': clean['timing']['steady_tpot_ms_median'],
                'prefill_seconds': clean['prefill_seconds'],
                'skip_validation': clean['skip_validation'],
                'skip': clean['skip'],
                'path': str(arm_path(args, length, batch, 'nfa-clean-e2e')),
            }
        if flashinfer_profile is not None:
            record['arms']['flashinfer_operator_profile'] = {
                'operator_summary': flashinfer_profile['operator_summary'],
                'kernel_summary': flashinfer_profile['kernel_summary'],
                'profiled_decode_steps': flashinfer_profile['profiled_decode_steps'],
                'nsys_receipt': flashinfer_profile['nsys_receipt'],
                'path': str(arm_path(
                    args, length, batch, 'flashinfer-operator-profile')),
            }
        if flashinfer is not None:
            record['arms']['flashinfer_e2e'] = {
                'tpot_ms': flashinfer['steady_tpot_ms_median'],
                'prefill_seconds': flashinfer['prefill_seconds'],
                'runtime': flashinfer['flashinfer_runtime'],
                'path': str(arm_path(args, length, batch, 'flashinfer-e2e')),
            }
        if profile is not None and clean is not None:
            profile_tokens = profile['generated_token_ids']
            clean_tokens = clean['generated_token_ids']
            prefix = [row[:len(profile_tokens[index])]
                      for index, row in enumerate(clean_tokens)]
            if profile_tokens != prefix:
                raise ValueError(f'NFA profile/clean token 分歧: L{length} B{batch}')
            record['nfa_profile_clean_token_identity_common_prefix'] = True
        if flashinfer_profile is not None and flashinfer is not None:
            profile_tokens = flashinfer_profile['generated_token_ids']
            clean_tokens = flashinfer['generated_token_ids']
            divergence = first_divergence(profile_tokens, clean_tokens)
            if divergence is not None:
                raise ValueError(
                    f'FlashInfer profile/clean token 分歧: L{length} B{batch}: '
                    f'{divergence}')
            record['flashinfer_profile_clean_token_identity'] = True
        if profile is not None and flashinfer_profile is not None:
            nfa_summary = profile['operator_summary']
            fi_summary = flashinfer_profile['operator_summary']
            fi_attention = fi_summary[
                'attention_without_append_ms']['mean_ms_per_step']
            nfa_attention = nfa_summary['main_attention_ms']['mean_ms_per_step']
            nfa_path = nfa_summary['nfa_operator_total_ms']['mean_ms_per_step']
            if min(fi_attention, nfa_attention, nfa_path) <= 0:
                raise ValueError(f'算子时间必须为正: L{length} B{batch}')
            record['speedups'] = {
                'pure_attention_kernel': fi_attention / nfa_attention,
                'attention_path_including_nfa_selection': fi_attention / nfa_path,
                'flashinfer_attention_without_append_ms_per_step': fi_attention,
                'flashinfer_kv_append_ms_per_step': fi_summary[
                    'kv_append_ms']['mean_ms_per_step'],
                'nfa_selected_attention_ms_per_step': nfa_attention,
                'nfa_selection_plus_attention_ms_per_step': nfa_path,
                'definitions': {
                    'pure_attention_kernel': (
                        'FlashInfer dense attention main+merge kernels / '
                        'NFA selected-attention kernels'),
                    'attention_path_including_nfa_selection': (
                        'FlashInfer dense attention main+merge kernels / '
                        '(NFA selector + selected-attention kernels)'),
                    'kv_append': 'reported separately; excluded from both speedups',
                },
            }
        if clean is not None and flashinfer is not None:
            record['nfa_vs_flashinfer_tpot_speedup'] = (
                clean['timing']['steady_tpot_ms_median'] and
                flashinfer['steady_tpot_ms_median']
                / clean['timing']['steady_tpot_ms_median'])
            if 'speedups' in record:
                record['speedups']['clean_decode_tpot'] = (
                    record['nfa_vs_flashinfer_tpot_speedup'])
                record['speedups']['definitions']['clean_decode_tpot'] = (
                    'FlashInfer clean end-to-end decode TPOT / '
                    'NFA clean end-to-end decode TPOT')
            record['nfa_flashinfer_token_identity'] = (
                clean['generated_token_ids'] == flashinfer['generated_token_ids'])
            record['nfa_flashinfer_first_divergence'] = first_divergence(
                clean['generated_token_ids'], flashinfer['generated_token_ids'])
        record['input'] = (
            clean or profile or flashinfer_profile or flashinfer)['input']
        records.append(record)
    if missing:
        raise RuntimeError('矩阵尚缺臂或回执无效:\n' + '\n'.join(missing[:30]))
    result = {
        'schema': 'nfa-flashinfer-graph-performance-v3',
        'validation_status': 'pending-independent-verification',
        'source_by_arm': sources,
        'matrix_driver_sha256': sha256(Path(__file__)),
        'task': args.task,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'protocol': {
            'length_order': 'descending',
            'execution': 'CUDA Graph only',
            'nfa_operator': 'separate nsys CUDA Graph node trace process',
            'nfa_clean_e2e': 'separate process without nsys or torch.profiler',
            'flashinfer_operator': (
                'separate nsys CUDA Graph node trace process; dense attention '
                'main+merge and KV append are classified separately'),
            'flashinfer_clean_e2e': (
                'one clean run per cell without nsys or torch.profiler; valid '
                'receipts are cached and skipped'),
            'speedup_definitions': (
                'pure attention = FI dense attention main+merge / NFA selected '
                'attention; attention path = FI dense attention main+merge / '
                '(NFA selector + selected attention); clean TPOT = FI clean '
                'end-to-end decode TPOT / NFA clean end-to-end decode TPOT'),
            'host_overhead_derivable_by_subtraction': False,
            'observer_effect': (
                'nsys Graph node tracing may perturb kernel time; never subtract '
                'profiled operator total from clean TPOT to infer host overhead'),
            'timing_metric': 'end-to-end decode TPOT, not pure attention latency',
        },
        'records': records,
        'infeasible_memory_cells': infeasible,
    }
    output = args.output_dir / 'summary.json'
    write_json(output, result)
    print(f'saved summary -> {output}; records={len(records)}; infeasible={len(infeasible)}')
    return result


def main():
    args = parse_args()
    args.repo = args.repo.resolve()
    args.output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(args.repo))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('invalid shard')
    for length in args.lengths:
        prepare_extended_input(args, length)
    if args.prepare_only:
        return
    if args.summarize_only:
        summarize(args)
        return
    sources = make_sources(args)
    receipt_path = args.output_dir / f'driver_gpu{args.gpu}_shard{args.shard_index}.json'
    receipt = {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'source_by_arm': sources,
        'matrix_driver_sha256': sha256(Path(__file__)),
        'events': [], 'infeasible': [],
    }
    write_json(receipt_path, receipt)
    for index, (length, batch) in enumerate(cells(args)):
        if index % args.num_shards != args.shard_index:
            continue
        memory = memory_lower_bound(args, length, batch)
        if not memory['feasible']:
            event = {'length': length, 'batch': batch,
                     'status': 'infeasible-memory', 'memory': memory}
            receipt['infeasible'].append(event)
            write_json(receipt_path, receipt)
            print(json.dumps(event), flush=True)
            continue
        for arm in args.arms:
            event = run_arm(args, length, batch, arm, sources[arm])
            receipt['events'].append({
                'length': length, 'batch': batch, 'arm': arm,
                'status': event['status'], 'output': event['output'],
            })
            write_json(receipt_path, receipt)
    print(f'driver complete -> {receipt_path}', flush=True)


if __name__ == '__main__':
    main()
