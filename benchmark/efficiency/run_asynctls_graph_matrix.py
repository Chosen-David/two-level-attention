"""AsyncTLS PyTorch 语义复现的 Graph 算子观测与干净 decode TPOT 矩阵。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.benchmark_asynctls_graph import _source_receipt
from benchmark.efficiency.measure_asynctls_graph_operator import finalize
from benchmark.efficiency.run_nfa_perf_matrix import (
    GpuMonitor, cells, memory_lower_bound, prepare_extended_input, sha256,
    write_json,
)

BENCHMARK = REPO / 'benchmark/efficiency/benchmark_asynctls_graph.py'
ARMS = ('asynctls-operator-profile', 'asynctls-clean-e2e')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--data-root', type=Path, default=Path('/tmp/nfa_kernel_inputs'))
    parser.add_argument('--channel-file', type=Path, default=(
        REPO / 'dimension_channels/qwen3-8b/qasper_channels_per_group.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[524288, 262144, 131072, 65536, 32768])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--task', default='niah_multikey_2')
    parser.add_argument('--sample-index', type=int, default=2)
    parser.add_argument('--visible-len', type=int, default=8192)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--top-kb', type=int, default=128)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def arm_path(args, length, batch, arm):
    return args.output_dir / arm / f'{arm}_l{length}_b{batch}_graph.json'


def async_memory(args, length, batch):
    memory = memory_lower_bound(args, length, batch)
    metadata_gib = memory['kv_gib'] / args.block_size
    memory['asynctls_block_metadata_lower_bound_gib'] = metadata_gib
    memory['required_lower_bound_gib'] += metadata_gib
    memory['feasible'] = memory['required_lower_bound_gib'] < memory['gpu_total_gib']
    memory['note'] = (
        'BF16 K/V plus BF16 per-block K min/max lower bound; fixed candidate '
        'workspace and runtime reserve are not a measured peak-memory claim.')
    return memory


def expected(args, length, batch, arm):
    profile = arm == 'asynctls-operator-profile'
    return {
        'schema': 'qwen-asynctls-semantic-graph-e2e-v1',
        'method': 'asynctls-semantic-pytorch',
        'execution': 'graph',
        'measurement_mode': 'operator-profile' if profile else 'clean-e2e',
        'context_length': length,
        'batch_size': batch,
        'task': args.task,
        'sample_index': args.sample_index,
        'input_provenance': (
            'native-ruler' if length <= 131072
            else 'repeated-needle-body-extension'),
        'visible_len': args.visible_len,
        'warmup_steps': args.warmup_steps,
        'max_new_tokens': args.warmup_steps + args.decode_steps + 1,
        'profiled_decode_steps': args.decode_steps if profile else 0,
        'channel_file': str(args.channel_file.resolve()),
        'channel_file_sha256': sha256(args.channel_file),
        'official_tilelang': False,
        'real_async_offload': False,
    }


def valid_existing(args, length, batch, arm, source):
    path = arm_path(args, length, batch, arm)
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    if (row.get('source') != source or row.get('gpu_hygiene') != 'clean'
            or any(row.get(key) != value
                   for key, value in expected(args, length, batch, arm).items())):
        return False
    input_path = args.data_root / str(length) / f'{args.task}.jsonl'
    if row.get('input', {}).get('sha256') != sha256(input_path):
        return False
    runtime = row.get('asynctls_runtime', {})
    if (row.get('graph_capture_count') != 1
            or row.get('graph_eager_fallbacks') != 0
            or runtime.get('token_budget') != args.budget
            or runtime.get('block_size') != args.block_size
            or runtime.get('top_kb') != args.top_kb
            or runtime.get('channels_per_kv_group') != 32
            or runtime.get('pipeline') != 'Stage2 uses previous-step coarse blocks M_(t-1)'):
        return False
    if arm == 'asynctls-clean-e2e':
        return row.get('tpot_ms_median', 0) > 0
    if not row.get('nsys_profile_finalized'):
        return False
    for key in ('report', 'sqlite'):
        receipt = row.get('nsys_receipt', {})
        artifact = Path(receipt.get(f'{key}_path', ''))
        if (not artifact.is_file()
                or receipt.get(f'{key}_sha256') != sha256(artifact)):
            return False
    return bool(row.get('operator_summary') and row.get('kernel_summary'))


def command(args, length, batch, arm, output):
    profile = arm == 'asynctls-operator-profile'
    repeat = args.decode_steps
    base = [
        sys.executable, '-u', str(args.repo / BENCHMARK.relative_to(REPO)),
        '--model-path', args.model_path,
        '--data-root', str(args.data_root),
        '--output', str(output),
        '--task', args.task,
        '--context-length', str(length),
        '--batch-size', str(batch),
        '--execution', 'graph',
        '--sample-index', str(args.sample_index),
        '--input-provenance',
        'native-ruler' if length <= 131072 else 'repeated-needle-body-extension',
        '--max-new-tokens', str(args.warmup_steps + repeat + 1),
        '--warmup-steps', str(args.warmup_steps),
        '--visible-len', str(args.visible_len),
        '--measurement-mode', 'operator-profile' if profile else 'clean-e2e',
        '--asynctls-channel-file', str(args.channel_file),
        '--asynctls-budget', str(args.budget),
        '--asynctls-block-size', str(args.block_size),
        '--asynctls-top-kb', str(args.top_kb),
    ]
    if not profile:
        return base
    return [
        'nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
        '--cuda-event-trace=false', '--cuda-graph-trace=node',
        '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
        '--export=sqlite', '--force-overwrite=true',
        f'--output={output.with_suffix("")}', *base,
    ]


def run_arm(args, length, batch, arm, source):
    output = arm_path(args, length, batch, arm)
    if valid_existing(args, length, batch, arm, source):
        print(json.dumps({'skip': output.name, 'reason': 'valid-existing'}), flush=True)
        return
    log = output.with_suffix('.log')
    hygiene = output.with_suffix('.hygiene.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['NFA_REPO'] = str(args.repo.resolve())
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [
        str(args.repo.resolve()), env.get('PYTHONPATH')]))
    if length > 131072:
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    for attempt in range(1, args.max_attempts + 1):
        if _source_receipt() != source:
            raise RuntimeError('测量期间 AsyncTLS 或公共模型路径源码身份发生变化')
        hygiene.unlink(missing_ok=True)
        monitor = GpuMonitor(args.gpu, hygiene)
        try:
            monitor.start()
        except RuntimeError as exc:
            print(f'WARNING {output.name} 起跑 GPU 不干净: {exc}', flush=True)
            if attempt == args.max_attempts:
                raise
            continue
        try:
            with open(log, 'w', encoding='utf-8') as handle:
                process = subprocess.Popen(
                    command(args, length, batch, arm, output),
                    cwd=str(args.repo), env=env,
                    stdout=handle, stderr=subprocess.STDOUT)
                monitor.set_active(True)
                returncode = process.wait()
                monitor.set_active(False)
        finally:
            monitor.stop()
        if returncode != 0:
            raise RuntimeError(f'{output.name} 失败，日志: {log}')
        row = json.loads(output.read_text(encoding='utf-8'))
        if row.get('source') != source:
            raise RuntimeError(f'{output.name} 运行前后源码身份变化')
        row['gpu_hygiene'] = 'polluted-rerun' if monitor.pollution else 'clean'
        row['gpu'] = args.gpu
        row['arm'] = arm
        if not monitor.pollution and arm == 'asynctls-operator-profile':
            finalize(row, output.with_suffix('.sqlite'))
            report = output.with_suffix('.nsys-rep')
            sqlite = output.with_suffix('.sqlite')
            row['nsys_receipt'] = {
                'version': subprocess.check_output(
                    ['nsys', '--version'], text=True).strip(),
                'report_path': str(report), 'report_sha256': sha256(report),
                'sqlite_path': str(sqlite), 'sqlite_sha256': sha256(sqlite),
                'trace': 'cuda', 'cuda_graph_trace': 'node',
                'capture_range': 'cudaProfilerApi around steady decode only',
                'observer_effect': (
                    'Graph node tracing may perturb runtime; values are observed '
                    'kernel times, not zero-overhead ground truth'),
            }
        write_json(output, row)
        if not monitor.pollution:
            if not valid_existing(args, length, batch, arm, source):
                raise RuntimeError(f'{output} 回执校验失败')
            summary = {'done': output.name, 'hygiene': row['gpu_hygiene']}
            if arm == 'asynctls-clean-e2e':
                summary['tpot_ms'] = row['tpot_ms_median']
            else:
                summary.update({
                    key: row['operator_summary'][key]['mean_ms_per_step']
                    for key in ('topk_sort_family_ms', 'gather_family_ms',
                                'index_family_ms',
                                'all_observed_model_kernels_ms')})
            print(json.dumps(summary), flush=True)
            return
        print(f'WARNING {output.name} 检测到其他 GPU 进程，作废重跑', flush=True)
    raise RuntimeError(f'{output.name} 多次受 GPU 污染')


def summarize(args):
    source = _source_receipt()
    records, missing, infeasible = [], [], []
    for length, batch in cells(args):
        memory = async_memory(args, length, batch)
        if not memory['feasible']:
            infeasible.append({'length': length, 'batch': batch, 'memory': memory})
            continue
        rows = {}
        for arm in args.arms:
            if not valid_existing(args, length, batch, arm, source):
                missing.append(str(arm_path(args, length, batch, arm)))
            else:
                rows[arm] = json.loads(
                    arm_path(args, length, batch, arm).read_text(encoding='utf-8'))
        if len(rows) != len(args.arms):
            continue
        profile = rows.get('asynctls-operator-profile')
        clean = rows.get('asynctls-clean-e2e')
        record = {
            'length': length, 'batch': batch, 'execution': 'graph',
            'input': (clean or profile)['input'],
            'actual_prefill_length': (clean or profile)['input']['actual_prompt_tokens'],
        }
        if profile is not None:
            record['operator'] = {
                'summary': profile['operator_summary'],
                'profiled_decode_steps': profile['profiled_decode_steps'],
                'nsys_receipt': profile['nsys_receipt'],
                'path': str(arm_path(
                    args, length, batch, 'asynctls-operator-profile')),
            }
        if clean is not None:
            record['clean_e2e'] = {
                'tpot_ms': clean['tpot_ms_median'],
                'prefill_seconds': clean['prefill_seconds'],
                'path': str(arm_path(args, length, batch, 'asynctls-clean-e2e')),
            }
        if profile is not None and clean is not None:
            profile_tokens = profile['generated_token_ids']
            if profile_tokens != clean['generated_token_ids']:
                raise ValueError(f'AsyncTLS profile/clean token 分歧: L{length} B{batch}')
            record['profile_clean_token_identity_common_prefix'] = True
            record['profile_clean_token_identity_full'] = True
        records.append(record)
    if missing:
        raise RuntimeError('AsyncTLS 矩阵尚缺臂或回执无效:\n' + '\n'.join(missing[:30]))
    result = {
        'schema': 'qwen-asynctls-semantic-graph-performance-v1',
        'validation_status': 'pending-independent-verification',
        'source': source,
        'matrix_driver_sha256': sha256(Path(__file__)),
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'semantics': {
            'implementation': 'independent PyTorch semantic reimplementation',
            'official_tilelang': False,
            'source_verified_official': False,
            'paper_pdf_verified': True,
            'real_async_offload': False,
            'int4': 'per-token quantize-dequantize; BF16 storage remains',
            'budget': args.budget,
            'block_size': args.block_size,
            'top_kb': args.top_kb,
            'channels_per_kv_group': 32,
            'pipeline': 'Stage2 uses previous-step coarse blocks M_(t-1)',
        },
        'protocol': {
            'operator': (
                'separate nsys CUDA Graph node trace process; identical '
                'max_new_tokens/cache geometry and decode-step count as clean'),
            'clean_e2e': 'separate process without nsys or torch.profiler',
            'host_overhead_derivable_by_subtraction': False,
            'timing_metric': 'end-to-end decode TPOT, not pure attention latency',
            'operator_scope': (
                'index-family kernels only; no fabricated stage totals for '
                'ambiguous PyTorch GEMM/softmax/reduction names'),
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
    args.channel_file = args.channel_file.resolve()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('invalid shard')
    for length in args.lengths:
        prepare_extended_input(args, length)
    if args.summarize_only:
        summarize(args)
        return
    source = _source_receipt()
    receipt_path = args.output_dir / f'driver_gpu{args.gpu}_shard{args.shard_index}.json'
    receipt = {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'source': source, 'matrix_driver_sha256': sha256(Path(__file__)),
        'events': [], 'infeasible': [],
    }
    write_json(receipt_path, receipt)
    for index, (length, batch) in enumerate(cells(args)):
        if index % args.num_shards != args.shard_index:
            continue
        memory = async_memory(args, length, batch)
        if not memory['feasible']:
            event = {'length': length, 'batch': batch,
                     'status': 'infeasible-memory', 'memory': memory}
            receipt['infeasible'].append(event)
            write_json(receipt_path, receipt)
            print(json.dumps(event), flush=True)
            continue
        for arm in args.arms:
            run_arm(args, length, batch, arm, source)
            receipt['events'].append({
                'length': length, 'batch': batch, 'arm': arm,
                'output': str(arm_path(args, length, batch, arm)),
            })
            write_json(receipt_path, receipt)
    print(f'driver complete -> {receipt_path}', flush=True)


if __name__ == '__main__':
    main()
