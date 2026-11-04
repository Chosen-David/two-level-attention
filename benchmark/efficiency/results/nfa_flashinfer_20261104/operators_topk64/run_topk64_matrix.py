"""在单张干净 GPU 上执行 NFA/FlashInfer 纯 kernel ABBA 矩阵。"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

PYTHON = '/home/wangyuanshuo02/two-level-attention/.venv/bin/python'
REPO = '/tmp/nfa-topk64-opt'
MODEL = '/home/wangyuanshuo02/Qwen3-8B'
DATA = '/tmp/nfa_kernel_inputs'
MEASURE = '/tmp/nfa_kernel_audit/measure_nfa_kernels.py'


def gpu_state(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True)
    for row in csv.reader(output.splitlines()):
        if int(row[0].strip()) == gpu:
            uuid = row[1].strip()
            used = int(row[2].strip().split()[0])
            util = int(row[3].strip().split()[0])
            break
    else:
        raise RuntimeError(f'GPU {gpu} not found')
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader'], text=True)
    rows = [line for line in apps.splitlines()
            if line.strip() and line.split(',')[0].strip() == uuid]
    return {'used_mib': used, 'utilization_percent': util,
            'compute_process_rows': rows}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--gate-epsilon', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    # 128K SDPA prefill 需要额外大块临时显存；扩展段避免 reserved 碎片阻塞分配。
    base_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    base_env['NFA_REPO'] = REPO
    python_paths = ['/tmp/nfa-fi-no-deps', REPO]
    if base_env.get('PYTHONPATH'):
        python_paths.append(base_env['PYTHONPATH'])
    base_env['PYTHONPATH'] = os.pathsep.join(python_paths)
    state = gpu_state(args.gpu)
    if state['used_mib'] >= 500 or state['compute_process_rows']:
        raise RuntimeError(f'GPU {args.gpu} is not clean: {state}')
    receipt = {'gpu': args.gpu, 'task': args.task,
               'sample_index': args.sample_index,
               'gate_epsilon': args.gate_epsilon,
               'baseline': state, 'arms': []}
    receipt_path = output_dir / f'{args.task}_s{args.sample_index}_driver.json'
    with receipt_path.open('w', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)

    for length in args.lengths:
        for execution in ('graph', 'eager'):
            sequence = (('nfa', 'a1'), ('flashinfer', 'b1'),
                        ('flashinfer', 'b2'), ('nfa', 'a2'))
            for method, abba in sequence:
                stem = f'{args.task}_s{args.sample_index}_{length}_{execution}_{abba}'
                output = output_dir / f'{stem}.json'
                log = output_dir / f'{stem}.log'
                command = [PYTHON, '-u', MEASURE,
                    '--method', method, '--execution', execution,
                    '--model-path', MODEL, '--data-root', DATA,
                    '--task', args.task, '--context-length', str(length),
                    '--sample-index', str(args.sample_index),
                    '--gate-epsilon', str(args.gate_epsilon),
                    '--input-provenance',
                    'repeated-needle-body-extension' if length == 98304 else 'native-ruler',
                    '--warmup-steps', '3', '--repeat', '5',
                    '--output', str(output)]
                for attempt in range(1, args.max_attempts + 1):
                    before = gpu_state(args.gpu)
                    if before['used_mib'] >= 500 or before['compute_process_rows']:
                        raise RuntimeError(f'GPU became dirty before {stem}: {before}')
                    env = base_env.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                    samples = []
                    stop = threading.Event()
                    with log.open('w', encoding='utf-8') as log_handle:
                        process = subprocess.Popen(
                            command, env=env, stdout=log_handle,
                            stderr=subprocess.STDOUT)
                        def monitor():
                            while not stop.wait(30.0):
                                if process.poll() is not None:
                                    return
                                samples.append(gpu_state(args.gpu))
                        watcher = threading.Thread(target=monitor)
                        watcher.start()
                        returncode = process.wait()
                        stop.set()
                        watcher.join()
                    after = gpu_state(args.gpu)
                    polluted = (any(len(row['compute_process_rows']) > 1
                                    for row in samples)
                                or bool(after['compute_process_rows']))
                    arm = {'length': length, 'execution': execution,
                           'method': method, 'abba': abba, 'attempt': attempt,
                           'returncode': returncode, 'before': before,
                           'samples': samples, 'after': after,
                           'gpu_hygiene': 'polluted-rerun' if polluted else 'clean',
                           'output': str(output), 'log': str(log)}
                    receipt['arms'].append(arm)
                    with receipt_path.open('w', encoding='utf-8') as handle:
                        json.dump(receipt, handle, indent=2)
                    if returncode != 0:
                        raise RuntimeError(f'{stem} failed; see {log}')
                    result = json.loads(output.read_text(encoding='utf-8'))
                    result['gpu_hygiene'] = arm['gpu_hygiene']
                    result['abba'] = abba
                    result['attempt'] = attempt
                    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                    if not polluted:
                        break
                else:
                    raise RuntimeError(
                        f'{stem} remained polluted after {args.max_attempts} attempts')
                print(json.dumps({'done': stem, 'hygiene': arm['gpu_hygiene']}), flush=True)


if __name__ == '__main__':
    main()
