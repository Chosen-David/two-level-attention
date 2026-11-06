"""修前/修后 NFA 真实完整 decode 的独立进程 ABBA 配对驱动。"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-repo', required=True)
    parser.add_argument('--candidate-repo', required=True)
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--data-root', default='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER')
    parser.add_argument('--task', default='niah_multikey_1')
    parser.add_argument('--context-length', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--executions', nargs='+', default=['eager', 'graph'])
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def source_receipt(repo):
    head = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', repo, 'diff', '--binary'])
    status = subprocess.check_output(['git', '-C', repo, 'status', '--short'], text=True)
    if status:
        raise RuntimeError(f'源码工作树必须干净: {repo}: {status}')
    return {'repo': repo, 'head': head, 'diff_sha256': hashlib.sha256(diff).hexdigest()}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sys.path.insert(0, args.candidate_repo)
    from benchmark.efficiency.benchmark_nfa_vs_fa2 import GpuMonitor, _preflight
    _preflight(args.gpu_id)
    monitor = GpuMonitor(args.gpu_id, os.path.join(args.output_dir, 'gpu_monitor.jsonl'))
    monitor.start()
    manifest = {
        'config': vars(args),
        'sources': {
            'baseline': source_receipt(args.baseline_repo),
            'candidate': source_receipt(args.candidate_repo),
        },
        'runs': [],
        'status': 'running',
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    try:
        for execution in args.executions:
            order = [('baseline', 'a'), ('candidate', 'a'),
                     ('candidate', 'b'), ('baseline', 'b')]
            for method, leg in order:
                repo = args.baseline_repo if method == 'baseline' else args.candidate_repo
                stem = f'{execution}_{method}_{leg}'
                output = os.path.join(args.output_dir, stem + '.json')
                log_path = os.path.join(args.output_dir, stem + '.log')
                runner = os.path.join(repo, 'benchmark', 'efficiency', 'benchmark_nfa_end_to_end.py')
                command = [
                    sys.executable, runner,
                    '--model-path', args.model_path,
                    '--data-root', args.data_root,
                    '--output', output,
                    '--task', args.task,
                    '--context-length', str(args.context_length),
                    '--batch-size', str(args.batch_size),
                    '--method', 'nfa', '--execution', execution,
                    '--sample-index', str(args.sample_index),
                    '--input-provenance', 'native-ruler',
                    '--max-new-tokens', str(args.max_new_tokens),
                    '--warmup-steps', str(args.warmup_steps),
                    '--measurement-mode', 'latency',
                ]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
                env['PYTHONPATH'] = repo
                print(f'RUN {stem}: {" ".join(command)}', flush=True)
                with open(log_path, 'w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                    monitor.register(process.pid)
                    try:
                        returncode = process.wait()
                    finally:
                        monitor.unregister(process.pid)
                record = {'execution': execution, 'method': method, 'leg': leg,
                          'pid': process.pid, 'returncode': returncode,
                          'json': output, 'log': log_path, 'command': command}
                manifest['runs'].append(record)
                with open(manifest_path, 'w', encoding='utf-8') as handle:
                    json.dump(manifest, handle, indent=2)
                if returncode:
                    raise RuntimeError(f'{stem} 失败，日志: {log_path}')
        if monitor.pollution:
            manifest['status'] = 'polluted-rerun'
            manifest['pollution'] = monitor.pollution
            raise RuntimeError('GPU 旁路监控发现他方进程，整组作废重跑')
        summaries = {}
        for execution in args.executions:
            runs = {}
            for method in ('baseline', 'candidate'):
                runs[method] = []
                for leg in ('a', 'b'):
                    path = os.path.join(args.output_dir, f'{execution}_{method}_{leg}.json')
                    with open(path, encoding='utf-8') as handle:
                        runs[method].append(json.load(handle))
            baseline_ms = [r['steady_tpot_ms_median'] for r in runs['baseline']]
            candidate_ms = [r['steady_tpot_ms_median'] for r in runs['candidate']]
            summaries[execution] = {
                'baseline_tpot_ms': baseline_ms,
                'candidate_tpot_ms': candidate_ms,
                'baseline_median_ms': statistics.median(baseline_ms),
                'candidate_median_ms': statistics.median(candidate_ms),
                'baseline_over_candidate': statistics.median(baseline_ms) / statistics.median(candidate_ms),
                'baseline_repeat_token_identity': runs['baseline'][0]['generated_token_ids'] == runs['baseline'][1]['generated_token_ids'],
                'candidate_repeat_token_identity': runs['candidate'][0]['generated_token_ids'] == runs['candidate'][1]['generated_token_ids'],
                'baseline_candidate_token_identity': runs['baseline'][0]['generated_token_ids'] == runs['candidate'][0]['generated_token_ids'],
                'baseline_skip_policy_counts': runs['baseline'][0]['skip_policy_counts'],
                'candidate_skip_policy_counts': runs['candidate'][0]['skip_policy_counts'],
                'input_sha256': runs['candidate'][0]['input']['sha256'],
                'actual_prompt_tokens': runs['candidate'][0]['input']['actual_prompt_tokens'],
            }
        manifest['summary'] = summaries
        manifest['gpu_hygiene'] = 'clean'
        manifest['status'] = 'complete'
    except Exception as error:
        manifest['error'] = str(error)
        if manifest['status'] != 'polluted-rerun':
            manifest['status'] = 'failed'
        raise
    finally:
        monitor.stop()
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
