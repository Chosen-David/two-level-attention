"""运行或验收 strict-skip 真实 RULER correctness 烟测，并记录 GPU hygiene。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = '/tmp/nfa-strict-skip-20261106'
HEAD = '2f702c222186bdf17de996e981c52af467957ee5'


def receipt(path):
    result = json.loads(Path(path).read_text())
    strict = result.get('strict_skip_first_step', {})
    return {
        'output': str(path),
        'strict_valid': strict.get('valid'),
        'violations': strict.get('violations'),
        'generated_token_ids': result.get('generated_token_ids'),
        'decode_steps': result.get('decode_steps'),
        'gpu_hygiene': result.get('gpu_hygiene'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', choices=('eager', 'graph'), required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    args = parser.parse_args()
    output = Path(f'/tmp/nfa-strict-results/nfa32k_{args.execution}_smoke.json')
    if output.exists():
        print(json.dumps({'reused': True, **receipt(output)}, ensure_ascii=False))
        return
    actual_head = subprocess.check_output(
        ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', REPO, 'status', '--short'], text=True).strip()
    if actual_head != HEAD or dirty:
        raise RuntimeError(f'源码身份不符: HEAD={actual_head}, dirty={dirty!r}')
    sys.path.insert(0, REPO)
    from benchmark.efficiency.benchmark_nfa_vs_fa2 import GpuMonitor, _preflight
    _preflight(args.gpu_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = GpuMonitor(args.gpu_id, str(output) + '.gpu_monitor.jsonl')
    monitor.start()
    command = [
        sys.executable, f'{REPO}/benchmark/efficiency/benchmark_nfa_end_to_end.py',
        '--model-path', '/home/wangyuanshuo02/Qwen3-8B',
        '--data-root', '/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER',
        '--output', str(output), '--task', 'niah_multikey_1',
        '--context-length', '32768', '--batch-size', '1',
        '--method', 'nfa', '--execution', args.execution,
        '--sample-index', '0', '--input-provenance', 'native-ruler',
        '--max-new-tokens', '32', '--warmup-steps', '3',
        '--measurement-mode', 'correctness',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    env['PYTHONPATH'] = REPO
    log_path = str(output) + '.log'
    post_error = None
    with open(log_path, 'w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        monitor.register(process.pid)
        try:
            returncode = process.wait()
        finally:
            monitor.unregister(process.pid)
    try:
        _preflight(args.gpu_id)
    except Exception as error:
        post_error = str(error)
    monitor.stop()
    hygiene = 'polluted-rerun' if monitor.pollution or post_error else 'clean'
    if output.exists():
        result = json.loads(output.read_text())
        result['gpu_hygiene'] = hygiene
        result['gpu_monitor'] = str(output) + '.gpu_monitor.jsonl'
        result['postflight_error'] = post_error
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    Path(str(output) + '.hygiene.json').write_text(json.dumps({
        'status': hygiene, 'pollution': monitor.pollution,
        'postflight_error': post_error, 'returncode': returncode,
        'command': command, 'log': log_path,
    }, ensure_ascii=False, indent=2) + '\n')
    if returncode or hygiene != 'clean':
        raise RuntimeError(f'烟测未采用: returncode={returncode}, hygiene={hygiene}, log={log_path}')
    result_receipt = receipt(output)
    if result_receipt['strict_valid'] is not True:
        raise RuntimeError(f'strict 回执失败: {result_receipt}')
    print(json.dumps({'reused': False, **result_receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
