"""按任务隔离 max_gen 后调用正式 RULER scorer；原始 generation 只读。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from benchmark.RULER.yarn_receipt import resolve_generation_pointer
    root = Path(args.root)
    pred_dir = root / 'L32768' / 'pred_strict_pair'
    mapping = {}
    for task in ('niah_multikey_1', 'cwe'):
        pointers = list(pred_dir.glob(f'{task}-*.jsonl.nfa_gen'))
        if len(pointers) != 1:
            raise RuntimeError(f'{task}: 预期一个完成指针，实际 {pointers}')
        pred_out = str(pointers[0])[:-len('.nfa_gen')]
        resolved = resolve_generation_pointer(pred_out)
        isolated = root / 'score_roots' / task / 'L32768' / 'pred_strict_pair'
        isolated.mkdir(parents=True, exist_ok=True)
        dest_gen = isolated / Path(resolved['gen_dir']).name
        shutil.copytree(resolved['gen_dir'], dest_gen, dirs_exist_ok=True)
        dest_pointer = isolated / pointers[0].name
        shutil.copy2(pointers[0], dest_pointer)
        out = root / f'formal_{task}.json'
        log = root / f'formal_{task}.log'
        command = [
            sys.executable, f'{args.repo}/benchmark/RULER/score_ruler_formal.py',
            '--root', str(root / 'score_roots' / task),
            '--pred-postfix', '_strict_pair', '--out', str(out),
            '--manifest-out', str(root / f'formal_{task}_manifest.json'),
            '--expect-tasks', '0', '--min-samples', '5',
            '--data-root', args.data_root, '--model-path', args.model_path,
            '--extra-param', 'tia_level1_topk=128',
            '--extra-param', 'tia_level2_topk=1024',
            '--extra-param', 'nfa_subspace=tail',
            '--extra-param', 'strict_skip_quality=true',
        ]
        env = os.environ.copy()
        env['PYTHONPATH'] = args.repo + ':/tmp/nfa_quality_deps'
        with log.open('w', encoding='utf-8') as handle:
            subprocess.run(command, cwd=args.repo, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads(out.read_text())
        assert all(n == 5 for cell in result['n'].values() for n in cell.values())
        assert sum(len(cell) for cell in result['n'].values()) == 1
        mapping[task] = {
            'source_pointer': str(pointers[0]),
            'source_generation': resolved['gen_dir'],
            'isolated_root': str(root / 'score_roots' / task),
            'formal_score': str(out),
            'scores': result['scores'], 'n': result['n'],
        }
        print(json.dumps({task: mapping[task]}, ensure_ascii=False), flush=True)
    (root / 'formal_task_map.json').write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
