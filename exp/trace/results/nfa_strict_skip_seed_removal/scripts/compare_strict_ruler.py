"""比较 strict-skip 修前/修后原生 RULER 正式评分与完整生成序列。"""
import argparse
import json
from pathlib import Path
import sys


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def gate_counts(row):
    decisions = row.get('_frozen_region', {}).get('decisions', {})
    counts = {'keep': 0, 'far': 0, 'near': 0, 'both': 0}
    for decision in decisions.values():
        far = bool(decision.get('skip_far', False))
        near = bool(decision.get('skip_near', False))
        counts['both' if far and near else 'far' if far else 'near' if near else 'keep'] += 1
    return counts


def load_root(root, repo):
    sys.path.insert(0, repo)
    from benchmark.RULER.yarn_receipt import resolve_generation_pointer
    root = Path(root)
    result = {'tasks': {}}
    pred_dir = root / 'L32768' / 'pred_strict_pair'
    for task in ('niah_multikey_1', 'cwe'):
        formal = json.loads((root / f'formal_{task}.json').read_text())
        formal_receipt = json.loads((root / f'formal_{task}.json.receipt.json').read_text())
        pointers = list(pred_dir.glob(f'{task}-*.jsonl.nfa_gen'))
        assert len(pointers) == 1, (task, pointers)
        pred_out = str(pointers[0])[:-len('.nfa_gen')]
        resolved = resolve_generation_pointer(pred_out)
        assert resolved is not None
        rows = [json.loads(line) for line in Path(resolved['pred_path']).read_text().splitlines()]
        receipt = json.loads(Path(resolved['rcp_path']).read_text())
        assert len(rows) == receipt['prediction_lines'] == 5
        result['tasks'][task] = {'rows': rows, 'receipt': receipt,
                                 'pred_path': resolved['pred_path'],
                                 'formal': formal, 'formal_receipt': formal_receipt}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--candidate-repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline = load_root(args.baseline, args.candidate_repo)
    candidate = load_root(args.candidate, args.candidate_repo)
    summary = {}
    rows_out = {}
    for task in ('niah_multikey_1', 'cwe'):
        b_formal = baseline['tasks'][task]['formal']
        c_formal = candidate['tasks'][task]['formal']
        assert b_formal['scores'].keys() == c_formal['scores'].keys()
        assert b_formal['n'] == c_formal['n']
        assert b_formal['n'].keys() == b_formal['scores'].keys()
        assert len(b_formal['scores']) == 1
        key = next(iter(b_formal['scores']))
        assert list(b_formal['scores'][key]) == [task]
        b_score = b_formal['scores'][key][task]
        c_score = c_formal['scores'][key][task]
        n = b_formal['n'][key][task]
        assert n == 5
        b_rows = {row['_id']: row for row in baseline['tasks'][task]['rows']}
        c_rows = {row['_id']: row for row in candidate['tasks'][task]['rows']}
        assert b_rows.keys() == c_rows.keys()
        assert len(b_rows) == 5
        task_rows = {}
        changed = 0
        changed_predictions = 0
        skip_rows = 0
        for row_id in sorted(b_rows):
            b = b_rows[row_id]
            c = c_rows[row_id]
            assert b['answers'] == c['answers']
            assert b['_answers_sha'] == c['_answers_sha']
            assert b['length'] == c['length']
            bt = b['generated_token_ids']
            ct = c['generated_token_ids']
            assert bt[0] == ct[0], f'dense prefill first token changed: {task}/{row_id}'
            b_gates = gate_counts(b)
            c_gates = gate_counts(c)
            assert b_gates == c_gates
            no_skip = b_gates['keep'] == 36
            identical = bt == ct
            if no_skip:
                assert identical, f'no-skip row changed: {task}/{row_id}'
            changed += not identical
            changed_predictions += b['pred'] != c['pred']
            skip_rows += not no_skip
            task_rows[row_id] = {
                'answers_sha': b['_answers_sha'],
                'gate_counts': b_gates,
                'token_identity': identical,
                'prediction_identity': b['pred'] == c['pred'],
                'first_divergence': first_divergence(bt, ct),
                'baseline_generated_token_ids': bt,
                'candidate_generated_token_ids': ct,
                'baseline_prediction': b['pred'],
                'candidate_prediction': c['pred'],
            }
        rows_out[task] = task_rows
        summary[task] = {
            'score_key': key, 'n': n,
            'baseline_score': b_score, 'candidate_score': c_score,
            'delta': c_score - b_score,
            'changed_token_sequences': changed,
            'changed_predictions': changed_predictions,
            'natural_skip_rows': skip_rows,
        }
    result = {
        'valid': True,
        'summary': summary,
        'rows': rows_out,
        'baseline_formal_receipts': {
            task: baseline['tasks'][task]['formal_receipt'] for task in baseline['tasks']},
        'candidate_formal_receipts': {
            task: candidate['tasks'][task]['formal_receipt'] for task in candidate['tasks']},
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'valid': True, 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
