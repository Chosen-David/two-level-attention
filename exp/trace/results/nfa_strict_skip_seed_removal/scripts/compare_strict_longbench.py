"""比较 strict-skip 修前/修后原生 LongBench 配对质量输出。"""
import argparse
import json
from pathlib import Path


def load_root(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest_rank0.json').read_text())
    records = {}
    for path in sorted((root / 'requests').glob('*/*.json')):
        record = json.loads(path.read_text())
        key = f"{record['dataset']}:{record['row_index']}"
        records[key] = record
    return manifest, records


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    baseline_manifest, baseline = load_root(args.baseline)
    candidate_manifest, candidate = load_root(args.candidate)
    assert baseline.keys() == candidate.keys()
    assert len(baseline) == 5
    bp = baseline_manifest['protocol']
    cp = candidate_manifest['protocol']
    for field in ('selected_rows', 'dataset_sha256', 'model_config_sha256',
                  'tokenizer_config_sha256', 'seed', 'samples_per_task', 'max_new_tokens'):
        assert bp[field] == cp[field], field
    rows = {}
    totals = {}
    for arm in ('full', 'far_0.05', 'dual_0.05'):
        totals[arm] = {'baseline': 0.0, 'candidate': 0.0}
    for key in sorted(baseline):
        b = baseline[key]
        c = candidate[key]
        assert b['status'] == c['status'] == 'complete'
        assert b['input']['input_sha256'] == c['input']['input_sha256']
        assert b['answers'] == c['answers']
        assert b['arms'].keys() == c['arms'].keys()
        row = {'input_sha256': b['input']['input_sha256'], 'arms': {}}
        for arm in ('full', 'far_0.05', 'dual_0.05'):
            ba = b['arms'][arm]
            ca = c['arms'][arm]
            assert ba['gate_counts'] == ca['gate_counts']
            assert ba['generated_ids'][0] == b['arms']['full']['generated_ids'][0]
            assert ca['generated_ids'][0] == c['arms']['full']['generated_ids'][0]
            bt = ba['generated_ids']
            ct = ca['generated_ids']
            row['arms'][arm] = {
                'baseline_score': ba['score'],
                'candidate_score': ca['score'],
                'score_delta': ca['score'] - ba['score'],
                'gate_counts': ca['gate_counts'],
                'token_identity': bt == ct,
                'prediction_identity': ba['prediction'] == ca['prediction'],
                'first_divergence': first_divergence(bt, ct),
                'baseline_generated_ids': bt,
                'candidate_generated_ids': ct,
            }
            totals[arm]['baseline'] += ba['score']
            totals[arm]['candidate'] += ca['score']
        assert row['arms']['full']['token_identity'], f'full arm changed: {key}'
        rows[key] = row
    count = len(rows)
    summary = {
        arm: {
            'baseline_mean': values['baseline'] / count,
            'candidate_mean': values['candidate'] / count,
            'delta': (values['candidate'] - values['baseline']) / count,
            'changed_token_sequences': sum(
                not row['arms'][arm]['token_identity'] for row in rows.values()),
            'changed_predictions': sum(
                not row['arms'][arm]['prediction_identity'] for row in rows.values()),
        }
        for arm, values in totals.items()
    }
    result = {
        'valid': True,
        'samples': count,
        'baseline_head': bp['git_head'],
        'candidate_head': cp['git_head'],
        'summary': summary,
        'rows': rows,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'valid': True, 'samples': count, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
