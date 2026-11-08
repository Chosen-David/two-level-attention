"""Local recomputation for aligned glm_nope_last32 full-task raw records."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ARMS = ('dense', 'selected')
TASKS = ('niah_multikey_2', 'qasper', 'hotpotqa', 'musique', 'gov_report', 'repobench-p')
LB_TASKS = TASKS[1:]


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def percentile(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def paired_bootstrap(deltas, n=5000, seed=42):
    rng = random.Random(seed)
    m = len(deltas)
    return [mean(rng.choices(deltas, k=m)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    shared = json.loads((run / 'protocol.json').read_text())
    identity = shared['identity']
    protocol = shared['protocol']
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == identity
    assert set(protocol['arms']) == set(ARMS)
    assert protocol.get('source_commit') == '42821f81a4739d05d8bc81e1d4cc58307e34bb7e'
    expected = protocol['expected']
    for t in TASKS:
        assert expected.get(t, 0) == {'niah_multikey_2': 100, 'qasper': 200, 'hotpotqa': 200,
                                      'musique': 200, 'gov_report': 200, 'repobench-p': 500}[t]

    records = {}
    total_files = 0
    for task in TASKS:
        rows = sorted((run / 'requests' / task).glob('*.json'))
        if len(rows) != expected[task]:
            raise RuntimeError(f'{task}: found {len(rows)}, expected {expected[task]}')
        rows = [json.loads(p.read_text()) for p in rows]
        for rec in rows:
            total_files += 1
            if rec.get('identity') != identity:
                raise RuntimeError(f'{task}/{rec.get("row")}: identity mismatch')
            if rec.get('status') != 'complete':
                raise RuntimeError(f'{task}/{rec.get("row")}: status {rec.get("status")}')
            if set(rec.get('arms', {})) != set(ARMS):
                raise RuntimeError(f'{task}/{rec.get("row")}: arms {set(rec.get("arms", {}))}')
            for arm, value in rec['arms'].items():
                if not math.isfinite(float(value.get('score_pct', float('nan')))):
                    raise RuntimeError(f'{task}/{rec.get("row")}/{arm}: non-finite score')
                if not value.get('generated_ids'):
                    raise RuntimeError(f'{task}/{rec.get("row")}/{arm}: empty generation')
        records[task] = rows

    task_scores = {}
    task_pairs = {}
    for task in TASKS:
        pairs = [(rec['arms']['dense']['score_pct'], rec['arms']['selected']['score_pct']) for rec in records[task]]
        task_pairs[task] = pairs
        task_scores[task] = {arm: mean([p[i] for p in pairs]) for i, arm in enumerate(ARMS)}

    boot = {}
    for task in TASKS:
        deltas = [s - d for d, s in task_pairs[task]]
        b = paired_bootstrap(deltas)
        boot[task] = {
            'n': len(deltas),
            'mean_delta': mean(deltas),
            'dense_mean': task_scores[task]['dense'],
            'selected_mean': task_scores[task]['selected'],
            'ci95_low': percentile(b, 0.025),
            'ci95_high': percentile(b, 0.975),
        }

    avg_dense = mean([task_scores[t]['dense'] for t in LB_TASKS])
    avg_selected = mean([task_scores[t]['selected'] for t in LB_TASKS])
    avg_deltas = []
    rng = random.Random(42)
    for _ in range(5000):
        stratified = []
        for t in LB_TASKS:
            deltas = [s - d for d, s in task_pairs[t]]
            stratified.append(mean(rng.choices(deltas, k=len(deltas))))
        avg_deltas.append(mean(stratified))
    boot['AVG5'] = {
        'n': sum(len(task_pairs[t]) for t in LB_TASKS),
        'mean_delta': avg_selected - avg_dense,
        'dense_mean': avg_dense,
        'selected_mean': avg_selected,
        'ci95_low': percentile(avg_deltas, 0.025),
        'ci95_high': percentile(avg_deltas, 0.975),
    }

    summary = {
        'schema': 'glm-nope-last32-full-aligned-recompute-v1',
        'run_dir': str(run),
        'identity': identity,
        'source_commit': protocol.get('source_commit'),
        'total_request_files': total_files,
        'expected_total_files': sum(expected.values()),
        'arms': list(ARMS),
        'tasks': TASKS,
        'task_scores': task_scores,
        'paired_bootstrap': boot,
        'scope': 'Raw complete-record recomputation only; not independent review or benchmark acceptance',
    }
    out = Path(args.out or run / 'summary_recompute.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
