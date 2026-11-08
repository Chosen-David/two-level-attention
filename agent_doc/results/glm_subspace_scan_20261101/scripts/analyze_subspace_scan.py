"""Analyze GLM subspace screen/full paired records.

Computes RULER MultiKey2 dense/selected mean, LongBench AVG5 dense/selected,
paired bootstrap 95% CI for selected-dense, and per-layer skip consistency
across tasks for dual/far/near gate runs.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ('dense', 'selected')
RULER_TASK = 'niah_multikey_2'
LB_TASKS = ('qasper', 'hotpotqa', 'musique', 'gov_report', 'repobench-p')


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def percentile(xs, q):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def paired_bootstrap(deltas, n=5000, seed=42):
    rng = random.Random(seed)
    m = len(deltas)
    return [mean(rng.choices(deltas, k=m)) for _ in range(n)]


def read_records(run):
    shared = json.loads((run / 'protocol.json').read_text())
    identity = shared['identity']
    protocol = shared['protocol']
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == identity
    expected = protocol.get('expected', {})
    records = {}
    for task in expected:
        rows = sorted((run / 'requests' / task).glob('*.json'))
        if len(rows) != expected[task]:
            raise RuntimeError(f'{task}: {len(rows)} != {expected[task]}')
        records[task] = []
        for p in rows:
            rec = json.loads(p.read_text())
            if rec['identity'] != identity or rec['status'] != 'complete':
                raise RuntimeError(f'{task}/{p.name}: bad identity/status')
            if set(rec['arms']) != set(ARMS):
                raise RuntimeError(f'{task}/{p.name}: incomplete arms')
            records[task].append(rec)
    return identity, protocol, records


def task_score(records):
    return {task: {arm: mean([r['arms'][arm]['score_pct'] for r in rows])
                   for arm in ARMS}
            for task, rows in records.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    identity, protocol, records = read_records(run)

    scores = task_score(records)
    boot = {}
    for task, rows in records.items():
        deltas = [r['arms']['selected']['score_pct'] - r['arms']['dense']['score_pct'] for r in rows]
        b = paired_bootstrap(deltas)
        boot[task] = dict(n=len(deltas), dense_mean=scores[task]['dense'],
                          selected_mean=scores[task]['selected'],
                          mean_delta=mean(deltas),
                          ci95_low=percentile(b, 0.025), ci95_high=percentile(b, 0.975))

    if all(t in records for t in LB_TASKS):
        avg_dense = mean([scores[t]['dense'] for t in LB_TASKS])
        avg_selected = mean([scores[t]['selected'] for t in LB_TASKS])
        rng = random.Random(42)
        avg_deltas = []
        for _ in range(5000):
            stratified = []
            for t in LB_TASKS:
                deltas = [r['arms']['selected']['score_pct'] - r['arms']['dense']['score_pct']
                          for r in records[t]]
                stratified.append(mean(rng.choices(deltas, k=len(deltas))))
            avg_deltas.append(mean(stratified))
        boot['AVG5'] = dict(n=sum(len(records[t]) for t in LB_TASKS),
                            dense_mean=avg_dense, selected_mean=avg_selected,
                            mean_delta=avg_selected - avg_dense,
                            ci95_low=percentile(avg_deltas, 0.025),
                            ci95_high=percentile(avg_deltas, 0.975))

    skip_summary = None
    if any(r['arms']['selected'].get('tli_audit_summary') for rows in records.values() for r in rows):
        per_task = {}
        n_layers = None
        for task, rows in records.items():
            skip_near = []
            skip_far = []
            states = Counter()
            n = 0
            for r in rows:
                audit = r['arms']['selected'].get('tli_audit_summary') or {}
                sn = audit.get('skip_near')
                sf = audit.get('skip_far')
                if sn is None or sf is None:
                    continue
                if n_layers is None:
                    n_layers = len(sn)
                skip_near.append(sn)
                skip_far.append(sf)
                for s in audit.get('layer_states', []):
                    states[s] += 1
                n += 1
            if not n:
                continue
            per_task[task] = dict(
                n=n,
                skip_near_freq=[sum(v[i] for v in skip_near) / n for i in range(n_layers or 0)],
                skip_far_freq=[sum(v[i] for v in skip_far) / n for i in range(n_layers or 0)],
                state_counts=dict(states),
            )
        pairwise = {}
        tasks = sorted(per_task)
        for i, a in enumerate(tasks):
            for b in tasks[i:]:
                va = per_task[a]['skip_far_freq']
                vb = per_task[b]['skip_far_freq']
                both_zero = not any(va) and not any(vb)
                if both_zero:
                    sim = 1.0
                else:
                    inter = sum(min(x, y) for x, y in zip(va, vb))
                    union = sum(max(x, y) for x, y in zip(va, vb))
                    sim = inter / union if union else 1.0
                pairwise[f'{a}|{b}'] = sim
        skip_summary = dict(per_task=per_task, pairwise_task_jaccard=dict(pairwise))

    out = dict(
        schema='glm-subspace-scan-analysis-v1',
        run_dir=str(run),
        identity=identity,
        source_commit=protocol.get('source_commit'),
        selector=protocol.get('selector'),
        selector_dims=protocol.get('selector_dims'),
        gate=dict(mode=protocol.get('gate_mode'), epsilon=protocol.get('epsilon'),
                  statistic=protocol.get('statistic')),
        screen=protocol.get('screen'),
        scores=scores,
        paired_bootstrap=boot,
        skip_consistency=skip_summary,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
