"""Paired task scores and activation/physical storage audit for the screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark.LongBench.prefill_frozen_skip import atomic_json


def paired_ci(values, rng, repeats=5000):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0,len(values),(repeats,len(values)))].mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def analyze(outdir):
    outdir=Path(outdir)
    manifests=[json.loads(p.read_text()) for p in sorted(outdir.glob('manifest_rank*.json'))]
    if not manifests: return None
    identities={m['identity'] for m in manifests}
    if len(identities)!=1: raise ValueError('mixed experiment identities')
    protocol=manifests[0]['protocol'];identity=manifests[0]['identity']
    expected={(ds,i) for ds,ids in protocol['selected_rows'].items() for i in ids}
    records=[];partial=0
    for p in outdir.glob('requests/*/*.json'):
        r=json.loads(p.read_text())
        if r['identity']!=identity: raise ValueError('result identity mismatch')
        if (r['dataset'],r['row_index']) not in expected:raise ValueError('unexpected row')
        if r['status']=='complete':records.append(r)
        else:partial+=1
    arm_names=['full']+[f'{mode}_{eps:g}' for eps in protocol['thresholds'] for mode in ['far','dual']]
    for r in records:
        if set(r['arms'])!=set(arm_names):raise ValueError('completed row has incomplete arms')
    result=dict(identity=identity,status='complete' if len(records)==len(expected) else 'running',
                completed_requests=len(records),expected_requests=len(expected),partial_requests=partial,
                protocol=protocol,arms={},comparisons={})
    rng=np.random.default_rng(42)
    for arm in arm_names:
        scores={ds:float(np.mean([r['arms'][arm]['score'] for r in records if r['dataset']==ds])*100)
                for ds in protocol['datasets'] if any(r['dataset']==ds for r in records)}
        counts={name:sum(r['arms'][arm]['gate_counts'][name] for r in records) for name in ['keep','near','far','both']}
        near_active=sum(any(d['skip_near'] for d in r['arms'][arm]['decisions'].values()) for r in records)
        far_active=sum(any(d['skip_far'] for d in r['arms'][arm]['decisions'].values()) for r in records)
        ratios=[r['arms'][arm]['frozen_kv']['kv_bytes']/r['arms'][arm]['prefill_full_kv']['kv_bytes'] for r in records]
        result['arms'][arm]=dict(per_task=scores,avg5=float(np.mean(list(scores.values()))) if len(scores)==5 else None,
            gate_counts=counts,near_active_requests=near_active,far_active_requests=far_active,
            exact_full_generation=sum(r['arms'][arm]['generated_ids']==r['arms']['full']['generated_ids'] for r in records),
            frozen_kv_ratio_mean=float(np.mean(ratios)) if ratios else None)
    pairs=[(arm,'full') for arm in arm_names if arm!='full']+[(f'dual_{eps:g}',f'far_{eps:g}') for eps in protocol['thresholds']]
    for arm,base in pairs:
        tasks={};groups=[]
        for ds in protocol['datasets']:
            rows=[r for r in records if r['dataset']==ds]
            if not rows:continue
            values=[100*(r['arms'][arm]['score']-r['arms'][base]['score']) for r in rows]
            tasks[ds]=dict(n=len(values),delta_pp=float(np.mean(values)),ci95_pp=paired_ci(values,rng),
                degraded=sum(x<0 for x in values),improved=sum(x>0 for x in values))
            groups.append(np.array(values))
        avg=None
        if len(groups)==5:
            resampled=np.stack([g[rng.integers(0,len(g),(5000,len(g)))].mean(1) for g in groups]).mean(0)
            avg=dict(delta_pp=float(np.mean([g.mean() for g in groups])),ci95_pp=[float(x) for x in np.quantile(resampled,[.025,.975])])
        result['comparisons'][arm+'-vs-'+base]=dict(per_task=tasks,avg5=avg)
    result['scope']='Exploratory paired request bootstrap, 5000 repeats, task-stratified macro AVG5; CI containing zero is not proof of equivalence. Physical eager node deletion, not TLI champion selection or performance claim.'
    atomic_json(outdir/'summary.json',result)
    lines=['# Frozen near/far cache exploration','',f"Status: {result['status']}; paired requests {len(records)}/{len(expected)}.",'',
           '| arm | AVG5 | near active | far active | frozen KV / full |','|---|---:|---:|---:|---:|']
    for arm,a in result['arms'].items():
        avg='pending' if a['avg5'] is None else f"{a['avg5']:.4f}"
        ratio='pending' if a['frozen_kv_ratio_mean'] is None else f"{a['frozen_kv_ratio_mean']:.4f}"
        lines.append(f"| {arm} | {avg} | {a['near_active_requests']} | {a['far_active_requests']} | {ratio} |")
    lines+=['','Paired differences (percentage points):','']
    for pair,c in result['comparisons'].items():
        if c['avg5']:lines.append(f"- {pair}: {c['avg5']['delta_pp']:+.4f}; bootstrap 95% CI {c['avg5']['ci95_pp']}")
    lines+=['',result['scope'],'','Nodes deleted permanently. Near-skip discards SWA overflow forever; it cannot later recover missing KV into far.']
    (outdir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');args=ap.parse_args()
    r=analyze(args.outdir)
    print(json.dumps(None if r is None else {k:r[k] for k in ['status','completed_requests','expected_requests','arms','comparisons']},indent=2))
