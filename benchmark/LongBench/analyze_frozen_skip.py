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
    independent = protocol.get('schema')=='frozen-near-far-screen-v3-independent-gates'
    staging_policy = independent or protocol.get('schema')=='frozen-near-far-screen-v2-near-staging'
    expected={(ds,i) for ds,ids in protocol['selected_rows'].items() for i in ids}
    records=[];partial=0;seen=set()
    for p in outdir.glob('requests/*/*.json'):
        r=json.loads(p.read_text())
        if r['identity']!=identity: raise ValueError('result identity mismatch')
        if (r['dataset'],r['row_index']) not in expected:raise ValueError('unexpected row')
        key=(r['dataset'],r['row_index'])
        if key in seen:raise ValueError('duplicate request identity')
        seen.add(key)
        if r['status'] not in ('complete','running'):raise ValueError('unknown request status')
        if r['status']=='complete':records.append(r)
        else:partial+=1
    task_order={ds:i for i,ds in enumerate(protocol['datasets'])}
    records.sort(key=lambda r:(task_order[r['dataset']],r['row_index']))
    arm_names=['full']+[f'{mode}_{eps:g}' for eps in protocol['thresholds'] for mode in ['far','dual']]
    for r in records:
        if set(r['arms'])!=set(arm_names):raise ValueError('completed row has incomplete arms')
    result=dict(identity=identity,status='complete' if len(records)==len(expected) else 'running',
                completed_requests=len(records),expected_requests=len(expected),partial_requests=partial,
                protocol=protocol,arms={},comparisons={})
    result['completed_per_task']={ds:sum(r['dataset']==ds for r in records) for ds in protocol['datasets']}
    if (outdir/'STOPPED.json').exists(): result['status']='superseded'
    rng=np.random.default_rng(42)
    for arm in arm_names:
        scores={ds:float(np.mean([r['arms'][arm]['score'] for r in records if r['dataset']==ds])*100)
                for ds in protocol['datasets'] if any(r['dataset']==ds for r in records)}
        counts={name:sum(r['arms'][arm]['gate_counts'][name] for r in records) for name in ['keep','near','far','both']}
        near_active=sum(any(d['skip_near'] for d in r['arms'][arm]['decisions'].values()) for r in records)
        far_active=sum(any(d['skip_far'] for d in r['arms'][arm]['decisions'].values()) for r in records)
        task_activation={}
        for ds in protocol['datasets']:
            group=[r for r in records if r['dataset']==ds]
            task_activation[ds]=dict(n=len(group),
                near_active=sum(any(d['skip_near'] for d in r['arms'][arm]['decisions'].values()) for r in group),
                far_active=sum(any(d['skip_far'] for d in r['arms'][arm]['decisions'].values()) for r in group),
                exact_full_generation=sum(r['arms'][arm]['generated_ids']==r['arms']['full']['generated_ids'] for r in group))
        ratios=[r['arms'][arm]['frozen_kv']['kv_bytes']/r['arms'][arm]['prefill_full_kv']['kv_bytes'] for r in records]
        work_ratios=[]
        near_only_kv_ratios=[]
        near_only_work_ratios=[]
        far_choices_changed=0
        for r in records:
            frozen=r['arms'][arm]['frozen_kv'].get('layers',[])
            full=r['arms'][arm]['prefill_full_kv'].get('layers',[])
            if frozen and full and all('attention_tokens' in layer for layer in frozen+full):
                work_ratios.append(sum(layer['attention_tokens'] for layer in frozen)/sum(layer['attention_tokens'] for layer in full))
                for li,d in r['arms'][arm]['decisions'].items():
                    if d['skip_near'] and not d['skip_far']:
                        before,after=full[int(li)],frozen[int(li)]
                        near_only_kv_ratios.append(sum(n['kv_bytes'] for n in after['nodes'].values())/sum(n['kv_bytes'] for n in before['nodes'].values()))
                        near_only_work_ratios.append(after['attention_tokens']/before['attention_tokens'])
            if arm.startswith('dual_'):
                counterpart='far_'+arm.split('_',1)[1]
                far_choices_changed+=sum(d['skip_far']!=r['arms'][counterpart]['decisions'][li]['skip_far'] for li,d in r['arms'][arm]['decisions'].items())
        result['arms'][arm]=dict(per_task=scores,per_task_activation=task_activation,avg5=float(np.mean(list(scores.values()))) if len(scores)==5 else None,
            gate_counts=counts,near_active_requests=near_active,far_active_requests=far_active,
            exact_full_generation=sum(r['arms'][arm]['generated_ids']==r['arms']['full']['generated_ids'] for r in records),
            frozen_kv_ratio_mean=float(np.mean(ratios)) if ratios else None,
            frozen_attention_token_ratio_mean=float(np.mean(work_ratios)) if work_ratios else None,
            near_only_layer_count=len(near_only_kv_ratios),
            near_only_layer_kv_ratio_mean=float(np.mean(near_only_kv_ratios)) if near_only_kv_ratios else None,
            near_only_layer_attention_ratio_mean=float(np.mean(near_only_work_ratios)) if near_only_work_ratios else None,
            far_decisions_differ_from_far_only=far_choices_changed)
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
    policy = 'Near-only retains staging KV and skips attention work only; far-only/both physically evict nodes.' if staging_policy else 'Superseded v1: near-only permanently evicted KV and did not migrate it into far.'
    result['scope']='Exploratory paired request bootstrap, 5000 repeats, task-stratified macro AVG5; CI containing zero is not proof of equivalence. '+policy+(' TLI tail32 L1/L2 reference adapter; full denotes gates off, not dense attention. No performance claim.' if protocol.get('attention','').startswith('tli') else ' No TLI champion selection or performance claim.')
    atomic_json(outdir/'summary.json',result)
    lines=['# Frozen near/far cache exploration','',f"Status: {result['status']}; paired requests {len(records)}/{len(expected)}.",'',
           '| arm | AVG5 | near-active requests | far-active requests | frozen KV / full-node KV | attention tokens / full |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,a in result['arms'].items():
        avg='pending' if a['avg5'] is None else f"{a['avg5']:.4f}"
        ratio='pending' if a['frozen_kv_ratio_mean'] is None else f"{a['frozen_kv_ratio_mean']:.4f}"
        work='pending' if a['frozen_attention_token_ratio_mean'] is None else f"{a['frozen_attention_token_ratio_mean']:.4f}"
        lines.append(f"| {arm} | {avg} | {a['near_active_requests']} | {a['far_active_requests']} | {ratio} | {work} |")
    lines+=['','Per-task scores (0–100):','',
            '| arm | '+' | '.join(protocol['datasets'])+' |',
            '|---|'+'---:|'*len(protocol['datasets'])]
    for arm,a in result['arms'].items():
        lines.append('| '+arm+' | '+' | '.join(f"{a['per_task'][ds]:.4f}" if ds in a['per_task'] else 'pending' for ds in protocol['datasets'])+' |')
    lines+=['','Paired differences (percentage points):','']
    for pair,c in result['comparisons'].items():
        if c['avg5']:lines.append(f"- {pair}: {c['avg5']['delta_pp']:+.4f}; bootstrap 95% CI {c['avg5']['ci95_pp']}")
    lines+=['', 'Attention baseline: '+protocol.get('attention','native SDPA full retained KV')+'. Arm full means both region gates off.', '',result['scope'],'',
            'KV ratios measure allocated cache-node capacity, relative to the same full-node cache; they exclude model weights, gather temporaries and allocator behavior. Attention-token ratios are a pre-decode work proxy, not measured speedups.',
            ('Near and Far gates compare their own mass independently; adding Near gating does not change Far decisions.' if independent else 'Historical dual gates share a combined calibrated mass budget; adding Near gating may replace Far decisions.'),
            'Near-active and far-active request counts include both-skip states.']
    (outdir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');args=ap.parse_args()
    r=analyze(args.outdir)
    print(json.dumps(None if r is None else {k:r[k] for k in ['status','completed_requests','expected_requests','arms','comparisons']},indent=2))
