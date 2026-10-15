"""CPU LongBench paired diagnostic. Never certifies producer identity or a winner.

Frozen plan supplies independently recovered document/input identities. Official
run-cap gates remain mandatory. E119 RULER results are not accepted by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exp.figures.figure_data_contract import need, label, snapshot, write_new, build_rows, tsv_bytes

IDENTITY = ('model_revision', 'tokenizer_revision', 'data_revision', 'split',
            'scorer_sha256', 'scorer_backend', 'phase', 'dtype', 'rope',
            'generation', 'protection', 'producer_commit', 'producer_files_sha256')


def record_status(row):
    """Classify recorded failures without copying potentially private error text."""
    if row is None:
        return 'missing'
    if row.get('error'):
        return 'error-present'
    if row.get('failed'):
        return 'failed-flag'
    return row.get('status', 'ok')


def paired_stats(tasks, draws=10000, seed=20261014):
    """Task-stratified document cluster bootstrap, sample-weighted within task.

    Each document is drawn as a whole; ratio of sampled score sums/counts keeps
    original sample-weighted task estimand even with unequal cluster sizes.
    """
    need(type(draws) is int and 100 <= draws <= 100000, 'invalid bootstrap count')
    need(type(seed) is int and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    reports, distributions = {}, []
    for task, groups in sorted(tasks.items()):
        need(len(groups) >= 2, 'CI requires two documents per task')
        arrays = [np.asarray(groups[k], dtype=float) for k in sorted(groups)]
        need(all(a.ndim == 1 and len(a) and np.isfinite(a).all() for a in arrays), 'invalid differences')
        sums = np.array([a.sum() for a in arrays]); sizes = np.array([len(a) for a in arrays])
        dist = np.empty(draws)
        # Bounded memory: no B x N allocation.
        for b in range(draws):
            idx = rng.integers(0, len(arrays), len(arrays))
            dist[b] = sums[idx].sum() / sizes[idx].sum()
        distributions.append(dist)
        reports[task] = dict(estimate=float(sums.sum()/sizes.sum()),
                            ci_low=float(np.quantile(dist, .025)), ci_high=float(np.quantile(dist, .975)),
                            n_independent_seq=len(groups), n_valid=int(sizes.sum()))
    need(bool(reports), 'no tasks')
    dist = np.mean(distributions, axis=0)
    overall = dict(estimate=float(np.mean([r['estimate'] for r in reports.values()])),
                   ci_low=float(np.quantile(dist,.025)), ci_high=float(np.quantile(dist,.975)))
    return reports, overall


def validate_plan(p):
    need(p['schema'] == 'paired-longbench-v1' and p['status'] == 'frozen', 'unfrozen plan')
    need(type(p['seed']) is int and p['seed'] >= 0, 'seed must be a nonnegative integer')
    label(p['run_id'], 'run_id')
    need(p['run_mode'] == 'natural', 'only natural generation supported')
    need(set(p['arms']) == {'sub_sub', 'dense'}, 'need sub_sub and separate dense')
    need(p['arms']['dense']['selection'] == 'dense_fullkv', 'full/full is not dense')
    need(p['arms']['sub_sub']['selection'] == 'sparse_sub_sub', 'not sub/sub')
    for field in IDENTITY:
        a = p['arms']['dense']['identity'][field]; b = p['arms']['sub_sub']['identity'][field]
        need(a == b, 'identity mismatch: '+field)
        label(a, field)
    need(isinstance(p['tasks'], dict) and p['tasks'], 'missing planned tasks')
    # Threshold provenance is preserved, not accepted on a self-declared timestamp.
    if p.get('noninferiority_delta_pp') is not None:
        deltas = p['noninferiority_delta_pp']
        need(set(deltas) == set(p['tasks']), 'threshold task coverage differs')
        need(all(type(v) in (int,float) and np.isfinite(v) and v >= 0 for v in deltas.values()), 'invalid thresholds')
        need(isinstance(p.get('preregistration'),dict), 'missing preregistration artifact')


def analyze(p, base):
    validate_plan(p)
    if p.get('noninferiority_delta_pp') is not None:
        reg_sha = p['preregistration']['sha256']
        need(isinstance(reg_sha,str) and len(reg_sha)==64 and all(c in '0123456789abcdef' for c in reg_sha), 'invalid preregistration hash')
        registration,_ = snapshot(base / p['preregistration']['path'], reg_sha)
        need(registration['noninferiority_delta_pp'] == p['noninferiority_delta_pp'], 'threshold registration mismatch')
        label(registration['frozen_before_run'], 'freeze evidence reference')
    from benchmark.LongBench.eval import sample_score, validate_runcap_sidecar, SCORER_BACKEND_ID, dataset2metric
    from benchmark.LongBench.run_cap_manifest import runcap_gate_lock
    scorer_hash = hashlib.sha256(b''.join((ROOT / f).read_bytes() for f in
        ('benchmark/LongBench/eval.py','benchmark/LongBench/metrics.py','benchmark/LongBench/lbv2_choice.py'))).hexdigest()
    identity = p['arms']['dense']['identity']
    need(identity['scorer_sha256'] == scorer_hash and identity['scorer_backend'] == SCORER_BACKEND_ID, 'scorer identity mismatch')
    audit, groups, sources = {}, {}, {}
    absolute = {}
    document_tasks = {}
    for task, spec in sorted(p['tasks'].items()):
        label(task, 'task')
        need(task in dataset2metric, 'unknown official task')
        planned = spec['samples']; need(bool(planned), 'empty planned task')
        seen_inputs = {}; rows = {}; caps = {}; manifests = {}
        for sid, item in planned.items():
            label(sid, 'sample_id'); label(item['document_id'], 'document_id')
            doc = item['document_id']
            need(doc not in document_tasks or document_tasks[doc] == task, 'cross-task document dependence unsupported')
            document_tasks[doc] = task
            for h in ('input_ids_sha256', 'record_sha256_dense', 'record_sha256_sub_sub'):
                val = item[h]; need(isinstance(val,str) and len(val)==64 and all(c in '0123456789abcdef' for c in val), 'invalid '+h)
            inp = item['input_ids_sha256']
            need(inp not in seen_inputs or seen_inputs[inp] == item['document_id'], 'input relabeled as independent documents')
            seen_inputs[inp] = item['document_id']
        for arm in ('dense','sub_sub'):
            src = spec[arm]; path = base / src['path']
            for field in ('sha256', 'runcap_sha256'):
                digest = src[field]
                need(isinstance(digest, str) and len(digest) == 64
                     and all(c in '0123456789abcdef' for c in digest),
                     'invalid source ' + field)
            with runcap_gate_lock(str(path)):
                raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
                need(sha == src['sha256'], 'prediction hash mismatch')
                records = [json.loads(line) for line in raw.splitlines()]
                caps[arm] = validate_runcap_sidecar(str(path), False, sha, len(records))
                manifest, _ = snapshot(Path(str(path)+'.runcap.json'), src['runcap_sha256'])
                need(manifest['provenance']=='inline', 'legacy/incomplete producer forbidden')
                need(manifest['method'] == ('none' if arm=='dense' else 'tli'), 'producer method differs')
                need(manifest['dataset']==task, 'producer task differs')
                manifests[arm] = manifest
            rows[arm] = {}
            for row in records:
                sid = row['_id']; need(sid in planned and sid not in rows[arm], 'unknown/duplicate sample ID')
                canonical = json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
                need(hashlib.sha256(canonical).hexdigest()==planned[sid]['record_sha256_'+arm], 'record binding mismatch')
                need(isinstance(row.get('answers'), list) and bool(row['answers']),
                     'missing or empty references')
                rows[arm][sid] = row
            sources[task+':'+arm] = sha
        for field in ('model_name','model_path','dataset_fingerprint','max_new_tokens','producer'):
            need(manifests['dense'][field] == manifests['sub_sub'][field], 'producer identity mismatch: '+field)
        need(caps['dense']['cap_comparability_key']==caps['sub_sub']['cap_comparability_key'], 'cap mismatch')
        classes = [row['all_classes'] for arm in rows for row in rows[arm].values()]
        need(not classes or all(value == classes[0] for value in classes),
             'task class vocabulary mismatch')
        pairs, excluded = {}, []
        for sid, item in sorted(planned.items()):
            a,b = rows['sub_sub'].get(sid),rows['dense'].get(sid)
            a_status, b_status = record_status(a), record_status(b)
            if a_status != 'ok' or b_status != 'ok':
                excluded.append(dict(sample_id=sid,dense_status=b_status,sub_sub_status=a_status)); continue
            need(a['answers']==b['answers'] and a['all_classes']==b['all_classes'], 'reference mismatch')
            method_score=100*sample_score(task,a['pred'],a['answers'],a['all_classes'])
            dense_score=100*sample_score(task,b['pred'],b['answers'],b['all_classes'])
            absolute.setdefault(task, []).append((dense_score, method_score))
            delta=method_score-dense_score
            pairs.setdefault(item['document_id'],[]).append(delta)
        groups[task]=pairs
        audit[task]=dict(n_planned=len(planned),n_valid=sum(map(len,pairs.values())),n_excluded=len(excluded),excluded=excluded)
    complete=all(v['n_excluded']==0 and len(groups[t])>=2 for t,v in audit.items())
    reports,overall=paired_stats(groups,p['bootstrap_draws'],p['seed']) if complete else ({},None)
    for task, report in reports.items():
        delta = (p.get('noninferiority_delta_pp') or {}).get(task)
        report['dense_score_pp'] = float(np.mean([r[0] for r in absolute[task]]))
        report['sub_sub_score_pp'] = float(np.mean([r[1] for r in absolute[task]]))
        report['delta_pp'] = delta
        report['lower_bound_above_minus_delta'] = None if delta is None else report['ci_low'] > -delta
    return dict(status='diagnostic-pending-independent-identity-review' if complete else 'blocked-missing-failed-or-insufficient-documents',
                unit='percentage_points',tasks=reports,overall=overall,audit=audit,source_sha256=sources,
                worst_task=min(reports,key=lambda t:reports[t]['estimate']) if reports else None,
                winner=None,noninferiority='not-assessed',
                limitations='Marginal 95% percentile intervals; crossing zero is not equivalence. Frozen declarations do not verify producer identity. No scientific acceptance.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plot',action='store_true');a=ap.parse_args()
    p,sha=snapshot(a.protocol); result=analyze(p,a.protocol.parent);result['protocol_sha256']=sha
    from exp.figures.o3_retrieval import publish_directory
    final_out = a.out
    need(not final_out.exists(), 'output already exists')
    final_out.parent.mkdir(parents=True,exist_ok=True)
    a.out = Path(tempfile.mkdtemp(prefix='.paired-',dir=final_out.parent))
    write_new(a.out/'summary.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    if result['tasks']:
        source_sha=hashlib.sha256((a.out/'summary.json').read_bytes()).hexdigest()
        cells=[]
        for task in result['tasks']:
            ptr=task.replace('~','~0').replace('/','~1')
            cells.append(dict(figure_id='E1',panel_id=task,arm_id='sub_sub',reference_id='dense_fullkv',run_mode='natural',metric_id='paired_quality_delta',unit='percentage_points',aggregation='task-sample-mean-document-bootstrap',transform='none',source_id='summary',fields={k:'/tasks/'+ptr+'/'+k for k in ('estimate','ci_low','ci_high','n_independent_seq')}))
        binding=dict(schema='figure-bindings-v1',status='frozen',sources={'summary':dict(path='summary.json',sha256=source_sha,run_id=p['run_id'])},cells=cells)
        write_new(a.out/'figure_bindings.json',json.dumps(binding,indent=2).encode())
        rows,_=build_rows(a.out/'figure_bindings.json');write_new(a.out/'figure_data.tsv',tsv_bytes(rows))
        if a.plot:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(8,max(3,len(rows)*.35)))
            for i,r in enumerate(rows):
                delta = result['tasks'][r['panel_id']]['delta_pp']
                if delta is not None: ax.plot(-delta,i,'|',color='darkred',markersize=12)
                ax.plot([r['ci_low'],r['ci_high']],[i,i],color='steelblue');ax.plot(r['estimate'],i,'o',color='steelblue')
            ax.set_yticks(range(len(rows)),[r['panel_id'] for r in rows]);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('sub/sub − dense FullKV (percentage points)');ax.set_title('Diagnostic only · identity review pending');fig.tight_layout();fig.savefig(a.out/'paired_quality.svg');fig.savefig(a.out/'paired_quality.pdf');plt.close(fig)
    publish_directory(a.out, final_out)
    print(result['status'])

if __name__=='__main__':
    main()
