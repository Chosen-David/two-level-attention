"""Read-only historical re-scoring. Never edits source results or manifests."""
import collections
import hashlib
import json
from pathlib import Path
from benchmark.LongBench.eval import sample_score,validate_runcap_sidecar

PROJECT=Path('/home/wangyuanshuo02/two-level-attention')
OUT=PROJECT/'agent_doc/results/nfa_ablation_20261030'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    paths=[]
    root=PROJECT/'exp/results_longbench/Qwen3-8B'
    for folder in ['e90_subspace','pred_b7','pred_e72_mminmax','pred_e72_mavg','pred_e72_aavg','pred_E149_sink4']:
        paths.extend((root/folder).rglob('*.jsonl'))
    for folder in ['e134_l1sub','e134_l2full']:
        paths.extend((PROJECT/'exp/trace/results'/folder).rglob('*.jsonl'))
    for host,tag in [('local','pred_nb32ntoffnl2048'),('pc32','pred_nb32ntoffnl4096'),('pc31','pred_nb32ntoffnl8192')]:
        paths.extend((Path('/home/wangyuanshuo02/tmp_e155d_collect')/host/tag).glob('*.jsonl'))
    records=[]
    for p in sorted(set(paths)):
        task=p.name.split('-')[0]
        if task not in ('hotpotqa','gov_report'):continue
        rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not rows:continue
        params_path=Path(str(p)+'.tli_manifest.json')
        params=json.loads(params_path.read_text()) if params_path.exists() else {}
        rawscore=sum(sample_score(task,x['pred'],x['answers'],x.get('all_classes',[])) for x in rows)*100/len(rows)
        canonical=[json.loads(line) for line in (Path('/home/wangyuanshuo02/datasets/LongBench/data')/(task+'.jsonl')).read_text().splitlines()]
        ah=lambda data:collections.Counter(json.dumps(x['answers'],sort_keys=True) for x in data)
        binding='legacy missing sidecar; input tokens/code version not established'
        sidecar=Path(str(p)+'.runcap.json')
        if sidecar.exists():
            try:
                validate_runcap_sidecar(str(p),False)
                binding='run-cap binding passed'
            except (SystemExit,Exception) as e:binding='binding failed: '+str(e)
        records.append(dict(path=str(p),sha256=sha(p),task=task,rows=len(rows),score=round(rawscore,4),
            official_answers_multiset_matches=ah(rows)==ah(canonical),binding=binding,
            parameters=params,parameters_source=str(params_path) if params else None,
            sink=params.get('tli_sink_tokens','unknown'),swa=params.get('tli_swa_size','unknown'),
            validation='pending independent historical-reuse review'))
    (OUT/'historical_quality_inventory.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# Historical exploratory quality draft (pending independent review)','',
           'Scores re-computed from immutable raw predictions; no strict single-factor or matched-input claim. Unknown protection parameters remain unknown. Actual recorded (128,128) groups may be reused independently of main (4,128).','',
           '| Source | Task | n | Score | sink/swa | Original data checks |','|---|---|---:|---:|---|---|']
    for r in records:
        label='/'.join(Path(r['path']).parts[-3:-1])
        lines.append(f"| {label} | {r['task']} | {r['rows']} | {r['score']:.2f} | {r['sink']}/{r['swa']} | answers={'match' if r['official_answers_multiset_matches'] else 'MISMATCH'}; {r['binding']} |")
    (OUT/'HISTORICAL_DRAFT.md').write_text('\n'.join(lines)+'\n')
    print('rescored files',len(records))

if __name__=='__main__':main()
