"""Scope-limited historical B raw-score inventory; no selector fidelity claim."""
import collections,hashlib,json
from pathlib import Path
from benchmark.LongBench.eval import sample_score
P=Path('/home/wangyuanshuo02/two-level-attention');R=Path('/tmp/e109_scan_v2');O=P/'agent_doc/results/nfa_ablation_20261030'
choices={
 'global_minmax_declared':'pred_E109_mavg_a0_b0_g0',
 'partition_minmax_minmax_declared':'pred_E109_mminmax_a0.5_b0.5_g0.5',
 'partition_avg_avg_declared':'pred_E109_aavg_a0.25_b0.125_g0.375',
 'partition_nearavg_farminmax_declared':'pred_E109_mavg_a0.25_b0.125_g0.625'}
manifest=json.load((P/'exp/trace/results/e109_lbv1_manifest.json').open())
known={v['file']:v['sha256'] for arm in manifest['arms'].values() for task,v in arm['tasks'].items() if task in ('hotpotqa','gov_report')}
output=[]
for arm,folder in choices.items():
 for task in ('hotpotqa','gov_report'):
  path=next((R/folder).glob(task+'*.jsonl'));raw=path.read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
  official=[json.loads(x) for x in (Path('/home/wangyuanshuo02/datasets/LongBench/data')/(task+'.jsonl')).read_text().splitlines()]
  identity=lambda rr:collections.Counter(json.dumps([r.get('answers'),r.get('length')],sort_keys=True) for r in rr)
  assert len(rows)==200 and identity(rows)==identity(official)
  digest=hashlib.sha256(raw).hexdigest();bound=known.get(str(path))
  if bound is not None:assert digest==bound
  scores=[sample_score(task,r['pred'],r['answers'],r.get('all_classes',[])) for r in rows]
  output.append(dict(arm=arm,task=task,rows=200,score=100*sum(scores)/200,path=str(path),sha256=digest,
     historical_manifest_hash_match=True if bound else None,answers_lengths_multiset_match=True,
     declaration=folder,geometry='historical source era128/128; exact run binding missing',
     limitations='Different alpha/beta/gamma; no fixed-quota causal claim. Actual Avg selector/source binding requires independent legacy review.',
     validation='pending independent historical-selector/data admission'))
(O/'historical_regional_quality.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('arm','task','score','historical_manifest_hash_match')} for r in output],indent=2))
