"""Explicit conditional reuse ledger; pending independent admission, not400 new runs."""
import hashlib,json
from pathlib import Path
import numpy as np
from benchmark.LongBench.eval import sample_score
P=Path('/home/wangyuanshuo02/two-level-attention');R=P/'agent_doc/results/frozen_near_far_full5_independent005_20261029';O=P/'agent_doc/results/nfa_ablation_20261030'
binding=json.load((O/'historical_frozen_recheck.json').open());hashes={x['path']:x['sha256'] for x in binding['raw_bindings']}
result=dict(status='pending independent conditional-reuse admission',new_paired_requests=5,reused_gateoff_requests=395,threshold=.05,per_task={},ledger=[])
for task in ('hotpotqa','gov_report'):
 arms={k:[] for k in ('full','near_0.05','far_0.05','dual_0.05')};fresh=0
 for path in sorted((R/'requests'/task).glob('*.json')):
  raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert hashes[str(path)]==sha
  x=json.loads(raw);near_active=any(d['skip_near'] for d in x['arms']['dual_0.05']['decisions'].values())
  row=dict(task=task,row_index=x['row_index'],old_raw_path=str(path),old_raw_sha256=sha,near_active=near_active)
  if near_active:
   probe=O/'C_near_only_probe'/f"{task}_{x['row_index']:04d}.json";a=json.load(probe.open())
   assert a['status']=='complete-pending-independent-verification'
   assert a['old_full_tokens_exact'] and a['near_decisions_match_historical'] and a['max_profile_mean_difference']==0
   assert a['input']==x['input'] and a['old_result_sha256']==sha
   score=sample_score(task,a['arms']['near']['prediction'],x['answers'],[])
   assert abs(score-a['arms']['near']['score'])<1e-12
   row.update(near_only_source='new paired near-only probe',path=str(probe),sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),environment=a['environment']);fresh+=1
  else:
   # At near-only mode far is always false. With no near gate this is exactly
   # the no-skip decision vector for this same request, not another request.
   score=x['arms']['full']['score'];row['near_only_source']='conditional reuse: same request, all gates keep'
  arms['near_0.05'].append(100*score)
  for arm in ('full','far_0.05','dual_0.05'):arms[arm].append(100*x['arms'][arm]['score'])
  row['near_only_score_points']=100*score;result['ledger'].append(row)
 rng=np.random.default_rng(20261030);indices=rng.integers(0,200,(10000,200));summary={}
 for arm,values in arms.items():
  d=np.array(values)-np.array(arms['full']);boot=d[indices].mean(1)
  summary[arm]=dict(score_points=float(np.mean(values)),delta_vs_no_skip=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
 result['per_task'][task]=dict(n=200,new_paired=fresh,conditional_reuse=200-fresh,arms=summary)
(O/'C_conditional_near_only.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['per_task'],indent=2))
