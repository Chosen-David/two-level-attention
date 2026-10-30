"""Scoped C quality plan; excludes.1, deduplicates by request/gate vector."""
import hashlib,json
from pathlib import Path
P=Path('/home/wangyuanshuo02/two-level-attention');R=P/'agent_doc/results/frozen_near_far_full5_independent005_20261029';O=P/'agent_doc/results/nfa_ablation_20261030'
requests=[];totals={}
for task in ('hotpotqa','gov_report'):
 counts={'bridges':0,'new_far_generations':0,'epsilon_0.01':0,'epsilon_0.025':0}
 for path in sorted((R/'requests'/task).glob('*.json')):
  raw=path.read_bytes();x=json.loads(raw);a=x['arms']['full'];cases=[]
  for eps in (.01,.025):
   decisions={}
   for li,profile in a['profiles'].items():
    nodes=a['prefill_full_kv']['layers'][int(li)]['nodes']
    assert not (nodes['near']['tokens']>0 and profile['near']['mean']<eps),'near gate unexpectedly active'
    decisions[li]=dict(skip_near=False,skip_far=nodes['far']['tokens']>0 and profile['far']['mean']<eps)
   if any(d['skip_far'] for d in decisions.values()):
    cases.append(dict(epsilon=eps,mode='far',equivalent_mode='dual',expected_decisions=decisions))
    counts['new_far_generations']+=1;counts[f'epsilon_{eps:g}']+=1
  if cases:
   counts['bridges']+=1
   requests.append(dict(task=task,row_index=x['row_index'],old_raw_path=str(path),old_raw_sha256=hashlib.sha256(raw).hexdigest(),cases=cases))
 totals[task]=counts
plan=dict(status='proposed pending independent C admission; not launched',datasets=['hotpotqa','gov_report'],thresholds=[.01,.025,.05],excluded_thresholds=[.1],source_revision='3e4a6885604ac3e719ee92a4906f87b532535314',
    reference_threshold=.05,sink=4,swa=128,near_len=4096,
    protocol='Exact historical full-prompt frozen TLI source. Each active-request union gets a new no-skip bridge first; require exact old tokens/profile/gate match before conditional reuse.',
    equivalence='At.01/.025 near gates are all false in both tasks: near-only reuses no-skip; dual uses the exact same frozen decision vector as far-only. Inactive requests reuse same-request no-skip. Every reference explicit.',
    scope='quality and gate/cache correctness only; no timing/throughput/profiling claims',totals=totals,requests=requests)
assert sum(x['new_far_generations'] for x in totals.values())==139
assert len(requests)==127
(O/'C_LOW_THRESHOLD_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(totals,indent=2))
