"""Count missing request-specific frozen-gate signatures, without generation."""
import collections,json
from pathlib import Path
P=Path('/home/wangyuanshuo02/two-level-attention');R=P/'agent_doc/results/frozen_near_far_full5_independent005_20261029';O=P/'agent_doc/results/nfa_ablation_20261030'
report=dict(scope='HotpotQA/GovReport only; last256 mean independent regional thresholds; offline dispatch inventory, not quality estimates',thresholds=[.01,.025,.05,.1],tasks={})
for task in ('hotpotqa','gov_report'):
 counts=collections.Counter();missing=collections.Counter();unique=0
 for path in sorted((R/'requests'/task).glob('*.json')):
  x=json.load(path.open());a=x['arms']['full'];layers=sorted(a['profiles'],key=int)
  signature=lambda ds:tuple((ds[i]['skip_near'],ds[i]['skip_far']) for i in layers)
  known={signature(arm['decisions']) for arm in x['arms'].values()}
  probe=O/'C_near_only_probe'/f"{task}_{x['row_index']:04d}.json"
  if probe.exists():known.add(signature(json.load(probe.open())['arms']['near']['decisions']))
  pending=set()
  for eps in report['thresholds']:
   for mode in ('near','far','dual'):
    d={}
    for i in layers:
     nodes=a['prefill_full_kv']['layers'][int(i)]['nodes'];p=a['profiles'][i]
     d[i]=dict(skip_near=mode in ('near','dual') and nodes['near']['tokens']>0 and p['near']['mean']<eps,
               skip_far=mode in ('far','dual') and nodes['far']['tokens']>0 and p['far']['mean']<eps)
    sig=signature(d);key=f'{mode}_{eps:g}'
    counts[key]+=any(n or f for n,f in sig)
    if sig not in known:missing[key]+=1;pending.add(sig)
  unique+=len(pending)
 report['tasks'][task]=dict(active_requests=dict(counts),missing_request_arms=dict(missing),unique_missing_request_signatures=unique)
(O/'C_gate_signature_inventory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
