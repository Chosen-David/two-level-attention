from pathlib import Path
import json,hashlib,shutil,datetime
B=Path(__file__).resolve().parent
V=B.parent.parent
M=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030')
sha=lambda b:hashlib.sha256(b).hexdigest()
bind={}; counts={}; excluded={}
q=json.loads((V/'queue.json').read_text());assert len(q['jobs'])==128 and all(j['status']=='done' for j in q['jobs']),'full queue required'
assert len(list(V.glob('requests/*/*.json')))==3750,'full3750 required'
def cp(p,d):
 b=p.read_bytes();assert b==p.read_bytes(),str(p)
 q=B/d;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 bind[str(d)]={'origin':str(p),'sha256':sha(b)}
 return b
for label,run,pattern in [('v1',V,'requests/*/*.json')]:
 counts[label]=0;excluded[label]=[]
 for p in sorted(run.glob(pattern)):
  b=p.read_bytes();r=json.loads(b)
  if r['status']!='complete':excluded[label].append(str(p));continue
  assert b==p.read_bytes()
  cp(p,Path('fixed')/label/p.relative_to(run));counts[label]+=1
 for pat in (['manifest_rank*.json','worker_done_rank*.json','queue.json','dispatch_plan.json','validation_plan.json','run_contract.json','summary.json','validation.json','predictions.jsonl','raw_index.json','scores.tsv','paired_differences.tsv','activation_storage.tsv','per_task_activation.tsv','per_request_scores.tsv','REPORT.md'] if label=='v1' else ['ruler64k/manifest.json','run_contract.json','validation_plan.json','smoke_verification.md','reuse_decision.json','smoke/manifest.json','smoke/requests/*.json','smoke/validation.json','smoke/summary.json','smoke/GENERATED.json']):
  for p in run.glob(pat):cp(p,Path('fixed')/label/p.relative_to(run))
for name in ['NEXT_STEPS.md','independent_review_20261030T0313/REPORT_DETAIL.md','independent_review_20261030T0213/REPORT_DETAIL.md','independent_manifest_20261030T0415/REPORT_DETAIL.md']:
 cp(V/'ai_supervision'/name,Path('context/v1')/name)
for name in ['independent_review_20261030T0243/REPORT_DETAIL.md','independent_review_20261030T0343/REPORT_DETAIL.md']:
 cp(M/name,Path('context/mk2')/name)
for name in ['guide/GUIDE.md','task/TASK.md','task/task_details/TL-FROZEN-V1-16-20261030.md','task/task_details/TL-FROZEN-QASPER-MK2-20261030.md']:
 cp(Path('/home/wangyuanshuo02/two-level-attention/agent_doc')/name,Path('context/project')/name)
(B/'CAPTURE_BINDINGS.json').write_text(json.dumps(bind,indent=2))
(B/'SNAPSHOT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,excluded=excluded,atomic_directory=False),indent=2))
print(counts)
