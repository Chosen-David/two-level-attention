import json,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.generation.logits_process import MinLengthLogitsProcessor
from benchmark.LongBench.prefill_frozen_skip import V1_TASKS,NO_CHAT_TASKS,greedy_token,stopping_policy,prepare_ids
from benchmark.LongBench.analyze_frozen_skip import analyze
from benchmark.LongBench.export_frozen_skip import export
assert len(V1_TASKS)==len(set(V1_TASKS))==16
class Tokenizer:
 eos_token_id=2
 def encode(self,s,**kw):return [3]
tok=Tokenizer();model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2,9]))
stops,minimum=stopping_policy(model,tok,'samsum');assert stops==[2,3] and minimum==1
for generated in [0,1,2]:
 logits=torch.tensor([[0.,1.,4.,5.,2.]])
 ref=MinLengthLogitsProcessor(11,eos_token_id=stops)(torch.zeros(1,10+generated,dtype=torch.long),logits.clone())
 assert torch.equal(greedy_token(logits,generated,stops,minimum),ref.argmax(-1,keepdim=True))
assert stopping_policy(model,tok,'trec')==([2,9],0)
with tempfile.TemporaryDirectory() as d:
 p=Path(d);protocol=dict(schema='frozen-near-far-v1-v4-independent-gates',selected_rows={t:list(range(3 if i==0 else 1)) for i,t in enumerate(V1_TASKS)},datasets=list(V1_TASKS),thresholds=[.05],full_dataset=True,dataset_counts={t:3 if i==0 else 1 for i,t in enumerate(V1_TASKS)})
 (p/'manifest_rank0.json').write_text(json.dumps(dict(identity='fixture',protocol=protocol)))
 for j,t in enumerate(V1_TASKS):
  path=p/'requests'/t;path.mkdir(parents=True)
  for i in protocol['selected_rows'][t]:
   arms={}
   for name,loss in [('full',0),('far_0.05',.1),('dual_0.05',.2)]:
    arms[name]=dict(score=j/20-loss,gate_counts=dict(keep=1,near=0,far=0,both=0),decisions={'0':dict(skip_near=False,skip_far=False)},frozen_kv=dict(kv_bytes=100),prefill_full_kv=dict(kv_bytes=100),generated_ids=[1],prediction='x',generated_tokens=1)
   (path/f'{i}.json').write_text(json.dumps(dict(identity='fixture',dataset=t,row_index=i,status='complete',arms=arms,input=dict(input_sha256='fixture',input_tokens=20),answers=['x'])))
 s=analyze(p);assert s['completed_requests']==18 and s['arms']['full']['avg16']==37.5
 assert abs(s['comparisons']['dual_0.05-vs-full']['avg16']['delta_pp']+20)<1e-10
 (p/'validation.json').write_text(json.dumps(dict(status='complete-validated',identity='fixture')))
 export(p)
 assert 'AVG16' in (p/'scores.tsv').read_text().splitlines()[0]
 assert 'AVG5' not in (p/'REPORT.md').read_text()
print('PASS 16-task macro/export unequal-count fixture; samsum HF min_length parity; EOS scope')
