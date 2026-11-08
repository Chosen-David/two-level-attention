import os,sys,json,hashlib,math
from pathlib import Path
os.environ['TLI_SCORER_BACKEND']='difflib'
src=Path('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030');sys.path.insert(0,str(src))
from transformers import AutoTokenizer
from benchmark.LongBench.eval import sample_score
p=Path('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_v1_16_smoke_20261030')
m=json.loads((p/'manifest_rank0.json').read_text());cfg=m['protocol']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['identity']
assert cfg['git_head']=='08ce44d6027a235080e7ff4127754af6b1ae91e0'
assert cfg['thresholds']==[.05] and cfg['statistic']=='mean'
for f,h in cfg['source_sha256'].items():assert sha(src/f)==h
model=Path(cfg['model']);tok=AutoTokenizer.from_pretrained(model)
eos=json.loads((model/'generation_config.json').read_text())['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
prompts=json.loads((src/'benchmark/LongBench/config/dataset2prompt.json').read_text());found=[];native=0
for ds,ids in cfg['selected_rows'].items():
 data=Path('/home/wangyuanshuo02/datasets/LongBench/data')/(ds+'.jsonl');assert sha(data)==cfg['dataset_sha256'][ds]
 rows=[json.loads(s) for s in data.read_text().splitlines()]
 for i in ids:
  f=p/'requests'/ds/f'{i:04d}.json';r=json.loads(f.read_text());assert r['status']=='complete'
  assert r['identity']==m['identity'] and r['answers']==rows[i]['answers'];assert set(r['arms'])=={'full','far_0.05','dual_0.05'}
  text=prompts[ds].format(**rows[i]);tokens=tok(text,add_special_tokens=False).input_ids
  raw=len(tokens)
  if raw>cfg['context_cap']:
   half=cfg['context_cap']//2;text=tok.decode(tokens[:half],skip_special_tokens=True)+tok.decode(tokens[-(cfg['context_cap']-half):],skip_special_tokens=True)
  # All three declared smoke tasks are no-chat tasks: reconstruct directly.
  assert ds in ['samsum','trec','lcc'];tokens=tok(text,add_special_tokens=False).input_ids
  assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==r['input']['input_sha256']
  stop=[tok.eos_token_id,tok.encode('\n',add_special_tokens=False)[-1]] if ds=='samsum' else eos
  for arm,a in r['arms'].items():
   g=a['generated_ids'];assert 0<len(g)<=cfg['max_new_tokens'][ds]
   assert not set(g[:-1])&set(stop) and (g[-1] in stop or len(g)==cfg['max_new_tokens'][ds])
   if ds=='samsum':assert g[0] not in stop
   assert a['prediction']==tok.decode(g,skip_special_tokens=True)
   assert math.isfinite(a['score']) and abs(a['score']-sample_score(ds,a['prediction'],rows[i]['answers'],rows[i].get('all_classes',[])))<1e-12
   native+=bool(a.get('native_full_parity'))
   for li,d in a['decisions'].items():
    pr=a['profiles'][li];nodes=a['prefill_full_kv']['layers'][int(li)]['nodes']
    assert d['skip_near']==(arm=='dual_0.05' and pr['near']['mean']<.05 and nodes['near']['tokens']>0)
    assert d['skip_far']==(arm!='full' and pr['far']['mean']<.05 and nodes['far']['tokens']>0)
    assert d['skip_far']==r['arms']['far_0.05']['decisions'][li]['skip_far'] if arm=='dual_0.05' else True
  found.append(dict(dataset=ds,row=i,sha256=sha(f),arms=3))
assert len(found)==3 and native>=1
print(json.dumps(dict(status='independent_checks_passed',records=found,source_files=len(cfg['source_sha256']),native_parity_records=native,manifest_sha256=sha(p/'manifest_rank0.json')),indent=2))
