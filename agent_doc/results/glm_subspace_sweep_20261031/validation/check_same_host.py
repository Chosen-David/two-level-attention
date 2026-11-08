from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,'/home/wangyuanshuo02/two-level-attention/.venv/lib/python3.11/site-packages')
import transformers
from transformers import AutoTokenizer
R=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/results/glm_subspace_sweep_20261031')
MODEL='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash'
DATA='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER/65536/niah_multikey_2.jsonl'
SOURCE=Path('/home/wangyuanshuo02/tli-glm-subspace')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tok=AutoTokenizer.from_pretrained(MODEL)
rows=[json.loads(x) for x in Path(DATA).read_text().splitlines()][:10]
inputs=[]
for row in rows:
 ids=tok.apply_chat_template([{'role':'user','content':row['input']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if hasattr(ids,'input_ids'): ids=ids.input_ids
 inputs.append((len(ids),hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
variants=['rope_low32','rope64','nope32','nope64','mixed32','mixed64','full256','alltoken_control']
protocols={};records=[];dense={};errors=[];sourcechecks=0;alltoken_parity=[]
raw_roots=[R/'raw125_full',R/'raw125_prior']
for root in raw_roots:
 for variant in variants:
  vdir=root/variant
  files=sorted(vdir.glob('[0-9][0-9][0-9][0-9].json'))
  if not files: continue
  p=json.loads((vdir/'protocol.json').read_text());key=f'{root.name}/{variant}'
  protocols[key]=p
  assert p['variant']==variant
  assert p['known_rows']==[0,1,2] and p['heldout_rows']==list(range(3,10))
  for f in files:
   d=json.loads(f.read_text()); i=d['row']
   assert i in range(10) and f.stem==f'{i:04}'
   assert d['identity']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
   assert (d['input_tokens'],d['input_ids_sha256'])==inputs[i]
   assert d['answers']==rows[i]['outputs'] and d['source_index']==rows[i].get('index',i)
   assert d['status']=='complete', (key,i,d.get('status'))
   assert set(d['arms'])=={'dense','selected'}
   for arm,v in d['arms'].items():
    assert tok.decode(v['generated_ids'],skip_special_tokens=True)==v['prediction'],(key,i,arm)
    assert tok.decode(v['generated_ids'])==v['raw_text'],(key,i,arm)
    assert v['answer_hits']==[str(a).lower() in v['prediction'].lower() for a in d['answers']],(key,i,arm)
    assert 2<=len(v['generated_ids'])<=64
   if i in dense and d['arms']['dense']['generated_ids']!=dense[i]: errors.append('dense mismatch '+key+'/'+str(i))
   dense.setdefault(i,d['arms']['dense']['generated_ids'])
   if variant=='alltoken_control':
    eq=d['arms']['selected']['generated_ids']==d['arms']['dense']['generated_ids']; alltoken_parity.append({'row':i,'exact':eq}); assert eq and d.get('alltoken_exact_dense')
   records.append({'host':root.name,'variant':variant,'row':i,'status':d['status'],'hits':{a:v['answer_hits'] for a,v in d['arms'].items()},'path':str(f.relative_to(R)),'sha256':sha(f)})
assert len(records)==80, len(records)
assert len(alltoken_parity)==10
assert not errors, errors
common=('source','source_commit','model_manifest','script_sha256','dataset_sha256','rows','max_new_tokens','min_new_tokens','seed','k1','k2','sink','swa','near_len','skip','rebuild_each_step','scoring_backend','prompt')
pp=list(protocols.values()); assert all(all(p[k]==pp[0][k] for k in common) for p in pp)
for p in pp:
 assert p['dataset_sha256']==sha(Path(DATA)) and p['script_sha256']==sha(R/'sweep_v2.py')
 assert p['source_commit']=='0a72672f3aa3d37f12545eeb8f397bbcc4c91d0b'
 for name,h in p['source'].items():
  assert sha(SOURCE/name)==h, name; sourcechecks+=1
scores=[]
for variant in variants:
 for arm in ['dense','selected']:
  rec=[r for r in records if r['variant']==variant]
  known=[r for r in rec if r['row'] in (0,1,2)]
  held=[r for r in rec if r['row'] in range(3,10)]
  def hit_count(rs):
   return float(sum(any(r['hits'][arm]) for r in rs))
  scores.append({'variant':variant,'arm':arm,'known':{'hits':hit_count(known),'n':len(known),'score_percent':100*hit_count(known)/len(known)},'heldout':{'hits':hit_count(held),'n':len(held),'score_percent':100*hit_count(held)/len(held)},'all':{'hits':hit_count(rec),'n':len(rec),'score_percent':100*hit_count(rec)/len(rec)}})
out={'schema_version':'glm-subspace-samehost-check/v1','review_transformers':transformers.__version__,'input_tokenization':inputs,'records':records,'source_hash_checks':sourcechecks,'protocols':{k:{x:p[x] for x in ('variant','dimensions','torch','transformers','gpu')} for k,p in protocols.items()},'alltoken_parity':alltoken_parity,'dense_parity_errors':errors,'bound_common_fields':common,'total_records':len(records),'total_arms':sum(len(x['arms']) if 'arms' in x else 0 for x in []),'observed_scores':scores,'scope':'Same-host 33.253.189.125 eight-variant ten-case recorded-output check; not benchmark/paper acceptance'}
Path(R/'validation/checks_same_host.json').write_text(json.dumps(out,indent=2)+'\n')
Path(R/'validation/observed_scores_same_host.json').write_text(json.dumps({'scope':'Same-host fixed10case recorded-output answer-hit recomputation','scores':scores},indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('records','input_tokenization')},indent=2))
