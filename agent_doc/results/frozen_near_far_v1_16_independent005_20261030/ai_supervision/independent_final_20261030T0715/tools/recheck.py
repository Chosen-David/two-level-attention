"""Read-only CPU independent snapshot audit. Run with python -B; output only --output.
Does not import producer generation/validation/scoring, launch GPU, or mutate run.
For ROUGE tasks requires rouge (same metric library, separately invoked).
"""
import argparse, collections, datetime, difflib, hashlib, json, math, os, re, string, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent/'reference_dependency'))
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);ap.add_argument('--require-full',action='store_true');ap.add_argument('--no-snapshot',action='store_true');args=ap.parse_args()
p=Path(args.run);src=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(f):return json.loads(f.read_bytes())
evidence={};errors=[]
def check(ok,label):
 if not ok:errors.append(label)
ms=[]
for f in sorted(p.glob('manifest_rank*.json')):
 b=f.read_bytes();evidence[str(f)]=sha(b);ms.append(json.loads(b))
m=ms[0];cfg=m['protocol'];identity=m['identity'];check(sha(json.dumps(cfg,sort_keys=True).encode())==identity,'protocol digest')
check(all(x['protocol']==cfg and x['identity']==identity for x in ms),'manifest agreement')
check(len({x['rank'] for x in ms})==len(ms),'manifest unique ranks')
check(cfg['world_size']==128 and cfg['thresholds']==[.05] and cfg['statistic']=='mean','fixed config')
check(len(cfg['datasets'])==16 and cfg['full_dataset'],'full16 config')
for name,h in cfg['source_sha256'].items():
 for root in [src,Path('/home/wangyuanshuo02/two-level-attention-frozen-main')]:
  f=root/name;actual=sha(f.read_bytes());evidence[str(f)]=actual;check(actual==h,'source '+str(f))
from transformers import AutoTokenizer
model=Path(cfg['model']);tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
for n,key in [('config.json','model_config_sha256'),('tokenizer_config.json','tokenizer_config_sha256')]:
 evidence[str(model/n)]=sha((model/n).read_bytes());check(evidence[str(model/n)]==cfg[key],n)
# Record additional unbound artifacts without pretending the producer bound them.
for n in ['generation_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json']:
 if (model/n).exists():evidence[str(model/n)]=sha((model/n).read_bytes())
mc=read(model/'config.json');eos=read(model/'generation_config.json')['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
rows={};expected=set()
for ds in cfg['datasets']:
 f=Path('/home/wangyuanshuo02/datasets/LongBench/data')/(ds+'.jsonl');b=f.read_bytes();evidence[str(f)]=sha(b);check(sha(b)==cfg['dataset_sha256'][ds],ds+' dataset hash')
 rows[ds]=[json.loads(x) for x in b.splitlines()];inds=cfg['selected_rows'][ds]
 check(len(inds)==len(set(inds))==len(rows[ds]) and set(inds)==set(range(len(rows[ds]))),ds+' full row coverage')
 expected.update((ds,i) for i in inds)
check(len(expected)==3750,'3750 expected')
pr=read(src/'benchmark/LongBench/config/dataset2prompt.json');maxlen=read(src/'benchmark/LongBench/config/dataset2maxlen.json')
nochat={'trec','triviaqa','samsum','lcc','repobench-p'}
def norm(s):return re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split()
def score(ds,pred,ans,classes):
 if ds in {'trec','triviaqa','samsum'}:pred=pred.lstrip('\n').split('\n')[0]
 scores=[]
 for gold in ans:
  if ds in {'narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique','triviaqa'}:
   a,b=norm(pred),norm(gold);overlap=sum((collections.Counter(a)&collections.Counter(b)).values());s=2*overlap/(len(a)+len(b)) if overlap else 0.
  elif ds in {'lcc','repobench-p'}:
   line=next((x for x in pred.lstrip('\n').split('\n') if not any(z in x for z in ['`','#','//'])),'');s=round(100*difflib.SequenceMatcher(None,line,gold).ratio())/100
  elif ds in {'passage_count','passage_retrieval_en'}:
   nums=re.findall(r'\d+',pred);target=re.search(r'Paragraph (\d+)',gold).group(1) if ds=='passage_retrieval_en' else str(gold);s=sum(x==target for x in nums)/len(nums) if nums else 0
  elif ds=='trec':
   hits=[x for x in classes if x in pred]
   for x in hits[:]:
    if x in gold and x!=gold:hits.remove(x)
   s=1/len(hits) if gold in hits else 0
  else:
   from rouge import Rouge
   try:s=Rouge().get_scores([pred],[gold],avg=True)['rouge-l']['f']
   except ValueError:s=0.
  scores.append(s)
 return max([0.]+scores)
def bounds(n):return min(4,n),min(max(min(4,n),n-128),max(min(4,n),((n-4224)//64)*64)),max(min(4,n),n-128)
records=[];partial=[];seen=set();tot=collections.Counter();activation=collections.Counter();recomputed=collections.defaultdict(list)
for f in sorted(p.glob('requests/*/*.json')):
 b=f.read_bytes();r=json.loads(b);key=(r['dataset'],r['row_index']);check(key not in seen and key in expected,'unique expected '+str(key));seen.add(key)
 check(r['identity']==identity,'record identity '+str(key))
 if r['status']!='complete':partial.append(str(f));continue
 evidence[str(f)]=sha(b);dest=out/'snapshot'/f.relative_to(p);
 if not args.no_snapshot:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 ds,i=key;row=rows[ds][i];check(r['answers']==row['answers'],'answers '+str(key));text=pr[ds].format(**row);ids=tok(text,add_special_tokens=False).input_ids;raw=len(ids)
 if raw>cfg['context_cap']:
  n=cfg['context_cap']//2;text=tok.decode(ids[:n],skip_special_tokens=True)+tok.decode(ids[-(cfg['context_cap']-n):],skip_special_tokens=True)
 if ds not in nochat:text=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tok(text,add_special_tokens=False).input_ids;check(r['input']['input_sha256']==sha(json.dumps(ids).encode()) and r['input']['input_tokens']==len(ids) and r['input']['raw_tokens']==raw,'input '+str(key))
 check(set(r['arms'])=={'full','far_0.05','dual_0.05'},'arms '+str(key));tot[ds]+=1
 for arm,a in r['arms'].items():
  tag=f'{ds}/{i}/{arm}';g=a['generated_ids'];stop=[tok.eos_token_id,tok.encode('\n',add_special_tokens=False)[-1]] if ds=='samsum' else eos
  check(cfg['max_new_tokens'][ds]==maxlen[ds],'maxlen '+ds)
  check(0<len(g)<=maxlen[ds] and len(g)==a['generated_tokens'],'length '+tag)
  check(not any(t in stop for t in g[:-1]) and (g[-1] in stop or len(g)==maxlen[ds]),'stop '+tag)
  if ds=='samsum':check(g[0] not in stop,'samsum min1 '+tag)
  check(g[0]==r['arms']['full']['generated_ids'][0],'first token '+tag)
  check(tok.decode(g,skip_special_tokens=True)==a['prediction'],'decode '+tag)
  s=score(ds,a['prediction'],row['answers'],row.get('all_classes',[]));check(math.isfinite(a['score']) and abs(s-a['score'])<1e-12,'score '+tag);recomputed[(ds,arm)].append(s)
  counts=collections.Counter()
  factor=2*2*mc['num_key_value_heads']*mc.get('head_dim',mc['hidden_size']//mc['num_attention_heads'])
  for phase in ['prefill_full_kv','frozen_kv','final_kv']:
   layers=a[phase]['layers'];check(len(layers)==mc['num_hidden_layers'],'layer count '+tag)
   check(a[phase]['kv_bytes']==sum(node['kv_bytes'] for st in layers for node in st['nodes'].values()),'total bytes '+tag)
   for st in layers:
    for node in st['nodes'].values():check(0<=node['tokens']<=node['capacity'] and node['kv_bytes']==factor*node['capacity'],'node bytes '+tag)
  if len(g)>1:
   check(len(a['tli_receipts'])==mc['num_hidden_layers'],'receipt layers '+tag)
   for li,receipt in a['tli_receipts'].items():
    st=a['final_kv']['layers'][int(li)]
    check(receipt['scorer']=='TLIIndexer.compute_score' and receipt['selector']=='TLIIndexer.compute_mask','actual tli '+tag)
    check(receipt['dimensions']==list(range(48,64))+list(range(112,128)),'tail32 '+tag)
    check(receipt['logical_tokens']==st['logical_length'] and receipt['visible_tokens']==st['attention_tokens'],'receipt lengths '+tag)
    check(0<min(receipt['selected_per_kv_head'])<=max(receipt['selected_per_kv_head'])<=1024,'L2 budget '+tag)

  for li in range(mc['num_hidden_layers']):
   profile=a['profiles'][str(li)];d=a['decisions'][str(li)];before=a['prefill_full_kv']['layers'][li]
   n=profile['near']['mean'];far=profile['far']['mean'];sn=arm=='dual_0.05' and n<.05 and before['nodes']['near']['tokens']>0;sf=arm!='full' and far<.05 and before['nodes']['far']['tokens']>0
   check(d=={'skip_near':sn,'skip_far':sf},'gate '+tag+str(li));counts['both' if sn and sf else 'near' if sn else 'far' if sf else 'keep']+=1
   for region in ['near','far']:
    for stat in ['mean','p95','max']:
     value=profile[region][stat];check(math.isfinite(value) and 0<=value<=1+1e-6,'finite mass '+tag);check(abs(value-r['arms']['full']['profiles'][str(li)][region][stat])<1e-6,'profile parity '+tag)
   if arm=='dual_0.05':check(sf==r['arms']['far_0.05']['decisions'][str(li)]['skip_far'],'independent far '+tag)
   for phase,length in [('frozen_kv',len(ids)),('final_kv',len(ids)+len(g)-1)]:
    st=a[phase]['layers'][li];sink,fh,nh=bounds(length);wanted={'sink':sink,'swa':length-nh}
    if not sf:wanted['far']=fh-sink
    if not(sn and sf):wanted['near_staging' if sn else 'near']=nh-fh
    check(st['logical_length']==length and {k:v['tokens'] for k,v in st['nodes'].items()}==wanted,'geometry '+tag+phase+str(li))
    check(st['stored_tokens']+st['permanently_dropped']==length,'conservation '+tag)
    check(st['attention_tokens']==sum(v for k,v in wanted.items() if k!='near_staging'),'visible tokens '+tag)
    for name,node in st['nodes'].items():check(node['attention_visible']==(name!='near_staging'),'staging invisible '+tag)
    if sn and not sf:check(st['permanently_dropped']==0,'near no eviction '+tag)
   if sn and not sf and a['final_kv']['layers'][li]['nodes']['far']['tokens']>a['frozen_kv']['layers'][li]['nodes']['far']['tokens']:activation['near_only_far_growth']+=1
  check(all(counts[k]==a['gate_counts'][k] for k in ['keep','near','far','both']),'gate counts '+tag);activation.update(counts)
 records.append({'dataset':ds,'row':i,'sha256':sha(b)})
per_task_scores={arm:{ds:100*sum(v)/len(v) for (ds,a),v in recomputed.items() if a==arm} for arm in ['full','far_0.05','dual_0.05']}
if args.require_full:
 check(len(ms)==128 and set(x['rank'] for x in ms)==set(range(128)),'all128 manifests')
 check(len(records)==3750 and not partial and seen==expected,'full3750 completion')
 for rank in range(128):
  f=p/f'worker_done_rank{rank}.json';check(f.exists() and read(f)['identity']==identity,'worker done '+str(rank))
 for name in ['summary.json','validation.json','raw_index.json','predictions.jsonl']:check((p/name).exists(),'final artifact '+name)
 if (p/'summary.json').exists():
  summary=read(p/'summary.json');check(summary['identity']==identity and summary['status']=='complete','summary identity/status')
  for arm,scores in per_task_scores.items():
   check(summary['arms'][arm]['per_task']==scores or all(abs(summary['arms'][arm]['per_task'][ds]-v)<1e-10 for ds,v in scores.items()),'summary task means '+arm)
   check(len(scores)==16 and abs(summary['arms'][arm]['avg16']-sum(scores.values())/16)<1e-10,'AVG16 '+arm)
 if (p/'raw_index.json').exists():
  index=read(p/'raw_index.json');check(len(index['records'])==3750,'export index3750')
  for item in index['records']:check(sha((p/item['path']).read_bytes())==item['sha256'],'export hash '+item['path'])
 if (p/'predictions.jsonl').exists():
  pp=[json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()];check(len(pp)==11250,'11250 predictions');check(len({(r['dataset'],r['row_index'],r['arm']) for r in pp})==11250,'unique predictions')
  for pred in pp:
   raw=read(p/pred['raw_path']);arm=raw['arms'][pred['arm']]
   check(pred['score']==arm['score'] and pred['prediction']==arm['prediction'] and pred['generated_ids']==arm['generated_ids'],'export prediction binding')
result={'status':'scoped_cpu_checks_pass' if not errors else 'failed','full_acceptance':'pending','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':identity,'source_files':len(cfg['source_sha256']),'manifest_count':len(ms),'expected_manifests':128,'completed_requests_checked':len(records),'generated_outputs_checked':3*len(records),'expected_requests':3750,'per_task':dict(tot),'independent_per_task_scores':per_task_scores,'running_files_skipped':partial,'gate_layer_counts':dict(activation),'errors':errors,'records':records,'artifact_hashes':evidence,'independent_gpu_rerun':False,'score_reference':'independent implementations; ROUGE library reference when applicable','limits':['No GPU rerun or full-run acceptance','Logged masses are not recomputed from actual Q/K; those tensors are not retained','Tokenizer/weight/generation dependencies not all bound in producer identity']}
(out/'cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_hashes','records','running_files_skipped']},indent=2))
if errors:raise SystemExit(1)
