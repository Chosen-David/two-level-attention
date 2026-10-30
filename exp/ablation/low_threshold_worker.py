"""Quality-only C worker: exact old source, mandatory no-skip bridge, frozen gates."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
B=Path.home()/'tli-nfa-ablation';code=B/'code-c-legacy';rank=int(sys.argv[1]);gpu=int(sys.argv[2]);worker_count=int(sys.argv[3]);out=B/'C_low_threshold'
out.mkdir(exist_ok=True)
lease=open(f'/tmp/nfa-gpu-{gpu}.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
public_dir=Path('/tmp/tli-gpu-locks');public_dir.mkdir(exist_ok=True)
public_lease=(public_dir/(uuid+'.lock')).open('a');fcntl.flock(public_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
metadata=public_dir/(uuid+'.lease.json')
if metadata.exists():
 previous_lease=json.loads(metadata.read_text())
 previous_pid=previous_lease.get('pid')
 if previous_lease.get('status')!='released':
  raise RuntimeError(f'Existing live GPU lease must be coordinated: {previous_lease}')
metadata.write_text(json.dumps(dict(owner='nfa_ablation:C_low_threshold',pid=os.getpid(),gpu=gpu,uuid=uuid,time=time.time(),status='running'),indent=2))
m,u=map(int,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert m<1000 and u<5
os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),TLI_FINE_GATHER='0',TLI_OPERATOR_MODE='off',TLI_SPARSE_PREFILL='0',TLI_DECODE_DENSE='0',TLI_SCORER_BACKEND='difflib',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(code))
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from benchmark.LongBench.prefill_frozen_skip import prepare_ids,generate,atomic_json,sha_file,sample_score
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
plan=json.load((B/'C_LOW_THRESHOLD_PLAN.json').open());admit=json.load((B/'C_reuse_admission.json').open())
# Independent design admission is kept verbatim beside the execution plan.
assert admit['scope']=='HotpotQA/GovReport historical ring quality only; conditional Near-only .05'
assert admit['status']=='usable-with-scope' and admit['low_threshold_plan']=='design-admitted; results pending independent verification'
assert hashlib.sha256((B/'C_LOW_THRESHOLD_PLAN.json').read_bytes()).hexdigest()==admit['bindings']['C_LOW_THRESHOLD_PLAN.json']
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();assert sha==plan['source_revision']
source=json.load((B/'nfa_c_baselines/source_manifest.json').open())
for name,digest in source['protocol']['source_sha256'].items():assert sha_file(code/name)==digest
model_path=Path.home()/'tli-yarn-eval/models/Qwen3-8B'
assert sha_file(model_path/'config.json')==source['protocol']['model_config_sha256']
assert sha_file(model_path/'tokenizer_config.json')==source['protocol']['tokenizer_config_sha256']
for task in plan['datasets']:assert sha_file(B/'data'/f'{task}.jsonl')==source['protocol']['dataset_sha256'][task]
# Deterministic cost-balanced partition; max generation length is the work weight,
# not a measured throughput estimate and not a reported performance result.
bins=[[] for _ in range(worker_count)];weights=[0]*worker_count
for item in sorted(plan['requests'],key=lambda r:((512 if r['task']=='gov_report' else 32)*(1+len(r['cases']))),reverse=True):
 dest=min(range(worker_count),key=lambda i:weights[i]);bins[dest].append(item);weights[dest]+=(512 if item['task']=='gov_report' else 32)*(1+len(item['cases']))
assign=bins[rank];torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
a=tli_args();assert register_patch(model,a,resolve_treatment_snapshot(a))==model.config.num_hidden_layers
model._frozen_tli=True
prompts=json.load((code/'benchmark/LongBench/config/dataset2prompt.json').open());lengths=json.load((code/'benchmark/LongBench/config/dataset2maxlen.json').open())
datasets={t:[json.loads(x) for x in (B/'data'/f'{t}.jsonl').read_text().splitlines()] for t in plan['datasets']}
for item in assign:
 if (out/'STOP.json').exists():raise SystemExit(3)
 metadata.write_text(json.dumps(dict(owner='nfa_ablation:C_low_threshold',pid=os.getpid(),gpu=gpu,uuid=uuid,time=time.time(),status='running'),indent=2))
 task=item['task'];row_index=item['row_index'];key=f'{task}_{row_index:04d}';path=out/'requests'/f'{key}.json'
 request_lock=out/(key+'.lock');request_lock.parent.mkdir(exist_ok=True)
 with request_lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if path.exists():
   previous=json.load(path.open())
   assert previous['source_revision']==sha and previous['old_raw_sha256']==item['old_raw_sha256']
   assert previous['script_sha256']==sha_file(Path(__file__)) and previous['plan_sha256']==sha_file(B/'C_LOW_THRESHOLD_PLAN.json')
   if previous['status']=='complete-pending-independent-verification':continue
  oldpath=B/'C_low_baselines'/f'{key}.json';assert sha_file(oldpath)==item['old_raw_sha256']
  old=json.load(oldpath.open());row=datasets[task][row_index]
  ids,info=prepare_ids(tokenizer,row,task,prompts[task],31500);assert info==old['input']
  record=dict(status='running',task=task,row_index=row_index,input=info,source_revision=sha,
      old_raw_sha256=item['old_raw_sha256'],script_sha256=sha_file(Path(__file__)),plan_sha256=sha_file(B/'C_LOW_THRESHOLD_PLAN.json'),
      environment=dict(torch=torch.__version__,transformers=transformers.__version__),gpu=gpu,rank=rank,pid=os.getpid(),arms={})
  atomic_json(path,record)
  atomic_json(out/f'heartbeat{rank}.json',dict(time=time.time(),task=task,row_index=row_index,arm='full_bridge',pid=os.getpid()))
  print('START',key,'full_bridge',flush=True)
  full=generate(model,tokenizer,ids,lengths[task],GateConfig(mode='full',epsilon=.05),'cuda:0')
  full['score']=sample_score(task,full['prediction'],row['answers'],row.get('all_classes',[]));record['full_bridge']=full
  tokens_match=full['generated_ids']==old['arms']['full']['generated_ids']
  profiles_match=all(full['profiles'][li]==old['arms']['full']['profiles'][str(li)] for li in full['profiles'])
  record['full_bridge_tokens_exact']=tokens_match;record['full_bridge_profiles_exact']=profiles_match
  if not (tokens_match and profiles_match):
   record['status']='bridge-failed-no-conditional-reuse';atomic_json(path,record)
   atomic_json(out/'STOP.json',dict(reason='mandatory old Full bridge failed',request=key,tokens_match=tokens_match,profiles_match=profiles_match));raise SystemExit(3)
  atomic_json(path,record)
  for case in item['cases']:
   arm=f"far_{case['epsilon']:g}"
   atomic_json(out/f'heartbeat{rank}.json',dict(time=time.time(),task=task,row_index=row_index,arm=arm,pid=os.getpid()))
   print('START',key,arm,flush=True)
   r=generate(model,tokenizer,ids,lengths[task],GateConfig(mode='far',epsilon=case['epsilon']),'cuda:0')
   if {str(k):v for k,v in r['decisions'].items()}!=case['expected_decisions']:
    record['status']='gate-mismatch-no-conditional-reuse';record['failed_arm']=r;atomic_json(path,record)
    atomic_json(out/'STOP.json',dict(reason='new gate differs from predeclared input-derived vector',request=key,arm=arm));raise SystemExit(3)
   r['score']=sample_score(task,r['prediction'],row['answers'],row.get('all_classes',[]));record['arms'][arm]=r;atomic_json(path,record)
   print('DONE',key,arm,'score',r['score'],'tokens',r['generated_tokens'],flush=True)
  record['status']='complete-pending-independent-verification';atomic_json(path,record);torch.cuda.empty_cache()
atomic_json(out/f'worker{rank}_done.json',dict(source_revision=sha,requests=len(assign),status='raw pending independent verification'))

current_lease=json.loads(metadata.read_text())
assert current_lease['pid']==os.getpid() and current_lease['owner']=='nfa_ablation:C_low_threshold'
metadata.write_text(json.dumps(dict(owner='nfa_ablation:C_low_threshold',pid=os.getpid(),gpu=gpu,uuid=uuid,time=time.time(),status='released'),indent=2))
