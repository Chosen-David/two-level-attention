"""Quality-only paired probe on the five near-active historical requests.

Exact old production source is imported from a separate immutable checkout.
No globalNear-only score is synthesized here; inactive-request reuse needs
independent admission of source/input/decision/runtime scope.
"""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
B=Path.home()/'tli-nfa-ablation';code=B/'code-c-legacy';gpu=int(sys.argv[1])
JOBS=[('hotpotqa',69),('hotpotqa',73),('hotpotqa',80),('hotpotqa',135),('gov_report',98)]
task,row_index=JOBS[gpu]
lease=open(f'/tmp/nfa-gpu-{gpu}.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();assert sha=='3e4a6885604ac3e719ee92a4906f87b532535314'
source=json.load((B/'nfa_c_baselines/source_manifest.json').open())
for name,digest in source['protocol']['source_sha256'].items():assert sha_file(code/name)==digest,(name,'source mismatch')
model_path=Path.home()/'tli-yarn-eval/models/Qwen3-8B'
assert sha_file(model_path/'config.json')==source['protocol']['model_config_sha256']
assert sha_file(model_path/'tokenizer_config.json')==source['protocol']['tokenizer_config_sha256']
data=B/'data'/f'{task}.jsonl';assert sha_file(data)==source['protocol']['dataset_sha256'][task]
row=json.loads(data.read_text().splitlines()[row_index]);old=json.load((B/'nfa_c_baselines'/f'{task}_{row_index:04d}.json').open())
out=B/'near_only_probe';out.mkdir(exist_ok=True)
record=dict(status='running',task=task,row_index=row_index,source_sha=sha,script_sha256=sha_file(Path(__file__)),
    old_result_sha256=sha_file(B/'nfa_c_baselines'/f'{task}_{row_index:04d}.json'),
    environment=dict(torch=torch.__version__,transformers=transformers.__version__),old_environment={k:source['protocol'][k] for k in ('torch','transformers')},
    gpu=gpu,pid=os.getpid(),arms={},scope='quality/semantics probe; no performance comparison')
path=out/f'{task}_{row_index:04d}.json';atomic_json(path,record)
torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(model_path)
prompts=json.load((code/'benchmark/LongBench/config/dataset2prompt.json').open());lengths=json.load((code/'benchmark/LongBench/config/dataset2maxlen.json').open())
ids,input_info=prepare_ids(tokenizer,row,task,prompts[task],31500);assert input_info==old['input'],'input token identity differs'
record['input']=input_info
model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
a=tli_args();assert register_patch(model,a,resolve_treatment_snapshot(a))==model.config.num_hidden_layers
model._frozen_tli=True
for mode in ('full','near'):
 r=generate(model,tokenizer,ids,lengths[task],GateConfig(mode=mode,epsilon=.05),'cuda:0')
 r['score']=sample_score(task,r['prediction'],row['answers'],row.get('all_classes',[]));record['arms'][mode]=r
 atomic_json(path,record);print(task,row_index,mode,'score',r['score'],'tokens',r['generated_tokens'],flush=True)
record['old_full_tokens_exact']=record['arms']['full']['generated_ids']==old['arms']['full']['generated_ids']
record['old_full_score_delta']=record['arms']['full']['score']-old['arms']['full']['score']
record['max_profile_mean_difference']=max(abs(record['arms']['full']['profiles'][i][reg]['mean']-old['arms']['full']['profiles'][str(i)][reg]['mean']) for i in record['arms']['full']['profiles'] for reg in ('near','far'))
record['near_decisions_match_historical']=all(d['skip_near']==old['arms']['dual_0.05']['decisions'][str(i)]['skip_near'] and not d['skip_far'] for i,d in record['arms']['near']['decisions'].items())
record['status']='complete-pending-independent-verification';atomic_json(path,record)
