"""Extended fixed-budget GLM 32D subspace candidate search (same-host diagnostic)."""
import argparse,gc,hashlib,json,os,subprocess,time,fcntl
import transformers
from pathlib import Path
from types import MethodType
for key,val in dict(TLI_OPERATOR_MODE='off',TLI_FINE_GATHER='0',TLI_SPARSE_PREFILL='0',TLI_DECODE_DENSE='0',TLI_CUDA_GRAPH='0',TLI_BATCH_INDEXER='legacy').items():os.environ[key]=val
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from benchmark.LongBench.prefill_frozen_skip import generate_native,atomic_json,sha_file,ROOT
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard

def rng(a,b,step=1):
    return list(range(a,b,step))
DIMS={
 'nope_mid32_64':rng(64,96),
 'nope_mid32_96':rng(96,128),
 'nope_mid32_128':rng(128,160),
 'nope_last32':rng(160,192),
 'nope_even32':rng(0,64,2),
 'nope_odd32':rng(1,64,2),
 'nope_first16_mid16':rng(0,16)+rng(96,112),
 'nope_first16_last16':rng(0,16)+rng(176,192),
 'rope_high32':rng(224,256),
 'rope_mid32':rng(208,240),
 'rope_even32':rng(192,256,2),
 'mixed_nope16_rope_high16':rng(0,16)+rng(224,240),
 'mixed_nope16_rope_low8_high8':rng(0,16)+rng(208,216)+rng(248,256),
 'tail32':rng(208,224)+rng(240,256),
 'mixed_nope24_rope_low8':rng(0,24)+rng(208,216),
}
p=argparse.ArgumentParser()
for name in ('model','data','out','variant','model-manifest'):p.add_argument('--'+name,required=True)
p.add_argument('--rows',type=int,default=10);a=p.parse_args();dims=DIMS[a.variant]
assert len(set(dims))==len(dims)==32
lines=Path(a.data).read_text().splitlines();assert 0<a.rows<=len(lines)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit=='0a72672f3aa3d37f12545eeb8f397bbcc4c91d0b'
out=Path(a.out)/a.variant;out.mkdir(parents=True,exist_ok=True)
lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
model_manifest=json.loads(Path(a.model_manifest).read_text())
for name,metadata in model_manifest.items():
 assert (Path(a.model)/name).stat().st_size==metadata['size']
 if not name.endswith('.safetensors'):assert sha_file(Path(a.model)/name)==metadata['sha256']
source={str(f.relative_to(ROOT)):sha_file(f) for base in ('sparse_attn','benchmark/LongBench','benchmark/RULER') for f in (ROOT/base).rglob('*.py')}
protocol=dict(known_rows=list(range(min(3,a.rows))),heldout_rows=list(range(3,a.rows)),transformers=transformers.__version__,model_manifest=model_manifest,source_commit=commit,source=source,script_sha256=sha_file(__file__),dataset_sha256=sha_file(a.data),variant=a.variant,dimensions=dims,rows=list(range(a.rows)),max_new_tokens=64,min_new_tokens=2,seed=42,k1=128,k2=1024,sink=4,swa=128,near_len=4096,skip=False,rebuild_each_step=True,scoring_backend='torch reference',prompt='GLM chat template enable_thinking=False',scope='extended 32D subspace candidate diagnostic; no benchmark or paper repair acceptance',torch=torch.__version__,gpu=torch.cuda.get_device_name())
identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
atomic_json(out/'protocol.json',protocol)
torch.manual_seed(42);torch.set_num_threads(6)
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
native=[l.self_attn.forward for l in model.model.layers]
args=tli_args();assert register_patch(model,args,resolve_treatment_snapshot(args))==47
patched=[l.self_attn.forward for l in model.model.layers]
for layer in model.model.layers:
 idx=layer.self_attn.indexer
 idx._tail_indices=MethodType(lambda self,device:torch.tensor(dims,device=device,dtype=torch.long),idx)
 idx._idx_sub_cache=None;idx._qat_idx_cache=None
 idx._triton_ok=False;idx._stat_ok=False
 old=idx.prepare_mask
 def prepare(self,q,qi,k,cu,softmax_scale=None,_old=old,**kw):
  result=_old(q,qi,k,cu,softmax_scale=softmax_scale,**kw)
  self._inc=None
  return result
 idx.prepare_mask=MethodType(prepare,idx)
with torch.inference_mode():
 for rowid,line in enumerate(lines[:a.rows]):
  row=json.loads(line)
  text=tok.apply_chat_template([{'role':'user','content':row['input']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=tok(text,add_special_tokens=False).input_ids
  assert len(ids)+64<=model.config.max_position_embeddings
  ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest();file=out/f'{rowid:04}.json'
  rec=dict(identity=identity,row=rowid,source_index=row.get('index',rowid),input_ids_sha256=ih,input_tokens=len(ids),answers=row['outputs'],arms={})
  if file.exists():
   old=json.loads(file.read_text());assert all(old[k]==rec[k] for k in ('identity','row','source_index','input_ids_sha256','answers'));rec=old
   if rec.get('status','').startswith('failed'):raise RuntimeError('Prior failed evidence requires review; not a resumable success')
  for arm in ('dense','selected'):
   if arm in rec['arms']:continue
   for layer,forward in zip(model.model.layers,native if arm=='dense' else patched):
    layer.self_attn.forward=forward;layer.self_attn.indexer.clear()
   model._frozen_tli=arm!='dense';torch.manual_seed(42);gc.collect();torch.cuda.empty_cache()
   with eager_decoding_request_guard('cuda:0'):
    gen=generate_native(model,ids,64,'cuda:0',min_new_tokens=2)
   pred=tok.decode(gen,skip_special_tokens=True)
   if arm=='selected':assert gen[0]==rec['arms']['dense']['generated_ids'][0]
   rec['arms'][arm]=dict(generated_ids=gen,prediction=pred,raw_text=tok.decode(gen),answer_hits=[str(x).lower() in pred.lower() for x in row['outputs']])
   rec['status']='complete' if len(rec['arms'])==2 else 'running';atomic_json(file,rec)
   print(json.dumps(dict(row=rowid,variant=a.variant,arm=arm,tokens=len(gen),answer_hits=rec['arms'][arm]['answer_hits'])),flush=True)
atomic_json(out/'worker_done.json',dict(identity=identity,rows=a.rows,status='raw_complete_pending_independent_review'))
print('DONE',flush=True)
