"""Real-weight GLM four-state integration smoke; no benchmark score claim."""
import gc
import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from benchmark.LongBench.prefill_frozen_skip import generate,generate_native
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot

os.environ['TLI_FINE_GATHER']='0'
os.environ['TLI_OPERATOR_MODE']='off'
os.environ['TLI_CUDA_GRAPH']='0'
torch.manual_seed(42)
torch.set_num_threads(8)
path='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash'
tokenizer=AutoTokenizer.from_pretrained(path)
model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,
    device_map='cuda:0',attn_implementation='sdpa').eval()
args=tli_args()
assert register_patch(model,args,resolve_treatment_snapshot(args))==47
model._frozen_tli=True
model.generation_config.eos_token_id=None
ids=tokenizer('Remember the hidden number 57291. '*2000,add_special_tokens=False).input_ids[:4352]
assert len(ids)==4352
records={}
records['native_tli']=generate_native(model,ids,4,'cuda:0')
for mode,epsilon in [('full',.05),('far',1.),('near',1.),('dual',1.),('natural_dual',.05)]:
    gc.collect();torch.cuda.empty_cache()
    result=generate(model,tokenizer,ids,4,GateConfig(mode='dual' if mode=='natural_dual' else mode,epsilon=epsilon),'cuda:0')
    assert len(result['profiles'])==47
    assert all(result['tli_receipts'].values())
    assert all(x['logical_length']==4355 for x in result['final_kv']['layers'])
    records[mode]=result
    Path(sys.argv[1]).write_text(json.dumps({'input_ids':ids,'records':records,'scope':'GLM B1 4352 tokens four-step smoke, no accuracy claim'},indent=2)+'\n')
    print(mode,result['generated_ids'],result['gate_counts'],flush=True)
assert records['full']['generated_ids']==records['native_tli']
for mode in ('far','near','dual'):
    assert all(d['skip_far']==(mode in ('far','dual')) and d['skip_near']==(mode in ('near','dual')) for d in records[mode]['decisions'].values())
print('PASS fullgate/native parity and all47layers fourstates/positions/receipts',flush=True)
