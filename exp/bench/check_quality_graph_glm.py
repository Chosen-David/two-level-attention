"""Real-weight GLM Graph/eager token smoke; not a benchmark quality score."""
import gc
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from sparse_attn.ops.quality_graph import from_dynamic


def main():
    torch.manual_seed(42)
    root=Path(__file__).resolve().parents[2]
    helpers=runpy.run_path(str(root/'exp/bench/test_glm47_d256_gates.py'))
    path='/home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash'
    tokenizer=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    args=helpers['make_prod_args']()
    args.tli_dyn_dgate=True
    args.tli_far_skip_backend='legacy';args.tli_far_skip_cache='retain'
    resolve_far_skip_modes(args,config=model.config)
    assert register_patch(model,args,resolve_treatment_snapshot(args))==47
    text='The hidden key is 57291. Remember this key and answer the question. '
    ids=tokenizer(text*3000,return_tensors='pt').input_ids[:,:16384].to('cuda')
    assert ids.shape[-1]==16384
    records=[]
    for policy in ('0','1'):
        os.environ['TLI_CUDA_GRAPH']=policy
        with torch.inference_mode(),eager_decoding_request_guard():
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            past=out.past_key_values;first=out.logits[:,-1].argmax(-1).item();del out
            generated=from_dynamic(model,past,first,5,set())
            if generated is None:
                generated=[first]
                for _ in range(4):
                    out=model(input_ids=torch.tensor([[generated[-1]]],device='cuda'),
                              past_key_values=past,use_cache=True,logits_to_keep=1)
                    generated.append(out.logits[:,-1].argmax(-1).item())
                    past=out.past_key_values;del out
            records.append(dict(policy=policy,tokens=generated,
                execution=dict(model._quality_graph_execution),
                normalization_capture_routes=[getattr(x.self_attn.indexer, "_graph_normalization_route", None) for x in model.model.layers],
                frozen=[x.self_attn.indexer._dyn_frozen for x in model.model.layers],
                skip_far=[x.self_attn.indexer.skip_far for x in model.model.layers]))
        del past
        gc.collect();torch.cuda.empty_cache()
    result=dict(scope='real-weight GLM B1 16384-token deterministic smoke, not RULER accuracy',
                records=records,tokens_equal=records[0]['tokens']==records[1]['tokens'])
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    assert result['tokens_equal']
    assert records[1]['execution']['replay_steps']==4
    assert records[0]['skip_far']==records[1]['skip_far']
    assert all(records[1]['frozen'])
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
