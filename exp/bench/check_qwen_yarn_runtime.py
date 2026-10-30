"""Real-weight old/new YaRN diagnosis; not a benchmark score or speed result."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
import transformers
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from benchmark.rope_policy import configure_rope,verify_runtime_rope
from benchmark.LongBench.prefill_frozen_skip import atomic_json,sha_file


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--model',default='/home/wangyuanshuo02/Qwen3-8B')
    ap.add_argument('--data-root',default='/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER')
    ap.add_argument('--out',required=True)
    ap.add_argument('--lengths',default='65536,131072')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(42)
    config=AutoConfig.from_pretrained(args.model)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    report=dict(torch=torch.__version__,transformers=transformers.__version__,
        model_config_sha256=sha_file(Path(args.model)/'config.json'),
        gpu=torch.cuda.get_device_name(0),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha_file(p) for p in [Path(__file__),Path('benchmark/rope_policy.py')]},
        scope='1 source row per length, old/new dense first logits and8 greedy tokens; no benchmark quality/latency claim',cases=[])
    for length in map(int,args.lengths.split(',')):
        data=Path(args.data_root)/str(length)/'niah_multikey_2.jsonl'
        row=json.loads(data.read_text().splitlines()[0]);ids=tokenizer(row['input'],truncation=False).input_ids
        case=dict(length=length,index=row['index'],input_tokens=len(ids),data_sha256=sha_file(data),
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),variants={})
        logits={}
        for variant in ['legacy_original40960','corrected_original32768']:
            c=copy.deepcopy(config)
            if variant.startswith('legacy'):
                c.rope_scaling=dict(rope_type='yarn',type='yarn',factor=length/32768,
                    original_max_position_embeddings=40960,beta_fast=32,beta_slow=1)
            else: configure_rope(c,length,64,yarn=True)
            # Hold runtime capacity constant to isolate the original-length correction.
            c.max_position_embeddings=length+64
            c._attn_implementation="sdpa"
            if len(ids)+8>c.max_position_embeddings:raise ValueError('input/output exceeds runtime capacity')
            model.config=c;model.model.config=c
            for layer in model.model.layers:
                if hasattr(layer,'config'):layer.config=c
                layer.self_attn.config=c
            model.model.rotary_emb=Qwen3RotaryEmbedding(config=c,device='cuda:0')
            evidence=verify_runtime_rope(model) if variant.startswith('corrected') else dict(
                inv_freq=model.model.rotary_emb.inv_freq.float().cpu().tolist(),
                attention_scaling=float(model.model.rotary_emb.attention_scaling))
            generated=[]
            with torch.inference_mode():
                result=model(torch.tensor([ids],device='cuda:0'),use_cache=True,logits_to_keep=1)
                logits[variant]=result.logits[0,-1].float().cpu()
                if not torch.isfinite(logits[variant]).all():raise ValueError('nonfinite logits')
                cache=result.past_key_values
                for step in range(8):
                    token=result.logits[:,-1].argmax(-1,keepdim=True);generated.append(int(token))
                    if step<7:result=model(token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            case['variants'][variant]=dict(config=c.to_dict(),rope=evidence,generated_ids=generated,
                text=tokenizer.decode(generated,skip_special_tokens=True))
            del result,cache;gc.collect();torch.cuda.empty_cache()
            print(length,variant,generated,flush=True)
        old,new=logits.values()
        case['first_logits_max_abs_difference']=float((old-new).abs().max())
        case['first_logits_l2_difference']=float(torch.linalg.vector_norm(old-new))
        case['same_generated_ids']=case['variants']['legacy_original40960']['generated_ids']==case['variants']['corrected_original32768']['generated_ids']
        logits_path=out/f'first_logits_{length}.pt'
        torch.save(logits,logits_path)
        case['logits_sha256']=sha_file(logits_path)
        report['cases'].append(case);atomic_json(out/'raw.json',report)
    atomic_json(out/'DONE.json',dict(status='pending independent verification'))


if __name__=='__main__':main()
