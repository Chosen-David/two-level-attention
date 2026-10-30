"""Paired fixed-input/frozen-gate ring versus paged model validation and ABBA timing.

Reuses the frozen-region prefill profiler, gate policy and actual TLI selection.
Known decode replay tokens are not reported as generated-answer throughput.
"""
import argparse,copy,gc,hashlib,json,os,subprocess,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.frozen_region_cache import FrozenRegionCache,RegionGeometry
from sparse_attn.ops.paged_region_cache import PagedRegionCache
from sparse_attn.ops.frozen_region_gate import GateConfig,decide_regions
from sparse_attn.ops.frozen_region_tli import tli_args
from benchmark.LongBench.prefill_frozen_skip import capture_profiles

@torch.inference_mode()
def trial(model,ids,tokens,decisions,kind,geometry):
    for layer in model.model.layers:layer.self_attn.indexer.clear();layer.self_attn.indexer.skip_far=False
    cls=FrozenRegionCache if kind=='ring' else PagedRegionCache
    cache=cls(model.config.num_hidden_layers,geometry);cache.tli_regions=True
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start_alloc=torch.cuda.memory_allocated();torch.cuda.synchronize();t=time.perf_counter()
    out=model(ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.cuda.synchronize();prefill_ms=(time.perf_counter()-t)*1000
    prefill_alloc=torch.cuda.memory_allocated();t=time.perf_counter();cache.freeze(decisions)
    torch.cuda.synchronize();freeze_ms=(time.perf_counter()-t)*1000
    frozen_alloc=torch.cuda.memory_allocated();frozen=cache.report()
    ms=[];logits=[];generated=[]
    for token in tokens:
        torch.cuda.synchronize();t=time.perf_counter()
        out=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,attention_mask=None if model.config.model_type=='glm4_moe_lite' else {'full_attention':None},use_cache=True,logits_to_keep=1)
        torch.cuda.synchronize();ms.append((time.perf_counter()-t)*1000)
        logits.append(out.logits[:,-1].float().cpu());generated.append(int(out.logits[:,-1].argmax()))
    result=dict(kind=kind,prefill_ms=prefill_ms,freeze_ms=freeze_ms,decode_step_ms=ms,
        generated_ids=generated,model_start_allocated=start_alloc,prefill_allocated=prefill_alloc,
        after_freeze_allocated=frozen_alloc,peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),frozen=frozen,final=cache.report(),
        layer_receipts={i:layer.tli_receipt for i,layer in enumerate(cache.layers)})
    del out,cache
    return result,torch.stack(logits)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--length',type=int,default=4352);ap.add_argument('--steps',type=int,default=65)
    ap.add_argument('--modes',nargs='+',default=['keep','near','far','both']);ap.add_argument('--cycles',type=int,default=1)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    os.environ.update(TLI_FINE_GATHER='0',TLI_OPERATOR_MODE='off',TLI_CUDA_GRAPH='0',TLI_BATCH_INDEXER='legacy')
    torch.manual_seed(42);torch.set_num_threads(6)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    treatment=tli_args();assert register_patch(model,treatment,resolve_treatment_snapshot(treatment))==model.config.num_hidden_layers
    ids=tokenizer('Remember hidden number 57291. Explain the evidence precisely. '*args.length,add_special_tokens=False).input_ids[:args.length]
    tensor=torch.tensor([ids],device='cuda');geometry=RegionGeometry(max_new_tokens=args.steps+1)
    tokens=tokenizer(' evidence and context support the answer. '*args.steps,add_special_tokens=False).input_ids[:args.steps]
    result=dict(schema='nfa-paged-paired-v1',status='pending independent verification',args=vars(args),
        input_ids=ids,decode_replay_tokens=tokens,scope='fixed known-input replay, not generated answer throughput',
        device=torch.cuda.get_device_name(),torch=torch.__version__,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'sparse_attn').rglob('*.py')},records=[])
    destination=Path(args.out);destination.parent.mkdir(parents=True,exist_ok=True)
    for mode in args.modes:
        if mode not in ['keep','near','far','both']:raise ValueError(mode)
        decisions={i:dict(skip_near=mode in ['near','both'],skip_far=mode in ['far','both']) for i in range(model.config.num_hidden_layers)}
        # Full-state forcing is a diagnostic, not input-calibrated task quality.
        reference=None
        for kind in ['ring','paged']:
            warm,_=trial(model,tensor,tokens[:min(2,len(tokens))],decisions,kind,geometry)
            warm.update(mode=mode,phase='warmup_excluded_from_comparison')
            result['records'].append(warm)
        for cycle in range(args.cycles):
            for kind in ['ring','paged','paged','ring']:
                record,logits=trial(model,tensor,tokens,decisions,kind,geometry)
                if reference is None:reference=logits
                record.update(mode=mode,cycle=cycle,phase='measured',max_logit_difference=float((logits-reference).abs().max()),
                              same_argmax=bool(torch.equal(logits.argmax(-1),reference.argmax(-1))))
                result['records'].append(record);destination.write_text(json.dumps(result,indent=2)+'\n')
                print(mode,kind,record['max_logit_difference'],record['same_argmax'],sum(record['decode_step_ms'])/len(tokens),flush=True)
    print('DONE pending independent verification',flush=True)
if __name__=='__main__':main()
