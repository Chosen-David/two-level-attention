"""Paired model correctness replay, without timing or performance measurement."""
import argparse,gc,hashlib,json,os,subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.frozen_region_cache import FrozenRegionCache,RegionGeometry
from sparse_attn.ops.paged_region_cache import PagedRegionCache
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--length',type=int,default=4352);ap.add_argument('--steps',type=int,default=65)
    a=ap.parse_args();destination=Path(a.out);destination.mkdir(parents=True,exist_ok=False)
    if a.length<4352 or a.steps<1:raise ValueError('nonempty Far and positive replay required')
    os.environ.update(TLI_FINE_GATHER='0',TLI_OPERATOR_MODE='off',TLI_CUDA_GRAPH='0',TLI_BATCH_INDEXER='legacy')
    torch.manual_seed(42);torch.set_num_threads(6)
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    treatment=tli_args();assert register_patch(model,treatment,resolve_treatment_snapshot(treatment))==model.config.num_hidden_layers
    ids=tokenizer('Remember hidden number 57291. Explain the evidence precisely. '*a.length,add_special_tokens=False).input_ids[:a.length]
    tokens=tokenizer(' evidence and context support the answer. '*a.steps,add_special_tokens=False).input_ids[:a.steps]
    result=dict(status='pending_independent_validation',scope='fixed replay correctness; not free generation quality, speed or memory performance',
                pid=os.getpid(),device=torch.cuda.get_device_name(),
                args=vars(a),input_ids=ids,replay_ids=tokens,commit=subprocess.check_output(
                    ['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip(),records=[])
    for mode in ['keep','near','far','both']:
        reference=None
        for kind,cls in [('ring',FrozenRegionCache),('paged',PagedRegionCache)]:
            for layer in model.model.layers:layer.self_attn.indexer.clear();layer.self_attn.indexer.skip_far=False
            cache=cls(model.config.num_hidden_layers,RegionGeometry(max_new_tokens=a.steps+1));cache.tli_regions=True
            record=dict(mode=mode,kind=kind,steps=[]);outputs=[]
            with eager_decoding_request_guard():
                out=model(torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
                if not torch.isfinite(out.logits).all():raise RuntimeError('nonfinite prefill')
                decisions={i:dict(skip_near=mode in ['near','both'],skip_far=mode in ['far','both']) for i in range(model.config.num_hidden_layers)}
                torch.cuda.synchronize();record['active_cuda_bytes_before_freeze']=torch.cuda.memory_allocated()
                record['prefill_cache']=cache.report()
                cache.freeze(decisions);gc.collect();torch.cuda.synchronize()
                record['active_cuda_bytes_after_freeze']=torch.cuda.memory_allocated()
                record['active_cuda_bytes_reclaimed']=(record['active_cuda_bytes_before_freeze']-
                    record['active_cuda_bytes_after_freeze'])
                record['frozen_cache']=cache.report()
                if kind=='paged':
                    kv_delta=record['prefill_cache']['kv_bytes']-record['frozen_cache']['kv_bytes']
                    # active allocation 必须与实际整页释放完全一致，而非理论 mask 字节。
                    assert record['active_cuda_bytes_reclaimed']==kv_delta,(mode,kv_delta,record['active_cuda_bytes_reclaimed'])
                for step,token in enumerate(tokens,1):
                    out=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
                    logits=out.logits[:,-1].float().cpu();finite=bool(torch.isfinite(logits).all());outputs.append(logits)
                    summary=dict(step=step,finite=finite,argmax=int(logits.argmax()),logical_lengths=[l.logical_length for l in cache.layers])
                    if kind=='paged':summary['page_lifecycle']=[dict(live_pages=len(l.pages),reclaimed=l.reclaimed_pages,reused_ids=l.reused_page_ids,aging_copy_bytes=l.aging_copy_bytes) for l in cache.layers]
                    record['steps'].append(summary)
                    if not finite:
                        torch.save(outputs,destination/f'{mode}_{kind}_failed.pt')
                        raise RuntimeError('nonfinite decode; outputs retained')
                record['final_cache']=cache.report()
                record['final_receipts']=[l.tli_receipt for l in cache.layers]
                if kind=='paged' and mode=='both':
                    assert all(r['scorer']=='protected_only' and r['l1_calls']==0 and
                               r['l2_calls']==0 and r['selection_topk_calls']==0
                               for r in record['final_receipts'])
            outputs=torch.stack(outputs);tensor_path=destination/f'{mode}_{kind}.pt';torch.save(outputs,tensor_path)
            record.update(logits_file=tensor_path.name,logits_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest())
            if reference is None:reference=outputs
            else:record.update(max_logit_error=float((outputs-reference).abs().max()),argmax_mismatch_steps=int((outputs.argmax(-1)!=reference.argmax(-1)).sum()))
            result['records'].append(record);(destination/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
            print(mode,kind,'complete',flush=True)
            del out,cache
    print('DONE correctness-only; pending independent validation',flush=True)
if __name__=='__main__':main()
