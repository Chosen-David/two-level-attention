"""Correctness-only layer-local probe with identical Q/K/V for both caches.

Only the ring output advances model hidden states. Paged outputs are diagnostic,
so differences in later layer inputs cannot contaminate selection comparisons.
No timing, warmup, ABBA, throughput or allocator peak measurement is performed.
"""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sparse_attn.patches import register_patch
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops import frozen_region_tli as ring, paged_region_tli as paged
from sparse_attn.ops.frozen_region_cache import FrozenRegionCache,RegionGeometry
from sparse_attn.ops.paged_region_cache import PagedRegionCache
from sparse_attn.ops.paged_region_kernels import paged_attention
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--length',type=int,default=4352);ap.add_argument('--steps',type=int,default=4)
    ap.add_argument('--modes',nargs='+',default=['keep','near','far','both']);a=ap.parse_args()
    if a.length<4352 or a.steps<1:raise ValueError('need nonempty Far and positive replay')
    destination=Path(a.out);destination.mkdir(parents=True,exist_ok=False)
    os.environ.update(TLI_FINE_GATHER='0',TLI_OPERATOR_MODE='off',TLI_CUDA_GRAPH='0',TLI_BATCH_INDEXER='legacy')
    torch.manual_seed(42);torch.set_num_threads(6)
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    treatment=ring.tli_args();assert register_patch(model,treatment,resolve_treatment_snapshot(treatment))==model.config.num_hidden_layers
    ids=tok('Remember hidden number 57291. Explain the evidence precisely. '*a.length,add_special_tokens=False).input_ids[:a.length]
    tokens=tok(' evidence and context support the answer. '*a.steps,add_special_tokens=False).input_ids[:a.steps]
    result=dict(status='pending_independent_validation',scope=__doc__,args=vars(a),input_ids=ids,replay_ids=tokens,
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=[])
    original_decode=ring.decode
    for mode in a.modes:
        if mode not in ['keep','near','far','both']:raise ValueError(mode)
        for layer in model.model.layers:layer.self_attn.indexer.clear();layer.self_attn.indexer.skip_far=False
        geometry=RegionGeometry(max_new_tokens=a.steps+1)
        rc=FrozenRegionCache(model.config.num_hidden_layers,geometry);rc.tli_regions=True
        pc=PagedRegionCache(model.config.num_hidden_layers,geometry)
        original_update=rc.update
        def dual_update(k,v,layer_idx,cache_kwargs=None):
            pc.update(k,v,layer_idx,cache_kwargs)
            return original_update(k,v,layer_idx,cache_kwargs)
        def probe(module,q,k,v,cache,position):
            reference=original_decode(module,q,k,v,cache,position)
            li=module.layer_idx;pl=pc.layers[li];rl=rc.layers[li]
            rk,rv,positions=rl.gather();logical=pl.logical_length
            rm,_=ring.region_mask(module.indexer,q.transpose(1,2).contiguous(),rk.transpose(1,2).contiguous(),positions,logical,module.scaling)
            pm,receipt=paged.region_selection(module.indexer,q,pl,module.scaling)
            densemask=torch.zeros_like(pm);densemask[...,positions]=rm
            dims=module.indexer._subspace_indices(q.device);bs=module.indexer.args.tia_block_size
            projection=pl.project_visible(dims,bs);expected=torch.zeros_like(projection)
            expected[:,positions]=rk.index_select(-1,dims).transpose(1,2)
            actual=paged_attention(pl,q,pm,module.scaling,ring_arithmetic=True)
            same_mask=paged_attention(pl,q,densemask,module.scaling,ring_arithmetic=True)
            record=dict(mode=mode,step=logical-a.length,layer=li,projection_equal=bool(torch.equal(projection,expected)),
                        mask_mismatches=int((pm!=densemask).sum()),selected_ring=rm.sum(-1).flatten().tolist(),selected_paged=pm.sum(-1).flatten().tolist(),
                        max_output_error=float((actual.float()-reference.float()).abs().max()),
                        same_mask_max_output_error=float((same_mask.float()-reference.float()).abs().max()),
                        reference_abs_max=float(reference.abs().max()),finite=bool(torch.isfinite(actual).all()),
                        reference_finite=bool(torch.isfinite(reference).all()),same_mask_finite=bool(torch.isfinite(same_mask).all()),receipt=receipt)
            worst_case={'keep':(33,2),'near':(33,4),'far':(34,4),'both':(30,3)}[mode]
            if (logical==a.length+1 and li in [0,8,26]) or (li,logical-a.length)==worst_case:
                filename=f'{mode}_layer{li}_step{logical-a.length}.pt'
                torch.save(dict(q=q.cpu(),k=rk.cpu(),v=rv.cpu(),positions=positions,ring_mask=rm.cpu(),paged_mask=pm.cpu(),ring_output=reference.cpu(),paged_output=actual.cpu(),scale=module.scaling),destination/filename)
                record['tensor_file']=filename;record['tensor_sha256']=hashlib.sha256((destination/filename).read_bytes()).hexdigest()
            result['records'].append(record)
            if not all(record[x] for x in ['finite','reference_finite','same_mask_finite']):raise RuntimeError('nonfinite attention output')
            return reference
        with eager_decoding_request_guard(),patch.object(rc,'update',dual_update),patch.object(ring,'decode',probe):
            output=model(torch.tensor([ids],device='cuda'),past_key_values=rc,use_cache=True,logits_to_keep=1)
            if not torch.isfinite(output.logits).all():raise RuntimeError('nonfinite prefill logits')
            decisions={i:dict(skip_near=mode in ['near','both'],skip_far=mode in ['far','both']) for i in range(model.config.num_hidden_layers)}
            rc.freeze(decisions);pc.freeze(decisions)
            for step,token in enumerate(tokens,1):
                output=model(torch.tensor([[token]],device='cuda'),past_key_values=rc,attention_mask={'full_attention':None},use_cache=True,logits_to_keep=1)
                finite=bool(torch.isfinite(output.logits).all())
                result.setdefault('model_logits_finite',[]).append(dict(mode=mode,step=step,finite=finite))
                (destination/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
                if not finite:raise RuntimeError('nonfinite model logits')
        result.setdefault('cache_reports',{})[mode]=dict(ring=rc.report(),paged=pc.report())
        (destination/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
        print(mode,'complete',flush=True)
        del output,rc,pc
    print('DONE correctness only; pending independent review',flush=True)
if __name__=='__main__':main()
