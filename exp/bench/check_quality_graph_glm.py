"""Real-weight GLM Graph/eager token smoke; not a benchmark quality score."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from sparse_attn.ops.quality_graph import from_dynamic
from exp.bench.quality_pairing import glm_request_arms


def run(args_cli):
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
    arms = glm_request_arms(args_cli.repeats, args_cli.device_input_reference)
    for request_index, (policy, input_mode) in enumerate(arms):
        os.environ['TLI_CUDA_GRAPH']=policy
        if input_mode is not None:
            os.environ['TLI_DEVICE_DECODE_INPUTS'] = input_mode
            os.environ['TLI_REUSE_WARMUP_STREAM'] = input_mode
        if args_cli.measure:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            request_start = time.perf_counter()
        with torch.inference_mode(),eager_decoding_request_guard():
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            past=out.past_key_values;first=out.logits[:,-1].argmax(-1).item();del out
            if args_cli.measure:
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - request_start) * 1000
            generated=from_dynamic(model,past,first,args_cli.max_gen,set())
            if generated is None:
                generated=[first]
                step_ms = []
                for _ in range(args_cli.max_gen - 1):
                    if args_cli.measure:
                        step_start = time.perf_counter()
                    out=model(input_ids=torch.tensor([[generated[-1]]],device='cuda'),
                              past_key_values=past,use_cache=True,logits_to_keep=1)
                    generated.append(out.logits[:,-1].argmax(-1).item())
                    past=out.past_key_values;del out
                    if args_cli.measure:
                        step_ms.append((time.perf_counter() - step_start) * 1000)
                if args_cli.measure:
                    model._quality_graph_execution['timing'] = dict(
                        scope='dynamic eager decode; inject, forward, argmax and readback',
                        step_wall_ms=step_ms, replay_gpu_ms=[], replay_gpu_mean_ms=None,
                        decode_steps=len(step_ms), generated_decode_tokens=len(step_ms),
                        tpot_wall_ms=sum(step_ms)/len(step_ms) if step_ms else None)
            records.append(dict(request_index=request_index,pair=request_index//2,policy=policy,input_mode=input_mode,tokens=generated,
                execution=dict(model._quality_graph_execution),
                normalization_capture_routes=[getattr(x.self_attn.indexer, "_graph_normalization_route", None) for x in model.model.layers],
                normalization_stats_capture_routes=[getattr(x.self_attn.indexer, "_graph_normalization_stats_route", None) for x in model.model.layers],
                ranking_capture_routes=[getattr(x.self_attn.indexer, "_graph_ranking_route", None) for x in model.model.layers],
                ranking_kth_capture_routes=[getattr(x.self_attn.indexer, "_graph_ranking_kth_route", None) for x in model.model.layers],
                ranking_gather_capture_routes=[getattr(x.self_attn.indexer, "_graph_ranking_gather_route", None) for x in model.model.layers],
                frozen=[x.self_attn.indexer._dyn_frozen for x in model.model.layers],
                skip_far=[x.self_attn.indexer.skip_far for x in model.model.layers]))
        if args_cli.measure:
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - request_start) * 1000
            records[-1]['request_timing'] = dict(
                prefill_wall_ms=prefill_ms, total_wall_ms=total_ms,
                output_tokens_per_second=len(generated)*1000/total_ms,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope='after model/tokenizer/input preparation through prefill, conversion, capture, generation and final error check; excludes post-request cache cleanup')
        Path(args_cli.output).write_text(json.dumps(dict(status='pending',records=records),indent=2)+'\n')
        del past
        gc.collect();torch.cuda.empty_cache()
    result=dict(scope='real-weight GLM B1 16384-token deterministic requests, not RULER accuracy',
                comparison='same_graph_device_inputs' if args_cli.device_input_reference else 'eager_graph',
                measurement_enabled=args_cli.measure, records=records,
                tokens_equal=all(r['tokens']==records[0]['tokens'] for r in records))
    Path(args_cli.output).write_text(json.dumps(result,indent=2)+'\n')
    assert result['tokens_equal']
    for record in records:
        if record['policy']=='1':
            assert record['execution']['replay_steps']==args_cli.max_gen-1
            assert record['execution']['captures']==1
            assert all(record['frozen'])
        if args_cli.device_input_reference:
            receipt = record['execution']
            assert receipt['actual'] == 'graph' and receipt['fallback_reason'] is None
            assert receipt['input_update_route'] == ('device_row_fused' if record['input_mode'] == '1' else 'host_reference')
            assert receipt['input_update_fallback'] is None
        assert record['skip_far']==records[0]['skip_far']
    if args_cli.device_input_reference:
        candidates = [r['execution'] for r in records if r['input_mode'] == '1']
        assert len({r['warmup_stream'] for r in candidates}) == 1
        assert all(r['warmup_stream_reused'] for r in candidates[1:])
    print(json.dumps(result),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-gen', type=int, default=5)
    parser.add_argument('--device-input-reference', action='store_true',
                        help='Compare host/device inputs on the same strict Graph; pair stream reuse with device inputs')
    args_cli = parser.parse_args()
    if args_cli.repeats < 1 or args_cli.max_gen < 2:
        parser.error('repeats >= 1 and max-gen >= 2 required')
    if args_cli.measure:
        os.environ['TLI_QUALITY_TIMING'] = '1'
        from request_gpu_monitor import GPU0Monitor
        monitor = GPU0Monitor(Path(args_cli.output).with_suffix('.gpu.json'))
    else:
        monitor = contextlib.nullcontext()
    with monitor:
        run(args_cli)


if __name__=='__main__':main()
