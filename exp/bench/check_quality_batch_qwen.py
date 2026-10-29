"""Real no-gate Qwen batch requests against independent single-row Graphs."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.patches import register_patch
from sparse_attn.baseline_runtime import BatchCtx
from sparse_attn.ops.quality_graph import static_decode
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
from sparse_attn.ops.tli_static_cache import TLIStaticCache
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from request_gpu_monitor import GPU0Monitor


def run(cli):
    inherited_env={k:v for k,v in os.environ.items() if k.startswith("TLI_")}
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(); add_sparse_attn_args(parser)
    args = parser.parse_args([
        '--method','tli','--tia_block_size','64','--tia_level1_topk','128',
        '--tia_level2_topk','1024','--tia_level2_cmp_ratio','4',
        '--tli_near_pages','32','--tli_near_tokens','off','--tli_near_len','4096',
        '--tli_subspace','tail','--tli_sink_tokens','4','--tli_swa_size','128',
        '--tli_enable_layer_skip','false','--tli_enable_kmeans','false',
        '--tli_far_skip_backend','legacy','--tli_far_skip_cache','retain'])
    os.environ.update(TLI_OPERATOR_MODE='serial',TLI_OPERATOR_BACKEND='simt',
        TLI_OPERATOR_SELECTED_STAGES='1',TLI_SPARSE_SELECTION='1',
        TLI_LIVE_NORMALIZATION='1',TLI_SPARSE_STATS='owner',TLI_GRAPH_INC='1',
        TLI_L2_BYPASS='0',TLI_CUDA_GRAPH='auto',
        TLI_QUALITY_TIMING='1' if cli.measure else '0')
    model_path='/home/wangyuanshuo02/models/huggingface.co/Qwen/Qwen3-32B'
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    resolve_far_skip_modes(args,config=model.config)
    patched=register_patch(model,args,resolve_treatment_snapshot(args))
    assert patched==model.config.num_hidden_layers
    assert all(not m.indexer.dyn_dgate for m in model.modules() if hasattr(m,'indexer'))
    rows=[json.loads(line) for line in Path(cli.input).read_text().splitlines() if line.strip()]
    if len(rows)<cli.batch: raise ValueError('input needs distinct original rows for requested batch')
    rows=rows[:cli.batch]
    ids=[tokenizer(row['input'],return_tensors='pt').input_ids.to('cuda') for row in rows]
    lengths=[x.shape[-1] for x in ids]
    if min(lengths)<9217 or max(lengths)+cli.max_gen>=65536:
        raise ValueError('unsupported live-length/capacity geometry')
    records=[]
    output=Path(cli.output)
    output.with_suffix('.configuration.json').write_text(json.dumps(dict(
        resolved_args=vars(args),inherited_tli_environment=inherited_env,model=model_path,input_indices=[r['index'] for r in rows],
        lengths=lengths,eos_policy='empty EOS diagnostic; fixed generation count',
        scope='explicit no-gate Qwen; unchanged 64/128/1024 budget, not ring-gate or GLM B>1'),indent=2))

    def request(which, batch_mode, row_reference=False, eager_reference=False, legacy_graph=False):
        os.environ['TLI_CUDA_GRAPH']='off' if eager_reference else 'auto'
        os.environ['TLI_OPERATOR_MODE']='off' if (eager_reference or legacy_graph) else 'serial'
        os.environ['TLI_EXACT_BATCH_REFERENCE']='1' if row_reference else '0'
        os.environ['TLI_BATCH_INDEXER']=batch_mode
        batch=len(which); lens=[lengths[b] for b in which]
        capacity=max(lens)+cli.max_gen
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        cache=TLIStaticCache(model.config,capacity,batch);first=[];prefilled=[]
        prefill_ms=conversion_ms=0.0
        with torch.inference_mode(),eager_decoding_request_guard():
            for slot,b in enumerate(which):
                torch.cuda.synchronize();t=time.perf_counter()
                out=model(input_ids=ids[b],use_cache=True,logits_to_keep=1)
                first.append(int(out.logits[0,-1].argmax()))
                torch.cuda.synchronize();prefill_ms+=(time.perf_counter()-t)*1000
                prefilled.append(out.past_key_values)
                del out
            from sparse_attn.ops.static_batch_prefill import consume_prefill_rows
            torch.cuda.synchronize();t=time.perf_counter()
            consume_prefill_rows(cache,prefilled,lens)
            prefilled.clear()
            torch.cuda.synchronize();conversion_ms=(time.perf_counter()-t)*1000
            ctx=BatchCtx(batch,list(lens),[0]*batch,[True]*batch,0)
            tokens=static_decode(model,cache,[[x] for x in first],lens,set(),cli.max_gen,ctx)
            torch.cuda.synchronize();total_ms=(time.perf_counter()-start)*1000
            execution=dict(model._quality_graph_execution)
            rec=dict(cycle=cycle,arm=current_arm,rows=which,batch=batch,batch_indexer=batch_mode,tokens=tokens,
                     execution=execution,environment={k:v for k,v in os.environ.items() if k.startswith('TLI_')})
            if cli.measure:
                rec['request_timing']=dict(total_wall_ms=total_ms,prefill_wall_ms=prefill_ms,
                    cache_conversion_wall_ms=conversion_ms,
                    output_tokens_per_second=sum(map(len,tokens))*1000/total_ms,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
            records.append(rec)
            output.write_text(json.dumps(dict(status='pending',records=records),indent=2))
            if eager_reference:
                assert execution['actual']=='staged_static' and execution['captures']==0
                assert execution['replay_steps']==0 and execution['fallback_reason'] is None
                if batch>1:
                    assert len(execution['eager_layers'])==model.config.num_hidden_layers
                    assert all(r['attention_route']=='batch_eager_tli' for r in execution['eager_layers'])
            else:
                assert execution['actual']=='graph' and execution['captures']==1
                assert execution['replay_steps']==cli.max_gen-1 and execution['fallback_reason'] is None
                assert len(execution['capture_layers'])==model.config.num_hidden_layers
                if legacy_graph:
                    assert all(r['attention_route']=='legacy_graph_eager_tli' for r in execution['capture_layers'])
                if batch_mode=='exact':
                    expected_route=('linear_batch_reference_simt' if row_reference else
                        'exact_batch_ordered_eager' if os.environ.get('TLI_EXACT_ATTENTION','split')=='ordered'
                        else 'exact_batch_simt')
                    assert all(r['attention_route']==expected_route for r in execution['capture_layers'])
        del cache
        return tokens

    def run_reference():
        if cli.eager_batch_reference:
            return request(list(range(cli.batch)), 'legacy', eager_reference=True)
        if cli.legacy_batch_reference:
            return request(list(range(cli.batch)), 'legacy', legacy_graph=True)
        if cli.same_batch_reference:
            return request(list(range(cli.batch)), 'exact', True)
        return [request([b], 'legacy')[0] for b in range(cli.batch)]

    pairs=[]
    comparison=('same_batch_eager' if cli.eager_batch_reference else
                'same_batch_legacy_graph' if cli.legacy_batch_reference else
                'same_batch_rows' if cli.same_batch_reference else 'independent_B1')
    for cycle in range(cli.cycles):
        pair={'cycle':cycle, 'order':(['reference','candidate'] if cycle%2==0
                                    else ['candidate','reference'])}
        values={}
        for current_arm in pair['order']:
            start_record=len(records)
            if current_arm=='reference':
                values[current_arm]=run_reference()
            else:
                native=cli.legacy_graph_candidate or cli.batch==1
                values[current_arm]=request(list(range(cli.batch)),
                    'legacy' if native else 'exact', legacy_graph=native)
            pair[current_arm+'_records']=list(range(start_record,len(records)))
        pair['tokens_equal']=values['reference']==values['candidate']
        pairs.append(pair)
        output.write_text(json.dumps(dict(status='pending_independent_verification',
            tokens_equal=all(p['tokens_equal'] for p in pairs),pairs=pairs,
            candidate='legacy_graph' if cli.legacy_graph_candidate or cli.batch==1 else 'exact_batch',
            comparison=comparison,records=records),indent=2))
        assert pair['tokens_equal'],'real batch tokens differ from the recorded reference control'



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,choices=[1,2,4,8],required=True)
    p.add_argument('--max-gen',type=int,default=16);p.add_argument('--measure',action='store_true')
    ref=p.add_mutually_exclusive_group()
    ref.add_argument('--same-batch-reference',action='store_true')
    ref.add_argument('--eager-batch-reference',action='store_true')
    ref.add_argument('--legacy-batch-reference',action='store_true',
        help='Same-batch operator-off native Graph reference for fused Graph comparison')
    p.add_argument('--cycles',type=int,default=1,help='Paired cycles with alternating arm order')
    p.add_argument('--legacy-graph-candidate',action='store_true',
        help='Compare operator-off legacy Graph against same-batch eager, checking per-layer receipts')
    p.add_argument('--trace-dir')
    cli=p.parse_args()
    if cli.legacy_graph_candidate and (not cli.eager_batch_reference or cli.batch<2):
        p.error('legacy Graph candidate requires same-batch eager reference and B>=2')
    if cli.trace_dir and (cli.measure or not cli.eager_batch_reference):
        p.error('trace requires eager-batch-reference and forbids timing')
    if cli.same_batch_reference and cli.batch<2:p.error('same-batch reference needs B>=2')
    if cli.cycles<1:p.error('cycles must be >=1')
    if cli.max_gen<2:p.error('max-gen must be >=2')
    with GPU0Monitor(Path(cli.output).with_suffix('.gpu.json')):
        if cli.trace_dir:
            from batch_decode_trace import record_batch_decode_trace
            with record_batch_decode_trace(cli.trace_dir):run(cli)
        else:run(cli)
