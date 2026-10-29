"""Same-input, same-budget Qwen compact consumers through the quality entry."""
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
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
from sparse_attn.ops.quality_graph import from_dynamic
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from request_gpu_monitor import GPU0Monitor


def run(cli):
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    args = parser.parse_args([
        '--method', 'tli', '--tia_block_size', '64', '--tia_level1_topk', '128',
        '--tia_level2_topk', '1024', '--tia_level2_cmp_ratio', '4',
        '--tli_near_pages', '32', '--tli_near_tokens', 'off', '--tli_near_len', '4096',
        '--tli_subspace', 'tail', '--tli_sink_tokens', '4', '--tli_swa_size', '128',
        '--tli_enable_layer_skip', 'false', '--tli_enable_kmeans', 'false',
        '--tli_dyn_dgate', '--tli_far_skip_backend', 'compact', '--tli_far_skip_cache', 'evict'])
    path = '/home/wangyuanshuo02/models/huggingface.co/Qwen/Qwen3-32B'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    resolve_far_skip_modes(args, config=model.config)
    patched = register_patch(model, args, resolve_treatment_snapshot(args))
    assert patched == model.config.num_hidden_layers
    row = json.loads(Path(cli.input).read_text().splitlines()[0])
    ids = tokenizer(row['input'], truncation=False, return_tensors='pt').input_ids.to('cuda')
    records = []
    for cycle in range(cli.cycles):
        for mode in (('off', 'serial', 'overlap') if cycle % 2 == 0 else ('overlap', 'serial', 'off')):
            os.environ['TLI_OPERATOR_MODE'] = mode
            os.environ['TLI_OPERATOR_BACKEND'] = 'simt'
            os.environ['TLI_CUDA_GRAPH'] = 'auto'
            os.environ['TLI_QUALITY_TIMING'] = '1' if cli.measure else '0'
            with torch.inference_mode(), eager_decoding_request_guard():
                if cli.measure:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                first = out.logits[:, -1].argmax(-1).item()
                del out
                if cli.measure:
                    torch.cuda.synchronize(); prefill_ms = (time.perf_counter()-start)*1000
                generated = from_dynamic(model, past, first, cli.max_gen, set())
                execution = dict(model._quality_graph_execution)
                record = dict(cycle=cycle, mode=mode, input_index=row['index'],
                              input_tokens=ids.shape[-1], tokens=generated, execution=execution)
                if cli.measure:
                    torch.cuda.synchronize(); total_ms = (time.perf_counter()-start)*1000
                    record['request_timing'] = dict(prefill_wall_ms=prefill_ms,
                        total_wall_ms=total_ms, output_tokens_per_second=len(generated)*1000/total_ms,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        scope='after model/input preparation through prefill, conversion, capture, sampling/readback and final error check')
                records.append(record)
                Path(cli.output).write_text(json.dumps(dict(status='pending', records=records), indent=2))
                assert execution['actual'] == 'graph' and execution['captures'] == 1
                assert execution['replay_steps'] == cli.max_gen-1 and execution['fallback_reason'] is None
                rings = [r for r in execution['capture_layers'] if r['skip_far']]
                assert rings, 'must exercise an actual naturally frozen ring layer'
                route = 'ring_sdpa' if mode == 'off' else f'ring_{mode}_simt'
                assert all(r['attention_route'] == route and r['ring_capacity'] < ids.shape[-1] for r in rings)
                assert generated == records[0]['tokens'], 'consumer token mismatch'
                assert [r['skip_far'] for r in execution['capture_layers']] == [
                    r['skip_far'] for r in records[0]['execution']['capture_layers']]
            del past
            gc.collect(); torch.cuda.empty_cache()
    Path(cli.output).write_text(json.dumps(dict(status='pending_independent_verification',
        scope='Qwen B1 explicit compact configuration, not paper accuracy or GLM support',
        records=records, tokens_equal=True), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-gen', type=int, default=16); p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--measure', action='store_true'); cli = p.parse_args()
    if cli.max_gen < 2 or cli.cycles < 1: p.error('max-gen >= 2 and cycles >= 1 required')
    with GPU0Monitor(Path(cli.output).with_suffix('.gpu.json')):
        run(cli)
