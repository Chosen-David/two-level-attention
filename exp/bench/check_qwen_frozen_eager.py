"""真实 Qwen3 BF16 frozen eager 配对正确性与性能验收。

每个进程只加载一次模型并处理一个长度；performance 使用 ABBA 顺序。profiler
单独运行，不把 profiler 开销混入 TPOT。默认不宣称等价或加速，全部原始指标
写入 JSON，由判决方依据生成一致性和实测数据下结论。
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from benchmark.RULER.yarn_receipt import resolve_yarn_config

from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops.frozen_region_runtime import (
    complete_frozen_request, prefill_frozen_request)
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches import register_patch


STATES = {
    'keep': dict(skip_near=False, skip_far=False),
    'far': dict(skip_near=False, skip_far=True),
    'near': dict(skip_near=True, skip_far=False),
    'both': dict(skip_near=True, skip_far=True),
}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _memory(device):
    if device.type != 'cuda':
        return {}
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def _work_summary(audit):
    receipts = audit.get('tli_receipts', {})
    selected = []
    l1 = l2 = 0
    physical = True
    for receipt in receipts.values():
        if not receipt:
            continue
        selected.extend(receipt.get('selected_per_kv_head', []))
        l1 += receipt.get('l1_calls', 0)
        l2 += receipt.get('l2_topk_calls', 0)
        physical = physical and receipt.get('physical_arena', False)
    return dict(
        decode_receipts=len(receipts), l1_calls=l1, l2_topk_calls=l2,
        selected_kv_total=sum(selected),
        selected_kv_min=min(selected) if selected else None,
        selected_kv_max=max(selected) if selected else None,
        all_physical=bool(receipts) and physical,
    )


def _public_result(result, include_audit=False):
    keep = dict(
        arm=result['arm'], state=result['state'], generated_ids=result['generated_ids'],
        generated_sha256=hashlib.sha256(
            json.dumps(result['generated_ids']).encode()).hexdigest(),
        decisions=result['audit']['decisions'], gate_counts=result['audit']['gate_counts'],
        prefill_full_kv=result['audit']['prefill_full_kv'],
        frozen_kv=result['audit']['frozen_kv'], final_kv=result['audit']['final_kv'],
        work=_work_summary(result['audit']), timings=result['timings'],
        memory=result['memory'])
    if include_audit:
        keep['audit'] = result['audit']
    return keep


@torch.inference_mode()
def run_request(model, ids, max_new, arm, state, device, capture=False,
                forced_tokens=None, profile_path=None):
    reference = arm == 'reference'
    os.environ['TLI_FROZEN_EAGER_REFERENCE'] = '1' if reference else '0'
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    initial_memory = _memory(device)
    start = time.perf_counter()
    decision = STATES[state]
    with (patch('sparse_attn.ops.frozen_region_runtime.decide_regions',
                return_value=decision) if state != 'dual' else contextlib.nullcontext()):
        out, audit = prefill_frozen_request(
            model, ids, GateConfig(mode='dual'), RegionGeometry(max_new_tokens=max_new))
    cache = out.past_key_values
    _sync(device)
    prefill_done = time.perf_counter()
    prefill_memory = _memory(device)

    captured_layers = []
    captured_logits = []
    handles = []

    def capture_output(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured_layers.append(value.detach().float().cpu())

    if capture:
        for layer in model.model.layers:
            handles.append(layer.self_attn.register_forward_hook(capture_output))
    generated = []
    profiler = None
    if profile_path:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True)
        profiler.__enter__()
    decode_start = time.perf_counter()
    try:
        for step in range(max_new):
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(token))
            if capture:
                captured_logits.append(out.logits.detach().float().cpu())
            if step+1 == max_new:
                break
            next_token = (token if forced_tokens is None else
                          torch.tensor([[forced_tokens[step]]], device=device))
            out = model(next_token, past_key_values=cache,
                        attention_mask={'full_attention': None},
                        use_cache=True, logits_to_keep=1)
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
        for handle in handles:
            handle.remove()
    _sync(device)
    decode_done = time.perf_counter()
    audit = complete_frozen_request(model, cache, len(generated), audit)
    final_memory = _memory(device)
    decode_steps = max_new-1
    result = dict(
        arm=arm, state=state, generated_ids=generated, audit=audit,
        timings=dict(
            prefill_seconds=prefill_done-start,
            decode_seconds=decode_done-decode_start,
            total_seconds=decode_done-start,
            decode_steps=decode_steps,
            tpot_ms=(decode_done-decode_start)*1000/decode_steps if decode_steps else None),
        memory=dict(initial=initial_memory, prefill=prefill_memory, final=final_memory),
        captured_layers=captured_layers, captured_logits=captured_logits)
    if profiler is not None:
        profiler.export_chrome_trace(str(profile_path))
        events = profiler.key_averages()
        result['profiler'] = dict(
            trace=str(profile_path),
            top_ops=[dict(key=e.key, count=e.count, self_cpu_ms=e.self_cpu_time_total/1000,
                          self_cuda_ms=e.self_device_time_total/1000)
                     for e in sorted(events, key=lambda e: e.self_device_time_total,
                                     reverse=True)[:40]],
            transfer_ops=[dict(key=e.key, count=e.count,
                               self_cuda_ms=e.self_device_time_total/1000)
                          for e in events if 'memcpy' in e.key.lower()],
            sync_ops=[dict(key=e.key, count=e.count)
                      for e in events if any(name in e.key.lower() for name in
                          ('synchronize', '_local_scalar_dense', 'aten::item'))])
    del out, cache
    return result


def _max_delta(left, right):
    if not left and not right:
        return 0.
    if len(left) != len(right):
        return float('inf')
    return max((a-b).abs().max().item() for a, b in zip(left, right))


def compare_pair(reference, physical):
    layer_delta = _max_delta(reference['captured_layers'], physical['captured_layers'])
    logit_delta = _max_delta(reference['captured_logits'], physical['captured_logits'])
    return dict(
        decisions_equal=reference['audit']['decisions'] == physical['audit']['decisions'],
        generated_equal=reference['generated_ids'] == physical['generated_ids'],
        reference_generated=reference['generated_ids'],
        physical_generated=physical['generated_ids'],
        layer_output_max_abs=layer_delta,
        logits_max_abs=logit_delta,
        decode_layer_tensors=len(reference['captured_layers']),
        decode_logit_tensors=len(reference['captured_logits']))


def performance_summary(runs):
    summary = {}
    for arm in ('reference', 'physical', 'gates_off'):
        selected = [r for r in runs if r['arm'] == arm]
        if not selected:
            continue
        summary[arm] = dict(
            runs=len(selected),
            median_tpot_ms=statistics.median(r['timings']['tpot_ms'] for r in selected),
            median_prefill_seconds=statistics.median(
                r['timings']['prefill_seconds'] for r in selected),
            median_final_allocated_bytes=statistics.median(
                r['memory']['final']['allocated_bytes'] for r in selected),
            median_peak_allocated_bytes=statistics.median(
                r['memory']['final']['peak_allocated_bytes'] for r in selected))
    if 'reference' in summary and 'physical' in summary:
        summary['tpot_speedup_reference_over_physical'] = (
            summary['reference']['median_tpot_ms']/summary['physical']['median_tpot_ms'])
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--task', default='niah_single_1')
    parser.add_argument('--context-length', type=int, required=True)
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--max-new', type=int, default=8)
    parser.add_argument('--states', default='keep,far,near,both')
    parser.add_argument('--mode', choices=['correctness', 'performance', 'all'], default='all')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.max_new < 2 or args.reps < 1 or args.row < 0:
        raise ValueError('max-new>=2, reps>=1 and row>=0 required')
    for name, value in dict(TLI_OPERATOR_MODE='off', TLI_FINE_GATHER='0',
                            TLI_SPARSE_PREFILL='0', TLI_DECODE_DENSE='0',
                            TLI_CUDA_GRAPH='off').items():
        os.environ[name] = value
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    rope_scaling = None
    if args.context_length > 40960:
        _, rope_scaling = resolve_yarn_config(
            True, None, args.context_length, {65536: 2., 131072: 4.}, 40960)
        config.rope_scaling = rope_scaling
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map=device,
        attn_implementation='sdpa').eval()
    tli = tli_args()
    if register_patch(model, tli, resolve_treatment_snapshot(tli)) != model.config.num_hidden_layers:
        raise ValueError('incomplete TLI registration')
    model._frozen_tli = True
    source = Path(args.data_root)
    row = json.loads((source/str(args.context_length)/(args.task+'.jsonl'))
                     .read_text().splitlines()[args.row])
    ids = tokenizer(row['input'], return_tensors='pt', truncation=False).input_ids.to(device)
    protocol = dict(model=args.model, task=args.task, row=args.row,
                    context_length=args.context_length, input_tokens=ids.shape[-1],
                    max_new=args.max_new, dtype='bfloat16', device=str(device),
                    rope_scaling=rope_scaling,
                    torch=torch.__version__, tli_args=vars(tli))
    (outdir/'protocol.json').write_text(json.dumps(protocol, indent=2, default=str)+'\n')
    report = dict(protocol=protocol, correctness=[], performance=[])
    states = args.states.split(',')
    if args.mode in ('correctness', 'all'):
        for state in states:
            if state == 'dual':
                decision_state = 'dual'
            elif state in STATES:
                decision_state = state
            else:
                raise ValueError(f'unknown state {state}')
            reference = run_request(model, ids, args.max_new, 'reference',
                                    decision_state, device, capture=True)
            physical = run_request(model, ids, args.max_new, 'physical',
                                   decision_state, device, capture=True,
                                   forced_tokens=reference['generated_ids'])
            comparison = compare_pair(reference, physical)
            comparison['state'] = state
            report['correctness'].append(dict(
                comparison=comparison,
                reference=_public_result(reference, include_audit=True),
                physical=_public_result(physical, include_audit=True)))
            (outdir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print('CORRECTNESS', state, json.dumps(comparison), flush=True)
            del reference, physical
    if args.mode in ('performance', 'all'):
        runs = []
        for rep in range(args.reps):
            for arm in ('reference', 'physical', 'physical', 'reference'):
                result = run_request(model, ids, args.max_new, arm, 'dual', device)
                runs.append(result)
                report['performance'].append(_public_result(result))
                (outdir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print('PERFORMANCE', rep, arm, result['timings'],
                      result['audit']['gate_counts'], flush=True)
                del result
            if args.profile:
                profiled = run_request(
                    model, ids, args.max_new, 'physical', 'dual', device,
                    profile_path=outdir/f'profile_rep{rep}.json')
                report['profile'] = _public_result(profiled)
                report['profile']['profiler'] = profiled['profiler']
                del profiled
            # gates-off 独立臂：full 决策通过逐层 patch 固定为 keep，仍执行 TLI。
            off = run_request(model, ids, args.max_new, 'physical', 'keep', device)
            off['arm'] = 'gates_off'
            runs.append(off)
            report['performance'].append(_public_result(off))
            (outdir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            del off
        report['performance_summary'] = performance_summary(runs)
        (outdir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('SUMMARY', json.dumps(report['performance_summary']), flush=True)


if __name__ == '__main__':
    main()
