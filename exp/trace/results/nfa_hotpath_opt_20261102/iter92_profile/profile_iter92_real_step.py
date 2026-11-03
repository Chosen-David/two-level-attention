"""在当前真实生产 runner 的单个稳态 step 上采集结构化 torch profiler。"""
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

REPO = '/home/wangyuanshuo02/wt-nfa-iter89-rope'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'benchmark', 'efficiency'))
import benchmark_nfa_end_to_end as bench


def main():
    args = bench.parse_args()
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt, _ = bench._load_input(args, tokenizer)
    model = bench._load_model(args, args.max_new_tokens)
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)

    original_events = bench._events
    prof = None
    event_calls = 0

    def profiled_events():
        nonlocal prof, event_calls
        event_calls += 1
        if event_calls == args.warmup_steps + 1:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
        elif prof is not None and event_calls == args.warmup_steps + 2:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
        return original_events()

    bench._events = profiled_events
    fa2_scope = (bench._fa2_no_packed_position_ids()
                 if args.method == 'fa2' else bench.nullcontext())
    with torch.inference_mode(), fa2_scope as fa2_state:
        if args.method == 'nfa':
            result = bench._run_nfa(args, model, prompt, args.max_new_tokens, eos_ids)
        else:
            result = bench._run_fa2(
                args, model, prompt, args.max_new_tokens, eos_ids, fa2_state)
    if prof is not None and event_calls <= args.warmup_steps + 1:
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)

    rows = []
    for event in prof.key_averages():
        device_ms = event.self_device_time_total / 1000.0
        if device_ms > 0:
            rows.append({
                'name': event.key,
                'calls': event.count,
                'device_ms_total': device_ms,
                'cpu_ms_total': event.cpu_time_total / 1000.0,
            })
    rows.sort(key=lambda row: row['device_ms_total'], reverse=True)
    summary = {
        'scope': 'one production decode step with torch profiler; not clean TPOT',
        'method': args.method,
        'execution': args.execution,
        'profiled_step_zero_based': args.warmup_steps,
        'profiled_receipt_tpot_ms_not_verdict': result['steady_tpot_ms_median'],
        'self_device_ms_total': sum(row['device_ms_total'] for row in rows),
        'kernels': rows,
    }
    profile_path = args.output + '.profile.json'
    with open(profile_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    prof.export_chrome_trace(args.output + '.trace.json')
    print(json.dumps({**summary, 'kernels': rows[:45]}, indent=2))


if __name__ == '__main__':
    main()
