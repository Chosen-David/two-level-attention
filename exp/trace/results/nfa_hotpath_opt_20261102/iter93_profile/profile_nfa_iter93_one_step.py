"""对 iter93 生产 benchmark 的单个 GraphRunner.step 做独立 torch profiler。"""
import runpy
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from sparse_attn.engine.graph_runner import GraphRunner

execution = sys.argv[sys.argv.index('--execution') + 1]
method = sys.argv[sys.argv.index('--method') + 1]
trace_path = f'/tmp/iter93_one_step_{method}_{execution}.trace.json'
original_step = GraphRunner.step
state = {'step': 0}


def profiled_step(*args, **kwargs):
    step = state['step']
    state['step'] += 1
    if step != 3:
        return original_step(*args, **kwargs)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = original_step(*args, **kwargs)
    prof.export_chrome_trace(trace_path)
    print(prof.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=90), flush=True)
    print('PROFILE_TRACE=' + trace_path, flush=True)
    return result


GraphRunner.step = profiled_step
runpy.run_path(
    '/home/wangyuanshuo02/wt-nfa-iter89-rope/benchmark/efficiency/benchmark_nfa_end_to_end.py',
    run_name='__main__')
