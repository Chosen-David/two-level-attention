"""iter94：Qwen3-8B B1 decode QKV 拼接投影的数值、存储与延迟探针。"""
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sparse_attn.patches.qwen3_attn_patch import prepare_qkv_fused_weight  # noqa: E402


def separate(x, qw, kw, vw):
    return F.linear(x, qw), F.linear(x, kw), F.linear(x, vw)


def fused(x, fw):
    y = F.linear(x, fw)
    return y[..., :4096], y[..., 4096:5120], y[..., 5120:]


def graph_time(fn, warmup=10, reps=5, inner=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / inner)
    return statistics.median(times), static_out


def eager_time(fn, warmup=20, reps=5, inner=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / inner)
    return statistics.median(times)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(20261104)
    module = torch.nn.Module()
    module.q_proj = torch.nn.Linear(4096, 4096, bias=False)
    module.k_proj = torch.nn.Linear(4096, 1024, bias=False)
    module.v_proj = torch.nn.Linear(4096, 1024, bias=False)
    module = module.to(device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, 1, 4096, device='cuda', dtype=torch.bfloat16)
    allocated_before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        ref = separate(x, module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)
        assert prepare_qkv_fused_weight(module)
        fw = module._nfa_qkv_weight
        assert module.q_proj.weight.data_ptr() == fw.data_ptr()
        assert module.k_proj.weight.data_ptr() == fw[4096:].data_ptr()
        assert module.v_proj.weight.data_ptr() == fw[5120:].data_ptr()
        allocated_after = torch.cuda.memory_allocated()
        fallback = separate(x, module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)
        print('shared_storage', True, 'allocated_delta_bytes', allocated_after - allocated_before)
        print('fallback_bit_exact', all(torch.equal(a, b) for a, b in zip(ref, fallback)))
        got = fused(x, fw)
        for name, a, b in zip(('q', 'k', 'v'), ref, got):
            print(name, 'fused_bit_exact', torch.equal(a, b),
                  'different', torch.count_nonzero(a != b).item(),
                  'max_abs', (a.float() - b.float()).abs().max().item())
        te_sep = eager_time(lambda: separate(
            x, module.q_proj.weight, module.k_proj.weight, module.v_proj.weight))
        te_fused = eager_time(lambda: fused(x, fw))
        tg_sep, _ = graph_time(lambda: separate(
            x, module.q_proj.weight, module.k_proj.weight, module.v_proj.weight))
        tg_fused, _ = graph_time(lambda: fused(x, fw))
    print('eager_ms', te_sep, te_fused, 'ratio', te_sep / te_fused)
    print('graph_ms', tg_sep, tg_fused, 'ratio', tg_sep / tg_fused)


if __name__ == '__main__':
    main()
