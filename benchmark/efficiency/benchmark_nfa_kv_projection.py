"""iter111：B1 decode 独立 K/V 与融合 KV skinny GEMM 的逐位和延迟对拍。"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from triton.testing import do_bench


def _load_kv_weights(model_path, layer_idx):
    """只读取指定层，避免为单算子基准加载完整 8B 模型。"""
    root = Path(model_path)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    prefix = f'model.layers.{layer_idx}.self_attn'
    tensors = []
    for name in ('k_proj', 'v_proj'):
        key = f'{prefix}.{name}.weight'
        with safe_open(str(root / index['weight_map'][key]), framework='pt') as shard:
            tensors.append(shard.get_tensor(key).cuda())
    return tensors


def _reference(x, k_weight, v_weight, outputs):
    outputs['k'] = F.linear(x, k_weight)
    outputs['v'] = F.linear(x, v_weight)


def _candidate(x, kv_weight, outputs):
    kv = F.linear(x, kv_weight)
    outputs['k'], outputs['v'] = kv.split(1024, dim=-1)


def _capture_and_time(launch, outputs, inner=16, replays=100, reps=5):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    snapshot = {name: value.clone() for name, value in outputs.items()}
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (inner * replays))
    return snapshot, statistics.median(samples), samples


def _mismatch(reference, candidate):
    diff = reference != candidate
    return {
        'count': int(diff.sum()),
        'max_abs': float((reference.float() - candidate.float()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(20261111)

    all_ok = True
    cases = []
    for layer_idx in (0, 1, 17, 35):
        k_weight, v_weight = _load_kv_weights(args.model_path, layer_idx)
        kv_weight = torch.cat((k_weight, v_weight), dim=0).contiguous()
        for seed in range(4):
            generator = torch.Generator(device='cuda').manual_seed(20261111 + seed)
            x = torch.randn(
                (1, 1, 4096), generator=generator, device='cuda',
                dtype=torch.float32).to(torch.bfloat16)
            if seed == 3:
                x = (x.float() * 0.02).to(torch.bfloat16)
            reference, candidate = {}, {}
            _reference(x, k_weight, v_weight, reference)
            _candidate(x, kv_weight, candidate)
            torch.cuda.synchronize()
            result = {
                'layer': layer_idx, 'seed': seed,
                'k_bit_exact': torch.equal(reference['k'], candidate['k']),
                'v_bit_exact': torch.equal(reference['v'], candidate['v']),
            }
            if not result['k_bit_exact']:
                result['k_mismatch'] = _mismatch(reference['k'], candidate['k'])
            if not result['v_bit_exact']:
                result['v_mismatch'] = _mismatch(reference['v'], candidate['v'])
            result['bit_exact'] = result['k_bit_exact'] and result['v_bit_exact']
            all_ok &= result['bit_exact']
            cases.append(result)
            print('CASE ' + json.dumps(result), flush=True)
        del k_weight, v_weight, kv_weight

    summary = {
        'model_path': args.model_path,
        'shape': {'x': [1, 1, 4096], 'k': [1024, 4096],
                  'v': [1024, 4096], 'kv': [2048, 4096]},
        'dtype': 'bfloat16',
        'bf16_reduced_precision_reduction':
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'correctness_cases': cases,
        'all_bit_exact': all_ok,
    }
    if args.correctness_only or not all_ok:
        print('SUMMARY ' + json.dumps(summary), flush=True)
        return 0 if all_ok else 1

    k_weight, v_weight = _load_kv_weights(args.model_path, 0)
    kv_weight = torch.cat((k_weight, v_weight), dim=0).contiguous()
    x = torch.randn((1, 1, 4096), device='cuda', dtype=torch.bfloat16)
    ref_out, new_out = {}, {}
    reference = lambda: _reference(x, k_weight, v_weight, ref_out)
    candidate = lambda: _candidate(x, kv_weight, new_out)
    reference()
    candidate()
    torch.cuda.synchronize()
    eager_ref = do_bench(reference, warmup=25, rep=100)
    eager_new = do_bench(candidate, warmup=25, rep=100)
    ref_snapshot, graph_ref, graph_ref_samples = _capture_and_time(reference, ref_out)
    new_snapshot, graph_new, graph_new_samples = _capture_and_time(candidate, new_out)
    graph_exact = all(torch.equal(ref_snapshot[name], new_snapshot[name])
                      for name in ('k', 'v'))
    summary['performance'] = {
        'eager_reference_ms': eager_ref,
        'eager_candidate_ms': eager_new,
        'eager_speedup': eager_ref / eager_new,
        'graph_reference_ms': graph_ref,
        'graph_candidate_ms': graph_new,
        'graph_speedup': graph_ref / graph_new,
        'graph_reference_samples_ms': graph_ref_samples,
        'graph_candidate_samples_ms': graph_new_samples,
        'graph_bit_exact': graph_exact,
        'graph_candidate_threshold_ms_for_0p2_step': 0.008208,
        'estimated_36_layer_graph_saving_ms': 36 * (graph_ref - graph_new),
    }
    summary['all_bit_exact'] = all_ok and graph_exact
    print('SUMMARY ' + json.dumps(summary), flush=True)
    return 0 if summary['all_bit_exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
