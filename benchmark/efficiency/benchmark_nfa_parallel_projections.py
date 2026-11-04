"""iter112：原形状独立投影的跨 stream 并行逐位与延迟对拍。"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from triton.testing import do_bench

from benchmark.efficiency.benchmark_nfa_kv_projection import (
    _capture_and_time, _mismatch)


def _load_weights(model_path, layer_idx, names):
    root = Path(model_path)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    prefix = f'model.layers.{layer_idx}'
    result = {}
    for name in names:
        key = (f'{prefix}.self_attn.{name}.weight' if name.endswith('_proj')
               and name != 'down_proj' else f'{prefix}.mlp.{name}.weight')
        # gate/up 属于 MLP；q/k/v/o 属于 attention。
        if name in ('gate_proj', 'up_proj', 'down_proj'):
            key = f'{prefix}.mlp.{name}.weight'
        with safe_open(str(root / index['weight_map'][key]), framework='pt') as shard:
            result[name] = shard.get_tensor(key).cuda()
    return result


def _attention_reference(x, weights, outputs):
    outputs['q'] = F.linear(x, weights['q_proj'])
    outputs['k'] = F.linear(x, weights['k_proj'])
    outputs['v'] = F.linear(x, weights['v_proj'])


def _attention_parallel(x, weights, outputs, side_stream):
    current = torch.cuda.current_stream()
    side_stream.wait_stream(current)
    with torch.cuda.stream(side_stream):
        outputs['k'] = F.linear(x, weights['k_proj'])
        outputs['v'] = F.linear(x, weights['v_proj'])
    outputs['q'] = F.linear(x, weights['q_proj'])
    current.wait_stream(side_stream)
    outputs['k'].record_stream(current)
    outputs['v'].record_stream(current)


def _mlp_reference(x, weights, outputs):
    outputs['gate'] = F.linear(x, weights['gate_proj'])
    outputs['up'] = F.linear(x, weights['up_proj'])


def _mlp_parallel(x, weights, outputs, side_stream):
    current = torch.cuda.current_stream()
    side_stream.wait_stream(current)
    with torch.cuda.stream(side_stream):
        outputs['up'] = F.linear(x, weights['up_proj'])
    outputs['gate'] = F.linear(x, weights['gate_proj'])
    current.wait_stream(side_stream)
    outputs['up'].record_stream(current)


def _check(outputs_ref, outputs_new, names):
    result = {}
    for name in names:
        exact = torch.equal(outputs_ref[name], outputs_new[name])
        result[f'{name}_bit_exact'] = exact
        if not exact:
            result[f'{name}_mismatch'] = _mismatch(
                outputs_ref[name], outputs_new[name])
    result['bit_exact'] = all(result[f'{name}_bit_exact'] for name in names)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    side_stream = torch.cuda.Stream()
    all_ok = True
    cases = []

    for layer_idx in (0, 1, 17, 35):
        weights = _load_weights(
            args.model_path, layer_idx,
            ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'))
        for seed in range(4):
            generator = torch.Generator(device='cuda').manual_seed(20261112 + seed)
            x = torch.randn(
                (1, 1, 4096), generator=generator, device='cuda',
                dtype=torch.float32).to(torch.bfloat16)
            if seed == 3:
                x = (x.float() * 0.02).to(torch.bfloat16)
            reference, candidate = {}, {}
            _attention_reference(x, weights, reference)
            _attention_parallel(x, weights, candidate, side_stream)
            _mlp_reference(x, weights, reference)
            _mlp_parallel(x, weights, candidate, side_stream)
            torch.cuda.synchronize()
            result = {'layer': layer_idx, 'seed': seed}
            result['attention'] = _check(reference, candidate, ('q', 'k', 'v'))
            result['mlp'] = _check(reference, candidate, ('gate', 'up'))
            result['bit_exact'] = result['attention']['bit_exact'] and result['mlp']['bit_exact']
            all_ok &= result['bit_exact']
            cases.append(result)
            print('CASE ' + json.dumps(result), flush=True)
        del weights

    summary = {
        'model_path': args.model_path,
        'shape': {'x': [1, 1, 4096], 'q': [4096, 4096],
                  'k_v': [1024, 4096], 'gate_up': [11008, 4096]},
        'correctness_cases': cases,
        'all_bit_exact': all_ok,
    }
    if args.correctness_only or not all_ok:
        print('SUMMARY ' + json.dumps(summary), flush=True)
        return 0 if all_ok else 1

    weights = _load_weights(
        args.model_path, 0, ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'))
    x = torch.randn((1, 1, 4096), device='cuda', dtype=torch.bfloat16)
    performance = {}
    for name, reference_fn, candidate_fn, output_names in (
        ('attention', _attention_reference, _attention_parallel, ('q', 'k', 'v')),
        ('mlp', _mlp_reference, _mlp_parallel, ('gate', 'up')),
    ):
        ref_out, new_out = {}, {}
        reference = lambda: reference_fn(x, weights, ref_out)
        candidate = lambda: candidate_fn(x, weights, new_out, side_stream)
        reference()
        candidate()
        torch.cuda.synchronize()
        eager_ref = do_bench(reference, warmup=25, rep=100)
        eager_new = do_bench(candidate, warmup=25, rep=100)
        ref_snapshot, graph_ref, graph_ref_samples = _capture_and_time(reference, ref_out)
        new_snapshot, graph_new, graph_new_samples = _capture_and_time(candidate, new_out)
        graph_exact = all(torch.equal(ref_snapshot[key], new_snapshot[key])
                          for key in output_names)
        performance[name] = {
            'eager_reference_ms': eager_ref,
            'eager_candidate_ms': eager_new,
            'eager_speedup': eager_ref / eager_new,
            'graph_reference_ms': graph_ref,
            'graph_candidate_ms': graph_new,
            'graph_speedup': graph_ref / graph_new,
            'graph_reference_samples_ms': graph_ref_samples,
            'graph_candidate_samples_ms': graph_new_samples,
            'graph_bit_exact': graph_exact,
            'estimated_36_layer_graph_saving_ms': 36 * (graph_ref - graph_new),
        }
        all_ok &= graph_exact
    summary['performance'] = performance
    summary['all_bit_exact'] = all_ok
    summary['estimated_total_36_layer_graph_saving_ms'] = sum(
        value['estimated_36_layer_graph_saving_ms'] for value in performance.values())
    print('SUMMARY ' + json.dumps(summary), flush=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
