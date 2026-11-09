"""Measure fused-LSE calibration as incremental time over identical dense prefill."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from benchmark.efficiency import benchmark_nfa_end_to_end as bench
from benchmark.efficiency.benchmark_nfa_vs_fa2 import (
    GpuMonitor, _model_receipt, _smi)
from sparse_attn.engine.frozen_runner import (
    _region_present_from_length, build_frozen_cache, prefill_frozen_request)
from sparse_attn.policies.frozen_gate import (
    GateConfig, _profile_summary, _stream_tail_profile, decide_regions)
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.policies.static_gate import StaticGateDecision


@contextmanager
def _timed_profile_regions(validate_legacy=False):
    """计时并证明真实调用 fused kernel； warmup 可选逐层旧实现对拍。"""
    import sparse_attn.engine.frozen_runner as frozen_runner
    import sparse_attn.ops.frozen_gate_triton as fused_triton

    original = frozen_runner.profile_regions
    original_fused = fused_triton.fused_region_profile
    samples = []
    fused_calls = []
    legacy = {'profiles': [], 'max_abs_error': 0.0}

    def fused_spy(*fn_args, **fn_kwargs):
        fused_calls.append(1)
        return original_fused(*fn_args, **fn_kwargs)

    def timed(*fn_args, **fn_kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        profile = original(*fn_args, **fn_kwargs)
        end.record()
        samples.append((start, end, time.perf_counter() - host_start))
        if validate_legacy:
            q, k, geometry = fn_args[:3]
            rows = fn_kwargs.get('rows', 256)
            chunk = fn_kwargs.get('chunk', 16)
            scale = fn_kwargs.get('scale')
            if scale is None:
                scale = q.shape[-1] ** -.5
            reference = _profile_summary(_stream_tail_profile(
                q, k, geometry, rows, chunk, scale))
            for region in ('near', 'far'):
                for statistic in ('mean', 'p95', 'max'):
                    error = abs(profile[region][statistic]
                                - reference[region][statistic])
                    legacy['max_abs_error'] = max(
                        legacy['max_abs_error'], error)
                    if not math.isclose(
                            profile[region][statistic], reference[region][statistic],
                            rel_tol=0.0, abs_tol=5e-6):
                        raise RuntimeError(
                            f'legacy profile mismatch at call {len(samples) - 1}: '
                            f'{region}.{statistic} fused={profile[region][statistic]} '
                            f'reference={reference[region][statistic]}')
            legacy['profiles'].append(reference)
        return profile

    frozen_runner.profile_regions = timed
    fused_triton.fused_region_profile = fused_spy
    try:
        yield samples, fused_calls, legacy
    finally:
        frozen_runner.profile_regions = original
        fused_triton.fused_region_profile = original_fused


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument(
        '--data-root', default='/home/wangyuanshuo02/sparse-bench/third_party/'
                               'KVCache-Factory/data/RULER')
    parser.add_argument('--task', default='niah_multikey_1')
    parser.add_argument('--input-provenance', default='native-ruler')
    parser.add_argument('--context-length', type=int, required=True,
                        choices=(32768, 65536, 131072))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _load_model(args):
    load_args = SimpleNamespace(
        model_path=args.model_path, context_length=args.context_length,
        method='nfa', subspace='tail', l2_subspace='same')
    return bench._load_model(load_args, 8)


def _static_gate(model):
    layers = model.config.num_hidden_layers
    return StaticGateDecision(
        model_type=model.config.model_type, num_layers=layers,
        skip_far=(False,) * layers, skip_near=(False,) * layers,
        gate={'mode': 'full', 'role': 'identical dense prefill without calibrator'},
        source='measure_prefill_lse_calibrator')


def _run_prefill(model, prompt, geometry, calibrated, validate_legacy=False):
    cache = build_frozen_cache(model, geometry, prompt.numel(), max_requests=1)
    static_gate = None if calibrated else _static_gate(model)
    gate = GateConfig(epsilon=.05, rows=256, statistic='mean', mode='dual')
    torch.cuda.synchronize()
    start = time.perf_counter()
    with _timed_profile_regions(validate_legacy) as (
            samples, fused_calls, legacy):
        output, audit = prefill_frozen_request(
            model, prompt.unsqueeze(0), gate, geometry,
            static_gate=static_gate, cache=cache, request_id=0)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    token = int(output.logits[:, -1].argmax(-1)[0])
    device_ms = [begin.elapsed_time(end) for begin, end, _ in samples]
    result = {
        'wall_seconds': wall_seconds,
        'next_token': token,
        'calibrator': {
            'calls': len(samples),
            'fused_kernel_calls': len(fused_calls),
            'device_ms_per_layer': device_ms,
            'device_seconds_total': sum(device_ms) / 1000.0,
            'host_seconds_total': sum(sample[2] for sample in samples),
            'legacy_validation': None,
        },
        'decisions': {
            'far': [layer for layer, decision in audit['decisions'].items()
                    if decision['skip_far']],
            'near': [layer for layer, decision in audit['decisions'].items()
                     if decision['skip_near']],
        },
    }
    expected_calls = model.config.num_hidden_layers if calibrated else 0
    if len(samples) != expected_calls or len(fused_calls) != expected_calls:
        raise RuntimeError(
            f'calibrator/fused calls {len(samples)}/{len(fused_calls)} '
            f'!= {expected_calls}')
    if validate_legacy:
        if not calibrated or len(legacy['profiles']) != expected_calls:
            raise RuntimeError('legacy validation requires every calibrated layer')
        present_near, present_far = _region_present_from_length(
            prompt.numel(), geometry)
        for layer_idx, reference in enumerate(legacy['profiles']):
            expected = decide_regions(
                reference, gate, present_near=present_near, present_far=present_far)
            actual = audit['decisions'][layer_idx]
            if (actual['skip_near'] != expected['skip_near']
                    or actual['skip_far'] != expected['skip_far']):
                raise RuntimeError(
                    f'legacy gate mismatch at layer {layer_idx}: '
                    f'fused={actual} reference={expected}')
        result['calibrator']['legacy_validation'] = {
            'status': 'PASS', 'layers': expected_calls,
            'max_abs_error': legacy['max_abs_error'],
            'abs_tolerance': 5e-6, 'gate_decisions_identity': True,
        }
    del output, audit, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _summary(measurements):
    dense = [row['dense']['wall_seconds'] for row in measurements]
    fused = [row['fused']['wall_seconds'] for row in measurements]
    paired_delta = [row['delta_seconds'] for row in measurements]
    dense_median = statistics.median(dense)
    fused_median = statistics.median(fused)
    calibrator_device = statistics.median([
        row['fused']['calibrator']['device_seconds_total']
        for row in measurements])
    calibrator_host = statistics.median([
        row['fused']['calibrator']['host_seconds_total']
        for row in measurements])
    return {
        'repeat': len(measurements),
        'dense_prefill_seconds': dense,
        'fused_prefill_seconds': fused,
        'paired_increment_seconds': paired_delta,
        'dense_prefill_median_seconds': dense_median,
        'fused_prefill_median_seconds': fused_median,
        'increment_over_dense_median_seconds': fused_median - dense_median,
        'increment_over_dense_ratio': (fused_median - dense_median) / dense_median,
        'paired_increment_median_seconds': statistics.median(paired_delta),
        'calibrator_device_median_seconds': calibrator_device,
        'calibrator_host_median_seconds': calibrator_host,
        'calibrator_device_over_dense_prefill': calibrator_device / dense_median,
        'calibrator_host_over_dense_prefill': calibrator_host / dense_median,
        'all_next_token_identity': all(row['token_identity'] for row in measurements),
    }


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError('--repeat must be positive')
    torch.cuda.set_device(args.gpu_id)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    monitor = GpuMonitor(args.gpu_id, args.output + '.hygiene.jsonl')
    monitor.start()
    monitor.register(os.getpid())
    try:
        tokenizer = bench.AutoTokenizer.from_pretrained(args.model_path)
        load_args = SimpleNamespace(
            data_root=args.data_root, task=args.task,
            context_length=args.context_length, sample_index=args.sample_index,
            input_provenance=args.input_provenance)
        prompt, input_receipt = bench._load_input(load_args, tokenizer)
        model = _load_model(args)
        geometry = RegionGeometry(
            sink=4, near=4096, window=128, page=64, max_new_tokens=8)

        # Compile Triton and warm both paths before any measured request.  The
        # fused warmup additionally compares every real Qwen layer and gate to
        # the legacy implementation; none of this work enters measured walls.
        _run_prefill(model, prompt, geometry, calibrated=False)
        warmup_fused = _run_prefill(
            model, prompt, geometry, calibrated=True, validate_legacy=True)
        measurements = []
        for repeat in range(args.repeat):
            order = (('dense', False), ('fused', True)) if repeat % 2 == 0 else (
                ('fused', True), ('dense', False))
            row = {}
            for name, calibrated in order:
                print(f'PREFILL {args.context_length} repeat={repeat} arm={name}',
                      flush=True)
                row[name] = _run_prefill(model, prompt, geometry, calibrated)
            row['delta_seconds'] = (
                row['fused']['wall_seconds'] - row['dense']['wall_seconds'])
            row['token_identity'] = (
                row['fused']['next_token'] == row['dense']['next_token'])
            if not row['token_identity']:
                raise RuntimeError('fused calibration changed dense prefill next token')
            measurements.append(row)
            print(json.dumps(_summary(measurements), indent=2), flush=True)
        if monitor.pollution:
            raise RuntimeError('GPU hygiene pollution detected; refusing performance verdict')
        result = {
            'schema': 'nfa-fused-lse-prefill-increment-v1',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'scope': ('Identical Qwen3 SDPA dense prefill with and without fused-LSE '
                      'calibration; independent requests, alternating order, one warmup '
                      'per arm; no decode time included'),
            'arguments': vars(args),
            'source': {
                'head': subprocess.check_output(
                    ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
                'diff_sha256': hashlib.sha256(subprocess.check_output(
                    ['git', '-C', REPO, 'diff', '--binary'])).hexdigest(),
            },
            'model': _model_receipt(args.model_path),
            'input': input_receipt,
            'gpu': {
                'id': args.gpu_id,
                'name': _smi(args.gpu_id, 'name'),
                'driver_version': _smi(args.gpu_id, 'driver_version'),
            },
            'software': {
                'torch': torch.__version__,
                'transformers': __import__('transformers').__version__,
                'triton': __import__('triton').__version__,
            },
            'gpu_hygiene': {
                'status': 'clean', 'pollution_samples': len(monitor.pollution),
                'log': args.output + '.hygiene.jsonl',
            },
            'correctness': {
                'warmup_fused': warmup_fused['calibrator'],
                'warmup_next_token': warmup_fused['next_token'],
                'scope': ('Real Qwen per-layer near/far summaries and gate decisions '
                          'against legacy chunked softmax; fused kernel call count '
                          'checked on every calibrated prefill'),
            },
            'measurements': measurements,
            'summary': _summary(measurements),
        }
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        print('saved ->', args.output, flush=True)
    finally:
        monitor.unregister(os.getpid())
        monitor.stop()


if __name__ == '__main__':
    main()
