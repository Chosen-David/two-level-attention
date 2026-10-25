"""Recompute summaries and verify final receipt/source bindings; no GPU claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'agent_doc/results/far_skip_three_rounds_20261025'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RESULTS/name).read_text())


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='checkout of the measured round-3 code when current code has advanced')
    ROOT = parser.parse_args().source_root.resolve()
    summary = {'scope': 'single-layer synthetic CPU FP32, one thread; not CUDA Graph or model throughput',
               'parent_commit': '3e55156bf375a2f925172b3e6ec1da74a383f17e',
               'round1_commit': 'd7fab93f2142c07acd2cb222aa191c7de0221d14',
               'round2_commit': 'ed308eac7c2da6c0c2814db6cff149ab8f48189e',
               'cuda_executed': False, 'independent_review': False}
    outputs, error = 0, 0.
    for file in ['final_eager_cpu.json', 'final_eager_holdout_cpu.json']:
        d = read(file)
        for name, expected in d['source_sha256']['optimized'].items():
            assert sha(ROOT/'sparse_attn/ops'/name) == expected, name
        for name, expected in d['source_sha256']['shared_harness'].items():
            assert sha(ROOT/name) == expected, name
        rows = []
        for length in d['config']['lengths']:
            receipts = [r for r in d['measurements'] if r['length'] == length]
            med = {n: statistics.median([v for r in receipts for v in r['runs'][n]['wall_ms']])
                   for n in ['v1', 'optimized']}
            assert d['medians_ms'][str(length)] == med
            rows.append(dict(length=length, near=d['config']['near_len'], parent_ms=med['v1'],
                             final_ms=med['optimized'], ratio=med['v1']/med['optimized'],
                             cold_ms={n: statistics.median([r['runs'][n]['first_decode_wall_ms'] for r in receipts])
                                      for n in ['v1', 'optimized']},
                             kv_bytes=receipts[0]['runs']['optimized']['kv_storage_bytes']))
        summary[file] = rows
        outputs += sum(r['outputs_checked'] for r in d['measurements'])
        error = max(error, max(r['max_abs_error'] for r in d['measurements']))
    for file in ['final_fixed_cpu.json', 'final_fixed_holdout_cpu.json', 'final_fixed_vs_eager_cpu.json']:
        d = read(file)
        for name, expected in d['source_sha256']['candidate'].items():
            assert sha(ROOT/name) == expected, name
        assert sha(ROOT/'exp/bench/far_skip_graph_compare.py') == d['source_sha256']['harness']
        med = {n: statistics.median([v for r in d['measurements'] for v in r[n]['wall_ms']])
               for n in ['baseline', 'candidate']}
        assert med == d['medians_ms']
        summary[file] = dict(length=d['config']['length'], near=d['config']['near_len'],
                             median_ms=med, ratio=med['baseline']/med['candidate'],
                             setup_ms={n: statistics.median([r[n]['setup_ms'] for r in d['measurements']])
                                       for n in ['baseline', 'candidate']})
        outputs += sum(r['outputs_checked'] for r in d['measurements'])
        error = max(error, max(r['max_abs_error'] for r in d['measurements']))
    summary['final_timing_outputs_checked'] = outputs
    summary['max_absolute_error'] = error
    e = read('final_eager_profile_cpu.json')
    f = read('final_fixed_profile_cpu.json')['measurements'][0]
    summary['profile_cpu'] = {
        'eager_positive_allocation_bytes': {n: e[n]['positive_self_cpu_allocated_bytes'] for n in ['v1', 'optimized']},
        'eager_repeat_interleave': {n: e[n]['operator_counts'].get('aten::repeat_interleave', 0) for n in ['v1', 'optimized']},
        'fixed_positive_allocation_bytes': {n: f[n]['positive_self_cpu_allocation_bytes'] for n in ['baseline', 'candidate']},
        'note': 'positive self allocation traffic, not net/peak memory; counts are not CUDA launches'}
    log = (RESULTS/'final_tests.txt').read_text()
    counts = re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed', log)
    assert counts, 'test log lacks completed test counts'
    summary['tests'] = dict(zip(['passed', 'skipped_cuda', 'subtests_passed'], map(int, counts.groups())))
    (RESULTS/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    paths = list((ROOT/'sparse_attn/ops').glob('far_skip*.py')) + [ROOT/p for p in [
        'sparse_attn/ops/tli_static_cache.py', 'sparse_attn/patches/qwen3_attn_patch.py',
        'tests/test_far_skip_graph.py', 'tests/test_far_skip_compact.py', 'tests/test_far_skip_optimized.py',
        'exp/bench/e152v4_graph_latency.py', 'exp/bench/far_skip_graph_compare.py',
        'exp/bench/far_skip_compare.py', 'exp/bench/far_skip_decode.py',
        'exp/bench/summarize_far_skip_three_rounds.py']]
    validation = dict(status='CPU subset verified; CUDA unrun', tests=summary['tests'],
                      final_timing_outputs_checked=outputs, final_source_bindings_match=True,
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                      receipt_sha256={p.name: sha(p) for p in RESULTS.iterdir()
                                      if p.suffix in ['.json', '.txt'] and p.name != 'validation.json'})
    (RESULTS/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
