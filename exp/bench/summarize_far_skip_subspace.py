"""Recompute the two subspace iterations and validate their final source binding."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'agent_doc/results/far_skip_subspace_20261025'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT)
    root = p.parse_args().source_root.resolve()
    files = ['final_vs_round3_cpu.json', 'final_quantized_cpu.json', 'final_holdout_cpu.json',
             'final_short_cpu.json', 'final_bfloat16_cpu.json', 's2_all_fit_vs_s1_cpu.json']
    summary = dict(scope='paired same-width single-layer CPU; fixed shapes are not CUDA Graph timings',
                   baseline_round3='f3b0347fd5b52fd6145a8db1dde407c9157a0b8b',
                   subspace_round1='b5d724fa540e835f47bea164fea58f8f1dac0140',
                   cuda_executed=False, pretrained_quality_evaluated=False, independent_review=False)
    checked, error = 0, 0.
    for name in files:
        d = json.loads((RESULTS/name).read_text())
        for path, expected in d['source_sha256']['candidate'].items():
            assert sha(root/path) == expected, path
        assert sha(root/'exp/bench/far_skip_subspace_compare.py') == d['source_sha256']['harness']
        rows = []
        for width in d['config']['widths']:
            for mode in d['config']['modes']:
                rr = [r for r in d['measurements'] if r['width'] == width and r['mode'] == mode]
                med = {n: statistics.median([v for r in rr for v in r[n]['wall_ms']])
                       for n in ['baseline', 'candidate']}
                rows.append(dict(width=width, mode=mode, ms=med, ratio=med['baseline']/med['candidate'],
                                 setup_ms={n: statistics.median([r[n]['setup_ms'] for r in rr])
                                           for n in ['baseline', 'candidate']},
                                 setup_plus_first_ms={n: statistics.median([r[n]['setup_ms']+r[n]['first_step_ms'] for r in rr])
                                                      for n in ['baseline', 'candidate']},
                                 fine_key_cache_bytes=rr[0]['candidate']['fine_key_cache_bytes'],
                                 kv_bytes=rr[0]['candidate']['kv_bytes']))
        summary[name] = rows
        checked += sum(r['outputs_checked'] for r in d['measurements'])
        error = max(error, max(r['max_abs_error'] for r in d['measurements']))
    summary.update(outputs_and_decisions_checked=checked, max_absolute_error=error,
                   all_compared_decisions_equal=True)
    log = (RESULTS/'final_tests.txt').read_text()
    match = re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed', log)
    assert match
    summary['tests'] = dict(zip(['passed', 'skipped_cuda', 'subtests_passed'], map(int, match.groups())))
    summary['profile_allocation_bytes'] = [dict(mode=r['mode'],
        **{n: r[n]['profile']['positive_self_allocation_bytes'] for n in ['baseline', 'candidate']})
        for r in json.loads((RESULTS/'final_profile_cpu.json').read_text())['measurements']]
    (RESULTS/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    paths = list((root/'sparse_attn/ops').glob('far_skip*.py')) + [root/f for f in [
        'sparse_attn/ops/tli_static_cache.py', 'sparse_attn/patches/qwen3_attn_patch.py',
        'tests/test_far_skip_compact.py', 'tests/test_far_skip_graph.py',
        'tests/test_far_skip_optimized.py', 'tests/test_far_skip_subspace.py',
        'exp/bench/e152v4_graph_latency.py', 'exp/bench/far_skip_decode.py',
        'exp/bench/far_skip_graph_compare.py', 'exp/bench/far_skip_subspace_compare.py',
        'exp/bench/summarize_far_skip_subspace.py']]
    (RESULTS/'validation.json').write_text(json.dumps(dict(
        status='CPU subset verified; target CUDA and quality gates remain unrun',
        final_source_bindings_match=True, tests=summary['tests'],
        outputs_and_decisions_checked=checked, max_absolute_error=error,
        source_sha256={str(path.relative_to(root)): sha(path) for path in paths},
        receipt_sha256={path.name: sha(path) for path in RESULTS.iterdir()
                        if path.suffix in ['.json', '.txt'] and path.name != 'validation.json'}), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
