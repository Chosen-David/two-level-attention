"""Paired v1 versus optimized eager runs, including real cache update.

Prepare a checkout of PR v1 (3c95ea167d66b462e2c64840adcc7c532a5115bd), then:
python -m exp.bench.far_skip_compare --baseline-root /tmp/far-v1 --device cuda
Run one process at a time on an otherwise idle device. CPU results are CPU-only.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import types

import torch
import transformers

from .far_skip_decode import run


def load_baseline(root):
    path = Path(root).resolve() / 'sparse_attn/ops'
    package = types.ModuleType('_far_skip_baseline')
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    runtime = importlib.import_module(package.__name__ + '.far_skip_runtime')
    return runtime.try_far_skip_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--lengths', nargs='+', type=int, default=[8192, 32768])
    parser.add_argument('--near-len', type=int, default=4096)
    parser.add_argument('--near-pages', type=int, default=32)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    opts = parser.parse_args()
    if min(opts.lengths + [opts.steps, opts.warmup, opts.repeats]) < 1:
        parser.error('lengths/steps/warmup/repeats must be positive')
    if opts.device == 'cuda' and not torch.cuda.is_available():
        parser.error('requires real CUDA; never substitute CPU timings')
    if opts.device == 'cpu':
        torch.set_num_threads(1)
    baseline = load_baseline(opts.baseline_root)
    receipts = []
    for length in opts.lengths:
        for repeat in range(opts.repeats):
            pair, outputs = {}, {}
            for name in (['v1', 'optimized'] if repeat % 2 == 0 else ['optimized', 'v1']):
                kw = dict(runtime=baseline) if name == 'v1' else {}
                pair[name], outputs[name] = run(opts, length, 'evict', collect_outputs=True, **kw)
            error = 0.
            for expected, actual in zip(outputs['v1'], outputs['optimized']):
                torch.testing.assert_close(actual, expected)
                error = max(error, float((actual.float()-expected.float()).abs().max()))
            receipts.append(dict(length=length, repeat=repeat, outputs_checked=len(outputs['v1']),
                                 max_abs_error=error, runs=pair))
    medians = {}
    for length in opts.lengths:
        medians[length] = {name: statistics.median([x for r in receipts if r['length'] == length
                                                    for x in r['runs'][name]['wall_ms']])
                           for name in ['v1', 'optimized']}
    source_paths = ['far_skip.py', 'far_skip_cache.py', 'far_skip_runtime.py']
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources = {name: {p: sha(root / 'sparse_attn/ops' / p) for p in paths}
               for name, root, paths in [('v1', Path(opts.baseline_root), source_paths),
                                        ('optimized', Path.cwd(), source_paths + ['far_skip_workspace.py'])]}
    sources['shared_harness'] = {p: sha(p) for p in
        ['exp/bench/far_skip_compare.py', 'exp/bench/far_skip_decode.py',
         'sparse_attn/indexer/tli_indexer.py', 'sparse_attn/arguments.py']}
    print(json.dumps(dict(scope='paired single-layer synthetic eager decode, not model throughput',
                          config=vars(opts), python=platform.python_version(), torch=torch.__version__,
                          transformers=transformers.__version__, source_sha256=sources,
                          gpu=torch.cuda.get_device_name() if opts.device == 'cuda' else None,
                          medians_ms=medians, measurements=receipts), indent=2))


if __name__ == '__main__':
    main()
