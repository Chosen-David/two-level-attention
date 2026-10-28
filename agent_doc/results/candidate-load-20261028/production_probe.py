"""Actual wrapper/tile validation. CPU interpreter is NOT CUDA acceptance.

TRITON_INTERPRET=1 python agent_doc/results/candidate-load-20261028/production_probe.py
python agent_doc/results/candidate-load-20261028/production_probe.py --bench
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import triton

from baseline_kernel import _score_fine_cand_kernel as baseline
from sparse_attn.indexer import tli_indexer as module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be >= 2 for alternating order')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.bench and (device != 'cuda' or triton.knobs.runtime.interpret):
        raise RuntimeError('Timing requires actual CUDA execution')
    torch.set_num_threads(1)
    candidate = module._score_fine_cand_kernel
    indexer = object.__new__(module.TLIIndexer)
    rows = []
    dtypes = (torch.float32, torch.float16, torch.bfloat16) if device == 'cuda' else (torch.float32, torch.float16)
    for dtype in dtypes:
        for d in (32, 64, 128):
            bt, nw = indexer._TILE_FINE[d]
            for g in (1, 4):
                for layout in ('token', 'head'):
                    for r in (bt + 1, 2 * bt + 1, 4224):
                        for occupancy in ('empty', 'mixed', 'full'):
                            torch.manual_seed(281028)
                            t, h = (32768 if args.bench else 257), 3
                            k = torch.randn((t, h, d) if layout == 'token' else (h, t, d), device=device, dtype=dtype)
                            if layout == 'head':
                                k = k.transpose(0, 1)
                            # Contiguous feature dimension, nontrivial row stride.
                            q = torch.randn(h * g, d + 8, device=device)[:, :d]
                            ids = torch.randint(t, (h, r), device=device)
                            lens = {'empty': [0, 0, 0], 'mixed': [0, r // 2, r], 'full': [r, r, r]}[occupancy]
                            lengths = torch.tensor(lens, device=device)
                            valid = torch.arange(r, device=device)[None] < lengths[:, None]
                            ids = torch.where(valid, ids, t - 1)

                            def wrapper(fn):
                                # Exercise the real _TILE_FINE lookup, strides,
                                # allocation, and launch arguments on both arms.
                                saved = module._score_fine_cand_kernel
                                module._score_fine_cand_kernel = fn
                                try:
                                    return indexer._fast_score_fine_cand(k, q, ids, lengths, r, dtype)
                                finally:
                                    module._score_fine_cand_kernel = saved

                            old, new = wrapper(baseline), wrapper(candidate)
                            torch.testing.assert_close(new, old, rtol=0, atol=0)
                            assert torch.isneginf(new[~valid.repeat_interleave(g, 0)]).all()
                            record = dict(dtype=str(dtype), d=d, g=g, r=r, t=t, layout=layout,
                                          occupancy=occupancy, bt=bt, num_warps=nw, exact=True)
                            if args.bench:
                                from triton.testing import do_bench, do_bench_cudagraph
                                # Preallocate output so kernel timing does not
                                # conflate allocator/Python with this predicate.
                                def launch(fn, out):
                                    fn[(triton.cdiv(r, bt), h)](
                                        k, q, ids, lengths, out, r, k.stride(0), k.stride(1),
                                        q.stride(0), ids.stride(0), ids.stride(1),
                                        out.stride(0), out.stride(1), Hkv=h, G=g, D=d,
                                        BT=bt, num_warps=nw)
                                for mode, bench in (('eager', do_bench), ('graph', do_bench_cudagraph)):
                                    samples = []
                                    for rep in range(args.repeats):
                                        pair = {}
                                        arms = [('baseline', baseline, old), ('candidate', candidate, new)]
                                        if rep % 2:
                                            arms.reverse()
                                        for name, fn, out in arms:
                                            pair[name] = bench(lambda: launch(fn, out))
                                        samples.append(pair)
                                    record[mode] = dict(samples_ms=samples,
                                        median_ratio=statistics.median(x['candidate'] / x['baseline'] for x in samples))
                            rows.append(record)
    result = dict(scope='CUDA kernel microbenchmark' if args.bench else 'correctness only',
                  device=device, torch=torch.__version__, triton=triton.__version__,
                  candidate_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(Path(__file__).with_name('baseline_kernel.py').read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=rows)
    out = args.out or Path(__file__).with_name('production-gpu.json' if args.bench else 'production-pairs.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(device=device, pairs=len(rows), exact=True, bench=args.bench)))


if __name__ == '__main__':
    main()
