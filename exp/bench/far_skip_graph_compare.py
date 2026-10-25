"""Source-bound fixed-path ablation. CPU is not a CUDA Graph performance result.

--baseline-root must contain round-2 far_skip_graph.py and its relative imports.
Both arms include one-token ring writes, selection and selected-KV SDPA. Setup
and graph capture are separately reported. Fresh Q/K/V each step; comparisons
are outside timing. --capture requires CUDA and measures injection plus replay.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import types

import torch
from transformers.cache_utils import DynamicCache
from sparse_attn.ops.far_skip import FarSkipConfig, select_far_skip, attend_far_skip
from sparse_attn.ops.far_skip_cache import evict_far
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace


def load_fixed(root):
    package = types.ModuleType('_far_fixed_baseline')
    package.__path__ = [str(Path(root).resolve() / 'sparse_attn/ops')]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__+'.far_skip_graph').FarSkipGraphWorkspace


@torch.inference_mode()
def run(opts, cls, *, profile=False):
    torch.manual_seed(opts.seed)
    cfg = FarSkipConfig(64, opts.near_len, opts.near_pages, 1024, 128, 128)
    dev = torch.device(opts.device)
    dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
    coarse, fine = torch.arange(32, device=dev), torch.arange(64, device=dev)
    cache = DynamicCache()
    cache.update(torch.randn(1, 4, opts.length, 128, device=dev, dtype=dtype),
                 torch.randn(1, 4, opts.length, 128, device=dev, dtype=dtype), 0)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else lambda: None
    sync()
    start = time.perf_counter()
    ring = evict_far(cache, 0, cfg)
    ws = cls(cfg, ring.layout, 4, coarse, fine) if cls else FarSkipWorkspace()
    kk, vv = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
    if hasattr(ws, 'initialize'):
        ws.initialize(kk, opts.length)
    sync()
    setup_ms = (time.perf_counter()-start)*1e3
    inputs = [(torch.randn(1, 1, 16, 128, device=dev, dtype=dtype),
               torch.randn(1, 4, 1, 128, device=dev, dtype=dtype),
               torch.randn(1, 4, 1, 128, device=dev, dtype=dtype))
              for _ in range(opts.steps+opts.warmup)]
    pos = torch.tensor([opts.length], device=dev)
    qbuf, kbuf, vbuf = [torch.empty_like(t) for t in inputs[0]]
    def fixed_step():
        slot = ring.layout.map(pos)
        ring.keys.index_copy_(2, slot, kbuf)
        ring.values.index_copy_(2, slot, vbuf)
        positions, valid = ws.select(qbuf, kk, pos+1, 128**-.5)
        out = ws.attend(qbuf, kk, vv, positions, valid, 128**-.5)
        return out, positions, valid
    graph, graph_output, capture_ms = None, None, None
    if opts.capture:
        if cls is None:
            raise ValueError('dynamic eager oracle cannot be captured')
        for target, source in zip((qbuf, kbuf, vbuf), inputs[0]):
            target.copy_(source)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fixed_step()
        torch.cuda.current_stream().wait_stream(stream)
        sync()
        start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fixed_step()
        sync()
        capture_ms = (time.perf_counter()-start)*1e3
    samples, outputs, decisions = [], [], []
    events = []
    for step, (q, nk, nv) in enumerate(inputs):
        length = opts.length+step+1
        sync()
        profiler = None
        if profile and step == opts.warmup:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                               profile_memory=True, record_shapes=True)
            profiler.__enter__()
        start = time.perf_counter()
        if cls:
            pos.fill_(length-1)
            qbuf.copy_(q); kbuf.copy_(nk); vbuf.copy_(nv)
            if graph:
                graph.replay()
                out, positions, valid = graph_output
            else:
                out, positions, valid = fixed_step()
        else:
            ring.update(nk, nv)
            selection = select_far_skip(q, kk, length, cfg, coarse, fine, layout=ring.layout, workspace=ws)
            out = attend_far_skip(q, kk, vv, selection, cfg, layout=ring.layout, workspace=ws)
            positions = selection.positions
            valid = torch.ones_like(positions, dtype=torch.bool)
        sync()
        elapsed = (time.perf_counter()-start)*1e3
        if profiler:
            profiler.__exit__(None, None, None)
            events = {e.key: e.count for e in profiler.key_averages()}
            allocations = sum(max(0, e.self_cpu_memory_usage) for e in profiler.events())
        if step >= opts.warmup:
            samples.append(elapsed)
        outputs.append(out.cpu().clone())
        p, m = positions.cpu(), valid.cpu()
        decisions.append([p[h, m[h]].tolist() for h in range(4)])
    receipt = dict(setup_ms=setup_ms, capture_ms=capture_ms, wall_ms=samples,
                   median_ms=statistics.median(samples), kv_bytes=sum(t.untyped_storage().nbytes()
                     for t in (ring.keys, ring.values)))
    if profile:
        receipt.update(cpu_ops=events, positive_self_cpu_allocation_bytes=allocations)
    return receipt, outputs, decisions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--length', type=int, default=8192)
    p.add_argument('--near-len', type=int, default=4096)
    p.add_argument('--near-pages', type=int, default=32)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261026)
    opts = p.parse_args()
    if (opts.device == 'cuda' and not torch.cuda.is_available()) or (opts.capture and opts.device != 'cuda'):
        p.error('requires real CUDA; never emulate capture with CPU')
    if min(opts.steps, opts.warmup, opts.repeats, opts.length, opts.near_len) < 1 or opts.near_pages < 0:
        p.error('invalid benchmark geometry/count')
    if opts.capture and not opts.baseline_root:
        p.error('--capture comparison requires a graph-compatible baseline-root')
    if opts.device == 'cpu':
        torch.set_num_threads(1)
    baseline = load_fixed(opts.baseline_root) if opts.baseline_root else None
    rows = []
    for repeat in range(opts.repeats):
        result = {}
        for name in (['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']):
            result[name] = run(opts, baseline if name == 'baseline' else FarSkipGraphWorkspace,
                               profile=opts.profile)
        assert result['baseline'][2] == result['candidate'][2], 'selected token mismatch'
        errors = []
        for a, b in zip(result['baseline'][1], result['candidate'][1]):
            torch.testing.assert_close(a, b, atol=1e-2 if opts.device == 'cuda' else 2e-6,
                                       rtol=1e-2 if opts.device == 'cuda' else 2e-5)
            errors.append(float((a.float()-b.float()).abs().max()))
        rows.append(dict(repeat=repeat, outputs_checked=len(errors), max_abs_error=max(errors),
                         baseline=result['baseline'][0], candidate=result['candidate'][0]))
    source_roots = {'candidate': Path(__file__).resolve().parents[2]}
    if opts.baseline_root:
        source_roots['baseline'] = Path(opts.baseline_root)
    hashes = {name: {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (root/'sparse_attn/ops').glob('far_skip*.py')}
              for name, root in source_roots.items()}
    hashes['harness'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(dict(scope='single-layer fixed selector; CPU is not CUDA validation',
                         config=vars(opts), python=platform.python_version(), torch=torch.__version__,
                         source_sha256=hashes, measurements=rows,
                         medians_ms={name: statistics.median([v for r in rows for v in r[name]['wall_ms']])
                                     for name in ['baseline', 'candidate']}), indent=2))


if __name__ == '__main__':
    main()
