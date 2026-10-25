"""Same-subspace paired comparison; changing width changes selection semantics.

Both paths include one-token KV update, projection/QDQ, selection and full-D
attention. CPU fixed shapes are not CUDA Graph timing. --capture is actual CUDA
capture/replay of the fixed path, including fresh input injection in wall time.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch
from transformers.cache_utils import DynamicCache


def load_ops(root, name):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(Path(root).resolve()/'sparse_attn/ops')]
    sys.modules[name] = pkg
    return types.SimpleNamespace(**{key: importlib.import_module(name+'.'+file) for key, file in [
        ('core', 'far_skip'), ('cache', 'far_skip_cache'), ('graph', 'far_skip_graph'),
        ('workspace', 'far_skip_workspace')]})


@torch.inference_mode()
def run(opts, ops, width, mode):
    torch.manual_seed(opts.seed)
    dev, dtype = torch.device(opts.device), getattr(torch, opts.dtype)
    cfg = ops.core.FarSkipConfig(64, opts.near_len, opts.near_pages, 1024, 128, 128,
                                'avg', opts.quantize)
    half = width//2
    dims = torch.cat([torch.arange(64-half, 64, device=dev),
                      torch.arange(128-half, 128, device=dev)])
    cache = DynamicCache()
    cache.update(torch.randn(1, 4, opts.length, 128, device=dev, dtype=dtype),
                 torch.randn(1, 4, opts.length, 128, device=dev, dtype=dtype), 0)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else lambda: None
    sync(); start = time.perf_counter()
    ring = ops.cache.evict_far(cache, 0, cfg)
    kk, vv = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
    ws = (ops.graph.FarSkipGraphWorkspace(cfg, ring.layout, 4, dims, dims)
          if mode == 'fixed' else ops.workspace.FarSkipWorkspace())
    if mode == 'fixed':
        ws.initialize(kk, opts.length)
    sync(); setup_ms = (time.perf_counter()-start)*1000
    inputs = [(torch.randn(1, 1, 16, 128, device=dev, dtype=dtype),
               torch.randn(1, 4, 1, 128, device=dev, dtype=dtype),
               torch.randn(1, 4, 1, 128, device=dev, dtype=dtype))
              for _ in range(opts.steps+opts.warmup)]
    pos = torch.tensor([opts.length], device=dev)
    qbuf, kbuf, vbuf = [torch.empty_like(x) for x in inputs[0]]
    def fixed():
        slot = ring.layout.map(pos)
        ring.keys.index_copy_(2, slot, kbuf); ring.values.index_copy_(2, slot, vbuf)
        ids, valid = ws.select(qbuf, kk, pos+1, 128**-.5)
        return ws.attend(qbuf, kk, vv, ids, valid, 128**-.5), ids, valid
    graph, capture_ms = None, None
    if opts.capture:
        for dst, src in zip((qbuf, kbuf, vbuf), inputs[0]):
            dst.copy_(src)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fixed(); fixed()
        torch.cuda.current_stream().wait_stream(stream)
        sync(); start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fixed()
        sync(); capture_ms = (time.perf_counter()-start)*1000
    times, outputs, decisions, profile = [], [], [], None
    for step, (q, nk, nv) in enumerate(inputs):
        length = opts.length+step+1
        sync()
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                           profile_memory=True, record_shapes=True)
                    if opts.profile and step == opts.warmup else None)
        with profiler if profiler else nullcontext():
            start = time.perf_counter()
            if mode == 'fixed':
                pos.fill_(length-1)
                qbuf.copy_(q); kbuf.copy_(nk); vbuf.copy_(nv)
                if graph:
                    graph.replay(); out, ids, valid = graph_output
                else:
                    out, ids, valid = fixed()
            else:
                ring.update(nk, nv)
                selection = ops.core.select_far_skip(q, kk, length, cfg, dims, dims,
                                                      layout=ring.layout, workspace=ws)
                out = ops.core.attend_far_skip(q, kk, vv, selection, cfg, layout=ring.layout, workspace=ws)
                ids, valid = selection.positions, None
            sync(); elapsed = (time.perf_counter()-start)*1000
        if profiler:
            profile = dict(ops={e.key: {'count': e.count, 'self_cpu_us': e.self_cpu_time_total}
                                for e in profiler.key_averages()},
                           positive_self_allocation_bytes=sum(max(0, e.self_cpu_memory_usage)
                                                              for e in profiler.events()))
        times.append(elapsed)
        outputs.append(out.cpu().clone())
        cpu_ids = ids.cpu()
        cpu_valid = valid.cpu() if valid is not None else torch.ones_like(cpu_ids, dtype=torch.bool)
        decisions.append([cpu_ids[h, cpu_valid[h]].tolist() for h in range(4)])
    fine = getattr(ws, 'fine_cache', getattr(ws, '_fine_cache', None))
    auxiliary = fine.keys.untyped_storage().nbytes() if fine is not None else 0
    return dict(setup_ms=setup_ms, first_step_ms=times[0], capture_ms=capture_ms,
                wall_ms=times[opts.warmup:], median_ms=statistics.median(times[opts.warmup:]),
                fine_key_cache_bytes=auxiliary,
                kv_bytes=sum(t.untyped_storage().nbytes() for t in (ring.keys, ring.values)),
                profile=profile), outputs, decisions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    p.add_argument('--widths', nargs='+', type=int, default=[32, 128])
    p.add_argument('--modes', nargs='+', choices=['eager', 'fixed'], default=['eager', 'fixed'])
    p.add_argument('--quantize', action='store_true')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--length', type=int, default=8192)
    p.add_argument('--near-len', type=int, default=4096)
    p.add_argument('--near-pages', type=int, default=32)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261028)
    opts = p.parse_args()
    if opts.device == 'cuda' and not torch.cuda.is_available():
        p.error('real CUDA is required')
    if opts.capture and (opts.device != 'cuda' or opts.modes != ['fixed']):
        p.error('capture requires CUDA and --modes fixed')
    if any(d < 2 or d > 128 or d % 2 for d in opts.widths):
        p.error('widths must be even, 2..128')
    if min(opts.steps, opts.warmup, opts.repeats, opts.length, opts.near_len) < 1 or opts.near_pages < 0:
        p.error('invalid geometry/count')
    if opts.device == 'cpu':
        torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    modules = {'baseline': load_ops(opts.baseline_root, '_subspace_old'),
               'candidate': load_ops(root, '_subspace_new')}
    receipts = []
    for width in opts.widths:
        for mode in opts.modes:
            for repeat in range(opts.repeats):
                results = {}
                for name in (['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']):
                    results[name] = run(opts, modules[name], width, mode)
                assert results['baseline'][2] == results['candidate'][2], 'selection IDs differ'
                error = 0.
                for a, b in zip(results['baseline'][1], results['candidate'][1]):
                    torch.testing.assert_close(a, b, atol=2e-6 if opts.dtype == 'float32' else 1e-2,
                                               rtol=2e-5 if opts.dtype == 'float32' else 1e-2)
                    error = max(error, float((a.float()-b.float()).abs().max()))
                receipts.append(dict(width=width, mode=mode, repeat=repeat,
                                     outputs_checked=len(results['baseline'][1]), max_abs_error=error,
                                     **{name: r[0] for name, r in results.items()}))
    hashes = {name: {str(f.relative_to(folder)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (folder/'sparse_attn/ops').glob('far_skip*.py')}
              for name, folder in [('baseline', Path(opts.baseline_root)), ('candidate', root)]}
    hashes['harness'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(dict(scope='same-width CPU/real CUDA only; no across-width quality parity claim',
                         config=vars(opts), torch=torch.__version__, source_sha256=hashes,
                         measurements=receipts), indent=2))


if __name__ == '__main__':
    main()
