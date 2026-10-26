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
import os
from pathlib import Path
import statistics
import sys
import time
import types

import torch
from transformers.cache_utils import DynamicCache


def load_ops(root, name):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(Path(root).resolve()/'sparse_attn')]
    sys.modules[name] = pkg
    # Shared host BatchCtx contract; algorithm/cache modules remain checkout-local.
    sys.modules[name+'.baseline_runtime'] = importlib.import_module('sparse_attn.baseline_runtime')
    sub = types.ModuleType(name+'.ops')
    sub.__path__ = [str(Path(root).resolve()/'sparse_attn/ops')]
    sys.modules[name+'.ops'] = sub
    modules = types.SimpleNamespace(**{key: importlib.import_module(name+'.ops.'+file) for key, file in [
        ('core', 'far_skip'), ('cache', 'far_skip_cache'), ('graph', 'far_skip_graph'),
        ('workspace', 'far_skip_workspace')]})
    modules.accel = (importlib.import_module(name+'.ops.far_skip_accel')
                     if (Path(root)/'sparse_attn/ops/far_skip_accel.py').exists() else None)
    return modules


def workspace_bytes(workspace):
    """Count owned/referenced tensor storage once, including nested GPU scratch."""
    seen, storage = set(), {}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            s = value.untyped_storage()
            storage[(value.device, s.data_ptr())] = s.nbytes()
        elif isinstance(value, dict):
            for x in value.values():
                visit(x)
        elif isinstance(value, (list, tuple)):
            for x in value:
                visit(x)
        elif hasattr(value, '__dict__'):
            visit(vars(value))
    visit(workspace)
    return sum(storage.values())


@torch.inference_mode()
def run(opts, ops, width, mode):
    torch.manual_seed(opts.seed)
    dev, dtype = torch.device(opts.device), getattr(torch, opts.dtype)
    cfg = ops.core.FarSkipConfig(opts.block_size, opts.near_len, opts.near_pages,
                                opts.budget, opts.sink, opts.window, opts.near_method, opts.quantize)
    half = width//2
    dims = torch.cat([torch.arange(64-half, 64, device=dev),
                      torch.arange(128-half, 128, device=dev)])
    cache = DynamicCache()
    cache.update(torch.randn(1, opts.kv_heads, opts.length, 128, device=dev, dtype=dtype),
                 torch.randn(1, opts.kv_heads, opts.length, 128, device=dev, dtype=dtype), 0)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else lambda: None
    sync(); start = time.perf_counter()
    if mode == 'staged':
        static_ops = importlib.import_module(ops.core.__package__+'.far_skip_static')
        ring = static_ops.FarSkipStaticLayer(cache.layers[0], cfg,
                    opts.length+opts.steps+opts.warmup+1, dims, dims)
        del cache
        static_cache = types.SimpleNamespace(layers=[ring], _far_skip_prepared=True,
                                             _far_skip_attention_mask=None)
        module = types.SimpleNamespace(training=False, layer_idx=0, scaling=128**-.5,
                    indexer=types.SimpleNamespace(skip_far=True, far_skip_config=cfg,
                                                  far_skip_cache='evict'))
        ctx = types.SimpleNamespace(batch=1, seq_lens=[opts.length])
    else:
        ring = ops.cache.evict_far(cache, 0, cfg)
    kk, vv = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
    ws = (ops.graph.FarSkipGraphWorkspace(cfg, ring.layout, opts.kv_heads, dims, dims)
          if mode == 'fixed' else ops.workspace.FarSkipWorkspace())
    if mode == 'fixed':
        ws.initialize(kk, opts.length)
    sync(); setup_ms = (time.perf_counter()-start)*1000
    inputs = [(torch.randn(1, 1, opts.heads, 128, device=dev, dtype=dtype),
               torch.randn(1, opts.kv_heads, 1, 128, device=dev, dtype=dtype),
               torch.randn(1, opts.kv_heads, 1, 128, device=dev, dtype=dtype))
              for _ in range(opts.steps+opts.warmup)]
    pos = torch.tensor([opts.length], device=dev)
    qbuf, kbuf, vbuf = [torch.empty_like(x) for x in inputs[0]]
    def fixed():
        if ops.accel is not None and ops.accel.enabled(kbuf):
            ops.accel.write_pair(ring.keys, ring.values, kbuf, vbuf, pos, ring.layout)
        else:
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
        activities = [torch.profiler.ProfilerActivity.CPU]
        if dev.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = (torch.profiler.profile(activities=activities,
                                           profile_memory=True, record_shapes=True)
                    if opts.profile and step == opts.warmup else None)
        with profiler if profiler else nullcontext():
            start = time.perf_counter()
            if mode == 'staged':
                ctx.seq_lens[0] = length
                out = static_ops.try_far_skip_static(module, q.transpose(1, 2), nk, nv,
                                                     static_cache, None, ctx)
                ids = valid = None
            elif mode == 'fixed':
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
            profile['cuda_events'] = [dict(name=e.name, device_us=e.device_time_total)
                                      for e in profiler.events()
                                      if e.device_type == torch.autograd.DeviceType.CUDA]
        # An instrumented step cannot contribute to the latency verdict.
        times.append(None if profiler else elapsed)
        outputs.append(out.cpu().clone())
        if ids is not None:
            cpu_ids = ids.cpu()
            cpu_valid = valid.cpu() if valid is not None else torch.ones_like(cpu_ids, dtype=torch.bool)
            decisions.append([cpu_ids[h, cpu_valid[h]].tolist() for h in range(opts.kv_heads)])
    if mode == 'staged':
        ws = getattr(ring, 'eager_workspace', None) or ring.workspace
    fine = getattr(ws, 'fine_cache', getattr(ws, '_fine_cache', None))
    auxiliary = fine.keys.untyped_storage().nbytes() if fine is not None else 0
    measured = [x for x in times[opts.warmup:] if x is not None]
    return dict(setup_ms=setup_ms, first_step_ms=times[0], capture_ms=capture_ms,
                wall_ms=measured, median_ms=statistics.median(measured),
                fine_key_cache_bytes=auxiliary,
                workspace_storage_bytes=workspace_bytes(ws),
                kv_bytes=sum(t.untyped_storage().nbytes() for t in (ring.keys, ring.values)),
                profile=profile), outputs, decisions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    p.add_argument('--widths', nargs='+', type=int, default=[32, 128])
    p.add_argument('--modes', nargs='+', choices=['eager', 'fixed', 'staged'], default=['eager', 'fixed'])
    p.add_argument('--quantize', action='store_true')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--kernels', choices=['torch', 'triton'], default='torch')
    p.add_argument('--baseline-kernels', choices=['torch', 'triton'], default='torch')
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--block-size', type=int, default=64)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--sink', type=int, default=128)
    p.add_argument('--window', type=int, default=128)
    p.add_argument('--near-method', choices=['avg', 'minmax'], default='avg')
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
    if 'triton' in (opts.kernels, opts.baseline_kernels) and opts.device != 'cuda':
        p.error('Triton timing requires real CUDA')
    if opts.profile and opts.steps < 2:
        p.error('profiling needs at least two measured steps')
    if min(opts.heads, opts.kv_heads, opts.block_size) < 1 or opts.heads % opts.kv_heads:
        p.error('invalid head/block geometry')
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
                    os.environ['TLI_FAR_SKIP_KERNELS'] = (opts.baseline_kernels if name == 'baseline' else opts.kernels)
                    results[name] = run(opts, modules[name], width, mode)
                assert results['baseline'][2] == results['candidate'][2], 'selection IDs differ'
                error = 0.
                for a, b in zip(results['baseline'][1], results['candidate'][1]):
                    torch.testing.assert_close(a, b, atol=2e-6 if opts.dtype == 'float32' else 1e-2,
                                               rtol=2e-5 if opts.dtype == 'float32' else 1e-2)
                    error = max(error, float((a.float()-b.float()).abs().max()))
                receipts.append(dict(width=width, mode=mode, repeat=repeat,
                                     outputs_checked=len(results['baseline'][1]), max_abs_error=error,
                                     decision_steps_checked=len(results['baseline'][2]),
                                     **{name: r[0] for name, r in results.items()}))
    hashes = {name: {str(f.relative_to(folder)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (folder/'sparse_attn/ops').glob('far_skip*.py')}
              for name, folder in [('baseline', Path(opts.baseline_root)), ('candidate', root)]}
    hashes['harness'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes['shared_batch_context'] = hashlib.sha256((root/'sparse_attn/baseline_runtime.py').read_bytes()).hexdigest()
    print(json.dumps(dict(scope='same-width CPU/real CUDA only; no across-width quality parity claim',
                         config=vars(opts), torch=torch.__version__, source_sha256=hashes,
                         gpu=torch.cuda.get_device_name() if opts.device == 'cuda' else None,
                         measurements=receipts), indent=2))


if __name__ == '__main__':
    main()
