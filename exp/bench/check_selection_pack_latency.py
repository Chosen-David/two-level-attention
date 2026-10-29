"""Paired full-selector Graph replay diagnostic; not model TPOT."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops import selection_pack


def main():
    output = Path(sys.argv[1])
    component = sys.argv[2] if len(sys.argv) > 2 else 'selection'
    if component not in ('selection', 'domain', 'mean', 'normalization', 'ranking', 'sparse', 'narrow'):
        raise ValueError('component must be selection, domain, mean, normalization, ranking, sparse or narrow')
    observations = []
    stop = threading.Event()
    uuid = 'GPU-05a75a00-5bfc-a167-09e9-8bbf429846c9'
    device_pid = os.getpid()

    def query():
        return subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True)

    def observe():
        raw = query()
        foreign = [line for line in raw.splitlines()
                   if uuid in line and int(line.split(',')[1]) != device_pid]
        observations.append(dict(time=time.time(), raw=raw, foreign=foreign))

    def monitor():
        while not stop.wait(0.2):
            try:
                observe()
            except Exception as exc:
                observations.append(dict(error=str(exc)))

    # NVML reports host PIDs in this container. Establish ownership before
    # timing: empty GPU -> our CUDA context -> sole NVML PID, then confirm
    # that same PID's allocation grows with our explicit 64 MiB allocation.
    before = query()
    assert not any(uuid in line for line in before.splitlines()), before
    torch.cuda.init()
    torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    initialized = query()
    rows = [line.split(',') for line in initialized.splitlines() if uuid in line]
    assert len(rows) == 1, initialized
    device_pid = int(rows[0][1])
    initial_mib = int(rows[0][2])
    marker = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    allocated = query()
    rows = [line.split(',') for line in allocated.splitlines() if uuid in line]
    assert len(rows) == 1 and int(rows[0][1]) == device_pid, allocated
    assert int(rows[0][2]) >= initial_mib + 60, allocated
    ownership = dict(container_pid=os.getpid(), nvml_pid=device_pid,
                     before=before, initialized=initialized, allocated=allocated,
                     method='empty device, sole initialized context, 64MiB allocation delta')
    del marker
    torch.cuda.empty_cache()
    observe()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    torch.manual_seed(729)
    capacity = int(sys.argv[3]) if len(sys.argv) > 3 else 32832
    if capacity <= 9216:
        raise ValueError("normalization diagnostic requires capacity > 9216")
    q = torch.randn(1, 1, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, capacity, 8, 128, device='cuda', dtype=q.dtype)
    length = torch.tensor([capacity], device='cuda')
    cu = torch.tensor([0, capacity], device='cuda', dtype=torch.int32)
    helpers = runpy.run_path(str(Path(__file__).with_name('test_tli_graph_gates.py')))
    idx = TLIIndexer(helpers['_make_prod_args']())
    idx.layer_idx = 2
    idx.clear()
    index = idx.prepare_index(k, cu)
    graphs, outputs, calls = {}, {}, {}
    if component in ('sparse', 'narrow'):
        from sparse_attn.ops import sparse_normalization
        target, attribute = sparse_normalization, 'supported_geometry'
    elif component == 'ranking':
        from sparse_attn.ops import compact_ranking
        target, attribute = compact_ranking, 'compact_topk'
    elif component == 'normalization':
        from sparse_attn.ops import compact_normalization
        target, attribute = compact_normalization, 'supported'
    elif component == 'mean':
        from sparse_attn.ops import compact_normalization
        target, attribute = compact_normalization, 'compact_probabilities'
    elif component == 'domain':
        from sparse_attn.indexer import candidate_domain_triton
        target, attribute = candidate_domain_triton, 'complete_and_pack'
    else:
        target, attribute = selection_pack, 'pack_unique_top'
    native = getattr(target, attribute)
    for mode in ('torch', 'fused'):
        calls[mode] = {'calls': 0, 'fused': 0}

        def dispatch(*args):
            calls[mode]['calls'] += 1
            if component == 'mean':
                with patch.dict(os.environ, {'TLI_COMPACT_FUSED_MEAN': '1' if mode == 'fused' else '0'}):
                    result = native(*args)
            else:
                result = native(*args) if mode == 'fused' or component == 'narrow' else None
            calls[mode]['fused'] += int(result is not None and (component != 'mean' or mode == 'fused'))
            return result

        native_sort = torch.Tensor.sort
        def observed_sort(tensor, *args, **kwargs):
            calls[mode].setdefault('sort_input_dtypes', []).append(str(tensor.dtype))
            return native_sort(tensor, *args, **kwargs)

        with patch.dict(os.environ, {'TLI_COMPACT_NORMALIZATION': '1' if component in ('mean', 'normalization', 'ranking', 'sparse', 'narrow') else '0', 'TLI_COMPACT_FUSED_MEAN': '1' if mode == 'fused' or component in ('ranking', 'sparse', 'narrow') else '0', 'TLI_COMPACT_RANKING': '1' if component in ('ranking', 'sparse', 'narrow') else '0', 'TLI_SPARSE_SELECTION': '1' if component in ('sparse', 'narrow') else '0', 'TLI_SPARSE_NARROW_KEYS': '0' if component == 'narrow' and mode == 'torch' else '1', 'TLI_COMPACT_KTH': 'histogram', 'TLI_COMPACT_GATHER': 'blocked'}), patch.object(target, attribute, dispatch), patch.object(torch.Tensor, 'sort', observed_sort):
            for _ in range(3):
                idx.compute_selection_graph(q, index, 128**-.5, length)
            torch.cuda.synchronize()
            graphs[mode] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[mode]):
                outputs[mode] = idx.compute_selection_graph(q, index, 128**-.5, length)
            calls[mode]['sparse_selection_fallback'] = idx._graph_sparse_selection_fallback
            calls[mode]['normalization_capture_route'] = idx._graph_normalization_route
            calls[mode]['ranking_capture_route'] = idx._graph_ranking_route
            calls[mode]['kth_capture_route'] = idx._graph_ranking_kth_route
            calls[mode]['gather_capture_route'] = idx._graph_ranking_gather_route
        graphs[mode].replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs['torch'].indices, outputs['fused'].indices)
    assert torch.equal(outputs['torch'].counts, outputs['fused'].counts)
    assert calls['fused']['fused'] > 0
    if component == 'sparse':
        assert calls['fused']['normalization_capture_route'] == 'sparse_candidate_domain'
        assert calls['fused']['ranking_capture_route'] == 'candidate_domain_native_1024'
        assert calls['fused']['sparse_selection_fallback'] is None
        assert calls['torch']['normalization_capture_route'] == 'compact_fused_mean'
    if os.environ.get('TLI_SELECTOR_PROFILE') == '1':
        # Diagnostic only; profile runs are outside the paired timing samples.
        for mode in ('torch', 'fused'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    graphs[mode].replay()
                torch.cuda.synchronize()
            output.with_suffix('.' + mode + '.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50))
    if component == 'narrow':
        for mode in ('torch', 'fused'):
            assert calls[mode]['normalization_capture_route'] == 'sparse_candidate_domain'
            assert calls[mode]['ranking_capture_route'] == 'candidate_domain_native_1024'
        assert 'torch.int16' in calls['fused']['sort_input_dtypes']
        assert 'torch.int16' not in calls['torch']['sort_input_dtypes']
    samples = []
    for pair in range(30):
        for mode in (('torch', 'fused') if pair % 2 == 0 else ('fused', 'torch')):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            begin.record()
            for _ in range(20):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            samples.append(dict(pair=pair, mode=mode, replays=20,
                                gpu_ms=begin.elapsed_time(end)/20,
                                wall_ms=(time.perf_counter()-wall)*1000/20))
    stop.set()
    watcher.join()
    observe()
    valid = not any(row.get('foreign') or row.get('error') for row in observations)
    output.write_text(json.dumps(dict(
        scope='B1 synthetic full selector replay only; excludes model/inject/sampling/prefill/capture',
        seed=729, capacity=capacity, component=component,
        pid=os.getpid(), ownership=ownership, calls=calls,
        captures=2, measured_replays=1200, samples=samples,
        gpu_observations=observations, uncontaminated=valid,
        max_allocated_bytes=torch.cuda.max_memory_allocated()), indent=2)+'\n')
    assert valid, 'GPU interference or monitoring failure; timing invalid'


if __name__ == '__main__':
    main()
