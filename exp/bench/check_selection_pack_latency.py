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
    if component not in ('selection', 'domain'):
        raise ValueError('component must be selection or domain')
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
    capacity = 32832
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
    if component == 'domain':
        from sparse_attn.indexer import candidate_domain_triton
        target, attribute = candidate_domain_triton, 'complete_and_pack'
    else:
        target, attribute = selection_pack, 'pack_unique_top'
    native = getattr(target, attribute)
    for mode in ('torch', 'fused'):
        calls[mode] = {'calls': 0, 'fused': 0}

        def dispatch(*args):
            calls[mode]['calls'] += 1
            result = native(*args) if mode == 'fused' else None
            calls[mode]['fused'] += int(result is not None)
            return result

        with patch.object(target, attribute, dispatch):
            for _ in range(3):
                idx.compute_selection_graph(q, index, 128**-.5, length)
            torch.cuda.synchronize()
            graphs[mode] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[mode]):
                outputs[mode] = idx.compute_selection_graph(q, index, 128**-.5, length)
        graphs[mode].replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs['torch'].indices, outputs['fused'].indices)
    assert torch.equal(outputs['torch'].counts, outputs['fused'].counts)
    assert calls['fused']['fused'] > 0
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
