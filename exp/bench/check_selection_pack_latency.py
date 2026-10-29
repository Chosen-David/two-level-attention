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
    observations = []
    stop = threading.Event()
    uuid = 'GPU-05a75a00-5bfc-a167-09e9-8bbf429846c9'

    def observe():
        raw = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
        foreign = [line for line in raw.splitlines()
                   if uuid in line and int(line.split(',')[1]) != os.getpid()]
        observations.append(dict(time=time.time(), raw=raw, foreign=foreign))

    def monitor():
        while not stop.wait(0.2):
            try:
                observe()
            except Exception as exc:
                observations.append(dict(error=str(exc)))

    observe()
    assert not observations[-1]['foreign'], observations[-1]
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
    native = selection_pack.pack_unique_top
    for mode in ('torch', 'fused'):
        calls[mode] = {'calls': 0, 'fused': 0}

        def dispatch(*args):
            calls[mode]['calls'] += 1
            result = native(*args) if mode == 'fused' else None
            calls[mode]['fused'] += int(result is not None)
            return result

        with patch.object(selection_pack, 'pack_unique_top', dispatch):
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
        seed=729, capacity=capacity, pid=os.getpid(), calls=calls,
        captures=2, measured_replays=1200, samples=samples,
        gpu_observations=observations, uncontaminated=valid,
        max_allocated_bytes=torch.cuda.max_memory_allocated()), indent=2)+'\n')
    assert valid, 'GPU interference or monitoring failure; timing invalid'


if __name__ == '__main__':
    main()
