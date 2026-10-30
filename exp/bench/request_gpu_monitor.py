"""Fail-closed GPU0 ownership observations for request benchmarks.

The cooperative lock is acquired by the launcher. NVML uses host PIDs, so
ownership is established by empty-device -> unique context -> allocation delta.
Sampling cannot rule out interference shorter than the observation interval.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch


class GPU0Monitor:
    def __init__(self, path):
        self.path = Path(path)
        self.uuid = 'GPU-05a75a00-5bfc-a167-09e9-8bbf429846c9'
        self.observations = []
        self.stop = threading.Event()
        self.ownership = None

    def query(self):
        return subprocess.check_output([
            'nvidia-smi', '-i', self.uuid,
            '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True)

    def observe(self):
        try:
            raw = self.query()
            foreign = [line for line in raw.splitlines() if self.uuid in line
                       and int(line.split(',')[1]) != self.pid]
            self.observations.append(dict(time=time.time(), raw=raw, foreign=foreign))
        except Exception as exc:
            self.observations.append(dict(time=time.time(), error=str(exc)))

    def watch(self):
        while not self.stop.wait(0.2):
            self.observe()

    def __enter__(self):
        before = self.query()
        if any(self.uuid in line for line in before.splitlines()):
            self.path.write_text(json.dumps(dict(admission='busy', before=before)))
            raise RuntimeError('GPU0 is occupied; no benchmark admitted')
        torch.cuda.init()
        torch.empty(1, device='cuda:0')
        torch.cuda.synchronize()
        initialized = self.query()
        rows = [line.split(',') for line in initialized.splitlines() if self.uuid in line]
        assert len(rows) == 1, initialized
        self.pid = int(rows[0][1])
        initial_mib = int(rows[0][2])
        marker = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
        torch.cuda.synchronize()
        allocated = self.query()
        rows = [line.split(',') for line in allocated.splitlines() if self.uuid in line]
        assert len(rows) == 1 and int(rows[0][1]) == self.pid, allocated
        assert int(rows[0][2]) >= initial_mib + 60, allocated
        self.ownership = dict(container_pid=os.getpid(), nvml_pid=self.pid,
                              before=before, initialized=initialized, allocated=allocated)
        del marker
        torch.cuda.empty_cache()
        self.observe()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.observe()
        invalid = any(row.get('foreign') or row.get('error') for row in self.observations)
        self.path.write_text(json.dumps(dict(
            ownership=self.ownership, observations=self.observations,
            interference_detected=invalid, exception=None if exc is None else str(exc),
            limitation='cooperative lock and sampled observations; no OS-exclusive guarantee'), indent=2)+'\n')
        if exc is None and invalid:
            raise RuntimeError('GPU0 interference/monitor failure: request timings invalid')
        return False
