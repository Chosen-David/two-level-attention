"""Monitored, process-separated model ABBA measurements on reserved GPU1.

Raw results remain pending independent validation. This runner does not establish
executed attention backend identity or replace a separate Nsight capture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


RING_ARMS = ('graph_ring_legacy', 'graph_ring_simt', 'graph_ring_tc', 'graph_ring_overlap')


UUID = 'GPU-a9292656-54e2-7d03-d327-b307c5381cc1'


def snapshot():
    gpu = subprocess.check_output([
        'nvidia-smi', '-i', UUID,
        '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip()
    rows = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    processes = [r for r in rows if r.split(',')[0].strip() == UUID]
    return dict(timestamp=time.time(), gpu=gpu, processes=processes)


def require_lease():
    # The persistent lock is held by the already-authorized lease process.
    owner = Path('/tmp/tli-gpu-locks') / (UUID + '.owner')
    pid = int(owner.read_text().split('pid=')[1].split()[0])
    os.kill(pid, 0)
    locks = Path(f'/proc/{pid}/fdinfo/9').read_text()
    if 'FLOCK' not in locks:
        raise RuntimeError('GPU1 lease process is not holding its flock')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs=2, choices=['fa2', 'legacy', 'serial', 'overlap', 'graph_serial', 'graph_compact', 'graph_sparse', *RING_ARMS],
                        default=['fa2', 'serial'])
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--max-gen', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--ring-mask', help='Explicit frozen skip mask; required for paired ring arms')
    args = parser.parse_args()
    if min(args.cycles, args.max_gen, args.repeats) < 1:
        parser.error('positive cycles, max-gen and repeats required')
    ring = any(arm in RING_ARMS for arm in args.arms)
    if ring and (not all(arm in RING_ARMS for arm in args.arms) or not args.ring_mask):
        parser.error('ring comparison requires two ring arms and --ring-mask')
    if args.ring_mask and not ring:
        parser.error('--ring-mask requires ring arms')
    root = Path(__file__).resolve().parents[2]
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    require_lease()
    if ring:
        mask_bytes = Path(args.ring_mask).read_bytes()
        skip = json.loads(mask_bytes).get('skip')
        if not isinstance(skip, list) or not skip or any(type(x) is not int or x < 0 for x in skip):
            parser.error('ring mask must contain a nonempty list of nonnegative skip layer IDs')
        frozen_mask = args.out_dir / 'skip-mask.json'
        frozen_mask.write_bytes(mask_bytes)
    files = [p for p in (root / 'sparse_attn').rglob('*') if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.cuh')]
    files += [Path(__file__).resolve(), root / 'exp/bench/operator_model_latency.py', root / 'exp/bench/e152v4_graph_latency.py']
    receipt = dict(status='pending', pid_identity='inferred from idle-to-single-stable NVML host PID; container PID differs', root=str(root), args=vars(args) | {'out_dir': str(args.out_dir)},
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if ring:
        receipt['ring_mask_sha256'] = hashlib.sha256(mask_bytes).hexdigest()
        receipt['ring_skip_ids'] = skip
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    a, b = args.arms
    order = [a, b, b, a] * args.cycles
    with (args.out_dir / 'telemetry.jsonl').open('x', buffering=1) as telemetry:
        for index, arm in enumerate(order):
            require_lease()
            stable = 0
            deadline = time.monotonic() + 30
            while stable < 3:
                sample = snapshot()
                telemetry.write(json.dumps(sample | {'phase': 'idle', 'arm': arm}) + '\n')
                values = sample['gpu'].split(',')
                if sample['processes'] or float(values[1]) > 16:
                    raise RuntimeError('GPU1 is occupied; no model job launched')
                # Utilization is a trailing window and can remain nonzero after
                # our preceding process exited. Wait for three actual idle
                # samples; never treat that tail as a new arm's idle admission.
                stable = stable + 1 if float(values[2]) == 0 else 0
                if time.monotonic() > deadline:
                    raise RuntimeError('GPU1 did not settle to stable idle')
                time.sleep(1)
            env = {k: v for k, v in os.environ.items() if not k.startswith('TLI_')}
            env.update(CUDA_VISIBLE_DEVICES='1', TLI_OPERATOR_MODE='off' if arm in ['fa2', 'legacy'] else ('serial' if arm in ('graph_serial', 'graph_compact', 'graph_sparse') else arm),
                       TLI_OPERATOR_FUSED_SELECTION='0', TLI_SPARSE_PREFILL='0',
                       TLI_COMPACT_NORMALIZATION='1' if arm == 'graph_compact' else '0',
                       TLI_SPARSE_SELECTION='1' if arm == 'graph_sparse' else '0')
            if arm in RING_ARMS:
                env.update(TLI_OPERATOR_MODE={'graph_ring_legacy':'off', 'graph_ring_simt':'serial', 'graph_ring_tc':'serial', 'graph_ring_overlap':'overlap'}[arm],
                           TLI_OPERATOR_BACKEND='tc' if arm == 'graph_ring_tc' else 'simt',
                           TLI_BENCH_REQUIRE_RING='1')
            command = [sys.executable, str(root / ('exp/bench/e152v4_graph_latency.py' if arm in ('graph_serial', 'graph_compact', 'graph_sparse', *RING_ARMS) else 'exp/bench/operator_model_latency.py')),
                       '--method', 'none' if arm == 'fa2' else 'tli',
                       '--attention-backend', 'flash_attention_2', '--context-length', '32768',
                       '--warmup', '3', '--repeats', str(args.repeats), '--max-gen', str(args.max_gen),
                       '--out', str(args.out_dir / 'samples.json'), '--label', f'{index:02d}-{arm}',
                       '--tli_subspace', 'tail', '--tia_level2_cmp_ratio', '4',
                       '--tli_enable_kmeans', 'false', '--tli_enable_layer_skip', 'true' if ring else 'false',
                       '--tli_near_pages', '32', '--tli_near_tokens', 'off', '--tli_near_len', '4096',
                       '--tli_sink_tokens', '4', '--tli_swa_size', '128',
                       '--tli_far_skip_backend', 'compact' if ring else 'legacy',
                       '--tli_far_skip_cache', 'evict' if ring else 'retain']
            if arm in RING_ARMS:
                command += ['--tli_layer_skip_path',str(frozen_mask)]
            with (args.out_dir / 'commands.jsonl').open('a') as commands:
                commands.write(json.dumps(dict(command=command, cwd=str(root),
                    environment={k: v for k, v in env.items() if k.startswith('TLI_') or k == 'CUDA_VISIBLE_DEVICES'})) + '\n')
            with (args.out_dir / f'{index:02d}-{arm}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                observed_gpu_pid = None
                try:
                    while child.poll() is None:
                        sample = snapshot()
                        telemetry.write(json.dumps(sample | {'phase': 'running', 'arm': arm, 'pid': child.pid}) + '\n')
                        # NVML reports host PIDs outside this container's PID
                        # namespace. Infer ownership from the stable idle-to-one
                        # transition; require the same sole PID for the arm.
                        gpu_pids = {int(r.split(',')[1]) for r in sample['processes']}
                        if len(gpu_pids) > 1:
                            raise RuntimeError('GPU1 contention detected; this round is invalid')
                        if gpu_pids:
                            current = next(iter(gpu_pids))
                            if observed_gpu_pid is None:
                                observed_gpu_pid = current
                            elif current != observed_gpu_pid:
                                raise RuntimeError('GPU1 process identity changed; round invalid')
                        time.sleep(.5)
                    if child.returncode:
                        raise RuntimeError(f'model arm {arm} failed: {child.returncode}')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
    receipt['execution_complete'] = True
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
