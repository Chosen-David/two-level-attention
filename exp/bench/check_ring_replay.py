"""Paired fixed-shape ring selector+attention replay, no model TPOT claim."""
import argparse
import json
import os
from pathlib import Path
import torch
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
from request_gpu_monitor import GPU0Monitor


def run(cli):
    torch.manual_seed(884)
    cfg = FarSkipConfig(64, 4096, 32, 1024, 4, 128, 'avg', False)
    layout = FarSkipLayout(4, 4287)
    q = torch.randn(1, 1, 64, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, layout.capacity, 8, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    length = torch.tensor([32613], device='cuda', dtype=torch.int64)
    dims = torch.arange(96, 128, device='cuda')
    arms = [('off', 'off', 'simt'), ('serial', 'serial', 'simt'),
            ('overlap', 'overlap', 'simt'), ('tc', 'serial', 'tc')]
    captures, workspaces = {}, {}
    with torch.inference_mode():
        for name, mode, backend in arms:
            os.environ['TLI_OPERATOR_MODE'] = mode
            os.environ['TLI_OPERATOR_BACKEND'] = backend
            os.environ['TLI_OPERATOR_SELECTED_STAGES'] = '1'
            ws = FarSkipGraphWorkspace(cfg, layout, 8, dims, dims)
            ws.initialize(k, 32613)
            for _ in range(3): ws(q, k, v, length, 128**-.5)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): output = ws(q, k, v, length, 128**-.5)
            captures[name] = graph, output
            workspaces[name] = ws
        checks = []
        # Different queries and last-slot writes; immutable logical length keeps
        # each seeded summary's ownership intact. Reference uses separate state.
        for sample in range(8):
            q.normal_()
            slot = layout.slot(32612)
            k[:, slot].normal_(); v[:, slot].normal_()
            for graph, _ in captures.values(): graph.replay()
            reference = captures['off'][1]
            for name in captures:
                actual = captures[name][1]
                torch.testing.assert_close(actual, reference, atol=.003, rtol=.02)
                checks.append(dict(sample=sample, arm=name,
                    max_abs_error=(actual.float()-reference.float()).abs().max().item()))
        for graph, _ in captures.values():
            for _ in range(30): graph.replay()
        torch.cuda.synchronize()
        pairs = []
        def measure(name):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(cli.replays): captures[name][0].replay()
            end.record(); end.synchronize()
            return start.elapsed_time(end)/cli.replays
        for pair in range(cli.pairs):
            for candidate in ('serial', 'overlap', 'tc'):
                order = ['off', candidate] if pair % 2 == 0 else [candidate, 'off']
                values = {name: measure(name) for name in order}
                pairs.append(dict(pair=pair, candidate=candidate, order=order, gpu_ms=values))
        result = dict(status='pending_independent_verification', seed=884,
            scope='synthetic B1 fixed-shape ring selector+attention GPU replay only; no model TPOT/quality claim',
            geometry=dict(length=32613, capacity=layout.capacity, hq=64, hkv=8, d=128, fine=32),
            pairs=pairs, replays_per_observation=cli.replays, correctness=checks,
            routes={name: ws.attention_route for name, ws in workspaces.items()})
        Path(cli.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('output')
    p.add_argument('--pairs', type=int, default=30); p.add_argument('--replays', type=int, default=40)
    cli = p.parse_args()
    if cli.pairs < 1 or cli.replays < 1: p.error('positive repetition counts required')
    with GPU0Monitor(Path(cli.output).with_suffix('.gpu.json')): run(cli)
