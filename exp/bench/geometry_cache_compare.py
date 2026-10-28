"""Paired tiny-Qwen eager/Graph validation; not production-model TPOT evidence."""
import argparse
import contextlib
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import torch
import sparse_attn.indexer.tli_indexer as tli


def legacy_functions():
    """Frozen main8f213e4/PR19e69bc cache policy; independent dictionary per arm."""
    cache = {}
    def lookup(valid_len, key):
        try:
            version = valid_len._version
        except RuntimeError:
            return None
        entry = cache.get(valid_len.data_ptr())
        if entry is not None and entry[1] == version and entry[2] == key:
            return entry[3]
        return None
    def store(valid_len, key, geo):
        try:
            version = valid_len._version
        except RuntimeError:
            return geo
        if len(cache) > 8:
            cache.clear()
        cache[valid_len.data_ptr()] = (valid_len, version, key, geo)
        return geo
    return lookup, store


@contextlib.contextmanager
def policy(functions):
    with patch.object(tli, '_graph_geo_lookup', functions[0]), patch.object(tli, '_graph_geo_store', functions[1]):
        yield


def tiny_model(device='cpu', dtype=torch.float32):
    # Reuse the existing actual-HF fixture; no model download or synthetic kernel.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
    from test_far_skip_hybrid_cache import make_model
    from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward
    model = make_model(device=device).to(dtype)
    first, second = [layer.self_attn for layer in model.model.layers]
    second.indexer = copy.deepcopy(first.indexer)
    second.indexer.layer_idx = 1
    second.indexer._skip_ids = None
    second.indexer.dyn_dgate = True
    second.indexer.dyn_dgate_eps = -1.0
    second.forward = MethodType(qwen3_attn_forward, second)
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--batches', type=int, nargs='+', default=[2, 8, 10, 32])
    p.add_argument('--mode', choices=['eager', 'graph', 'both'], default='both')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; CPU is not a performance substitute')
    candidate = tli._graph_geo_lookup, tli._graph_geo_store
    models = [tiny_model('cuda', torch.float16) for _ in range(2)]
    from test_hybrid_batch_cache import build
    samples = []
    for mode in (['eager', 'graph'] if a.mode == 'both' else [a.mode]):
        for batch in a.batches:
            for repeat in range(a.repeats):
                lengths = [23+b % 5 for b in range(batch)]
                arms = [legacy_functions(), candidate]
                runners = [build(m, lengths, [False]*batch, capacity=max(lengths)+a.steps+8) for m in models]
                for runner, funcs in zip(runners, arms):
                    with policy(funcs):
                        for _ in range(3):
                            lengths_w = [n+1 for n in runner.ctx.seq_lens]
                            runner.inject([3]*batch, lengths_w)
                            runner.run()
                        runner.inject([4]*batch, [n+1 for n in runner.ctx.seq_lens])
                        if mode == 'graph':
                            runner.capture()
                        runner.run()
                lengths = list(runners[0].ctx.seq_lens)
                for step in range(a.steps):
                    alive = [not (step >= a.steps//2 and b % 3 == 0) for b in range(batch)]
                    lengths = [n+int(live) for n, live in zip(lengths, alive)]
                    outputs = {}
                    for arm in ([0, 1] if step % 2 == 0 else [1, 0]):
                        with policy(arms[arm]):
                            torch.cuda.synchronize()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            wall = time.perf_counter()
                            start.record()
                            runners[arm].inject([(step+b)%32 for b in range(batch)], lengths, alive)
                            output = runners[arm].run().logits
                            end.record()
                            end.synchronize()
                            samples.append(dict(mode=mode, batch=batch, repeat=repeat, step=step,
                                                arm=['legacy', 'request_cache'][arm],
                                                wall_ms=(time.perf_counter()-wall)*1000,
                                                event_ms=start.elapsed_time(end)))
                            outputs[arm] = output.clone()
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                del runners, arms, output, outputs, runner
    source = Path(tli.__file__).read_bytes()
    a.out.write_text(json.dumps(dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(source).hexdigest(),
        scope='tiny random 2-layer Qwen; paired whole-step incl inject, excludes setup; not production TPOT',
        samples=samples), indent=2)+'\n')


if __name__ == '__main__':
    main()
