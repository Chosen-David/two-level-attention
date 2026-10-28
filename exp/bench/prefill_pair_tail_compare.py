"""CUDA-only observer comparison; not decode TPOT or end-to-end model timing."""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer


def baseline(q):
    qm = q.detach().float()[:, :, -256:].reshape(-1, q.shape[-1]).abs().mean(0)
    pairs = torch.topk(qm[:64] + qm[64:], 16).indices
    return torch.cat([pairs, pairs + 64]).sort().values.to(q.device)


def candidate(q):
    state = SimpleNamespace(_pair_idx=None, static_pair=True, layer_idx=None)
    TLIIndexer.observe_prefill_q(state, q)
    return state._pair_idx


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=int, nargs='+', default=[32768, 131072])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; no CPU timing substituted')
    torch.manual_seed(784)
    result = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                  source_sha256=hashlib.sha256(inspect.getsource(TLIIndexer.observe_prefill_q).encode()).hexdigest(),
                  scope='prefill observer only, bf16 transposed Q, static_pair enabled', cases=[])
    for s in args.lengths:
        q = torch.randn(args.batch, s, args.heads, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
        assert torch.equal(baseline(q), candidate(q))
        for _ in range(3):
            baseline(q); candidate(q)
        torch.cuda.synchronize()
        records = []
        for i in range(args.repeats):
            for name in (['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline']):
                fn = baseline if name == 'baseline' else candidate
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                start.record()
                out = fn(q)
                end.record()
                end.synchronize()
                records.append(dict(arm=name, event_ms=start.elapsed_time(end),
                                    wall_ms=(time.perf_counter()-t0)*1000,
                                    peak_allocated_delta=torch.cuda.max_memory_allocated()-before))
                del out
        result['cases'].append(dict(shape=list(q.shape), stride=list(q.stride()), samples=records))
        del q
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
