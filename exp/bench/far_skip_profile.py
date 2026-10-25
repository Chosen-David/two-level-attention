"""CPU operator accounting for a warmed production decode step (not GPU timing)."""
import argparse
import json
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from sparse_attn.arguments import _add_tia_args, _add_tli_args
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.far_skip_runtime import try_far_skip_decode
from .far_skip_compare import load_baseline


@torch.inference_mode()
def measure(runtime):
    torch.manual_seed(31)
    args = _add_tli_args(_add_tia_args(argparse.ArgumentParser())).parse_args([])
    args.tli_enable_layer_skip = args.tli_enable_kmeans = False
    args.tli_near_len, args.tli_near_pages, args.tli_near_tokens = 4096, 32, 'off'
    args.tli_far_skip_backend, args.tli_far_skip_cache = 'compact', 'evict'
    idx = TLIIndexer(args)
    idx.layer_idx, idx._skip_ids = 0, {0}
    module = SimpleNamespace(indexer=idx, layer_idx=0, training=False,
                             sliding_window=None, scaling=128**-.5)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 4, 8192, 128)
    cache.update(k, v, 0)
    del k, v
    for _ in range(8):
        q = torch.randn(1, 16, 1, 128)
        k, v = torch.randn(2, 1, 4, 1, 128)
        runtime(module, q, k, v, cache, None, None, None)
    q = torch.randn(1, 16, 1, 128)
    k, v = torch.randn(2, 1, 4, 1, 128)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True, record_shapes=True) as profile:
        runtime(module, q, k, v, cache, None, None, None)
    counts = {e.key: e.count for e in profile.key_averages()}
    events = profile.events()
    # Self memory excludes child allocations to avoid double-counting.
    allocated = sum(max(0, e.self_cpu_memory_usage) for e in events)
    return dict(operator_counts=counts, positive_self_cpu_allocated_bytes=allocated,
                copy_shapes=[e.input_shapes for e in events if e.name == 'aten::copy_'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(dict(scope='one warm CPU decode at S=8201; counts are not GPU launches',
                          v1=measure(load_baseline(args.baseline_root)),
                          optimized=measure(try_far_skip_decode)), indent=2))


if __name__ == '__main__':
    main()
