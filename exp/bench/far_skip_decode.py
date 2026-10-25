"""Single-layer decode microbenchmark; prints a JSON receipt, no speed claim.

Run from repo root: python -m exp.bench.far_skip_decode --device cuda --legacy
CPU mode validates the executable pipeline only. GPU/model quality are separate.
"""
import argparse
import json
import platform
import statistics
import time
from types import SimpleNamespace

import torch
import transformers
from transformers.cache_utils import DynamicCache

from sparse_attn.arguments import _add_tia_args, _add_tli_args
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.far_skip_runtime import try_far_skip_decode
from sparse_attn.ops.eager_decoding import eager_decoding_attn


@torch.inference_mode()
def run(options, length, mode, *, runtime=try_far_skip_decode, collect_outputs=False):
    torch.manual_seed(20261025)
    dev = torch.device(options.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    parser = _add_tli_args(_add_tia_args(argparse.ArgumentParser()))
    args = parser.parse_args([])
    args.tli_enable_layer_skip = args.tli_enable_kmeans = False
    args.tli_near_len, args.tli_near_pages = options.near_len, options.near_pages
    args.tli_near_tokens = "off"
    args.tia_level2_topk = options.budget
    args.tli_far_skip_backend = "legacy" if mode == "legacy" else "compact"
    args.tli_far_skip_cache = "evict" if mode == "evict" else "retain"
    idx = TLIIndexer(args)
    idx.layer_idx, idx._skip_ids = 0, {0}
    module = SimpleNamespace(indexer=idx, training=False, sliding_window=None,
                             layer_idx=0, scaling=128 ** -.5)
    cache = DynamicCache()
    k, v = [torch.randn(1, 4, length, 128, device=dev, dtype=dtype) for _ in range(2)]
    cache.update(k, v, 0)
    del k, v  # do not keep external references to full-history storage
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    wall, gpu, outputs = [], [], []
    first_ms = None
    for step in range(options.warmup + options.steps):
        q = torch.randn(1, 16, 1, 128, device=dev, dtype=dtype)
        k, v = [torch.randn(1, 4, 1, 128, device=dev, dtype=dtype) for _ in range(2)]
        if dev.type == "cuda":
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        begin = time.perf_counter()
        if mode != "legacy":
            out = runtime(module, q, k, v, cache, None, None, None)
        else:
            k, v = cache.update(k, v, 0)
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
            size = k.shape[1]
            cu = torch.tensor([0, size], dtype=torch.int32, device=dev)
            mask, bs = idx.prepare_mask(q, cu[1:] - 1, k, cu)
            out = eager_decoding_attn(q, k, v, mask, bs, cu, softmax_scale=module.scaling)
        if dev.type == "cuda":
            end.record()
            end.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        if step == 0:
            first_ms = elapsed
        if step >= options.warmup:
            wall.append(elapsed)
            if dev.type == "cuda":
                gpu.append(start.elapsed_time(end))
        if collect_outputs:
            outputs.append(out.detach().cpu().clone())
        del q, k, v
    layer = cache.layers[0]
    result = dict(mode=mode, prompt_length=length, logical_length=cache.get_seq_length(),
                  physical_tokens=layer.keys.shape[-2],
                  kv_storage_bytes=sum(x.untyped_storage().nbytes() for x in (layer.keys, layer.values)),
                  first_decode_wall_ms=first_ms, wall_ms=wall, cuda_event_ms=gpu,
                  median_wall_ms=statistics.median(wall), output_sum=float(out.float().sum()))
    if mode == "evict":
        result["live_retained_tokens"] = min(idx.sink_tokens, cache.get_seq_length()) + cache.get_seq_length() - layer.tail_start
    ws = getattr(idx, "_far_skip_workspace", None)
    if ws is not None:
        result["summary_pages_rebuilt"] = ws.pages_rebuilt
        result["workspace_bytes"] = sum(t.untyped_storage().nbytes() for t in
            list(ws._ranges.values()) + list(ws._buffers.values()) + (ws.summaries or [])
            + ([ws.full_dims] if hasattr(ws, "full_dims") else []))
    if dev.type == "cuda":
        result["cuda_peak_allocated_bytes_including_initial_cache"] = torch.cuda.max_memory_allocated()
    return (result, outputs) if collect_outputs else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--near-len", type=int, default=4096)
    parser.add_argument("--near-pages", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--legacy", action="store_true", help="also measure historical skip path (CUDA only)")
    opts = parser.parse_args()
    if opts.steps < 1 or opts.warmup < 1 or min(opts.lengths) < 1:
        parser.error("lengths, steps and warmup must be positive")
    if opts.device == "cuda" and not torch.cuda.is_available():
        parser.error("real CUDA hardware is required; no CPU timing substitution")
    if opts.legacy and opts.device != "cuda":
        parser.error("historical TLI requires CUDA")
    if opts.device == "cpu":
        torch.set_num_threads(1)
    rows = [run(opts, length, mode) for length in opts.lengths
            for mode in (["legacy"] if opts.legacy else []) + ["retain", "evict"]]
    print(json.dumps(dict(scope="single-layer synthetic decode; not model throughput or quality",
                          config=vars(opts), python=platform.python_version(), torch=torch.__version__,
                          transformers=transformers.__version__,
                          gpu=torch.cuda.get_device_name() if opts.device == "cuda" else None,
                          measurements=rows), indent=2))


if __name__ == "__main__":
    main()
