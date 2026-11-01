"""Kernel-level decode benchmark: NFA sparse gather vs FlashAttention-2 varlen.

This script measures only the attention kernel.  It builds a multi-request
``GraphReadyPagedRegionCache``, fills fixed-width physical gather slots, and
replays ``sparse_paged_attention_gather``.  FA2 runs the same batched decode
over full KV with causal varlen inputs.

Set ``FLASH_ATTN_WHEEL_DIR`` when flash-attn is not importable from the active
environment, e.g. the BeeGFS wheel directory extracted under /tmp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.ops.paged_region_kernels import sparse_paged_attention_gather
from sparse_attn.policies.region_geometry import RegionGeometry


def _load_flash_attn():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        return flash_attn_varlen_func
    except Exception:
        wheel_dir = os.environ.get("FLASH_ATTN_WHEEL_DIR")
        if wheel_dir:
            sys.path.insert(0, wheel_dir)
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "flash-attn is unavailable; install it or set "
                "FLASH_ATTN_WHEEL_DIR to an extracted wheel directory"
            ) from exc


def _build_cache(B, L, K, HQ, HK, D, DV, page):
    geometry = RegionGeometry(sink=4, near=8192, window=128, page=page,
                              max_new_tokens=128)
    max_blocks = ((L + page - 1) // page) + 8
    max_physical_pages = max_blocks * B + 8
    cache = GraphReadyPagedRegionCache(
        1, geometry,
        max_requests=B,
        max_physical_pages=max_physical_pages,
        max_blocks=max_blocks,
        max_visible_len=K,
        max_query_tokens=B,
        num_kv_heads=HK,
        num_query_heads=HQ,
        head_dim_k=D,
        head_dim_v=DV,
        dtype=torch.bfloat16,
        device='cuda',
    )

    req_ids = []
    k_seqs = []
    v_seqs = []
    lengths = []
    for b in range(B):
        length = max(1, L - (b % 3) * 7)
        request_id = cache.add_request()
        keys = torch.randn(1, HK, length, D, dtype=torch.bfloat16, device='cuda')
        values = torch.randn(1, HK, length, DV, dtype=torch.bfloat16, device='cuda')
        cache.write_prefill(0, request_id, keys, values)
        cache.freeze(request_id, [False], [False])
        req_ids.append(request_id)
        k_seqs.append(keys.squeeze(0).transpose(0, 1).contiguous())
        v_seqs.append(values.squeeze(0).transpose(0, 1).contiguous())
        lengths.append(length)

    cache.new_step(req_ids)
    cache.plan_common(req_ids)

    logical_ids = torch.arange(
        K, dtype=torch.int32, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
    counts = torch.full((B,), K, dtype=torch.int32, device='cuda')
    cache._store_gather_slots_batch(0, logical_ids, counts, req_ids)

    query = torch.randn(B, HQ, D, dtype=torch.bfloat16, device='cuda')
    query_to_req = torch.arange(B, dtype=torch.int32, device='cuda')
    return cache, query, query_to_req, k_seqs, v_seqs, lengths


def _measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--seq-len', type=int, nargs='+', default=[4096, 8192])
    parser.add_argument('--topk', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--kv-heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--page', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    flash_attn_varlen_func = _load_flash_attn()
    scale = args.head_dim ** -0.5
    results = []

    for B in args.batch:
        for L in args.seq_len:
            cache, q, query_to_req, k_seqs, v_seqs, lengths = _build_cache(
                B, L, args.topk, args.heads, args.kv_heads,
                args.head_dim, args.head_dim, args.page)

            layer = cache.layers[0]
            gather_indices = cache.gather_indices_buf[0, :B]
            visible_lens = cache.visible_lens_buf[0, :B]

            def sparse_fn():
                sparse_paged_attention_gather(
                    layer, q, gather_indices, visible_lens,
                    query_to_req, scale=scale)

            k_flat = torch.cat(k_seqs, dim=0).contiguous()
            v_flat = torch.cat(v_seqs, dim=0).contiguous()
            cu_q = torch.arange(B + 1, dtype=torch.int32, device='cuda')
            cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
            cu_k[1:] = torch.tensor(lengths, device='cuda').cumsum(0).to(torch.int32)
            max_len = max(lengths)

            def fa2_fn():
                flash_attn_varlen_func(
                    q, k_flat, v_flat, cu_q, cu_k,
                    1, max_len,
                    dropout_p=0.0, softmax_scale=scale, causal=True)

            t_sparse = _measure(sparse_fn, args.warmup, args.iters)
            t_fa2 = _measure(fa2_fn, args.warmup, args.iters)
            speedup = t_fa2 / t_sparse

            row = {
                'batch': B,
                'seq_len': L,
                'topk': args.topk,
                'sparse_ms': round(t_sparse, 5),
                'fa2_ms': round(t_fa2, 5),
                'speedup': round(speedup, 3),
            }
            results.append(row)
            print(
                f"B={B:2d} L={L:6d} K={args.topk:5d} | "
                f"nfa={t_sparse:.4f}ms fa2={t_fa2:.4f}ms "
                f"speedup={speedup:.2f}x"
            )

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'head_dim': args.head_dim,
                'results': results,
            }, f, indent=2)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()

