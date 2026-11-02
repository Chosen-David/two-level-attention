"""End-to-end per-decode-step benchmark: NFA selector + sparse attention vs FA2.

This is a closer analogue of one model decode step than the pure-kernel
benchmark.  For every layer it:
  * runs the layer-batched NFA selector once (T-1 Q -> physical gather slots),
  * replays sparse_paged_attention_gather for each layer,
and compares that total against running FlashAttention-2 varlen for each layer.

It also reports the T-1 prediction IoU: the selected physical-slot mask
computed from the previous-step Q versus the same selector run with the
current-step Q.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import torch

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.ops.paged_region_kernels import sparse_paged_attention_gather
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.indexer.nfa_indexer import NFAIndexer
from sparse_attn.indexer import nfa_layer_batch_selector
from sparse_attn.policies.region_geometry import RegionGeometry


def _load_flash_attn():
    try:
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func
    except Exception:
        wheel_dir = os.environ.get("FLASH_ATTN_WHEEL_DIR")
        if wheel_dir:
            sys.path.insert(0, wheel_dir)
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func


def _build(B, L, HQ, HK, D, DV, page, layers):
    geometry = RegionGeometry(sink=4, near=4096, window=128, page=page,
                              max_new_tokens=128)
    max_blocks = ((L + page - 1) // page) + 8
    max_physical_pages = max_blocks * B + 8
    cache = GraphReadyPagedRegionCache(
        layers, geometry,
        max_requests=B,
        max_physical_pages=max_physical_pages,
        max_blocks=max_blocks,
        max_visible_len=8192,
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
        rid = cache.add_request()
        keys = torch.randn(1, HK, length, D, dtype=torch.bfloat16, device='cuda')
        values = torch.randn(1, HK, length, DV, dtype=torch.bfloat16, device='cuda')
        for li in range(layers):
            cache.write_prefill(li, rid, keys, values)
        cache.freeze(rid, [False] * layers, [False] * layers)
        cache.states[rid].first_decode_seeded = True
        req_ids.append(rid)
        k_seqs.append(keys.squeeze(0).transpose(0, 1).contiguous())
        v_seqs.append(values.squeeze(0).transpose(0, 1).contiguous())
        lengths.append(length)

    cache.new_step(req_ids)
    cache.plan_common(req_ids)

    idx = NFAIndexer(nfa_args())
    idx.layer_idx = 0
    attn = SimpleNamespace(indexer=idx, scaling=D ** -0.5)
    model_layers = [SimpleNamespace(self_attn=attn, layer_idx=i)
                    for i in range(layers)]
    model = SimpleNamespace(model=SimpleNamespace(layers=model_layers))

    # T-1 previous-frame per-layer Q.
    for li in range(layers):
        for bi, rid in enumerate(req_ids):
            cache._last_q_buf[li, rid].normal_()

    q_current = [torch.randn(B, HQ, D, dtype=torch.bfloat16, device='cuda')
                 for _ in range(layers)]
    query_to_req = torch.arange(B, dtype=torch.int32, device='cuda')
    return (cache, model, req_ids, k_seqs, v_seqs, lengths,
            q_current, query_to_req)


def _mask_sets(cache, req_ids):
    """Return a set of selected physical slots for every (layer, request)."""
    masks = []
    for li in range(cache.n_layers):
        for bi, rid in enumerate(req_ids):
            count = int(cache.visible_lens_buf[li, bi].item())
            ids = cache.gather_indices_buf[li, bi, :count].tolist()
            masks.append(set(x for x in ids if x >= 0))
    return masks


def _iou(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


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
    os.environ.setdefault('NFA_LAYER_BATCH_SELECTOR', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--seq-len', type=int, nargs='+', default=[4096])
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--kv-heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--page', type=int, default=64)
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    flash_attn_varlen_func = _load_flash_attn()
    scale = args.head_dim ** -0.5
    results = []

    for B in args.batch:
        for L in args.seq_len:
            (cache, model, req_ids, k_seqs, v_seqs, lengths,
             q_current, query_to_req) = _build(
                B, L, args.heads, args.kv_heads, args.head_dim,
                args.head_dim, args.page, args.layers)

            padded_ids = req_ids + [-1] * (B - len(req_ids)) if B > len(req_ids) else req_ids
            batch = B

            def selection_fn():
                nfa_layer_batch_selector.layer_batched_select(
                    cache, model, padded_ids)

            selection_fn()
            torch.cuda.synchronize()
            base_masks = _mask_sets(cache, req_ids)

            # current-Q reference masks for IoU
            for li in range(cache.n_layers):
                for bi, rid in enumerate(req_ids):
                    cache._last_q_buf[li, rid].copy_(q_current[li][bi])
            selection_fn()
            torch.cuda.synchronize()
            cur_masks = _mask_sets(cache, req_ids)
            # restore T-1 Q for timing path
            for li in range(cache.n_layers):
                for bi, rid in enumerate(req_ids):
                    cache._last_q_buf[li, rid].normal_()
            selection_fn()

            ious = [_iou(a, b) for a, b in zip(base_masks, cur_masks)]
            mean_iou = sum(ious) / len(ious)

            def nfa_step_fn():
                nfa_layer_batch_selector.layer_batched_select(
                    cache, model, padded_ids)
                for li in range(cache.n_layers):
                    sparse_paged_attention_gather(
                        cache.layers[li], q_current[li],
                        cache.gather_indices_buf[li, :batch],
                        cache.visible_lens_buf[li, :batch],
                        query_to_req, scale=scale)

            k_flat = torch.cat(k_seqs, dim=0).contiguous()
            v_flat = torch.cat(v_seqs, dim=0).contiguous()
            cu_q = torch.arange(B + 1, dtype=torch.int32, device='cuda')
            cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
            cu_k[1:] = torch.tensor(lengths, device='cuda').cumsum(0).to(torch.int32)
            max_len = max(lengths)

            def fa2_step_fn():
                for li in range(cache.n_layers):
                    flash_attn_varlen_func(
                        q_current[li], k_flat, v_flat, cu_q, cu_k,
                        1, max_len,
                        dropout_p=0.0, softmax_scale=scale, causal=True)

            t_nfa = _measure(nfa_step_fn, args.warmup, args.iters)
            t_fa2 = _measure(fa2_step_fn, args.warmup, args.iters)
            speedup = t_fa2 / t_nfa

            row = {
                'batch': B,
                'seq_len': L,
                'layers': args.layers,
                't1_iou_mean': round(mean_iou, 5),
                'nfa_step_ms': round(t_nfa, 5),
                'fa2_step_ms': round(t_fa2, 5),
                'speedup': round(speedup, 3),
            }
            results.append(row)
            print(
                f"B={B:2d} L={L:6d} layers={args.layers:2d} | "
                f"IoU={mean_iou:.4f} | nfa={t_nfa:.4f}ms fa2={t_fa2:.4f}ms "
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
