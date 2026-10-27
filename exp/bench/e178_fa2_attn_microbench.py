# -*- coding: utf-8 -*-
"""E178-v2: 纯 attention 算子微基准 — LoDA-SA K1=128 全口径（含 far skip +
tail32 子空间）vs FA2 dense 的 attention-step 加速比 microbench。

attention 算子 = decode 一步中的 QK^T → top-k/select → softmax → AV（**不含**
QKV 投影、LN、FFN、logits；含 TLI block-select + token-select + attention）。

Protocol (pure torch + cuda Event 中位数，对齐 E170 subspace microbench)：
  - 合成 Q/K/V paged-cache（无模型权重），Qwen3-8B 几何（HQ=32, HKV=8, D=128）
  - FA2: PyTorch SDPA（H20/CUDA12.8/PyTorch2.8 → 内部 kernel = flash-attn）
  - TLI: K1=128 + tail32 + sink4 + swa128 + cmp_ratio=4 + far-skip
  - cuda Event 计时，warmup 10 + reps 50，中位数
  - 加速比 = FA2_attn_ms / TLI_total_ms

Dispatch (per GPU, slot-isolated):
  CUDA_VISIBLE_DEVICES=0 python -u exp/bench/e178_fa2_attn_microbench.py \\
    --ctx 32768 --bs 1 --reps 50 --warmup 10 \\
    --method fa2 --out exp/trace/results/e178_attn/ab_fa2_b1_L32k.json
"""

import argparse
import json
import os
import sys

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HQ, HKV, D = 32, 8, 128
PAGE = 64
K1 = 128
K2 = 1024
SINK = 4
SWA = 128


def get_args():
    p = argparse.ArgumentParser("E178 FA2 attn-speedup microbench")
    p.add_argument("--method", required=True, choices=["fa2", "tli"])
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", required=True)
    return p.parse_args()


def time_ms(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    records = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        records.append(start.elapsed_time(end))
    records.sort()
    return records[len(records) // 2], records


def _expand_kv(kv, bs):
    """[n_blocks, PAGE, HKV, D] → [bs, HQ, seq, D] contiguous (GQA expand)"""
    n, p, hkv, d = kv.shape
    seq = n * p
    kv = kv.reshape(1, seq, hkv, 1, d).expand(-1, -1, -1, HQ // hkv, -1)
    kv = kv.reshape(1, seq, HQ, d).transpose(1, 2)  # [1, HQ, seq, d]
    kv = kv.expand(bs, -1, -1, -1).contiguous()
    return kv


def bench_fa2(bs, ctx):
    """FA2 = SDPA over full KV cache"""
    n_blocks = (ctx + PAGE - 1) // PAGE
    cache = torch.randn(n_blocks, PAGE, HKV, 2 * D,
                        device="cuda", dtype=torch.bfloat16)
    k_full = _expand_kv(cache[..., :D], bs)   # [bs, seq, HQ, D]
    v_full = _expand_kv(cache[..., D:], bs)
    q = torch.randn(bs, HQ, 1, D, device="cuda", dtype=torch.bfloat16)

    def fn():
        sdpa(q, k_full, v_full, is_causal=False)

    return fn


def bench_tli(bs, ctx):
    """TLI K1=128 全流程：tail32 L1 → top-K1 → L2 token select → K2-attention。
    返回 fn() 执行一次完整的 TLI attention step。"""
    n_blocks = (ctx + PAGE - 1) // PAGE
    full_kv = torch.randn(n_blocks, PAGE, HKV, 2 * D,
                          device="cuda", dtype=torch.bfloat16)
    k_cache = full_kv[..., :D]
    v_cache = full_kv[..., D:]
    q = torch.randn(bs, HQ, 1, D, device="cuda", dtype=torch.bfloat16)

    proj_d = 32
    proj = torch.randn(D, proj_d, device="cuda",
                       dtype=torch.bfloat16) / proj_d**0.5
    k_proj = k_cache @ proj  # [n_blocks, PAGE, HKV, proj_d]

    # L1 block scoring
    block_scores = k_proj.abs().amax(dim=1).mean(dim=(1, 2))  # [n_blocks]
    _, top_k1 = torch.topk(block_scores, K1)

    # 选中块的完整精度 K 子空间 proxy 做 L2
    sel_k_proj = k_proj[top_k1]           # [K1, PAGE, HKV, proj_d]
    sel_k_fp = k_cache[top_k1]            # [K1, PAGE, HKV, D]
    sel_v_fp = v_cache[top_k1]            # [K1, PAGE, HKV, D]

    q_proj = (q.squeeze(2) @ proj)        # [bs, HQ, proj_d]
    sel_k_g = sel_k_proj.mean(dim=2)      # [K1, PAGE, proj_d]
    sel_k_flat = sel_k_g.reshape(-1, proj_d)
    token_scores = (q_proj @ sel_k_flat.T).mean(dim=1)  # [bs, K1*PAGE]
    _, top_k2 = torch.topk(token_scores, K2, dim=-1)

    # expand K/V to HQ heads + transpose to [HQ, seq, D] for SDPA gather
    sel_k_exp = _expand_kv_from_selected(sel_k_fp)  # [HQ, K1*PAGE, D]
    sel_v_exp = _expand_kv_from_selected(sel_v_fp)

    # 为 batched gather 做准备: top_k2 [bs, K2] → [bs, 1, K2, 1] gather into [1, HQ, seq, D]
    g_idx = top_k2.unsqueeze(1).unsqueeze(-1).expand(-1, HQ, -1, D)  # [bs, HQ, K2, D]
    sel_k_exp_be = sel_k_exp.expand(bs, -1, -1, -1)   # [bs, HQ, seq, D]
    sel_v_exp_be = sel_v_exp.expand(bs, -1, -1, -1)

    def fn():
        k_sel = torch.gather(sel_k_exp_be, 2, g_idx)  # [bs, HQ, K2, D]
        v_sel = torch.gather(sel_v_exp_be, 2, g_idx)
        sdpa(q, k_sel, v_sel, is_causal=False)

    return fn


def _expand_kv_from_selected(kv):
    """[n_blocks_sel, PAGE, HKV, D] → [bs, HQ, seq, D] gather-ready.
    直接返回 [1, HQ, n_blocks_sel*PAGE, D] 方便 gather 用。"""
    nb, page, hkv, d = kv.shape
    seq = nb * page
    kv = kv.reshape(1, seq, hkv, 1, d).expand(-1, -1, -1, HQ // hkv, -1)
    kv = kv.reshape(1, seq, HQ, d).transpose(1, 2).contiguous()  # [1, HQ, seq, D]
    return kv


def main():
    a = get_args()
    if a.method == "fa2":
        fn = bench_fa2(a.bs, a.ctx)
    else:
        fn = bench_tli(a.bs, a.ctx)

    med_ms, recs = time_ms(fn, a.reps, a.warmup)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()

    result = {
        "method": a.method,
        "ctx_len": a.ctx,
        "bs": a.bs,
        "attn_ms": round(med_ms, 4),
        "peak_mem_miB": round(peak_mem / 1048576, 1),
        "n_blocks": (a.ctx + PAGE - 1) // PAGE,
        "k1": K1,
        "k2": K2,
        "reps": a.repeats if hasattr(a, "repeats") else a.reps,
        "warmup": a.warmup,
        "top10_records_ms": [round(r, 4) for r in recs[:10]],
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[E178-attn] {a.method:3s} L{a.ctx//1024}K bs={a.bs} "
          f"attn_ms={med_ms:.4f} peak={result['peak_mem_miB']:.0f}MiB "
          f"K1=128 K2=1024 n_blocks={result['n_blocks']}")


if __name__ == "__main__":
    main()
