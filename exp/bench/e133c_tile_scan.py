#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E133c-A6：打分 Triton kernel D=32 tile 参数扫描（BT × num_warps）。
从真实 mavg+tail32 链路捕获 k_min/k_max/k_avg/k_qat/b_q，固定输入后
逐参数组合计时（cudaEvent 稳态中位数，warmup 50 + 200 rep）。
粗筛（coarse，行=块数）与细筛（fine，行=token 数）分开扫；avg 粗筛
复用 fine kernel 同构。输出最优 (BT, num_warps) 供 _TILE_* 查表回填。

用法：python3 e133c_tile_scan.py [out_json]（默认 /tmp/e133c_tile_scan.json）
"""
import itertools
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import torch  # noqa: E402
import triton  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402

dev = "cuda:0"
S = 65536
ns = make_args("mavg_tail")
torch.manual_seed(20261013)
idx = TLIIndexer(ns)
idx.layer_idx = 2
idx.clear()

H, Hkv, D = 32, 8, 128
scale = D ** -0.5
g = torch.Generator(device=dev).manual_seed(777001)
k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)
q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16, generator=g)
nk = torch.randn(1, 1, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)

# 建 8 步增量把缓存建热，取最后一步真实输入
sd = None
for i in range(8):
    cu = prepare_cu_seqlens(k)
    q_ids = prepare_seqlens(cu) - 1
    idct = idx.prepare_index(k, cu)
    sd = idx.compute_score(q, q_ids, idct, scale)
    k = torch.cat([k, nk], dim=1)
q_ids = prepare_seqlens(prepare_cu_seqlens(k)) - 1
torch.cuda.synchronize()

# 捕获打分输入：重跑一次 prepare/compute 拿真实 index_dict
cu = prepare_cu_seqlens(k)
idct = idx.prepare_index(k, cu)
sd = idx.compute_score(q, q_ids, idct, scale)
k_min = idct["k_min"].squeeze(0).contiguous()
k_max = idct["k_max"].squeeze(0).contiguous()
k_avg = idct["k_avg"].squeeze(0).contiguous()
k_qat = idct["k_qat"].squeeze(0).contiguous()
DD = k_min.shape[-1]
assert DD == 32, f"期望 D=32 实际 {DD}"
T_blk = k_min.shape[0]
T_tok = k_qat.shape[0]
q_full = (q.squeeze(0).squeeze(0) * scale).to(torch.float32)
idx_sub = idx._subspace_indices(dev)
b_q = q_full[..., idx_sub].contiguous()
print(f"输入: coarse {tuple(k_min.shape)} T_blk={T_blk} / fine {tuple(k_qat.shape)} "
      f"T_tok={T_tok} / b_q {tuple(b_q.shape)}")

import sparse_attn.indexer.tli_indexer as TI  # noqa: E402


def bench(fn, warmup=50, rep=200):
    for _ in range(warmup):
        fn()
    evs = []
    for _ in range(rep):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        evs.append((a, b))
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in evs)
    return ts[len(ts) // 2]


out = {"D": DD, "coarse_T": T_blk, "fine_T": T_tok, "scan": {}}
combos = list(itertools.product([16, 32, 64, 128, 256], [1, 2, 4, 8]))
for name, kmat, which in (
    ("coarse", (k_min, k_max), "coarse"),
    ("avg(fine-kernel)", (k_avg,), "fine"),
    ("fine", (k_qat,), "fine"),
):
    rows = {}
    for bt, nw in combos:
        if which == "coarse":
            TI.TLIIndexer._TILE_COARSE = {DD: (bt, nw)}
        else:
            TI.TLIIndexer._TILE_FINE = {DD: (bt, nw)}
        try:
            if which == "coarse":
                fn = lambda: idx._fast_score_coarse(k_min, k_max, b_q, torch.bfloat16)
            else:
                fn = lambda m=kmat[0]: idx._fast_score_fine(m, b_q, torch.bfloat16)
            t = bench(fn)
            rows[f"BT{bt}_W{nw}"] = round(t, 5)
        except Exception as e:
            rows[f"BT{bt}_W{nw}"] = f"ERR {type(e).__name__}"
    best = min((v, k) for k, v in rows.items() if isinstance(v, float))
    print(f"[{name}] 最优 {best[1]} = {best[0]:.4f}ms")
    out["scan"][name] = {"rows": rows, "best": best[1], "best_ms": best[0]}

# 基线（D=128 原参数 64/4 对照，验证扫描口径合理性）
TI.TLIIndexer._TILE_COARSE = {32: (64, 4)}
TI.TLIIndexer._TILE_FINE = {32: (64, 4)}
out["baseline_64_4"] = {
    "coarse": bench(lambda: idx._fast_score_coarse(k_min, k_max, b_q, torch.bfloat16)),
    "avg": bench(lambda: idx._fast_score_fine(k_avg, b_q, torch.bfloat16)),
    "fine": bench(lambda: idx._fast_score_fine(k_qat, b_q, torch.bfloat16)),
}
print("基线 (64,4):", {k: round(v, 4) for k, v in out["baseline_64_4"].items()})

out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/e133c_tile_scan.json"
json.dump(out, open(out_path, "w"), indent=1)
print(f"saved {out_path}")
