# -*- coding: utf-8 -*-
"""E139 纲领④ 快检 bench：L=32K grp64 各开关组合 wall + dense 参照。"""
import os
import sys
import time

TREE = "/tmp/wt_e138_ablation"
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402

DEV = "cuda:0"
os.environ["TLI_PREFILL_PREFETCH"] = "0"


def make_qkv(L, seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh():
    idx = TLIIndexer(make_args("mavg_tail"))
    idx.layer_idx = 2
    idx.clear()
    return idx


def bench(fn, iters=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


L = 32768
q, k, v = make_qkv(L)

def combo(batch, fused, label):
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "1" if batch else "0"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1" if fused else "0"
    t = bench(lambda: sparse_prefill_attn(fresh(), q, k, v))
    print(f"{label:34s}: {t*1e3:7.1f} ms", flush=True)
    return t

t_orig = combo(False, False, "逐组选择+原gather attention")
t_bs = combo(True, False, "批量选择+原gather attention")
t_fu = combo(True, True, "批量选择+融合attention（目标态）")
combo(False, True, "逐组选择+融合attention")

qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
td = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True))
print(f"{'dense SDPA 参照':34s}: {td*1e3:7.1f} ms", flush=True)
print(f"目标态 vs 逐组基线: {t_orig/t_fu:.2f}x | vs 批量基线: {t_bs/t_fu:.2f}x | vs dense: {td/t_fu:.2f}x")
