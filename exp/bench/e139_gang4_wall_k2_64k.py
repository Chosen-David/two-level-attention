# -*- coding: utf-8 -*-
"""E139 纲领④ 融合态 K2 上探 @64K（.187 GPU6/7 双卡交叉）：
K2 ∈ {1024,1536,2048} 融合态 wall + dense 参照——判定「64K 对 dense
净加速」主张的预算适用范围（1024 质量档 vs 2048 质量恢复档）。
"""
import os
import sys
import time

TREE = os.path.expanduser("~/tmp_e138_abl_tree_187")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops import eager_prefill as PF  # noqa: E402

DEV = "cuda:0"
L = 65536


def make_qkv(seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh_idx(K2):
    ns = make_args("mavg_tail")
    ns.tia_level2_topk = K2
    idx = TLIIndexer(ns)
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


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 融合态 K2 上探 L={L} GPU{gpu} ===", flush=True)
q, k, v = make_qkv()
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"

qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
td = bench(lambda: F.scaled_dot_product_attention(
    qt, kt, vt, is_causal=True, enable_gqa=True))
del qt, kt, vt
torch.cuda.empty_cache()
print(f"dense SDPA           = {td*1e3:7.1f}ms", flush=True)

for K2 in (1024, 1536, 2048):
    t = bench(lambda: PF.sparse_prefill_attn(fresh_idx(K2), q, k, v))
    print(f"融合态 K2={K2:5d}    = {t*1e3:7.1f}ms  ({td/t:.2f}x vs dense)", flush=True)
    torch.cuda.empty_cache()
print("=== DONE ===", flush=True)
