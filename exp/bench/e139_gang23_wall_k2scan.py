# -*- coding: utf-8 -*-
"""E139 纲领②③ wall 补测：批量版 K2 上探曲线（1024/1536/2048）
+ dense SDPA 参照。协议克隆 e139_gang23_wall_bench：fresh indexer 每 iter、
7 iter 中位、warmup 2、单流（PREFETCH=0）、锁卡 GPU0 计时。
产出：批量版 wall 随 K2 变化（列宽 ×1/×1.5/×2），并入三轴 trade-off 表。
"""
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
from sparse_attn.ops import eager_prefill as PF  # noqa: E402

DEV = "cuda:0"
L = 32768


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
print(f"=== E139 纲领②③ wall K2 上探（批量版）L={L} GPU{gpu} ===", flush=True)
q, k, v = make_qkv()
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)

# dense SDPA 参照
qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
td = bench(lambda: F.scaled_dot_product_attention(
    qt, kt, vt, is_causal=True, enable_gqa=True))
del qt, kt, vt
torch.cuda.empty_cache()

os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
results = {}
for K2 in (1024, 1536, 2048):
    t = bench(lambda: PF.sparse_prefill_attn(fresh_idx(K2), q, k, v))
    results[K2] = t
    print(f"批量版 K2={K2:5d}    = {t*1e3:7.1f}ms  ({td/t:.2f}x vs dense)", flush=True)
    torch.cuda.empty_cache()
os.environ["TLI_PREFILL_BATCH_SELECT"] = "0"

print(f"dense SDPA           = {td*1e3:7.1f}ms", flush=True)
print(f"K2 敏感度 1024→2048 = {results[2048]/results[1024]:.2f}x", flush=True)
print("=== DONE ===", flush=True)
