# -*- coding: utf-8 -*-
"""E139 纲领④ 融合态 wall 全曲线（.187 GPU4/5 双卡交叉）：
L ∈ {8K,16K,32K,64K} × {dense SDPA, 逐组, 批量, 批量+融合}，K2=1024。
协议克隆 e139_gang23_wall_lsweep：fresh indexer 每 iter、7 iter 中位、
warmup 2、单流（PREFETCH=0）、锁卡计时。
产出：融合态 wall vs L 曲线（对比 b0f40cc 无融合版）+ 64K 用户目标档。
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


def make_qkv(L, seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh_idx(K2=1024):
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
print(f"=== E139 纲领④ 融合态 wall L 曲线 GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"

for L in (8192, 16384, 32768, 65536):
    q, k, v = make_qkv(L)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    td = bench(lambda: F.scaled_dot_product_attention(
        qt, kt, vt, is_causal=True, enable_gqa=True))
    del qt, kt, vt
    torch.cuda.empty_cache()

    os.environ["TLI_PREFILL_BATCH_SELECT"] = "0"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "0"
    t_off = bench(lambda: PF.sparse_prefill_attn(fresh_idx(), q, k, v))
    torch.cuda.empty_cache()
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
    t_bat = bench(lambda: PF.sparse_prefill_attn(fresh_idx(), q, k, v))
    torch.cuda.empty_cache()
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"
    t_fus = bench(lambda: PF.sparse_prefill_attn(fresh_idx(), q, k, v))
    torch.cuda.empty_cache()
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "0"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "0"

    print(f"L={L:6d}  dense={td*1e3:7.1f}ms  逐组={t_off*1e3:7.1f}ms  "
          f"批量={t_bat*1e3:7.1f}ms  融合={t_fus*1e3:7.1f}ms  "
          f"({t_off/t_fus:.2f}x vs 逐组 / {t_bat/t_fus:.2f}x vs 批量 / "
          f"{td/t_fus:.2f}x vs dense)", flush=True)
    del q, k, v
    torch.cuda.empty_cache()
print("=== DONE ===", flush=True)
