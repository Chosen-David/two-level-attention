# -*- coding: utf-8 -*-
"""E139 纲领④ tile 参数扫参：BRG×BK×warps 小扫描（含正确性复查）。"""
import os
import sys
import time

TREE = "/tmp/wt_e138_ablation"
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402
from sparse_attn.ops import fused_attn_triton as FAT  # noqa: E402

DEV = "cuda:0"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"


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

# 参考输出（原路径）
os.environ["TLI_PREFILL_FUSED_ATTN"] = "0"
o_ref = sparse_prefill_attn(fresh(), q, k, v)
os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"

combos = [
    (32, 64, 4, 2), (32, 128, 4, 2),
    (64, 64, 4, 2), (64, 64, 8, 2),
    (64, 128, 4, 2), (64, 128, 4, 3), (64, 128, 8, 2), (64, 128, 8, 3),
    (128, 64, 8, 2), (128, 128, 8, 2), (128, 128, 8, 3),
]
best = (None, 1e9)
for brg, bk, w, st in combos:
    FAT._BRG, FAT._BK, FAT._NUM_WARPS, FAT._NUM_STAGES = brg, bk, w, st
    try:
        o = sparse_prefill_attn(fresh(), q, k, v)
        ok = torch.allclose(o_ref.float(), o.float(), rtol=1e-2, atol=1e-3)
        md = (o_ref.float() - o.float()).abs().max().item()
        t = bench(lambda: sparse_prefill_attn(fresh(), q, k, v))
        tag = "PASS" if ok else "FAIL"
        print(f"BRG={brg:4d} BK={bk:4d} w={w} st={st}: {t*1e3:7.1f} ms "
              f"{tag} maxdiff={md:.2e}", flush=True)
        if ok and t < best[1]:
            best = ((brg, bk, w, st), t)
        del o
    except Exception as e:
        print(f"BRG={brg:4d} BK={bk:4d} w={w} st={st}: EXC {type(e).__name__}: {e}", flush=True)
    torch.cuda.empty_cache()
print(f"BEST: {best[0]} -> {best[1]*1e3:.1f} ms")
