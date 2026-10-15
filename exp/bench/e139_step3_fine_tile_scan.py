# -*- coding: utf-8 -*-
"""E139 #253 Step3 预扫描：fine kernel D=32 tile 对 prefill 选择链形状的适配。
E133c A6 的 (16,1) 是 decode 形状（T 小、单次调用）调参；prefill 选择链的
fine 调用形状 = L1 near 段全前缀 + L2 候选 [C, Hkv, 32]（C≤8192）。
候选 tile 逐个 patch TLIIndexer._TILE_FINE[32]，计时 batched_group_sels
全链（7-iter 中位）+ mask 逐位对拍（BT 只影响 T 维 grid 分块，不改
per-output D 维归约树 → 数值上逐位安全，实测须 diff_rows=0）。"""
import os
import sys
import time

TREE = os.environ.get("E139_TREE",
                      os.path.expanduser("~/tmp_e138_abl_tree_187"))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.batch_select import _gate_ok  # noqa: E402

DEV = "cuda:0"
L = 65536
D = 128


def make_qkv(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv = 32, 8
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
print(f"=== E139 #253 Step3 fine tile 扫描 GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
os.environ.pop("TLI_PREFILL_SELECT_GRAPH", None)   # eager 路径扫 tile

bs = 64
dense_below = 1024
grp = 64
chunk = 2048
bounds = [(0, dense_below)] + [
    (c0, min(c0 + chunk, L)) for c0 in range(dense_below, L, chunk)
]
pairs = []
for (bc0, bc1) in bounds:
    if bc0 >= dense_below:
        for bg0 in range(bc0, bc1, grp):
            pairs.append((bg0, min(bg0 + grp, bc1)))
group_ends = [g1 for (_, g1) in pairs if g1 % bs == 0]
N = len(group_ends)
print(f"N={N}", flush=True)

from sparse_attn.ops.batch_select import batched_group_sels  # noqa: E402
from sparse_attn.ops import batch_select as _bs  # noqa: E402

idx = fresh_idx()
q, k, v = make_qkv(999)
assert _gate_ok(idx, q, k)
SS = D ** -0.5

m_base = batched_group_sels(idx, q, k, SS, group_ends, dense_below)
assert m_base is not None

# ---- graph 稳态扫描（PoC A 段 32.2ms 是 graph 化 GPU 地板口径）----
# 每 tile 清 _GRAPH_CACHE 重 capture（replay 固化 capture 时的 kernel 参数）。
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
TILES = [(16, 1), (32, 1), (32, 2), (64, 1), (64, 2), (64, 4),
         (128, 1), (128, 2), (128, 4), (256, 2), (256, 4), (256, 8)]
results = []
for (bt, nw) in TILES:
    TLIIndexer._TILE_FINE[32] = (bt, nw)
    _bs._GRAPH_CACHE.clear()
    try:
        m = batched_group_sels(idx, q, k, SS, group_ends, dense_below)
        assert m is not None
        d = sum(1 for a, b in zip(m, m_base) if not torch.equal(a, b))
        t = bench(lambda: batched_group_sels(
            idx, q, k, SS, group_ends, dense_below))
    except Exception as e:
        print(f"tile=({bt:3d},{nw}) EXCEPTION: {e}", flush=True)
        continue
    results.append((t, bt, nw, d))
    print(f"[graph] tile=({bt:3d},{nw}) diff_rows={d}/{N} "
          f"wall={t*1e3:7.1f}ms", flush=True)

os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
TLIIndexer._TILE_FINE[32] = (16, 1)   # 恢复默认，进程内不留污染

results.sort()
if results:
    t, bt, nw, d = results[0]
    print(f"[graph] 最优 tile=({bt},{nw}) wall={t*1e3:.1f}ms "
          f"diff_rows={d}（vs 基线 (16,1)）", flush=True)
print("=== DONE ===", flush=True)
