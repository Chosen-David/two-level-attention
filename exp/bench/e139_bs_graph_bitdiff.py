# -*- coding: utf-8 -*-
"""E139 #253 Step1 验收：batch_select graph 路径（env TLI_PREFILL_SELECT_GRAPH）
硬门 1：多层注入不同 q/k，graph replay vs eager 逐位一致（mask 级）
硬门 2：默认 env（GRAPH=0）eager 路径与重构后 body 提取零扰动（同输入两次）
性能：选择链 wall GRAPH=0 vs GRAPH=1（7-iter 中位，首层 capture 不计稳态）
协议克隆 e139_selchain_inner_seg（.187 GPU4 锁卡）。
"""
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
from e139_graph_checks import require_graph
from e139_validation_gates import require_equal_masks
from sparse_attn.ops.batch_select import _GRAPH_CACHE, _GRAPH_STATS
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
    def checked():
        out = fn()
        if out is None or len(out) != N:
            raise RuntimeError("timed selector fell back or returned incomplete masks")
        return out
    for _ in range(2):
        checked()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        checked()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 #253 Step1 验收 L={L} GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"

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

idx = fresh_idx()
q, k, v = make_qkv(999)
if not _gate_ok(idx, q, k):
    raise SystemExit("validation failed: selector is not eligible")
_GRAPH_CACHE.clear()
SS = D ** -0.5

from sparse_attn.ops.batch_select import batched_group_sels  # noqa: E402

# ---- 硬门 2：默认路径零扰动（同输入两次 eager）----
os.environ.pop("TLI_PREFILL_SELECT_GRAPH", None)
m_a1 = batched_group_sels(idx, q, k, SS, group_ends, dense_below)
m_a2 = batched_group_sels(idx, q, k, SS, group_ends, dense_below)
require_equal_masks("[G2] eager repeat", m_a1, m_a2, group_ends)
d0 = sum(1 for a, b in zip(m_a1, m_a2) if not torch.equal(a, b))
print(f"[G2] eager 同输入两次 diff_rows = {d0}/{N}", flush=True)

# ---- 硬门 1：graph 路径多层注入 vs eager 逐位 ----
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
# 层 0（capture 层）：graph 首调 = capture 执行结果
before = dict(_GRAPH_STATS)
m_g0 = batched_group_sels(idx, q, k, SS, group_ends, dense_below)
require_graph(before, _GRAPH_STATS, "capture")
require_equal_masks("[G1] capture", m_a1, m_g0, group_ends)
d_cap = sum(1 for a, b in zip(m_g0, m_a1) if not torch.equal(a, b))
print(f"[G1] capture 层 graph vs eager diff_rows = {d_cap}/{N}", flush=True)

for layer in range(1, 4):
    seed = 1000 + layer
    q_l, k_l, _ = make_qkv(seed)
    idx_l = fresh_idx()      # e2e 口径：每层独立 indexer 实例（跨实例共享 graph）
    # eager 先算（独立内存）
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
    m_e = batched_group_sels(idx_l, q_l, k_l, SS, group_ends, dense_below)
    require_equal_masks("[G1] eager input", m_e, m_e, group_ends)
    # graph replay（copy_ 注入 + replay）
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
    before = dict(_GRAPH_STATS)
    m_r = batched_group_sels(idx_l, q_l, k_l, SS, group_ends, dense_below)
    require_graph(before, _GRAPH_STATS, "replay")
    require_equal_masks("[G1] replay", m_e, m_r, group_ends)
    d = sum(1 for a, b in zip(m_e, m_r) if not torch.equal(a, b))
    print(f"[G1] 层 {layer}（seed {seed}，新 idx 实例）replay vs eager "
          f"diff_rows = {d}/{N}", flush=True)

# ---- 性能：选择链 wall（稳态 replay，不含首层 capture）----
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
t_eager = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, dense_below))
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
before = dict(_GRAPH_STATS)
t_graph = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, dense_below))
require_graph(before, _GRAPH_STATS, "replay")
if (_GRAPH_STATS["replay"] - before["replay"] != 9
        or _GRAPH_STATS["capture"] != before["capture"]):
    raise RuntimeError("timed Graph benchmark requires nine warm-cache replays")
print(f"选择链 wall: eager = {t_eager*1e3:7.1f}ms / graph = {t_graph*1e3:7.1f}ms"
      f"（speedup = {t_eager/t_graph:.2f}x）", flush=True)
print("=== DONE ===", flush=True)
