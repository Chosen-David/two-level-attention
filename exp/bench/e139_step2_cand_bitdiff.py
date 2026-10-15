# -*- coding: utf-8 -*-
"""E139 #253 Step2 验收：L2 双池 topk 候选列化（env TLI_PREFILL_L2_TOPK_CAND）
硬门 1：Step2（CAND=1）vs 旧路径（CAND=0）mask 逐位一致——eager 双臂
  （同输入同 indexer，唯一差异 = ④ 步 topk 实现；①②③ 段 gather/scatter
  与候选序无关 → 数值不变，diff 隔离在 topk 等价性）。
硬门 2：Step2 + GRAPH=1 capture/replay vs Step2 eager 逐位一致（分支免化
  设计——nbad host 常量分支 + 固定形状补位 kernel 进 capture）。
硬门 3：默认路径零扰动（CAND=0 同输入两次 diff=0——回归口径）。
性能：选择链 wall 四格（eager/graph × CAND 0/1，K2=1024 生产口径）。
多 K2 档：1024 生产（nbad=0）+ 8192 补位应力（触发 k>cnt 零补位仿真）。
协议克隆 e139_bs_graph_bitdiff。
"""
import os
import sys
import time

from e139_validation_gates import require_equal_masks

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


def make_qkv(seed, L_=L):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv = 32, 8
    return (torch.randn(1, L_, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L_, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L_, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


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


def group_ends_for(L_):
    bs, dense_below, grp, chunk = 64, 1024, 64, 2048
    bounds = [(0, dense_below)] + [
        (c0, min(c0 + chunk, L_)) for c0 in range(dense_below, L_, chunk)
    ]
    pairs = []
    for (bc0, bc1) in bounds:
        if bc0 >= dense_below:
            for bg0 in range(bc0, bc1, grp):
                pairs.append((bg0, min(bg0 + grp, bc1)))
    return [g1 for (_, g1) in pairs if g1 % bs == 0]


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 #253 Step2 验收 L={L} GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
os.environ.pop("TLI_PREFILL_L2_TOPK_CAND", None)

from sparse_attn.ops.batch_select import batched_group_sels  # noqa: E402
import sparse_attn.ops.batch_select as _bs  # noqa: E402

# K2=1024 生产口径（nbad=0 常规路径）；K2=8192 补位应力档
# （K2_mid=7872 > kf·64=7168 → 宽 far 池行 nbad=k-cnt>0，触发
# _zero_pad_pool 补位仿真分支——K2≤2048 时 K2_mid<7168 恒不触发）
def validate(K2):
    print(f"---- K2 = {K2} ----", flush=True)
    group_ends = group_ends_for(L)
    idx = fresh_idx(K2)
    q, k, _ = make_qkv(999)
    if not _gate_ok(idx, q, k):
        raise SystemExit(f"validation failed: K2={K2} selector is not eligible")
    SS = D ** -0.5

    # ---- 硬门 3：默认路径零扰动 ----
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
    m_a1 = batched_group_sels(idx, q, k, SS, group_ends, 1024)
    m_a2 = batched_group_sels(idx, q, k, SS, group_ends, 1024)
    require_equal_masks(f"[G3 K2={K2}] CAND=0 eager 同输入两次",
                        m_a1, m_a2, group_ends)

    # ---- 硬门 1：Step2 vs 旧路径（eager 双臂）----
    os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "1"
    m_s2 = batched_group_sels(idx, q, k, SS, group_ends, 1024)
    os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "0"
    require_equal_masks(f"[G1 K2={K2}] CAND=1 vs CAND=0",
                        m_a1, m_s2, group_ends)

    # ---- 硬门 2：Step2 + graph capture/replay vs Step2 eager ----
    os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "1"
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
    _bs._GRAPH_CACHE.clear()        # 排除 Step1 缓存（key 已含 CAND 指纹，防御性）
    m_g0 = batched_group_sels(idx, q, k, SS, group_ends, 1024)
    require_equal_masks(f"[G2 K2={K2}] CAND=1 capture 层 graph vs eager",
                        m_s2, m_g0, group_ends)
    for layer in range(1, 3):
        seed = 2000 + layer
        q_l, k_l, _ = make_qkv(seed)
        idx_l = fresh_idx(K2)
        os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
        m_e = batched_group_sels(idx_l, q_l, k_l, SS, group_ends, 1024)
        os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
        m_r = batched_group_sels(idx_l, q_l, k_l, SS, group_ends, 1024)
        require_equal_masks(f"[G2 K2={K2}] 层 {layer}（seed {seed}）replay vs eager",
                            m_e, m_r, group_ends)
    _bs._GRAPH_CACHE.clear()
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"

    os.environ.pop("TLI_PREFILL_L2_TOPK_CAND", None)


# Complete BOTH budget gates before any performance measurement. Function scope
# releases validation tensors before creating the production benchmark inputs.
for K2 in (1024, 8192):
    validate(K2)
print("=== TOTAL FAIL = 0 ===", flush=True)

# ---- 性能：选择链 wall 四格（K2=1024 生产口径）----
idx = fresh_idx(1024)
q, k, _ = make_qkv(999)
group_ends = group_ends_for(L)
SS = D ** -0.5
os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "0"
t_e0 = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, 1024))
os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "1"
t_e1 = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, 1024))
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1"
_bs._GRAPH_CACHE.clear()
batched_group_sels(idx, q, k, SS, group_ends, 1024)   # capture
t_g1 = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, 1024))
_bs._GRAPH_CACHE.clear()
os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "0"
batched_group_sels(idx, q, k, SS, group_ends, 1024)
t_g0 = bench(lambda: batched_group_sels(
    idx, q, k, SS, group_ends, 1024))
_bs._GRAPH_CACHE.clear()
os.environ["TLI_PREFILL_SELECT_GRAPH"] = "0"
print(f"选择链 wall: eager 旧 = {t_e0*1e3:7.1f}ms / "
      f"eager Step2 = {t_e1*1e3:7.1f}ms / "
      f"graph 旧 = {t_g0*1e3:7.1f}ms / "
      f"graph Step2 = {t_g1*1e3:7.1f}ms", flush=True)
os.environ.pop("TLI_PREFILL_L2_TOPK_CAND", None)
print("=== DONE ===", flush=True)
