# -*- coding: utf-8 -*-
"""E140 #256 前置：TL-E139-GRAPH-CACHE-20261015 修复验收。

缺陷：batch_select graph 缓存键只含 shape/分组/方法指纹——同进程 GRAPH=1
下 α/β/γ/sink/SWA/k1/subspace 变化不改输入 shape → 旧键命中 → replay
陈旧几何（GPT CPU 复现 36/256 mask 位差）。

修复：键绑定 capture 实际固化的 host 几何真值（nb_l/sl_l/k2_far_l/
k2_near_l）+ 保护区标量 + α/β/γ/k1 + subspace/dtype/dev 指纹。

本测试（同进程配置切换硬门，fail-closed sys.exit）：
  G1 α 切换 .25→.50：graph B == 新鲜 eager B（逐位）且 != A
  G2 γ 切换 .625→.375：同上
  G3 sink/SWA 切换 128/128→192/192：同上
  G4 k1 切换 128→256（β 同）：kf/kn 变化 + k2 预算变化双重路径
  G5 复用保持：A→B→A 第三调命中 A 图（_GRAPH_CACHE 条目数核算），
     replay 结果 == 新鲜 eager A——跨层复用未被修复误伤
  G6 严格 fail-closed：任一 diff 非零 / 缓存条目数异常 → 非零退出
协议克隆 e139_bs_graph_bitdiff（.187 GPU 空隙共驻，正确性测试非计时）。
"""
import os
import sys

TREE = os.environ.get("E140_TREE", "/tmp/wt_e138_ablation")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.batch_select import batched_group_sels, _GRAPH_CACHE  # noqa: E402

DEV = "cuda:0"
L = 16384          # 16K 足够触发多 chunk 几何差异，共驻快速
D = 128


def make_qkv(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv = 32, 8
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh_idx(**over):
    ns = make_args("mavg_tail")
    ns.tia_level2_topk = 1024
    idx = TLIIndexer(ns)
    idx.layer_idx = 2
    idx.clear()
    for k, v in over.items():
        setattr(idx, k, v)
    return idx


def masks_of(idx, q, k, graph):
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1" if graph else "0"
    m = batched_group_sels(idx, q, k, D ** -0.5, group_ends, dense_below)
    assert m is not None, "选择链返回 None——fail-closed 回退，配置不支持"
    return [t.clone() for t in m]        # clone：脱离 graph 静态缓冲生命周期


def diff(m1, m2):
    return sum(1 for a, b in zip(m1, m2) if not torch.equal(a, b))


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E140 graph-cachekey 修复验收 L={L} GPU{gpu} ===", flush=True)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ["TLI_PREFILL_PREFETCH"] = "0"
os.environ.pop("TLI_PREFILL_DENSE_BELOW", None)
os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"

bs = 64
dense_below = 1024
grp = 64
chunk = 2048
pairs = []
for c0 in range(dense_below, L, chunk):
    for g0 in range(c0, min(c0 + chunk, L), grp):
        pairs.append((g0, min(g0 + grp, min(c0 + chunk, L))))
group_ends = [g1 for (_, g1) in pairs if g1 % bs == 0]
N = len(group_ends)
print(f"N={N}", flush=True)

FAIL = []

q, k = make_qkv(999)
idx_A = fresh_idx()                          # α=.25 β=.125 γ=.625 k1=128

# ---- G1-G4：A capture → 各切换 B → graph B vs 新鲜 eager B ----
mA_graph = masks_of(idx_A, q, k, True)       # capture 层（config A）
mA_eager = masks_of(idx_A, q, k, False)
d = diff(mA_graph, mA_eager)
print(f"[capture] A graph vs A eager diff = {d}/{N}", flush=True)
if d != 0:
    FAIL.append("capture-A")

switches = [
    ("G1 alpha", dict(alpha=0.50)),
    ("G2 gamma", dict(gamma=0.375)),
    ("G3 sinkswa", dict(sink_tokens=192, sliding_window_size=192)),
    ("G4 k1=256", dict()),
]
for name, over in switches:
    if name.startswith("G4"):
        idx_B = fresh_idx()
        idx_B.args.tia_level1_topk = 256     # args 级 k1（走 args 读取路径）
    else:
        idx_B = fresh_idx(**over)
    mB_graph = masks_of(idx_B, q, k, True)   # 修复后：键不同 → 新 capture
    mB_eager = masks_of(idx_B, q, k, False)  # 新鲜 eager 参考（GRAPH=0 不经缓存）
    dB = diff(mB_graph, mB_eager)
    dX = diff(mB_graph, mA_graph)            # 应与 A 不同（几何确实变了）
    n_ent = len(_GRAPH_CACHE)
    print(f"[{name}] graph-B vs eager-B diff = {dB}/{N}；"
          f"B vs A diff = {dX}/{N}（>0 预期）；cache entries = {n_ent}",
          flush=True)
    if dB != 0:
        FAIL.append(f"{name}-bitdiff")
    if dX == 0:
        FAIL.append(f"{name}-no-geom-change")

# ---- G5：复用保持 A→B→A ----
mA2_graph = masks_of(idx_A, q, k, True)      # 应回命中 A 条目 replay
d5 = diff(mA2_graph, mA_eager)
print(f"[G5 reuse] A→B→A replay vs eager-A diff = {d5}/{N}", flush=True)
if d5 != 0:
    FAIL.append("reuse-A")
# A 条目数核算：A + 4 个 B 配置 = 5 条（k1=256 的 G4 走 args 路径同为新键）
n_ent = len(_GRAPH_CACHE)
print(f"[G5] cache entries = {n_ent}（预期 5）", flush=True)
if n_ent != 5:
    FAIL.append(f"cache-entries-{n_ent}")

# ---- G6 fail-closed 汇总 ----
if FAIL:
    print(f"=== FAIL: {FAIL} ===", flush=True)
    sys.exit(1)
print("=== ALL PASS ===", flush=True)
