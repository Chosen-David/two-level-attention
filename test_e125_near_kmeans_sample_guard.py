#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_e125_near_kmeans_sample_guard.py —— near 侧 kmeans 4 倍样本守卫门禁（CPU only，0 SKIP）。

【缺陷（DeepSeek 审计 round1 ③，2026-10-11，主 AI 派单确认）】
far 侧 _build_far_kmeans 有 4 倍样本守卫（Tfar < far_clusters*4 →
放弃建簇 _km_centroids=None 回退细筛分），near 侧 _update_near_cluster
的 kmeans 分支复用 far_clusters 作簇数（256 簇固定），却只有 Tn<256
下限守卫：Tn ∈ [256, far_clusters*4)（far_clusters=256 时 [256,1024)）
时以 256 簇对不足样本跑 kmeans —— 簇数逼近/≥样本数，退化为一簇一
样本/大量空簇，聚类完全失效但代码静默继续（LB 短序列 near 区可落入）。

【修复（本门禁守护的行为）】_update_near_cluster 补与 far 侧同口径
4 倍守卫：near_select="cluster" 且 Tn < far_clusters*4 时跳过 kmeans
更新，沿用本函数 Tn<256 的跳过惯例（_km_near_centroids 保持 None，
消费端回退细筛分——细筛分质量高于簇代表分，语义无损偏保守）。
sim_greedy 分支不受影响：贪心簇数自适应（Kcap=Tn，活簇随数据增长），
不存在簇数≥样本数退化，与 far 侧 greedy 无 4 倍守卫口径一致。

【行为变化边界（设计内，非回归）】near_select=cluster 且 near 区短样本
（Tn<4×far_clusters）的臂：修复前建退化簇、修复后回退细筛分。其余臂
（near_select=4bit 默认 / sim_greedy / Tn≥4×far_clusters 的 cluster 臂）
行为逐位不变。

【红绿口径】
  * T1 主证：Tn=300（∈[256,1024)）cluster 臂 → 期望跳过（centroids
    is None）。修复前红（以 256 簇对 300 样本建簇 = 退化聚类）；
    修复后绿。红阶段附加报告非空簇数（退化证据）。
  * T2 边界：Tn=1024（=4×256，与 far 侧 `<` 语义同口径，恰好不触发）
    → 必须建簇（守卫不过度）。
  * T3 范围：Tn=300 sim_greedy 臂 → 必须建簇（贪心自适应不受守卫）。
  * T4 既有下限回归：Tn=200 → 跳过（Tn<256 原守卫不回归）。
  * T5 far 侧同口径对照：Tfar=300 → _km_centroids is None（既有
    4 倍守卫，证 near 守卫补齐后两侧口径一致）。

用法：python3 test_e125_near_kmeans_sample_guard.py（two-level-attention/
下，python 与 python -O 双跑；显式 _check 不依赖 assert 语句）。
"""
import importlib
import os
import sys
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))

# 状态显式化：PASS/FAIL 字符串，门禁无 SKIP。
RESULTS = []   # (name, status, detail)


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def _check(cond, msg):
    # 065 口径：显式 raise（裸 assert 在 python -O 下被剥掉 → 假绿）。
    if not cond:
        raise AssertionError(msg)


def load_indexer():
    mod = types.ModuleType("sparse_attn")
    mod.__path__ = [os.path.join(REPO, "sparse_attn")]
    sys.modules["sparse_attn"] = mod
    return importlib.import_module("sparse_attn.indexer")


IDXPKG = load_indexer()


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# 几何：bs=64、swa=128（TIAIndexer 硬编码）。S=4424 → near_hi =
# (4424−128)//64·64 = 4288（块对齐 swa 起点）。far_hi 为本函数入参，
# 直接构造 Tn = near_hi − far_hi ∈ {200, 300, 1024}。
S = 4424
NEAR_HI = (S - 128) // 64 * 64   # 4288

g = torch.Generator().manual_seed(31)
_K = torch.randn(1, S, 2, 128, generator=g) * 0.5   # [1, S, Hkv=2, D]


def fresh_idx(near_select):
    args = make_args(
        tli_far_select="cluster", tli_near_select=near_select,
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625,
    )
    idx = IDXPKG.TLIIndexer(args)
    idx.layer_idx = 3
    return idx


def nonempty_clusters(idx):
    """红阶段退化证据：near kmeans 实际非空簇数（每 kv-head 取并集口径）。"""
    a = idx._km_near_assign            # [Hkv, Tn]
    K = idx._km_near_centroids.shape[1]
    ne = sum(int((torch.bincount(a[h], minlength=K) > 0).sum())
             for h in range(a.shape[0]))
    return ne, K, a.shape[1]


# ---- T1 主证：Tn=300 ∈ [256, far_clusters*4) 的 cluster 臂必须跳过 kmeans ----
idx = fresh_idx("cluster")
far_hi = NEAR_HI - 300                # Tn = 300
idx._update_near_cluster(_K, far_hi)
built = idx._km_near_centroids is not None
detail = "跳过 kmeans，_km_near_centroids=None，消费端回退细筛分"
if built:
    ne, K, Tn = nonempty_clusters(idx)
    detail = (f"退化实锤：以 K={K} 簇对 Tn={Tn} 样本建簇，"
              f"非空簇 {ne}/{K}（一簇一样本/大量空簇），聚类失效且静默继续")
report("T1 cluster Tn=300（∈[256,1024)）跳过 kmeans（守卫生效）",
       not built, detail)
_check(not built, "T1：Tn=300 < far_clusters*4=1024，near kmeans 必须跳过（修复前红：建了退化簇）")

# ---- T2 边界：Tn=1024（=4×far_clusters，far 侧 `<` 同口径恰好不触发）必须建簇 ----
idx = fresh_idx("cluster")
idx._update_near_cluster(_K, NEAR_HI - 1024)
report("T2 cluster Tn=1024（=4×256 边界）正常建簇（守卫不过度）",
       idx._km_near_centroids is not None
       and tuple(idx._km_near_centroids.shape[1:]) == (256, 32),
       f"centroids {tuple(idx._km_near_centroids.shape)}")

# ---- T3 范围：sim_greedy 自适应簇数不受守卫，Tn=300 必须建簇 ----
idx = fresh_idx("sim_greedy")
idx._update_near_cluster(_K, NEAR_HI - 300)
report("T3 sim_greedy Tn=300 正常建簇（贪心自适应 Kcap=Tn，无退化）",
       idx._km_near_centroids is not None,
       f"centroids {tuple(idx._km_near_centroids.shape)}")

# ---- T4 既有 Tn<256 下限守卫回归：Tn=200 必须跳过 ----
idx = fresh_idx("cluster")
idx._update_near_cluster(_K, NEAR_HI - 200)
report("T4 cluster Tn=200 跳过（既有 Tn<256 下限守卫不回归）",
       idx._km_near_centroids is None, "")

# ---- T5 far 侧同口径对照：Tfar=300 → 既有 4 倍守卫放弃建簇 ----
idx = fresh_idx("cluster")
idx._build_far_kmeans(_K, 0, 300)
report("T5 far 侧 Tfar=300 放弃建簇（既有 4 倍守卫，两侧口径一致）",
       idx._km_centroids is None, "")

# ---- 汇总（TL-TEST-SKIP-PASS-002 口径）----
n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
print(f"\n===== near kmeans 4 倍样本守卫门禁：executed {n_pass}/{len(RESULTS)} PASS"
      f"（0 SKIP；T1 修复前必须红）=====")
sys.exit(1 if n_fail else 0)
