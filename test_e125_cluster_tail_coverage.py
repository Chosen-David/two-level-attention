#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_e125_cluster_tail_coverage.py —— 静态簇尾巴覆盖缺陷门禁（CPU only，0 SKIP）。

【缺陷（GPT 审计 TL cluster tail coverage，主 AI 核验 confirmed，2026-10-11）】
触发五元组（全部满足才触发）：tli_alpha==0 且 tli_beta==0 + far_select!=4bit
（cluster / sim_greedy）+ enable_kmeans==true + skip_far==false + 簇缓存已初始化。
数值 gamma=0 与 gamma=off 均触发（gamma 与本缺陷无关）。

机制：生产者与消费者对 far 上界用不同对齐基准——
  * 生产者 _maybe_build_kmeans：far_hi = max(sink_blocks+1, (S−swa)//bs)·bs，
    按真实 S 块对齐**向下**，簇缓存实际覆盖 [far_lo, far_hi)；
  * 消费者（L916-923 (0,0) 分支）：near_base = kt*bs（**padding 过界的块基准**）
    → near_blks 被撑大 → far_tok_hi 被撑大 ≤ bs−1 token → 守卫
    swa_lo_tok > far_tok_hi 恒 FALSE → near 补齐分支被跳过，而 far 簇 topk
    候选只到缓存覆盖 far_hi 为止 → 残留尾巴 [far_hi, S−swa)（宽
    = (S−swa) mod bs）结构性不可达。
反例（S=4097/bs=64/sink=swa=128）：far_hi=3968、far_tok_hi=4032、
swa_lo_tok=3969 → token 3968 无论分数多高不可选。S%64==0 时缺口为零。

【修复（本门禁守护的行为）】三处，全部限定在触发分支：
  1. far_tok_hi 单一事实源：触发分支直接读生产者写进簇缓存条目的
     far_hi（_km_far_hi_cached，_build_far_kmeans/_update_far_greedy 全路径
     维护 = 缓存真实覆盖右缘），不再两侧各算各的公式；
  2. γ=off 自由竞争：触发组合的尾巴段改用 group-sum 原始细筛分（与
     _far_token_score 的 group-sum 簇分同量纲，ccluster near 池 C3 同款
     口径理由）真实参与竞争——softmax 概率 p（≤1）与无界簇分混池竞争
     会使高分配尾仍不可达；
  3. γ 数值 (0,0) 单池点：far 预算先扣除尾巴宽 gap_w = swa_lo_tok −
     far_tok_hi（< bs，簇缓存块对齐的结构性盲区），由 near 补齐分支用
     细筛分给尾巴 token 正常打分选择；mid 总预算 K2_mid 逐位不变
     （预算身份：K2 1024 = sink 128 + swa 128 + mid K2_mid 保持）。

【红绿口径】
  * A 组（尾巴可达性/预算/覆盖）：修复前必须红（尾巴 token 无论分数多高
    不可选）；修复后绿。
  * B 组：S%64==0 / !=0 边界公式哨兵。
  * C 组（10 配置逐位 fixture，zlib+b64 于本文件内联，捕获自修复前代码
    4a287049…）：非触发路径（e64 分区 / 4bit / 混合臂 / skip_far / σ /
    对齐 S 触发组合）修复前后必须逐位一致——零外溢证明，两组都得绿。

用法：python3 test_e125_cluster_tail_coverage.py（two-level-attention/ 下，
python 与 python -O 双跑；显式 _check 不依赖 assert 语句，-O 下不失效）。
"""
import base64
import importlib
import os
import sys
import types
import zlib

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))

# 状态显式化（TL-TEST-SKIP-PASS-002 口径）：PASS/FAIL 字符串，门禁无 SKIP。
RESULTS = []   # (name, status, detail)  status ∈ {"PASS", "FAIL"}


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


def gen_kq(S, seed=31):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, S, 2, 128, generator=g) * 0.5   # [1, S, Hkv=2, 128]
    q = torch.randn(1, 1, 4, 128, generator=g) * 0.5   # [1, 1, H=4, 128]（GQA G=2）
    return k, q


def align_tail(k, q, tok_lo, tok_hi=None, mag=8.0):
    """把 k 的 [tok_lo, tok_hi) token 对齐到 q（group-sum 口径）打高分：
    q_g[kv] = 组内 q-head 求和（与 _far_token_score 的 q_g 同口径），
    k token 取 q_g 方向单位向量 × mag → 细筛分 ≈ mag·||q_g||，
    远超随机簇心的 |q_g·c|。tok_hi=None 表示单 token。"""
    if tok_hi is None:
        tok_hi = tok_lo + 1
    q_dir = q[0, 0].reshape(2, 2, 128).sum(1)            # [Hkv=2, 128]（group-sum）
    q_dir = q_dir / (q_dir.norm(dim=-1, keepdim=True) + 1e-9)
    k[0, tok_lo:tok_hi] = q_dir.unsqueeze(0) * mag


def run_mask(args, k, q, layer_idx=3, skip_ids=None):
    idx = IDXPKG.TLIIndexer(args)
    idx.layer_idx = layer_idx
    if skip_ids is not None:
        idx._skip_ids = skip_ids
    S = k.shape[1]
    cu = torch.tensor([0, S])
    q_ids = torch.tensor([S - 1])
    mask, _ = idx.prepare_mask(q, q_ids, k, cu, q.shape[-1] ** -0.5)
    return idx, mask


# 触发组合基础配置（far_select=cluster；A6 换 sim_greedy，A8 换 near cluster）
TRIGGER = dict(
    tli_far_select="cluster", tli_near_select="4bit",
    tli_enable_kmeans=True, tli_enable_layer_skip=False,
    tli_far_method="minmax", tli_near_method="avg",
    tli_alpha=0.0, tli_beta=0.0,
)


def _budget_checks(m, S, far_hi, k2_mid, gap_w, name):
    """触发组合（γ 数值）分区预算深检：逐行验证
    sink 128 + far 簇段 (K2_mid−gap_w) + near 尾巴段 gap_w + swa 128
    = K2 1024；段间不重叠（各段计数精确相加 → 无重复选择）；
    mask 宽 == 真实 S（无 padding 位混入）。"""
    swa_lo = S - 128
    _check(m.shape[-1] == S, f"{name}: mask 宽 {m.shape[-1]} != 真实 S {S}（padding 混入）")
    for h in range(m.shape[0]):
        row = m[h]
        n_sink = int(row[:128].sum())
        n_far = int(row[128:far_hi].sum())
        n_gap = int(row[far_hi:swa_lo].sum())
        n_swa = int(row[swa_lo:].sum())
        _check(n_sink == 128, f"{name}: sink 强制区 {n_sink} != 128")
        _check(n_far == k2_mid - gap_w,
               f"{name}: far 簇段 {n_far} != {k2_mid - gap_w}（K2_mid {k2_mid} − 缺口 {gap_w}）")
        _check(n_gap == gap_w, f"{name}: near 尾巴段 {n_gap} != 缺口宽 {gap_w}（覆盖不全）")
        _check(n_swa == 128, f"{name}: swa 强制区 {n_swa} != 128")
        _check(int(row.sum()) == 1024,
               f"{name}: 总选中 {int(row.sum())} != K2 1024（预算身份破坏）")


# ================================================================ A1 γ=0 S=4097
def a1_gamma0_s4097_tail_reachable():
    """γ=0 数值 + S=4097（S%64=1，缺口宽 1）：审计反例逐位复现 + 分区预算深检。
    修复前红：token 3968 无论分数多高不可选（簇 topk 到 3967 为止 +
    守卫 3969>4032 FALSE 跳过补齐）。"""
    name = "A1 γ=0 S=4097：高分尾巴 3968 可选 + 守卫 3969>3968 TRUE + 预算 767+1"
    try:
        S = 4097
        k, q = gen_kq(S)
        align_tail(k, q, 3968)
        idx, mask = run_mask(make_args(**TRIGGER, tli_gamma=0.0), k, q)
        # 生产者哨兵：far_hi = max(sink/bs+1, (S−swa)//bs)·bs = 3968（真实 S 口径）
        _check(int(idx._km_far_hi_cached) == 3968,
               f"生产者 far_hi={int(idx._km_far_hi_cached)} != 3968")
        # 消费者修复后 far_tok_hi == 生产者 far_hi（单一事实源）→
        # swa_lo_tok 3969 > far_tok_hi 3968 → near 补齐分支真正覆盖缺口
        m = mask[0, 0]
        _check(bool(m[:, 3968].all()),
               "高分尾巴 token 3968 不可选——静态簇尾巴覆盖缺陷仍在（A1 红）")
        _budget_checks(m, S, far_hi=3968, k2_mid=768, gap_w=1, name="A1")
        report(name, True, "far_hi=3968 哨兵、尾巴 3968 入选、far 767+near 1+强制 256=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A2 γ=0 S=4159
def a2_gamma0_s4159_gap_covered():
    """γ=0 数值 + S=4159（S%64=63，缺口宽 63）：整个尾巴段 [3968,4031)
    全覆盖 + far 预算正确扣减 63。修复前红：全段不可选。"""
    name = "A2 γ=0 S=4159：缺口 63 token 全覆盖 + far 预算扣减 63（705+63）"
    try:
        S = 4159
        k, q = gen_kq(S)
        align_tail(k, q, 3968, 4031)
        idx, mask = run_mask(make_args(**TRIGGER, tli_gamma=0.0), k, q)
        _check(int(idx._km_far_hi_cached) == 3968,
               f"生产者 far_hi={int(idx._km_far_hi_cached)} != 3968")
        m = mask[0, 0]
        _check(bool(m[:, 3968:4031].all()),
               "尾巴段 [3968,4031) 未全覆盖——静态簇尾巴覆盖缺陷仍在（A2 红）")
        _budget_checks(m, S, far_hi=3968, k2_mid=768, gap_w=63, name="A2")
        report(name, True, "缺口 63 全入选、far 705 + near 63 + 强制 256 = 1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A3 γ=off S=4097
def a3_gammaoff_s4097_tail_wins_competition():
    """γ=off（E122 自由竞争）+ S=4097：尾巴 token 以 group-sum 细筛分真实
    参与单池竞争胜出（非挂名）——softmax 概率 p（≤1）与无界簇分混池时
    高分配尾也竞争不过簇分，故触发组合尾巴段必须用同量纲原始分。
    修复前红：守卫 FALSE 尾巴段根本不进竞争。"""
    name = "A3 γ=off S=4097：尾巴 3968 以同量纲细筛分真实竞争胜出（非挂名）"
    try:
        S = 4097
        k, q = gen_kq(S)
        align_tail(k, q, 3968)
        idx, mask = run_mask(make_args(**TRIGGER, tli_gamma=None), k, q)
        _check(int(idx._km_far_hi_cached) == 3968,
               f"生产者 far_hi={int(idx._km_far_hi_cached)} != 3968")
        m = mask[0, 0]
        _check(bool(m[:, 3968].all()),
               "γ=off 高分尾巴 token 3968 不可选——尾巴覆盖缺陷仍在（A3 红）")
        # 预算身份：sink 128 + mid 768 + swa 128 = 1024/行；mask 宽 = S
        _check(m.shape[-1] == S, f"mask 宽 {m.shape[-1]} != S {S}")
        for h in range(m.shape[0]):
            row = m[h]
            _check(int(row[128:3969].sum()) == 768,
                   f"mid 选中 {int(row[128:3969].sum())} != K2_mid 768")
            _check(int(row.sum()) == 1024, f"总选中 {int(row.sum())} != 1024")
        report(name, True, "尾巴凭 mag·||q_g||≈64 的原始细筛分竞争入选；mid=768、总=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A4 γ=off S=4159
def a4_gammaoff_s4159():
    """γ=off + S=4159：缺口 63 token 全部以真实分数竞争入选 + 预算闭合。
    修复前红：守卫 FALSE 全段缺席竞争。"""
    name = "A4 γ=off S=4159：缺口 63 token 全部竞争入选 + mid=768"
    try:
        S = 4159
        k, q = gen_kq(S)
        align_tail(k, q, 3968, 4031)
        idx, mask = run_mask(make_args(**TRIGGER, tli_gamma=None), k, q)
        m = mask[0, 0]
        _check(bool(m[:, 3968:4031].all()),
               "γ=off 尾巴段 [3968,4031) 未全部竞争入选（A4 红）")
        _check(m.shape[-1] == S, f"mask 宽 {m.shape[-1]} != S {S}")
        for h in range(m.shape[0]):
            row = m[h]
            _check(int(row[128:4031].sum()) == 768,
                   f"mid 选中 {int(row[128:4031].sum())} != K2_mid 768")
            _check(int(row.sum()) == 1024, f"总选中 {int(row.sum())} != 1024")
        report(name, True, "缺口 63 全部竞争入选；mid=768、总=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A5 低分尾巴（覆盖保证语义）
def a5_gamma0_low_score_tail_covered():
    """γ=0 + S=4097 + 尾巴打**低分**（0.001 幅度）：仍被 carve-out 预算覆盖。
    这是修复选定的语义：尾巴段是簇缓存块对齐的结构性盲区（宽 < bs），
    覆盖优先（far 池让渡 gap_w 个名额，总预算不变）——「细筛分质量 ≥
    簇代表分，语义无损偏保守」同哲学。若业务上要竞争式入选，γ=off 臂
    即该口径（A3/A4）。"""
    name = "A5 γ=0 S=4097：低分尾巴仍获结构性覆盖（carve-out 语义哨兵）"
    try:
        S = 4097
        k, q = gen_kq(S, seed=47)
        k[0, 3968] = k[0, 3968] * 0.001     # 尾巴分数压到近零
        idx, mask = run_mask(make_args(**TRIGGER, tli_gamma=0.0), k, q)
        m = mask[0, 0]
        _check(bool(m[:, 3968].all()), "低分尾巴 3968 未被覆盖（A5）")
        _budget_checks(m, S, far_hi=3968, k2_mid=768, gap_w=1, name="A5")
        report(name, True, "低分尾巴入选（结构性盲区覆盖保证）；预算 767+1 不变")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A6 sim_greedy（E125 cavgsim 形态）
def a6_sim_greedy_s4097():
    """far_select=sim_greedy（E125 cavgsim 臂形态）同样触发五元组：
    _update_far_greedy 全路径维护 _km_far_hi_cached → 修复同样生效。
    修复前红。"""
    name = "A6 sim_greedy γ=0 S=4097：greedy 簇臂尾巴 3968 可选"
    try:
        S = 4097
        k, q = gen_kq(S)
        align_tail(k, q, 3968)
        spec = dict(TRIGGER, tli_far_select="sim_greedy", tli_gamma=0.0)
        idx, mask = run_mask(make_args(**spec), k, q)
        _check(int(idx._km_far_hi_cached) == 3968,
               f"greedy 生产者 far_hi={int(idx._km_far_hi_cached)} != 3968")
        m = mask[0, 0]
        _check(bool(m[:, 3968].all()), "sim_greedy 尾巴 3968 不可选（A6 红）")
        _budget_checks(m, S, far_hi=3968, k2_mid=768, gap_w=1, name="A6")
        report(name, True, "greedy 臂尾巴入选；预算 767+1+256=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A7 decode 两步（簇缓存复用）
def a7_decode_two_steps():
    """decode 语义：S=4097 → 4098 跨 step 复用簇缓存（far_hi 块对齐不变
    → E72 缓存命中），缺口从 {3968} 长到 {3968,3969}，两步尾巴全覆盖。
    修复前每一步 S%64!=0 的 decode 都有不可达尾巴（生产影响面）。
    修复前红。"""
    name = "A7 decode 两步 S=4097→4098：簇缓存复用下两步尾巴全覆盖"
    try:
        S2 = 4098
        k, q = gen_kq(S2)
        align_tail(k, q, 3968, 3970)
        args = make_args(**TRIGGER, tli_gamma=0.0)
        idx = IDXPKG.TLIIndexer(args)
        idx.layer_idx = 3
        # step 1：S=4097
        k1 = k[:, :4097]
        cu = torch.tensor([0, 4097])
        mask1, _ = idx.prepare_mask(q, torch.tensor([4096]), k1, cu,
                                    q.shape[-1] ** -0.5)
        _check(int(idx._km_far_hi_cached) == 3968, "step1 far_hi != 3968")
        _check(bool(mask1[0, 0][:, 3968].all()), "step1 尾巴 3968 不可选（A7 红）")
        centroids_s1 = idx._km_centroids
        # step 2：S=4098（decode +1 token，far 区块区间未增长 → 缓存必须复用）
        cu = torch.tensor([0, S2])
        mask2, _ = idx.prepare_mask(q, torch.tensor([4097]), k, cu,
                                    q.shape[-1] ** -0.5)
        _check(int(idx._km_far_hi_cached) == 3968, "step2 far_hi != 3968（缓存漂移）")
        _check(idx._km_centroids is centroids_s1,
               "step2 簇缓存被重建（E72 块对齐缓存命中语义被破坏）")
        m2 = mask2[0, 0]
        _check(bool(m2[:, 3968:3970].all()), "step2 尾巴 {3968,3969} 不可选（A7 红）")
        _budget_checks(m2, S2, far_hi=3968, k2_mid=768, gap_w=2, name="A7-step2")
        report(name, True, "两步均覆盖；缓存复用（对象同一）；step2 预算 766+2+256=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ A8 (0,0) far+near 双 cluster
def a8_ccluster00_s4097():
    """(0,0) + far=cluster + near=cluster（ccluster 式 (0,0) 点）：near 区
    在 (0,0) 单池口径下为空（_update_near_cluster Tn=0 不建簇），near 池
    走细筛分补齐路径，尾巴覆盖同样生效。修复前红。"""
    name = "A8 (0,0) far+near 双 cluster γ=0 S=4097：尾巴 3968 可选"
    try:
        S = 4097
        k, q = gen_kq(S)
        align_tail(k, q, 3968)
        spec = dict(TRIGGER, tli_near_select="cluster", tli_gamma=0.0)
        idx, mask = run_mask(make_args(**spec), k, q)
        _check(idx._km_near_centroids is None,
               "(0,0) 下 near 簇不应构建（Tn=0）")
        m = mask[0, 0]
        _check(bool(m[:, 3968].all()), "双 cluster 尾巴 3968 不可选（A8 红）")
        _budget_checks(m, S, far_hi=3968, k2_mid=768, gap_w=1, name="A8")
        report(name, True, "near 区空（Tn=0 不建簇）+ 尾巴入选；预算 767+1+256=1024")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ B 组：S%64 边界公式哨兵
def b_aligned_boundary():
    """S%64==0（4096/4160）：缺口恒零（far_hi == S−swa），修复前后行为均
    不变（逐位证明由 C5/C6 fixture 承担）；S%64!=0（4097/4159）：缺口宽
    = S−swa−far_hi ∈ (0, bs)。"""
    name = "B 边界哨兵：4096/4160 缺口=0；4097/4159 缺口∈(0,64)"
    try:
        for S in (4096, 4160):
            far_hi = max(2 + 1, (S - 128) // 64) * 64
            _check(far_hi == S - 128,
                   f"S={S}: far_hi={far_hi} != S−swa={S - 128}（对齐缺口应零）")
        for S, gap in ((4097, 1), (4159, 63)):
            far_hi = max(3, (S - 128) // 64) * 64
            _check(0 < (S - 128) - far_hi < 64 and (S - 128) - far_hi == gap,
                   f"S={S}: 缺口宽 {(S - 128) - far_hi} != {gap}")
        report(name, True, "4096/4160 缺口 0；4097 缺口 1；4159 缺口 63")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ C 组：非触发路径逐位一致（零外溢）
# fixture 捕获自修复前代码（tli_indexer.py sha256 4a287049…，主仓 main
# @3b66c3c），同一固定输入（seed=31）在修复后实现上重跑必须逐位一致。
FIXTURES = {
    "c10_sigma_cluster": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrtwQEJAAAAwyDWv/RzHNQCAAAAAAAAAP4NjgUgAw=="),
    "c1_e64_cluster_g0625": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrtmNl2pDAQQ3X//6cnC7hUZiuMSfIw5OR0N5ha5UWC373kFx9/yrd27rT7BxcnA8whxxYO7aIrr6UI7zk9u80TI5WRcP425YbcL/dhkv7gEgdXoxMOjtIppIlO/PC851TNMYYBRiJ8/aVDY5SgV/fF7bVkZnOndHBShLct83rqzxsZj6iufGdP3yvo3NZMWotndQQuK33PDyUQcr6fcwgUZqCL481kyhJCJRYuxl4db2oefxXacwyhp2i87fhRNp9n6M9ptfwvv9rX5f9r3PcP/2z39T1O8cw+lgkW72o1bqPUzvMRhXtigeTyNu18F2YsD4+thSZFOmuW2X/D8prRhm1ESi3MKFJ7azlfRC3DuiWunN6uLzU7hE374dNuKVEz5jnIKr90JLfDSqwcXdRYWDhYWlh6hAN1PbTQW+zRPQuCDjwb4KllkvqSMu462OHSmmdGI2OvUKDF4bc3JdS8xm5FQu1aJxl0Au4OWuFQsrstbvH/emmZH+b5vLjpMcTcpu+IT+kFD4OkSPMrZ0UKZ1x+4rzBe8cbBujMnfwpgZ4zTDOQI49m1n1Gxt5akE76gwSVSjtGBLA7BiiUkYEUmQ9rRk1T1QnZXw2qyVOvO8X68qSSHGpg9V1wfJOiMEuZoNxUtqUxqeCS5VFg6S8J/ONLAKOLDZNxWFGcXpeW6nsCP9a+ebXnjP8bx9kQWmd6yoQ1kUxpw9eDBOJULEijOu7knG19sRGl7lH+FuHKTWcFYMPiM6UzWubULXOweDMZ9efayB+J0RrJdEYcVhSV7Ytnnxa7V1CmxIieezt9lraN6rKmo8GbnlunlCUXo/itgAYHU0+UPJl+YDqBJSGTa6y7uUoy2zZGXR5KwoqSHmACjtnxciqJLXIdSb22lbQNJRVJm3bIRYQkGm1mIH9AAfgHS1EIAQ=="),
    "c2_e64_cluster_goff": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrtmcmSGzEMQ/n+/6dTcWwJCzuXHFKpWHMZd2ulSBBEw99tcxs//34+4vXj/fB00r58OrzG6I9Xv1/PmBzH66U+593fesjy7/9u3/cK2aEf0C/uVvi8fs9OnG/OAW5Pzmx3x8c0vw4dK8uyLMf5nOm1vkxvJqVms5Mg93SHT1h4dD0327A8+1wLYohzeF/crSazXQPzvjDODNbHHK5vcUgf6i6f8/cVISvGSPx6MNveqzwvsfs2R8JMh7n/Zhl3jTPDGWWbF/8h9o77mu7145DboVmMsOxzno1+7WKHxzfmVoBwDEbQwq+DJbjtIW6G45kevSR8KDoIrIVJ84Z00n6LDxX8wqHkBjPuUncnF0b8qpGQ4wPKLI4dK150Oid2225DB7sVFJcTJxPbzxD5D0OAYdolmSlMNxPadOpL6Vj2wmHbENByWoAoe/Yw0PRkQibLIfMAqCd4Fq3o0ouwIJOYhhVViKUcFzBEbBytZElfRu9XADPnQlHfQTpISIS34osndaa9hQxdDEywvJpERgL05GLm4cRz0zW7s1rSgo2lqGlPeiaojyUcOzPl5pO+byTt4I1l8nLHwJhgLpPu8BtW4hi9GAeF53PSg1MsSA/NrZiFk9bNKnXLGGKWbIPiJsweCTyZ4du+7dv+kcZ/3pS/ZQl8mGNV1oqQW3kmRAjjfsDCN71gXOtmK7oJEaKmcj4qpJqgbAu9M/Ii/GWq9uu0NhsvyV02+xPa/5RTHuyy5t9Jjs/2y2tItDBTBjpNFfBikWTzwcSN++PsFC9wYVElCDpVfuI144SG5UXGsJd2E/UisfVVQ2rFCi+kQ+FgraeNkhsH0TIvBCKxFKEEsNZ045W2hBPDHgSlWYk/6GlQJZAilF7YN6E8YbbHpFdUVEF1meOl0Vv15kKezxJyTqEei4xE8MzUN5ojJoWMMBBjoLBA3AmKq9McG3aajtVWLe2pw7MVpPBYjChN9qqHctk23mc82pGcYTrLFDTT1acAEoHZgCg2LCFj+kVpCw/yScqWbCUbdUTGRDuJ7EW6NXDBxGc0UELnwvwANt3viralMRK1VOt0DH5iRFFRQctLaJPEGocCOihnZgvoujQqUQe1AQUaUlUvOYFNTmMedNc0ntS2sMicRRwetDuCq+SWk7rwpBfMIjil5Bwp3zrnd5UO1nthNKPhwiy5qwXUyfSMfH2a1TvHPro8iZmskrN9lvkWTt/2bV8B4M/bD/DaCAE="),
    "c3_00_4bit_g0": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrlmdt6wzAIg/nf/6W3tY2NQDjZ1S7Wb4emdXwAIQSBv31FfP/8vHj9fn/weROfj68R64PPn33NGsT+k37yi3RNnXX97wt+7n0vxNpfniLSHiKfbJ+QSJvNm9pbvu5/W4Lro7Ukfr+QbFa2vUZhNvr6mmT4cv/rK9JkyNrsTV2joBr77aJmjb18vkj2/7gz3ypTvffNgok4jwyE4qyKIXf0BCWQbSVv5nuJvAcUuahP9O3a7J6DZOzucjFsmzj5pOHdgCOa+cvCkV1HAW6LLxxM9YsKRIlVir0oAGfPtrCD209+l+xq3BwteARt4uN1hDqbub0GfwI6dKB5gqMShFDMvr7GpNsxVCe0kMkFHYBDzD6yt8oKygqnK24RxnDIrNHV4c8OmmWPtDE9td6lAwtsK5b7PASFpCabNhOyeFSIikbHF3FwIPKoOSznnUYEitqekaKxWmF7kt0Uekf8Dym0EGYdjJ+FIYvfxh8NOxRKiNO6HpuV2gbHPLgS5VKnn8ICdV9fm4rQTsi0C25cMB5BvsV5BGWXHjX1dqzFzmC7RWKeBMsSNnOZWDUpxDusUVIPCWcJpQU0budUYQISY0RuDjqaj3agRnExU+gT/ExG3IrPYP0AW7Nd7mkripgcHRdTrv0FVIeRKrpjQCuES5T5HcRJe251hYE6MSzNjdyxWYCk47g3wenoSO0ktQsuufSwNVAbsoHQrHML8YwxB8b73y/VdN0fWKhUXRem3sjCmBr7TLqKXH1qJVXKDEo+psl9Wa9kMiKMKombADvpLiadgRuIF9a9ZhYORUu1RysSTTdHqTRVEWq/AWMXpKgA70hlQFy6WYdGyLKKp9UIQO7JaqKPFz0mCzVJU7IY3c6Ox68CmGjNplqf0VSebFvqKFkNg49dF28T4zUITf3vJelNgO69CrTCCbTPcgFWq+bSvWrxCaUJYktNohMEOfGogUEd30UMVvnTRBaDuIBJA1ILBq0ienSX02qHgqlEX63Ni3YpyFYLshu1tZLBJn0aRBR9KzhNQQ3L8ahZnH1jT5U5XHHlxTQLHJTKPPelKeGHSwAULNBLOTyQ6LE7lfzhezElNZL7NtTGwtAscuoH15CJUxrr/SihXldsKNOd6swhRbU2uO1iHB8TDMYcOhil4OVcW5xPceg4cGoCD3fiGwVPZPtsQCdSHk1kZBSn4nkoH7it8LkpD7PHYPLB2Ls59LWIEd2UOgo8D1lfYXPAYbccOgN4gWebYEOkmhCQ5gEH4WtLwCNkogujX5Tj8kThQVWPb33AkDRtSNTCh6OwPockw5PO8M/77s7DXW+nPNjqos4HLO33YVjJU4maQ/FUydx4ZujgcCQ/bB+IeQlpKPzt6wuKdQgB"),
    "c4_00_4bit_goff": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrlmdt62zAMg/W//0uva2OLAEFpd71Yvq7NEtuieABACn73tdbXz98X3/++Pvi8WZ+PnyueD56L3wt+7vq57f31XiO37S/3ivv9/utr7QdSbWtP3yvUne0dsoqx9RbKnmRjZVvyrdsLo9n7KrN775Xi+OVG243I2vUrfBk0tP3BHkzUPORviTZ7xXfVd7nqbcsUiS8a2pYpxfHV9zU8ci/FBnkmKa/wZC4r0G4050Qb9mdEN8e3GpqVQ0R+yHcAiHem1cn5VDbECtZgNqALY/4IFal+TUVALuX9cHo8/Wl5TXF1SfTJbz05t8NYA0gVfJCsT47Eqw4eCKWZHa2iLGlekew1C1ipiq7um5Kx7pASCwancrpw/VARU0gkNZkjwQSyD1RhucaQc0PZ9uu9qBQqWyxhBYw8MB4bcelAxiV+OA9Y2kjBp6SxjHp9PJZChkG0UDaPMEaC/cPo82YfK29i/J8oF08AWh2zVOFk9PqUtF0bSAnDmIM3c1F2LCOR1va0Cg96yYrTY5iZoeJfshHZegdK4k4JIiroqJgkOAy6OGlQBwU0RPEoXYJVB/YsfN+16nDFMnsX4a8sAT/Wtv2wUkYEkjVWaYnSjS0uyBWczI2wR741y68RKdaZ36Jvh+SN2NTgACJR1ncQZKNqP0oDoGhGUgwto7uEJ5HWk8okTJt5F4E1l3k/LoDVrL6weZL8DFeL7AthCXqcMdSON//5q2o6IvRwICIrU3pPXYuEBoOXysRIVdWk83EiRiYVTqBMbjgcjb/qDPKTTtnvzaa4jatoPzWUNOXaGzVGgKSZBaPsErAIpVka7djOYckExARxXW/tnXAUBB6EFhv6RCi1Ay9yD523DykIEQ/B3aQepwBKoRhbcRe+uFTJpR+yJOhLhmUh4C55mkeSV73yo0OoSEXsDEh0cakhxkFG1wqDlIHUBOyh3+otGN0FmS5xDmc3XUhmm0KA0Nex1tTDE+Zcomo4pMWufvpzh3a1RFLJI2QpZqZPiUrb62n6wkIiAKJwmIfZmdSalrYy4dJQoSTn8kuHcJmgWHF0wjh4g9Tu4cQbRRqCdMNcga7UaOcP5+aXwzy4J29CwiZgCGPOelpALG8muYTxESnPo8AKTfcwZjjID+JRjfZJaWh+GQOCyv7UwnEgy3TY0NKOqcwlBx/Ls9QPJZsGezD2B72Jrn1Uz5M8QTFnFUpYU0M04U1qKnUkRyp6JlmoPADjVBSf+k7tVzp46QjCykcqGWRF8nGC25WPlsqU3Q8BrCtBe0zSWCc2x4R0Wn6IRGxygxZmzWtEmXEabgaAIroKId52HHGa94ZTiQfJ/eAwnRJpQUwjIo6ksMmDQR/ZtFXOJH739Qf6NwgB"),
    "c5_00_cluster_g0_aligned4096": dict(S=4096, shape=[1, 1, 2, 4096],
        b64="eNrlmduS40AIQzn//9NbtYndIAk7b/uwmZrJ5NJtDEIIGv7to359cP5+VvH55e/PeX398916Pn2efZOzsF9HtqSt/lwUzs5iLGPxNGK5t+sFl32cdbSNo0/0o2lnHTu472Y1azETprssLPfe7XLto3YTtrLdxfSi2/MJwzGdthJdeoeZ7inUYf0flhCN4Jd4jvvnda/vF5NdHSrciKOb2lBRb3CajvHvD5OZtrQ107H+ubnpGM3Bsxjn+xA9JdelhkMk5rqmdGf8FgfKGHZeMfgGZE9fzOc5T1hzFfULGfzhqsC0m5Zv5nqIKMHiIyCHvJ8C7RWLH6d+DafhgePqZZvssw6F471EKdc9cCcfCoIIvlkpaNXFfCXxqX5ziRVggM0YxtyIbtKcxkrhKdYc8MBy3UBSOVsUGGIK6z0pTpMleGCIqQWj8PCsJ1JdZrqUrU4IS5QikM39kXtIxSLx4REWrPWxZmHfStAEQsOArSKw/HTJTZTk4mNpQabnmiZQb3oJDNS40oi7o0geFd51CC8U96OSxaAzuCgTCsrLQ64OF898giaPSCCgZwtHfGY3sFB9FsITDy09uPTN0Mik+tsuSGV2zUqVMoGLBrkQ5ToTco9cwBeuPl90tcJvRdGu7NFyvtSlgtYSrZdJsk4uQYyAshhesVhoUX1FUhYSKuENBkX1UouXGpSSn/hikCRbWWJnboRQTJjxnz9m8xF6C1DYBa0aMgoJ/5LZuGyjE5BIiozgllmDRrUx9raEHTfefGp2LzL/9id5Q8yZScgDjwwW26YgvXiQNKGKqIQiy8icqNTSIVRWMEQnPJtuBcMahKD00PYTg29olSeJEiYXXTuk/nR2VARBhrWa0s3GWYmUy1HABv2e5nRpPNO8LsGEUdrvMp51ilEyy9ApaBdCgaaVGGP9FPPcI3XgGUqgtt56TNLYqjaVArBzDFFoUWzyozcIU7hO+ECFAeAVNmwDdQAdPiUTgigze1rqYCnMhXIDC4ni0No9FRBpHIjpYmwGLQEmNdFhhEQ9iKWpfcKw4UE6w/Psd2kOX4ffm5h6UfAPTdOujxl4TEYgIy0WIv7pHdXuITcxYscAe3/regqVN1YofnPUKlpM02yw4uE85aFPGWN6vFUgdhFhSmInC3tL4VOZOfrLL3S2lgiAcEqBXo4wkk4UQciR0KezDXWkm2UlDp4kGOYqiRhPy2oXW/3wahIdteAgtbfkOWMYsmbvrw2052A/VeFhYJOKNT8n5TIrWXgqU7KwRX/jEluMMwCikpwMaXyGHSEa8dWim6iFNOedW11zJTniQ8VJl5yLMVtulGn0FMnGFv0MK4gG6qW9gXF29Nr3/Ov++w+mlQgB"),
    "c6_00_cluster_goff_aligned4160": dict(S=4160, shape=[1, 1, 2, 4160],
        b64="eNrlmduWqzAMQ7P//6dnnWkhtiUR3ubhsObSFhrimywZ+NtjfQ4+P9fbpS/Kq/0Wd/W/VevC7VuUyxa/r+5f6krX2X2Pz/l94tq12cc80LNYm1DLCR4pVuJXwt3Dv2/3pdyVYixmlRFB3Sf1KvCuGTtfsqb1LMFC61++1izNjCukdyyvy9ieqKn5/ZgZJrPxz9J4c0viMVxQzR5puVfRyEW/GsdvC0s91DVDrKh/FyVtm3N52pG3R66lLde/j0am2kN1cNgJLk9bsKiVX80qd2BE08AW1gNkrGBkX4AB7qSuVT8zW6KHKVhM0EeCqJXDElbze7e61BgSnBndifp3MTU4ntfhC63hZMsaFGgU2DRyuC3evnZlQ8lZHGSklsVTpmS8o2KdaRJsoOOpczFdBA5ZHGKPxKB0pFlocH2wfXToEhNEmfVMTYavme1ftZweJ+9xBtbb/AC0kzvcFyMRvzO75gBXAVlb9z36WOxrdcsVj9H2Cl/qviMAqw0atR4rKOz6sT16FHQIDQ6awAAu9iLlmzii86Y6CelpABNMgk1aMPfBTHJLvtiVpq5BkmZSi1cHq5tABs3l8RXTEm1bx9PXb2rGRPR0VG+wA4pScBzLM9vFXYFgYqCo3nO94Ecu+sxstdS0lYqdkeah5UOkhM1nDGePPlsoMbU2H5Res9CioWUcGJyz5XJrO+fyjBwDzLR2CBDhZEXoeQhT6mjB/350dyra3IHFKSJMmtt2qkAkvVz5bmwPQo+OCp1RCUP/8s2EsbnYvHqBwlmKRZHD2oQBgUyORDXSLNTTBHZTd9mZvYgYSzHBUhrW+MqBi4MFd6xxEPjBk5e0yWE6Nk203tmxbM/HK+++eJ87qCLu8kcAlEuNqCpM5A2XDtL00OI0VJjIkAgNOM6UGF5gi61ymlkOnXSVDghURZIpfZLmpo2XZoY282uEWZtkoydZyBp5CEwAJo7exId7CZN+osy8tOQJKnTMEVm6eY0RmS4WTAZ4hgkOg2yGpkyymNN4YOh2IVxxFiabQsuGpzJZcRZIA+qZ8cPcNE8xXZfYWXjBBDFOReZaAaEZAxVPkdswajSqPD0KncAODZwQIMWS2QXymF7MyHo3yc3DjJrHJHr7yOABM3VQLaOH9WqA58UlXoAWGtrY6RTM4OVgmhLh1PMjTgUmevOa0VKNonXJTSa3OHjBAh59jEp45GOH+SSxh9x6bwx5MsVz7vQCwxBQRklX2sibSrfziBVH7/S0Gl3cwm4bDhe0Y3zoycOJBBPGppbhyuJwmve+gAQYnGBl7rdcmpGEGo5zOfq8Z+/pCW7ofOMZ42k2XxIZzDPEHjIsFvDn84MfxW8IAQ=="),
    "c7_mixed_beta_cluster": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrlmdtu3DAMRHn+/6d72Ugih0PZAQrkoUabbde2xMuQ4kzgZ6+I33/+XHw+ifV5vtw/9t/9zN8V1s/P26wFI71G3mU/8PWPyPeRr9Iy297zUHoA4tgSnyfrg9tIubaXybgTAWJHqb5Mtno/wrn7sYXsOzmcKwbEykUy+KxTvTgJyevkzJ1EEsWYvGZyt+zAfo8UgWNozhvHox0mCkrIWT6RbMGsYTm3Ac2ioucEfJlOpM+9OCZ4J8IrmlBSGHmdHfJkAdFBtfyOmp9wWRzBLKntrgpy6TuI6RLFnS+itAGp8uSsK4S1xI4GYkZziB6BY/4ULSldzFpRy6cumXCHKWVapihGkyBV7C62UO60IDCU70oEZNRKG0TgtL3L5UrKRwbEyl1KjsKhI5XejqVP7r1rU49oLSla5rSml/fJPUwEtWQk1Pgiy90kuSQLnfuU7Jak2hpOTYrufMqOprGdRP/4QlNLOcXD9FX8KpgD88EDOvQhp5KHhSaD9Hvsmf7uf/u8wizvtpaeaC0ExU3PeBs14CEFc7FiU1PLq6fDhl6mwEdsfAeJfbB0RetLg6nB3BM2Hyu4YJpknQPSbGLmGObESWhHR8fw0RwqoySX3XmHnymIMKHrDltjLiZK845cE9fHHDdlvOuRvWQYkkj41oOBKu3ktggCOYocmhhq9ulIoYz2lEnmXRO/7A/IvIo7XMyoZrDP8DSVj01l+9Qxh473f1+ZyRLDSCCgQebQNpB/0UFKF02k3O3CiASkAWD68DSguFPatWCdzAqJJjGdrHiEMh5gPOMzkcoVQOHJQieplLIOySGSgemYENSwldzoqaxukN7JzlU6xUSi2OLGOEA7jSiqYOJP76w4KWmCyuH3CY7KDFi+kZh90T56u6cqFRVaKPMUecAMnyEMmmjiA3ZoYh6FC8VJgg0i4/i6FC2J8D1bXqZLNV0WAToAaUc/JMxSy7+E8fDSVvkcTDjmS6Fycg+noWRaDjine1RiGBV2T6EhESogp4XIFd5GYnRYroxDarjWuJsYqohHRBZ/8HMUIqGIsgdN46iMtUiKCQx+0M+4xLEsLWJL1FBtxwYztS/6KEYRVrPFt2r9Ps/nPv3owcJt+B7gPCjlTlSOSjm7gQ5OR8S/qxhWW5i8sY0L4vJrj5Fb3L24KA5Ms//lTbxQ8GZsnwOIV4mfF8LU5o08D/SBR4bPAz3MGYO4d1jb5SeyS4zoRnhUUyBvrORZ+uOF8sV4Dx08e13NcVzDDyOfosXGiIo3OWUc81/kDbhrW0+SZ7Tf2Zg2gtkTJ+twL52LeuoN42034AKvi7jJ5VckVuMexOEHqk1vCPhgO/G/cbGrADg53XUg5i2KoPCz1y97xggB"),
    # 【③ near kmeans 4 倍样本守卫（DeepSeek round1 ③，2026-10-11）设计内
    # 行为变化】c8 near 区 = [far_hi=3456, near_hi=3968)，Tn=512 <
    # far_clusters*4=1024 → 修复前以 256 簇对 512 样本跑 kmeans（簇数=
    # 样本一半，退化聚类）→ 修复后跳过、消费端回退细筛分 → mask 变化。
    # 本 fixture 于 ③ 守卫 commit 后重捕（原 fixture 为修复前口径，
    # 在 ③ 守卫下红=186 字节差即设计内变化证据）；其余 9 配置仍是
    # ① 修复前原 fixture（零外溢证明不受 ③ 影响）。
    "c8_e64_ccluster": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrlmcl25SAQQ7n//9N9EttQgwTssuh3kjfYGEMNKqkMf/sa4/l7/p+3n/ff38/n9+09Sjj5nl1j15CxptavOYh6sAwIB8jTQVxJm4YyFcelrFGErceF4K4Vs5LNR5yjbMTM2zfxzps8Fq0e3oFiDfotUD+sS6a/0UuLP6S5vbXU0d8ARQ9lNN8bF7H91m0DwuV87szXoG7qY94skBAuKoJ5fUkzr0uAfgxnrpyG1eA6LERuN4sKB41bxzvc6WMwGcS4tstm3Tr/VB5dut4FLMApSgwSUjFsBggy3TgY9ZSfCR+5yLqS+CpkC1KuxBIbyDBX3YNGporF177CHkdCK5sIFt+oKeahklY34BBVcZa8xr0haLfHLAufYQznu2C+BaIuEgvmidihkhJf6RkGZEEio4xCTpRAbSn6QV5YqnVBWjYUSeymldFKzkTyo7yHiz3jLDwfyJtAgC15UGFLByxjeMA5lReGrsgrTKlkQJcm6oqb4zCo0IjtotrUPRYEB1Kk0khJp1GdzhRzcQmMJfkQaERCZzaxRI14cAY6gTHx/vme1OyleH1Sz8++XyiwJmON+UrhEk7MncI8vTBmnpoMl8l055TfHt4B03hzTYyEoTPgZgiE+r/u8Hx8APWugJHCjbWX9+t8ez5COZ27eDcQdhyjkSIxw7ll6ZJ1yV9MF1Br+relQKLm2AAzjOgJ4tX/88srUAUFSGXAieQiNQh7BuLrOBpjhqHyUf1v0Y3dISRAsp2DAy+GhHGyk0JuT+hSNXZ6NqtaazEEc4kKC99ZGV5ajpGTu6KzpQuFwCAtQGKATh1SwJ4hCjgXDRl5GBOQmJ4FTUo2XqvsSxWAvp2l3UMh5silbeTbMESOo1by9Cu0NqiNLJwQsm20zHPAsPNinR18cd2/sR0EjXROHnFx7aGfYDqm7Azh9llMxK4l61k/FgYbV0bpMmM+8J4k8mf64H3tyuCCUwm0zhDcaWqEHuuivcifRWAuimIPT9Wq90oUNLJgm3GB5DbpXi1Br5MYpYlDuGM3h3Huxsiu5ylUJRLung+4VnubGyfPCzyqnj6q+5Cqk0ZR91gA40OQyXiuAvRsTQWTjR8xWIWW49jHEdx0zE6srpivBxeN4eV1lPzDrJX2wIObRxtcNDFJHX90p/Hcg9cMF9UTxT9awnW3UO2l4ndEjmBpHJ00EJUlU4MTJCdh8JTqYwRiGloBS89P6TkfrU7lDqnrH/Q/a+ogsQtjDKq/6fzUc4hrSo0NYvOAUWR8skV+SkGQLWGbK4FXr2RtYXUOINbUJd5jCQ3thLVJ2aEc5NWzHtct5kFqL6RuwZ8L8H96EQgB"),
    "c9_skipfar_cluster": dict(S=4097, shape=[1, 1, 2, 4097],
        b64="eNrt17EKACAIBcDe//900NTWFCXeTeEUKabJWwMA6CXd/PHOW/iYBPnF7A7+QE0EADDzVFz71/HmjS3GShko0S4ncE4IAQ=="),
}

C_CONFIGS = {
    "c1_e64_cluster_g0625": dict(
        S=4097, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.25, tli_beta=0.125, tli_gamma=0.625),
    "c2_e64_cluster_goff": dict(
        S=4097, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.25, tli_beta=0.125, tli_gamma=None),
    "c3_00_4bit_g0": dict(
        S=4097, tli_alpha=0.0, tli_beta=0.0, tli_gamma=0.0,
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg"),
    "c4_00_4bit_goff": dict(
        S=4097, tli_alpha=0.0, tli_beta=0.0, tli_gamma=None,
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg"),
    "c5_00_cluster_g0_aligned4096": dict(
        S=4096, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=0.0),
    "c6_00_cluster_goff_aligned4160": dict(
        S=4160, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=None),
    "c7_mixed_beta_cluster": dict(
        S=4097, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.25, tli_gamma=0.5),
    "c8_e64_ccluster": dict(
        S=4097, tli_far_select="cluster", tli_near_select="cluster",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.125, tli_beta=0.25, tli_gamma=0.125),
    "c9_skipfar_cluster": dict(
        S=4097, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=0.0, _skip_ids=True),
    "c10_sigma_cluster": dict(
        S=4097, tli_far_select="cluster", tli_near_select="4bit",
        tli_enable_kmeans=True, tli_enable_layer_skip=False,
        tli_far_method="minmax", tli_near_method="avg",
        tli_alpha=0.0, tli_beta=0.0, tli_gamma=0.0, tli_sigma_select="far"),
}


def c_regression_bitexact():
    """10 配置 × 固定输入（seed=31）：修复后 mask 与修复前 fixture 逐位一致
    （zlib 解压后字节级比较）。覆盖：e64 分区（γ 数值/off）、(0,0)+4bit
    （γ 数值/off）、触发组合对齐 S（4096/4160，缺口零）、混合臂（α=0
    β=0.25，审计范围外）、e64 ccluster、skip_far legacy 路径、σ 分支。
    修复前后都必须绿——这是「只允许改 (0,0)+非4bit 触发分支」的零外溢证明。"""
    name = "C 非触发路径逐位一致（10 配置，零外溢证明）"
    diffs = []
    try:
        for cname, cfg in C_CONFIGS.items():
            _check(cname in FIXTURES, f"fixture 缺 {cname}")
            fx = FIXTURES[cname]
            spec = dict(cfg)
            S = spec.pop("S")
            skip = spec.pop("_skip_ids", False)
            k, q = gen_kq(S)
            _, mask = run_mask(make_args(**spec), k, q,
                               skip_ids={3} if skip else None)
            raw = mask.numpy().tobytes()
            exp = zlib.decompress(base64.b64decode(fx["b64"]))
            _check(list(mask.shape) == fx["shape"],
                   f"{cname}: shape {list(mask.shape)} != {fx['shape']}")
            if raw != exp:
                nd = sum(1 for a, b in zip(raw, exp) if a != b)
                diffs.append(f"{cname}（{nd} 字节差）")
        _check(not diffs, "逐位不一致: " + "; ".join(diffs))
        report(name, True, "10/10 配置逐位一致（9 配置=① 前原 fixture 零外溢 + c8=③ near 守卫后口径重捕，Tn=512 设计内变化）")
    except AssertionError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    a1_gamma0_s4097_tail_reachable()
    a2_gamma0_s4159_gap_covered()
    a3_gammaoff_s4097_tail_wins_competition()
    a4_gammaoff_s4159()
    a5_gamma0_low_score_tail_covered()
    a6_sim_greedy_s4097()
    a7_decode_two_steps()
    a8_ccluster00_s4097()
    b_aligned_boundary()
    c_regression_bitexact()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _check(n_pass + n_fail == len(RESULTS) == 10, "门禁不允许 SKIP/漏项")
    print(f"\n===== 簇尾巴覆盖门禁：executed {n_pass}/{n_pass + n_fail} PASS"
          f"（0 SKIP；A 组修复前必须红）=====")
    sys.exit(1 if n_fail else 0)
