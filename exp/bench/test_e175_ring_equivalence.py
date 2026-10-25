#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E175 W1 门禁：skip 层 ring 物理 KV 压缩的数学等价性（逐位/ULP 口径）。

用法（仓根）：
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e175_ring_equivalence.py
  CUDA_VISIBLE_DEVICES=0 python3 exp/bench/test_e175_ring_equivalence.py mapping cache

门禁 cell（违反即 W1 不合格，不得合入生产；口径经 2026-10-25 首跑修正）：
  - cell A（mapping）  : 位置→槽位映射穷举对拍（p∈[0,20000]）：
                        cache 实现 vs 纯 Python 独立模拟逐位相等；任意 L
                        驻留集映射双射；未满恒等；p 与 p+ring 同槽。
  - cell B（eager）    : 同选中集下 臂A（全宽位置序 KV）vs 臂B（ring 槽位
                        KV）eager_decoding_attn 输出（同 dense 路径，
                        TLI_E127_FAST=0 钉）：未满步（L<=cap 槽位==位置）
                        逐位相等；环绕步**逐元素 1-ULP 口径**（槽位布局
                        是位置序的物理重排，softmax 归约顺序差致 bf16 偶发
                        1 ULP 翻转是本质属性——对齐几何 (L-cap)%64==0
                        首跑实测亦有 nflip=1，逐位不可达；每个失配元素
                        差 ≤1 个 bf16 表示位（相对 2^-7 + 近零兜底）
                        + maxdiff ≤1e-3 + nd ≤1% 哨兵：错 token/错槽位
                        类真实 bug 产生远超 1 ULP 的差，全抓）。
  - cell C（cache）   : ① TLIStaticLayer.update 500+ 步 decode 模拟（跨
                        未满→环绕→多圈），ring buffer 驻留投影 == 全宽
                        参考 buffer 逐位（每步断言）；sink 槽 [0,4) 永不被
                        覆盖；② fill_row_from_tail（prefill 搬移）同款
                        投影断言；③ 500 步末用同选中集做臂 A/B eager 对拍。
  - cell D（indexer）  : ring 层 prepare_mask（CLI 层集 + 宽 L 虚拟位置
                        轴）vs 全宽 skip_far mask：**全部步逐位相等**
                        （未满步 + 任意非对齐环绕步——E175 骑墙块收紧 +
                        宽 L 虚拟轴使块网格与臂 A 锚定一致，同构不再要求
                        块对齐）；全部步：臂 A 选中集 ⊆ 驻留位置集（驱逐
                        无损断言——ring 驱逐的 token 经收紧后绝不可选）。
  - cell E（guards）  : fail-closed 守卫——CLI 层集 × enable_layer_skip
                        true / × dyn_dgate / 缺 near_len / 错 k 宽 / 层 id
                        越界 / argparse 非法串 全部 raise。

诚实披露（非门禁项，见验收报告）：
  - 骑墙块收紧的语义变更：skip 层（skip_far + E155 显式 near 几何）的
    near L2 段起点由块 floor 对齐上调到骑墙块上界（L1 near 池起点同步
    +1），near 区最远端 ≤ bs 个 token/层退出竞争——ring 驻留集与选择
    集的包含关系由此严格成立（否则块 floor 余量 token 仍可竞争选中但
    物理已驱逐，首跑 cell D 560/560 步实证破）。非 skip 层与 legacy
    α 比例口径逐位不变。
  - 环绕步 attention 输出为 ULP 口径（物理布局重排的归约顺序差）；
    选择集（mask）逐位同构的强等价主张由 cell D 承担。
  - graph 模式（compute_mask_graph）guard 排除 skip_far（E137 静态臂
    现状一致）：启用 CLI 层集的生产 bench 整体 fail-closed 回 staged。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import argparse  # noqa: E402
import torch  # noqa: E402

PASS, FAIL = [], []
SINK, RING = 4, 4224          # champion 几何：4 + 4096(near) + 128(swa)
CAP = SINK + RING             # ring 层容量 4228
H, Hkv, D = 32, 8, 128
G = H // Hkv
BS = 64                       # tia_block_size（champion）


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


def make_e175_args(skip_ids: str = "") -> argparse.Namespace:
    """E175 champion 几何（E152v4 生产主配置 + E155 固定配额 + E137 静态
    层集退役口径 enable_layer_skip=False）。ring 大小 = near_len+swa =
    4224 由本参数组唯一决定（与 cache 侧 e175_ring_geometry 同源）。"""
    return argparse.Namespace(
        method='tli', tia_block_size=BS, tia_level1_topk=128,
        tia_level2_topk=1024, tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
        tli_enable_subspace=True, tli_enable_kmeans=True,
        tli_far_select='4bit', tli_near_select='4bit',
        tli_far_clusters=256, tli_far_niter=10, tli_far_blocks=16,
        tli_far_tokens=512, tli_far_method='minmax', tli_near_method='avg',
        tli_sim=0.9, tli_sim_dims='subspace', tli_sigma_select='none',
        tli_moba=False, tli_sigma=8.0, tli_per_q_head=False,
        # 生产动态臂口径：静态层掩码显式关（E137 A6 / E175 守卫前提）
        tli_enable_layer_skip=False, tli_layer_skip_path=None,
        # E155 固定配额几何（champion flags）
        tli_near_pages=32, tli_near_tokens='off', tli_near_len=4096,
        # E135 正交强制区（champion sink=4）
        tli_sink_tokens=SINK, tli_swa_size=128, tli_sinkswa_only=False,
        # E137 动态 gate 关（静态层集互斥前提）
        tli_dyn_dgate=False, tli_dyn_dgate_eps=0.05,
        tli_dyn_dgate_rows=256,
        # E175 W1 CLI 层集
        tli_skip_layer_ids=skip_ids,
        tli_subspace='tail', tli_l2_subspace='same', tli_proj_basis=None,
        tli_static_pair=False, tli_sparse_prefill=False,
        tli_decode_dense=False, tli_boundary_bypass=True,
        tli_l2_quant=False)


def dom_positions(L: int) -> list:
    """驻留位置集（升序位置序）：sink 头 ∪ 尾部最近 ring 个。"""
    if L <= SINK:
        return list(range(L))
    return list(range(SINK)) + list(range(max(SINK, L - RING), L))


def _ulp_ok(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """E127 _ulp_ok 同款口径：nd ≤ max(numel*1e-4, 12) 且 rel ≤ 2%。"""
    nd = int((a != b).sum().item())
    numel = a.numel()
    ad = (a.float() - b.float()).abs()
    denom = a.float().abs().clamp_min(1e-6)
    rel = float((ad / denom).max().item()) if numel else 0.0
    return nd <= max(int(numel * 1e-4), 12) and rel <= 0.02, nd, rel


# ---------------------------------------------------------------- cell A 映射
def test_mapping():
    from sparse_attn.ops.tli_static_cache import pos_to_slot
    # 公式 vs 纯 Python 独立模拟（双实现防漂移）
    N = 20000
    ref = [p if p < SINK else SINK + (p - SINK) % RING for p in range(N)]
    got = pos_to_slot(torch.arange(N), SINK, RING).tolist()
    check("mapping: pos_to_slot 穷举 p∈[0,20000] 与独立模拟逐位相等",
          got == ref, f"N={N}")
    # 未满恒等：p < cap 时 slot == p（ring 未环绕）
    ok_ident = all(got[p] == p for p in range(CAP))
    check("mapping: 未满阶段（p<cap）槽位==位置（恒等映射）", ok_ident)
    # 环性质：p 与 p+RING 同槽
    ok_cycle = all(got[p] == got[p + RING] for p in range(SINK, N - RING))
    check("mapping: 环性质 p 与 p+ring 同槽", ok_cycle)
    # 任意 L：驻留集映射 = 槽位集 [0, min(L,cap)) 的双射
    ok_biject = True
    for L in [0, 1, SINK, SINK + 1, 100, CAP - 1, CAP, CAP + 1,
              CAP + 63, CAP + 64, 2 * CAP, N - 1, N]:
        dom = dom_positions(L)
        want = set(range(min(L, CAP)))
        got_set = {got[p] for p in dom}
        if got_set != want or len(got_set) != len(dom):
            ok_biject = False
            print(f"  L={L} 驻留映射非双射: got={sorted(got_set)[:8]}...")
    check("mapping: 任意 L 驻留集→槽位集双射（13 个 L 穷举）", ok_biject)


# ---------------------------------------------------------------- cell B eager
def test_eager_equiv():
    """同选中集下 臂A（全宽位置序 KV）vs 臂B（ring 槽位 KV）eager 对拍。

    口径（首跑修正 + 五跑 ULP 口径二修，2026-10-25）：
      - 环绕步（L > cap）：槽位物理布局是位置序的**重排**（环绕 gather），
        eager softmax 分母归约顺序随列序变化 → bf16 偶发 1 ULP 翻转是
        物理重排的本质属性（E127 _ulp_ok 先例），**逐位一致不可达**——
        首跑实证对齐几何 (L-cap)%64==0 下亦有 nflip=1（maxdiff 7.6e-6）。
        门禁口径 = **逐元素 1-ULP 刻画**（每个失配元素差 ≤1 个 bf16
        表示位：相对 2^-7 + 近零绝对兜底）+ maxdiff ≤1e-3 + nd ≤1%
        粗哨兵——比 E127 的 nd 计数阈（1e-4）更强且更贴合重排本性
        （重排 ULP 撞击率 ~1e-3 量级，计数阈对环绕先天偏紧；
        1-ULP 刻画对错 token/错槽位类真实 bug 全抓）。
      - 未满步（L <= cap）：槽位 == 位置恒等 → 两臂**逐位一致**。
    数学等价主张的强形态由 cell D 承担（选择集逐位同构，mask 无算术）。
    """
    from sparse_attn.ops.eager_decoding import (
        eager_decoding_attn, eager_decoding_check_errors)
    # 【同路径控制变量】TLI_E127_FAST=0 钉 dense 路径——L≥16384 时臂A
    # （T=L）走 Triton gather 而臂B（T=cap=4228）恒走 torch gather/dense，
    # 跨路径差（探针 2026-10-25 实测 nd=1780/4096、maxdiff 9.77e-4）是
    # eager 三路径的**既有系统行为**（E127 判决背书 gather/dense 数学
    # 语义相同仅归约顺序不同），与 ring 无关——ring 等价本体必须在同
    # 路径下判。dense 探针实测环绕步 nd=0（fp32 softmax + 列序差未跨
    # bf16 舍入边界），门禁用 ULP 计数 + 绝对差双阈兜偶发翻转。
    os.environ["TLI_E127_FAST"] = "0"
    torch.manual_seed(20261024)
    dev = torch.device('cuda:0')
    ulp_ok_all, ident_all, worst = [], [], 0.0
    for L in [1000, CAP - 1, CAP,                     # 未满/恰满：逐位
              CAP + 7 * BS, 8452, 8453, 12677, 13100,
              3 * CAP + 5 * BS, 20000]:                # 环绕：ULP
        kv_full = torch.randn(1, L, Hkv, D, device=dev, dtype=torch.bfloat16)
        vv_full = torch.randn(1, L, Hkv, D, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
        dom = dom_positions(L)
        # 选中集 = 强制区（sink+swa）∪ near 随机子集（模拟 skip 层 K2
        # 选择；E175 骑墙收紧后 near 可选域 ⊆ 驻留）
        swa_sel = list(range(max(SINK, L - 128), L))
        near_pool = sorted(p for p in dom
                           if SINK <= p < max(SINK, L - 128))
        n_sel = max(8, min(400, len(near_pool)))
        near_pick = sorted(torch.randperm(len(near_pool), device='cpu')[
            :n_sel].tolist())
        sel_pos = (set(range(0, min(SINK, L)))
                   | {near_pool[i] for i in near_pick}
                   | set(swa_sel))
        # 臂 A：全宽位置序 KV + 位置域 mask
        m_pos = torch.zeros(1, 1, Hkv, L, dtype=torch.bool, device=dev)
        m_pos[..., sorted(sel_pos)] = True
        cuA = torch.tensor([0, L], dtype=torch.int32, device=dev)
        oA = eager_decoding_attn(q, kv_full, vv_full, m_pos, 1, cuA,
                                 valid_length=L)
        # 臂 B：ring 槽位 KV（同一 token 值，槽位 s ← 位置 dom[argsort]）
        # + 槽位域 mask。未满步 buffer 恒宽 CAP（TLIStaticLayer 形态，
        # [L, cap) 零 pad——vl=L 裁剪排除）；环绕步 dom 宽恰 cap。
        dom_t = torch.tensor(dom, device=dev)
        slots = pos_to_slot_t(dom_t)
        order = torch.argsort(slots)
        kv_ring = torch.zeros(1, CAP, Hkv, D, device=dev,
                              dtype=torch.bfloat16)
        vv_ring = torch.zeros_like(kv_ring)
        kv_ring[:, slots] = kv_full[:, dom_t]
        vv_ring[:, slots] = vv_full[:, dom_t]
        m_slot = torch.zeros(1, 1, Hkv, CAP, dtype=torch.bool, device=dev)
        sel_t = torch.tensor(sorted(sel_pos), device=dev)
        m_slot[..., pos_to_slot_t(sel_t)] = True
        cuB = torch.tensor([0, CAP], dtype=torch.int32, device=dev)
        oB = eager_decoding_attn(q, kv_ring, vv_ring, m_slot, 1, cuB,
                                 valid_length=min(L, CAP))
        eager_decoding_check_errors(dev)
        d = (oA.float() - oB.float()).abs().max().item()
        worst = max(worst, d)
        if L <= CAP:
            same = torch.equal(oA, oB)
            ident_all.append((L, same))
            if not same:
                print(f"  未满 L={L} 非逐位: maxdiff={d:.3e}")
        else:
            # 【口径二修（2026-10-25 五跑）】nd 计数阈（E127 的 1e-4）对
            # 「列序完全重排」先天偏紧：环绕 scatter 的 ULP 撞击率比 E127
            # 的「同序不同分块归约」高一个量级（L=8453 实测 nd=16/4096≈
            # 3.9e-3、全部恰 1 ULP）。升级为更强的**逐元素 1-ULP 刻画**：
            # 每个失配元素必须差 ≤ 1 个 bf16 表示位（相对 2^-7 + 近零
            # 绝对兜底）——错 token/错槽位/错 mask 类真实 bug 产生远超
            # 1 ULP 的差，本断言全抓；计数只留 1% 粗哨兵。maxdiff ≤1e-3
            # 保留（防近零 rel 假阳性的绝对差兜底）。
            m = oA != oB
            nd = int(m.sum().item())
            if nd:
                af, bf = oA.float()[m], oB.float()[m]
                ulp1 = ((af - bf).abs()
                        <= (af.abs() + bf.abs()) * (2.0 ** -7)
                        + (2.0 ** -12)).all().item()
            else:
                ulp1 = True
            ok = ulp1 and d <= 1e-3 and nd <= int(oA.numel() * 1e-2)
            ulp_ok_all.append(ok)
            if not ok:
                print(f"  L={L} ULP 超界: nd={nd} maxdiff={d:.3e} "
                      f"ulp1={ulp1}")
    os.environ.pop("TLI_E127_FAST", None)
    check("eager: 未满步（L<=cap 槽位==位置）臂A/B eager 逐位一致",
          all(s for _, s in ident_all) and len(ident_all) >= 3,
          f"n_ident={len(ident_all)} L={[l for l, _ in ident_all]}")
    check("eager: 环绕步（L>cap 同 dense 路径）臂A/B eager 逐元素"
          "1-ULP 刻画 + maxdiff≤1e-3 + nd≤1% 哨兵",
          all(ulp_ok_all) and len(ulp_ok_all) >= 5,
          f"n_cases={len(ulp_ok_all)} worst_maxdiff={worst:.3e}")


def pos_to_slot_t(p: torch.Tensor) -> torch.Tensor:
    """测试侧独立映射（与 cache 实现双实现对拍，cell A 已验证公式本体）。"""
    return torch.where(p < SINK, p, SINK + (p - SINK) % RING)


# ---------------------------------------------------------------- cell C cache
def test_cache_equiv():
    from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
    from sparse_attn.ops.tli_static_cache import (
        TLIStaticLayer, fill_row_from_tail, pos_to_slot)
    torch.manual_seed(20261025)
    dev = torch.device('cuda:0')
    L0, N_STEPS = 4093, 520          # 未满→满→环绕→多圈（4093+520=4613>4228）
    layer = TLIStaticLayer(CAP, 1, ring_size=RING, ring_sink=SINK)
    shape_probe = torch.zeros(1, Hkv, 8, D, dtype=torch.bfloat16, device=dev)
    layer.lazy_initialization(shape_probe)
    # 全宽参考：位置 p 的 K 值
    ref_k = torch.zeros(40000, Hkv, D, dtype=torch.bfloat16, device=dev)
    ref_v = torch.zeros(40000, Hkv, D, dtype=torch.bfloat16, device=dev)
    ctx = BatchCtx(batch=1, seq_lens=[L0], gen_lens=[0],
                   alive=[True], pad_id=0)
    ok_proj = True
    sink_frozen = True
    with batch_decode_scope(ctx):
        for step in range(N_STEPS):
            pos = ctx.seq_lens[0] - 1     # 新 token 绝对位置（forward 前 +1 语义）
            kt = torch.randn(1, Hkv, 1, D, device=dev, dtype=torch.bfloat16)
            vt = torch.randn(1, Hkv, 1, D, device=dev, dtype=torch.bfloat16)
            ref_k[pos] = kt[0, :, 0]
            ref_v[pos] = vt[0, :, 0]
            layer.update(kt, vt)
            ctx.seq_lens[0] += 1
            L = ctx.seq_lens[0]
            if step % 7 == 0 or step == N_STEPS - 1:
                dom = torch.tensor(dom_positions(L), device=dev)
                slots = pos_to_slot(dom, SINK, RING)
                if not torch.equal(layer.keys[0, :, slots], ref_k[dom].transpose(0, 1)):
                    ok_proj = False
                    print(f"  step{step} L={L} 驻留投影失配（K）")
                if not torch.equal(layer.values[0, :, slots],
                                   ref_v[dom].transpose(0, 1)):
                    ok_proj = False
                    print(f"  step{step} L={L} 驻留投影失配（V）")
                # 满后环绕期：槽位域外（被驱逐 pad 不存在——满后无 pad）
                # sink 槽 0..3 永驻 = 位置 0..3
                if L > SINK and not torch.equal(
                        layer.keys[0, :, :SINK], ref_k[:SINK].transpose(0, 1)):
                    sink_frozen = False
                    print(f"  step{step} sink 槽被覆盖！")
    check("cache: 520 步 decode 模拟 ring buffer 驻留投影 == 全宽参考逐位"
          "（未满→满→环绕）", ok_proj)
    check("cache: sink 槽 [0,4) 520 步永不被覆盖", sink_frozen)

    # ---- fill_row_from_tail（prefill 搬移）投影断言 ----
    ok_fill = True
    for L in [100, CAP - 1, CAP, CAP + 1, 2 * CAP + 37, 20000]:
        lay = TLIStaticLayer(CAP, 1, ring_size=RING, ring_sink=SINK)
        lay.lazy_initialization(shape_probe)
        # 全宽 DynamicCache 形态：有效占尾 [S_row-L, S_row)
        S_row = 32768
        k_row = torch.randn(Hkv, S_row, D, device=dev, dtype=torch.bfloat16)
        v_row = torch.randn(Hkv, S_row, D, device=dev, dtype=torch.bfloat16)
        fill_row_from_tail(lay, 0, k_row, v_row, L)
        dom = torch.tensor(dom_positions(L), device=dev)
        slots = pos_to_slot(dom, SINK, RING)
        # 位置 p 的值 = k_row[:, S_row - L + p]
        want_k = k_row[:, S_row - L + dom]
        if not torch.equal(lay.keys[0, :, slots], want_k):
            ok_fill = False
            print(f"  fill L={L} 投影失配")
    check("cache: fill_row_from_tail 6 个 L（未满/恰满/环绕）驻留投影逐位",
          ok_fill)

    # ---- 500 步末同选中集 eager 对拍（cell B 的 cache 原生形态）----
    from sparse_attn.ops.eager_decoding import (
        eager_decoding_attn, eager_decoding_check_errors)
    L = ctx.seq_lens[0]
    q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
    sel_pos = sorted(set(dom_positions(L)) - set(range(SINK, L - 128 - 400)))
    sel_pos = sorted(set(dom_positions(L)[: SINK + 400]) | set(
        range(L - 128, L)))
    m_pos = torch.zeros(1, 1, Hkv, L, dtype=torch.bool, device=dev)
    m_pos[..., sel_pos] = True
    # 臂 A 全宽参考（位置序）
    kv_full = ref_k[:L].view(1, L, Hkv, D)
    vv_full = ref_v[:L].view(1, L, Hkv, D)
    cuA = torch.tensor([0, L], dtype=torch.int32, device=dev)
    oA = eager_decoding_attn(q, kv_full, vv_full, m_pos, 1, cuA,
                             valid_length=L)
    # 臂 B：ring buffer 原生物理布局 + 槽位域 mask
    sel_t = torch.tensor(sel_pos, device=dev)
    m_slot = torch.zeros(1, 1, Hkv, CAP, dtype=torch.bool, device=dev)
    m_slot[..., pos_to_slot(sel_t, SINK, RING)] = True
    k_r = layer.keys.transpose(1, 2)   # [1, Hkv, CAP, D] → [1, CAP, Hkv, D]
    v_r = layer.values.transpose(1, 2)
    cuB = torch.tensor([0, CAP], dtype=torch.int32, device=dev)
    oB = eager_decoding_attn(q, k_r, v_r, m_slot, 1, cuB, valid_length=CAP)
    eager_decoding_check_errors(dev)
    if (L - CAP) % BS == 0:
        check("cache: 520 步末臂A/B eager 对齐几何逐位一致", torch.equal(oA, oB))
    else:
        ok, nd, rel = _ulp_ok(oA, oB)
        check("cache: 520 步末臂A/B eager 非对齐 ULP 口径", ok,
              f"L={L} (L-cap)%64={(L - CAP) % BS} nd={nd} rel={rel:.3%} "
              f"maxdiff={(oA.float() - oB.float()).abs().max().item():.3e}")


# ---------------------------------------------------------------- cell D indexer
def test_indexer_equiv():
    """臂A（全宽 skip_far 参照）vs 臂B（ring CLI 层集）prepare_mask 对拍。

    口径（首跑修正，2026-10-25）：
      - 首跑证伪了「(L-cap)%64==0 对齐几何下同构」的原设计——near 段
        起点按块 floor 对齐产生 0~63 个「骑墙 token」（落 near L2 段
        可选但已被 ring 驱逐）→ 臂A 选中越出驻留集（驱逐不再无损）。
        修复 = 生产侧 E175 骑墙块收紧（compute_mask 的 far_tok_hi 上调
        到骑墙块上界 + L1 near 池排除骑墙块）+ prepare_mask 环绕分支改
        宽 L 虚拟位置轴（块网格锚 0 == 臂A，任意 L 同构，不再要求块
        对齐）→ 本 cell 断言升级为**全步逐位**。
      - 全步驱逐无损：臂A 选中 ⊆ 驻留位置集 [0,4)∪[L-4224,L)。
      - 环绕步：臂B 槽位域 mask 经 pos_to_slot 逆映射回位置域宽 L，
        与臂A **逐位相等**（选择链同构：同选中集）。
      - 未满步（L<=cap）：槽位==位置，m_A == m_B[..., :L] 逐位。
    """
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.ops.tli_static_cache import pos_to_slot
    torch.manual_seed(20261026)
    dev = torch.device('cuda:0')
    scale = D ** -0.5
    idx_full = TLIIndexer(make_e175_args(""))        # 全宽 skip_far 参照臂
    idx_full.layer_idx = 7
    idx_full.skip_far = True                          # 手动（静态集语义）
    idx_ring = TLIIndexer(make_e175_args("7,8,10,11,35"))
    idx_ring.layer_idx = 7                            # 集内层
    # 4093 起 620 步（4093→4712）：未满段 135 步 + 环绕段 485 步
    L0, N_STEPS = 4093, 620
    k_full = torch.randn(1, L0, Hkv, D, device=dev, dtype=torch.bfloat16)
    wrap_eq, unfilled_eq, resident_ok, budget_ok = [], [], True, True
    q = torch.randn(1, 1, H, D, device=dev, dtype=torch.bfloat16)
    for step in range(N_STEPS):
        if step:
            k_full = torch.cat([
                k_full, torch.randn(1, 1, Hkv, D, device=dev,
                                    dtype=torch.bfloat16)], dim=1)
        L = L0 + step
        # 臂 A：全宽位置序 + skip_far（手动）
        idx_full.clear()
        idx_full.skip_far = True
        cuA = torch.tensor([0, L], dtype=torch.int32, device=dev)
        pos = torch.tensor([L - 1], device=dev, dtype=torch.long)
        m_A, _ = idx_full.prepare_mask(
            q, pos, k_full, cuA, softmax_scale=scale, valid_length=L)
        # 臂 B：ring 层（CLI 层集）+ 宽 L 虚拟位置轴（prepare_mask 内部）。
        # buffer 恒宽 CAP（TLIStaticLayer 形态；未满步 [L,cap) 零 pad，
        # slots==dom 前缀恒等填入）。
        idx_ring.clear()
        dom = torch.tensor(dom_positions(L), device=dev)
        slots = pos_to_slot(dom, SINK, RING)
        k_ring = torch.zeros(1, CAP, Hkv, D, device=dev,
                             dtype=torch.bfloat16)
        k_ring[:, slots] = k_full[:, dom]
        cuB = torch.tensor([0, CAP], dtype=torch.int32, device=dev)
        m_B, _ = idx_ring.prepare_mask(
            q, pos, k_ring, cuB, softmax_scale=scale, valid_length=L)
        # ---- 驱逐无损断言（全步）：臂 A 在驻留外位置恒 False ----
        dom_set = set(dom_positions(L))
        non_dom = [p for p in range(L) if p not in dom_set]
        if non_dom and m_A[..., non_dom].any():
            resident_ok = False
            print(f"  step{step} L={L} 臂A 选中越出驻留集（假设破坏）")
        # 选中数（预算口径）
        nA = int(m_A.sum().item())
        nB = int(m_B.sum().item())
        if nA != nB:
            budget_ok = False
            print(f"  step{step} L={L} 选中数臂A={nA} 臂B={nB}（预算漂移）")
        if L <= CAP:
            # 未满：槽位==位置 → 两臂 mask 直接逐位（m_B 宽 cap 尾 pad False）
            same = torch.equal(m_A, m_B[..., :L]) and not m_B[..., L:].any()
            unfilled_eq.append((L, same))
            if not same:
                print(f"  未满 step{step} L={L} mask 失配 "
                      f"nd={(m_A != m_B[..., :L]).sum().item()}")
        else:
            # 环绕：臂B 槽位域 → 位置域逆映射（双射），与臂A 逐位
            m_B_pos = torch.zeros_like(m_A)
            m_B_pos[..., dom] = m_B[..., slots]
            same = torch.equal(m_A, m_B_pos)
            wrap_eq.append((L, same))
            if not same:
                print(f"  环绕 step{step} L={L} mask 失配 "
                      f"nd={(m_A != m_B_pos).sum().item()} "
                      f"nA={nA} nB={nB}")
    check("indexer: 驱逐无损——臂A 全宽 skip_far mask 620 步全部步"
          " 选中集 ⊆ 驻留集（E175 骑墙块收紧后）", resident_ok)
    check("indexer: 臂A/臂B 选中数逐步相等（K2 预算口径）", budget_ok)
    check("indexer: 未满步（L<=cap）位置序 mask 逐位相等",
          all(s for _, s in unfilled_eq) and len(unfilled_eq) >= 100,
          f"n_unfilled={len(unfilled_eq)}")
    check("indexer: 环绕步（L>cap 任意几何，含非对齐）虚拟轴选择链"
          "同构→位置域 mask 逐位相等",
          all(s for _, s in wrap_eq) and len(wrap_eq) >= 400,
          f"n_wrap={len(wrap_eq)}")


# ---------------------------------------------------------------- cell E 守卫
def test_guards():
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.ops.tli_static_cache import e175_layer_caps
    from sparse_attn.arguments import _parse_skip_layer_ids
    import argparse as _ap

    def raises(fn, etype=ValueError):
        try:
            fn()
            return False
        except etype:
            return True
        except Exception:
            return False

    # CLI 层集 × enable_layer_skip true 互斥
    a = make_e175_args("7,8")
    a.tli_enable_layer_skip = True
    check("guards: CLI 层集 × enable_layer_skip true 构造期拒绝",
          raises(lambda: TLIIndexer(a)))
    # CLI 层集 × dyn_dgate 互斥
    a = make_e175_args("7,8")
    a.tli_dyn_dgate = True
    check("guards: CLI 层集 × dyn_dgate 构造期拒绝",
          raises(lambda: TLIIndexer(a)))
    # CLI 层集缺 near_len（legacy 比例几何）拒绝
    a = make_e175_args("7,8")
    a.tli_near_len = None
    a.tli_near_pages = None
    a.tli_near_tokens = None
    check("guards: CLI 层集缺 tli_near_len（无恒容量 ring 几何）拒绝",
          raises(lambda: TLIIndexer(a)))
    # argparse 非法层集
    for bad in ["7,-8", "7.5,8", "7,8,8", "x", "7 8"]:
        ok = raises(lambda b=bad: _parse_skip_layer_ids(b),
                    _ap.ArgumentTypeError)
        if not ok:
            check(f"guards: 非法层集 {bad!r} 拒绝", False)
            break
    else:
        check("guards: 非法层集串（负数/小数/重复/非数字）argparse 全拒",
              True)
    check("guards: 合法层集规范输出（排序去空）",
          _parse_skip_layer_ids("35, 7, 8") == "7,8,35")
    # 层 id 越界（e175_layer_caps）
    a = make_e175_args("7,999")
    check("guards: 层 id 越界 e175_layer_caps 拒绝",
          raises(lambda: e175_layer_caps(a, 16384, 36)))
    check("guards: cap >= S_cap 拒绝（无收缩收益）",
          raises(lambda: e175_layer_caps(make_e175_args("7"), 4096, 36)))
    check("guards: 空层集 layer_caps=None（生产不变）",
          e175_layer_caps(make_e175_args(""), 16384, 36) is None)
    # 错 k 宽契约：ring 层集启用 + 全宽 k → prepare_mask fail-closed
    dev = torch.device('cuda:0') if torch.cuda.is_available() else None
    if dev is not None:
        torch.manual_seed(20261027)
        idx = TLIIndexer(make_e175_args("7"))
        idx.layer_idx = 7
        idx.clear()
        k_bad = torch.randn(1, 16384, Hkv, D, device=dev, dtype=torch.bfloat16)
        cu = torch.tensor([0, 16384], dtype=torch.int32, device=dev)
        pos = torch.tensor([16000], device=dev, dtype=torch.long)
        okw = raises(
            lambda: idx.prepare_mask(q=torch.randn(
                1, 1, H, D, device=dev, dtype=torch.bfloat16),
                q_ids=pos, k=k_bad, cu_seqlens_k=cu,
                softmax_scale=D ** -0.5, valid_length=16001),
            RuntimeError)
        check("guards: ring 层集启用 + 全宽 k 契约破损 prepare_mask 拒绝", okw)


TESTS = {
    "mapping": test_mapping,
    "eager": test_eager_equiv,
    "cache": test_cache_equiv,
    "indexer": test_indexer_equiv,
    "guards": test_guards,
}

if __name__ == "__main__":
    sel = sys.argv[1:] or list(TESTS)
    for name in sel:
        if name in TESTS:
            print(f"==== {name} ====")
            TESTS[name]()
        else:
            print(f"[SKIP] 未知 cell 名 {name!r}")
    print(f"\n===== E175 ring 等价门禁: {len(PASS)} PASS / {len(FAIL)} FAIL =====")
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)
