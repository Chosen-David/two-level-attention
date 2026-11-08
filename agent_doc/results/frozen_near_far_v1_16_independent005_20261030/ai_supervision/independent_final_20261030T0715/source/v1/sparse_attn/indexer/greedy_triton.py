#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E113：sim_greedy 增量贪心聚类的 Triton kernel（方案1：单 kernel 顺序化）。

== 地位（E113b e2e 集成）==
本文件是 **生产路径 kernel 本体**（tli_indexer.py `_greedy_cluster_pass`
调度器的 CUDA 路径，env 开关 SGLANG_TLI_GREEDY_KERNEL，默认 1）。
原型/单测/bench 脚本在 two-level-attention/exp/trace/e113_*.py（exp/trace
的 e113_greedy_triton.py 是本文件的转发入口，单一事实源在这里——远程部署
只需 rsync sparse_attn 目录即可携带 kernel）。
设计文档：research/docs/e113_method_kernel_design.md（实证篇）+
research/docs/sim_greedy_kernel_design.md（设计篇）。

== 背景 ==
E110 落地的 `_greedy_cluster_pass`（two-level-attention sparse_attn/indexer/
tli_indexer.py）是逐 token 的 Python 循环：每步 bmm/argmax/gather/index_add
约 10 次 kernel launch，E108 probe 实测 T≈16K 时 ~5.8 s/层（sim=0.9 时 27 s），
是远程 sim_greedy e2e 臂 13 h/臂 的直接根因。

== 本文件语义目标（对齐铁律）==
与 `_greedy_cluster_pass` 同签名同语义：
    greedy_pass_triton(x, sim, sums, cnt, sq, k_live)
        -> (sums, cnt, sq, k_live, assign)
- 贪心时序语义逐 token 精确保留：token i 与「i 时刻的簇心」（running mean）
  比较余弦，argmax + sim 阈值 → 归并或新建；
- 增量续跑（传入非零 k_live 的既有状态）≡ 全量重放（E110 T3 口径）；
- fp 注记：dot 用 tl.sum 逐 tile 归约，与 torch bmm 的归约顺序可能在
  1e-7 量级上不同 → 近并列边界可能翻转（E108 probe 已有先例：sim≥0.85
  逐步精确 norm vs 增量 sq 维护 0.03% 边界差）。单测报告逐位一致率。

== kernel 设计（grid=(H,) 单程序一头，token 循环在 kernel 内）==
- 每 program 独占一个 kv-head 的全部簇状态（sums/cnt/sq），无跨 program
  竞争，无需原子操作；
- 每 token：K 分 tile 扫描 sums 行（HBM/L2 读，tile [BT,DD]），cos =
  dot/(norm·xn+EPS)，tile 内 argmax（first-occurrence tie-break）+
  跨 tile 严格 > 比较（早 tile 优先 = 全局 first-occurrence，对齐 torch
  argmax 语义）；
- 决策与更新 branchless：dst = upd ? best_k : k_live；sums[dst] = upd ?
  row+x : x；cnt/sq 同式（新建簇槽必为零槽，与参考实现「零权重
  index_add 落零槽/活簇」的语义一致）；
- 容量由 wrapper 保证 K ≥ max(k_live)+T（最坏每 token 新建一簇），
  与参考实现的 F.pad 扩容点一致；
- 状态体量小（[H,K,dd] fp32，T=16K 时 ~16MB×H/8…见设计文档 roofline），
  L2 常驻，K 扫描走 L2 带宽。

用法（对拍单测见 exp/trace/test_e113_greedy_triton.py）：
    from sparse_attn.indexer.greedy_triton import greedy_pass_triton, greedy_build_triton
（兼容旧路径：exp/trace/e113_greedy_triton.py 转发导出本模块。）
"""

from __future__ import annotations

import time as _time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------- kernel 本体
@triton.jit
def _greedy_pass_kernel(
    x_ptr,          # [T, H, DD] fp32 连续（与参考实现 x [T,H,dd] 同布局）
    sums_ptr,       # [H, K, DD] fp32 连续（原地更新）
    cnt_ptr,        # [H, K]     fp32 连续（原地更新）
    sq_ptr,         # [H, K]     fp32 连续（原地更新，||sums||^2 增量维护）
    klive_ptr,      # [H] int64：入=初始活簇数，出=终态活簇数
    assign_ptr,     # [H, T] int64 出口
    H: tl.constexpr, DD: tl.constexpr, BT: tl.constexpr,
    K,              # 簇容量（runtime，≥ max(初始 k_live)+T 由 wrapper 保证）
    T,              # 本趟 token 数（runtime）
    SIM,
    EPS: tl.constexpr,
):
    h = tl.program_id(0)
    k_live = tl.load(klive_ptr + h).to(tl.int32)
    offs_d = tl.arange(0, DD)
    for i in range(T):
        x = tl.load(x_ptr + (i * H + h) * DD + offs_d).to(tl.float32)
        xn = tl.sqrt(tl.sum(x * x))
        # ---- K 分 tile 扫描：cos = dot/(norm·xn+EPS)，死簇 -inf ----
        best_cos = float("-inf")
        best_k = tl.zeros((), dtype=tl.int32)
        best_dot = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, k_live, BT):
            offs_k = k0 + tl.arange(0, BT)
            m = offs_k < k_live
            sums_t = tl.load(
                sums_ptr + (h * K + offs_k[:, None]) * DD + offs_d[None, :],
                mask=m[:, None], other=0.0,
            )
            dot_t = tl.sum(sums_t * x[None, :], axis=1)
            sq_t = tl.load(sq_ptr + h * K + offs_k, mask=m, other=0.0)
            norm_t = tl.sqrt(tl.maximum(sq_t, 0.0))
            cos_t = dot_t / (norm_t * xn + EPS)
            cos_t = tl.where(m, cos_t, float("-inf"))
            am = tl.argmax(cos_t, axis=0)          # tile 内 first-occurrence
            tmax = tl.max(cos_t, axis=0)
            lane = tl.arange(0, BT)
            dot_am = tl.sum(tl.where(lane == am, dot_t, 0.0))
            take = tmax > best_cos                 # 严格 >：早 tile 赢并列
            best_cos = tl.where(take, tmax, best_cos)
            best_k = tl.where(take, (k0 + am).to(tl.int32), best_k)
            best_dot = tl.where(take, dot_am, best_dot)
        # ---- 决策 + 更新（branchless；新建簇槽必为零槽）----
        upd = best_cos >= SIM
        dst = tl.where(upd, best_k, k_live)
        row = tl.load(sums_ptr + (h * K + dst) * DD + offs_d)
        tl.store(sums_ptr + (h * K + dst) * DD + offs_d,
                 tl.where(upd, row + x, x))
        c_old = tl.load(cnt_ptr + h * K + dst)
        tl.store(cnt_ptr + h * K + dst, tl.where(upd, c_old + 1.0, 1.0))
        s_old = tl.load(sq_ptr + h * K + dst)
        # 归并：sq += 2·(sums_a·x) + ||x||^2（dot_a 取扫描时的 tile 值，更新前口径）
        tl.store(sq_ptr + h * K + dst,
                 tl.where(upd, s_old + 2.0 * best_dot + xn * xn, xn * xn))
        tl.store(assign_ptr + h * T + i, dst.to(tl.int64))
        k_live = tl.where(upd, k_live, k_live + 1)
    tl.store(klive_ptr + h, k_live.to(tl.int64))


# ---------------------------------------------------------------- wrapper（对齐参考签名）
def greedy_pass_triton(x, sim, sums, cnt, sq, k_live, BT=512, num_warps=4):
    # BT/num_warps 默认值 = E113 GPU1 干净口径扫描最优（T=16384，struct K≈2K）：
    #   BT=128/nw=8: 41.0μs/token → BT=512/nw=4: 9.8μs/token（4.2×）；
    #   BT=1024/nw=4 与 512 持平（9.6 vs 9.8），K=16K 单例臂 512 略优（41 vs 43）；
    #   BT 不影响语义：dot 归约沿 dd 维（行内 32 元素，与 BT 无关），
    #   跨 tile 严格 > 的 first-occurrence tie-break 与 BT 无关 → 改 BT 逐位不变。
    # num_warps=4 时单 tile [512,32] = 64KB fp32，寄存器/L1 平衡点。
    """与 TLIIndexer._greedy_cluster_pass 同签名同语义的 Triton 版。

    x: [T, H, dd]（按序）；sums [H,K,dd] / cnt [H,K] / sq [H,K] / k_live [H]
    为既有簇状态（全零 = 冷启动；非零 = 续跑，与全量重放逐位一致——
    贪心决策只依赖先验状态）。返回 (sums, cnt, sq, k_live, assign[H,T])；
    容量不足时返回扩容新张量（与参考实现的 F.pad 扩容同语义）。
    注意：与参考实现一致，未扩容时 sums/cnt/sq 被原地更新。
    """
    T, H, dd = x.shape
    assert x.is_cuda, "Triton 路径需要 CUDA"
    x = x.float().contiguous()
    need = int(k_live.max().item()) + T
    K = sums.shape[1]
    if need > K:  # 扩容（新簇上界 = T，与参考实现同判据）
        pad = need - K
        sums = F.pad(sums, (0, 0, 0, pad))
        cnt = F.pad(cnt, (0, pad))
        sq = F.pad(sq, (0, pad))
        K = sums.shape[1]
    sums = sums.contiguous()
    cnt = cnt.contiguous()
    sq = sq.contiguous()
    kl = k_live.to(torch.int64).contiguous().clone()
    assign = torch.empty(H, T, dtype=torch.int64, device=x.device)
    BT = min(BT, triton.next_power_of_2(max(K, 16)))
    _greedy_pass_kernel[(H,)](
        x, sums, cnt, sq, kl, assign,
        H=H, DD=dd, BT=BT, K=K, T=T, SIM=float(sim), EPS=1e-9,
        num_warps=num_warps,
    )
    return sums, cnt, sq, kl, assign


def greedy_build_triton(x, sim, chunk=16384, BT=512, num_warps=4):
    """冷启动全序列构建（prefill 路径）：分段喂 greedy_pass_triton。

    分段不改变语义（贪心时序跨段精确延续，同 _update_far_greedy 的
    增量续跑论证）；chunk 只是单 launch 的 token 上限（防超长序列单
    kernel 占用与编译期 K 循环过长）。E113 实测 chunk=16384 比 8192
    快 1.75×（9.8 vs 17.1μs/token，launch 间隙效应），默认 16384。
    返回 (sums, cnt, sq, k_live, assign)。
    """
    T, H, dd = x.shape
    K = T
    sums = x.new_zeros(H, K, dd)
    cnt = x.new_zeros(H, K)
    sq = x.new_zeros(H, K)
    k_live = torch.zeros(H, dtype=torch.int64, device=x.device)
    assign = torch.empty(H, T, dtype=torch.int64, device=x.device)
    for c0 in range(0, T, chunk):
        seg = x[c0:c0 + chunk]
        sums, cnt, sq, k_live, a = greedy_pass_triton(
            seg, sim, sums, cnt, sq, k_live, BT=BT, num_warps=num_warps)
        assign[:, c0:c0 + seg.shape[0]] = a
    return sums, cnt, sq, k_live, assign


# ====================================================================== #
# 【E131-S2】SPLIT 协作版贪心 kernel（v2）：同头多 program 并行扫 K 维。
#
# == 动机（E131 阶段 1 瓶颈实测）==
# v1（grid=(H,) 单 program 一头）在 randn 最坏口径下冷建 S=64K 实测
# 8077ms / 57088 token = 141μs/token（cavgsim/cclustersim 冷建 8-11s 主项；
# near 重建 spike 137ms/8192 token 同源）。根因不是 launch（E113b 已消灭
# Python 循环），而是：
#   a) grid 仅 H=8 个 program，H20 78 个 SM 闲置 90%；
#   b) randn 最坏数据 sim=0.9 下簇永不归并 → k_live 涨满 T，每 token 的
#      活簇扫描 O(k_live·dd) → 全程 O(T²) 扫描，总流量 ~1.6TB / 8.4s ≈
#      190GB/s（HBM 4TB/s 的 5%）——单 program 串行 tile 循环是瓶颈，
#      带宽远未饱和，纯并行度问题。
# BT/num_warps 调参实测仅 1.2×（131→111μs/token），证实须增 program 数。
#
# == 协议（per-token 两阶段 + 两个软件 barrier）==
# grid=(H, SPLIT)：同头 SPLIT 个 program 协作。每 token i：
#   phase1：program p 扫本头活簇区间 [p·kl/SPLIT, (p+1)·kl/SPLIT)（KLIVE
#           大时均分扫描量），局部 best (cos,k,dot) 写 scratch[h,p]；
#   barrier1（原子计数 + spin，release-acquire）：scratch 写全局可见；
#   phase2：每 program 独立读 scratch 全 SPLIT 行做全局归约（p 升序严格 >
#           比较 = 簇索引升序 first-occurrence，与 v1 跨 tile 语义一致）
#           → upd/dst 决策；p==0 单写者更新 sums/cnt/sq/assign（避免
#           多 program 同地址 read-modify-write 竞态），其余 program 仅
#           本地推进 k_live（决策幂等，各 program 独立算出同值）；
#   barrier2：p==0 的状态更新对下一 token phase1 可见。
# counter 单调递增无重置（target=(2i+1/2)·SPLIT），无 ABA。
#
# == 逐位一致性论证（对拍口径同 test_e113_greedy_triton.py）==
#   dot/cos 的归约沿 dd 维逐行进行（tl.sum(axis=1)），与 tile 边界/program
#   切分完全无关 → 行级数值与 v1 逐位同；跨行 max/argmax 的比较序 =
#   簇索引升序严格 >（program p 升序 × 区间内 tile 升序），与 v1 跨 tile
#   严格 > 序完全一致 → argmax/dot_a/gather 序列逐位同。sq 更新用扫描
#   时 best_dot（更新前口径），同 v1。故 (sums,cnt,sq,k_live,assign)
#   与 v1 数学逐位一致（对拍单测验证）。
#
# == 死锁防御 ==
# grid block 数 H·SPLIT ≤ SM 数时保守安全（每 block 仅 4 warps + 少量
# scratch 寄存器，每 SM 至少驻留 1 block → 全 grid 必然并发驻留，spin
# barrier 不会等未调度的 program）。wrapper 按 device SM 数自动钳制
# SPLIT，env 可覆盖（TLI_E131_GREEDY_SPLIT）。
#
# == 【E132b TL-E131S2-BARRIER-RESIDENCY-130】limitation（如实声明）==
# 跨架构 residency 未证明，H20 实证通过（e131s2_v2_validate V1-V5 全
# PASS + e2e mask 逐位 12585120/12585120；本机 2×H20-3e）。_v2_
# residency_clamp 的寄存器计数法不构成跨架构无死锁证明（未覆盖编译后
# shared memory 占用与寄存器分配粒度，见其 docstring）；未知架构/
# 新编译器版本上首次运行前应先小规模探测。带外 watchdog 钩子见
# greedy_pass_triton_v2 wrapper（TLI_E131_V2_TIMEOUT_MS）。
# ====================================================================== #

@triton.jit
def _greedy_pass_kernel_v2(
    x_ptr,          # [T, H, DD] fp32 连续
    sums_ptr,       # [H, K, DD] fp32 连续
    cnt_ptr,        # [H, K] fp32 连续
    sq_ptr,         # [H, K] fp32 连续
    klive_ptr,      # [H] int64：入=初始活簇数，出=终态活簇数
    assign_ptr,     # [H, T] int64 出口
    scos_ptr,       # [H, SPLIT] fp32：phase1 局部 best cos scratch
    sk_ptr,         # [H, SPLIT] int32：局部 best 簇号
    sdot_ptr,       # [H, SPLIT] fp32：局部 best dot
    bar_ptr,        # [H] int32：barrier 计数（单调递增，无重置）
    H: tl.constexpr, DD: tl.constexpr, BT: tl.constexpr, SPLIT: tl.constexpr,
    K, T, SIM,
    EPS: tl.constexpr,
):
    h = tl.program_id(0)
    p = tl.program_id(1)
    k_live = tl.load(klive_ptr + h).to(tl.int32)
    offs_d = tl.arange(0, DD)
    for i in range(T):
        x = tl.load(x_ptr + (i * H + h) * DD + offs_d).to(tl.float32)
        xn = tl.sqrt(tl.sum(x * x))
        # ---- phase1：本 p 扫活簇区间 [p·kl/SPLIT, (p+1)·kl/SPLIT) ----
        # 整数除法均分（差 ≤1 行），与 v1 的行级数值无关（dot 归约沿 dd 维）
        lo_k = p * k_live // SPLIT
        hi_k = (p + 1) * k_live // SPLIT
        best_cos = float("-inf")
        best_k = tl.zeros((), dtype=tl.int32)
        best_dot = tl.zeros((), dtype=tl.float32)
        for k0 in range(lo_k, hi_k, BT):
            offs_k = k0 + tl.arange(0, BT)
            m = offs_k < hi_k
            sums_t = tl.load(
                sums_ptr + (h * K + offs_k[:, None]) * DD + offs_d[None, :],
                mask=m[:, None], other=0.0,
            )
            dot_t = tl.sum(sums_t * x[None, :], axis=1)
            sq_t = tl.load(sq_ptr + h * K + offs_k, mask=m, other=0.0)
            norm_t = tl.sqrt(tl.maximum(sq_t, 0.0))
            cos_t = dot_t / (norm_t * xn + EPS)
            cos_t = tl.where(m, cos_t, float("-inf"))
            am = tl.argmax(cos_t, axis=0)          # tile 内 first-occurrence
            tmax = tl.max(cos_t, axis=0)
            lane = tl.arange(0, BT)
            dot_am = tl.sum(tl.where(lane == am, dot_t, 0.0))
            take = tmax > best_cos                 # 严格 >：早 tile 赢并列
            best_cos = tl.where(take, tmax, best_cos)
            best_k = tl.where(take, (k0 + am).to(tl.int32), best_k)
            best_dot = tl.where(take, dot_am, best_dot)
        tl.store(scos_ptr + h * SPLIT + p, best_cos)
        tl.store(sk_ptr + h * SPLIT + p, best_k)
        tl.store(sdot_ptr + h * SPLIT + p, best_dot)
        # ---- barrier1：scratch 全局可见（release-acquire 对同地址原子）----
        target1 = (2 * i + 1) * SPLIT
        tl.atomic_add(bar_ptr + h, 1, sem="acq_rel", scope="gpu")
        v = tl.atomic_add(bar_ptr + h, 0, sem="acquire", scope="gpu")
        while v < target1:
            v = tl.atomic_add(bar_ptr + h, 0, sem="acquire", scope="gpu")
        # ---- phase2：全局归约（p 升序严格 > = 簇索引升序 first-occurrence）----
        g_cos = float("-inf")
        g_k = tl.zeros((), dtype=tl.int32)
        g_dot = tl.zeros((), dtype=tl.float32)
        for q in range(SPLIT):                     # constexpr 展开小循环
            c = tl.load(scos_ptr + h * SPLIT + q)
            take = c > g_cos
            g_cos = tl.where(take, c, g_cos)
            g_k = tl.where(take, tl.load(sk_ptr + h * SPLIT + q), g_k)
            g_dot = tl.where(take, tl.load(sdot_ptr + h * SPLIT + q), g_dot)
        upd = g_cos >= SIM
        dst = tl.where(upd, g_k, k_live)
        if p == 0:                                 # 单写者：无 RMW 竞态
            row = tl.load(sums_ptr + (h * K + dst) * DD + offs_d)
            tl.store(sums_ptr + (h * K + dst) * DD + offs_d,
                     tl.where(upd, row + x, x))
            c_old = tl.load(cnt_ptr + h * K + dst)
            tl.store(cnt_ptr + h * K + dst, tl.where(upd, c_old + 1.0, 1.0))
            s_old = tl.load(sq_ptr + h * K + dst)
            # 归并：sq += 2·(sums_a·x) + ||x||^2（g_dot 为扫描时口径，更新前）
            tl.store(sq_ptr + h * K + dst,
                     tl.where(upd, s_old + 2.0 * g_dot + xn * xn, xn * xn))
            tl.store(assign_ptr + h * T + i, dst.to(tl.int64))
        k_live = tl.where(upd, k_live, k_live + 1)  # 各 program 幂等同值
        # ---- barrier2：p==0 的状态更新对下一 token phase1 可见 ----
        target2 = (2 * i + 2) * SPLIT
        tl.atomic_add(bar_ptr + h, 1, sem="acq_rel", scope="gpu")
        v = tl.atomic_add(bar_ptr + h, 0, sem="acquire", scope="gpu")
        while v < target2:
            v = tl.atomic_add(bar_ptr + h, 0, sem="acquire", scope="gpu")
    if p == 0:
        tl.store(klive_ptr + h, k_live.to(tl.int64))


def _v2_split_bound(H, device):
    """SPLIT 默认值（env 覆盖优先；0=禁用 v2 回退 v1）。

    实测拐点（H20 78SM，T=14272 randn 最坏）：SPLIT 9→7.97μs/tok、
    16→6.94、24→6.62（拐点）、32→6.81（phase2/barrier 税反超）→ 默认
    min(24, 4·SM//H)。死锁安全由 _v2_residency_clamp 二次钳制（按编译
    后实测寄存器数），本函数只定「想要多少」。"""
    import os as _os
    env = _os.environ.get("TLI_E131_GREEDY_SPLIT", "")
    if env:
        return max(0, int(env))
    try:
        nsm = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 0
    return min(24, max(1, nsm * 4 // max(1, H)))


# 寄存器探测缓存：key=(device_idx, H, dd, BT, num_warps, SPLIT) → n_regs。
# 探测用 warmup 只编译不 launch（grid 无须全驻留，无死锁面）。
_V2_NREGS_CACHE = {}


def _v2_residency_clamp(SPLIT, H, dd, BT, num_warps, device, probe_tensors):
    """死锁防御（实证必要）：BT=512/nw=4 组合实测 n_regs 上升 → 3
    blocks/SM → SPLIT=32（256 blocks > 78×3=234 容量）grid 无法全驻留
    → spin barrier 死锁（E131-S2 扫描实证挂死）。修法：warmup 编译
    探测真实 n_regs → 驻留上界 SM·min(blocks_reg, blocks_thread,
    blocks_limit) // H 钳制 SPLIT。探测一次缓存（n_regs 对 SPLIT 不敏
    感，phase2 小展开差异不改变寄存器峰值）。探测异常 → 返回 1（禁用
    v2 回退 v1，fail-closed）。

    【E132b TL-E131S2-BARRIER-RESIDENCY-130 limitation（如实声明）】
    本 clamp 是寄存器计数启发式，【不是】跨架构无死锁证明：
      - 未纳入编译后 shared memory 占用（本 kernel 静态 smem 少，但
        跨 Triton 版本/新特化不保证；smem 超限会把 blocks/SM 压到
        估算值之下 → 高估驻留数 → barrier 永久自旋）；
      - 未纳入寄存器分配粒度（n_regs 按 warp 均匀分配假设；NVIDIA
        最佳实践明确要求用 occupancy API 而非简单寄存器计数）；
      - 探测的 SPLIT 特化与实际 launch 的 SPLIT 特化可能不同（返回
        更小 SPLIT 时靠「n_regs 对 SPLIT 不敏感」的经验假设）。
    跨架构 residency 未证明；H20-3e 实证通过（V1-V5 + e2e mask 逐位）。
    未知架构/新编译器首次运行应先小规模探测（探测失败已 fail-closed
    回退 v1）。带外 watchdog：TLI_E131_V2_TIMEOUT_MS（见 wrapper）。"""
    try:
        dev = torch.device(device) if isinstance(device, str) else device
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev_idx)
        nsm = prop.multi_processor_count
        threads = num_warps * 32
        key = (dev_idx, H, dd, BT, num_warps, SPLIT)
        n_regs = _V2_NREGS_CACHE.get(key)
        if n_regs is None:
            # warmup 只编译不 launch（无死锁面）；CompiledKernel 的
            # n_regs 惰性填充在 _init_handles（装载 cubin 句柄），
            # 手动触发后即可读
            x_t = probe_tensors
            ck = _greedy_pass_kernel_v2.warmup(
                *x_t[:10], K=x_t[10], T=x_t[11], SIM=x_t[12],
                H=H, DD=dd, BT=BT, SPLIT=SPLIT, EPS=1e-9,
                num_warps=num_warps, grid=(H, SPLIT))
            try:
                ck._init_handles()
                n_regs = int(getattr(ck, "n_regs", 0) or 0)
                if n_regs <= 0:
                    raise RuntimeError(f"n_regs 探测失败: {n_regs}")
                n_regs = max(n_regs, 32)
            except Exception:
                # fail-closed：探测失败 → 返回 1（禁用 v2 回退 v1），
                # 不能假设小寄存器（方向反了会放行死锁 grid）
                return 1
            _V2_NREGS_CACHE[key] = n_regs
        regs_per_sm = getattr(prop, "regs_per_multiprocessor", 65536)
        blocks_reg = max(1, regs_per_sm // (threads * n_regs))
        blocks_thr = max(1, prop.max_threads_per_multi_processor // threads)
        blocks_hw = max(1, getattr(prop, "max_blocks_per_multi_processor", 32))
        bps = min(blocks_reg, blocks_thr, blocks_hw)
        return min(SPLIT, max(1, nsm * bps // max(1, H)))
    except Exception:
        return 1


def greedy_pass_triton_v2(x, sim, sums, cnt, sq, k_live, BT=256, num_warps=4):
    """与 greedy_pass_triton 同签名同语义的 SPLIT 协作版（逐位一致，
    对拍见 exp/trace/test_e113_greedy_triton.py 同口径）。

    仅并行化不同（同头 SPLIT 个 program 均分活簇扫描），语义/数值/扩容
    判据与 v1 完全一致；SPLIT 由 _v2_split_bound 按 SM 数钳制，SPLIT<2
    时由调度器回退 v1（单 program 并行无收益只加 barrier 税）。
    BT 默认 256 ≠ v1 的 512：SPLIT 均分后每 program 扫描区间小，实测
    （H20 SPLIT=24，randn 最坏）256 比 512 快 ~6%（6.62 vs 7.07μs/
    token）；BT 只影响 tile 边界不影响行级数值 → 改 BT 逐位不变。
    回退 v1 时显式用 v1 默认 BT=512：E131-S2 实证大 k_live 续跑
    （T=64 @k_live≈57K randn singleton）下 BT=256 比 512 慢 2.83×
    （61.6 vs 21.8ms）——每 tile 固定开销（sq 加载/argmax/lane 归约）
    在 57K 行 / 224 tiles 时主导，E113 的「6% 差」结论只在 struct
    K≈2K 小簇口径成立，大簇态必须用大 BT。

    【E132b TL-E131S2-BARRIER-RESIDENCY-130】跨架构 residency 未证明
    （H20 实证通过；_v2_residency_clamp 寄存器计数法的 limitation 见
    其 docstring）。带外 watchdog 钩子：launch 前 wrapper 只记录 host
    时间戳 + 超时阈值（TLI_E131_V2_TIMEOUT_MS，默认 120000ms，≤0 禁
    用）——【不】在 launch 后加 synchronize 计时（会破坏异步语义：
    decode step 内 kernel 本就异步排队，强同步会把流水线串行化，且
    他进程占卡时 sync 等待会被误报成 kernel 超时）。超时检查由带外
    消费：调用方在【自有】自然同步点（k_live.max().item() / evs
    synchronize() / 结果消费）之后调用 v2_watchdog_overdue() 查询
    （advisory：未同步前 elapsed 含排队时间，不得据此判定死锁）。"""
    T, H, dd = x.shape
    assert x.is_cuda, "Triton 路径需要 CUDA"
    SPLIT = _v2_split_bound(H, x.device)
    if SPLIT < 2 or T < 256:
        # 小 T / SPLIT 不可用 → v1 路径（barrier 税不划算；语义同；
        # BT 用 v1 默认 512，见 docstring 大簇态实证）
        return greedy_pass_triton(x, sim, sums, cnt, sq, k_live,
                                  BT=512, num_warps=num_warps)
    x = x.float().contiguous()
    need = int(k_live.max().item()) + T
    K = sums.shape[1]
    if need > K:  # 扩容（新簇上界 = T，与参考实现同判据）
        pad = need - K
        sums = F.pad(sums, (0, 0, 0, pad))
        cnt = F.pad(cnt, (0, pad))
        sq = F.pad(sq, (0, pad))
        K = sums.shape[1]
    sums = sums.contiguous()
    cnt = cnt.contiguous()
    sq = sq.contiguous()
    kl = k_live.to(torch.int64).contiguous().clone()
    assign = torch.empty(H, T, dtype=torch.int64, device=x.device)
    # phase1 scratch + barrier counter（每次调用新分配：T≤16K/段 → 计数
    # ≤ 2·T·SPLIT < 2^31 无溢出；单调递增无需重置，kernel 返回即废弃）
    # ---- BT 自适应（E131-S2 实证）：每 program 扫描行数 k_live/SPLIT ≥
    # 1024 时 per-tile 固定开销（sq 加载/argmax/lane 归约）占比上升，
    # BT=512 比 256 快 14%（续跑 8192@k_live=28K：116.9 vs 135.1ms）；
    # 冷建/小簇（k_live 起步≈0）保持 256（V6 实测 256 比 512 快 6%）。
    # BT 只影响 tile 边界不影响行级数值（dot 归约沿 dd 维）→ 逐位不变。
    if max(int(k_live.max().item()), 1) // SPLIT >= 1024:
        BT = max(BT, 512)
    BT = min(BT, triton.next_power_of_2(max(K, 16)))
    scos = torch.empty(H, SPLIT, dtype=torch.float32, device=x.device)
    sk = torch.empty(H, SPLIT, dtype=torch.int32, device=x.device)
    sdot = torch.empty(H, SPLIT, dtype=torch.float32, device=x.device)
    bar = torch.zeros(H, dtype=torch.int32, device=x.device)
    # ---- 死锁防御：按编译后实测 n_regs 钳制 SPLIT（见函数 docstring
    # 与 _v2_residency_clamp；warmup 只编译不 launch，无死锁面）----
    SPLIT = _v2_residency_clamp(
        SPLIT, H, dd, BT, num_warps, x.device,
        (x, sums, cnt, sq, kl, assign, scos, sk, sdot, bar, K, T, float(sim)))
    if SPLIT < 2:
        # clamp 失败回退 v1（BT 同上：v1 默认 512）
        return greedy_pass_triton(x, sim, sums, cnt, sq, k_live,
                                  BT=512, num_warps=num_warps)
    scos = torch.empty(H, SPLIT, dtype=torch.float32, device=x.device)
    sk = torch.empty(H, SPLIT, dtype=torch.int32, device=x.device)
    sdot = torch.empty(H, SPLIT, dtype=torch.float32, device=x.device)
    bar = torch.zeros(H, dtype=torch.int32, device=x.device)
    # ---- 【E132b TL-E131S2-BARRIER-RESIDENCY-130】带外 watchdog：launch
    # 前记录 host 时间戳 + 超时阈值（不 synchronize、不计时 launch——
    # 破坏异步语义）。barrier 死锁的带外检测由调用方在自有同步点后调
    # v2_watchdog_overdue() 完成（advisory，见其 docstring）。----
    global _V2_LAUNCH_TS
    _V2_LAUNCH_TS = (_time.monotonic(), _v2_timeout_s())
    _greedy_pass_kernel_v2[(H, SPLIT)](
        x, sums, cnt, sq, kl, assign, scos, sk, sdot, bar,
        H=H, DD=dd, BT=BT, SPLIT=SPLIT, K=K, T=T, SIM=float(sim), EPS=1e-9,
        num_warps=num_warps,
    )
    return sums, cnt, sq, kl, assign


# ---- 【E132b TL-E131S2-BARRIER-RESIDENCY-130】带外 watchdog 钩子框架 ----
# v2 kernel 的跨 program 自旋 barrier 在 residency 高估时会永久自旋
# （见 _v2_residency_clamp limitation）。带外 watchdog 只做两件事：
#   1) wrapper launch 前记录 (host 时间戳, 超时秒)——零 GPU 开销、不改
#      变异步语义；
#   2) 调用方在【自有】自然同步点之后调 v2_watchdog_overdue() 查询是否
#      超时（elapsed 含 kernel 执行 + 排队时间；同步点之后仍超时才有
#      死锁嫌疑——advisory 判定，非精确 kernel 计时）。
# 在 wrapper 内 launch 后加 torch.cuda.synchronize + 计时会破坏异步
# 语义（串行化 decode 流水线 + 他进程占卡时排队等待会被误报），故
# 不做——死锁检测完全走带外（本框架）+ residency clamp（预防侧）。
_V2_LAUNCH_TS = None   # (launch_host_ts, timeout_s)：最近一次 v2 launch


def _v2_timeout_s():
    """TLI_E131_V2_TIMEOUT_MS（毫秒，默认 120000）→ 秒；≤0 / 非法值 = 0
    （禁用带外检查）。"""
    import os as _os
    env = _os.environ.get("TLI_E131_V2_TIMEOUT_MS", "")
    try:
        ms = float(env) if env else 120000.0
    except ValueError:
        ms = 120000.0
    return max(0.0, ms / 1000.0)


def v2_watchdog_overdue():
    """带外 advisory 检查：返回 (overdue: bool, elapsed_s: float|None)。

    调用方必须已越过【自有】同步点（wrapper 之后的 k_live.max().item()
    / evs.synchronize() / 结果消费等）再判定 overdue=True——否则 kernel
    可能只是仍在队列中未执行，elapsed 含排队时间会误报。overdue=True
    的处置建议：进程级带外手段（独立线程 + cuda launch abort /
    torch.cuda._sleep 探针 / 直接重启该请求），不要在当前流上加同步。
    无 launch 记录或超时禁用 → (False, None)。"""
    if _V2_LAUNCH_TS is None:
        return False, None
    ts, timeout_s = _V2_LAUNCH_TS
    if timeout_s <= 0.0:
        return False, None
    elapsed = _time.monotonic() - ts
    return elapsed > timeout_s, elapsed


__all__ = ["greedy_pass_triton", "greedy_build_triton", "_greedy_pass_kernel",
           "greedy_pass_triton_v2", "_greedy_pass_kernel_v2",
           "_v2_timeout_s", "v2_watchdog_overdue"]
