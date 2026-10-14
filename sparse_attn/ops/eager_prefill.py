# -*- coding: utf-8 -*-
"""稀疏 prefill（质量路径，TLI_SPARSE_PREFILL=1 门控）——gather 紧凑高性能版。

用户 2026-10-14 指令：全部 fix 成高性能版本，低效率版本不要留——原
einsum 全 T 算 score + torch.where(m, -inf) 的 mask 方法实现已删除
（全序列 K/V 读带宽 + 全序列 QK 计算，32K 序列下稀疏选择零节省，
还倒贴 mask 构造开销）。

chunk 共享选择语义（MoBA 口径，MoBA 论文 §3.2 TopK Gating）：
  prefill 的 L 行 query 按 CHUNK=2048 行切块；对每个 chunk 取末行 q 调
  indexer.prepare_mask 做一次选择（末行因果可见全部前缀 → 选择对块内
  所有行合法），chunk 内所有行共享该选择集，再与各自行因果下三角做
  交集（行 r 只可见 [0, r]）；sink 区 [0,128) 由 indexer 的 mask 强制
  保留（compute_mask 的 topk_mask[..., :sink_tok]=True）。

首 chunk 与短行 dense：chunk 起始位置 < K2（tia_level2_topk，默认 1024）
的行（含整个首 chunk）走 dense——等价 identity 选择，对齐 B02/S1 的
短行语义（短序列 dense 等价）。

swa 语义【10-09 TL-PREFILL-SWA-002 修复（GPT 审计）】：共享选择只强制
chunk 末行的 sink/SWA，块内早行自己的 128-token 局部窗口不保证在选集内
（例：chunk [2048,3000) 首行的自身 SWA [1921,2048] 与末行 SWA [2872,2999]
零重叠）→ 每行显式 OR 自身保护带（sink [0,128) + 自身窗口 [p-127,p]，
∩ 因果），保护 token 不占 mid top-k 预算（对齐 decode 侧 sink/swa 强制
区在 K2_mid 之外的契约）。

【E138b gather 化（2026-10-14 用户指令）】高性能实现：
  行块内各行可见集的并集 = (共享 sel ∩ [0, p_max]) ∪ sink [0, sink_tok)
  ∪ 保护带并集 [max(0, p_first-swa_tok+1), p_max]（行间自身窗口连续
  拼接成一条带）。并集 stable argsort 选中前置 compact 成 per-head 索引
  [RhG, W]（W ≈ K2 + R + swa + sink），score einsum 与 K/V 读带宽只对
  W 列计算——T=32K、W≈1.2K 时 ~25× 计算与带宽节省。行级因果/自身
  保护带在 W 宽上做小 mask（被 mask 列 exp=0 不进 softmax 分母，与原
  mask 版同可见集 softmax 数学等价；compact 用 stable 升序保持非零项
  reduce 顺序，仅树形归约分组 ULP 级漂移，对拍 allclose 验证）。
  索引重建：chunk 间 append-only 走 E127 增量缓存（_prepare_index_inc
  命中条件 S_now > inc["S"]），无 O(S²) 重建。
"""
import os

import torch
import torch.nn.functional as F
from einops import rearrange, einsum

# MoBA 口径的 chunk 行数（gate 粒度）
SPARSE_PREFILL_CHUNK = 2048
# 行子块 fp32 score 峰值元素数预算（128M 元素 ≈ 512MB fp32）
_ROW_BLOCK_MAX_ELEM = 1 << 27


def _row_block_size(rhg, w_sel_max, swa_tok, sink_tok=0, budget=_ROW_BLOCK_MAX_ELEM):
    """行子块尺寸 R：score 峰值 [R, RhG, W] 元素预算，其中
    W ≈ w_sel_max + R + swa_tok + sink_tok（保护带并集宽含 R 行自身
    窗口，另加 sink 强制区）。解
    R² + (w_sel_max + swa_tok + sink_tok)·R − budget/RhG ≤ 0 的正根。"""
    b = w_sel_max + swa_tok + sink_tok + 1
    c = budget // max(rhg, 1)
    r = int(0.5 * (-b + (b * b + 4 * c) ** 0.5))
    return max(1, r)


def _chunk_rows_attn(q_c, k_c, v_c, sel, pos_lo, G, softmax_scale,
                     sink_tok=0, swa_tok=0, w_sel_cap=0):
    """单个 chunk 的行级 attention（高性能 gather 紧凑版）。

    q_c: [1, len, H, D]；k_c/v_c: [1, T, Hkv, D]；sel: [Hkv, T]（kv-head
    共享）或 [H, T]（E103 per_q_head）的 token 级 bool 选择，None=dense。
    行 r 的绝对位置 = pos_lo + r，可见 keys = sel ∪ 自身保护带
    （sink [0, sink_tok) ∪ [p-swa_tok+1, p]）∩ 因果 [0, p]。
    dense（sel=None）保持全 T einsum（首 chunk identity 选择无浪费）。

    数值口径对齐 decode 侧 eager_decoding_attn：score 在 fp32 计算
    （q*scale → fp32、k → fp32），softmax fp32 后转回模型 dtype 与 v 相乘。
    """
    len_r, H, D = q_c.shape[1], q_c.shape[2], q_c.shape[3]
    T, Hkv = k_c.shape[1], k_c.shape[2]
    device = q_c.device
    per_qh = sel is not None and sel.shape[0] == H
    # RhG = gather 索引行数：共享 [Hkv]，per_q_head [H]（展开到 q-head）
    rhg = H if per_qh else Hkv

    o_c = torch.empty_like(q_c)
    if sel is None:
        # dense：全 T 计算（首 chunk/短行 identity 语义）
        R = _row_block_size(H, T, 0)
        kcol = torch.arange(T, device=device)
        b_k = k_c[0].to(torch.float32)  # 行块外一次物化（块间重复 cast 浪费）
        for r0 in range(0, len_r, R):
            r1 = min(r0 + R, len_r)
            b_q = rearrange(
                q_c[0, r0:r1] * softmax_scale, "r (h g) d -> r h g d", g=G
            ).to(torch.float32)
            b_s = einsum(b_q, b_k, "r h g d, t h d -> r h g t")
            causal = (
                torch.arange(pos_lo + r0, pos_lo + r1, device=device)[:, None]
                >= kcol[None, :]
            )
            # masked_fill_ 与原 torch.where 逐位相同（b_s 来自 einsum 为新鲜
            # 张量，原地安全），省一份 score 峰值分配
            b_s = b_s.masked_fill_(~causal[:, None, None, :], float("-inf"))
            b_p = F.softmax(b_s, dim=-1).to(q_c.dtype)
            b_o = einsum(b_p, v_c[0], "r h g t, t h d -> r h g d")
            o_c[0, r0:r1] = rearrange(b_o, "r h g d -> r (h g) d").to(q_c.dtype)
        return o_c

    # ---- 稀疏：行块级 gather 紧凑计算 ----
    sel_e = sel if not per_qh else sel.reshape(-1, T)      # [RhG, T]
    # 共享选择满额上界：w_sel_cap（调用方按 indexer 配置算的理论满额，
    # K2 mid + sink + 末行 swa + 块对齐松弛）；w_sel_cap=0 时回退 GPU
    # 实测（.item() 同步，仅独立调用形态兜底）。上界宽松无害——W 用
    # 上界多出的 pad 列由 valid mask 屏蔽（exp=0 不进 softmax 分母）
    if w_sel_cap > 0:
        w_sel_max = int(w_sel_cap)
    else:
        w_sel_max = int(sel_e[:, : T].sum(dim=-1).max().item())
    R = _row_block_size(rhg, w_sel_max, swa_tok, sink_tok)
    for r0 in range(0, len_r, R):
        r1 = min(r0 + R, len_r)
        p_first = pos_lo + r0
        p_max = pos_lo + r1 - 1
        # 行块可见并集：sel ∩ [0, p_max] ∪ sink ∪ 保护带并集
        # [max(0, p_first-swa_tok+1), p_max]（行间窗口连续拼接成一条带）
        vis = sel_e[:, : p_max + 1].clone()
        vis[:, : min(sink_tok, p_max + 1)] = True
        prot_lo = max(0, p_first - swa_tok + 1)
        if swa_tok > 0:
            vis[:, prot_lo:] = True
        else:
            # swa=0 时对角线段须显式并集进 vis（保证空行兜底的
            # 对角线位置 ∈ idx）；swa>0 时对角线段 ⊆ 保护带并集为
            # no-op，跳过省一次 kernel 发射
            vis[:, p_first : p_max + 1] = True
        # compact：选中前置（stable 保持升序——softmax 非零项顺序与
        # 原 mask 版一致，reduce 分组漂移最小化）。W 用静态上界替代
        # counts.max().item()（每行块一次 host-GPU 同步——上界差量由
        # valid mask 屏蔽，数值不变；行块形状恒定对 allocator 友好）
        order = torch.argsort(~vis, dim=-1, stable=True)
        counts = vis.sum(dim=-1)                           # [RhG]
        W = min(w_sel_max + sink_tok + (r1 - r0) + max(swa_tok, 1),
                p_max + 1)
        idx = order[:, :W]                                 # [RhG, W] 升序位置
        pos_w = idx
        in_sel = torch.gather(sel_e[:, : p_max + 1], 1, idx)
        valid = (
            torch.arange(W, device=device)[None, :] < counts[:, None]
        )                                                  # [RhG, W] pad 列
        # gather K/V：只读 W 列。per_qh 不做 K/V 按 GQA 展开的整体物化
        # （原 advanced indexing `k[:, kv_of, :]` 会复制全 K 的 4× 展开体，
        # T=32K 时 ~268MB/chunk），直接在 [Hkv,·,T,D] 扩展视图上按组 gather
        b_q = rearrange(
            q_c[0, r0:r1] * softmax_scale, "r (h g) d -> r h g d", g=G
        ).to(torch.float32)                                # [R, Hkv, G, D]
        if per_qh:
            k_hv = k_c[0].permute(1, 0, 2)                 # [Hkv, T, D] 视图
            v_hv = v_c[0].permute(1, 0, 2)
            # q-head 行序 (h g)：h = hkv*G + g 连续 → idx 直接 reshape 分组
            g_idx = idx.reshape(Hkv, G, W).unsqueeze(-1).expand(
                Hkv, G, W, D
            )                                              # [Hkv, G, W, D]
            kg = torch.gather(
                k_hv.unsqueeze(1).expand(Hkv, G, T, D), 2, g_idx
            )                                              # [Hkv, G, W, D]
            vg = torch.gather(
                v_hv.unsqueeze(1).expand(Hkv, G, T, D), 2, g_idx
            )
        else:
            k_h = k_c[0].permute(1, 0, 2)                  # [Hkv, T, D]
            v_h = v_c[0].permute(1, 0, 2)
            kg = torch.gather(k_h, 1, idx.unsqueeze(-1).expand(-1, -1, D))
            vg = torch.gather(v_h, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        # score：fp32 只算 W 列
        b_k = kg.to(torch.float32)
        if per_qh:
            b_s = einsum(b_q, b_k, "r h g d, h g w d -> r h g w")
        else:
            b_s = einsum(b_q, b_k, "r h g d, h w d -> r h g w")
        # 行级可见 mask（W 宽小 mask）：行 p_r 可见 = pos ≤ p_r 且
        #   (in_sel | pos < sink_tok | pos ≥ p_r-swa_tok+1)，pad 列恒 False
        p_r = torch.arange(pos_lo + r0, pos_lo + r1, device=device)
        pos_ok = pos_w[None, :, :] <= p_r[:, None, None]    # [R, RhG, W]
        row_ok = (
            pos_ok
            & (in_sel[None, :, :]
               | (pos_w[None, :, :] < sink_tok)
               | (pos_w[None, :, :] >= (p_r[:, None, None] - swa_tok + 1)))
            & valid[None, :, :]
        )
        diag = (pos_w[None, :, :] == p_r[:, None, None]) & valid[None, :, :]
        # 空行对角线兜底（防御：选择∩因果全空行——sink 强制 + pos≥K2
        # 下不应触发；对角线段已并入 vis，兜底位置 ∈ idx 恒可精确 OR；
        # 非空行不 OR 对角线，与原 mask 版逐行可见集严格一致）。
        # 向量化无分支：消除 bool(any) 的行块内 host-GPU 同步
        empty = ~row_ok.any(dim=-1)                         # [R, RhG]
        row_ok = row_ok | (diag & empty[..., None])
        # b_s 是 [R, Hkv, G, W]：per_qh 时 RhG=H 需展开回 h/g 两维
        if per_qh:
            m_fill = row_ok.view(r1 - r0, Hkv, G, W)
        else:
            m_fill = row_ok[:, :, None, :]
        # masked_fill_ 原地（b_s 来自 einsum 为新鲜张量），省一份
        # [R, RhG, W] fp32 峰值分配
        b_s = b_s.masked_fill_(~m_fill, float("-inf"))
        b_p = F.softmax(b_s, dim=-1).to(q_c.dtype)          # [R, Hkv, G, W]
        # 输出：vg 转模型 dtype 后 einsum（与 v 相乘口径同原版）
        vg_d = vg.to(q_c.dtype)
        if per_qh:
            b_o = einsum(b_p, vg_d, "r h g w, h g w d -> r h g d")
        else:
            b_o = einsum(b_p, vg_d, "r h g w, h w d -> r h g d")
        o_c[0, r0:r1] = rearrange(b_o, "r h g d -> r (h g) d").to(q_c.dtype)
    return o_c


def sparse_prefill_attn(
    indexer,
    q,
    k,
    v,
    softmax_scale=None,
    chunk_size=None,
    dense_below=None,
):
    """稀疏 chunked prefill attention 主入口（patch 层调用）。

    q: [1, L, H, D]；k/v: [1, S, Hkv, D]（S == L，B=1 无 padding）。
    返回 o: [1, L, H, D]。

    注意：不改 tli_indexer.py 的单行 query 断言（保护在跑语义）——本函数
    每 chunk 只喂末行单行 q（reshape 成 [1,1,H,D]）。
    """
    L, H, D = q.shape[1], q.shape[2], q.shape[3]
    S = k.shape[1]
    assert q.shape[0] == 1 and k.shape[0] == 1, "质量路径假设 B=1"
    assert L == S, "质量路径假设 prefill 时 q/k 等长（无 prefix 续写）"
    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if chunk_size is None:
        chunk_size = int(
            os.environ.get("TLI_SPARSE_PREFILL_CHUNK", SPARSE_PREFILL_CHUNK)
        )
    if dense_below is None:
        dense_below = getattr(indexer.args, "tia_level2_topk", 1024)

    G = H // k.shape[2]
    # 行级保护带尺寸（与 indexer mask 强制区同源；getattr 兜底对齐默认配置）
    swa_tok = int(getattr(indexer, "sliding_window_size", 128))
    _bs = int(getattr(indexer.args, "tia_block_size", 64))
    sink_tok = int(getattr(indexer, "sink_blocks", 2)) * _bs
    # 共享选择满额理论上界（host 静态算，零 GPU 同步）：mid topk（K2，
    # 块对齐 +1 块松弛）+ sink 强制 + 末行 swa 强制 + 1 块对齐余量。
    # W 用该上界多出的 pad 列由 valid mask 屏蔽，数值不变
    w_sel_cap = (
        int(getattr(indexer.args, "tia_level2_topk", 1024)) + _bs
        + sink_tok + swa_tok + _bs
    )
    o = torch.empty_like(q)
    # 【E138c chunk 级双流预取（2026-10-14 用户「全面性能优化」指令）】
    # chunk c+1 的选择（prepare_mask 只依赖 K/V 前缀与末行 q，不依赖
    # chunk c 的 attention 输出——prefill 开始时全序列 K/V 已物化）在
    # side stream 上计算，与主 stream 的 chunk c attention 重叠，隐藏
    # 选择时间（选择链 mavg 生产口径无 host 同步，全部 launch 异步）。
    # 数值不变性：选择 kernel 输入相同 + side stream FIFO 保 indexer
    # prepare 调用序（E127 inc 缓存 append-only 语义不破——所有 prep
    # 调用串行在单一 side stream 上）+ 主 stream 经 event 依赖后才消费
    # sel + sel 跨流 record_stream 标记（caching allocator 在主 stream
    # kernel 完成前不得重用其存储）。层间安全：每层入口 side.wait_stream
    # (主 stream) 使本层 prep 排在上一层的 observe/calibrate 之后；尾
    # chunk 消费点 ev.wait() 保证层返回时 side 无在飞工作。
    # 逃生口 TLI_PREFILL_PREFETCH=0（默认开）。
    bounds = [(c0, min(c0 + chunk_size, L)) for c0 in range(0, L, chunk_size)]
    n_sparse = sum(1 for c0, _ in bounds if c0 >= dense_below)
    # 门槛 n_sparse >= 4（.251 H20 长度扫描实测：n_sparse=2 净负 0.74x——
    # 流交错/SM 争用开销 > 重叠收益；=3 持平 1.02x；>=5 稳定 +5~9%，
    # =15 时 1.07x。短序列走逐 chunk 串行路径零扰动）
    use_pf = (
        n_sparse >= 4
        and q.is_cuda
        and os.environ.get("TLI_PREFILL_PREFETCH", "1") == "1"
    )
    side = None
    if use_pf:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())

    def _prepare_sel(c0, c1):
        """末行 q 单行选择（indexer 单行 query 设计，规避多行 assert）。
        因果可见前缀切片：末行不可见 [c1, L) 的未来 token，切片保证选择
        严格因果（末行视角 mask 的 swa 强制区同样是块末视角最近 128）。"""
        last_q = q[:, c1 - 1 : c1]  # [1, 1, H, D]
        cu = torch.tensor([0, c1], dtype=torch.int32, device=k.device)
        q_ids = cu[1:] - 1  # 末行位置（= prepare_seqlens(cu) - 1 口径）
        mask, _bs = indexer.prepare_mask(
            last_q, q_ids, k[:, :c1], cu, softmax_scale
        )
        m0 = mask[0, 0]        # [h, T_pad]（h=Hkv 共享 或 H per_q_head）
        return m0[..., :c1]    # 截到真实 token 数（pad 块尾巴去除）

    pending = None  # (c0, sel, event)
    for i, (c0, c1) in enumerate(bounds):
        if c0 < dense_below:
            # 首 chunk / 短行：dense（identity 选择，对齐 B02/S1 短行语义；
            # 保护带 ⊆ 因果恒 no-op，不传保持 dense 路径逐位不变）
            sel = None
            prot_args = dict(sink_tok=0, swa_tok=0)
        else:
            if pending is not None and pending[0] == c0:
                # 消费预取：主 stream 等 side 完成后读 sel；record_stream
                # 防 allocator 跨流重用竞争
                _, sel, ev = pending
                ev.wait()
                sel.record_stream(torch.cuda.current_stream())
                pending = None
            else:
                sel = _prepare_sel(c0, c1)
            # 稀疏 chunk：行级保护带生效（TL-PREFILL-SWA-002）+ 满额
            # 上界传参（消 w_sel_max/W 两处 .item() host-GPU 同步）
            prot_args = dict(sink_tok=sink_tok, swa_tok=swa_tok)
            # 发起下一个稀疏 chunk 的选择预取（side stream FIFO：prep
            # 调用保持 chunk 顺序）
            if use_pf and pending is None:
                for j in range(i + 1, len(bounds)):
                    nc0, nc1 = bounds[j]
                    if nc0 >= dense_below:
                        with torch.cuda.stream(side):
                            sel_next = _prepare_sel(nc0, nc1)
                            ev_next = torch.cuda.Event()
                            ev_next.record(side)
                        pending = (nc0, sel_next, ev_next)
                        break
        o[:, c0:c1] = _chunk_rows_attn(
            q[:, c0:c1], k[:, :c1], v[:, :c1], sel, c0, G, softmax_scale,
            w_sel_cap=(0 if sel is None else w_sel_cap), **prot_args
        )
    return o
