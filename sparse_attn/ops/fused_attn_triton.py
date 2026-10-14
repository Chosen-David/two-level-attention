# -*- coding: utf-8 -*-
"""【E139 纲领④】稀疏 prefill 融合紧凑 attention Triton kernel。

用户纲领④原话：「融合紧凑 attention。消除 2048 行跨组候选并集、重复搬运、
.item() 同步及完整 score/P 中间量。kernel 按每行绝对 token 位置处理
causal、sink、SWA；普通 varlen FlashAttention 不能直接替换这套不规则掩码」。

设计（对齐 eager_prefill._chunk_rows_attn 组版语义，L62-271）：
  1. **token 列表直取，不走并集 argsort**：每行（绝对位置 p = pos_lo + r）
     的有效列谓词 = (j <= p) AND (j < sink_tok OR j >= p-swa_tok+1 OR
     sel[j(r)//grp][h, j])——列遍历按 K 原始位置序天然因果，无需
     argsort/compact/并集物化（原实现对全 T 列 argsort，L=32K 每行块一次）。
  2. **flash 风格在线 softmax**：行内分块累积 m_i/l_i/acc，不物化完整
     score/P；q tile [BR*G, D]（GQA 折叠：行维 = 行×组内 q-head，K/V 每
     program 只载一次，G 个 q-head 共享）× K tile [D, BK] → tl.dot
     （bf16 输入 fp32 累加——bf16×bf16 乘积在 fp32 中精确，累加序与
     参考 fp32 einsum 仅树形归约分组差，口径同 E138b/SDPA-PoC 判决）。
  3. **块级跳空**（主要加速来源，每行有效 ~1.2K/32K 列）：每 K tile 先读
     组选择块位图 sel_blk[j, h, b]（64-token 块 any 聚合，host 一次
     预计算）；若 tile 内无任何 sel 块且不与 sink 区 / 行块保护带窗口
     [p_lo-swa+1, p_hi] 相交 → 整 tile 跳过（不载 K/V、不算 dot）。
     跳空判定保守正确：bitmap any==False ⇒ 块内 sel 全 False；sink/swa
     窗口相交条件是「存在有效列」的必要条件补集——漏跳只损失性能，
     误跳不可能（窗口外且 sel 全 False ⇒ 谓词全 False）。
  4. **空行兜底对齐原版**：l_i==0（该行无任何有效列）→ 输出 = v[p]
     （原版 diag OR 单列 softmax=1·v 语义的精确等价，L249-255）。
  5. **数值口径**：score bf16 dot fp32 累加（参考为 fp32 einsum，bf16
     输入乘积精确、仅累加树差 → ULP 级）；p_exp 转 bf16 后与 v dot
     （对齐原版 softmax→bf16→einsum 的 cast 序，L264-270）；对拍硬门
     allclose(rtol=1e-2, atol=1e-3) + maxdiff 报告（E138b 口径）。

fail-closed：仅组版（sel_list 非 None）+ kv 共享（非 per_q_head）+ CUDA
+ G 为 2 的幂（tl.arange 约束）启用；异常由调用方捕获回退原 gather 路径。
"""
import torch
import triton
import triton.language as tl

# 块粒度（= tia_block_size 生产值；位图聚合粒度）
_BLK = 64


@triton.jit
def _fused_sparse_prefill_attn(
    Q, K, V, SEL, SELBLK, O,
    len_r, T, pos_lo, grp, n_g, sink_tok, swa_tok, scale,
    H, Hkv, G, Tb, Tp,
    BRG: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr,
):
    """单 chunk 组选择融合 attention。

    Grid: (cdiv(len_r, BR), Hkv)；每 program 处理 BR 行 × G 个 q-head
    （折叠成 BRG 行 tile）× 1 个 kv head，扫全 T 列（块级跳空）。
    布局（均为连续张量，wrapper 侧断言）：
      Q/O: [len_r, H, D]；K/V: [T, Hkv, D]；SEL: [n_g, Hkv, Tp]（bool，
      右 pad 到 64 倍数，pad 列 False）；SELBLK: [n_g, Hkv, Tb]（bool，
      64-token 块 any 聚合）。
    行 r 局部（0 起），组号 j = r // grp（sel_list 为本 chunk 组列表，
    j 直接索引）；绝对位置 p = pos_lo + r。
    """
    rt = tl.program_id(0)
    h = tl.program_id(1)
    BR = BRG // G                       # 每 program 行数（G 折叠进行维）
    rg = tl.arange(0, BRG)              # [BRG] = r*G + g 展平
    r = rg // G
    g = rg % G
    rows = rt * BR + r                  # [BRG] 局部行号
    row_m = rows < len_r
    j = tl.minimum(rows // grp, n_g - 1)  # 组号（masked 行 clamp，仅掩码加载用）
    p = pos_lo + rows                   # [BRG] 绝对位置

    # q tile [BRG, D]：q-head = h*G + g（q_c 行主序 (h g) 布局）
    d = tl.arange(0, D)
    q_ptrs = Q + rows[:, None] * (H * D) + (h * G + g)[:, None] * D \
        + d[None, :]
    q = tl.load(q_ptrs, mask=row_m[:, None], other=0.0)
    # scale 施加口径对齐原版 L210-212：bf16 标量乘（fp32 中间再舍回 bf16）
    qs = (q * scale).to(q.dtype)

    m_i = tl.full([BRG], float("-inf"), tl.float32)
    l_i = tl.zeros([BRG], tl.float32)
    acc = tl.zeros([BRG, D], tl.float32)

    # 跳空窗口标量（行块内真实行的绝对位置界——masked 行不参与）
    n_rows_tile = tl.minimum(BR, len_r - rt * BR)
    p_lo = pos_lo + rt * BR
    p_hi = pos_lo + rt * BR + n_rows_tile - 1

    NB: tl.constexpr = BK // BLK        # 每 K tile 覆盖的位图块数
    b = tl.arange(0, NB)
    for kb in range(0, T, BK):
        # ---- 块级跳空判定（保守正确，见模块 docstring §3）----
        bm_ptrs = SELBLK + j[:, None] * (Hkv * Tb) + h * Tb \
            + (kb // BLK + b)[None, :]
        bm = tl.load(bm_ptrs, mask=row_m[:, None], other=0)
        sel_any = tl.max(tl.max(bm.to(tl.int32), axis=1), axis=0)
        # sink 区 [0, sink_tok) 与 tile 相交（fused 路径 pos_lo >= sink_tok
        # 恒成立，此处仍保守含 kb <= p_hi 因果界）
        sink_hit = (kb < sink_tok) & (kb <= p_hi)
        # 保护带窗口 [p_lo-swa+1, p_hi]（行间窗口连续拼接，原版 L164-166
        # 并集语义）与 tile [kb, kb+BK) 相交
        win_hit = (kb <= p_hi) & (kb + BK > p_lo - swa_tok + 1)
        need = (sel_any != 0) | sink_hit | win_hit
        if need:
            cols = kb + tl.arange(0, BK)            # [BK] 绝对列位置
            col_m = cols < T
            # 行级组选择（组内 partition 语义核心，原版 L189-201）：
            # sel[r, c] = sel_stack[r//grp][h, c]（pad 列 False）
            sel_ptrs = SEL + j[:, None] * (Hkv * Tp) + h * Tp + cols[None, :]
            sel = tl.load(sel_ptrs,
                          mask=row_m[:, None] & col_m[None, :], other=0)
            # K tile [D, BK]（转置布局直接喂 tl.dot）
            k_ptrs = K + cols[None, :] * (Hkv * D) + h * D + d[:, None]
            kt = tl.load(k_ptrs, mask=col_m[None, :], other=0.0)
            # score：bf16 dot fp32 累加（bf16×bf16 乘积 fp32 精确）
            s = tl.dot(qs, kt)                      # [BRG, BK] fp32
            # 行级可见谓词（= 原版 row_ok，L239-247）：
            # j <= p 且 (sel | j < sink | j >= p-swa+1)；col_m 屏蔽
            # K 尾部（T..Tp pad 列——causal p<=T-1 已截，此处双保险）
            valid = col_m[None, :] \
                & (cols[None, :] <= p[:, None]) \
                & (sel | (cols[None, :] < sink_tok)
                   | (cols[None, :] >= (p[:, None] - swa_tok + 1)))
            s = tl.where(valid, s, float("-inf"))
            # ---- 在线 softmax（flash 两遍式；-inf 列 exp=0 不进分母，
            # 与原版 masked_fill_(-inf)+softmax 数学等价）----
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            # m 全 -inf 行（本 tile 无有效列）：m_safe 兜底防 -inf-(-inf)=nan
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)            # m_i=-inf → 0（幂零）
            p_exp = tl.exp(s - m_safe[:, None])     # -inf 列 → 0
            l_i = l_i * alpha + tl.sum(p_exp, axis=1)
            # V tile [BK, D]；p_exp 转 bf16 后 dot（cast 序对齐原版
            # L264-270：softmax fp32 → 模型 dtype → 与 v 相乘）
            v_ptrs = V + cols[:, None] * (Hkv * D) + h * D + d[None, :]
            vt = tl.load(v_ptrs, mask=col_m[:, None], other=0.0)
            acc = acc * alpha[:, None] \
                + tl.dot(p_exp.to(vt.dtype), vt)   # [BRG, D] fp32
            m_i = m_new

    # ---- 收尾：归一化 + 空行兜底（原版 L249-255 diag OR 单列语义：空行
    # 输出 = v[p]，softmax=1 精确等价）----
    l_safe = tl.where(l_i == 0, 1.0, l_i)
    out = acc / l_safe[:, None]
    v_d_ptrs = V + p[:, None] * (Hkv * D) + h * D + d[None, :]
    v_d = tl.load(v_d_ptrs,
                  mask=row_m[:, None] & (p[:, None] < T), other=0.0)
    out = tl.where((l_i == 0)[:, None], v_d.to(tl.float32), out)
    o_ptrs = O + rows[:, None] * (H * D) + (h * G + g)[:, None] * D \
        + d[None, :]
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_m[:, None])


# tile 参数（扫参写死：BRG∈{32,64,128}×BK∈{64,128}×warps∈{4,8}×stages∈{2,3}
# 共 11 组，L=32K grp64 组合态最优 234.4ms；次优 (64,128,4,2) 245.0ms。
# BK=64=单位图块粒度跳空最细 + w=4 acc [64,128] fp32 寄存器压力适中）
_BRG = 64      # 行 tile = BR 行 × G（GQA 折叠），BRG=64/D=128 时 acc 32KB
_BK = 64       # K 列 tile（= 1 个 64-token 位图块，跳空粒度最细）
_NUM_WARPS = 4
_NUM_STAGES = 2


def fused_group_attn(q_c, k_c, v_c, sel_list, pos_lo, G, softmax_scale,
                     sink_tok=0, swa_tok=0, grp=0):
    """组选择融合 attention wrapper（_chunk_rows_attn 组版分流入口）。

    返回 o_c [1, len_r, H, D]；不支持组合 raise（调用方 fail-closed 回退）。
    sel_list[j]: [Hkv, g1_j] bool（组末行前缀选择，g1_j ≤ T；与批量版
    batched_group_sels 输出同格式）。
    """
    len_r, H, D = q_c.shape[1], q_c.shape[2], q_c.shape[3]
    T, Hkv = k_c.shape[1], k_c.shape[2]
    n_g = len(sel_list)
    if not (q_c.is_cuda and k_c.is_cuda and v_c.is_cuda):
        raise ValueError("fused_group_attn 仅支持 CUDA")
    if sel_list[0].shape[0] != Hkv:
        raise ValueError("fused_group_attn 仅支持 kv 共享选择（非 per_q_head）")
    if G < 1 or (G & (G - 1)) != 0 or G > _BRG or H % Hkv != 0 or H // Hkv != G:
        raise ValueError("G 须为 2 的幂且 ≤ BRG（tl.arange 约束）")
    if grp < 1:
        raise ValueError("组版须 grp >= 1")
    if D not in (64, 128) or _BK % _BLK != 0 or _BRG % G != 0:
        raise ValueError("D/BK/BRG 形状不支持")
    if not (q_c.is_contiguous() and k_c.is_contiguous()
            and v_c.is_contiguous()):
        raise ValueError("q/k/v 须连续")

    # sel 统一 pad 到 Tp（64 倍数）：pad 列 False（> 组末行的未来 token，
    # causal 谓词已截断，双重不可见——与原版 pad 语义一致 L120-137）
    Tb = (T + _BLK - 1) // _BLK
    Tp = Tb * _BLK
    dev = q_c.device
    if Tp == T and all(s.shape[-1] == T for s in sel_list):
        sel_stack = torch.stack(sel_list) if n_g > 1 else sel_list[0].unsqueeze(0)
    else:
        sels = []
        for s in sel_list:
            if s.shape[-1] < Tp:
                pad = torch.zeros((*s.shape[:-1], Tp - s.shape[-1]),
                                  dtype=torch.bool, device=s.device)
                s = torch.cat([s, pad], dim=-1)
            else:
                s = s[..., :Tp]
            sels.append(s)
        sel_stack = torch.stack(sels)               # [n_g, Hkv, Tp]
    sel_stack = sel_stack.contiguous()
    # 块位图（any 聚合）：bitmap False ⇒ 块内 sel 全 False ⇒ 可跳空判定用。
    # 混合块（非生产口径，如 T7 monkeypatch 最小选集）由 need 分支内
    # 的逐列 sel 读取精确处理——any 聚合只用于跳空，不参与正确性判定
    sel_blk = sel_stack.view(n_g, Hkv, Tb, _BLK).any(dim=-1).contiguous()

    o_c = torch.empty_like(q_c)
    BR = _BRG // G
    grid = ((len_r + BR - 1) // BR, Hkv)
    _fused_sparse_prefill_attn[grid](
        q_c, k_c, v_c, sel_stack, sel_blk, o_c,
        len_r, T, pos_lo, grp, n_g, sink_tok, swa_tok, softmax_scale,
        H, Hkv, G, Tb, Tp,
        BRG=_BRG, BK=_BK, BLK=_BLK, D=D,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return o_c
