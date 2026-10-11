# -*- coding: utf-8 -*-
"""【E127 iter4+iter5】TLIIndexer 增量缓存子类（decode 热路径加速）。

锚点文件 sparse_attn/indexer/tli_indexer.py 零改动（md5 身份不变）；
本子类经 sparse_attn/patches/patch.py 在 TLI_E127_FAST=1 时替换实例化
（默认 0 走原类，在飞链零扰动）。

iter4：prepare_index 增量缓存
  原实现每次 decode step 对全序列 S 做 O(S) 重算（S=64K 每层步 2.89ms），
  而 decode 每步只追加 1 个 token。增量版：k_min/k_max/k_avg 复用块
  [0, fb) 缓存 + 尾段重算（pad 语义与全量版一致）；k_qat 预分配 buffer
  步进写。逐位一致为数学保证（块内独立归约 + quant 逐 token 独立）。

iter5：compute_score 物化缓存 + compute_mask 同步消除
  - compute_score：b_k_qat = repeat(k_qat, "t h d -> t (h g) d").to(fp32)
    每步物化 [S,H,128]（S=64K：bf16 repeat 536MB 写 + fp32 转换 1.07GB
    写/536MB 读），profile 实测占该阶段 78%（拷贝/转换 2.23ms vs einsum
    本体 0.58ms）。改为惰性增量缓存（_fast_b_k_qat）：尾段步进写 +
    容量翻倍摊销扩容，einsum 输入形状/布局/值逐位同 → score 逐位一致。
  - compute_mask：skip 层两处 n_valid 统计的 .item() host 同步（skip
    层 compute_mask CPU 3.0ms 的停顿主源），数学上恒为 host 可算值：
    site1（e64 分支）far 区间刚被整段置 -inf → 计数恒 0；site2（单池
    分支）sc_pool 除 far 区外全有限 → 计数 = kt − far 区块数。
  - 两个方法均从锚点源码文本派生（_derive_anchor_method）：除上述
    数学等价替换点外逐字节同锚点，锚点文本变更时 fail-closed 拒绝加载
    （杜绝手工拷贝 500+ 行方法的漂移风险）。
  - _prepare_index_inc 的 pad_cu 向量化（原 prepare_pad_mask 对 GPU
    张量逐元素 Python 迭代 = 隐式 .item() 同步）+ quant indices 缓存
    （原每步 H2D 构造）。

iter6：score_fine Triton GQA K 复用（compute_score 派生替换点②改写）
  - 原版每步物化 b_k_qat = repeat(k_qat)[S,H,128].to(fp32)（S=64K
    1.07GB fp32 写 + gemv 读 1.07GB = HBM 带宽瓶颈，实测 einsum 本体
    0.55ms 阶段主项）；iter5 曾用 _rep 增量缓存压尾段写但常驻 1.07GB/层。
  - 现改为 Triton kernel 直接读 k_qat [T,Hkv,D] bf16（134MB，GQA 组内
    寄存器复用 4× 读带宽削减），fp32 乘加累加 → 输出 RNE 舍入 bf16，
    microbench 2.63ms → 0.11ms；_rep 常驻缓存整体退役（省 1.07GB/层）。
  - 数值口径：fp32 累加顺序与 cublas gemv 不同 → fp32 ULP 级漂移，bf16
    舍入后实测 ~5e-5 比例元素差 1 ULP；mask/topk 选择多步多配置实测
    逐位不变（经验验证口径，非数学保证）；attention 输出 allclose(1e-2)
    任务容差内。投影/异形布局/Triton 编译失败 → 回退原版 repeat+einsum。

iter8：coarse 段 Triton GQA K 复用（compute_score 派生替换点③④）
  - coarse minmax 段：原版每步 repeat(k_min/k_max) 物化 [kt,H,d'] fp32
    ×2 + clamp(max/min=0) 双 gemv + copy 胶水（iter7 附带发现：该段
    gemv 本体仅 ~32μs，repeat 物化/胶水 ~120μs 主导，~10 launches）。
    现 _score_coarse_kernel 单 kernel：每 program 载 k_min/k_max 块
    [BT,d']，循环 g 个 q 头做 min(q,0)·k_min + max(q,0)·k_max 双点积
    （GQA 组内 K 寄存器复用），块级 kt=S/64（64K 档仅 1024）。
  - score_coarse_avg 段：einsum(b_q, repeat(k_avg)) 与 _score_fine_kernel
    同构 → 复用 iter6 kernel。
  - 数值：fp32 累加顺序与 cublas 不同 → bf16 舍入后 ULP 漂移（与
    iter6 fine 同口径）；coarse 分数决定块级 topk → mask 选择为经验
    验证口径（门禁 4 配置 × normal/skip × 10 步实测）。

契约（fail-closed，全部回退原路径）：
  - k append-only（HF DynamicCache decode 语义）。S_now ≤ S_prev →
    全量重建缓存（含 _rep）。
  - clear() / _prepare_index_full 使 _inc、_rep 失效 → 新请求首步全量。
  - _fast_b_k_qat 仅当 index_dict 的 k_qat 与增量缓存同一底层存储
    （data_ptr host 判同，零同步）才复用前缀；投影基/moba/外部合成
    k_qat 一律全量重建（值同原版路径，行为逐位不变）。
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

import triton
import triton.language as tl
from triton.language.extra import libdevice   # 【iter9】精确 expf（__nv_expf）

from .tli_indexer import TLIIndexer


class TLIIndexerFast(TLIIndexer):

    def clear(self):
        super().clear()
        self._inc = None

    # ------------------------------------------------------------------ 入口
    def prepare_index(self, k, cu_seqlens_k):
        inc = getattr(self, "_inc", None)
        S_now = k.shape[1]
        if (
            inc is not None
            and S_now > inc["S"]
            and self._basis is None
            and self._basis_all is None
            and not self.moba_gate
            and k.shape[-1] == 128
            and inc["sub_key"] == self._sub_key()
            and (not self._need_avg_score() or inc["k_avg"] is not None)
        ):
            return self._prepare_index_inc(k, cu_seqlens_k, inc)
        return self._prepare_index_full(k, cu_seqlens_k)

    def _sub_key(self):
        """子空间一致性 key：跨步 idx_sub 不变才可增量（static_pair 的
        _pair_idx 在 prefill 后冻结；rope/nope/tail 为常量分支）。"""
        if not self.enable_subspace:
            return ("off", 0)
        if self.static_pair and self._pair_idx is not None:
            return ("pair", int(self._pair_idx.numel()))
        return (self.subspace, self.cmp_ratio)

    # ---------------------------------------------------------------- 全量
    def _prepare_index_full(self, k, cu_seqlens_k):
        d = super().prepare_index(k, cu_seqlens_k)
        # 建缓存（k_qat 仅非投影路径可作增量 buffer——投影路径 k_qat 为
        # 全维量化特征，散布语义不同；_basis 非 None 时 can_inc 已挡，
        # 此处再防御一次）
        k_qat_buf = None
        if self._basis is None and not self.moba_gate and k.shape[-1] == 128:
            k_qat_buf = d["k_qat"].squeeze(0).clone() if d["k_qat"].dim() == 4 \
                else None
        self._inc = {
            "S": k.shape[1],
            "k_min": d["k_min"],
            "k_max": d["k_max"],
            "k_avg": d.get("k_avg"),
            "k_qat_buf": k_qat_buf,
            "k_qat_cap": k.shape[1] if k_qat_buf is not None else 0,
            "sub_key": self._sub_key(),
        }
        return d

    # ---------------------------------------------------------------- 增量
    def _prepare_index_inc(self, k, cu_seqlens_k, inc):
        self._resolve_skip()   # 与原版头部同构（幂等；影响 _maybe_build_kmeans 条件）
        bs = self.args.tia_block_size
        S_prev, S_now = inc["S"], k.shape[1]
        fb = (S_prev - 1) // bs        # 第一个可能变更的块（含新 token 的块）
        lo = fb * bs                   # 复用前缀 = token [0, lo)（块 [0, fb)）
        idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None

        # ---- 尾段 [lo, S_now) 重算（pad 语义与全量版一致：块对齐、pad 0）----
        # 【iter5】pad_cu 向量化：原 prepare_pad_mask 对 seqlens 逐元素
        # Python 迭代（GPU 张量迭代 = 隐式 .item() host 同步，每步一次）；
        # 整数算术逐位等价：pad = (x+bs-1)//bs*bs，cumsum 前接 0。
        _seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        _pad_seqlens_k = (_seqlens_k + bs - 1) // bs * bs
        pad_cu_seqlens_k = torch.cat(
            (_pad_seqlens_k.new_zeros((1,)), _pad_seqlens_k.cumsum(dim=0)))
        tail_k = k[:, lo:S_now]                              # [1, n, Hkv, D]
        # 尾段块数与 pad：与全量版一致 = ceil(S_now/bs)*bs 覆盖下的尾块集
        # （lo 为块边界，前缀块已对齐——尾段 pad 量 = 全量 pad 量）
        pad_total = (bs - S_now % bs) % bs
        tail_pad = tail_k.new_zeros((1, S_now - lo + pad_total) + tail_k.shape[2:])
        tail_pad[:, : S_now - lo] = tail_k
        t_coarse = rearrange(
            tail_pad[..., idx_sub] if idx_sub is not None else tail_pad,
            "b (t bs) h d -> b t bs h d", bs=bs,
        )
        k_min = torch.cat([inc["k_min"][:, :fb], t_coarse.amin(dim=2)], dim=1)
        k_max = torch.cat([inc["k_max"][:, :fb], t_coarse.amax(dim=2)], dim=1)
        k_avg = None
        if self._need_avg_score():
            k_avg = torch.cat([inc["k_avg"][:, :fb], t_coarse.mean(dim=2)], dim=1)

        # ---- k_qat 增量：预分配 buffer 步进写（尾段 zero_ + scatter， ----
        # 与全量版 zeros_like+scatter 逐位同构）
        indices = idx_sub if idx_sub is not None else self._qat_indices(k.device)
        buf, cap = inc["k_qat_buf"], inc["k_qat_cap"]
        if S_now > cap:   # 扩容（翻倍摊销）
            new_cap = max(2 * cap, S_now)
            new_buf = buf.new_zeros((new_cap,) + buf.shape[1:])
            new_buf[:S_prev] = buf[:S_prev]
            inc["k_qat_buf"], inc["k_qat_cap"] = new_buf, new_cap
            buf = new_buf
        buf[lo:S_now].zero_()
        buf[lo:S_now, :, indices] = self.min_max_per_token_quant(
            k[0, lo:S_now, :, indices])
        k_qat = buf[:S_now].unsqueeze(0)                     # [1, S, Hkv, D] 视图

        index_dict = {
            "k_min": k_min,
            "k_max": k_max,
            "k_avg": k_avg,
            "k_avg_full": None,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": pad_cu_seqlens_k // bs,
            "cu_seqlens_k_fine": cu_seqlens_k,
            "valid_length": k.shape[1],
        }
        if (self.enable_kmeans and not self.skip_far
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            self._maybe_build_kmeans(k, index_dict)

        # 缓存滚动（cat 产物作为新缓存；k_qat buffer 原地复用）
        inc["S"] = S_now
        inc["k_min"], inc["k_max"], inc["k_avg"] = k_min, k_max, k_avg
        return index_dict

    def _qat_indices(self, device):
        """【iter5】非子空间 quant indices 缓存（原每步 list→tensor 的
        pageable H2D 构造；cmp_ratio 为实例级常量 → 命中后零构造零拷贝）。
        GPU 端 arange 构造（纯 kernel 零 H2D，int64 与原 torch.tensor(list)
        同 dtype，值逐位同）。"""
        key = (self.cmp_ratio, device)
        cache = getattr(self, "_qat_idx_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        delta = 64 // self.cmp_ratio
        idx = torch.cat([torch.arange(64 - delta, 64, device=device),
                         torch.arange(128 - delta, 128, device=device)])
        self._qat_idx_cache = (key, idx)
        return idx

    # ------------------------------------------------ 【iter6】Triton score_fine
    def _fast_score_fine(self, kq, b_q_full, out_dtype, want_g=False):
        """score_fine 的 Triton GQA K 复用 kernel（非投影路径专用）。

        数学：score[h, t] = sum_d b_q_full[h, d] * k_qat32[t, h//g, d]，
        与原版 repeat(k_qat,"t h d -> t (h g) d") 后 einsum("h d, kt h d
        -> h kt") 的数学等价；GQA 组内 K 在寄存器复用（每块 K 只读一次
        服务 g 个 q 头），读带宽 [T,H,D] → [T,Hkv,D]（4× 削减），且无需
        物化 fp32 repeat 缓存（iter5 _rep 常驻 [S,H,128] fp32 =
        1.07GB/层 @S=64K 整体退役）。

        数值：kernel 内 bf16→fp32 精确转换 + fp32 乘加累加 + 输出
        RNE 舍入到 out_dtype，与原版 fp32 einsum → .to(bf16) 相比仅
        累加顺序不同（tl.sum D 维 tree-reduce vs cublas gemv 调度）→
        fp32 结果 ULP 级漂移，bf16 舍入后实测差异 ~5e-5 比例的元素
        差 1 ULP；mask/topk 选择实测逐位不变（门禁实测口径，非数学
        保证）。注意原版输出为 bf16（q_sq.dtype，非投影路径），fp32
        ULP 漂移绝大部分被 bf16 舍入吸收。

        fail-closed：Triton 编译/执行异常 → 置 _triton_ok=False 并
        回退原版 repeat+einsum 表达式（值同原版路径，仅性能回退）。
        """
        try:
            T, Hkv, D = kq.shape
            H = b_q_full.shape[0]
            g = H // Hkv
            out = torch.empty((H, T), device=kq.device, dtype=out_dtype)
            if want_g:
                outg = torch.empty((Hkv, T), device=kq.device, dtype=out_dtype)
            else:
                outg = out   # HAS_G=False 占位（编译期裁剪不读）
            grid = (triton.cdiv(T, 64), Hkv)
            _score_fine_kernel[grid](
                kq, b_q_full, out, outg, T,
                kq.stride(0), kq.stride(1), b_q_full.stride(0),
                out.stride(0), out.stride(1),
                outg.stride(0), outg.stride(1),
                Hkv=Hkv, G=g, D=D, BT=64, HAS_G=want_g, num_warps=4)
            return (out, outg) if want_g else out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_score_coarse(self, k_min, k_max, b_q, out_dtype, want_g=False):
        """【iter8】score_coarse 的 Triton GQA K 复用 kernel（非投影路径专用）。

        数学：score_coarse[h, t] = Σ_d min(b_q[h,d],0)·k_min32[t,h//g,d]
                                   + Σ_d max(b_q[h,d],0)·k_max32[t,h//g,d]，
        与原版 repeat(k_min/k_max) 后 clamp(max=0)/clamp(min=0) 双 einsum
        数学等价（minmax 上界分：负部配 k_min、正部配 k_max）；GQA 组内
        k_min/k_max 块寄存器复用（读带宽 /g），免 [kt,H,d'] fp32 repeat
        物化 + clamp 胶水（~10 launches → 1 kernel）。

        数值：fp32 累加顺序与 cublas 双 gemv 不同 → bf16 舍入后 ULP 级
        漂移（实测 ~5e-5 比例元素差 1 ULP，与 iter6 fine 同量级）；
        mask/topk 选择为经验验证口径（非数学保证）。

        fail-closed：异常置 _triton_ok=False 并回退原版表达式。"""
        try:
            T, Hkv, D = k_min.shape
            H = b_q.shape[0]
            g = H // Hkv
            out = torch.empty((H, T), device=k_min.device, dtype=out_dtype)
            if want_g:
                outg = torch.empty((Hkv, T), device=k_min.device, dtype=out_dtype)
            else:
                outg = out   # HAS_G=False 占位（编译期裁剪不读）
            grid = (triton.cdiv(T, 64), Hkv)
            _score_coarse_kernel[grid](
                k_min, k_max, b_q, out, outg, T,
                k_min.stride(0), k_min.stride(1),
                k_max.stride(0), k_max.stride(1),
                b_q.stride(0),
                out.stride(0), out.stride(1),
                outg.stride(0), outg.stride(1),
                Hkv=Hkv, G=g, D=D, BT=64, HAS_G=want_g, num_warps=4)
            return (out, outg) if want_g else out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_mask_p(self, topk_mask, score_fine, bs, swa_lo, s_real):
        """【iter9】fused masked-softmax-gmean：compute_mask 共享口径的
        repeat + where + softmax + nan_to_num + mean + swa 置位
        6 op（3 次全宽 [H,T] 中间物化 + 6 launch）→ 2 kernel。

        数学语义（与原段逐位等价的构成）：
          - softmax(全 T 行、未选位 -inf) ≡ exp(s−m)/l（m/l 为全 T 归约），
            s = 块级 mask 展开后的 score_fine（连续 bs token 读同一块
            bool，L1 命中）；libdevice.exp = __nv_expf 与 PyTorch softmax
            内部 expf 同实现 → 分子逐位同。
          - l 的 fp32 分块归约顺序差异 → 相对 ~1e-6 量级，几乎全被
            bf16 舍入吸收（原型实测对齐 S 档 0 ULP / padding 档
            4/524K ≈ 7.6e-6 比例，下游 topk mask 逐位不变）。
          - mean(dim=-2) 口径：softmax 输出先舍入 P dtype 再 fp32 顺序
            累加 /G（= torch mean opmath 顺序累加），原型实测逐位一致。
          - 全 -inf 行：l=0 → p=0 ≡ softmax NaN → nan_to_num(0)。
          - swa 强制区 [swa_lo, S_real) 直接写 1.0 ≡ 原版 mean 后置位；
            padding 区 [S_real, T) score_fine 已 -inf → p=0 同原版。

        fail-closed：异常置 _triton_ok=False 并向上抛（调用方回退原段）。
        NC ≤ 64（BT=1024 → T ≤ 64K，调用方路由已保证）。
        """
        try:
            blkm = topk_mask[0, 0]                     # [Hkv, kt] bool
            sf = score_fine[0, 0]                      # [H, T]
            H, T = sf.shape
            Hkv = blkm.shape[0]
            G = H // Hkv
            NC = triton.cdiv(T, 1024)
            ms = torch.empty((H, NC), device=sf.device, dtype=torch.float32)
            ls = torch.empty((H, NC), device=sf.device, dtype=torch.float32)
            p = torch.empty((Hkv, T), device=sf.device, dtype=sf.dtype)
            _mask_p_stat_kernel[(H, NC)](
                sf, blkm, ms, ls, T, sf.stride(0), blkm.stride(0), NC,
                G=G, BS=bs, BT=1024, num_warps=4)
            _mask_p_out_kernel[(Hkv, NC)](
                sf, blkm, ms, ls, p, T, sf.stride(0), blkm.stride(0),
                p.stride(0), NC, swa_lo, s_real, G=G, BS=bs, BT=1024, num_warps=4)
            return p[None, None]
        except Exception:
            self._triton_ok = False
            raise


@triton.jit
def _score_fine_kernel(
    KQ, Q, OUT, OUTG,   # KQ [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T]；OUTG [Hkv,T]
    T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
    stride_gh, stride_gt,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    HAS_G: tl.constexpr,
):
    """GQA K 复用 score_fine：每 program 处理 (token 块 × 1 个 kv 头)，
    K 块 [BT, D] 载入寄存器后循环 g 个 q 头逐头点积，K 读带宽 /g。
    【iter9】HAS_G=True 时双输出 OUTG[h_kv,t] = mean_g s_j（s_j 先舍入
    OUT dtype 再 fp32 顺序累加——与原版「softmax/mean 前先 bf16 舍入」
    的 torch mean(dim=-2) 顺序累加口径同构，原型实测逐位一致）。"""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask_t = offs_t < T
    k = tl.load(KQ + offs_t[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :],
                mask=mask_t[:, None], other=0.0).to(tl.float32)
    gm = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(G):
        h = pid_h * G + j
        q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已乘 scale）
        s = tl.sum(k * q[None, :], axis=1)            # [BT] fp32
        s_out = s.to(OUT.dtype.element_ty)
        tl.store(OUT + h * stride_oh + offs_t * stride_ot, s_out, mask=mask_t)
        if HAS_G:
            gm += s_out.to(tl.float32)
    if HAS_G:
        tl.store(OUTG + pid_h * stride_gh + offs_t * stride_gt,
                 (gm / G).to(OUTG.dtype.element_ty), mask=mask_t)


@triton.jit
def _score_coarse_kernel(
    KMIN, KMAX, Q, OUT, OUTG,   # KMIN/KMAX [T,Hkv,D] bf16；Q [H,D]；OUT [H,T]；OUTG [Hkv,T]
    T, stride_kt, stride_kh, stride_kt2, stride_kh2,
    stride_qh, stride_oh, stride_ot, stride_gh, stride_gt,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    HAS_G: tl.constexpr,
):
    """【iter8】GQA K 复用 coarse minmax 上界分（min/max 两份 K 独立 stride）：
    s[h,t] = Σ_d min(q,0)·k_min + Σ_d max(q,0)·k_max（负部配 k_min、正部
    配 k_max，与原版 clamp 双 einsum 数学等价）。K 块 [BT,D] 寄存器复用
    服务 g 个 q 头（读带宽 /g，免 repeat 物化）。"""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask_t = offs_t < T
    kmin = tl.load(KMIN + offs_t[:, None] * stride_kt + pid_h * stride_kh
                   + offs_d[None, :],
                   mask=mask_t[:, None], other=0.0).to(tl.float32)   # [BT, D]
    kmax = tl.load(KMAX + offs_t[:, None] * stride_kt2 + pid_h * stride_kh2
                   + offs_d[None, :],
                   mask=mask_t[:, None], other=0.0).to(tl.float32)
    gm = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(G):
        h = pid_h * G + j
        q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已缩放）
        q_low = tl.minimum(q, 0.0)                   # clamp(max=0)：负部 → k_min
        q_high = tl.maximum(q, 0.0)                  # clamp(min=0)：正部 → k_max
        s = tl.sum(kmin * q_low[None, :], axis=1) + tl.sum(kmax * q_high[None, :], axis=1)
        s_out = s.to(OUT.dtype.element_ty)
        tl.store(OUT + h * stride_oh + offs_t * stride_ot, s_out, mask=mask_t)
        if HAS_G:   # 【iter9】gmean 双输出（先舍入再 fp32 顺序平均，原型逐位一致）
            gm += s_out.to(tl.float32)
    if HAS_G:
        tl.store(OUTG + pid_h * stride_gh + offs_t * stride_gt,
                 (gm / G).to(OUTG.dtype.element_ty), mask=mask_t)


# ---- 【iter9】fused masked-softmax-gmean（compute_mask 共享口径前段）----
@triton.jit
def _mask_p_stat_kernel(
    SF, BLKM, MS, LS,   # SF [H,T]；BLKM [Hkv,kt] bool；MS/LS [H,NC] fp32
    T, stride_sf, stride_bm, NC,
    G: tl.constexpr, BS: tl.constexpr, BT: tl.constexpr,
):
    """(h_all, c)：单头单 chunk 的 masked s chunk-local (m_c, l_c)。
    s = 块级 mask 展开后的 score_fine（块外 -inf）；l_c 以 m_c 归一
    （flash 式，out kernel 跨 chunk 合并）；全 -inf chunk → (m_c,l_c)
    = (-inf, 0)（exp(-inf-0)=0 精确）。exp 用 libdevice.exp（__nv_expf，
    与 PyTorch softmax 内部 expf 同实现）。"""
    hg = tl.program_id(0)
    h = hg // G
    c = tl.program_id(1)
    offs_t = c * BT + tl.arange(0, BT)
    tm = offs_t < T
    # token → 块 bool gather：连续 BS 个 token 读同一字节，L1 完全命中
    bidx = (offs_t // BS).to(tl.int32)
    tokm = tl.load(BLKM + h * stride_bm + bidx, mask=tm, other=0).to(tl.int32)
    sf = tl.load(SF + hg * stride_sf + offs_t, mask=tm, other=0.0).to(tl.float32)
    s = tl.where((tokm != 0) & tm, sf, float('-inf'))
    mc = tl.max(s, axis=0)
    mc_safe = tl.where(mc == float('-inf'), 0.0, mc)
    lc = tl.sum(libdevice.exp(s - mc_safe), axis=0)
    tl.store(MS + hg * NC + c, mc)
    tl.store(LS + hg * NC + c, lc)


@triton.jit
def _mask_p_out_kernel(
    SF, BLKM, MS, LS, P,   # P [Hkv, T]
    T, stride_sf, stride_bm, stride_p,
    NC,
    SWA_LO, S_REAL,
    G: tl.constexpr, BS: tl.constexpr, BT: tl.constexpr,
):
    """(h_kv, c)：循环 g 个 q 头合并 [nC] chunk 统计 → 全局 (m_g, l_g)
    （M=max_c m_c；L=Σ_c l_c·exp(m_c−M)），重载 sf 块重算 masked s →
    p[h,t] = Σ_g exp(s_g−m_g)/l_g / G；swa 强制区直接 1.0。
    l_g=0（该头全行 -inf）⟹ 本 chunk s 全 -inf ⟹ 分子 exp(-inf)=0 →
    p=0，等价原版 softmax NaN → nan_to_num(0)（lg 守卫防 0 除）。
    mean 口径：pj 先舍入 P dtype 再 fp32 累加（= torch mean opmath 顺序）。"""
    h = tl.program_id(0)
    c = tl.program_id(1)
    offs_t = c * BT + tl.arange(0, BT)
    tm = offs_t < T
    bidx = (offs_t // BS).to(tl.int32)
    tokm = tl.load(BLKM + h * stride_bm + bidx, mask=tm, other=0).to(tl.int32)
    offs_c = tl.arange(0, 64)      # NC ≤ 64（BT=1024 → T ≤ 64K，路由保证）
    cm = offs_c < NC
    pacc = tl.zeros((BT,), tl.float32)
    for j in tl.static_range(G):
        ms = tl.load(MS + (h * G + j) * NC + offs_c, mask=cm, other=float('-inf'))
        ls = tl.load(LS + (h * G + j) * NC + offs_c, mask=cm, other=0.0)
        mg = tl.max(ms, axis=0)
        mg_safe = tl.where(mg == float('-inf'), 0.0, mg)
        lg = tl.sum(ls * libdevice.exp(ms - mg_safe), axis=0)
        sf = tl.load(SF + (h * G + j) * stride_sf + offs_t,
                     mask=tm, other=0.0).to(tl.float32)
        s = tl.where((tokm != 0) & tm, sf, float('-inf'))
        pj = libdevice.exp(s - mg_safe) / tl.where(lg > 0.0, lg, 1.0)
        pacc += pj.to(P.dtype.element_ty).to(tl.float32)
    p = pacc / G
    # swa 强制区 [swa_lo, S_real) 直接 1.0（≡ 原版 mean 后置位）；
    # padding 区 [S_real, T) 保持 0（score_fine 已被置 -inf → p=0）
    p = tl.where((offs_t >= SWA_LO) & (offs_t < S_REAL), 1.0, p)
    tl.store(P + h * stride_p + offs_t, p.to(P.dtype.element_ty), mask=tm)


# ======================================================================
# 【E127 iter5】锚点方法文本派生：compute_score / compute_mask
#
# 方法：inspect 读取锚点源码 → 对数学等价替换点做唯一文本替换 → exec
# 挂到 TLIIndexerFast。除替换点外与锚点逐字节相同（杜绝 500+ 行方法
# 手工拷贝的漂移风险）；锚点文本变更（替换点不再唯一匹配）时 fail-closed
# raise，拒绝在失配锚点上加载派生方法。
# ======================================================================
def _derive_anchor_method(name, replacements):
    import inspect
    import textwrap
    # 注意：先在原始缩进（getsource 返回类体缩进文本）上做唯一文本替换，
    # 再 dedent —— 替换串按锚点文件缩进书写，锚点文本变更即失配 fail-closed。
    src = inspect.getsource(getattr(TLIIndexer, name))
    for old, new in replacements:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(
                f"[TLIIndexerFast] 锚点方法 {name} 的替换点文本不再唯一匹配"
                f"（锚点 sparse_attn/indexer/tli_indexer.py 已变更？）："
                f"expected 1 occurrence of {old[:70]!r}..., got {n}。"
                f"E127 派生方法与锚点的逐位一致契约被破坏，拒绝加载。")
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    ns = {"torch": torch, "F": F, "rearrange": rearrange,
          "repeat": repeat, "einsum": einsum, "TLIIndexer": TLIIndexer}
    exec(compile(src, f"<tli_indexer_fast:derived-{name}>", "exec"), ns)
    return ns[name]


# ---- compute_score：四处替换 ----
# ① 零参 super() 在 exec 派生的独立函数中无 __class__ cell → 显式等价
#   改写（Fast 的 MRO 下 super().compute_score ≡ TLIIndexer.compute_score）；
# ② b_k_qat 物化 + 双分支 einsum（该两分支唯一消费点）→ 非投影
#   Triton GQA K 复用（_fast_score_fine；数学等价、bf16 舍入后 ULP 级
#   漂移、mask 门禁实测逐位不变）；投影/异形/编译失败回退原版表达式。
# ③【iter8】coarse minmax 段（b_k_min/b_k_max repeat 物化 + clamp 双
#   gemv + 胶水 ~10 launches）→ _fast_score_coarse 单 kernel（负部·k_min
#   + 正部·k_max 数学等价，GQA 组内 K 复用；ULP 口径同 ②）。
# ④【iter8】score_coarse_avg 段（repeat(k_avg) + gemv）→ 复用 ② 的
#   _score_fine_kernel 同构（einsum(b_q, k_avg) 数学等价；ULP 口径同 ②）。
_SCORE_REPL = [
    (
        "            d = super().compute_score(q, q_ids, index_dict, softmax_scale)",
        "            d = TLIIndexer.compute_score(self, q, q_ids, index_dict, softmax_scale)",
    ),
    (
        '        b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", '
        'g=self.group_size).to(torch.float32)\n'
        '        if self._basis is not None:\n'
        '            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt")\n'
        '        else:\n'
        '            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt").to(q_sq.dtype)',
        "        # 【E127 iter6】非投影路径 Triton GQA K 复用（K 读带宽 4× 削减，\n"
        "        # 消除 [T,H,128] fp32 repeat 物化与 iter5 _rep 常驻缓存；fp32\n"
        "        # 累加与原版 einsum 仅累加顺序不同 → bf16 舍入后 ULP 级漂移，\n"
        "        # mask 逐位门禁实测不变）。投影/异形布局/Triton 不可用回退原版。\n"
        "        if (self._basis is None and k_qat.dim() == 4 and k_qat.shape[0] == 1\n"
        "                and k_qat.shape[-1] == 128\n"
        "                and b_q_full.dtype == torch.float32\n"
        "                and k_qat.shape[2] * self.group_size == b_q_full.shape[0]\n"
        "                and getattr(self, \"_triton_ok\", True)):\n"
        "            try:\n"
        "                score_fine[0] = self._fast_score_fine(\n"
        "                    k_qat.squeeze(0), b_q_full, q_sq.dtype)\n"
        "            except Exception:\n"
        "                b_k_qat = repeat(k_qat.squeeze(0), \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "                score_fine[0] = einsum(b_q_full, b_k_qat, \"h d, kt h d -> h kt\").to(q_sq.dtype)\n"
        "        elif self._basis is not None:\n"
        "            b_k_qat = repeat(k_qat.squeeze(0), \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "            score_fine[0] = einsum(b_q_full, b_k_qat, \"h d, kt h d -> h kt\")\n"
        "        else:\n"
        "            b_k_qat = repeat(k_qat.squeeze(0), \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "            score_fine[0] = einsum(b_q_full, b_k_qat, \"h d, kt h d -> h kt\").to(q_sq.dtype)",
    ),
    # ---- 【iter8】coarse minmax 段 Triton GQA K 复用：b_k_min/b_k_max
    # 两次 [kt,H,d'] fp32 repeat 物化 + clamp 双 gemv + 胶水（~10
    # launches）→ 1 kernel。数学等价（负部·k_min + 正部·k_max）；fp32
    # 累加顺序差异 → bf16 舍入后 ULP 漂移（同 iter6 fine 口径）。
    # 投影/异形布局/Triton 异常回退原版表达式（fail-closed）。
    (
        '        b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)\n'
        '        b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)\n'
        '        b_score = (\n'
        '            einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")\n'
        '            + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")\n'
        '        )\n'
        '        score_coarse[0] = b_score.to(q_sq.dtype)',
        "        _sc_g = None\n"
        "        # 【E127 iter8】coarse minmax 段 Triton GQA K 复用（同 iter6\n"
        "        # fine 模式）：免 b_k_min/b_k_max 两次 [kt,H,d'] fp32 repeat\n"
        "        # 物化 + clamp 胶水（~10 launches → 1 kernel）；fp32 累加顺序\n"
        "        # 与 cublas 不同 → bf16 舍入后 ULP 级漂移（实测 ~5e-5 比例\n"
        "        # 差 1 ULP）。投影/异形布局/Triton 异常回退原版表达式。\n"
        "        # 【iter9】want_g=True 双输出 gmean（组内 mean 前移，与\n"
        "        # 现场 rearrange+mean 逐位一致，原型实测），供 compute_mask\n"
        "        # 消费（score_coarse_g 键；无键回退现场 mean）。\n"
        "        if (self._basis is None and k_min.dim() == 3 and k_max.dim() == 3\n"
        "                and (k_min.shape[-1] & (k_min.shape[-1] - 1)) == 0\n"
        "                and k_min.stride(-1) == 1 and k_max.stride(-1) == 1\n"
        "                and b_q.stride(-1) == 1 and b_q.dtype == torch.float32\n"
        "                and k_min.shape[-1] == b_q.shape[-1]\n"
        "                and k_min.shape[1] * self.group_size == b_q.shape[0]\n"
        "                and getattr(self, \"_triton_ok\", True)):\n"
        "            try:\n"
        "                # 【iter10】按档路由：gmean 双输出仅在 T ≤ 32768 启用\n"
        "                # （launch-gap 主导口径下小 S 档赚 / 64K 大 S 档 kernel\n"
        "                # 特化+clone 开销超收益，实测净回归 → 回退 iter8 行为）\n"
        "                if k_min.shape[0] * self.args.tia_block_size <= 32768:\n"
        "                    score_coarse[0], _sc_g = self._fast_score_coarse(\n"
        "                        k_min, k_max, b_q, q_sq.dtype, want_g=True)\n"
        "                else:\n"
        "                    score_coarse[0] = self._fast_score_coarse(\n"
        "                        k_min, k_max, b_q, q_sq.dtype)\n"
        "            except Exception:\n"
        "                b_k_min = repeat(k_min, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "                b_k_max = repeat(k_max, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "                b_score = (\n"
        "                    einsum(b_q.clamp(max=0), b_k_min, \"h d, kt h d -> h kt\")\n"
        "                    + einsum(b_q.clamp(min=0), b_k_max, \"h d, kt h d -> h kt\")\n"
        "                )\n"
        "                score_coarse[0] = b_score.to(q_sq.dtype)\n"
        "        else:\n"
        "            b_k_min = repeat(k_min, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "            b_k_max = repeat(k_max, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "            b_score = (\n"
        "                einsum(b_q.clamp(max=0), b_k_min, \"h d, kt h d -> h kt\")\n"
        "                + einsum(b_q.clamp(min=0), b_k_max, \"h d, kt h d -> h kt\")\n"
        "            )\n"
        "            score_coarse[0] = b_score.to(q_sq.dtype)",
    ),
    # ---- 【iter8】score_coarse_avg 段：einsum(b_q, repeat(k_avg)) 与
    # _score_fine_kernel 同构 → 复用 iter6 kernel（GQA K 复用免物化），
    # ULP 口径同 iter6；回退原版表达式（fail-closed）。
    (
        '        if k_avg is not None:\n'
        '            b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)\n'
        '            score_coarse_avg = q_sq.new_full(coarse_shape, float("-inf"))\n'
        '            score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)',
        "        _sc_avg_g = None\n"
        "        if k_avg is not None:\n"
        "            # 【E127 iter8】avg 段复用 iter6 _score_fine_kernel 同构\n"
        "            # （einsum(b_q, repeat(k_avg)) 数学等价，GQA K 复用免物\n"
        "            # 化）；ULP 口径同 iter6；异形/异常回退原版表达式。\n"
        "            # 【iter9】want_g=True 双输出 gmean（同 ③）。\n"
        "            if (self._basis is None and k_avg.dim() == 3\n"
        "                    and (k_avg.shape[-1] & (k_avg.shape[-1] - 1)) == 0\n"
        "                    and k_avg.stride(-1) == 1\n"
        "                    and b_q.dtype == torch.float32\n"
        "                    and k_avg.shape[-1] == b_q.shape[-1]\n"
        "                    and k_avg.shape[1] * self.group_size == b_q.shape[0]\n"
        "                    and getattr(self, \"_triton_ok\", True)):\n"
        "                try:\n"
        "                    score_coarse_avg = q_sq.new_full(coarse_shape, float(\"-inf\"))\n"
        "                    # 【iter10】按档路由（同 ③）：gmean 仅 T ≤ 32768\n"
        "                    if k_avg.shape[0] * self.args.tia_block_size <= 32768:\n"
        "                        score_coarse_avg[0], _sc_avg_g = self._fast_score_fine(\n"
        "                            k_avg, b_q, q_sq.dtype, want_g=True)\n"
        "                    else:\n"
        "                        score_coarse_avg[0] = self._fast_score_fine(\n"
        "                            k_avg, b_q, q_sq.dtype)\n"
        "                except Exception:\n"
        "                    b_k_avg = repeat(k_avg, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "                    score_coarse_avg = q_sq.new_full(coarse_shape, float(\"-inf\"))\n"
        "                    score_coarse_avg[0] = einsum(b_q, b_k_avg, \"h d, kt h d -> h kt\").to(q_sq.dtype)\n"
        "            else:\n"
        "                b_k_avg = repeat(k_avg, \"t h d -> t (h g) d\", g=self.group_size).to(torch.float32)\n"
        "                score_coarse_avg = q_sq.new_full(coarse_shape, float(\"-inf\"))\n"
        "                score_coarse_avg[0] = einsum(b_q, b_k_avg, \"h d, kt h d -> h kt\").to(q_sq.dtype)",
    ),
    # ---- 【iter9】⑤：返回 dict 透传 gmean 双输出（compute_mask 消费；
    # 无键 = 回退/父类捷径/投影路径，消费端 get 返回 None 走现场 mean）。
    (
        '        return {\n'
        '            "score_coarse": score_coarse.unsqueeze(0),\n'
        '            "score_coarse_avg": None if score_coarse_avg is None else score_coarse_avg.unsqueeze(0),\n'
        '            "score_moba": None if score_moba is None else score_moba.unsqueeze(0),\n'
        '            "score_fine": score_fine.unsqueeze(0),\n'
        '            # 075：真实序列长度随分数透传给 compute_mask（投影路径\n'
        '            # score_fine 宽 = Tpad ≠ 真实 S；无值时消费端回退宽度口径）\n'
        '            "valid_length": index_dict.get("valid_length"),\n'
        '        }',
        '        _e127_ret = {\n'
        '            "score_coarse": score_coarse.unsqueeze(0),\n'
        '            "score_coarse_avg": None if score_coarse_avg is None else score_coarse_avg.unsqueeze(0),\n'
        '            "score_moba": None if score_moba is None else score_moba.unsqueeze(0),\n'
        '            "score_fine": score_fine.unsqueeze(0),\n'
        '            # 075：真实序列长度随分数透传给 compute_mask（投影路径\n'
        '            # score_fine 宽 = Tpad ≠ 真实 S；无值时消费端回退宽度口径）\n'
        '            "valid_length": index_dict.get("valid_length"),\n'
        '        }\n'
        '        # 【E127 iter9】gmean 前移产物透传（kernel 双输出 [Hkv,kt]，\n'
        '        # 原型实测与现场 rearrange+mean 逐位一致）；无值不写键 →\n'
        '        # 消费端回退。两次 unsqueeze 补到 4 维 [1,1,Hkv,kt]（= 原版\n'
        '        # 消费端 rearrange+mean 输出形状，M-A/M-B 守卫 dim()==4）。\n'
        '        _sc_g_out = locals().get("_sc_g")\n'
        '        if _sc_g_out is not None:\n'
        '            _e127_ret["score_coarse_g"] = _sc_g_out.unsqueeze(0).unsqueeze(0)\n'
        '        _sc_avg_g_out = locals().get("_sc_avg_g")\n'
        '        if _sc_avg_g_out is not None:\n'
        '            _e127_ret["score_coarse_avg_g"] = _sc_avg_g_out.unsqueeze(0).unsqueeze(0)\n'
        '        return _e127_ret',
    ),
]

# ---- compute_mask：两处 .item() 同步消除（数学恒等替换）----
# site1（e64 分支 + skip_far）：far 区间 [far_lo_blk, far_hi_blk) 刚在
#   上方被 far_range 整段置 -inf → (> -inf).sum() 数学恒 0（区间空时
#   空张量计数亦为 0）→ host 常量 0 逐位等价。
# site2（单池分支 + skip_far）：sc_pool 除 far 区外全有限（score_coarse/
#   score_coarse_avg 均在 compute_score 中被全量覆写；父类捷径同构；
#   far 区 minmax/avg 两口径均已在上方整段置 -inf）→ 有效块数
#   = kt − far 区块数，host 可算（区间倒挂时两者同为 kt−0）。
_MASK_REPL = [
    (
        "                n_valid_far = int(\n"
        "                    (score_coarse[..., far_lo_blk:far_hi_blk] > float(\"-inf\")).sum(dim=-1).max().item()\n"
        "                )",
        "                # 【E127 iter5】far 区间刚被上方 far_range 整段置 -inf，\n"
        "                # 计数数学恒 0——host 常量替代 GPU 统计 + .item()\n"
        "                # host 同步（逐位等价；区间空时原计数亦为 0）\n"
        "                n_valid_far = 0",
    ),
    (
        "                n_valid = int((sc_pool > float(\"-inf\")).sum(dim=-1).max().item())",
        "                # 【E127 iter5】sc_pool 除 far 区（minmax/avg 两口径均在\n"
        "                # 上方整段置 -inf）外全有限（分数矩阵被全量覆写过）→\n"
        "                # 有效块数 = kt − far 区块数，host 可算免同步（逐位等价）\n"
        "                n_valid = kt - max(0, far_hi_blk - far_lo_blk)",
    ),
    (
        "            far_range = torch.arange(far_lo_blk, far_hi_blk, device=score_coarse.device)\n"
        "            score_coarse[..., far_range] = float(\"-inf\")",
        "            # 【E127 iter5】far_range 恒为连续区间 [far_lo_blk, far_hi_blk)——\n"
        "            # tensor 高级索引赋值（index_put_）触发 host 同步，连续\n"
        "            # 区间切片赋值数学恒等（区间空时两者同为空操作）→ 零同步改写\n"
        "            score_coarse[..., far_lo_blk:far_hi_blk] = float(\"-inf\")",
    ),
    (
        "                    sc_pool[..., torch.arange(far_lo_blk, far_hi_blk, device=sc_pool.device)] = float(\"-inf\")",
        "                    # 【E127 iter5】同上：连续区间切片赋值零同步等价改写\n"
        "                    sc_pool[..., far_lo_blk:far_hi_blk] = float(\"-inf\")",
    ),
    # ---- 【iter9】M-A：score_coarse group-mean 前移消费（kernel 双输出
    # 直供，原型实测与现场 rearrange+mean 逐位一致；无键/形状不符回退
    # 原式）。【clone 必要性】compute_mask 下方对 score_coarse 有原地写
    # （skip_far far 区 -inf / 末块置 +inf 排除）：原版 mean 产物是独立
    # 新张量，score_dict 键永不被 mask 阶段污染；直供版若绑引用则写
    # 透传进 dict 的 score_coarse_g 键（破坏「dict = compute_score 输出」
    # 契约）。此处 clone 恢复原版语义（[1,1,Hkv,kt] tiny copy，换掉
    # rearrange 物化 + MeanOps kernel 仍净赚）；下方原地写落在副本上，
    # 与原版「写 mean 产物」逐位等价。
    (
        '        if not self.per_q_head:\n'
        '            score_coarse = rearrange(\n'
        '                score_coarse, "b qt (h g) kt -> b qt h g kt", g=self.group_size\n'
        '            ).mean(dim=-2)  # [1,1,Hkv,kt]',
        "        _sc_g_pre = score_dict.get(\"score_coarse_g\")\n"
        "        if (not self.per_q_head and _sc_g_pre is not None\n"
        "                and _sc_g_pre.dim() == 4\n"
        "                and _sc_g_pre.shape[-1] == score_coarse.shape[-1]\n"
        "                and _sc_g_pre.shape[-2] * self.group_size == score_coarse.shape[-2]):\n"
        "            # 【E127 iter9】gmean 前移直供（compute_score kernel 双输出；\n"
        "            # clone 保「score_dict 键不被 mask 阶段原地写污染」契约）\n"
        "            score_coarse = _sc_g_pre.clone()\n"
        "        elif not self.per_q_head:\n"
        "            score_coarse = rearrange(\n"
        "                score_coarse, \"b qt (h g) kt -> b qt h g kt\", g=self.group_size\n"
        "            ).mean(dim=-2)  # [1,1,Hkv,kt]",
    ),
    # ---- 【iter9】M-B：score_coarse_avg group-mean 前移消费（同 M-A，
    # clone 保 dict 键不被 mask 阶段原地写污染；单池分支 L1036 的 clone
    # 后写不触及本张量）。
    (
        '        score_coarse_avg = score_dict.get("score_coarse_avg")\n'
        '        if score_coarse_avg is not None and not self.per_q_head:\n'
        '            score_coarse_avg = rearrange(\n'
        '                score_coarse_avg, "b qt (h g) kt -> b qt h g kt", g=self.group_size\n'
        '            ).mean(dim=-2)',
        "        score_coarse_avg = score_dict.get(\"score_coarse_avg\")\n"
        "        if score_coarse_avg is not None and not self.per_q_head:\n"
        "            _sc_avg_g_pre = score_dict.get(\"score_coarse_avg_g\")\n"
        "            if (_sc_avg_g_pre is not None\n"
        "                    and _sc_avg_g_pre.dim() == 4\n"
        "                    and _sc_avg_g_pre.shape[-1] == score_coarse_avg.shape[-1]\n"
        "                    and _sc_avg_g_pre.shape[-2] * self.group_size == score_coarse_avg.shape[-2]):\n"
        "                # 【E127 iter9】gmean 前移直供（同 M-A，clone 保契约）\n"
        "                score_coarse_avg = _sc_avg_g_pre.clone()\n"
        "            else:\n"
        "                score_coarse_avg = rearrange(\n"
        "                    score_coarse_avg, \"b qt (h g) kt -> b qt h g kt\", g=self.group_size\n"
        "                ).mean(dim=-2)",
    ),
    # ---- 【iter9】M-E：fused masked-softmax-gmean（共享口径前段 6 op →
    # 2 kernel；原型逐位一致：libdevice.exp=__nv_expf 与 PyTorch softmax
    # 内部 expf 同实现 + mean 前 bf16 舍入口径对齐 + l 分块归约差异被
    # bf16 舍入吸收，下游 topk mask 逐位不变）。路由条件全 host 已知量
    # （零同步判走）；异常 → 置 _triton_ok=False 回退原段（值同原路径）。
    (
        '        if self.per_q_head:\n'
        '            # E103：mask 已是 per-q-head [1,1,H,kt]，只做块→token 展开（不按 G 扩）\n'
        '            topk_mask = repeat(\n'
        '                topk_mask, "b qt h kt -> b qt h (kt bs)", bs=bs\n'
        '            )\n'
        '        else:\n'
        '            topk_mask = repeat(\n'
        '                topk_mask, "b qt h kt -> b qt (h g) (kt bs)", g=self.group_size, bs=bs\n'
        '            )\n'
        '\n'
        '        if self.enable_async:\n'
        '            if self.prev_mask is not None:\n'
        '                topk_mask, self.prev_mask = self.prev_mask, topk_mask\n'
        '                if topk_mask.shape[-1] < score_fine.shape[-1]:\n'
        '                    pad_len = score_fine.shape[-1] - topk_mask.shape[-1]\n'
        '                    topk_mask = F.pad(topk_mask, (0, pad_len), value=True)\n'
        '            else:\n'
        '                self.prev_mask = topk_mask\n'
        '\n'
        '        topk_mask = topk_mask[..., : score_fine.shape[-1]]\n'
        '        score_fine = torch.where(topk_mask, score_fine, float("-inf"))\n'
        '        p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)\n'
        '        if not self.per_q_head:\n'
        '            # E103：共享口径组内 mean；per_q_head 保留 [1,1,H,T] 直接 topk\n'
        '            p = rearrange(p, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)\n'
        '        # 【075 修复】SWA 强制区按真实 S：p[..., -swa:] 在投影路径落 padded\n'
        '        # 尾部（消费端裁剪后真实序列尾只保 swa−(Tpad−S) 个 token，S=6145\n'
        '        # 实测 65/128）。改为 [S_real−swa, S_real)；非投影路径 S_real==宽度\n'
        '        # → 与旧 p[..., -swa:] 逐位等价（S_real<swa 时两者同为全宽度置位）。\n'
        '        p[..., max(0, S_real - swa_tok):S_real] = 1.0',
        "        _e127_p_fused = False\n"
        "        if (\n"
        "            not self.per_q_head\n"
        "            and not self.enable_async\n"
        "            and getattr(self, \"_triton_ok\", True)\n"
        "            and topk_mask.dim() == 4 and topk_mask.dtype == torch.bool\n"
        "            and topk_mask.stride(-1) == 1\n"
        "            and topk_mask.shape[0] == 1 and topk_mask.shape[1] == 1\n"
        "            and score_fine.dim() == 4 and score_fine.stride(-1) == 1\n"
        "            and score_fine.shape[0] == 1 and score_fine.shape[1] == 1\n"
        "            and topk_mask.shape[-2] * self.group_size == score_fine.shape[-2]\n"
        "            and topk_mask.shape[-1] * bs >= score_fine.shape[-1]\n"
        "            and 0 < score_fine.shape[-1] <= 32768\n"
        "        ):\n"
        "            # 【E127 iter9】fused masked-softmax-gmean：repeat+where+\n"
        "            # softmax+nan_to_num+mean+swa 置位 6 op → 2 kernel（原型\n"
        "            # 逐位一致验证；下游 topk mask 逐位不变）。异常置标志\n"
        "            # 回退原段（值同原路径）。\n"
        "            try:\n"
        "                p = self._fast_mask_p(\n"
        "                    topk_mask, score_fine, bs,\n"
        "                    max(0, S_real - swa_tok), S_real)\n"
        "                _e127_p_fused = True\n"
        "            except Exception:\n"
        "                self._triton_ok = False\n"
        "                _e127_p_fused = False\n"
        "        if not _e127_p_fused:\n"
        "            if self.per_q_head:\n"
        "                # E103：mask 已是 per-q-head [1,1,H,kt]，只做块→token 展开（不按 G 扩）\n"
        "                topk_mask = repeat(\n"
        "                    topk_mask, \"b qt h kt -> b qt h (kt bs)\", bs=bs\n"
        "                )\n"
        "            else:\n"
        "                topk_mask = repeat(\n"
        "                    topk_mask, \"b qt h kt -> b qt (h g) (kt bs)\", g=self.group_size, bs=bs\n"
        "                )\n"
        "\n"
        "            if self.enable_async:\n"
        "                if self.prev_mask is not None:\n"
        "                    topk_mask, self.prev_mask = self.prev_mask, topk_mask\n"
        "                    if topk_mask.shape[-1] < score_fine.shape[-1]:\n"
        "                        pad_len = score_fine.shape[-1] - topk_mask.shape[-1]\n"
        "                        topk_mask = F.pad(topk_mask, (0, pad_len), value=True)\n"
        "                else:\n"
        "                    self.prev_mask = topk_mask\n"
        "\n"
        "            topk_mask = topk_mask[..., : score_fine.shape[-1]]\n"
        "            score_fine = torch.where(topk_mask, score_fine, float(\"-inf\"))\n"
        "            p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)\n"
        "            if not self.per_q_head:\n"
        "                # E103：共享口径组内 mean；per_q_head 保留 [1,1,H,T] 直接 topk\n"
        "                p = rearrange(p, \"b qt (h g) kt -> b qt h g kt\", g=self.group_size).mean(dim=-2)\n"
        "            # 【075 修复】SWA 强制区按真实 S：p[..., -swa:] 在投影路径落 padded\n"
        "            # 尾部（消费端裁剪后真实序列尾只保 swa−(Tpad−S) 个 token，S=6145\n"
        "            # 实测 65/128）。改为 [S_real−swa, S_real)；非投影路径 S_real==宽度\n"
        "            # → 与旧 p[..., -swa:] 逐位等价（S_real<swa 时两者同为全宽度置位）。\n"
        "            p[..., max(0, S_real - swa_tok):S_real] = 1.0",
    ),
]

TLIIndexerFast.compute_score = _derive_anchor_method("compute_score", _SCORE_REPL)
TLIIndexerFast.compute_mask = _derive_anchor_method("compute_mask", _MASK_REPL)
