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
    def _fast_score_fine(self, kq, b_q_full, out_dtype):
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
            grid = (triton.cdiv(T, 64), Hkv)
            _score_fine_kernel[grid](
                kq, b_q_full, out, T,
                kq.stride(0), kq.stride(1), b_q_full.stride(0),
                out.stride(0), out.stride(1),
                Hkv=Hkv, G=g, D=D, BT=64, num_warps=4)
            return out
        except Exception:
            self._triton_ok = False
            raise


@triton.jit
def _score_fine_kernel(
    KQ, Q, OUT,   # KQ [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T] out_dtype
    T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
):
    """GQA K 复用 score_fine：每 program 处理 (token 块 × 1 个 kv 头)，
    K 块 [BT, D] 载入寄存器后循环 g 个 q 头逐头点积，K 读带宽 /g。"""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask_t = offs_t < T
    k = tl.load(KQ + offs_t[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :],
                mask=mask_t[:, None], other=0.0).to(tl.float32)
    for j in tl.static_range(G):
        h = pid_h * G + j
        q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已乘 scale）
        s = tl.sum(k * q[None, :], axis=1)            # [BT] fp32
        tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                 s.to(OUT.dtype.element_ty), mask=mask_t)


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


# ---- compute_score：两处替换 ----
# ① 零参 super() 在 exec 派生的独立函数中无 __class__ cell → 显式等价
#   改写（Fast 的 MRO 下 super().compute_score ≡ TLIIndexer.compute_score）；
# ② b_k_qat 物化 + 双分支 einsum（该两分支唯一消费点）→ 非投影
#   Triton GQA K 复用（_fast_score_fine；数学等价、bf16 舍入后 ULP 级
#   漂移、mask 门禁实测逐位不变）；投影/异形/编译失败回退原版表达式。
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
]

TLIIndexerFast.compute_score = _derive_anchor_method("compute_score", _SCORE_REPL)
TLIIndexerFast.compute_mask = _derive_anchor_method("compute_mask", _MASK_REPL)
