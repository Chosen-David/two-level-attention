# -*- coding: utf-8 -*-
"""【E127 iter4】TLIIndexer 增量缓存子类（decode prepare_index 加速）。

锚点文件 sparse_attn/indexer/tli_indexer.py 零改动（md5 身份不变）；
本子类经 sparse_attn/patches/patch.py 在 TLI_E127_FAST=1 时替换实例化
（默认 0 走原类，在飞链零扰动）。

优化目标：原 prepare_index 每次 decode step 对全序列 S 做 O(S) 重算
（S=64K 每层步 2.89ms：pad_tensor 全量复制 134MB + k[..., indices]
gather + 全量 quant + zeros_like 134MB + amax/amin/mean 全块归约），
而 decode 每步只追加 1 个新 token——统计量只有尾块变化。

逐位一致证明（数学保证，不依赖实测运气）：
  - k_min/k_max = k_coarse.amax/amin(dim=2)、k_avg = mean(dim=2)：bs 维
    （块内）独立归约，块间无交叉 → 块 [0, fb) 输入不变时其输出不变；
    缓存值即原 kernel 的历史输出，与本次全量重算逐位相同（同 kernel、
    同输入、确定性）。
  - k_qat = zeros_like(k) + [..., indices] 散布 min_max_per_token_quant：
    量化沿 dim=-1 逐 token 独立，indices 维外恒 0 → 尾段先 zero_ 再
    scatter 与全量版 zeros_like+scatter 逐位相同；前缀段保留历史即
    全量重算值。
  - 尾块重算用与全量版相同的 pad 语义（块对齐、pad 值 0）→ 尾块统计
    含 padding 的行为与全量版一致。

契约（fail-closed 防御，全部回退全量重算）：
  - k append-only（HF DynamicCache decode 语义：每步 cat 追加，前缀
    不可变）。S_now > S_prev 才增量；S_now ≤ S_prev（新请求/异常回退/
    前缀改写导致的长度回退）→ 全量重建缓存。
  - clear()（patch 每次 prefill 前调用）使缓存失效 → 新请求首步全量。
  - 非默认路径（投影基 _basis/_basis_all 非 None、moba gate、D≠128、
    子空间 key 变化）一律 super() 全量，行为与原类逐位相同。
  - 残余风险：调用方改写前缀但长度单调增（理论上的契约破坏）无法零
    成本检测——与 cu=[0,S] 契约同族，文档化；如需强校验可加全前缀
    逐位比对（成本 ~8µs/层步，见 E127 报告建议）。
"""

import torch
from einops import rearrange

from .tli_indexer import TLIIndexer
from .utils import prepare_pad_mask


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
        _, pad_cu_seqlens_k = prepare_pad_mask(bs, cu_seqlens_k)
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

        # ---- k_qat 增量：预分配 buffer 步进写（尾段 zero_ + scatter，
        # 与全量版 zeros_like+scatter 逐位同构）----
        delta = 64 // self.cmp_ratio
        indices = idx_sub if idx_sub is not None else torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128))
        ).to(k.device)
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
