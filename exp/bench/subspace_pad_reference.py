"""Frozen main8f213e4 _prepare_index_base; only import made absolute.
Benchmark/reference only; never imported by production.
"""
import torch
from einops import rearrange
from sparse_attn.indexer.tli_indexer import _HAS_TRITON

def _prepare_index_base(self, k, cu_seqlens_k, valid_length=None):
    from sparse_attn.indexer.utils import prepare_pad_mask, pad_tensor
    self._resolve_skip()
    idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None
    # 【阶段 2a 固定宽度】块统计有效区：固定全宽 KV（k.shape[1]=S_cap
    # 容量 > valid_length 有效长）下 pad 列进入块统计 → k_min/k_max/k_avg
    # 与变长路径（无 pad 列）不一致 → score_coarse 漂移 → L1 选择/mask
    # 漂移（G5 aavg_e64 normal 反例）。非投影分支的块统计从有效区切片算
    # （宽 ceil(vl/bs)），下游 kt=score_coarse 宽随之有效化 → 近池/
    # swa_lo_blk 边界与变长一致（G6 bypass 断言反例同源消解）。k_qat
    # 保持全宽（非候选化路径 score_fine 需全宽 S_cap；pad 列由
    # compute_mask 显式置 -inf）。投影分支（_basis 非 None，消融臂）
    # 保持全宽旧行为逐位不变。
    k_stat = k
    cu_stat = cu_seqlens_k
    if (valid_length is not None and valid_length < k.shape[1]
            and self._basis is None):
        k_stat = k[:, :valid_length]
        cu_stat = torch.tensor([0, valid_length],
                               dtype=cu_seqlens_k.dtype,
                               device=cu_seqlens_k.device)
    # 【阶段 2c-5】CUDA Graph capture 纯张量路径：prepare_pad_mask 的
    # 逐元素迭代与 pad_tensor 的 bool 掩码散射都需要 host 侧元素个数
    # 同步（capture 期非法）。单行（k_stat.shape[0]==1，本函数断言）且
    # pad_mask 恒为前缀 True 布局 → 静态 total + 切片写入，值逐位相同。
    if k.is_cuda and torch.cuda.is_current_stream_capturing():
        _pbs = self.args.tia_block_size
        _total = (k_stat.shape[1] + _pbs - 1) // _pbs * _pbs
        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(_pbs, cu_stat,
                                                      total=_total)
        pad_k = k_stat.new_zeros((k_stat.shape[0], _total)
                                 + tuple(k_stat.shape[2:]))
        pad_k[:, :k_stat.shape[1]] = k_stat
    else:
        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(
            self.args.tia_block_size, cu_stat)
        pad_k = pad_tensor(k_stat, pad_mask)
    # ---- 投影基惰性提取（每层一次）：full-D [128,r] → 子空间 [32,r] ----
    if self._basis_all is not None and self._basis is None and self.layer_idx is not None:
        li = min(self.layer_idx, self._basis_all.shape[0] - 1)
        # 校准时 W[:, D2I, :] = B_sub，故 [:, idx_sub, :] 精确还原子空间基
        idx_cpu = idx_sub.cpu() if idx_sub is not None else slice(None)
        self._basis = self._basis_all[li][:, idx_cpu, :].to(k.device)  # [Hkv, 32, r]
    k_for_coarse = pad_k[..., idx_sub] if idx_sub is not None else pad_k
    if self._basis is not None:
        # 投影路径（E64f 协议）：粗筛+细筛都在 d 维投影特征上
        kp = torch.einsum("bshd,hde->bshe", k_for_coarse.float(), self._basis)
        # 【E176】全量块统计融合（投影路径）：kp 投影后特征，
        # amin+amax+mean 一次读数据。
        _proj_need_avg = self._need_avg_score()
        _proj_use_fused = (_HAS_TRITON and self._triton_ok
                           and self._stat_ok and kp.is_cuda
                           and kp.stride(-1) == 1
                           and (self.args.tia_block_size & (self.args.tia_block_size - 1)) == 0)
        if _proj_use_fused:
            try:
                k_min, k_max, k_avg_proj = self._fast_stat_full(
                    kp, None, _proj_need_avg)
            except Exception:
                if not self._stat_ok:
                    _proj_use_fused = False
                else:
                    raise
        if not _proj_use_fused:
            k_coarse = rearrange(kp,
                "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
            k_min = k_coarse.amin(dim=2)
            k_max = k_coarse.amax(dim=2)
            k_avg_proj = k_coarse.mean(dim=2) if _proj_need_avg else None
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size
        # L2 4bit 精筛：投影特征逐 token 量化（任意 last-dim 通用）
        # 【E155b 去量化】l2_quant=False 直接用原始投影特征（同 dtype 视图语义）
        k_qat = self.min_max_per_token_quant(kp) if self.l2_quant else kp
        index_dict = {
            "k_min": k_min,
            "k_max": k_max,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
            "cu_seqlens_k_fine": cu_seqlens_k,
            # 【075 修复 2026-10-10（GPT 审计 TL-E121-PROJ-PAD-S-075）】投影
            # 路径 k_qat 从 pad_k 派生 → score_fine 宽度 = Tpad（pad 后），
            # 真实长度必须显式携带（= pad 前 k 的 token 数；B=1 下与
            # cu_seqlens_k[-1]-cu_seqlens_k[0] 相等），供 compute_mask 把
            # near/far 边界与 SWA 强制区落在真实域。
            "valid_length": k.shape[1],
        }
        if _proj_need_avg:
            # near/far 粗筛 avg 分数（E64f：near=avg / far=minmax；far=avg 单池也需）
            # 【E176】k_avg_proj 已由 _fast_stat_full 或 fallback 产出
            index_dict["k_avg"] = k_avg_proj
        # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
        if (self.enable_kmeans and not self.skip_far
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            self._maybe_build_kmeans(k, index_dict)
        return index_dict
    # 【E176】全量块统计融合：amin + amax + mean 一次读数据
    # （替代 rearrange + amin + amax + mean 四次读）。Triton 异常
    # 时 _fast_stat_full 置 _stat_ok=False 并 raise，except 走原版链。
    need_avg = self._need_avg_score()
    _use_fused = (_HAS_TRITON and self._triton_ok and self._stat_ok
                  and k_for_coarse.is_cuda
                  and (self.args.tia_block_size & (self.args.tia_block_size - 1)) == 0
                  and k_for_coarse.stride(-1) == 1)
    if _use_fused:
        try:
            # 注意：k_for_coarse 在此处已是子空间切片后的 tensor
            # （L1169: pad_k[..., idx_sub] if idx_sub is not None），
            # 不能再传 idx_sub 给 kernel 做二次 gather（越界）。
            # _fast_stat_full 内部会根据 stride(-1) 与 shape(-1) 判断是否需 sub。
            k_min, k_max, k_avg = self._fast_stat_full(
                k_for_coarse, None, need_avg)
        except Exception:
            if not self._stat_ok:
                _use_fused = False  # 永久回退原版链
            else:
                raise
    if not _use_fused:
        k_coarse = rearrange(k_for_coarse,
            "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
        k_min = k_coarse.amin(dim=2)
        k_max = k_coarse.amax(dim=2)
        k_avg = k_coarse.mean(dim=2) if need_avg else None
    # E89 MoBA 臂：全维 chunk-mean（gate 分数用完整 128 维 pad_k，严格 MoBA 口径）
    k_avg_full = None
    if self.moba_gate:
        kc_full = rearrange(
            pad_k, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
        )
        k_avg_full = kc_full.mean(dim=2)
    cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size

    # 【E142a】维度门：Qwen3 GQA 128 维 / DeepSeek MLA mla_rope 臂 192 维
    # （nope 128 + rope 64；选择子空间 mla_rope 只取尾 64 维）。
    # 其他模式（full/tail/rope/nope/random/highfreq）保持 128 硬门不变。
    # E85f 口径修正：L2 细筛 k_qat 与 L1 粗筛用同一子空间（idx_sub）——
    # E85e 重放口径是粗筛+细筛同特征；tail 臂 idx_sub 与旧硬编码 tail32
    # 逐位相同（cmp_ratio=2），仅 rope/nope 探索臂与静态 pair 臂行为改变
    # 【E133b 原生化】k_qat 直接产出 [1,S,Hkv,delta] 紧凑布局（原为
    # zeros_like(k) 128 维零填充 dense + 尾段 scatter——96 个零维参与
    # fine 打分读带宽 4× 浪费；零维乘积恒为 0，紧凑布局数学等价，仅
    # fp32 累加顺序变化 → bf16 舍入后 ULP 级漂移，E127 iter6 同口径）。
    # full 模式 L2 同样用 tail-delta 尾段（cmp_ratio 口径，idx_sub=None
    # 时 _qat_indices 缓存路径），full 锚点 fine 段同样受益。
    # 【E134】l2_subspace=full 时 L2 细筛全 128 维量化（L1 仍由 idx_sub
    # 决定）——sub+full 臂消费端，score_fine 宽度 = 全维（128 宽 dense，
    # 无紧凑化问题）。
    if self.l2_subspace == "full":
        # 【E155b 去量化】l2_quant=False 直接用原始 k（无压缩收益的假量化旁路）
        k_qat = self.min_max_per_token_quant(k) if self.l2_quant else k   # [1, S, Hkv, 128]
    else:
        indices = idx_sub if idx_sub is not None else self._qat_indices(k.device)
        # 【E155b 去量化】同上——原始子空间切片直用（gather 拷贝保持紧凑布局）
        k_qat = (self.min_max_per_token_quant(k[..., indices]) if self.l2_quant
                 else k[..., indices])
    index_dict = {
        "k_min": k_min,
        "k_max": k_max,
        "k_avg": k_avg,
        "k_avg_full": k_avg_full,
        "k_qat": k_qat,
        "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
        "cu_seqlens_k_fine": cu_seqlens_k,
        # 075：非投影路径 k_qat 从真实 k 派生，valid_length == score_fine
        # 宽度 → compute_mask 行为逐位不变（对称携带，接口统一）
        "valid_length": k.shape[1],
    }
    # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
    if (self.enable_kmeans and not self.skip_far
            and (self.far_select != "4bit" or self.near_select != "4bit")):
        self._maybe_build_kmeans(k, index_dict)
    return index_dict
