"""
LoDA-SA (论文名) / TLI (仓库内部实现) —— 解码阶段全张量流伪代码

对应生产配置 champion flags（Qwen3-8B, tail 子空间, H20）:
  --method tli
  --tli_far_method minmax --tli_near_method avg
  --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096
  --tli_subspace tail --tia_level2_cmp_ratio 4
  --tli_sink_tokens 4 --tli_swa_size 128
  --tia_level1_topk 128 --tia_level2_topk 1024

geometric constants
  D=128  full head dim   (Qwen3-8B)
  D_s=32 subspace (tail: [48:64]∪[112:128])
  H=32   query heads; Hkv=8 KV heads (GQA group G=4)
  B=64   block size (tokens per page)
  S_cap full-width KV buffer (static capacity, NOT sequence-dependent)

core shape legend (B=1 decode step):
  q      [1, 1, H,  D]    query (new token, post-RoPE, full dim)
  k_buf  [1, S_cap, Hkv, D]    static KV cache (ALL tokens, full-width, NEVER compressed)
  v_buf  [1, S_cap, Hkv, D]
  mask   [1, 1, Hkv, S_cap] bool → consumed by gather attention

File map (pseudo-code line → real-file:line):
  §0  constants         — arguments.py L6-8 (K1/K2/block defaults)
  §1  static cache init — tli_static_cache.py L141-168 (全宽 buffer copy)
  §2  prepare_index     — tli_indexer.py L794-855 (缓存 / 增量)
  §3  compute_mask      — tli_indexer.py L4151-4511 (主掩码流水线)
  §4  gather attention  — eager_decoding.py L719-834 (稀疏消费端)
  §5  head-forward      — qwen3_attn_patch.py L81-126 (完整 decode 外层)
"""

import torch
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════ §0
# 几何常量（champion flags，与 arguments.py L6-8 对应）
# ════════════════════════════════════════════════════════════════════

K1   = 128          # L1 粗筛页数（单位：「页」，非 token）
K2   = 1024         # L2 细筛 token 数（单池自由竞争 top-K2）
nb_near = 32        # near 区在 K1 中占 nb_near 页（剩余 nb_far 页给 far）
nb_far  = K1 - nb_near  # = 96
nl   = 4096         # near 区显式 token 宽度（nt=off 时自由竞争）
D    = 128          # full head dim
D_s  = 32           # L1/L2 选择子空间（tail-low-freq 32 维）
H    = 32           # query heads (Qwen3-8B)
Hkv  = 8            # KV heads (GQA 4:1)
G    = H // Hkv     # = 4 (group size)
B    = 64           # block/page size (tokens per page)
sink = 4            # sink token 数（前缀强制保留）
swa  = 128          # sliding window token 数（尾部强制保留）

# ── 导出：全局/设备常量，capture 前烘焙为常量张量 ──
# （对应 gather path qwen3_attn_patch.py L70-78，graph const cache）
kt_cap = None       # ← warmup 首步计算：S_cap // B（总块数）


# ════════════════════════════════════════════════════════════════════ §1
# 全宽 KV cache 初始化（TLIStaticCache: 全量静态，无物理压缩）
# ════════════════════════════════════════════════════════════════════
# 对应 sparse_attn/ops/tli_static_cache.py L141-194
#
# 核心语义：prefill 后 [0, L) 全部 token 的 K/V 一次性搬入固定宽 buffer，
# 后续 decode 不走删除/追加——追加是 pad 区静态覆盖（_ginc buffer）。
# 稀疏仅靠 attention mask，不靠物理 KV 裁剪。

def init_static_cache(k_prefill: torch.Tensor, v_prefill: torch.Tensor,
                      S_cap: int):
    """
    k_prefill: [1, L, Hkv, D]   prefill 后的全量 K
    v_prefill: [1, L, Hkv, D]   prefill 后的全量 V
    S_cap:     static capacity  （≥ max_seq_len）
    returns:   (k_buf, v_buf)   全宽静态 buffer，后续只覆写不分配
    """
    _, L, _, D = k_prefill.shape
    dev = k_prefill.device
    # ★ 全宽 buffer：每一层一个 [1, S_cap, Hkv, D]，零初始化
    k_buf = torch.zeros(1, S_cap, Hkv, D, device=dev, dtype=k_prefill.dtype)
    v_buf = torch.zeros(1, S_cap, Hkv, D, device=dev, dtype=v_prefill.dtype)
    # prefill token copy → 前缀，pad 区恒零
    k_buf[:, :L].copy_(k_prefill)   # 对应 L193: layer.keys[b, :, :L].copy_(...)
    v_buf[:, :L].copy_(v_prefill)
    return k_buf, v_buf


# ════════════════════════════════════════════════════════════════════ §2
# 子空间投影 + 页级块统计 (prepare_index)
# ════════════════════════════════════════════════════════════════════
# 对应 sparse_attn/indexer/tli_indexer.py L794-855 (dispatch)
#                                             L1359-1410 (base build)
# 输入：full-width k_buf（graph 路径容量宽）；输出：块级统计摘要。

# tail32 子空间索引（低频 RoPE 维，cmp_ratio=2时为 [48:64]∪[112:128]，
# 与 tail_d32 配置逐位对应，见 tli_indexer.py 内 _qat_indices 实现）
IDX_SUB = torch.tensor([48, 49, 50, 51, 52, 53, 54, 55,
                        56, 57, 58, 59, 60, 61, 62, 63,
                        112, 113, 114, 115, 116, 117, 118, 119,
                        120, 121, 122, 123, 124, 125, 126, 127],
                       dtype=torch.long)


def subspace_indices(device: torch.device) -> torch.Tensor:
    """返回 tail32 子空间 D_s=32 维的 long 索引（缓存复用）。"""
    # 对应 tli_indexer.py L860-864 _sub_key + _qat_indices cache
    return IDX_SUB.to(device)


def prepare_index(k_buf: torch.Tensor, cu_k: torch.Tensor, S_cap: int):
    """
    对全宽 KV buffer 做子空间投影 → 块级 min/max/avg 统计。

    k_buf:    [1, S_cap, Hkv, D]    full-width KV (static)
    cu_k:     [2]  int32   varlen 累计位置 [0, S_cap]（单 request）
    S_cap:    static 容量（真实序列长在下文 valid_len 传入）
    returns:  dict {k_min, k_max, k_avg, k_qat, cu_k_coarse, cu_k_fine, valid_length}

    形状说明:
      kt_cap = S_cap // B           总块数
      k_min  [1, kt_cap, Hkv, D_s]   块内 min（Quest 式符号感知上界估计）
      k_max  [1, kt_cap, Hkv, D_s]   块内 max
      k_avg  [1, kt_cap, Hkv, D_s]   块内 mean（near 区粗筛用）
      k_qat  [1, S_cap, Hkv, D_s]   per-token 量化特征（L2 细筛用）
    """
    idx_sub = subspace_indices(k_buf.device)

    # ★ 子空间投影: full D → D_s=32，节省后续所有 coarse/fine 带宽
    # 对应 tli_indexer.py L1359: k_for_coarse = k_full[..., idx_sub]
    k_s = k_buf[..., idx_sub]                       # [1, S_cap, Hkv, D_s]

    # ── 页级块统计: rearrange(S_cap → kt_cap blocks × B tokens) ──
    kt_cap = S_cap // B
    k_blk = k_s.reshape(1, kt_cap, B, Hkv, D_s)     # [1, kt_cap, B, Hkv, D_s]
    k_min = k_blk.amin(dim=2)                       # [1, kt_cap, Hkv, D_s]
    k_max = k_blk.amax(dim=2)                       # [1, kt_cap, Hkv, D_s]
    k_avg = k_blk.mean(dim=2)                       # [1, kt_cap, Hkv, D_s]

    # ── per-token 子空间切片 (L2 细筛复用) ──
    k_qat = k_s                                     # [1, S_cap, Hkv, D_s]

    return {
        "k_min": k_min,          # block-min (L1 far 粗筛)
        "k_max": k_max,          # block-max (L1 far 粗筛 Quest 上界)
        "k_avg": k_avg,          # block-mean (L1 near 粗筛)
        "k_qat": k_qat,          # per-token 子空间 (L2 细筛)
        "cu_k_coarse": cu_k // B,
        "cu_k_fine": cu_k,
        "valid_length": S_cap,
        "kt_cap": kt_cap,
    }


# ════════════════════════════════════════════════════════════════════ §3
# L1 双池粗筛 + L2 单池细筛 + 终掩码装配 (compute_mask)
# ════════════════════════════════════════════════════════════════════
# 对应 sparse_attn/indexer/tli_indexer.py L4151-4511 (graph 路径)
# 主配置守卫见 L4160-4172（alpha/beta=1,gamma=None,nb/nl 显式,
#                         非投影/非 per_q_head/非簇分/moba/sigma/async,
#                         skip_far=False）


def _resolve_geometry(valid_len: int, S_cap: int, kt_cap: int):
    """
    将标量 valid_len [int] → 近端/远端块划分。
    对应 tli_indexer.py L4223-4256 (graph 步级缓存 _graph_geo_lookup)

    划分（距离 query 由近到远）:
      [0, sink)              → sink 区   (强制 attend)
      [sink, far_tok_lo)     → far 区    (L1/L2 选择域)
      [far_tok_lo, swa_lo)   → near 区   (L1/L2 选择域)
      [swa_lo, valid_len)    → swa 强制区 (强制 attend)
      [valid_len, S_cap)     → pad 区    (mask False，不 attend)
    """
    mid_len  = valid_len - sink - swa         # 选择域总长
    far_tok_lo = sink                         # far 区起点（sink 之后）
    near_tok_lo = valid_len - nl              # near 区起点（nl token 宽）
    swa_lo_tok  = valid_len - swa             # swa 强制区起点

    # 切块（页对齐）
    far_lo_blk  = 0                           # 第 0 块起（含 sink 跨界页，后 bypass 补）
    near_blkt   = near_tok_lo // B            # near 区起始块
    swa_lo_blk  = swa_lo_tok // B             # swa 强制区起始块
    near_blks   = swa_lo_blk - near_blkt      # near 页数上限
    return far_tok_lo, near_blkt, swa_lo_blk, swa_lo_tok


def _coarse_score(k_min, k_max, b_q, method="minmax"):
    """
    L1 粗筛打分（Quest 式符号感知上界 / avg 点积）。

    k_min / k_max: [1, kt_cap, Hkv, D_s]   块级子空间极值
    b_q:           [H, D_s]               query 子空间投影（已 scale）

    Quest 上界（对应 _fast_score_coarse, tli_indexer.py L4290-4294）:
      score(t) = Σ_d [ q_-(d) · k_min(t,d) + q_+(d) · k_max(t,d) ]
      等价 einsum("h d, kt h d -> h kt", q_pos, k_max)
           + einsum("h d, kt h d -> h kt", q_neg, k_min)

    avg 点积（对应 _fast_score_fine, tli_indexer.py L4303-4305）:
      score(t) = einsum("h d, kt h d -> h kt", q, k_avg)

    returns: [1, H, kt_cap]
    """
    # group_size 展开: k_min [kt, Hkv, d] → [kt, H, d]（GQA 共享）
    k_min_g = k_min.repeat_interleave(G, dim=2).float()      # [1, kt, H, d]
    k_max_g = k_max.repeat_interleave(G, dim=2).float()
    if method == "minmax":
        q_pos = b_q.clamp(min=0)
        q_neg = b_q.clamp(max=0)
        score = (torch.einsum("h d, k h d -> h k", q_pos, k_max_g[0])
                 + torch.einsum("h d, k h d -> h k", q_neg, k_min_g[0]))
    else:  # avg
        raise NotImplementedError
    return score.unsqueeze(0)


def compute_mask(q: torch.Tensor, index: dict, valid_len: int, S_cap: int):
    """
    TLI 核心 mask 流水线：L1 双池粗筛 → L2 单池细筛 → 终掩码装配。

    q:         [1, 1, H, D]   query token (post-RoPE, full dim)
    index:     dict (prepare_index 输出)
    valid_len: int            真实序列长
    S_cap:    int             静容量

    returns:   mask [1, 1, Hkv, S_cap] bool
               被选中 (mask=True) 的 token 才会参与 attention。
    """
    kt_cap = index["kt_cap"]
    k_min, k_max, k_avg = index["k_min"], index["k_max"], index["k_avg"]
    k_qat = index["k_qat"]                          # [1, S_cap, Hkv, D_s]
    dev = q.device

    # ── 子空间投影（Q 侧）──
    idx_sub = subspace_indices(dev)
    q_s = q[..., idx_sub]                           # [1, 1, H, D_s]
    q_s = (q_s.squeeze(0).squeeze(0) * (D ** -0.5)).float()  # [H, D_s], scaled

    # ── 块划分 ──
    far_tok_lo, near_blkt, swa_lo_blk, swa_lo_tok = _resolve_geometry(
        valid_len, S_cap, kt_cap)
    near_blks = swa_lo_blk - near_blkt

    # ── L1: 双池 Top-K1（far / near 各自独立 tag，不等权混排）──
    sc_coarse = _coarse_score(k_min, k_max, q_s, method="minmax")  # [1,H,kt_cap]
    sc_avg    = None  # 若有 k_avg 则算 avg 打分（near 臂用）; 此处省略 avg 路径

    # GQA 聚合到 Hkv（组内 mean，对应 tli_indexer.py L4307）
    sc_kv = sc_coarse.view(1, Hkv, G, kt_cap).mean(dim=2)  # [1, Hkv, kt_cap]

    # 池掩码: far 池 = [far_blk_lo, near_blkt)；near 池 = [near_blkt, swa_lo_blk)
    far_lo_blk = 0
    far_pool_m  = torch.zeros(kt_cap, dtype=torch.bool, device=dev)
    near_pool_m = torch.zeros(kt_cap, dtype=torch.bool, device=dev)
    far_pool_m [far_lo_blk:near_blkt] = True
    near_pool_m[near_blkt:swa_lo_blk] = True

    # ★ 双池选择: far 区取 top nb_far 页 + near 区取 top nb_near 页
    far_scored  = sc_kv.masked_fill(~far_pool_m,  float("-inf"))
    near_scored = sc_kv.masked_fill(~near_pool_m, float("-inf"))
    i_f = torch.topk(far_scored,  nb_far,  dim=-1).indices  # [1,1,Hkv,nb_far]
    i_n = torch.topk(near_scored, nb_near, dim=-1).indices  # [1,1,Hkv,nb_near]

    # ── 候选集组装: L1 页展開 token 级（每块 B 个 token）──
    # 对应 tli_indexer.py L4344-4389 (iter3d 融合候选装配)
    device_list = torch.arange(B, device=dev).view(1,1,1,1,-1)
    l1_tok_f = i_f.unsqueeze(-1) * B + device_list          # [1,1,Hkv,nb_far, B]
    l1_tok_n = i_n.unsqueeze(-1) * B + device_list          # [1,1,Hkv,nb_near,B]
    cand_idx = torch.cat([
        l1_tok_f.reshape(1, 1, Hkv, nb_far  * B),
        l1_tok_n.reshape(1, 1, Hkv, nb_near * B),
    ], dim=-1)                                               # [1,1,Hkv,K1*B]
    keep = (cand_idx < valid_len)                           # 页内 PAD token 剔除
    cand_idx = cand_idx[0,0]                                 # [Hkv, K1*B]
    keep     = keep[0,0]                                     # [Hkv, K1*B]

    # ★ 紧凑压缩: keep=True 槽前置（对应 iter3 compact kernel / L4412-4425）
    sort_idx = torch.argsort(~keep, dim=-1, stable=True)     # garbage 后置
    cand_idx = cand_idx.gather(-1, sort_idx)                 # [Hkv, K1*B]
    keep     = keep.gather(-1, sort_idx)
    cand_len = keep.sum(-1).long()                           # [Hkv] 有效候选数

    # ── L2: 域内 Top-K2 token 精筛（near+far 候选「单池自由竞争」）──
    cand_idx_g = cand_idx.unsqueeze(1).expand(Hkv, G, K1*B)  # [Hkv,G,K1*B] → [H,...]
    # gather 候选 token 的 k_qat
    kv_id = torch.arange(Hkv, device=dev).view(-1, 1, 1)
    k_cand = k_qat[0, cand_idx, kv_id, :]                    # [Hkv, K1*B, D_s]
    k_cand_g = k_cand.repeat_interleave(G, dim=0).float()    # [H, K1*B, D_s]
    # fine 打分（子空间逐 token 点积）
    score_fine = torch.einsum("h d, h r d -> h r", q_s.float(), k_cand_g)  # [H, K1*B]
    score_fine = score_fine.view(Hkv, G, K1*B).mean(dim=1)    # [Hkv, K1*B] GQA
    # garbage 槽 -inf
    keep_g = keep.repeat_interleave(G, dim=0).view(Hkv, G, K1*B)[:, 0, :]  # [Hkv, K1*B]
    score_fine = torch.where(keep_g, score_fine,
                             torch.full_like(score_fine, float("-inf")))
    # softmax 归一化（紧凑域 [Hkv, R_cap]，对应 iter2b L4453-4454）
    p = F.softmax(score_fine, dim=-1)                        # [Hkv, K1*B]
    # ★ Top-K2: 单池自由竞争（near + far 候选混排 TopK2）
    _, i_top = torch.topk(p, K2, dim=-1)                    # [Hkv, K2]
    tok_top = cand_idx.gather(-1, i_top)                     # [Hkv, K2]

    # ── 终掩码装配: 选中 True + sink 强制 + swa 强制 ──
    # 对应 tli_indexer.py L4493-4498 + iter3a assembly fusion
    mask = torch.zeros(1, 1, Hkv, S_cap, dtype=torch.bool, device=dev)
    mask.scatter_(-1, tok_top.unsqueeze(0).unsqueeze(0), True)   # 选中位
    # sink 强制 (前缀)
    mask[..., :sink] = True
    # swa 强制 (尾部 recent window)
    mask[..., swa_lo_tok:valid_len] = True
    # pad 位置零（已经零初始化，边界安全）
    return mask                                                # [1,1,Hkv,S_cap]


# ════════════════════════════════════════════════════════════════════ §4
# 稀疏 attention 消费端（gather 路径）
# ════════════════════════════════════════════════════════════════════
# 对应 sparse_attn/ops/eager_decoding.py L719-767 (entry)
#                                             _eager_decoding_attn_gather (内层)


def sparse_decode_attn(q: torch.Tensor,
                       k_buf: torch.Tensor,
                       v_buf: torch.Tensor,
                       mask: torch.Tensor,
                       block_size: int = 1,
                       cu_k: torch.Tensor = None):
    """
    稀疏 decode attention（gather 路径，生产 main path）。

    q      [1, 1, H,  D]   query
    k_buf  [1, S_cap, Hkv, D]   full-width KV (ALL tokens, NEVER compressed)
    v_buf  [1, S_cap, Hkv, D]   full-width KV
    mask   [1, 1, Hkv, S_cap]   bool，True=attend

    选择语义: 只有 mask=True 的 token 参与注意力争和 softmax，其余被排除。
    KV 缓存 全部 token 仍保存在 k_buf/v_buf 中（物理不删）。

    三次 dispatch（对应 L819-829 + L832-834）
      (1)  T≥16K + triton avail + bool mask  → Triton gather fast-path
      (2)  否则                              → torch gather per-query
    """
    if cu_k is None:
        cu_k = torch.tensor([0, k_buf.shape[1]], device=q.device, dtype=torch.int32)

    B = q.shape[0]
    H = q.shape[2]
    Hkv = k_buf.shape[2]
    S_cap = k_buf.shape[1]
    dev = q.device
    m0 = mask[:, 0]                                    # [B, Hkv, S_cap]
    R = m0.shape[1]
    G_ = H // Hkv
    W_up = min(2048, S_cap)                            # TLI_E127_W_UPPER 默认 2048

    # ── (1) Triton gather fast-path (T≥16K + bool mask + head_dim 2 的幂) ──
    if (S_cap >= 16384 and m0.dtype == torch.bool
            and (k_buf.shape[-1] & (k_buf.shape[-1] - 1)) == 0
            and m0.stride(-1) == 1 and k_buf.stride(-1) == 1
            and v_buf.stride(-1) == 1 and cu_k.dtype in (torch.int32, torch.int64)):
        return _gather_triton(q, k_buf, v_buf, m0, G_, W_up, cu_k)

    # ── (2) torch gather fallback ──
    return _gather_torch(q, k_buf, v_buf, m0, G_, cu_k, S_cap)


def _gather_triton(q, k_buf, v_buf, m0, G_, W_up, cu_k):
    """
    Triton gather kernel（对应 _eager_gather_triton_batch）。

    操作摘要: 逐行 gather mask 选中的 token（~K2=1024 个 + sink+swa ≈ 256），
    做 GQA 展开 + online-softmax + output reduction; 远小于 dense [S_cap] 全长归约。
    输出与 dense 路径 ULP 级别等价（exp(-inf)=0 精确排除）。

    q [1,1,H,D], k_buf [1,S,Hkv,D], v_buf [1,S,Hkv,D], m0 [B,R,S]

    ► gather: select K=Σ(m0[b,r]) token per row → kv_sel [B, R, K, D]
    ► einsum("b h d, b r h d -> b r h", q, kv_sel) → QK [B,R,K]   (GQA mean)
    ► softmax(QK) → P [B,R,K]
    ► einsum("b r k, b r k d -> b r d", P, v_sel) → O [B,R,D]
    ► O residual add → Out [B, R, D]
    总带宽 O(K·D) 而非 O(S_cap·D)。K≈K2+sink+swa ≪ S_cap。
    """
    # 生产中此路径由 Triton 对应（手写 kernel 见 triton 包）；纯伪代码略具体 impl。
    raise NotImplementedError("Triton kernel 专用路径（有 triton 时自动路由）")


def _gather_torch(q, k_buf, v_buf, m0, G_, cu_k, S_cap):
    """
    torch gather fallback: 每行独立（对应 _eager_decoding_attn_gather_batch）。

    q [1,1,H,D], m0 [1, Hkv, S_cap] bool
    k_buf, v_buf 全宽 [S_cap, Hkv, D]
    """
    # ③ gather 选中 token 的 K/V（k_buf 全宽，但只读 mask=True 的 token）
    kv_id = torch.arange(q.shape[2] // m0.shape[0], device=q.device)  # [K_sel]
    k_sel = k_buf[0][m0[0]]                          # [K_sel, Hkv, D]  ◀ gather 读
    v_sel = v_buf[0][m0[0]]                          # [K_sel, Hkv, D]
    # GQA 组: k_sel [K_sel, Hkv, D] → [K_sel, H, D]
    k_g = k_sel.repeat_interleave(G_, dim=1)
    # QK 点积 + softmax
    scale = q.shape[-1] ** -0.5
    qk = torch.einsum("b h d, k h d -> b h k", q.view(1, -1, q.shape[-1]),
                      k_g.unsqueeze(0).float()) * scale       # [1, H, K_sel]
    p = F.softmax(qk, dim=-1)
    # attention 输出
    out = torch.einsum("b h k, k h d -> b h d", p,
                       v_sel.repeat_interleave(G_, dim=1).float())
    return out.view(q.shape)


# ════════════════════════════════════════════════════════════════════ §5
# 单层 TLI decode 入口（patch 外层调度）
# ════════════════════════════════════════════════════════════════════
# 对应 sparse_attn/patches/qwen3_attn_patch.py L81-126 (_tl_decode_batched_graph)


def tli_layer_forward(layer, q_full: torch.Tensor, k_buf: torch.Tensor,
                      v_buf: torch.Tensor, valid_len: int, S_cap: int,
                      cuda_graph: bool = True):
    """
    TLI 单层解码前置全管线（graph / eager 两路径共用）。

    q_full   [1,1,H,D]         query (post-RoPE)
    k_buf    [1,S_cap,Hkv,D]   全宽 KV cache (static, NEVER compressed)
    v_buf    [1,S_cap,Hkv,D]   全宽 KV cache
    valid_len int              当前真实有效序列长
    S_cap    int               静容量
    cuda_graph bool            是否走 CUDA Graph capture/replay

    每个 decode step 每层调用（Qwen3-8B × 36 层）：

      prepare_index → compute_mask → sparse_decode_attn

    其中 prepare_index 在 graph/iter3 模式下可以增量（~7 个 kernel/层/步）；
    iter 等迭代版本已融合多 kernel → 对应 kernel 数：
      原始 e152v4: ~8400 kernel/step/36层（每层 230+ kernel）
      iter2a+2b:  → ~3360 kernel/step（几何缓存 + 紧凑 softmax/topk）
      iter3d:     → ~2400 kernel/step（融合候选装配 + fine + assemble mask）
    """
    global kt_cap
    if kt_cap is None:
        kt_cap = S_cap // B

    # ① 准备子空间块统计（见 §2 prepare_index）
    cu_k = torch.tensor([0, S_cap], device=q_full.device, dtype=torch.int32)
    index = prepare_index(k_buf, cu_k, S_cap)
    index["valid_length"] = valid_len

    # ② L1 双池粗筛 + L2 单池细筛 → mask [1,1,Hkv,S_cap]（见 §3）
    #    对应 compute_mask_graph (tli_indexer.py L4151-4511)
    mask = compute_mask(q_full, index, valid_len, S_cap)

    # ③ 稀疏 attention（见 §4 gather 路径）
    #    对应 eager_decoding_attn (eager_decoding.py L719-757)
    out = sparse_decode_attn(q_full, k_buf, v_buf, mask)

    # ④ 残差 + FFN (dense, 不变)
    return out
