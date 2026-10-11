import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


def eager_decoding_attn(
    q: torch.Tensor, # [1, tq, hq, d]
    k: torch.Tensor, # [1, tk, h, d]
    v: torch.Tensor, # [1, tk, h, d]
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    """【E127 iter2】稀疏 attention 消费端入口（原实现原样保留为 _dense 版）。

    TLI_E127_FAST=1 时走 gather 路径：原实现对全序列 S 个 token 全量计算
    fp32 分数再 mask（稀疏性在消费端被完全丢弃——S=64K 时 k.to(float32)
    物化 + 全长 einsum + 全长 softmax ≈ 2.2ms/层步）；gather 路径只取
    每行 mask 选中的 token（~K2+sink+swa ≈ 1280）做 attention，带宽
    从 O(S·Hkv·D) 降到 O(R·W·D)。数学语义不变：softmax(全 S 行、未选
    位 -inf) ≡ softmax(选中子集)（exp(-inf)=0 精确），数值上仅归约顺序
    不同（输出 ULP 级漂移，allclose(1e-2) 门禁）。
    B7 全 False 行守卫语义保持：任一行空 → 与原路径相同的 RuntimeError
    （min(counts)==0 检测，同步次数不增——原 any().all() 本身就是一次
    host 同步）。
    """
    if os.environ.get("TLI_E127_FAST", "0") == "1":
        return _eager_decoding_attn_gather(
            q, k, v, mask, block_size, cu_seqlens_k, softmax_scale)
    return _eager_decoding_attn_dense(
        q, k, v, mask, block_size, cu_seqlens_k, softmax_scale)


def _eager_decoding_attn_dense(
    q: torch.Tensor, # [1, tq, hq, d]
    k: torch.Tensor, # [1, tk, h, d]
    v: torch.Tensor, # [1, tk, h, d]
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    if softmax_scale is None:
        softmax_scale = k.shape[-1] ** -0.5
    q, k, v, mask = (x.squeeze(0) for x in (q, k, v, mask))
    o = q.new_zeros(q.shape[:-1] + (v.shape[-1],))
    G = q.shape[-2] // k.shape[-2]
    bos_k, eos_k = cu_seqlens_k[0], cu_seqlens_k[1]
    # ---- E103：kv-head 共享消融消费端 ----
    # 共享口径：mask [Hkv, tk]（kv-head 级），repeat 后 [Hkv, tk*bs] 直接广播到组内
    # 全部 G 个 q-head（b_mask[:, None, :]）——「同组共享同一份选择」的落点；
    # per_q_head 口径：mask [H, tk]（q-head 级），按 G 拆回 [Hkv, G, tk*bs]
    # 逐元素对应——每个 q-head 用自己的选择，不做组内广播。
    m0 = mask[0]
    per_qh = m0.shape[0] != k.shape[-2]
    # B05 修复（GPT 审查 2026-10-08）：原版 [:, :eos_k-bos_k] 切的是 dim1
    # （per_qh 的 G 轴 / shared 的 Hkv 轴）而非最后一维 token 轴——无 padding
    # 时恰好是 no-op 故长期未暴露；投影路径 Tpad>T 时会广播失败或掩码不裁。
    # 改为裁 token 轴；无 padding（tk*bs == eos_k-bos_k）时与原版逐位等价。
    if per_qh:
        g_m = m0.shape[0] // k.shape[-2]
        b_mask = repeat(
            m0, '(h g) tk -> h g (tk bs)', g=g_m, bs=block_size
        )[..., :eos_k - bos_k]
    else:
        b_mask = repeat(mask[0], 'h tk -> h (tk bs)', bs=block_size)[..., :eos_k - bos_k]

    b_q = rearrange(q[0] * softmax_scale, '(h g) d -> h g d', g=G).to(torch.float32)
    b_k = k.to(torch.float32)
    b_v = v
    b_s = einsum(b_q, b_k, 'h g d, t h d -> h g t')
    if per_qh:
        # E103：b_mask 已是 [Hkv, G, T] 与 b_s 同形，逐元素对应（每 q-head 自己的选择）
        b_s = torch.where(b_mask, b_s, float('-inf'))
    else:
        b_s = torch.where(b_mask[:, None, :], b_s, float('-inf'))
    # 【B7 修复（kimi3 清单 F12，2026-10-08）】全 False 行防线（消费端）：
    # 任一 head 行全 False 时 softmax 对全 -inf 行产生 NaN 输出。indexer
    # 契约上由 sink/swa 正交强制保证不可达，此为防御性 fail-closed（契约
    # 破损时显式报错优于 NaN 静默扩散）。代价 = 每次调用一次 host 同步
    # （any().all() → bool）；本函数是 HF 参考路径，非 sglang 生产热路径。
    if not bool(b_mask.any(dim=-1).all()):
        raise RuntimeError(
            "eager_decoding_attn: 选择 mask 存在全 False 行——softmax 将"
            "产生 NaN。indexer 契约应保证 sink/swa 强制区使每行至少一个"
            "有效 token；请检查 indexer 的 mask 输出。"
        )
    b_p = F.softmax(b_s, dim=-1).to(q.dtype)
    b_o = rearrange(einsum(b_p, b_v, 'h g t, t h d -> h g d'), 'h g d -> (h g) d')
    o[0] = b_o
    return o.unsqueeze(0)


def _eager_decoding_attn_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    """E127 gather 路径（数学语义 = _dense，仅归约顺序不同）。

    行定义（与 _dense 的 b_mask 一致）：
      共享口径：R = Hkv 行，行 h = kv-head h，组内 G 个 q-head 广播同一选择；
      per_q_head：R = Hkv*G 行，行 (h,g) = q-head g 用 kv-head h。
    W = max 行选中数（一次 host 同步读 min/max，min==0 复用 B7 报错）。
    topk(int32(mask), W) 取每行选中位置（W ≥ 每行 1 的个数 → 全部命中），
    gather 后用 valid 掩掉补齐位（mask 值 0 的位置）。
    """
    if softmax_scale is None:
        softmax_scale = k.shape[-1] ** -0.5
    q, k, v, mask = (x.squeeze(0) for x in (q, k, v, mask))
    G = q.shape[-2] // k.shape[-2]
    bos_k, eos_k = cu_seqlens_k[0], cu_seqlens_k[1]
    m0 = mask[0]
    per_qh = m0.shape[0] != k.shape[-2]
    # 与 _dense 相同的 b_mask 构造（repeat 到 token 级 + 裁真实长度）
    if per_qh:
        g_m = m0.shape[0] // k.shape[-2]
        b_mask = repeat(
            m0, '(h g) tk -> h g (tk bs)', g=g_m, bs=block_size
        )[..., :eos_k - bos_k]
    else:
        b_mask = repeat(mask[0], 'h tk -> h (tk bs)', bs=block_size)[..., :eos_k - bos_k]

    # 统一为 2D [R, T] 行视图
    if per_qh:
        bm2 = b_mask.reshape(-1, b_mask.shape[-1])        # [Hkv*G, T]
        head_of_row = torch.arange(m0.shape[0], device=q.device) // G
    else:
        bm2 = b_mask                                     # [Hkv, T]
        head_of_row = None                              # 行号即 kv-head 号

    T = bm2.shape[-1]
    # 小序列路由：gather 的固定开销（topk + ~15 个小 op + 1 次读回同步）在
    # T < 16K 时得不偿失（S=8K 实测 gather 0.48ms > dense 0.28ms），且 T
    # 是 host 已知量——本分支零同步直接回退，8K 档与 baseline 逐位同路径。
    if T < 16384:
        return _eager_decoding_attn_dense(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            mask.unsqueeze(0), block_size, cu_seqlens_k, softmax_scale)
    counts = bm2.sum(dim=-1)                            # [R]
    # 一次 host 同步取 min/max：min==0 → B7 守卫（与 _dense 同一条报错）
    stats = torch.stack([counts.max(), counts.min()]).cpu()
    W, min_c = int(stats[0]), int(stats[1])
    if min_c == 0:
        raise RuntimeError(
            "eager_decoding_attn: 选择 mask 存在全 False 行——softmax 将"
            "产生 NaN。indexer 契约应保证 sink/swa 强制区使每行至少一个"
            "有效 token；请检查 indexer 的 mask 输出。"
        )
    # 稀疏度门限：gather 版有 topk（O(T)/行）+ 索引/gather 小 op ~15 个的
    # 固定开销，W/T 过高（短序列或稠密 mask 臂）时收益为负——S=8K 实测
    # gather 0.48ms > dense 0.28ms。W*8>T 一律回退稠密（生产 mask 行选中
    # ≈ K2+sink+swa ≈ 1280-1600，S≥16K 时满足 W ≤ T/8）。
    if W * 8 > T:
        return _eager_decoding_attn_dense(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            mask.unsqueeze(0), block_size, cu_seqlens_k, softmax_scale)

    bm_int = bm2.to(torch.int32)
    idx = torch.topk(bm_int, W, dim=-1).indices          # [R, W]
    valid = torch.gather(bm_int, 1, idx).to(torch.bool)  # [R, W] 补齐位 False

    # KV gather：kv_t [Hkv, T, D]；行 → head 映射（共享口径行号即 head 号）
    kv_t = k[bos_k:eos_k].permute(1, 0, 2)               # [Hkv, T, D]
    vv_t = v[bos_k:eos_k].permute(1, 0, 2)
    if head_of_row is None:
        idx3 = idx.unsqueeze(-1).expand(*idx.shape, kv_t.shape[-1])
        gk = torch.gather(kv_t, 1, idx3)                 # [Hkv, W, D]
        gv = torch.gather(vv_t, 1, idx3)
    else:
        h_idx = head_of_row.unsqueeze(-1).expand_as(idx)  # [R, W]
        gk = kv_t[h_idx, idx, :]                         # [R, W, D] 高级索引 gather
        gv = vv_t[h_idx, idx, :]

    if per_qh:
        b_q = (q[0] * softmax_scale).to(torch.float32)   # [H=R, D]
        b_s = einsum(b_q, gk.to(torch.float32), 'r d, r w d -> r w')
        b_s = torch.where(valid, b_s, float('-inf'))
        b_p = F.softmax(b_s, dim=-1).to(q.dtype)
        b_o = einsum(b_p, gv, 'r w, r w d -> r d')       # [H, D]
    else:
        b_q = rearrange(q[0] * softmax_scale, '(h g) d -> h g d', g=G).to(torch.float32)
        b_s = einsum(b_q, gk.to(torch.float32), 'h g d, h w d -> h g w')
        b_s = torch.where(valid[:, None, :], b_s, float('-inf'))
        b_p = F.softmax(b_s, dim=-1).to(q.dtype)
        b_o = rearrange(einsum(b_p, gv, 'h g w, h w d -> h g d'),
                        'h g d -> (h g) d')
    o = q.new_zeros(q.shape[:-1] + (v.shape[-1],))
    o[0] = b_o
    return o.unsqueeze(0)
