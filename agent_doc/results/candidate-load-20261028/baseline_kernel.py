import triton
import triton.language as tl
@triton.jit
def _score_fine_cand_kernel(
    KQ, Q, CAND, CLEN, OUT,
    # KQ [T,Hkv,D] bf16；Q [H,D] fp32（已乘 scale）；CAND [Hkv,R] int64；
    # CLEN [Hkv] int64；OUT [H,R] out_dtype
    R, stride_kt, stride_kh, stride_qh, stride_ch, stride_cr,
    stride_oh, stride_or,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
):
    """【E156c】候选 fine 打分融合 kernel：每 program 处理 (候选列块 × 1
    kv 头)，cand_idx gather k 行 [BT,D] 载入寄存器后循环 g 个 q 头逐头
    点积（K 读带宽 /g，同 _score_fine_kernel）；-inf 门控（r ≥ CLEN[kv]
    的 pad 列）在 kernel 内完成，替代 einsum 路径的 where。

    数学：score[h, r] = Σ_d b_q_full[h, d]·kq[CAND[kv, r], kv, d]，与
    einsum(b_q_full, kq[CAND] 展开) 数学等价；fp32 累加顺序与 einsum
    不同 → bf16 舍入后 ULP 级漂移（与 _fast_score_fine 同口径）。
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv = pid_h
    offs_r = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    mask_r = offs_r < R
    cand = tl.load(CAND + kv * stride_ch + offs_r * stride_cr,
                   mask=mask_r, other=0)                     # [BT] int64
    k = tl.load(KQ + cand[:, None] * stride_kt + kv * stride_kh
                + offs_d[None, :],
                mask=mask_r[:, None], other=0.0).to(tl.float32)   # [BT, D]
    cl = tl.load(CLEN + kv)
    for j in tl.static_range(G):
        h = kv * G + j
        q = tl.load(Q + h * stride_qh + offs_d)              # [D] fp32
        s = tl.sum(k * q[None, :], axis=1)                   # [BT] fp32
        res = tl.where(offs_r < cl, s, float("-inf"))
        tl.store(OUT + h * stride_oh + offs_r * stride_or,
                 res.to(OUT.dtype.element_ty), mask=mask_r)

