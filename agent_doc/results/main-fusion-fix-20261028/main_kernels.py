import triton
import triton.language as tl
@triton.jit
def _iter3_compact_kernel(CIDX, KEEP, MIDM, CIDX_OUT, KEEP_OUT, MID_OUT, CLEN, R_cap, S_CAP_M1: tl.constexpr, BLOCK: tl.constexpr):
    """iter3b：行内稳定压缩。每 program 一行（kv head）：
        pass1 算 keep 总数 total；pass2 keep=True 槽写前缀目的位
        （carry+cumsum-1）、False 槽写 total+序数目的位（尾部保序），
        同步输出 mid 值（MIDM[idx] 随槽位置换）——与
        argsort(~keep,stable)+gather×2+sum+mid gather 逐位一致
        （含 garbage 区内容保序复刻）。纯整数/布尔域。
        【iter3f-3】哨兵重定向内联：keep=False 槽写 S_CAP_M1 而非原值
        （替代 host 侧 torch.where(keep2, cand_idx2, S_cap-1)），
        省 1 kernel/层 + [Hkv,R_cap] int64 读+写。"""
    r = tl.program_id(0)
    base = r * R_cap
    total = tl.zeros((), tl.int32)
    for c0 in range(0, R_cap, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        m = offs < R_cap
        k = tl.load(KEEP + base + offs, mask=m, other=0).to(tl.int32)
        total += tl.sum(k, axis=0)
    carry = tl.zeros((), tl.int32)
    gcarry = tl.zeros((), tl.int32)
    for c0 in range(0, R_cap, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        m = offs < R_cap
        k = tl.load(KEEP + base + offs, mask=m, other=0).to(tl.int32)
        idx = tl.load(CIDX + base + offs, mask=m, other=0)
        mid = tl.load(MIDM + idx, mask=m, other=0)
        ksum = tl.cumsum(k, axis=0)
        gsum = tl.cumsum(1 - k, axis=0)
        kp = carry + ksum - 1
        gp = total + gcarry + gsum - 1
        pos = tl.where(k != 0, kp, gp)
        w = m
        sidx = tl.where(k != 0, idx, S_CAP_M1)
        tl.store(CIDX_OUT + base + pos, sidx, mask=w)
        tl.store(KEEP_OUT + base + pos, k != 0, mask=w)
        tl.store(MID_OUT + base + pos, mid != 0, mask=w)
        carry += tl.sum(k, axis=0)
        gcarry += tl.sum(1 - k, axis=0)
    tl.store(CLEN + r, total.to(tl.int64))

@triton.jit
def _iter3_fused_softmax_kernel(SFC, KEEP2, MID_SLOT, OUT_P, R_CAP: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    """iter3e+iter3f-4：fine 门控 + softmax + GQA mean + mid 门控五合一。

        输入 SFC [Hkv*group_size, R_CAP]、KEEP2 [Hkv, R_CAP]（head 共享）、
        MID_SLOT [Hkv, R_CAP]（mid 域布尔），
        输出 OUT_P [Hkv, R_CAP] = mid_slot ? mean_over_group(softmax(gated_sfc)) : -inf。

        【iter3f-4】mid 门控内联：~mid_slot | ~keep2 → masked_fill(-inf)
        由 kernel 在写输出前逐元素应用。全 -inf kv_head 行 → max_val=-inf
        → 输出全 -inf；mid=True 行正常输出 softmax prob。
        节省 3 kernel/层（mid_slot 计算 | ~mid_slot|~keep2 | masked_fill）。"""
    kv = tl.program_id(0)
    col = tl.program_id(1)
    ninf = float('-inf')
    offs = col * BLOCK + tl.arange(0, BLOCK)
    m = offs < R_CAP
    base_keep = kv * R_CAP
    k_row = tl.load(KEEP2 + base_keep + offs, mask=m, other=0) != 0
    mid_row = tl.load(MID_SLOT + base_keep + offs, mask=m, other=0) != 0
    gmax = ninf
    for g in range(GROUP_SIZE):
        h = kv * GROUP_SIZE + g
        s = tl.load(SFC + h * R_CAP + offs, mask=m, other=ninf).to(tl.float32)
        s_g = tl.where(k_row, s, ninf)
        gmax = tl.maximum(gmax, tl.max(s_g, axis=0))
    if gmax == ninf:
        tl.store(OUT_P + kv * R_CAP + offs, tl.full((BLOCK,), ninf, dtype=tl.float32), mask=m)
        return
    mean_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for g in range(GROUP_SIZE):
        h = kv * GROUP_SIZE + g
        s = tl.load(SFC + h * R_CAP + offs, mask=m, other=ninf).to(tl.float32)
        s_g = tl.where(k_row, s, ninf)
        e_g = tl.where(k_row, tl.exp(s_g - gmax), 0.0)
        row_sum = tl.sum(e_g, axis=0)
        p_norm = tl.where(row_sum > 0, e_g / row_sum, 0.0)
        mean_acc += p_norm
    mean_acc /= GROUP_SIZE
    mean_acc = tl.where(mid_row, mean_acc, ninf)
    tl.store(OUT_P + kv * R_CAP + offs, mean_acc, mask=m)
