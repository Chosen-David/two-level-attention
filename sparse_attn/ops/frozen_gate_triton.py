"""Fused online-softmax/LSE profiling for frozen near/far regions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _frozen_region_profile_kernel(
        Q, K, FAR_HI, NEAR_HI, OUT,
        length, start, n_rows, sink, scale,
        heads, group,
        q_stride_h, q_stride_l, q_stride_d,
        k_stride_h, k_stride_l, k_stride_d,
        BLOCK_R: tl.constexpr, GROUP: tl.constexpr,
        BLOCK_K: tl.constexpr, DIM: tl.constexpr):
    """One QK pass produces total, far and near online-softmax sums.

    Grid is (ceil(tail_rows / BLOCK_R), kv_heads).  GQA query heads are
    folded into the row tile; no [rows, length] score/probability tensor is
    materialized.  OUT has layout [2, heads, tail_rows] (near, then far).
    """
    row_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    flat_rows = tl.arange(0, BLOCK_R * GROUP)
    local_rows = flat_rows // GROUP
    group_heads = flat_rows % GROUP
    rows = row_tile * BLOCK_R + local_rows
    row_mask = rows < n_rows
    query_heads = kv_head * GROUP + group_heads

    dims = tl.arange(0, DIM)
    q_ptrs = (Q + query_heads[:, None] * q_stride_h
              + (start + rows)[:, None] * q_stride_l
              + dims[None, :] * q_stride_d)
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    qlen = start + rows + 1
    sink_hi = tl.minimum(qlen, sink)
    far_hi = tl.load(FAR_HI + rows, mask=row_mask, other=0)
    near_hi = tl.load(NEAR_HI + rows, mask=row_mask, other=0)

    m_i = tl.full([BLOCK_R * GROUP], float('-inf'), tl.float32)
    total = tl.zeros([BLOCK_R * GROUP], tl.float32)
    far = tl.zeros([BLOCK_R * GROUP], tl.float32)
    near = tl.zeros([BLOCK_R * GROUP], tl.float32)

    for key_start in range(0, length, BLOCK_K):
        cols = key_start + tl.arange(0, BLOCK_K)
        col_mask = cols < length
        k_ptrs = (K + kv_head * k_stride_h + cols[None, :] * k_stride_l
                  + dims[:, None] * k_stride_d)
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.0)
        scores = tl.dot(q, k) * scale
        valid = row_mask[:, None] & col_mask[None, :] \
            & (cols[None, :] < qlen[:, None])
        scores = tl.where(valid, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        probs = tl.exp(scores - m_safe[:, None])
        far_mask = (cols[None, :] >= sink_hi[:, None]) \
            & (cols[None, :] < far_hi[:, None])
        near_mask = (cols[None, :] >= far_hi[:, None]) \
            & (cols[None, :] < near_hi[:, None])
        total = total * alpha + tl.sum(probs, axis=1)
        far = far * alpha + tl.sum(tl.where(far_mask, probs, 0.0), axis=1)
        near = near * alpha + tl.sum(tl.where(near_mask, probs, 0.0), axis=1)
        m_i = m_new

    # Causal self-attention guarantees total > 0 for every valid row.
    far_mass = far / total
    near_mass = near / total
    out_offsets = query_heads * n_rows + rows
    tl.store(OUT + heads * n_rows + out_offsets, far_mass, mask=row_mask)
    tl.store(OUT + out_offsets, near_mass, mask=row_mask)


def fused_region_profile(q, k, geometry, rows, scale):
    """Return per-query head-averaged near/far masses on CUDA."""
    length = q.shape[2]
    n = min(rows, length)
    heads = q.shape[1]
    kv_heads = k.shape[1]
    group = heads // kv_heads
    start = length - n
    device = q.device

    # Reuse the exact legacy integer boundary semantics outside the kernel.
    qlen = torch.arange(start + 1, length + 1, device=device)[:, None]
    sink_hi = torch.minimum(qlen, torch.tensor(geometry.sink, device=device))
    near_hi = torch.maximum(sink_hi, qlen - geometry.window)
    far_hi = torch.minimum(near_hi, torch.maximum(
        sink_hi, ((qlen - geometry.window - geometry.near) // geometry.page)
        * geometry.page))
    far_hi = far_hi[:, 0].contiguous()
    near_hi = near_hi[:, 0].contiguous()
    output = torch.empty((2, heads, n), dtype=torch.float32, device=device)

    block_r = 16
    _frozen_region_profile_kernel[(triton.cdiv(n, block_r), kv_heads)](
        q, k, far_hi, near_hi, output,
        length, start, n, geometry.sink, scale,
        heads, group,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        BLOCK_R=block_r, GROUP=group, BLOCK_K=64,
        DIM=triton.next_power_of_2(q.shape[-1]),
        num_warps=4, num_stages=2,
    )
    return output.mean(dim=1)
