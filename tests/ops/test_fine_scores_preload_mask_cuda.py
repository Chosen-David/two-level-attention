"""L2 fine-score load 前过滤的逐位与越界读回归。"""
import unittest

import torch
import triton
import triton.language as tl

from sparse_attn.indexer.nfa_layer_batch_selector import (
    _nfa_fine_scores_paged_kernel,
)


@triton.jit
def _reference_fine_scores_paged_kernel(
    KV_PTRS, BLOCK_TABLE, VALID_REQ, DIMS, Q, CAND, KEEP, OWNERS,
    GENERATIONS, LOGITS, LOW, HIGH, QUOTA, KV_REF, R,
    stride_qm, stride_qh, stride_ch, stride_cr,
    S: tl.constexpr, MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
    HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
    D_BLOCK: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
    TILE: tl.constexpr, MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    TOKEN_BITS: tl.constexpr,
):
    """修改前实现，保留为独立 bit-exact oracle。"""
    row = tl.program_id(0)
    t = tl.program_id(1)
    m = row // HKV
    kv = row % HKV
    layer = m // B
    req = m % B
    req_id = tl.load(VALID_REQ + req).to(tl.int32)
    kv_addr = tl.load(KV_PTRS + layer)
    kv_ptr = kv_addr.to(tl.pointer_type(KV_REF.dtype.element_ty))
    offs_d = tl.arange(0, D_BLOCK)
    dim_mask = offs_d < DIM_NUMEL
    dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0).to(tl.int64)
    offs_r = t * TILE + tl.arange(0, TILE)
    mask_r = offs_r < R
    keep = tl.load(KEEP + row * stride_ch + offs_r * stride_cr,
                   mask=mask_r, other=False)
    read_r = mask_r & keep
    cand = tl.load(CAND + row * stride_ch + offs_r * stride_cr,
                   mask=read_r, other=0)
    in_range = (cand >= 0) & (cand < S)
    owner = tl.load(OWNERS + row * S + cand,
                    mask=read_r & in_range, other=-1)
    generation = tl.load(GENERATIONS + row)
    unique = read_r & in_range & (owner == generation * R + offs_r)
    safe = tl.where(read_r, cand, 0)
    lp = safe // PAGE
    off = safe % PAGE
    lp_valid = lp < MAX_BLOCKS
    phys = tl.load(
        BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
        + req_id * MAX_BLOCKS + lp,
        mask=read_r & lp_valid, other=0).to(tl.int64)
    src = (kv_ptr + phys[:, None] * HK * PAGE * HEAD_DIM
           + kv * PAGE * HEAD_DIM + off[:, None] * HEAD_DIM + dim[None, :])
    k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :],
                other=0.0)
    q = tl.load(Q + m * stride_qm + kv * stride_qh + offs_d,
                mask=dim_mask, other=0.0)
    score = tl.sum(k.to(tl.float32) * q[None, :], axis=1)
    score_bf16 = score.to(tl.bfloat16)
    score = score_bf16.to(tl.float32)
    good = (unique & lp_valid & (cand >= tl.load(LOW + m))
            & (cand < tl.load(HIGH + m)) & (tl.load(QUOTA + m) > 0)
            & (score == score))
    score = tl.where(good, score, float("-inf"))
    bits = score_bf16.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    ordered = tl.where(negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
    packed_good = good & (score != float("-inf"))
    delta = S - cand.to(tl.int32)
    invalid = -32768 << TOKEN_BITS
    key = tl.where(packed_good, (ordered << TOKEN_BITS) + delta, invalid)
    tl.store(LOGITS + row * R + offs_r, key, mask=mask_r)


class FineScoresPreloadMaskTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_bit_exact_and_discarded_pages_are_not_read(self):
        m, hkv, r = 8, 1, 32
        capacity, page, max_blocks, dim = 512, 64, 8, 32
        device = "cuda"
        torch.manual_seed(20261107)
        cand = torch.randint(64, 448, (m, hkv, r), dtype=torch.int64)
        keep = torch.ones((m, hkv, r), dtype=torch.bool)
        for row in range(m):
            cand[row, 0, :12] = torch.tensor(
                [0, 1, 2, 3, 100, 100, 120, 130, 480, 481, 500, 700])
            keep[row, 0, 15] = False
        owners = torch.zeros((m, capacity), dtype=torch.int32)
        for row in range(m):
            for slot in range(r):
                token = int(cand[row, 0, slot])
                if bool(keep[row, 0, slot]) and 0 <= token < capacity:
                    owners[row, token] = r + slot
        low = torch.full((m,), 64, dtype=torch.int64)
        high = torch.full((m,), 448, dtype=torch.int64)
        generations = torch.ones((m, hkv), dtype=torch.int32)
        valid_req = torch.arange(m, dtype=torch.int64)
        block_table = torch.arange(
            m * max_blocks, dtype=torch.int32).reshape(m, max_blocks)
        keys = torch.randn(
            m * max_blocks * page * dim, dtype=torch.bfloat16, device=device)
        query = torch.randn((m, hkv, dim), dtype=torch.bfloat16, device=device)
        dims = torch.arange(dim, dtype=torch.int64)

        def run(kernel, table, quota):
            logits = torch.empty((m, hkv, r), dtype=torch.int32, device=device)
            kv_ptrs = torch.tensor([keys.data_ptr()], dtype=torch.int64, device=device)
            kernel[(m * hkv, triton.cdiv(r, 8))](
                kv_ptrs, table.to(device), valid_req.to(device), dims.to(device),
                query.to(device), cand.to(device), keep.to(device), owners.to(device),
                generations.to(device), logits, low.to(device), high.to(device),
                quota.to(device), keys.to(device), r,
                stride_qm=query.stride(0), stride_qh=query.stride(1),
                stride_ch=cand.stride(1), stride_cr=cand.stride(2),
                S=capacity, MAX_BLOCKS=max_blocks, PAGE=page,
                HK=1, HEAD_DIM=dim, DIM_NUMEL=dim, D_BLOCK=dim,
                HKV=hkv, B=m, TILE=8, MAX_REQ=m, LAYER_OFFSET=0,
                TOKEN_BITS=16, num_warps=4)
            torch.cuda.synchronize()
            return logits

        quota = torch.full((m,), 5, dtype=torch.int64)
        reference = run(_reference_fine_scores_paged_kernel, block_table, quota)
        actual = run(_nfa_fine_scores_paged_kernel, block_table, quota)
        self.assertTrue(torch.equal(reference, actual), "packed logits 不一致")

        discarded = ~((cand >= 64) & (cand < 448))
        poisoned = block_table.clone()
        for row in range(m):
            pages = set((cand[row][discarded[row]] // page).tolist())
            for logical_page in pages:
                if 0 <= logical_page < max_blocks:
                    poisoned[row, logical_page] = 0x3FFFFFF
        masked = run(_nfa_fine_scores_paged_kernel, poisoned, quota)
        self.assertTrue(torch.equal(actual, masked), "丢弃候选仍影响输出")

        zero_quota = torch.zeros_like(quota)
        zero_reference = run(
            _reference_fine_scores_paged_kernel, block_table, zero_quota)
        zero_masked = run(
            _nfa_fine_scores_paged_kernel,
            torch.full_like(block_table, 0x3FFFFFF), zero_quota)
        self.assertTrue(torch.equal(zero_reference, zero_masked), "quota=0 仍读 K")


if __name__ == "__main__":
    unittest.main()
