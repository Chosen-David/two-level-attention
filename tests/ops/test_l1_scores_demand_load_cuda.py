"""L1 按候选池读取统计的逐位与越界读回归。"""
import unittest

import torch
import triton
import triton.language as tl

from sparse_attn.indexer.nfa_layer_batch_selector import _nfa_l1_scores_kernel


@triton.jit
def _reference_l1_scores_kernel(
    KMIN, KMAX, KSUM, Q, LENGTHS, ACTIVE_REQ, SKIP_FAR, SKIP_NEAR,
    VALID_REQ, STATS_LEN, FAR_OUT, NEAR_OUT, T,
    stride_kt, stride_kh, stride_kt2, stride_kh2,
    stride_kts, stride_khs, stride_qh,
    PAGE: tl.constexpr, BS: tl.constexpr,
    SINK: tl.constexpr, NEAR_LEN: tl.constexpr, WINDOW: tl.constexpr,
    POOL_NEAR_LEN: tl.constexpr, SINK_POOL_LO: tl.constexpr,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    B: tl.constexpr, STRIDE_KB: tl.constexpr, STRIDE_KB2: tl.constexpr,
    STRIDE_KS: tl.constexpr, STRIDE_KR: tl.constexpr,
    STRIDE_KR2: tl.constexpr, STRIDE_KRS: tl.constexpr,
    HAS_AVG: tl.constexpr, FAR_AVG: tl.constexpr, NEAR_AVG: tl.constexpr,
    MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
):
    """修改前实现，保留为独立 bit-exact oracle。"""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    batch = pid_h // Hkv
    kv = pid_h % Hkv
    layer = batch // B
    req = batch % B
    req_id = tl.load(VALID_REQ + req).to(tl.int32)
    length = tl.load(LENGTHS + req).to(tl.int32)
    if (pid_t == 0) & (kv == 0):
        if tl.load(ACTIVE_REQ + req) >= 0:
            tl.store(STATS_LEN + (layer + LAYER_OFFSET) * MAX_REQ + req_id,
                     length)
    skip_far = tl.load(SKIP_FAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
    skip_near = tl.load(SKIP_NEAR + (layer + LAYER_OFFSET) * MAX_REQ + req_id).to(tl.int32)
    far_on = skip_far == 0
    near_on = skip_near == 0
    sink_hi = tl.minimum(SINK, length)
    swa_hi = tl.maximum(sink_hi, length - WINDOW)
    far_blk = (length - WINDOW - NEAR_LEN) // PAGE
    orig_far_hi = tl.minimum(swa_hi, tl.maximum(sink_hi, far_blk * PAGE))
    offs_t = pid_t * BT + tl.arange(0, BT)
    mask_t = offs_t < T
    block_lo = offs_t * BS
    block_hi = block_lo + BS
    has_sink = block_lo < sink_hi
    has_far = far_on & (block_hi > sink_hi) & (block_lo < orig_far_hi)
    has_near = near_on & (block_hi > orig_far_hi) & (block_lo < swa_hi)
    has_swa = (block_hi > swa_hi) & (block_lo < length)
    block_valid = (block_lo < length) & (
        has_sink | has_far | has_near | has_swa)
    near_blks = tl.maximum(
        (length - WINDOW - POOL_NEAR_LEN) // BS, SINK_POOL_LO)
    kt_eff = (length + BS - 1) // BS
    swa_lo_blk = tl.maximum(near_blks, kt_eff - tl.maximum(1, WINDOW // BS))
    swa_lo_tok = tl.maximum(length - WINDOW, 0)
    swa_byp_blk = swa_lo_tok // BS
    swa_byp_cond = ((swa_lo_tok % BS != 0) & (swa_byp_blk >= SINK_POOL_LO))
    if SINK % BS != 0:
        swa_byp_cond = swa_byp_cond & (swa_byp_blk != SINK_POOL_LO)
    far_pool = ((offs_t >= SINK_POOL_LO) & (offs_t < near_blks) & far_on)
    near_pool = ((offs_t >= near_blks) & (offs_t < swa_lo_blk) & near_on
                 & ~((offs_t == swa_byp_blk) & swa_byp_cond))
    if SINK % BS != 0:
        far_pool = far_pool & (offs_t != SINK_POOL_LO)
        near_pool = near_pool & (offs_t != SINK_POOL_LO)
    far_pool = far_pool & block_valid
    near_pool = near_pool & block_valid
    KMIN += layer * STRIDE_KB + req_id * STRIDE_KR
    KMAX += layer * STRIDE_KB2 + req_id * STRIDE_KR2
    if HAS_AVG:
        KSUM += layer * STRIDE_KS + req_id * STRIDE_KRS
    offs_d = tl.arange(0, D)
    kmin = tl.load(KMIN + offs_t[:, None] * stride_kt + kv * stride_kh
                   + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.float32)
    kmax = tl.load(KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2
                   + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.float32)
    if HAS_AVG:
        ksum = tl.load(KSUM + offs_t[:, None] * stride_kts + kv * stride_khs
                       + offs_d[None, :], mask=mask_t[:, None], other=0.0)
        kavg = (ksum.to(tl.float32) / BS).to(
            FAR_OUT.dtype.element_ty).to(tl.float32)
    mm_acc = tl.zeros((BT,), dtype=tl.float32)
    avg_acc = tl.zeros((BT,), dtype=tl.float32)
    for j in tl.static_range(G):
        h = pid_h.to(tl.int64) * G + j
        q = tl.load(Q + h * stride_qh + offs_d)
        q_low = tl.minimum(q, 0.0)
        q_high = tl.maximum(q, 0.0)
        mm = (tl.sum(kmin * q_low[None, :], axis=1)
              + tl.sum(kmax * q_high[None, :], axis=1))
        mm_acc += mm.to(FAR_OUT.dtype.element_ty).to(tl.float32)
        if HAS_AVG:
            avg = tl.sum(kavg * q[None, :], axis=1)
            avg_acc += avg.to(FAR_OUT.dtype.element_ty).to(tl.float32)
    mm_score = (mm_acc / G).to(FAR_OUT.dtype.element_ty)
    far_score = mm_score
    near_score = mm_score
    if HAS_AVG:
        avg_score = (avg_acc / G).to(FAR_OUT.dtype.element_ty)
        if FAR_AVG:
            far_score = avg_score
        if NEAR_AVG:
            near_score = avg_score
    far_score = tl.where(far_pool, far_score, float("-inf"))
    near_score = tl.where(near_pool, near_score, float("-inf"))
    out_offs = pid_h.to(tl.int64) * T + offs_t
    tl.store(FAR_OUT + out_offs, far_score, mask=mask_t)
    tl.store(NEAR_OUT + out_offs, near_score, mask=mask_t)


class L1ScoresDemandLoadTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_bit_exact_all_methods_and_skip_states(self):
        layers, batch, hkv, group, d = 2, 2, 2, 2, 32
        t, bs, bt = 80, 16, 16
        shape = (layers, batch, t, hkv, d)
        device = "cuda"
        torch.manual_seed(20261107)
        kmin = torch.randn(shape, dtype=torch.bfloat16, device=device)
        kmax = torch.randn(shape, dtype=torch.bfloat16, device=device)
        ksum = torch.randn(shape, dtype=torch.bfloat16, device=device)
        kmin[0, 0, 3, 0, 0] = float("nan")
        kmax[0, 1, 7, 1, 1] = float("inf")
        ksum[1, 0, 9, 0, 2] = float("-inf")
        q = torch.randn((layers * batch * hkv * group, d),
                        dtype=torch.bfloat16, device=device)
        q[0, 3] = float("nan")
        lengths = torch.tensor([1280, 1003], dtype=torch.int64, device=device)
        active_req = torch.arange(batch, dtype=torch.int64, device=device)
        valid_req = torch.tensor([1, 0], dtype=torch.int64, device=device)
        # valid_req=[1,0] 反解 request slot；四个 layer/request 组合
        # 分别覆盖不跳、far-only、near-only、both-skip。
        skip_far = torch.tensor([1, 0, 1, 0], dtype=torch.bool, device=device)
        skip_near = torch.tensor([0, 0, 1, 1], dtype=torch.bool, device=device)

        def run(kernel, far_avg, near_avg, min_arg=None, max_arg=None):
            far = torch.full((layers * batch, hkv, t), 13.25,
                             dtype=torch.bfloat16, device=device)
            near = torch.full_like(far, 13.25)
            stats_len = torch.zeros((layers, batch), dtype=torch.int64, device=device)
            has_avg = far_avg or near_avg
            sum_arg = ksum if has_avg else kmin
            min_arg = kmin if min_arg is None else min_arg
            max_arg = kmax if max_arg is None else max_arg
            kernel[(triton.cdiv(t, bt), layers * batch * hkv)](
                min_arg, max_arg, sum_arg, q, lengths, active_req,
                skip_far, skip_near, valid_req, stats_len, far, near, t,
                kmin.stride(2), kmin.stride(3), kmax.stride(2), kmax.stride(3),
                sum_arg.stride(2) if has_avg else kmin.stride(2),
                sum_arg.stride(3) if has_avg else kmin.stride(3), q.stride(0),
                PAGE=bs, BS=bs, SINK=4, NEAR_LEN=256, WINDOW=64,
                POOL_NEAR_LEN=256, SINK_POOL_LO=0, Hkv=hkv, G=group,
                D=d, BT=bt, B=batch, STRIDE_KB=kmin.stride(0),
                STRIDE_KB2=kmax.stride(0), STRIDE_KS=sum_arg.stride(0),
                STRIDE_KR=kmin.stride(1), STRIDE_KR2=kmax.stride(1),
                STRIDE_KRS=sum_arg.stride(1), HAS_AVG=has_avg,
                FAR_AVG=far_avg, NEAR_AVG=near_avg,
                MAX_REQ=batch, LAYER_OFFSET=0, num_warps=4)
            torch.cuda.synchronize()
            return far, near, stats_len

        def assert_active_and_poison_tail(reference, actual):
            lengths_host = lengths.cpu().tolist()
            for row in range(layers * batch):
                active = triton.cdiv(lengths_host[row % batch], bs)
                for ref, out in zip(reference[:2], actual[:2]):
                    torch.testing.assert_close(
                        ref[row, :, :active].view(torch.int16),
                        out[row, :, :active].view(torch.int16),
                        rtol=0, atol=0)
                    if active < t:
                        torch.testing.assert_close(
                            out[row, :, active:],
                            torch.full_like(out[row, :, active:], 13.25),
                            rtol=0, atol=0)
            torch.testing.assert_close(reference[2], actual[2], rtol=0, atol=0)

        for far_avg, near_avg in ((False, True), (False, False),
                                  (True, False), (True, True)):
            with self.subTest(far_avg=far_avg, near_avg=near_avg):
                reference = run(_reference_l1_scores_kernel, far_avg, near_avg)
                actual = run(_nfa_l1_scores_kernel, far_avg, near_avg)
                assert_active_and_poison_tail(reference, actual)

        # both-avg 完全不消费 Min/Max；用单元素存储配合原大 stride，
        # 任何遗漏的 Min/Max load 都会越界并使本测试失败。
        tiny = torch.empty(1, dtype=torch.bfloat16, device=device)
        reference = run(_reference_l1_scores_kernel, True, True)
        actual = run(_nfa_l1_scores_kernel, True, True, tiny, tiny)
        assert_active_and_poison_tail(reference, actual)


if __name__ == "__main__":
    unittest.main()
