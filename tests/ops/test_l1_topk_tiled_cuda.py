"""长行 L1 局部 Top-K 与原整行实现的逐位一致性回归。"""
import unittest

import torch
import triton
import triton.language as tl

from sparse_attn.indexer.nfa_layer_batch_selector import (
    _candidate_expand_batched_kernel,
    _candidate_pack_batched_kernel,
    _l1_topk_partial_kernel,
    _topk_candidate_batched_kernel,
    _topk_candidate_prepare_kernel,
    _topk_indices_body,
    _topk_indices_merge_body,
)


@triton.jit
def _reference_topk(
    FAR, NEAR, I_F, I_N,
    KT: tl.constexpr, KT_PAD: tl.constexpr,
    K_F: tl.constexpr, K_N: tl.constexpr,
    K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
):
    _topk_indices_body(FAR, I_F, KT=KT, KT_PAD=KT_PAD, K=K_F, K_PAD=K_F_PAD)
    _topk_indices_body(NEAR, I_N, KT=KT, KT_PAD=KT_PAD, K=K_N, K_PAD=K_N_PAD)


@triton.jit
def _merge_topk(
    FAR, NEAR, I_F, I_N,
    KT: tl.constexpr, NTILES: tl.constexpr, NTILES_PAD: tl.constexpr,
    K_F: tl.constexpr, K_N: tl.constexpr,
    K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
    LOG2_F: tl.constexpr, LOG2_N: tl.constexpr,
):
    _topk_indices_merge_body(
        FAR, I_F, KT=KT, NTILES=NTILES, NTILES_PAD=NTILES_PAD,
        K=K_F, K_PAD=K_F_PAD, LOG2_K=LOG2_F)
    _topk_indices_merge_body(
        NEAR, I_N, KT=KT, NTILES=NTILES, NTILES_PAD=NTILES_PAD,
        K=K_N, K_PAD=K_N_PAD, LOG2_K=LOG2_N)


@triton.jit
def _reference_candidate(
    FAR, NEAR, I_F, I_N, LENGTHS, SKIP_FAR, SKIP_NEAR, VALID_REQ,
    OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
    KT: tl.constexpr, KT_PAD: tl.constexpr,
    K_F: tl.constexpr, K_N: tl.constexpr,
    K_F_PAD: tl.constexpr, K_N_PAD: tl.constexpr,
    R: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr,
    SINK: tl.constexpr, SWA: tl.constexpr, CAP: tl.constexpr,
    SINK_POOL_LO: tl.constexpr, BS: tl.constexpr, PAGE: tl.constexpr,
    NEAR_LEN: tl.constexpr, K2: tl.constexpr,
    HKV: tl.constexpr, B: tl.constexpr,
    MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
):
    """skip-aware 修改前的整行 Top-K + candidate，作为独立 oracle。"""
    _topk_indices_body(FAR, I_F, KT=KT, KT_PAD=KT_PAD, K=K_F, K_PAD=K_F_PAD)
    _topk_indices_body(NEAR, I_N, KT=KT, KT_PAD=KT_PAD, K=K_N, K_PAD=K_N_PAD)
    tl.debug_barrier()
    _candidate_pack_batched_kernel(
        I_F, I_N, FAR, NEAR, LENGTHS, SKIP_FAR, SKIP_NEAR, VALID_REQ,
        OUT, VALID, LOWER, UPPER, QUOTA, OWNERS, GENERATIONS,
        K_F=K_F, K_N=K_N, KT=KT, R=R, HEAD=HEAD, TAIL=TAIL,
        SINK=SINK, SWA=SWA, CAP=CAP, SINK_POOL_LO=SINK_POOL_LO,
        BS=BS, PAGE=PAGE, NEAR_LEN=NEAR_LEN, K2=K2, HKV=HKV, B=B,
        MAX_REQ=MAX_REQ, LAYER_OFFSET=LAYER_OFFSET)


class L1TopKTiledTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_exact_indices_with_ties_and_nonfinite_scores(self):
        rows = 8
        for seed, (kt, k_f, k_n) in enumerate((
            (513, 96, 32),
            (959, 96, 32),
            (1025, 96, 32),
            (1453, 96, 32),
            (1987, 96, 32),
            (513, 1, 0),
            (1987, 128, 64),
        )):
            with self.subTest(kt=kt, k_f=k_f, k_n=k_n):
                torch.manual_seed(20261105 + seed)
                shape = (rows, kt)
                far = (torch.randint(-32, 33, shape, device="cuda")
                       .to(torch.bfloat16) / 8)
                near = (torch.randint(-32, 33, shape, device="cuda")
                        .to(torch.bfloat16) / 8)
                far[0, 0] = float("nan")
                far[1, 1] = float("-inf")
                near[2, 2] = float("inf")
                near[3, -1] = float("nan")
                ref_f = torch.empty((rows, k_f), dtype=torch.int64, device="cuda")
                ref_n = torch.empty((rows, k_n), dtype=torch.int64, device="cuda")
                out_f = torch.empty_like(ref_f)
                out_n = torch.empty_like(ref_n)
                k_f_pad = triton.next_power_of_2(max(1, k_f))
                k_n_pad = triton.next_power_of_2(max(1, k_n))
                ntiles = triton.cdiv(kt, 512)
                skip_far = torch.zeros(1, dtype=torch.bool, device="cuda")
                skip_near = torch.zeros_like(skip_far)
                valid_req = torch.zeros(1, dtype=torch.int64, device="cuda")
                _reference_topk[(rows,)](
                    far, near, ref_f, ref_n,
                    KT=kt, KT_PAD=triton.next_power_of_2(kt),
                    K_F=k_f, K_N=k_n, K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                    num_warps=8)
                partial_f = torch.empty(
                    (rows, ntiles, k_f_pad), dtype=torch.int32, device="cuda")
                partial_n = torch.empty(
                    (rows, ntiles, k_n_pad), dtype=torch.int32, device="cuda")
                _l1_topk_partial_kernel[(rows, ntiles)](
                    far, near, partial_f, partial_n,
                    skip_far, skip_near, valid_req,
                    KT=kt, TILE=512, NTILES=ntiles,
                    K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                    HKV=rows, B=1, MAX_REQ=1, LAYER_OFFSET=0, num_warps=8)
                _merge_topk[(rows,)](
                    partial_f, partial_n, out_f, out_n,
                    KT=kt, NTILES=ntiles,
                    NTILES_PAD=triton.next_power_of_2(ntiles),
                    K_F=k_f, K_N=k_n, K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                    LOG2_F=k_f_pad.bit_length() - 1,
                    LOG2_N=k_n_pad.bit_length() - 1, num_warps=8)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(ref_f, out_f), "far 索引或顺序不同")
                self.assertTrue(torch.equal(ref_n, out_n), "near 索引或顺序不同")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_fused_candidate_preserves_original_score_inputs(self):
        """tiled Top-K 的 int32 partial 不得覆盖后续 BF16 分数指针。"""
        rows, hkv, kt = 8, 8, 513
        k_f, k_n, bs = 96, 32, 64
        sink, swa = 4, 128
        capacity = kt * bs
        torch.manual_seed(20261106)
        far = torch.randn((rows, kt), device="cuda", dtype=torch.bfloat16)
        near = torch.randn((rows, kt), device="cuda", dtype=torch.bfloat16)
        far[0, 0] = float("nan")
        near[1, 1] = float("-inf")
        lengths = torch.tensor([capacity], dtype=torch.int64, device="cuda")
        skip_far = torch.zeros(1, dtype=torch.bool, device="cuda")
        skip_near = torch.zeros_like(skip_far)
        valid_req = torch.zeros(1, dtype=torch.int64, device="cuda")
        k_f_pad = triton.next_power_of_2(k_f)
        k_n_pad = triton.next_power_of_2(k_n)
        ntiles = triton.cdiv(kt, 512)
        r = (k_f + k_n + 2) * bs
        width = r + sink + swa

        def allocate():
            return {
                "i_f": torch.empty((rows, k_f), dtype=torch.int64, device="cuda"),
                "i_n": torch.empty((rows, k_n), dtype=torch.int64, device="cuda"),
                "out": torch.empty((rows, width), dtype=torch.int64, device="cuda"),
                "valid": torch.empty((rows, width), dtype=torch.bool, device="cuda"),
                "lower": torch.empty(1, dtype=torch.int64, device="cuda"),
                "upper": torch.empty(1, dtype=torch.int64, device="cuda"),
                "quota": torch.empty(1, dtype=torch.int64, device="cuda"),
                "owners": torch.zeros((rows, capacity), dtype=torch.int32, device="cuda"),
                "generations": torch.zeros(rows, dtype=torch.int32, device="cuda"),
            }

        ref = allocate()
        out = allocate()
        common = dict(
            KT=kt, KT_PAD=triton.next_power_of_2(kt),
            K_F=k_f, K_N=k_n, K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
            R=r, HEAD=sink, TAIL=swa, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=0, BS=bs, PAGE=64, NEAR_LEN=4096, K2=1024,
            HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=0,
            LOG2_F=k_f_pad.bit_length() - 1,
            LOG2_N=k_n_pad.bit_length() - 1,
            num_warps=8)
        _topk_candidate_batched_kernel[(rows,)](
            far, near, far, near,
            ref["i_f"], ref["i_n"], lengths, skip_far, skip_near, valid_req,
            ref["out"], ref["valid"], ref["lower"], ref["upper"], ref["quota"],
            ref["owners"], ref["generations"],
            TILED=False, NTILES=1, NTILES_PAD=1, **common)
        partial_f = torch.empty(
            (rows, ntiles, k_f_pad), dtype=torch.int32, device="cuda")
        partial_n = torch.empty(
            (rows, ntiles, k_n_pad), dtype=torch.int32, device="cuda")
        _l1_topk_partial_kernel[(rows, ntiles)](
            far, near, partial_f, partial_n,
            skip_far, skip_near, valid_req,
            KT=kt, TILE=512, NTILES=ntiles,
            K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
            HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=0, num_warps=8)
        _topk_candidate_batched_kernel[(rows,)](
            partial_f, partial_n, far, near,
            out["i_f"], out["i_n"], lengths, skip_far, skip_near, valid_req,
            out["out"], out["valid"], out["lower"], out["upper"], out["quota"],
            out["owners"], out["generations"],
            TILED=True, NTILES=ntiles, NTILES_PAD=triton.next_power_of_2(ntiles),
            **common)
        torch.cuda.synchronize()
        for name in ref:
            self.assertTrue(torch.equal(ref[name], out[name]), f"{name} 不一致")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_skip_states_preserve_fused_and_parallel_outputs(self):
        rows, hkv, batch, layers, kt = 16, 2, 4, 2, 513
        k_f, k_n, bs = 16, 8, 64
        sink, swa = 4, 128
        capacity = kt * bs
        m = layers * batch
        torch.manual_seed(20261107)
        far = torch.randn((rows, kt), device="cuda", dtype=torch.bfloat16)
        near = torch.randn((rows, kt), device="cuda", dtype=torch.bfloat16)
        far[0, 0] = float("nan")
        near[1, 1] = float("-inf")
        lengths = torch.tensor(
            [capacity, capacity - 65, 1003, 1280],
            dtype=torch.int64, device="cuda")
        valid_req = torch.tensor([5, 1, 4, 0], dtype=torch.int64, device="cuda")
        layer_offset = 2
        skip_far = torch.zeros(
            (layers + layer_offset, 6), dtype=torch.bool, device="cuda")
        skip_near = torch.zeros_like(skip_far)
        states = ((False, False), (True, False), (False, True), (True, True))
        for layer in range(layers):
            for req, (sf, sn) in enumerate(states):
                req_id = int(valid_req[req])
                skip_far[layer + layer_offset, req_id] = sf ^ (layer == 1)
                skip_near[layer + layer_offset, req_id] = sn
        for row in range(rows):
            m_idx = row // hkv
            layer, req = m_idx // batch, m_idx % batch
            req_id = int(valid_req[req])
            if bool(skip_far[layer + layer_offset, req_id]):
                far[row] = float("-inf")
            if bool(skip_near[layer + layer_offset, req_id]):
                near[row] = float("-inf")
        k_f_pad, k_n_pad = 16, 8
        ntiles = triton.cdiv(kt, 512)
        r = (k_f + k_n + 2) * bs
        width = r + sink + swa

        def allocate():
            return {
                "i_f": torch.empty((rows, k_f), dtype=torch.int64, device="cuda"),
                "i_n": torch.empty((rows, k_n), dtype=torch.int64, device="cuda"),
                "out": torch.empty((rows, width), dtype=torch.int64, device="cuda"),
                "valid": torch.empty((rows, width), dtype=torch.bool, device="cuda"),
                "lower": torch.empty(m, dtype=torch.int64, device="cuda"),
                "upper": torch.empty(m, dtype=torch.int64, device="cuda"),
                "quota": torch.empty(m, dtype=torch.int64, device="cuda"),
                "owners": torch.zeros(
                    (rows, capacity), dtype=torch.int32, device="cuda"),
                "generations": torch.zeros(rows, dtype=torch.int32, device="cuda"),
            }

        common = dict(
            KT=kt, KT_PAD=triton.next_power_of_2(kt),
            K_F=k_f, K_N=k_n, K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
            R=r, HEAD=sink, TAIL=swa, SINK=sink, SWA=swa, CAP=capacity,
            SINK_POOL_LO=0, BS=bs, PAGE=64, NEAR_LEN=4096, K2=1024,
            HKV=hkv, B=batch, MAX_REQ=6, LAYER_OFFSET=layer_offset, num_warps=8)
        ref = allocate()
        _reference_candidate[(rows,)](
            far, near, ref["i_f"], ref["i_n"], lengths,
            skip_far, skip_near, valid_req, ref["out"], ref["valid"],
            ref["lower"], ref["upper"], ref["quota"],
            ref["owners"], ref["generations"], **common)

        eager = allocate()
        _topk_candidate_batched_kernel[(rows,)](
            far, near, far, near, eager["i_f"], eager["i_n"], lengths,
            skip_far, skip_near, valid_req, eager["out"], eager["valid"],
            eager["lower"], eager["upper"], eager["quota"],
            eager["owners"], eager["generations"],
            TILED=False, NTILES=1, NTILES_PAD=1,
            LOG2_F=4, LOG2_N=3, **common)

        partial_f = torch.empty(
            (rows, ntiles, k_f_pad), dtype=torch.int32, device="cuda")
        partial_n = torch.empty(
            (rows, ntiles, k_n_pad), dtype=torch.int32, device="cuda")
        _l1_topk_partial_kernel[(rows, ntiles)](
            far, near, partial_f, partial_n, skip_far, skip_near, valid_req,
            KT=kt, TILE=512, NTILES=ntiles,
            K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
            HKV=hkv, B=batch, MAX_REQ=6, LAYER_OFFSET=layer_offset, num_warps=8)
        tiled = allocate()
        _topk_candidate_batched_kernel[(rows,)](
            partial_f, partial_n, far, near,
            tiled["i_f"], tiled["i_n"], lengths,
            skip_far, skip_near, valid_req, tiled["out"], tiled["valid"],
            tiled["lower"], tiled["upper"], tiled["quota"],
            tiled["owners"], tiled["generations"],
            TILED=True, NTILES=ntiles, NTILES_PAD=2,
            LOG2_F=4, LOG2_N=3, **common)
        parallel = allocate()
        _topk_candidate_prepare_kernel[(rows,)](
            partial_f, partial_n, parallel["i_f"], parallel["i_n"], lengths,
            skip_far, skip_near, valid_req,
            parallel["lower"], parallel["upper"], parallel["quota"],
            parallel["owners"], parallel["generations"],
            TILED=True, NTILES=ntiles, NTILES_PAD=2,
            LOG2_F=4, LOG2_N=3, **common)
        expand_common = {
            k: v for k, v in common.items()
            if k in ("K_F", "K_N", "KT", "R", "HEAD", "TAIL", "SINK",
                     "SWA", "CAP", "SINK_POOL_LO", "BS", "HKV", "B")}
        _candidate_expand_batched_kernel[(rows, triton.cdiv(width, 1024))](
            parallel["i_f"], parallel["i_n"], far, near, lengths,
            parallel["out"], parallel["valid"],
            parallel["owners"], parallel["generations"],
            **expand_common, num_warps=8)
        torch.cuda.synchronize()
        for name in ref:
            self.assertTrue(torch.equal(ref[name], eager[name]),
                            f"non-tiled {name} 不一致")
            self.assertTrue(torch.equal(ref[name], tiled[name]),
                            f"tiled {name} 不一致")
            self.assertTrue(torch.equal(ref[name], parallel[name]),
                            f"parallel {name} 不一致")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_skipped_topk_k_boundaries(self):
        rows, hkv, kt, bs = 2, 2, 8, 1
        capacity = kt * bs
        far = torch.full(
            (rows, kt), float("-inf"), device="cuda", dtype=torch.bfloat16)
        near = torch.full_like(far, float("-inf"))
        lengths = torch.full((1,), capacity, dtype=torch.int64, device="cuda")
        valid_req = torch.zeros(1, dtype=torch.int64, device="cuda")
        skip_far = torch.ones((2, 1), dtype=torch.bool, device="cuda")
        skip_near = torch.ones_like(skip_far)
        for k_f, k_n in ((0, 0), (1, 0), (0, 1), (1, 1), (kt, kt)):
            for tiled in (False, True):
                with self.subTest(k_f=k_f, k_n=k_n, tiled=tiled):
                    k_f_pad = max(1, triton.next_power_of_2(k_f))
                    k_n_pad = max(1, triton.next_power_of_2(k_n))
                    r = (k_f + k_n + 2) * bs

                    def allocate():
                        return {
                            "i_f": torch.empty(
                                (rows, k_f), dtype=torch.int64, device="cuda"),
                            "i_n": torch.empty(
                                (rows, k_n), dtype=torch.int64, device="cuda"),
                            "out": torch.empty(
                                (rows, r), dtype=torch.int64, device="cuda"),
                            "valid": torch.empty(
                                (rows, r), dtype=torch.bool, device="cuda"),
                            "lower": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "upper": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "quota": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "owners": torch.zeros(
                                (rows, capacity), dtype=torch.int32, device="cuda"),
                            "generations": torch.zeros(
                                rows, dtype=torch.int32, device="cuda"),
                        }

                    common = dict(
                        KT=kt, KT_PAD=kt, K_F=k_f, K_N=k_n,
                        K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                        R=r, HEAD=0, TAIL=0, SINK=0, SWA=0, CAP=capacity,
                        SINK_POOL_LO=0, BS=bs, PAGE=64, NEAR_LEN=4096, K2=1024,
                        HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=1, num_warps=8)
                    ref = allocate()
                    _reference_candidate[(rows,)](
                        far, near, ref["i_f"], ref["i_n"], lengths,
                        skip_far, skip_near, valid_req, ref["out"], ref["valid"],
                        ref["lower"], ref["upper"], ref["quota"],
                        ref["owners"], ref["generations"], **common)
                    if tiled:
                        far_topk = torch.empty(
                            (rows, 1, k_f_pad), dtype=torch.int32, device="cuda")
                        near_topk = torch.empty(
                            (rows, 1, k_n_pad), dtype=torch.int32, device="cuda")
                        _l1_topk_partial_kernel[(rows, 1)](
                            far, near, far_topk, near_topk,
                            skip_far, skip_near, valid_req,
                            KT=kt, TILE=512, NTILES=1,
                            K_F_PAD=k_f_pad, K_N_PAD=k_n_pad,
                            HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=1, num_warps=8)
                    else:
                        far_topk, near_topk = far, near
                    actual = allocate()
                    _topk_candidate_batched_kernel[(rows,)](
                        far_topk, near_topk, far, near,
                        actual["i_f"], actual["i_n"], lengths,
                        skip_far, skip_near, valid_req,
                        actual["out"], actual["valid"],
                        actual["lower"], actual["upper"], actual["quota"],
                        actual["owners"], actual["generations"],
                        TILED=tiled, NTILES=1, NTILES_PAD=1,
                        LOG2_F=k_f_pad.bit_length() - 1,
                        LOG2_N=k_n_pad.bit_length() - 1, **common)
                    torch.cuda.synchronize()
                    for name in ref:
                        self.assertTrue(torch.equal(ref[name], actual[name]),
                                        f"{name} 不一致")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_skipped_pool_does_not_read_scores_or_partial(self):
        rows, hkv, k_f, k_n, bs = 8, 8, 16, 8, 64
        for tiled, kt in ((False, 256), (True, 513)):
            for sf, sn in ((False, False), (True, False),
                           (False, True), (True, True)):
                with self.subTest(tiled=tiled, skip_far=sf, skip_near=sn):
                    capacity = kt * bs
                    torch.manual_seed(20261107 + kt + int(sf) * 2 + int(sn))
                    far = torch.randn(
                        (rows, kt), device="cuda", dtype=torch.bfloat16)
                    near = torch.randn(
                        (rows, kt), device="cuda", dtype=torch.bfloat16)
                    if sf:
                        far.fill_(float("-inf"))
                    if sn:
                        near.fill_(float("-inf"))
                    tiny = torch.empty(1, dtype=torch.bfloat16, device="cuda")
                    tiny_partial = torch.empty(1, dtype=torch.int32, device="cuda")
                    far_arg = tiny if sf else far
                    near_arg = tiny if sn else near
                    skip_far = torch.full((1,), sf, dtype=torch.bool, device="cuda")
                    skip_near = torch.full(
                        (1,), sn, dtype=torch.bool, device="cuda")
                    valid_req = torch.zeros(1, dtype=torch.int64, device="cuda")
                    lengths = torch.ones(1, dtype=torch.int64, device="cuda")
                    ntiles = triton.cdiv(kt, 512)

                    def allocate():
                        return {
                            "i_f": torch.empty(
                                (rows, k_f), dtype=torch.int64, device="cuda"),
                            "i_n": torch.empty(
                                (rows, k_n), dtype=torch.int64, device="cuda"),
                            # Top-K 使用 tiny skipped 指针；candidate expansion
                            # 仍使用完整原始分数，符合生产路径的读取边界。
                            "out": torch.empty(
                                (rows, 1), dtype=torch.int64, device="cuda"),
                            "valid": torch.empty(
                                (rows, 1), dtype=torch.bool, device="cuda"),
                            "lower": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "upper": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "quota": torch.empty(1, dtype=torch.int64, device="cuda"),
                            "owners": torch.zeros(
                                (rows, capacity), dtype=torch.int32, device="cuda"),
                            "generations": torch.zeros(
                                rows, dtype=torch.int32, device="cuda"),
                        }

                    common = dict(
                        KT=kt, KT_PAD=triton.next_power_of_2(kt),
                        K_F=k_f, K_N=k_n, K_F_PAD=16, K_N_PAD=8,
                        R=0, HEAD=1, TAIL=0, SINK=1, SWA=128, CAP=capacity,
                        SINK_POOL_LO=0, BS=bs, PAGE=64, NEAR_LEN=4096, K2=1024,
                        HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=0, num_warps=8)
                    ref = allocate()
                    _reference_candidate[(rows,)](
                        far, near, ref["i_f"], ref["i_n"], lengths,
                        skip_far, skip_near, valid_req, ref["out"], ref["valid"],
                        ref["lower"], ref["upper"], ref["quota"],
                        ref["owners"], ref["generations"], **common)
                    if tiled:
                        partial_f = torch.empty(
                            (rows, ntiles, 16), dtype=torch.int32, device="cuda")
                        partial_n = torch.empty(
                            (rows, ntiles, 8), dtype=torch.int32, device="cuda")
                        _l1_topk_partial_kernel[(rows, ntiles)](
                            far_arg, near_arg,
                            tiny_partial if sf else partial_f,
                            tiny_partial if sn else partial_n,
                            skip_far, skip_near, valid_req,
                            KT=kt, TILE=512, NTILES=ntiles,
                            K_F_PAD=16, K_N_PAD=8,
                            HKV=hkv, B=1, MAX_REQ=1, LAYER_OFFSET=0, num_warps=8)
                        far_topk = tiny_partial if sf else partial_f
                        near_topk = tiny_partial if sn else partial_n
                    else:
                        far_topk, near_topk = far_arg, near_arg
                    actual = allocate()
                    _topk_candidate_batched_kernel[(rows,)](
                        far_topk, near_topk, far, near,
                        actual["i_f"], actual["i_n"], lengths,
                        skip_far, skip_near, valid_req,
                        actual["out"], actual["valid"],
                        actual["lower"], actual["upper"], actual["quota"],
                        actual["owners"], actual["generations"],
                        TILED=tiled, NTILES=ntiles,
                        NTILES_PAD=triton.next_power_of_2(ntiles),
                        LOG2_F=4, LOG2_N=3, **common)
                    parallel = allocate()
                    _topk_candidate_prepare_kernel[(rows,)](
                        far_topk, near_topk, parallel["i_f"], parallel["i_n"],
                        lengths, skip_far, skip_near, valid_req,
                        parallel["lower"], parallel["upper"], parallel["quota"],
                        parallel["owners"], parallel["generations"],
                        TILED=tiled, NTILES=ntiles,
                        NTILES_PAD=triton.next_power_of_2(ntiles),
                        LOG2_F=4, LOG2_N=3, **common)
                    torch.cuda.synchronize()
                    for name in ref:
                        self.assertTrue(torch.equal(ref[name], actual[name]),
                                        f"fused {name} 不一致")
                    for name in ("i_f", "i_n", "lower", "upper", "quota",
                                 "generations"):
                        self.assertTrue(torch.equal(ref[name], parallel[name]),
                                        f"prepare {name} 不一致")


if __name__ == "__main__":
    unittest.main()
