"""长行 L1 局部 Top-K 与原整行实现的逐位一致性回归。"""
import unittest

import torch
import triton
import triton.language as tl

from sparse_attn.indexer.nfa_layer_batch_selector import (
    _l1_topk_partial_kernel,
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
                    KT=kt, TILE=512, NTILES=ntiles,
                    K_F_PAD=k_f_pad, K_N_PAD=k_n_pad, num_warps=8)
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


if __name__ == "__main__":
    unittest.main()
