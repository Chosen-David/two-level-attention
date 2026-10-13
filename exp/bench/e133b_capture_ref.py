#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E133b：抓取当前实现（改造前）的 24 步 mask 作为原生实现对拍基准。

用法：python3 e133b_capture_ref.py <arm: full|tail|full_native|tail_native> <out.pt>
确定性 seed（20261012/777001），与 e133_mavg_tail_mask_bitexact 同口径。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "exp", "bench"))

import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.patches.utils import prepare_cu_seqlens, prepare_seqlens  # noqa: E402


def arm_args(arm):
    """cmp4 基座（e2e 锚点口径）；arm ∈ full / tail。"""
    ns = make_args("mavg_e2e")
    if arm == "tail":
        ns.tli_subspace = "tail"
    return ns


def main():
    arm = sys.argv[1]
    out_path = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    torch.manual_seed(20261012)
    idx = TLIIndexer(arm_args(arm))
    idx.layer_idx = 2
    idx.clear()
    dev = "cuda:0"
    H, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    S = 65536
    g = torch.Generator(device=dev).manual_seed(777001)
    k = torch.randn(1, S, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)
    qs = torch.randn(steps, 1, 1, H, D, device=dev, dtype=torch.bfloat16, generator=g)
    newks = torch.randn(steps, 1, 1, Hkv, D, device=dev, dtype=torch.bfloat16, generator=g)

    masks = []
    for i in range(steps):
        q = qs[i]
        cu = prepare_cu_seqlens(k)
        q_ids = prepare_seqlens(cu) - 1
        index_dict = idx.prepare_index(k, cu)
        score_dict = idx.compute_score(q, q_ids, index_dict, scale)
        mask = idx.compute_mask(q_ids, score_dict)
        masks.append(mask.cpu().clone())
        k = torch.cat([k, newks[i]], dim=1)
    torch.cuda.synchronize()
    torch.save({"arm": arm, "masks": masks}, out_path)
    print(f"[OK] arm={arm} steps={steps} -> {out_path}")


if __name__ == "__main__":
    main()
