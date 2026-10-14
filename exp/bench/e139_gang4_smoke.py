# -*- coding: utf-8 -*-
"""E139 纲领④ 冒烟：fused_attn_triton vs _chunk_rows_attn 组版（小规模）。"""
import os
import sys

TREE = "/tmp/wt_e138_ablation"
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402

DEV = "cuda:0"
os.environ["TLI_PREFILL_PREFETCH"] = "0"


def make_qkv(L, seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh(K2=1024):
    ns = make_args("mavg_tail")
    ns.tia_level2_topk = K2
    idx = TLIIndexer(ns)
    idx.layer_idx = 2
    idx.clear()
    return idx


def run(L, grp, fused):
    q, k, v = make_qkv(L)
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = str(grp)
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1" if fused else "0"
    o = sparse_prefill_attn(fresh(), q, k, v)
    return o, (q, k, v)


for L, grp in [(5000, 64), (8192, 64), (3000, 64)]:
    o_ref, _ = run(L, grp, False)
    o_new, _ = run(L, grp, True)
    ok = torch.allclose(o_ref.float(), o_new.float(), rtol=1e-2, atol=1e-3)
    md = (o_ref.float() - o_new.float()).abs().max().item()
    nn = torch.isnan(o_new.float()).sum().item()
    print(f"L={L:6d} grp={grp}: allclose={'PASS' if ok else 'FAIL'} "
          f"maxdiff={md:.3e} nan={nn}", flush=True)
    del o_ref, o_new
    torch.cuda.empty_cache()
print("SMOKE DONE")
