# -*- coding: utf-8 -*-
"""E139 #253 Step1 输出级硬门：graph 化选择链全链（BATCH+FUSED+GRAPH）vs
eager 全链（BATCH+FUSED, GRAPH=0）——生产口径 sparse_prefill_attn 输出
finite + allclose（选择 mask 恒等由 e139_bs_graph_bitdiff 单独核验）。
L∈{8192, 32768, 65536}。"""
import os
import sys
import time

TREE = os.environ.get("E139_TREE",
                      os.path.expanduser("~/tmp_e138_abl_tree_187"))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e139_graph_checks import require_outputs, require_graph
from sparse_attn.ops.batch_select import _GRAPH_CACHE, _GRAPH_STATS
from e131_arm_bench import make_args  # noqa: E402  (TREE/exp/bench)
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402


def fresh():
    idx = TLIIndexer(make_args("mavg_tail"))
    idx.layer_idx = 2
    idx.clear()
    return idx


def run(idx, q, k, v, graph):
    os.environ["TLI_PREFILL_RUNTIME"] = "candidate"
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"
    os.environ["TLI_PREFILL_PREFETCH"] = "0"
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1" if graph else "0"
    return sparse_prefill_attn(idx, q, k, v)


def col_set(o):
    """行级有效列集合（非零 → bool 列集），G2 口径。"""
    return (o.abs().sum(dim=(0, 1)) != 0) if o.dim() == 4 else o


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 #253 Step1 e2e 硬门 GPU{gpu} ===", flush=True)
for L in (8192, 32768, 65536):
    g = torch.Generator(device="cuda:0").manual_seed(999)
    H, Hkv, D = 32, 8, 128
    q = torch.randn(1, L, H, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, L, Hkv, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, L, Hkv, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    o_e = run(fresh(), q, k, v, graph=False)
    _GRAPH_CACHE.clear()
    before = dict(_GRAPH_STATS)
    o_g = run(fresh(), q, k, v, graph=True)     # 首调 = capture + replay
    require_graph(before, _GRAPH_STATS, "capture")
    require_outputs(o_g, o_e)
    before = dict(_GRAPH_STATS)
    # 第二次调用（replay 稳态路径）再对拍——覆盖 capture 层之后的层
    o_g2 = run(fresh(), q, k, v, graph=True)    # 跨 indexer 实例 replay
    require_graph(before, _GRAPH_STATS, "replay")
    require_outputs(o_g2, o_e)
    ok1 = torch.allclose(o_e.float(), o_g.float(), rtol=1e-2, atol=1e-3)
    md1 = (o_e.float() - o_g.float()).abs().max().item()
    ok2 = torch.allclose(o_e.float(), o_g2.float(), rtol=1e-2, atol=1e-3)
    md2 = (o_e.float() - o_g2.float()).abs().max().item()
    nan = int(torch.isnan(o_g.float()).sum().item())
    print(f"L={L:6d}: capture 层 allclose={ok1} maxdiff={md1:.3e} | "
          f"replay 层 allclose={ok2} maxdiff={md2:.3e} nan={nan}", flush=True)
print("=== DONE ===", flush=True)
