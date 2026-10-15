# -*- coding: utf-8 -*-
"""E139 #253 Step2 输出级硬门：L2 topk 候选列化（env TLI_PREFILL_L2_TOPK_CAND）
vs 旧 topk 路径——生产口径 sparse_prefill_attn 全链（BATCH+FUSED+GRAPH）
输出逐位一致（maxdiff=0.0）：
  硬门 1：Step2 eager（CAND=1）vs Step1 eager（CAND=0）——mask 逐位一致（
  e139_step2_cand_bitdiff G1 已证）经同一条 attention 链 → 输出须逐位相同。
  硬门 2：Step2 + GRAPH=1 capture/replay vs Step2 eager——分支免化设计
  （nbad host 常量 + 固定形状补位 kernel 进 capture）。
  硬门 3：跨 indexer 实例 replay（e2e 每层独立实例口径）。
L∈{8192, 32768, 65536}。协议克隆 e139_bs_graph_e2e_bitdiff。"""
import os
import sys

TREE = os.environ.get("E139_TREE",
                      os.path.expanduser("~/tmp_e138_abl_tree_187"))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402  (TREE/exp/bench)
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402
import sparse_attn.ops.batch_select as _bs  # noqa: E402


def fresh():
    idx = TLIIndexer(make_args("mavg_tail"))
    idx.layer_idx = 2
    idx.clear()
    return idx


def run(idx, q, k, v, graph, cand):
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = "64"
    os.environ["TLI_PREFILL_BATCH_SELECT"] = "1"
    os.environ["TLI_PREFILL_FUSED_ATTN"] = "1"
    os.environ["TLI_PREFILL_PREFETCH"] = "0"
    os.environ["TLI_PREFILL_SELECT_GRAPH"] = "1" if graph else "0"
    os.environ["TLI_PREFILL_L2_TOPK_CAND"] = "1" if cand else "0"
    return sparse_prefill_attn(idx, q, k, v)


gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"=== E139 #253 Step2 e2e 硬门 GPU{gpu} ===", flush=True)
fail = 0
for L in (8192, 32768, 65536):
    g = torch.Generator(device="cuda:0").manual_seed(999)
    H, Hkv, D = 32, 8, 128
    q = torch.randn(1, L, H, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, L, Hkv, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, L, Hkv, D, device="cuda:0",
                    dtype=torch.bfloat16, generator=g)
    # 同一输入：Step1 eager 锚点 / Step2 eager / Step2 graph capture / replay
    o_e0 = run(fresh(), q, k, v, graph=False, cand=False)
    o_e1 = run(fresh(), q, k, v, graph=False, cand=True)
    _bs._GRAPH_CACHE.clear()          # 排除 Step1 残留缓存（key 含 CAND 指纹，防御性）
    o_g1 = run(fresh(), q, k, v, graph=True, cand=True)   # capture + replay
    o_g2 = run(fresh(), q, k, v, graph=True, cand=True)   # 跨实例 replay
    _bs._GRAPH_CACHE.clear()
    md01 = (o_e0.float() - o_e1.float()).abs().max().item()
    md02 = (o_e1.float() - o_g1.float()).abs().max().item()
    md03 = (o_e1.float() - o_g2.float()).abs().max().item()
    nan = int(torch.isnan(o_g1.float()).sum().item())
    ok = (md01 == 0.0 and md02 == 0.0 and md03 == 0.0 and nan == 0)
    fail += 0 if ok else 1
    print(f"L={L:6d}: Step2 eager vs Step1 eager maxdiff={md01:.3e} | "
          f"capture vs Step2 eager maxdiff={md02:.3e} | "
          f"replay vs Step2 eager maxdiff={md03:.3e} nan={nan} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
print(f"=== TOTAL FAIL = {fail} ===", flush=True)
print("=== DONE ===", flush=True)
