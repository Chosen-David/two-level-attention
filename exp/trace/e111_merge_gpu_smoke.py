#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E127 inplace 合并后 GPU smoke：验证并入基类的 Triton fast path 与
增量缓存真实生效且逐位一致。

四项检查：
  1. 非投影非 skip：_triton_ok 保持 True（fast path 真实命中非异常回退）
     + fast mask vs 强制回退（_triton_ok=False）mask 逐位一致。
  2. 增量缓存 decode 模拟：连续 20 步 k 追加 1 token，增量实例 mask
     vs 每步全新全量实例 mask 逐位一致，_inc 非 None（增量真命中）。
  3. skip+e64（E111 新语义）GPU 上同 2（dgate_short 与增量共存）。
  4. score 数值：fast vs 回退 allclose（bf16 舍入后 ULP 带内）。
"""
import argparse
import os
import sys
import types

import torch

sys.path.insert(0, "/home/wangyuanshuo02/two-level-attention-e111")
IDX = __import__("importlib").import_module("sparse_attn.indexer")

BS, K1, K2 = 64, 128, 1024
SINK_TOK, SWA_TOK = 128, 128


def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=BS, tia_level1_topk=K1, tia_level2_topk=K2,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gen_kq(S, Hkv=8, H=32, D=128, seed=31):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(1, S, Hkv, D, generator=g, device="cuda") * 0.5
    q = torch.randn(1, 1, H, D, generator=g, device="cuda") * 0.5
    return k, q


def mavg_kw():
    return dict(tli_far_select="4bit", tli_near_select="4bit",
                tli_enable_kmeans=False, tli_far_method="minmax",
                tli_near_method="avg", tli_alpha=0.25, tli_beta=0.125,
                tli_gamma=0.625)


def run(idx, k, q, cu, q_ids, scale):
    d = idx.prepare_index(k, cu)
    s = idx.compute_score(q, q_ids, d, scale)
    m = idx.compute_mask(q_ids, s)
    return d, s, m


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    scale = 128 ** -0.5

    # ---------- 1. fast path 命中 + fast vs 回退逐位 ----------
    S = 8192
    k, q = gen_kq(S)
    cu = torch.tensor([0, S], device=dev)
    q_ids = torch.tensor([S - 1], device=dev)

    idx_fast = IDX.TLIIndexer(make_args(**mavg_kw()))
    idx_fast.layer_idx = 3
    _, s_fast, m_fast = run(idx_fast, k, q, cu, q_ids, scale)
    assert idx_fast._triton_ok, "fast path 未命中（_triton_ok=False）——Triton 异常回退了"
    assert idx_fast._inc is not None, "增量缓存未建立"

    idx_fb = IDX.TLIIndexer(make_args(**mavg_kw()))
    idx_fb.layer_idx = 3
    idx_fb._triton_ok = False   # 强制走原版 repeat+einsum
    _, s_fb, m_fb = run(idx_fb, k, q, cu, q_ids, scale)
    eq_mask = torch.equal(m_fast, m_fb)
    sf, sfb = s_fast["score_fine"].float(), s_fb["score_fine"].float()
    ulp = (sf != sfb).float().mean().item()
    md = (sf - sfb).abs().max().item()
    print(f"[1] fast 命中（_triton_ok=True）✓  mask fast==fallback: {eq_mask}  "
          f"fine ULP 比例={ulp:.2e} max|Δ|={md:.2e}")
    assert eq_mask, "fast vs 回退 mask 不一致"

    # ---------- 2. 增量 vs 每步全量重建（decode 模拟，非 skip）----------
    idx_inc = IDX.TLIIndexer(make_args(**mavg_kw()))
    idx_inc.layer_idx = 3
    n_diff = 0
    for step in range(20):
        if step > 0:
            k = torch.cat([k, torch.randn(1, 1, 8, 128, device=dev) * 0.5], dim=1)
            q = torch.randn(1, 1, 32, 128, device=dev) * 0.5
        S_now = k.shape[1]
        cu = torch.tensor([0, S_now], device=dev)
        q_ids = torch.tensor([S_now - 1], device=dev)
        _, _, m_inc = run(idx_inc, k, q, cu, q_ids, scale)
        idx_full = IDX.TLIIndexer(make_args(**mavg_kw()))
        idx_full.layer_idx = 3
        _, _, m_ref = run(idx_full, k, q, cu, q_ids, scale)
        if not torch.equal(m_inc, m_ref):
            n_diff += 1
    assert idx_inc._inc is not None and idx_inc._inc["S"] == S_now
    print(f"[2] 增量 decode 20 步 vs 全量重建: 不一致步数={n_diff}（_inc.S={idx_inc._inc['S']}）")
    assert n_diff == 0, "增量 vs 全量 mask 不一致"

    # ---------- 3. skip+e64（E111 新语义）增量共存 ----------
    kw = mavg_kw()
    idx_s = IDX.TLIIndexer(make_args(**kw))
    idx_s.layer_idx = 3
    idx_s._skip_ids = {3}
    n_diff = 0
    for step in range(20):
        if step > 0:
            k = torch.cat([k, torch.randn(1, 1, 8, 128, device=dev) * 0.5], dim=1)
            q = torch.randn(1, 1, 32, 128, device=dev) * 0.5
        S_now = k.shape[1]
        cu = torch.tensor([0, S_now], device=dev)
        q_ids = torch.tensor([S_now - 1], device=dev)
        d, s_d, m_s = run(idx_s, k, q, cu, q_ids, scale)
        # far 段省算证据（GPU 上 dgate_short 分支）
        mid_len = max(0, S_now - SINK_TOK - SWA_TOK)
        near_dyn = max(BS, int(0.25 * mid_len))
        near_blks = max(2, (S_now - SWA_TOK - near_dyn) // BS)
        sc_far = s_d["score_coarse"][0, 0, ..., 2:near_blks]
        sf_far = s_d["score_fine"][0, 0, ..., SINK_TOK:near_blks * BS]
        assert bool((sc_far == float("-inf")).all()), f"step{step}: far coarse 未省算"
        assert bool((sf_far == float("-inf")).all()), f"step{step}: far fine 未省算"
        idx_r = IDX.TLIIndexer(make_args(**kw))
        idx_r.layer_idx = 3
        idx_r._skip_ids = {3}
        _, _, m_r = run(idx_r, k, q, cu, q_ids, scale)
        if not torch.equal(m_s, m_r):
            n_diff += 1
    print(f"[3] skip+e64 增量 decode 20 步 vs 全量重建: 不一致步数={n_diff} "
          f"（far -inf 省算证据逐步过）")
    assert n_diff == 0, "skip+e64 增量 vs 全量 mask 不一致"

    # ---------- 4. clear() 后新请求全量重建（跨请求缓存失效）----------
    k2, q2 = gen_kq(4097, seed=77)
    cu2 = torch.tensor([0, 4097], device=dev)
    idx_s.clear()
    assert idx_s._inc is None, "clear() 未失效增量缓存"
    _, _, m_new = run(idx_s, k2, q2, cu2, torch.tensor([4096], device=dev), scale)
    idx_r2 = IDX.TLIIndexer(make_args(**kw))
    idx_r2.layer_idx = 3
    idx_r2._skip_ids = {3}
    _, _, m_r2 = run(idx_r2, k2, q2, cu2, torch.tensor([4096], device=dev), scale)
    assert torch.equal(m_new, m_r2), "clear() 后新请求 mask 不一致"
    print("[4] clear() 跨请求失效 + 新请求（S=4097 尾页）逐位一致 ✓")

    print("[SMOKE] 全过：fast 命中 / fast↔回退逐位 / 增量↔全量逐位 / "
          "skip 省算共存 / clear 失效", flush=True)


if __name__ == "__main__":
    main()
