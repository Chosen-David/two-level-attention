#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E131-S2：greedy_pass_triton_v2 对拍验证 + 计时（GPU0 开发卡）。

对拍口径（克隆 exp/trace/test_e113_greedy_triton.py）：
  V1 冷启动全量对拍：v2 vs Python 参考（assign 逐位 + 状态 allclose）
  V2 多 seed 逐位一致率（8 seed × 3 sim，断言 ==0；v2 vs Python 参考）
  V3 v2 vs v1 逐位（同输入直接对比两 kernel 输出，应 100%）
  V4 增量续跑 == 全量重放（v2 三段续跑 vs v2 全量）
  V5 randn 最坏（全单例）+ 大 k_live 续跑对拍（E131 口径主路径）
  V6 计时：far T=57088 / near T=8192 / 续跑 8192@k_live≈28K
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sparse_attn.indexer.greedy_triton import (  # noqa: E402
    greedy_pass_triton, greedy_build_triton, greedy_pass_triton_v2)
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)


def gen_structured(T, H, dd, g, n_base=64, noise=0.08):
    base = torch.randn(n_base, H, dd, generator=g, device="cuda")
    idx = torch.randint(0, n_base, (T,), generator=g, device="cuda")
    return base[idx] + noise * torch.randn(T, H, dd, generator=g, device="cuda")


def greedy_cold(fn, x, sim):
    T, H, dd = x.shape
    return fn(x, sim, x.new_zeros(H, T, dd), x.new_zeros(H, T),
              x.new_zeros(H, T), torch.zeros(H, dtype=torch.long, device="cuda"))


def cmp_state(r, t, tag):
    sums_r, cnt_r, sq_r, kl_r, a_r = r
    sums_t, cnt_t, sq_t, kl_t, a_t = t
    assert torch.equal(kl_r, kl_t), f"{tag}: k_live {kl_r.tolist()} vs {kl_t.tolist()}"
    n_mism = int((a_r != a_t).sum())
    assert n_mism == 0, f"{tag}: assign 不一致 {n_mism}/{a_r.numel()} 处"
    live = int(kl_r.max().item())
    assert live > 0
    assert torch.allclose(cnt_r[:, :live], cnt_t[:, :live], atol=1e-6), f"{tag}: cnt"
    assert torch.allclose(sq_r[:, :live], sq_t[:, :live], rtol=1e-5, atol=1e-5), f"{tag}: sq"
    assert torch.allclose(sums_r[:, :live], sums_t[:, :live], atol=1e-5), f"{tag}: sums"
    return live


ref = TLIIndexer._greedy_cluster_pass_python


def v1_cold(x, sim):
    T, H, dd = x.shape
    return ref(x, sim, x.new_zeros(H, T, dd), x.new_zeros(H, T),
               x.new_zeros(H, T), torch.zeros(H, dtype=torch.long, device="cuda"))


def main():
    torch.manual_seed(20261012)

    # ---- V1 冷启动 vs Python 参考 ----
    try:
        g = torch.Generator(device="cuda").manual_seed(20261007)
        for (T, H, dd, sim) in [(512, 4, 32, 0.9), (1024, 8, 32, 0.85), (768, 2, 64, 0.92)]:
            x = gen_structured(T, H, dd, g).contiguous()
            r = v1_cold(x, sim)
            t = greedy_cold(greedy_pass_triton_v2, x, sim)
            live = cmp_state(r, t, f"T={T},H={H},dd={dd},sim={sim}")
            assert live < T, "sim<=0.92 应有归并（测试无鉴别力）"
        report("V1 v2 冷启动 vs Python 参考", True, "3 组形状全过")
    except AssertionError as e:
        report("V1 v2 冷启动 vs Python 参考", False, str(e))

    # ---- V2 多 seed 逐位一致率 ----
    try:
        g = torch.Generator(device="cuda").manual_seed(314159)
        tot = mism = live_tot = 0
        for _ in range(8):
            x = gen_structured(2048, 8, 32, g, n_base=128, noise=0.10).contiguous()
            for sim in (0.80, 0.90, 0.95):
                r = v1_cold(x, sim)
                t = greedy_cold(greedy_pass_triton_v2, x, sim)
                tot += r[4].numel()
                mism += int((r[4] != t[4]).sum())
                assert torch.equal(r[3], t[3]), "k_live 不一致"
                live_tot += int(r[3].max().item())
                assert torch.allclose(r[0][:, :live_tot and int(r[3].max())],
                                      t[0][:, :int(r[3].max())], atol=1e-4)
        assert mism == 0, f"逐位不一致 {mism}/{tot}"
        report("V2 多 seed 逐位一致率", True, f"24 组 100%（{tot} 位）")
    except AssertionError as e:
        report("V2 多 seed 逐位一致率", False, str(e))

    # ---- V3 v2 vs v1 同输入逐位 ----
    try:
        g = torch.Generator(device="cuda").manual_seed(999983)
        bad = 0
        for (T, H, dd, sim) in [(3000, 8, 32, 0.9), (2048, 8, 32, 0.95),
                                (1500, 4, 32, 0.85), (512, 8, 128, 0.99)]:
            x = torch.randn(T, H, dd, generator=g, device="cuda").contiguous()
            a = greedy_cold(greedy_pass_triton, x, sim)
            b = greedy_cold(greedy_pass_triton_v2, x, sim)
            assert torch.equal(a[3], b[3]), "k_live"
            bad += int((a[4] != b[4]).sum())
        assert bad == 0, f"v2 vs v1 assign 差 {bad} 位"
        report("V3 v2 vs v1 逐位", True, "4 组 randn/结构 100%")
    except AssertionError as e:
        report("V3 v2 vs v1 逐位", False, str(e))

    # ---- V4 增量续跑 == 全量重放（v2）----
    try:
        g = torch.Generator(device="cuda").manual_seed(271828)
        T, H, dd, sim = 4096, 8, 32, 0.9
        x = gen_structured(T, H, dd, g, n_base=96, noise=0.09).contiguous()
        sums = x.new_zeros(H, 1024, dd)
        cnt = x.new_zeros(H, 1024)
        sq = x.new_zeros(H, 1024)
        kl = torch.zeros(H, dtype=torch.long, device="cuda")
        assign = torch.empty(H, T, dtype=torch.int64, device="cuda")
        for lo, hi in [(0, 1024), (1024, 3072), (3072, 4096)]:
            sums, cnt, sq, kl, a = greedy_pass_triton_v2(x[lo:hi], sim, sums, cnt, sq, kl)
            assign[:, lo:hi] = a
        r = v1_cold(x, sim)
        assert torch.equal(kl, r[3]) and int((assign != r[4]).sum()) == 0, "续跑≠重放"
        report("V4 v2 增量续跑 == 全量重放", True)
    except AssertionError as e:
        report("V4 v2 增量续跑 == 全量重放", False, str(e))

    # ---- V5 randn 最坏大 k_live 续跑对拍（E131 主路径：k_live≈28K）----
    try:
        g = torch.Generator(device="cuda").manual_seed(1234567)
        x_big = torch.randn(28672, 8, 32, generator=g, device="cuda").contiguous()
        st = greedy_cold(greedy_pass_triton, x_big, 0.9)  # 全单例 k_live=28672
        assert int(st[3].max()) == 28672
        x_cont = torch.randn(8192, 8, 32, generator=g, device="cuda").contiguous()
        # v1 续跑
        r1 = greedy_pass_triton(x_cont, 0.9, st[0].clone(), st[1].clone(),
                                st[2].clone(), st[3].clone())
        # v2 续跑
        r2 = greedy_pass_triton_v2(x_cont, 0.9, st[0].clone(), st[1].clone(),
                                   st[2].clone(), st[3].clone())
        assert torch.equal(r1[3], r2[3]), "k_live"
        n = int((r1[4] != r2[4]).sum())
        assert n == 0, f"续跑 assign 差 {n} 位"
        live = int(r1[3].max())
        assert torch.allclose(r1[2][:, :live], r2[2][:, :live], atol=1e-4), "sq"
        assert torch.allclose(r1[0][:, :live], r2[0][:, :live], atol=1e-3), "sums"
        report("V5 randn 最坏 k_live≈28K 续跑 v2 vs v1", True, "逐位一致")
    except AssertionError as e:
        report("V5 randn 最坏续跑", False, str(e))

    # ---- V6 计时对比 ----
    def timed(fn, n=1):
        ts = []
        for _ in range(n):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return ts[len(ts) // 2]

    g = torch.Generator(device="cuda").manual_seed(20261011)
    Tfar, Tn = 57088, 8192
    x_far = torch.randn(Tfar, 8, 32, generator=g, device="cuda").contiguous()
    x_near = torch.randn(Tn, 8, 32, generator=g, device="cuda").contiguous()
    t = timed(lambda: greedy_cold(greedy_pass_triton_v2, x_far, 0.9))
    print(f"[V6] v2 far 冷建 T={Tfar}: {t:.1f} ms ({t/Tfar*1e3:.2f} us/token)", flush=True)
    t = timed(lambda: greedy_cold(greedy_pass_triton_v2, x_near, 0.9))
    print(f"[V6] v2 near 冷建 T={Tn}: {t:.1f} ms ({t/Tn*1e3:.2f} us/token)", flush=True)
    st = greedy_cold(greedy_pass_triton_v2, x_far[:28672], 0.9)
    t = timed(lambda: greedy_pass_triton_v2(
        x_far[28672:28672+8192], 0.9, st[0].clone(), st[1].clone(),
        st[2].clone(), st[3].clone()))
    print(f"[V6] v2 续跑 8192 @k_live≈28K: {t:.1f} ms ({t/8192*1e3:.2f} us/token)",
          flush=True)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n=== {len(RESULTS)-n_fail}/{len(RESULTS)} PASS ===", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
