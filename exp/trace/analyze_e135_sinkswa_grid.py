#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E135：sink/swa 敏感性扫参 mass 分析器 + sinkswa_only 归因臂（用户 2026-10-13 批准）。

背景：生产 sink=128（tli_indexer sink_blocks=2×64）/ swa=128（tia_indexer
sliding_window_size=128）均为无实验依据的工程约定（文献惯例 + 块对齐）。
本分析器在 E98/E130 mass 协议（真实全维 softmax 行级覆盖率）上扫描
(sink, swa) 网格下 mavg 冠军 (α,β,γ)=(0.25,0.125,0.625) 的覆盖率：

  ① champion 网格臂：sink ∈ {16,64,128} × swa ∈ {128,256,512}
  ② sinkswa_only 归因臂（E135a mass 面）：cand = sink_c | swa_c——纯固定区
     覆盖率，与冠军臂覆盖率之差即两级索引在 mass 口径下的增量上限
  ③ 锚点复现自校验：(SINK,SWA)=(128,1024) 臂必须逐值复现
     e98_abg_full_grid.json 的 mavg_a0.25_b0.125_g0.625 per_sample 重算均值
     （E98 协议 SINK=128/SWA=1024/B_TOK=2048/BP=64/BS=64 同源克隆）

口径：TRACE=/tmp/trace/qwen3-8b 16 样本、B_TOK=2048、BP=64、BS=64、TAIL_N=4
（全部与 E98/E130 同协议）；行过滤统一按网格最坏点 (sink=16,swa=512) 的
mid_len ≥ 4096（跨臂同一行集，保证可比）。子块 sink（16<bs）语义与生产
E135 实现同源：页池左界 = floor(sink/bs) 块（16→0 块，sink 所在块参与
L1 池竞争），强制区宽 = sink token 级。

用法：E135_DEVICE=cuda:0 python3 analyze_e135_sinkswa_grid.py [--smoke]
输出：exp/trace/results/e135_sinkswa_grid.json
"""
import argparse
import json
import os
import sys

import torch

TRACE = "/tmp/trace/qwen3-8b"
OUT = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e135_sinkswa_grid.json"
E98_JSON = "/home/wangyuanshuo02/two-level-attention/exp/trace/results/e98_abg_full_grid.json"
BS = 64
D2I = list(range(48, 64)) + list(range(112, 128))
B_TOK = 2048
BP = 64                       # 页池总量（E98 同协议）
TAIL_N = 4
A_CH, B_CH, G_CH = 0.25, 0.125, 0.625     # mavg 冠军 (α,β,γ)
SINK_GRID = [16, 64, 128]
SWA_GRID = [128, 256, 512]
ANCHOR = (128, 1024)          # E98 协议锚点（自校验臂）
MIN_MID = 4096                # 行过滤（按网格最坏点 sink=16/swa=512 统一）


def eval_layer(lf, device):
    d = torch.load(lf, map_location="cpu", weights_only=False)
    k, q, qpos, S = d["k"].to(device).float(), d["q"].to(device).float(), d["qpos"], d["S"]
    Hkv, H, D = k.shape[1], q.shape[1], k.shape[-1]
    G = H // Hkv
    pos = torch.arange(S, device=device)
    res = {}
    idx = torch.tensor(D2I, device=device)
    ksub_layer = k[..., idx]
    for ri in range(TAIL_N):
        t_r = int(qpos[-TAIL_N + ri])
        # 统一行过滤：网格最坏点 (16,512) 的 mid_len ≥ MIN_MID（跨臂同行集）
        if t_r + 1 - 512 - 16 < MIN_MID:
            continue
        qg = q[-TAIL_N + ri].reshape(Hkv, G, D)
        k4 = k[:, :, None, :].expand(S, Hkv, G, D)
        s_full = torch.einsum("hgd,shgd->hgs", qg, k4) * (D ** -0.5)
        s_full = s_full.masked_fill((pos > t_r).view(1, 1, -1), float("-inf"))
        p_full = torch.softmax(s_full, dim=-1)

        def cov_mass(cand):
            return float((p_full * cand.unsqueeze(1)).sum(-1).mean())

        qsub = q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32).sum(1)
        nblk = (S + BS - 1) // BS
        kk = torch.nn.functional.pad(ksub_layer, (0, 0, 0, 0, 0, nblk * BS - S))
        kc = kk.reshape(nblk, BS, Hkv, 32)
        kmin, kmax = kc.amin(1), kc.amax(1)
        sc_mm = (torch.einsum("hd,nhd->hn", qsub.clamp(min=0), kmax) +
                 torch.einsum("hd,nhd->hn", qsub.clamp(max=0), kmin))
        sc_av = torch.einsum("hd,nhd->hn", qsub, kc.mean(1))
        sc_tok = torch.einsum("hgd,shd->hgs",
                              q[-TAIL_N + ri][..., idx].reshape(Hkv, G, 32),
                              ksub_layer).sum(1)

        def eval_sinkswa(sink, swa):
            """单 (sink,swa) 点的 mavg 冠军两级选择 mass + 纯固定区 mass。

            生产 E135 语义同源：页池左界 = floor(sink/bs) 块（sink 所在块参与
            竞争）；强制区宽 sink/swa token 级；near 右界 = swa 起点（B10）；
            γ 预算切分 = E98 gamma_arm 克隆（nt_near=min(nb_near·BS·γ,B_TOK)、
            nt_far=max(64,B_TOK−nt_near)，无泄漏口径——E98 原版）。"""
            mid_hi = t_r + 1 - swa
            mid_len = mid_hi - sink
            if mid_len < MIN_MID:
                return None
            pool_lo_blk = max(0, sink // BS)
            sc_blk = sc_mm[:, pool_lo_blk:]
            sc_blk_av = sc_av[:, pool_lo_blk:]
            # mid 相对坐标的页选择（页分数相对 pool_lo_blk，token 偏移 +sink）
            nb_near = int(round(BP * B_CH))
            near_L = int(A_CH * mid_len)
            near_lo_off = mid_len - near_L
            far_L = mid_len - near_L

            def select_pages(lo_off, hi_off, use_avg, n_pages):
                if n_pages <= 0 or hi_off - lo_off <= 0:
                    return torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
                sc = sc_blk_av if use_avg else sc_blk
                bi = torch.arange(sc.shape[1], device=device)
                # 页绝对 token 区间 [pool_lo_blk·BS + bi·BS, +BS) ↔ mid 相对
                s1 = sc.masked_fill(
                    ~(((bi * BS + sink) < sink + hi_off) &
                      ((bi * BS + BS - 1 + sink) >= sink + lo_off)).view(1, -1),
                    float("-inf"))
                ib = torch.topk(s1, min(n_pages, s1.shape[-1]), dim=-1).indices
                tok = (ib.unsqueeze(-1) * BS +
                       torch.arange(BS, device=device).view(1, 1, BS)).reshape(Hkv, -1)
                tok = tok + pool_lo_blk * BS - sink      # mid 相对坐标
                # 子块 sink（pool_lo_blk·BS < sink）时首块含负偏移，clamp 进 [0, mid_len)
                tok = tok.clamp(min=0, max=mid_len - 1)
                pool = torch.zeros(Hkv, mid_len, dtype=torch.bool, device=device)
                pool.scatter_(1, tok, True)
                ar = torch.arange(mid_len, device=device)
                pool &= ((ar >= lo_off) & (ar < hi_off)).view(1, -1)
                return pool

            ts = sc_tok[:, sink:mid_hi]
            cand = torch.zeros(Hkv, S, dtype=torch.bool, device=device)
            sink_c = (pos < sink).view(1, -1)
            swa_c = ((pos >= t_r + 1 - swa) & (pos <= t_r)).view(1, -1)
            cand |= sink_c | swa_c
            nt_near = min(int(nb_near * BS * G_CH), B_TOK)
            nt_far = max(64, B_TOK - nt_near)
            if nt_far > far_L:
                return None                       # E98 约束：far_budget ≤ far_L
            jobs = []
            if A_CH > 0 and B_CH > 0:
                jobs.append((select_pages(near_lo_off, mid_len, True, nb_near), nt_near))
            if A_CH < 1 and B_CH < 1:
                jobs.append((select_pages(0, near_lo_off, False, BP - nb_near), nt_far))
            for pool_x, nt in jobs:
                if nt <= 0:
                    continue
                tsx = ts.masked_fill(~pool_x, float("-inf"))
                _, ix = torch.topk(tsx, min(nt, mid_len), dim=-1)
                cand.scatter_(1, ix + sink, True)
            return cov_mass(cand), cov_mass(sink_c | swa_c)

        for swa in SWA_GRID:
            for sink in SINK_GRID:
                r = eval_sinkswa(sink, swa)
                if r is not None:
                    res.setdefault(f"mavg_champ_sk{sink}_sw{swa}", []).append(r[0])
                    res.setdefault(f"sinkswa_only_sk{sink}_sw{swa}", []).append(r[1])
        r = eval_sinkswa(*ANCHOR)
        if r is not None:
            res.setdefault("anchor_e98_sk128_sw1024", []).append(r[0])
        print(f"[E135 layer] {os.path.basename(lf)} ri={ri} arms={len(res)}", flush=True)
    del k, q, d, ksub_layer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def run_sample(name, device):
    n_layers = json.load(open(f"{TRACE}/{name}/meta.json"))["n_layers"]
    agg = {}
    for li in range(0, n_layers, max(1, n_layers // 12)):
        r = eval_layer(f"{TRACE}/{name}/layer{li:02d}.pt", device)
        if not r:
            continue
        for k2, v in r.items():
            agg.setdefault(k2, []).extend(v)
    return {k2: sum(v) / len(v) for k2, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ns = ap.parse_args()
    device = os.environ.get("E135_DEVICE", "cuda:0")
    samples = sorted(os.listdir(TRACE))
    if ns.smoke:
        samples = samples[:1]
    per_sample = {}
    for name in samples:
        per_sample[name] = run_sample(name, device)
        print(f"[E135 sample] {name}: "
              f"{ {k: round(v, 4) for k, v in list(per_sample[name].items())[:3]} }",
              flush=True)
    mean = {}
    for arm in next(iter(per_sample.values())):
        vals = [per_sample[s][arm] for s in per_sample if arm in per_sample[s]]
        mean[arm] = sum(vals) / len(vals)
    # ---- 锚点复现自校验：per-sample 级逐样本比较（smoke 子集同样有效）----
    # 行集天然对齐：ANCHOR 臂 eval_sinkswa(128,1024) 内部 mid_len 检查与 E98
    # 同式（t_r+1-1024-128 ≥ 4096），统一行过滤是 E98 行集的超集不引入额外行。
    # 容差 1e-5：行集/算法一致，残差为 fp32 聚合顺序噪声（实测 ~8e-7/样本）。
    e98 = json.load(open(E98_JSON))
    e98_ps = e98["per_sample"]

    def e98_mean(arm):
        vals = [e98_ps[s][arm] for s in e98_ps if arm in e98_ps[s]]
        return sum(vals) / len(vals)
    champ_e98 = e98_mean("mavg_a0.25_b0.125_g0.625")
    anchor_val = mean.get("anchor_e98_sk128_sw1024")
    ps_diffs = [abs(per_sample[s]["anchor_e98_sk128_sw1024"] -
                    e98_ps[s]["mavg_a0.25_b0.125_g0.625"])
                for s in per_sample
                if "anchor_e98_sk128_sw1024" in per_sample[s]
                and s in e98_ps and "mavg_a0.25_b0.125_g0.625" in e98_ps[s]]
    anchor_diff = None if anchor_val is None else abs(anchor_val - champ_e98)
    # ---- 报告表 ----
    grid = {}
    for swa in SWA_GRID:
        for sink in SINK_GRID:
            a = mean.get(f"mavg_champ_sk{sink}_sw{swa}")
            b = mean.get(f"sinkswa_only_sk{sink}_sw{swa}")
            grid[f"sk{sink}_sw{swa}"] = {
                "champion_mass": round(a, 6) if a is not None else None,
                "sinkswa_only_mass": round(b, 6) if b is not None else None,
                "indexer_gain": round(a - b, 6) if (a is not None and b is not None) else None,
            }
    base = grid["sk128_sw128"]
    out = {
        "experiment": "E135 sink/swa 敏感性 mass 扫参 + sinkswa_only 归因（E98/E130 同协议）",
        "protocol": f"TRACE={TRACE} 16 样本, B_TOK={B_TOK} BP={BP} BS={BS} TAIL_N={TAIL_N}, "
                    "真实全维 softmax 行级覆盖率; 行过滤按网格最坏点 (sk16,sw512) mid_len≥4096 统一; "
                    "冠军 (α,β,γ)=(0.25,0.125,0.625) E98 gamma_arm 克隆; "
                    "子块 sink 页池左界=floor(sink/bs)（生产 E135 语义同源）",
        "anchor_check": {
            "e98_champion_recomputed": round(champ_e98, 9),
            "anchor_e98_sk128_sw1024": round(anchor_val, 9) if anchor_val is not None else None,
            "abs_diff": anchor_diff,
            "per_sample_abs_diffs": [round(d, 10) for d in ps_diffs],
            "n_samples_checked": len(ps_diffs),
            "pass": len(ps_diffs) > 0 and max(ps_diffs) < 1e-5,
        },
        "grid": grid,
        "per_sample": per_sample,
        "mean": mean,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("=" * 72)
    print("[E135] sink/swa 敏感性 + 归因 mass 报告")
    ps_max = max(ps_diffs) if ps_diffs else float("nan")
    print(f"锚点自校验(逐样本): n={len(ps_diffs)} max|diff|={ps_max:.2e} "
          f"{'PASS' if out['anchor_check']['pass'] else 'FAIL'} "
          f"(e98冠军均值={champ_e98:.9f} 本器锚点均值={anchor_val})")
    print(f"{'(sink,swa)':<16}{'champion':<12}{'sinkswa_only':<14}{'indexer_gain':<13}{'Δvs(128,128)'}")
    for swa in SWA_GRID:
        for sink in SINK_GRID:
            g = grid[f"sk{sink}_sw{swa}"]
            print(f"sk{sink:<4}sw{swa:<4}{g['champion_mass']:<12}{g['sinkswa_only_mass']:<14}"
                  f"{g['indexer_gain']:<13}"
                  f"{g['champion_mass'] - base['champion_mass']:+.6f}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
