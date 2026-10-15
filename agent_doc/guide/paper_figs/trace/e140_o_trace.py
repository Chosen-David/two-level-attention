# -*- coding: utf-8 -*-
"""E140d O1/O3 observation trace 采集（figure_id=O1/O3，P0 主文）。

O1 距离分布：Qwen3-8B 真实 LongBench 样本（hotpotqa/musique 各 2 条，16K 截断），
layer {8,24}，手动 q/k_proj + RoPE 取 post-RoPE Q/K，GQA 组内逐 Q-head softmax
→ 组内概率均值（E0 口径）。逐 query 采样位置（16/样本）统计：
  - 保护区 sink[0,128) / swa(q-128,q] 单独记账
  - 非保护 mid 按 64-token 距离箱 mass 与 token 数（density = mass/tokens）
  - near/far 边界按生产公式 α=.25 64 对齐（near_len = floor(α·mid_len/64)·64）
  - far mass 组均值 + 组内最差头摘要
O3 频段对照：同 Q/K，32 维子空间三臂（tail=低频尾 pairs24-31 / highfreq=pairs0-7 /
random-unpaired=seed20261002 诊断臂），同规则两层选择（L1 页级 avg 分 top-K1=128 页
→ L2 候选页内 token 分 top-768），对照 full-ref 组均值概率 TopK-768：
  - L1 page recall（参考 token 被选中页覆盖比例）
  - 最终 overlap recall（|sel∩ref|/768）
  - retained mass（sel 集合 full-ref 概率质量和 / ref 集 mass）
口径声明：L1 页分用统一 avg（生产 far 池为 minmax、near 池 avg 的分型混合——本对照
三臂同规则公平，非生产 mavg 完整复刻）；random 臂为 unpaired 诊断（spec A2b 标注）。

输出：exp/trace/results/e140_o_trace.json（O1 bins + O3 三臂指标，逐格可回查）。
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
TASKS = ["hotpotqa", "musique"]
N_SAMPLE = 2
LAYERS = [8, 24]
MAXLEN = 16384
N_QUERY = 16
ALPHA, BS = 0.25, 64
K1, K2MID = 128, 768   # 页数 / 非保护 token 预算
SINK, SWA = 128, 128

SUBSPACES = {
    # tail：低频尾 16 完整 pairs（标量 [48,64)∪[112,128)——生产 _subspace_indices 同口径）
    "tail32": list(range(48, 64)) + list(range(112, 128)),
    # highfreq：高频头 16 完整 pairs（[0,16)∪[64,80)——E90 消融臂同口径）
    "high32": list(range(0, 16)) + list(range(64, 80)),
    # random-unpaired：seed20261002 randperm(128)[:32]（旧实现诊断臂）
    "random32": None,
}

DATA = os.environ.get("LB_DATA", os.path.expanduser(
    "~/datasets/LongBench/data"))
OUT = os.environ.get("E140_OUT", os.path.expanduser(
    "~/two-level-attention_git/exp/trace/results/e140_o_trace.json"))
MODEL = os.environ.get("QWEN_MODEL", os.path.expanduser("~/Qwen3-8B"))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    assert getattr(model.config, "rope_scaling", None) is None, \
        "rope_scaling 存在——手写 RoPE 需适配，fail loudly"
    theta = float(model.config.rope_theta)
    return tok, model, theta


def get_qk(model, theta, input_ids):
    """一次前向拿 hidden_states，对目标层手动 q/k_proj + RoPE。"""
    with torch.no_grad():
        out = model(input_ids=input_ids.to(DEV), output_hidden_states=True,
                    use_cache=False)
    qs, ks = {}, {}
    inv_freq = 1.0 / (theta ** (torch.arange(0, 64, device=DEV,
                                              dtype=torch.float32) / 64))
    pos = torch.arange(input_ids.shape[1], device=DEV, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # (T, 64)
    emb = torch.cat((freqs, freqs), dim=-1)         # (T, 128)
    cos = emb.cos().to(torch.bfloat16)              # (T,128)
    sin = emb.sin().to(torch.bfloat16)
    for L in LAYERS:
        hs = out.hidden_states[L]                   # (1,T,hidden) bf16
        attn = model.model.layers[L].self_attn
        q = attn.q_proj(hs).view(1, -1, 32, 128)    # (1,T,H,D)
        k = attn.k_proj(hs).view(1, -1, 8, 128)     # (1,T,Hkv,D)
        # Qwen3 per-head RMSNorm（q_norm/k_norm 必须在 RoPE 前加——
        # 诊断实证：跳过时高范数 token 主导 score，sink mass 病态 1.0；
        # 含 norm 后 sink .18/swa .29/mid .52 与生产 attention 一致）
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        # RoPE：q·cos + rotate_half(q)·sin（Qwen3 全维旋转，无 partial）
        qr = (q * cos.unsqueeze(0).unsqueeze(2) +
              rotate_half(q) * sin.unsqueeze(0).unsqueeze(2))
        kr = (k * cos.unsqueeze(0).unsqueeze(2) +
              rotate_half(k) * sin.unsqueeze(0).unsqueeze(2))
        qs[L] = qr[0].float()                       # (T,32,128)
        ks[L] = kr[0].float()                       # (T,8,128)
    del out
    torch.cuda.empty_cache()
    return qs, ks


def group_mean_probs(q_row, k_all, scale):
    """q_row (32,128) k_all (T,8,128) → 组内逐 head softmax 概率均值 (8,T)。
    scale = 1/sqrt(128)（原始全维口径）。GQA：32 Q-head 按 8 组 × 4 映射。"""
    Hkv = k_all.shape[1]
    Hq = q_row.shape[0]
    g = Hq // Hkv
    q_g = q_row.view(Hkv, g, -1)                        # (8,4,128)
    scores = torch.einsum("ghd,tgd->ght", q_g, k_all) * scale  # (8,4,T)
    probs = torch.softmax(scores, dim=-1)               # 逐 head softmax
    return probs.mean(dim=1)                            # 组内均值 (8,T)


def near_len_for(q_pos):
    mid_len = q_pos - SINK - SWA
    if mid_len <= 0:
        return 0
    return max(BS, (int(ALPHA * mid_len) // BS) * BS)


def o1_trace(P, q_pos):
    """P (8,T) 组均值概率 → 距离分箱 + 保护区 + near/far 记账。"""
    T = P.shape[1]
    mass = P.mean(dim=0)                             # 跨组均值 (T,)
    sink_m = mass[:SINK].sum().item()
    swa_lo = max(0, q_pos - SWA)
    swa_m = mass[swa_lo:q_pos + 1].sum().item()
    mid_lo, mid_hi = SINK, q_pos - SWA               # 非保护区间 [128, q-128)
    mid = mass[mid_lo:mid_hi]
    nl = near_len_for(q_pos)
    near_m = mid[:nl].sum().item() if nl > 0 else 0.0
    far_m = mid[nl:].sum().item() if nl > 0 else mid.sum().item()
    # 64-token 距离箱（在非保护 mid 上，d = q_pos - k_pos ∈ [SWA+1, q_pos-SINK]）
    bins = []
    n_mid = mid_hi - mid_lo
    for b0 in range(0, n_mid, BS):
        b1 = min(b0 + BS, n_mid)
        bins.append({
            "d_lo": int(q_pos - mid_lo - b1 + 1), "d_hi": int(q_pos - mid_lo - b0),
            "mass": float(mid[b0:b1].sum()), "n_tok": int(b1 - b0)})
    far_per_group = P[:, mid_lo + nl:mid_hi].sum(dim=1) if nl > 0 else P[:, mid_lo:mid_hi].sum(dim=1)
    return {"mass_sink": sink_m, "mass_swa": swa_m, "mass_near": near_m,
            "mass_far": float(far_m), "near_len": nl, "q_pos": int(q_pos),
            "far_worst_head": float(far_per_group.max()),
            "bins": bins}


def o3_trace(q_row, k_all, scale, sub_idx):
    """三臂同规则两层选择 vs full-ref TopK。逐 KV-group 独立选择，返回 per-group 指标。
    坐标统一用 mid 内偏移；mid 池 = [SINK, T'-SWA-1)（T'=因果前缀长 qp+1 含 query 自身，
    与 O1 的 mid 区间 [128, q-128) 一致）。sub_idx: 32 个标量坐标（None=random32）。"""
    T = k_all.shape[0]
    g = torch.Generator().manual_seed(20261002)
    if sub_idx is None:
        sub_idx = torch.randperm(128, generator=g)[:32].sort().values.tolist()
    idx = torch.tensor(sub_idx, device=DEV)
    P = group_mean_probs(q_row, k_all, scale)        # (8,T')
    mid_lo, mid_hi = SINK, T - SWA - 1               # 与 O1 mid 区间一致
    n_mid = mid_hi - mid_lo
    metrics = []
    for grp in range(8):
        mid_p = P[grp, mid_lo:mid_hi]                # (n_mid,)
        k_sel = min(K2MID, n_mid)
        ref = mid_p.topk(k_sel).indices              # mid 坐标
        ref_mass = mid_p[ref].sum()
        # ---- selector 臂（32 维，组内 4 Q-head 均值分）----
        q_s = q_row[grp * 4: (grp + 1) * 4][:, idx]         # (4,32)
        k_s = k_all[:, grp, :][:, idx]                        # (T',32)
        s_tok = torch.einsum("hd,td->t", q_s, k_s) * (128 ** -0.5)
        mid_s = s_tok[mid_lo:mid_hi]
        # L1：64-token 页 avg 分 → top-K1 页（非保护池）
        n_pg = n_mid // BS
        pg = mid_s[: n_pg * BS].view(n_pg, BS).mean(dim=1)   # (n_pg,)
        top_pages = pg.topk(min(K1, n_pg)).indices           # 页 id（mid 内）
        cand = torch.cat([torch.arange(int(p) * BS, int(p) * BS + BS, device=DEV)
                          for p in top_pages.tolist()])       # mid 坐标候选 token
        page_cov = float(torch.isin(ref, cand).float().mean())
        # L2：候选 token 32 维分 top-768
        k2 = min(K2MID, cand.numel())
        sel = cand[mid_s[cand].topk(k2).indices]
        ov = float(torch.isin(sel, ref).float().mean())       # overlap recall
        ret_mass = float(mid_p[sel].sum() / ref_mass)
        metrics.append({"page_recall": page_cov, "overlap_recall": ov,
                        "retained_mass": ret_mass})
    return metrics


def main():
    torch.cuda.set_device(DEV)
    print(f"=== E140d O1/O3 trace GPU {DEV} ===", flush=True)
    tok, model, theta = load_model()
    results = []
    for task in TASKS:
        path = os.path.join(DATA, f"{task}.jsonl")
        samples = [json.loads(l) for l in open(path)][:N_SAMPLE]
        for si, smp in enumerate(samples):
            text = smp["context"] + "\n\n" + smp["input"]
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=MAXLEN).input_ids
            T = ids.shape[1]
            print(f"[{task} s{si}] T={T}", flush=True)
            qs, ks = get_qk(model, theta, ids)
            qpos = torch.linspace(SINK + SWA + 256, T - 64, N_QUERY).long().tolist()
            for L in LAYERS:
                qL, kL = qs[L], ks[L]
                scale = 128 ** -0.5
                o1_cells, o3_cells = [], {k_: [] for k_ in SUBSPACES}
                for qp in qpos:
                    q_row = qL[qp]                     # (32,128)
                    k_all = kL[: qp + 1]               # 因果前缀 (T',8,128)
                    P = group_mean_probs(q_row, k_all, scale)
                    o1_cells.append(o1_trace(P, qp))
                    # O3：mid 池 = [SINK, qp-SWA)（与 O1 同界），k_all 前缀内
                    k_pre = k_all
                    for name, sub in SUBSPACES.items():
                        o3_cells[name].append(o3_trace(q_row, k_pre, scale, sub))
                # O3 汇总（跨 query × group mean/max）
                o3_sum = {}
                for name, cells in o3_cells.items():
                    import statistics as st
                    flat = [m for cell in cells for m in cell]
                    o3_sum[name] = {
                        k2: {"mean": st.mean(m[k2] for m in flat),
                             "worst": min(m[k2] for m in flat)}
                        for k2 in ("page_recall", "overlap_recall", "retained_mass")}
                results.append({
                    "task": task, "sample": si, "layer": L, "T": int(T),
                    "o1": o1_cells, "o3": o3_sum})
                print(f"  layer {L}: O1 {len(o1_cells)} queries, O3 3 arms done",
                      flush=True)
            del qs, ks
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meta = {"model": MODEL, "tasks": TASKS, "n_sample": N_SAMPLE,
            "layers": LAYERS, "maxlen": MAXLEN, "alpha": ALPHA, "bs": BS,
            "K1": K1, "K2_mid": K2MID, "sink": SINK, "swa": SWA,
            "subspaces": {k: (v if v else "random-unpaired seed20261002")
                          for k, v in SUBSPACES.items()},
            "o3_rule": "L1 page-avg top-K1 -> L2 token top-768, per-group, "
                       "same rule across arms (production far-minmax/near-avg "
                       "typing NOT replicated; diagnostic frequency contrast)",
            "o1_rule": "GQA per-head softmax -> group mean prob; sink/swa "
                       "separate; 64-token distance bins on mid"}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
