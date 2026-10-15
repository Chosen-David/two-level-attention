# -*- coding: utf-8 -*-
"""E140e O4 选择行为 trace 采集（figure_id=O4，补充 O1 的 oracle 视角）。

O1 回答「质量在哪里」（dense mass 分布 = 理想选择的先验）；本 trace 回答
「选择器实际选了哪里」——同一批 Qwen3-8B 真实 LongBench 样本（hotpotqa/
musique 各 2 条，16K 截断）、同一批层 {8,24}、同一批 query 位置（16/样本，
与 e140_o_trace.py 同种子同网格），逐 query 逐 KV-group 做三种选择并记录
选中 token 的距离分布（64-token 距离箱，与 O1 bins 同口径）：

  - oracle：full-128 维组均值概率 P（E0 口径：逐 Q-head softmax → 组内均值），
    mid 区纯 top-768（K2_mid=768，sink/swa 强制区不占预算、不参与比较）；
  - tli_k128 / tli_k64：生产 mavg 口径两级选择的逐字段复刻——
      L1：near 池 avg 页分（q·k_avg 块均值，tail32 子空间）top nb_near 页 +
          far 池 minmax 上界页分（Σ_d max(q·kmax, q·kmin)，tail32）top nb_far 页
          （nb_near=max(1,round(K1·β))，nb_far=K1−nb_near，tli_indexer.py
          L1947-1950 同公式）；
      L2：候选页内 token 级 tail32 细筛分 → 逐 head softmax（候选外 -inf，与
          生产 where(topk_mask,sf,-inf)→softmax 同口径）→ 组内均值 p →
          near 配额 nt_near=min(nb_near·bs·γ, K2_mid) / far 配额 K2_mid−nt_near
          分区 topk（L2300-2305 同公式——E140 far 配额归零机制的消费端）；
  - 双臂对照目的：K1=128（生产默认，far 配额 128）vs K1=64（E140 甜点，
    far 配额 448）——位置分布上应直接可见 far 区覆盖差。

区域几何逐字段复刻 compute_mask（tli_indexer.py L1880-1897，e64 分区臂）：
  mid_len = S_real−sink−swa；near_len_dyn = max(bs, int(α·mid_len))；
  near_base = S_real−swa；near 池左页 = max(sink_pages, (near_base−near_len_dyn)//bs)；
  far 池 = [sink_pages, near 池左页)。页边界 floor 对齐（生产保守多含口径）。

记录（逐 query 逐组）：每臂每 64-token 距离箱的选中 token 数 + 箱内 token 数
（density = Σsel/Σn_tok 聚合，与 O1 面板 A 的 log 网格聚合兼容）、near/far
分区选中数、与 oracle 的 mid 交集 recall（|∩|/768）、retained mass
（选中集的 full-ref 概率质量和 / oracle 集 mass，O3 同口径）。

统计纪律（observation.md）：head/query 为重复观测，独立单位 = 样本文档；
cells 保留 task/sample/layer/query 粒度可回查。sink/swa 强制区三臂相同，
只入摘要不入分布比较。

输出：exp/trace/results/e140_o4_select_trace.json
"""
import json
import os

import torch
import torch.nn.functional as F

DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
TASKS = ["hotpotqa", "musique"]
N_SAMPLE = 2
LAYERS = [8, 24]
MAXLEN = 16384
N_QUERY = 16
ALPHA, BETA, GAMMA, BS = 0.25, 0.125, 0.625, 64
K2_MID = 768            # K2=1024 − sink 128 − swa 128
SINK, SWA = 128, 128
ARMS = {"oracle": None, "tli_k128": 128, "tli_k64": 64}

# tail32：低频尾 16 完整 pairs（生产 _subspace_indices cmp4 同口径，O3 同）
SUB32 = list(range(48, 64)) + list(range(112, 128))

DATA = os.environ.get("LB_DATA", os.path.expanduser(
    "~/datasets/LongBench/data"))
OUT = os.environ.get("E140_OUT", os.path.expanduser(
    "~/two-level-attention/exp/trace/results/e140_o4_select_trace.json"))
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
    return tok, model, float(model.config.rope_theta)


def get_qk(model, theta, input_ids):
    """一次前向拿 hidden_states，目标层手动 q/k_proj + q/k_norm + RoPE
    （与 e140_o_trace.py 逐位同流程）。"""
    with torch.no_grad():
        out = model(input_ids=input_ids.to(DEV), output_hidden_states=True,
                    use_cache=False)
    qs, ks = {}, {}
    inv_freq = 1.0 / (theta ** (torch.arange(0, 64, device=DEV,
                                             dtype=torch.float32) / 64))
    pos = torch.arange(input_ids.shape[1], device=DEV, dtype=torch.float32)
    emb = torch.cat((torch.outer(pos, inv_freq),) * 2, dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    for L in LAYERS:
        hs = out.hidden_states[L]
        attn = model.model.layers[L].self_attn
        q = attn.q_norm(attn.q_proj(hs).view(1, -1, 32, 128))
        k = attn.k_norm(attn.k_proj(hs).view(1, -1, 8, 128))
        qs[L] = (q * cos[None, :, None] +
                 rotate_half(q) * sin[None, :, None])[0].float()   # (T,32,128)
        ks[L] = (k * cos[None, :, None] +
                 rotate_half(k) * sin[None, :, None])[0].float()   # (T,8,128)
    del out
    torch.cuda.empty_cache()
    return qs, ks


def group_mean_probs(q_row, k_all, scale):
    """q_row (32,128) k_all (T',8,128) → 逐 head softmax → 组内均值 (8,T')。"""
    g = 32 // 8
    q_g = q_row.view(8, g, -1)
    scores = torch.einsum("ghd,tgd->ght", q_g, k_all) * scale
    return torch.softmax(scores, dim=-1).mean(dim=1)


def zone_geometry(q_pos):
    """生产 e64 分区几何（token 级）：返回 mid/near/far 区间与页边界。"""
    S_real = q_pos + 1
    mid_len = max(0, S_real - SINK - SWA)
    near_len = max(BS, int(ALPHA * mid_len))
    near_base = S_real - SWA                       # near 右界 = swa 起点
    near_lo_pg = max(SINK // BS, (near_base - near_len) // BS)
    near_lo = near_lo_pg * BS                      # 页对齐（floor，保守多含）
    return dict(S_real=S_real, mid_lo=SINK, mid_hi=near_base,
                near_lo=near_lo, near_base=near_base, near_len=near_len)


def tli_select(q_row, k_all, geo, K1, idx):
    """生产 mavg 两级选择（tail32 子空间），逐组返回 mid 选中 token 掩码 (8, n_mid)。

    q_row (32,128) float32；k_all (T',8,128)；geo 见 zone_geometry；idx (32,) 子空间坐标。
    """
    T = k_all.shape[0]
    mid_lo, mid_hi = geo["mid_lo"], geo["mid_hi"]
    near_lo = geo["near_lo"]
    n_mid = mid_hi - mid_lo
    n_pg = n_mid // BS                                # mid 内完整页数（尾巴页含入最后一页）
    qs = q_row.view(8, 4, 128)[..., idx]              # (8,4,32)
    ks = k_all[:, :, idx]                             # (T',8,32)
    scale = 128 ** -0.5
    tok_score = torch.einsum("ghd,tgd->ght", qs, ks) * scale   # (8,4,T') 细筛分
    k_mid = ks[mid_lo:mid_hi]                         # (n_mid,8,32)

    sel = torch.zeros(8, n_mid, dtype=torch.bool, device=DEV)
    nb_near = max(1, round(K1 * BETA))
    nb_far = max(0, K1 - nb_near)
    nt_near = min(round(nb_near * BS * GAMMA), K2_MID)
    far_budget = max(0, K2_MID - nt_near)             # E140 配额公式
    near_pg_lo = (near_lo - mid_lo) // BS             # mid 内页坐标
    far_pg_hi = near_pg_lo
    for g in range(8):
        # ---- L1 页分（生产分型：near=avg / far=minmax 上界）----
        kg = k_mid[:, g, :]                           # (n_mid,32)
        pg_avg = kg[: n_pg * BS].view(n_pg, BS, -1).mean(dim=1)     # 页均值 k
        pg_max = kg[: n_pg * BS].view(n_pg, BS, -1).amax(dim=1)
        pg_min = kg[: n_pg * BS].view(n_pg, BS, -1).amin(dim=1)
        qm = qs[g].mean(dim=0)                        # 组均值 q (32,)
        near_sc = pg_avg @ qm                         # avg 页分
        far_sc = (torch.where(qm >= 0, qm * pg_max, qm * pg_min)
                  ).sum(dim=-1)                       # minmax 上界页分
        n_near_pool = n_pg - near_pg_lo
        k_n = min(nb_near, max(n_near_pool, 0))
        i_near = near_sc[near_pg_lo:].topk(k_n).indices + near_pg_lo \
            if k_n > 0 else torch.empty(0, dtype=torch.long, device=DEV)
        k_f = min(nb_far, max(far_pg_hi, 0))
        i_far = far_sc[:far_pg_hi].topk(k_f).indices \
            if k_f > 0 else torch.empty(0, dtype=torch.long, device=DEV)
        cand_pg = torch.cat([i_far, i_near]).unique()
        cand_tok = (cand_pg[:, None] * BS +
                    torch.arange(BS, device=DEV)[None]).reshape(-1)  # mid 内坐标
        # ---- L2：候选内逐 head softmax → 组均值 p → 分区配额 topk ----
        s_c = tok_score[g][:, mid_lo:mid_hi][:, cand_tok]            # (4, n_cand)
        p = torch.softmax(s_c, dim=-1).mean(dim=0)                   # (n_cand,)
        in_near = cand_tok >= (near_lo - mid_lo)
        c_near = cand_tok[in_near]
        c_far = cand_tok[~in_near]
        k2n = min(nt_near, c_near.numel())
        k2f = min(far_budget, c_far.numel())
        if k2n > 0:
            sel[g, c_near[p[in_near].topk(k2n).indices]] = True
        if k2f > 0:
            sel[g, c_far[p[~in_near].topk(k2f).indices]] = True
    return sel


def oracle_select(P, mid_lo, mid_hi):
    """oracle：mid 区组均值概率纯 top-K2_MID，返回 (8, n_mid) 掩码。"""
    mid_p = P[:, mid_lo:mid_hi]
    k = min(K2_MID, mid_p.shape[1])
    sel = torch.zeros_like(mid_p, dtype=torch.bool)
    sel.scatter_(1, mid_p.topk(k, dim=1).indices, True)
    return sel


def bin_counts(sel, q_pos, mid_lo, mid_hi):
    """选中掩码 (8, n_mid) → 64-token 距离箱计数（与 O1 bins 同口径）。"""
    n_mid = mid_hi - mid_lo
    counts = []
    for b0 in range(0, n_mid, BS):
        b1 = min(b0 + BS, n_mid)
        counts.append({
            "d_lo": int(q_pos - mid_lo - b1 + 1),
            "d_hi": int(q_pos - mid_lo - b0),
            "n_tok": int((b1 - b0) * 8),              # ×8 组（逐组独立选择）
            "n_sel": int(sel[:, b0:b1].sum())})
    return counts


def main():
    torch.cuda.set_device(DEV)
    idx = torch.tensor(SUB32, device=DEV)
    print(f"=== E140e O4 select-behavior trace GPU {DEV} ===", flush=True)
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
                for qp in qpos:
                    q_row, k_pre = qL[qp], kL[: qp + 1]
                    P = group_mean_probs(q_row, k_pre, scale)   # (8,T')
                    geo = zone_geometry(qp)
                    mid_lo, mid_hi = geo["mid_lo"], geo["mid_hi"]
                    sels = {"oracle": oracle_select(P, mid_lo, mid_hi)}
                    for name, k1 in ARMS.items():
                        if k1 is not None:
                            sels[name] = tli_select(q_row, k_pre, geo, k1, idx)
                    ref = sels["oracle"]
                    mid_p = P[:, mid_lo:mid_hi]
                    ref_mass = mid_p.masked_select(ref).sum()
                    rec = {"q_pos": int(qp), "near_lo": geo["near_lo"],
                           "near_len": geo["near_len"], "mid_lo": mid_lo,
                           "mid_hi": mid_hi, "arms": {}}
                    for name, s in sels.items():
                        ov = float((s & ref).sum()) / float(ref.sum())
                        rm = float(mid_p.masked_select(s).sum() / ref_mass)
                        nz = s[:, geo["near_lo"] - mid_lo:].sum()
                        rec["arms"][name] = {
                            "n_sel": int(s.sum()),
                            "near_sel": int(nz),
                            "far_sel": int(s.sum() - nz),
                            "overlap_recall": ov, "retained_mass": rm,
                            "bins": bin_counts(s, qp, mid_lo, mid_hi)}
                    results.append({"task": task, "sample": si, "layer": L,
                                    "T": int(T), "o4": rec})
                print(f"  layer {L}: {len(qpos)} queries done", flush=True)
            del qs, ks
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meta = {"model": MODEL, "tasks": TASKS, "n_sample": N_SAMPLE,
            "layers": LAYERS, "maxlen": MAXLEN, "n_query": N_QUERY,
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "bs": BS,
            "k2_mid": K2_MID, "sink": SINK, "swa": SWA,
            "subspace": "tail32 (pairs 24-31, production cmp4)",
            "arms": {k: (f"production-typed two-level, K1={v}" if v else
                         "full-128d group-mean prob mid top-768")
                     for k, v in ARMS.items()},
            "quota_formula": "nt_near=min(round(K1*b)*bs*g, K2_mid); "
                             "far_budget=K2_mid-nt_near (E140 mechanism)",
            "note": "selection BEHAVIOR trace (where tokens land); "
                    "complements O1 mass trace (where mass lives)"}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
