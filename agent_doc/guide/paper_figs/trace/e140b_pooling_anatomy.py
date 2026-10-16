# -*- coding: utf-8 -*-
"""E140b pooling anatomy probe：单一 pooling（avg）vs minmax 上界 vs oracle 的页级选择对照。

动机：图 (a) 需要在真实数据上展示「单一 avg pooling 把 far 稀疏尖峰页稀释误筛，
minmax 无漏上界保留之」的机理。v2 JSON 只有 64-token bin 聚合质量，无法分解
页内分布，因此本 probe 重采 token 级 32 维共享子空间分数。

口径（与 e140_o_trace.py v2 完全一致的部分直接 import 复用）：
  - Q/K = 实际 q_norm/k_norm 输出 + 实际 position_embeddings RoPE（get_qk）
  - 真实质量 = GQA 组内逐 head softmax 后组均（group_mean_probs），token 级
  - 子空间 = tail32（[48,64)∪[112,128)，16 完整低频对，生产 _subspace_indices 同口径）
  - token 诊断分 s_i = 32 个 Q-head 的 z_q·z_k（各 head 用本组 K）均值 × 128^-0.5
    （诊断简化：生产 L1 按 KV 组内 4 Q-head 均值、逐 KV 头选页；此处全局单一
    选择面，与图 (a) 的 GQA 组均质量口径一致）
  - 页 = o1 的 64-token bin（mid 内，[128, mid_hi)，末页可能不足 64）
  - 单池预算 = 2048 token（与图 (a) oracle 单池条带相同）：按页分降序，
    接纳前检查 acc + n_tok <= 预算，放不下则跳过该页继续（GPT 2026-10-16
    审查修正：旧版先接纳后检查，128 条中至多 97 条超额、最大 2110 token）
  - 三臂：oracle（真实页质量和）、avg（页内 token 分均值）、
    minmax（Quest 上界 U_b = Σ_j z_q+·M_j + z_q-·m_j，逐 Q-head 计算后取 32 head 均值）

输出（'x' 模式，不覆盖历史）：每 query × cell 的逐页数组
（mass/avg/minmax/选中标记）+ 汇总（avg 漏掉的 oracle 页数与质量、minmax 救回数）
+ 每 query 的 far-spike 代表页与 near 均匀代表页的 token 级细节（供解剖放大面板）。

诊断用途：机理展示，非生产打分器复刻，非任务质量结论。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import importlib

import torch

REPO = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, os.path.join(REPO, "agent_doc", "guide", "paper_figs", "trace"))
e = importlib.import_module("e140_o_trace")  # noqa: E402  复用 v2 采集器数学口径

OUT = os.path.join(REPO, "exp", "trace", "results", "e140b_pooling_anatomy.json")
BUDGET_TOK = 2048   # 单池 mid token 预算（与图 (a) 条带一致）
J = torch.tensor(list(range(48, 64)) + list(range(112, 128)))  # tail32
SCALE = 128 ** -0.5

# 用 v2 记录的真实声明（本机同一份本地模型/数据；身份声明沿用，不重新捏造）
_v2 = json.load(open(os.path.join(REPO, "exp", "trace", "results", "e140_o_trace_v2.json")))
for k, v in _v2["meta"]["declared_revisions"].items():
    os.environ.setdefault(k, v)


def token_scores(q_row, k_all):
    """q_row (32,128) k_all (T',8,128) → 32-head 均值的 tail32 token 分 (T',)。"""
    dev = q_row.device
    idx = J.to(dev)
    zq = q_row[:, idx]                                # (32,32)
    zk = k_all[:, :, idx]                             # (T',8,32)
    g = q_row.shape[0] // k_all.shape[1]              # 4
    zq_g = zq.reshape(k_all.shape[1], g, -1)          # (8,4,32)
    s = torch.einsum("ghd,tgd->ght", zq_g, zk) * SCALE  # (8,4,T')
    return s.mean(dim=(0, 1))                          # (T',)


def minmax_bounds(q_row, k_all, pages):
    """逐页 Quest 上界（32 head 均值）。pages: list[(b0,b1)] 绝对位置。"""
    dev = q_row.device
    idx = J.to(dev)
    zq = q_row[:, idx].float()                        # (32,32)
    zp, zn = zq.clamp(min=0), zq.clamp(max=0)
    bounds = []
    for b0, b1 in pages:
        zb = k_all[b0:b1][:, :, idx].float()          # (n,8,32)
        g = zq.shape[0] // k_all.shape[1]
        # Q head h 属于组 h//g：复制各组 K 到其 4 个 Q head（不可 reshape）
        zb_g = zb.repeat_interleave(g, dim=1)         # (n,32,32)
        M = zb_g.amax(dim=0)                          # (32,32) per q-head
        m = zb_g.amin(dim=0)
        U = (zp * M + zn * m).sum(-1) * SCALE         # (32,)
        bounds.append(float(U.mean()))
    return bounds


def select_pages(order_vals, ntoks, budget=BUDGET_TOK):
    """按分值降序逐页累积；接纳前检查 acc+n_tok<=budget，放不下则跳过该页。

    （与 fig_o_observation.py 同逻辑；GPT 审查修正：旧版先接纳后 break，
    末页不足 64 token 时超额至 2110。跳过而非 break：后续更小的页仍可填。）
    """
    sel = set()
    acc = 0
    for i in sorted(range(len(order_vals)), key=lambda i: -order_vals[i]):
        if acc + ntoks[i] > budget:
            continue
        sel.add(i)
        acc += ntoks[i]
    return sel


def main():
    if Path(OUT).exists():
        if "--overwrite" in sys.argv:
            # 预算修正重跑：旧 JSON（GPT 冻结哈希 3327e9…）被本版取代，
            # 取代记录见 observation/e140_validity.json 与审查回复
            Path(OUT).unlink()
        else:
            raise SystemExit(f"output exists (x-mode): {OUT}")
    torch.cuda.set_device(e.DEV)
    tok, model, _ = e.load_model()
    cells_out = []
    for task in e.TASKS:
        path = os.path.join(e.DATA, f"{task}.jsonl")
        samples = [json.loads(l) for l in open(path)][:e.N_SAMPLE]
        for si, smp in enumerate(samples):
            text = smp["context"] + "\n\n" + smp["input"]
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=e.MAXLEN).input_ids
            T = ids.shape[1]
            qs, ks = e.get_qk(model, None, ids)
            print(f"[{task} s{si}] T={T}", flush=True)
            for L in e.LAYERS:
                qL, kL = qs[L], ks[L]
                qcells = []
                for qp in e.query_schedule(T):
                    q_row = qL[qp]
                    k_all = kL[: qp + 1]
                    P = e.group_mean_probs(q_row, k_all, SCALE)
                    mass = P.mean(0)                  # (T',) token 级真实质量
                    r = e.regions(qp)
                    lo, hi, nl = r["mid_lo"], r["mid_hi"], r["near_lo"]
                    s_tok = token_scores(q_row, k_all)
                    pages, p_mass, p_avg, p_ntok = [], [], [], []
                    for b0 in range(lo, hi, e.BS):
                        b1 = min(b0 + e.BS, hi)
                        pages.append((b0, b1))
                        p_mass.append(float(mass[b0:b1].sum()))
                        p_avg.append(float(s_tok[b0:b1].mean()))
                        p_ntok.append(b1 - b0)
                    p_mm = minmax_bounds(q_row, k_all, pages)
                    sel_o = select_pages(p_mass, p_ntok)
                    sel_a = select_pages(p_avg, p_ntok)
                    sel_m = select_pages(p_mm, p_ntok)
                    missed = sel_o - sel_a
                    recov = missed & sel_m
                    o_mass = sum(p_mass[i] for i in sel_o)
                    miss_mass = sum(p_mass[i] for i in missed)
                    # far/near 页归属（页中心）
                    far_pg = [i for i, (b0, b1) in enumerate(pages)
                              if (b0 + b1) / 2 < nl]
                    miss_far = [i for i in missed if i in set(far_pg)]
                    # 代表页：avg 漏掉的 oracle far 页中质量最高者（尖峰页）；
                    # near 区内 oracle 选中且 avg 也选中的最均匀页（CV 最小）
                    spike, uniform = None, None
                    if miss_far:
                        spike = max(miss_far, key=lambda i: p_mass[i])
                    near_sel = [i for i in (sel_o & sel_a)
                                if i not in set(far_pg)]
                    if near_sel:
                        def cv(i):
                            b0, b1 = pages[i]
                            v = mass[b0:b1]
                            return float(v.std() / v.mean().clamp(min=1e-12))
                        uniform = min(near_sel, key=cv)
                    detail = {}
                    for tag, pi in (("spike", spike), ("uniform", uniform)):
                        if pi is None:
                            continue
                        b0, b1 = pages[pi]
                        detail[tag] = dict(
                            page_index=pi, pos_lo=b0, pos_hi=b1,
                            token_mass=[float(x) for x in mass[b0:b1]],
                            token_score=[float(x) for x in s_tok[b0:b1]],
                            page_mass=p_mass[pi], page_avg=p_avg[pi],
                            page_minmax=p_mm[pi],
                            in_oracle=pi in sel_o, in_avg=pi in sel_a,
                            in_minmax=pi in sel_m)
                    qcells.append(dict(
                        q_pos=qp, near_len=hi - nl, region_bounds=r,
                        n_pages=len(pages),
                        n_oracle=len(sel_o), n_avg_miss=len(missed),
                        n_avg_miss_far=len(miss_far),
                        n_minmax_recover=len(recov),
                        oracle_mass=o_mass,
                        miss_mass_frac=miss_mass / o_mass if o_mass > 0 else 0.0,
                        pages=dict(pos_lo=[p[0] for p in pages],
                                   n_tok=p_ntok, mass=p_mass,
                                   avg=p_avg, minmax=p_mm,
                                   sel_oracle=sorted(sel_o),
                                   sel_avg=sorted(sel_a),
                                   sel_minmax=sorted(sel_m)),
                        detail=detail))
                cells_out.append(dict(
                    task=task, sample=si, layer=L, T=int(T),
                    input_ids_sha256=hashlib.sha256(
                        ids.numpy().astype("<i8").tobytes()).hexdigest(),
                    queries=qcells))
                print(f"  L{L}: {len(qcells)} queries done", flush=True)
            del qs, ks
            torch.cuda.empty_cache()
    meta = dict(
        schema="e140b-pooling-anatomy-v2",
        status="diagnostic-mechanism-illustration",
        parent_trace="e140_o_trace_v2.json（同口径 import 复用）",
        budget_tok=BUDGET_TOK, subspace="tail32", scale=SCALE,
        score_rule="32 Q-head mean of z_q·z_k (own-group K), x 128^-0.5; "
                   "diagnostic single selection plane (production selects per KV head)",
        minmax_rule="Quest bound per Q-head, mean over 32 heads",
        selection_rule="pages ranked by score, accepted only if acc + n_tok "
                       "<= budget else skipped (v2: GPT-review budget fix; "
                       "v1 accepted-then-checked and could exceed budget)",
        supersedes="v1 sha256 3327e95d044087d5efb8315a9cf9b33e778e6a4fa028861dddedc2179324f6a7",
        declared_revisions=_v2["meta"]["declared_revisions"],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformers=_v2["meta"]["transformers_version"],
        torch=torch.__version__, device=e.DEV)
    with open(OUT, "x") as f:
        json.dump(dict(meta=meta, cells=cells_out), f)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
