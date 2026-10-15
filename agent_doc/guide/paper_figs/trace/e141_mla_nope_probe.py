# -*- coding: utf-8 -*-
"""E141 MLA 架构子空间 trace probe（DeepSeek-V2-Lite-Chat，figure_id=E141 探索）。

用户指令：GQA（Qwen3-8B）子空间结论扩展到 MLA——①只用 nope 维度打分（丢弃 rope 部分，
维度节约 192→128 = 1.5x；nope:rope=2:1）；②nope 维度内部压缩（{96,64,32} 随机子空间 +
variance 结构臂）看 retained mass 掉不掉，丰富论文「架构适用性」叙事。

口径（克隆 e140_o_trace O3）：真实 LongBench 样本（hotpotqa/musique 各 2 条，16K 截断），
layer {8,24}，手动复刻 MLA 注意力前向（q_proj / kv_a_proj_with_mqa / kv_b_proj +
kv_a_layernorm / yarn RoPE 走模型自带 rotary_emb 模块保证逐位一致，softmax_scale 直接读
attn 模块属性含 mscale²）。参考概率 = 逐 head softmax(full 192d score) → 16 head 均值
（MLA 无 GQA 分组，等价 O1 组均值口径换 head 均值）。选择 = head 均值分数单索引
（MLA 部署故事：latent 共享 → 单索引全体 head 共用），L1 页级 avg top-K1=128 页
→ L2 候选页内 token top-768，对照 head 均值概率 TopK-768。

臂（selector 打分子空间）：
  nope128   —— nope 全维（丢 rope，标题臂）
  rope64    —— rope-only 对照（纯位置信号，预期 recency 偏置）
  nope96r / nope64r / nope32r —— nope 随机子空间（seed 20261002 randperm）
  nope64hi / nope64lo —— nope 64 维按 K 方差排序 top/bottom（结构探针，
              对标 GQA tail/high 频段对照——nope 无 RoPE 无频段，方差是自然代理）

输出：exp/trace/results/e141_mla_nope_trace.json。
"""
import hashlib
import json
import os
import statistics as st
import sys

import torch

DEV = "cuda:0"
TASKS = ["hotpotqa", "musique"]
N_SAMPLE = 2
LAYERS = [4, 8, 12, 16, 20]
MAXLEN = 16384
N_QUERY = 16
ALPHA, BS = 0.25, 64
K1, K2MID = 128, 768
SINK, SWA = 128, 128
NOPE_D, ROPE_D = 128, 64

DATA = os.environ.get("LB_DATA", os.path.expanduser("~/datasets/LongBench/data"))
OUT = os.environ.get("E141_OUT", os.path.expanduser(
    "~/two-level-attention/exp/trace/results/e141_mla_nope_trace.json"))
MODEL = os.environ.get("MLA_MODEL", os.path.expanduser(
    "~/models/huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat"))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEV,
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    return tok, model


def get_mla_qk(model, input_ids):
    """一次前向拿 hidden_states，对目标层手动复刻 MLA 注意力的 q/k。"""
    H = model.config.num_attention_heads
    with torch.no_grad():
        out = model(input_ids=input_ids.to(DEV), output_hidden_states=True,
                    use_cache=False)
    T = input_ids.shape[1]
    qn, qr, kn, kr, scales = {}, {}, {}, {}, {}
    for L in LAYERS:
        # 【E141 修正 2026-10-15，TL-E141-QK-RECONSTRUCTION 审计缺陷①】
        # hidden_states[L] 是层边界记录 = 该层 input_layernorm 的输入——
        # 须先过该层 RMSNorm 再投影，否则 q/k 输入非 attention 真值。
        hs = model.model.layers[L].input_layernorm(out.hidden_states[L])
        attn = model.model.layers[L].self_attn
        # ---- q：nope 128 | rope 64（q_lora_rank=None → q_proj 直连）----
        q = attn.q_proj(hs).view(1, T, H, NOPE_D + ROPE_D)
        q_nope = q[..., :NOPE_D][0]                     # (T,H,128)
        # ---- kv：latent c_kv 512 + 共享 rope 64；kv_b_proj 上投影 → per-head nope ----
        ckv_kpe = attn.kv_a_proj_with_mqa(hs)
        c_kv, k_pe = ckv_kpe[..., :512], ckv_kpe[..., 512:]  # (1,T,512)/(1,T,64)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(c_kv))
        kv = kv.view(1, T, H, NOPE_D + 128)
        k_nope = kv[..., :NOPE_D][0]                    # (T,H,128)
        # 【E141 修正 2026-10-15，审计缺陷②】yarn RoPE 改走同源
        # apply_rotary_pos_emb（含 view(d//2,2).transpose 通道重排 +
        # position_ids 索引），不再手工 cos/sin + rotate_half——原实现
        # 漏 interleaved 布局前置重排，跨位置 QK 点积错误。
        rope_fn = sys.modules[type(attn).__module__].apply_rotary_pos_emb
        pos_ids = torch.arange(T, device=hs.device).unsqueeze(0)  # (1,T) 同产线口径
        cos, sin = attn.rotary_emb(k_pe, seq_len=T)     # (T,64) bf16
        q4 = q[..., NOPE_D:].transpose(1, 2)            # (1,H,T,64)
        k4 = k_pe.unsqueeze(1)                          # (1,1,T,64) MQA 共享
        q4r, k4r = rope_fn(q4, k4, cos, sin, pos_ids, unsqueeze_dim=1)
        q_pe_r = q4r[0].transpose(0, 1)                 # (T,H,64)
        k_pe_r = k4r[0, 0].unsqueeze(1)                 # (T,1,64) MQA 共享
        qn[L], qr[L] = q_nope.float(), q_pe_r.float()
        kn[L], kr[L] = k_nope.float(), k_pe_r.float()
        scales[L] = float(attn.softmax_scale)           # 192^-0.5 * mscale²
    del out
    torch.cuda.empty_cache()
    return qn, qr, kn, kr, scales


def head_mean_probs(q_row, k_nope_pre, k_pe_pre, scale):
    """因果前缀 → 逐 head softmax(full 192d score) 概率的 16-head 均值 (T',)。
    q_row: nope (H,128) / rope (H,64)；k_nope (T',H,128)；k_pe (T',1,64) MQA 共享。
    逐 head 算分后各自 softmax 再均值（O1 组均值口径的 head 均值版）。"""
    s_h = (torch.einsum("hd,thd->ht", q_row["nope"], k_nope_pre) +
           torch.einsum("hd,td->ht", q_row["rope"], k_pe_pre.squeeze(1))) * scale
    p_h = torch.softmax(s_h, dim=-1)                    # (H,T')
    return p_h.mean(dim=0)                              # (T',)


def arm_score(q_row, k_nope_pre, k_pe_pre, arm, idx=None):
    """选择器臂分数（head 均值原始分，无 softmax——排序用，单调不变量）。"""
    if arm == "nope":
        return torch.einsum("hd,thd->t", q_row["nope"], k_nope_pre)
    if arm == "rope":
        return torch.einsum("hd,td->t", q_row["rope"], k_pe_pre.squeeze(1))
    if arm == "full":                                   # 上限锚：全 192d 打分
        return (torch.einsum("hd,thd->t", q_row["nope"], k_nope_pre) +
                torch.einsum("hd,td->t", q_row["rope"], k_pe_pre.squeeze(1)))
    if arm == "nope_sub":                               # idx: 选定 nope 维坐标
        return torch.einsum("hd,thd->t", q_row["nope"][:, idx],
                            k_nope_pre[:, :, idx])
    raise ValueError(arm)


def two_level_select(s_tok, mid_lo, mid_hi):
    """L1 页级 avg top-K1 → L2 候选页内 token top-768（O3 同规则）。
    返回绝对坐标（sel/cand 都 +mid_lo，与 ref 同坐标系）。"""
    mid_s = s_tok[mid_lo:mid_hi]
    n_mid = mid_s.numel()
    n_pg = n_mid // BS
    pg = mid_s[: n_pg * BS].view(n_pg, BS).mean(dim=1)
    top_pages = pg.topk(min(K1, n_pg)).indices
    cand = torch.cat([torch.arange(int(p) * BS, int(p) * BS + BS, device=DEV)
                      for p in top_pages.tolist()]) + mid_lo
    k2 = min(K2MID, cand.numel())
    sel = cand[mid_s[cand - mid_lo].topk(k2).indices]
    return sel, cand


def probe_cell(q_row, k_nope_pre, k_pe_pre, scale, sub_spaces):
    """单 query 单层：参考概率 + O1 式质量分解 + 各臂两层选择三指标。"""
    T = k_nope_pre.shape[0]
    P = head_mean_probs(q_row, k_nope_pre, k_pe_pre, scale)
    mid_lo, mid_hi = SINK, T - SWA - 1
    n_mid = mid_hi - mid_lo
    # ---- O1 式分解：sink / swa / mid(near+far) 质量占比 ----
    q_pos = T - 1
    mid_len = q_pos - SINK - SWA
    nl = max(BS, (int(ALPHA * mid_len) // BS) * BS) if mid_len > 0 else 0
    mid_p = P[mid_lo:mid_hi]
    o1 = {"mass_sink": float(P[:SINK].sum()),
          "mass_swa": float(P[max(0, q_pos - SWA):].sum()),
          "mass_near": float(mid_p[:nl].sum()) if nl > 0 else 0.0,
          "mass_far": float(mid_p[nl:].sum()) if nl > 0 else float(mid_p.sum())}
    k_sel = min(K2MID, n_mid)
    ref = P[mid_lo:mid_hi].topk(k_sel).indices + mid_lo   # 绝对坐标
    ref_mass = P[ref].sum()
    if ref_mass < 1e-9:
        # 深层注意力超集中：16 head 概率全部精确落在 sink/swa，
        # mid 区 fp32 下溢为 0 —— 无可测 mid 质量，跳过该 query 记账
        return None
    out = {}
    for name, (arm, idx) in sub_spaces.items():
        s_tok = arm_score(q_row, k_nope_pre, k_pe_pre, arm, idx)
        sel, cand = two_level_select(s_tok, mid_lo, mid_hi)
        page_cov = float(torch.isin(ref, cand).float().mean())
        ov = float(torch.isin(sel, ref).float().mean())
        ret = float(P[sel].sum() / ref_mass)
        out[name] = {"page_recall": page_cov, "overlap_recall": ov,
                     "retained_mass": ret}
    out["o1"] = o1
    return out


def main():
    torch.cuda.set_device(DEV)
    print(f"=== E141 MLA nope probe GPU {DEV} model={MODEL} ===", flush=True)
    tok, model = load_model()
    g = torch.Generator().manual_seed(20261002)
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
            qn, qr, kn, kr, scales = get_mla_qk(model, ids)
            qpos = torch.linspace(SINK + SWA + 256, T - 64, N_QUERY).long().tolist()
            for L in LAYERS:
                scale = scales[L]
                # ---- 子空间：随机 {96,64,32} + 方差 top/bottom 64（本层 K 方差）----
                var_dim = kn[L].var(dim=0).mean(dim=0)     # (128,) 跨 head 均方差
                var_order = var_dim.argsort(descending=True)
                sub_spaces = {
                    "full192": ("full", None),
                    "nope128": ("nope", None),
                    "rope64": ("rope", None),
                    "nope96r": ("nope_sub", torch.randperm(128, generator=g)[:96].sort().values.to(DEV)),
                    "nope64r": ("nope_sub", torch.randperm(128, generator=g)[:64].sort().values.to(DEV)),
                    "nope32r": ("nope_sub", torch.randperm(128, generator=g)[:32].sort().values.to(DEV)),
                    "nope64hi": ("nope_sub", var_order[:64].to(DEV)),
                    "nope64lo": ("nope_sub", var_order[-64:].to(DEV)),
                }
                cells = {k_: [] for k_ in sub_spaces}
                o1_cells = []
                n_used = 0
                for qp in qpos:
                    q_row = {"nope": qn[L][qp], "rope": qr[L][qp]}
                    k_np = kn[L][: qp + 1]
                    k_rp = kr[L][: qp + 1]
                    cell = probe_cell(q_row, k_np, k_rp, scale, sub_spaces)
                    if cell is None:
                        continue
                    n_used += 1
                    o1_cells.append(cell.pop("o1"))
                    for k_, v in cell.items():
                        cells[k_].append(v)
                arm_sum = {}
                for name, cs in cells.items():
                    if not cs:
                        arm_sum[name] = None
                        continue
                    arm_sum[name] = {
                        m: {"mean": st.mean(c[m] for c in cs),
                            "worst": min(c[m] for c in cs)}
                        for m in ("page_recall", "overlap_recall", "retained_mass")}
                results.append({"task": task, "sample": si, "layer": L,
                                "T": int(T), "n_query_used": n_used,
                                "o1_mean": {k: st.mean(c[k] for c in o1_cells)
                                           for k in o1_cells[0]} if o1_cells else {},
                                "arms": arm_sum})
                if n_used:
                    print(f"  layer {L} (used {n_used}/{len(qpos)}): " + " ".join(
                        f"{n}={arm_sum[n]['retained_mass']['mean']:.3f}"
                        for n in sub_spaces), flush=True)
                else:
                    print(f"  layer {L}: ALL queries skipped (mid mass~0)",
                          flush=True)
            del qn, qr, kn, kr
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meta = {
        "model": MODEL, "arch": "MLA (DeepSeek-V2-Lite)", "tasks": TASKS,
        "modeling_sha256": hashlib.sha256(
            open(os.path.join(MODEL, "modeling_deepseek.py"), "rb").read()
        ).hexdigest(),
        "qk_reconstruction": "fixed 2026-10-15 (input_layernorm + same-source "
                             "apply_rotary_pos_emb; TL-E141-QK-RECONSTRUCTION)",
        "n_sample": N_SAMPLE, "layers": LAYERS, "maxlen": MAXLEN,
        "dims": {"nope": NOPE_D, "rope": ROPE_D, "kv_lora_rank": 512},
        "alpha": ALPHA, "bs": BS, "K1": K1, "K2_mid": K2MID,
        "sink": SINK, "swa": SWA,
        "caliber": "per-head softmax -> 16-head mean prob (O1 group-mean analog); "
                   "single shared selection on head-mean score (MLA latent-shared "
                   "index story); L1 page-avg top-K1 -> L2 token top-768; "
                   "ref = head-mean prob TopK-768",
        "arms": {"nope128": "nope only (drop rope, 192->128)",
                 "rope64": "rope only (positional control)",
                 "nope96r/64r/32r": "random nope subspace seed20261002",
                 "nope64hi/lo": "variance top/bottom 64 nope dims (per-layer K var)"},
    }
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
