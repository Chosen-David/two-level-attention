# -*- coding: utf-8 -*-
"""E140e O4 选择行为 trace 采集 v3（figure_id=O4，补充 O1 的 oracle 视角）。

v3: valid-partial-pages-v1；有效末页统计与候选裁界，近远配额不 refill。

【v2 修复（TL-E140-VALIDITY 采信清单）】旧版 get_qk 从 hidden_states[L] 手工
q/k_proj+手写 RoPE 重建 Q/K——漏 input_layernorm（Qwen3 v4.51.3 契约违反）。
v2 与 e140_o_trace.py v2 同口径：hook 实际 q_norm/k_norm 输出 + 实际前向
position_embeddings，异常 API fail closed；模型/分词器/数据 revision 声明
门禁 + --execute 显式执行 + 输出 x 模式防覆写（旧 JSON 字节保留，见
../observation/E140_VALIDITY.md）。区域几何（near=SWA 左侧右对齐段、SWA
[q+1-128,q+1) 128 位）与配额公式复刻逻辑自旧版保持不变。

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
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

DEV = os.environ.get("E140_DEVICE", "cuda:0")
TASKS = ["hotpotqa", "musique"]
N_SAMPLE = 2
LAYERS = [int(x) for x in os.environ.get("E140_LAYERS", "8,24").split(",")]
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
    "~/two-level-attention/exp/trace/results/e140_o4_select_trace_v3.json"))
MODEL = os.environ.get("QWEN_MODEL", os.path.expanduser("~/Qwen3-8B"))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=os.environ["E140_TOKENIZER_REVISION"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=os.environ["E140_MODEL_REVISION"], local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("collector requires the verified Qwen3 attention contract")
    return tok, model, float(model.config.rope_theta)


def get_qk(model, theta, input_ids):
    """Capture actual normalized projections and supplied RoPE during forward.

    与 e140_o_trace.py v2 逐位同实现：hook q_norm/k_norm 输出（已含
    input_layernorm 后的 q_proj/k_proj 链）与前向实际 position_embeddings；
    不再从 hidden_states 重建（旧版缺陷①）。异常 API fail closed。
    """
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or input_ids.shape[1] == 0
            or input_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("collector requires one nonempty integer input sequence")
    if model.training:
        raise ValueError("collector requires model.eval()")
    layers = tuple(LAYERS)
    if (not layers or any(type(i) is not int or i < 0 or i >= len(model.model.layers) for i in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("collector requires unique valid layer indices")
    qs, ks, rope, handles = {}, {}, {}, []
    def projection(target, layer):
        def hook(module, args, output):
            if layer in target or not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[:2] != input_ids.shape:
                raise ValueError("unexpected/repeated Qwen3 normalized projection")
            target[layer] = output.detach().clone()
        return hook
    def position_hook(layer):
        def hook(module, args, kwargs):
            value = kwargs.get('position_embeddings')
            if layer in rope or not isinstance(value, (tuple,list)) or len(value)!=2 or not all(isinstance(v,torch.Tensor) for v in value):
                raise ValueError("actual position_embeddings unavailable")
            rope[layer] = tuple(v.detach().clone() for v in value)
        return hook
    try:
        for layer in layers:
            attn = model.model.layers[layer].self_attn
            handles.append(attn.q_norm.register_forward_hook(projection(qs,layer)))
            handles.append(attn.k_norm.register_forward_hook(projection(ks,layer)))
            handles.append(attn.register_forward_pre_hook(position_hook(layer),with_kwargs=True))
        with torch.no_grad():
            model(input_ids=input_ids.to(DEV), use_cache=False)
        for layer in layers:
            if layer not in qs or layer not in ks or layer not in rope:
                raise ValueError("actual Q/K/RoPE hook coverage incomplete")
            q,k=qs[layer],ks[layer];cos,sin=rope[layer]
            if q.shape[-1]!=128 or k.shape[-1]!=128 or cos.shape!=(1,input_ids.shape[1],128) or sin.shape!=cos.shape:
                raise ValueError("unsupported Qwen3 projection/RoPE shape")
            if (not k.shape[2] or not q.shape[2] or q.shape[2] % k.shape[2]
                    or any(not v.is_floating_point() or v.device != q.device or v.dtype != q.dtype
                           or not torch.isfinite(v).all() for v in (q,k,cos,sin))):
                raise ValueError("invalid Qwen3 projection/RoPE values or GQA geometry")
            cos,sin=cos.unsqueeze(2),sin.unsqueeze(2)
            qs[layer]=(q*cos+rotate_half(q)*sin)[0].float()   # (T,32,128)
            ks[layer]=(k*cos+rotate_half(k)*sin)[0].float()   # (T,8,128)
        return qs,ks
    finally:
        for handle in handles: handle.remove()


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
    near_base = max(SINK, S_real - SWA)                       # near 右界 = swa 起点
    near_lo_pg = max(SINK // BS, (near_base - near_len) // BS)
    near_lo = near_lo_pg * BS                      # 页对齐（floor，保守多含）
    return dict(S_real=S_real, mid_lo=SINK, mid_hi=near_base,
                near_lo=near_lo, near_base=near_base, near_len=near_len)


def valid_page_stats(kg):
    """Page reductions over valid tokens only, including a short final page."""
    n = kg.shape[0]
    if n == 0:
        empty = kg.new_empty((0, kg.shape[1]))
        return empty, empty, empty
    pages = (n + BS - 1) // BS
    padded = F.pad(kg, (0, 0, 0, pages * BS - n)).view(pages, BS, -1)
    valid = (torch.arange(pages * BS, device=kg.device) < n).view(pages, BS, 1)
    avg = padded.sum(1) / valid.sum(1)
    maximum = padded.masked_fill(~valid, -torch.inf).amax(1)
    minimum = padded.masked_fill(~valid, torch.inf).amin(1)
    return avg, maximum, minimum


def tli_select(q_row, k_all, geo, K1, idx):
    """生产 mavg 两级选择（tail32 子空间），逐组返回 mid 选中 token 掩码 (8, n_mid)。

    q_row (32,128) float32；k_all (T',8,128)；geo 见 zone_geometry；idx (32,) 子空间坐标。
    """
    T = k_all.shape[0]
    mid_lo, mid_hi = geo["mid_lo"], geo["mid_hi"]
    near_lo = geo["near_lo"]
    n_mid = mid_hi - mid_lo
    if K1 < 0 or K2_MID < 0:
        raise ValueError("budgets must be nonnegative")
    if not (0 <= mid_lo <= near_lo <= mid_hi <= T):
        raise ValueError("invalid selection zone")
    if near_lo not in (mid_lo, mid_hi) and (near_lo - mid_lo) % BS:
        raise ValueError("internal near boundary must be page aligned")
    n_pg = (n_mid + BS - 1) // BS  # include every valid final-page token
    device = q_row.device
    qs = q_row.view(8, 4, 128)[..., idx]              # (8,4,32)
    ks = k_all[:, :, idx]                             # (T',8,32)
    scale = 128 ** -0.5
    tok_score = torch.einsum("ghd,tgd->ght", qs, ks) * scale   # (8,4,T') 细筛分
    k_mid = ks[mid_lo:mid_hi]                         # (n_mid,8,32)

    sel = torch.zeros(8, n_mid, dtype=torch.bool, device=device)
    if n_mid == 0 or K1 == 0 or K2_MID == 0:
        return sel
    nb_near = max(1, round(K1 * BETA))
    nb_far = max(0, K1 - nb_near)
    nt_near = min(round(nb_near * BS * GAMMA), K2_MID)
    far_budget = max(0, K2_MID - nt_near)             # E140 配额公式
    near_pg_lo = (near_lo - mid_lo + BS - 1) // BS             # mid 内页坐标
    far_pg_hi = near_pg_lo
    for g in range(8):
        # ---- L1 页分（生产分型：near=avg / far=minmax 上界）----
        kg = k_mid[:, g, :]                           # (n_mid,32)
        pg_avg, pg_max, pg_min = valid_page_stats(kg)
        qm = qs[g].mean(dim=0)                        # 组均值 q (32,)
        near_sc = pg_avg @ qm                         # avg 页分
        far_sc = (torch.where(qm >= 0, qm * pg_max, qm * pg_min)
                  ).sum(dim=-1)                       # minmax 上界页分
        n_near_pool = n_pg - near_pg_lo
        k_n = min(nb_near, max(n_near_pool, 0))
        i_near = near_sc[near_pg_lo:].topk(k_n).indices + near_pg_lo \
            if k_n > 0 else torch.empty(0, dtype=torch.long, device=device)
        k_f = min(nb_far, max(far_pg_hi, 0))
        i_far = far_sc[:far_pg_hi].topk(k_f).indices \
            if k_f > 0 else torch.empty(0, dtype=torch.long, device=device)
        cand_pg = torch.cat([i_far, i_near]).unique()
        cand_tok = (cand_pg[:, None] * BS +
                    torch.arange(BS, device=device)[None]).reshape(-1)  # mid 内坐标
        cand_tok = cand_tok[cand_tok < n_mid]
        if cand_tok.numel() == 0:
            continue
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


def query_schedule(length):
    if type(length) is not int or length <= SINK + SWA + 320:
        raise ValueError("sample too short for frozen query schedule")
    positions = torch.linspace(SINK + SWA + 256, length - 64, N_QUERY).long().tolist()
    if len(set(positions)) != N_QUERY:
        raise ValueError("query schedule requires distinct positions")
    return positions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='plan-only', gpu_started=False,
                              requires=['explicit --execute', 'model/data identity',
                                        'fresh E140_OUT'],
                              historical_results='not repaired by relabeling')))
        return
    if Path(OUT).exists():
        raise ValueError("output exists; historical observations cannot be overwritten")
    if not all(os.environ.get(k) for k in ('E140_MODEL_REVISION', 'E140_TOKENIZER_REVISION', 'E140_DATASET_REVISION')):
        raise ValueError("explicit model/tokenizer/dataset identity declarations required")
    torch.cuda.set_device(DEV)
    idx = torch.tensor(SUB32, device=DEV)
    print(f"=== E140e O4 select-behavior trace v3 GPU {DEV} ===", flush=True)
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
            qpos = query_schedule(T)
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
                                    "T": int(T),
                                    "input_ids_sha256": hashlib.sha256(
                                        ids.numpy().astype('<i8').tobytes()).hexdigest(),
                                    "o4": rec})
                print(f"  layer {L}: {len(qpos)} queries done", flush=True)
            del qs, ks
            torch.cuda.empty_cache()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    import transformers, inspect
    model_source = Path(inspect.getfile(type(model.model.layers[0].self_attn)))
    meta = {"schema": "e140-o4-select-v3", "page_statistics": "valid-partial-pages-v1",
            "status": "diagnostic-unverified-identity",
            "qk_capture": "actual q_norm/k_norm outputs + actual attention position_embeddings",
            "transformers_version": transformers.__version__,
            "model_source_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
            "declared_revisions": {k: os.environ[k] for k in
                                   ('E140_MODEL_REVISION', 'E140_TOKENIZER_REVISION',
                                    'E140_DATASET_REVISION')},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": MODEL, "tasks": TASKS, "n_sample": N_SAMPLE,
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
    with open(OUT, "x") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
