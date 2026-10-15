# -*- coding: utf-8 -*-
"""E140g L1 稀释机制判决探针（mass 级，配对 E140a k512 e2e 崩塌 -14）。

假说：K1 上探超池 → L1 被旁路 → 4bit 量化 L2 全池 top-K2 被量化噪声稀释。
判决签名：K1↑ 只在「L2 量化」臂伤 recall/mass，「L2 精确」臂单调不降
（超集单调性）——交互项即机制证据，排除「候选集本身变差」解释。

口径：克隆 e140_o_trace（Qwen3-8B 真实样本、q_norm/k_norm + RoPE、GQA 组
均值概率 full-ref、tail32 子空间、L1 页 avg 分）。L2 量化 = 逐 token
min-max 4bit（16 级均匀，对齐生产 cmp4 量级；非逐位复刻，机制探针口径）。

K1 扫 {16, 64, 128, 224, 448}（448=k512 档 nb_far 等效全池；224=k256 档）。
输出 exp/trace/results/e140_l1_dilution_probe.json。
"""
import json
import os

import torch

DEV = "cuda:0"
TASKS = ["hotpotqa", "musique"]
N_SAMPLE = 2
LAYERS = [8, 24]
MAXLEN = 16384
N_QUERY = 8
BS = 64
K2MID = 768
SINK, SWA = 128, 128
K1_SWEEP = [16, 64, 128, 224, 448]
L2_MODES = ["exact", "q4"]

DATA = os.path.expanduser("~/datasets/LongBench/data")
MODEL = os.path.expanduser("~/Qwen3-8B")
OUT = os.path.expanduser(
    "~/two-level-attention/exp/trace/results/e140_l1_dilution_probe.json")

SUB_IDX = list(range(48, 64)) + list(range(112, 128))   # tail32 生产同口径


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True)
    model.eval()
    return tok, model


def get_qk(model, cos, sin, input_ids):
    """一次前向拿 hidden_states，逐层手动 q/k_proj + RoPE（克隆 e140_o_trace
    口径；q_norm/k_norm 必须施加——E140d 根因坑。注意新版 transformers 下
    forward hook 位置参数为空，须走 output_hidden_states 路线）。"""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True,
                    use_cache=False)
    outs = {}
    for li in LAYERS:
        hs = out.hidden_states[li]               # (1,T,hidden) bf16
        attn = model.model.layers[li].self_attn
        q = attn.q_proj(hs).view(1, -1, 32, 128)
        k = attn.k_proj(hs).view(1, -1, 8, 128)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        # 克隆 e140_o_trace 口径：bf16 cos/sin + bf16 RoPE 后升 float
        cs = cos[: hs.shape[1]].view(1, -1, 1, 128)
        sn = sin[: hs.shape[1]].view(1, -1, 1, 128)
        q = (q * cs + rotate_half(q) * sn).float()
        k = (k * cs + rotate_half(k) * sn).float()
        outs[li] = (q[0], k[0])
    return outs


def group_mean_probs(q_row, k_all, scale):
    Hkv = k_all.shape[1]
    Hq = q_row.shape[0]
    g = Hq // Hkv
    q_g = q_row.view(Hkv, g, -1)
    scores = torch.einsum("ghd,tgd->ght", q_g, k_all) * scale
    probs = torch.softmax(scores, dim=-1)
    return probs.mean(dim=1)


def quant4_rows(k_s):
    """逐 token min-max 4bit（16 级均匀量化，对齐 cmp4 量级）。"""
    kmin = k_s.min(dim=-1, keepdim=True).values
    kmax = k_s.max(dim=-1, keepdim=True).values
    rng = (kmax - kmin).clamp_min(1e-12)
    q = torch.round((k_s - kmin) / rng * 15.0) / 15.0 * rng + kmin
    return q.to(k_s.dtype)


def probe_group(P, q_row, k_all, scale, k1, l2_mode):
    """单 KV-group 两级选择指标。返回 (overlap, retained_mass, page_recall)。"""
    T = k_all.shape[0]
    mid_lo, mid_hi = SINK, T - SWA - 1
    n_mid = mid_hi - mid_lo
    mid_p = P[mid_lo:mid_hi]
    k_sel = min(K2MID, n_mid)
    ref = mid_p.topk(k_sel).indices
    ref_mass = mid_p[ref].sum()

    idx = torch.tensor(SUB_IDX, device=DEV)
    # q_row 形状 (4,128)：组内 4 Q-head；k_all (T,32) 已切子空间
    q_s = q_row[:, idx]                                  # (4,32)
    s_tok = torch.einsum("hd,td->t", q_s, k_all) * scale
    mid_s = s_tok[mid_lo:mid_hi]

    n_pg = n_mid // BS
    pg = mid_s[: n_pg * BS].view(n_pg, BS).mean(dim=1)
    top_pages = pg.topk(min(k1, n_pg)).indices
    page_cov = float(torch.isin(ref // BS, top_pages).float().mean())
    cand = torch.cat([torch.arange(int(p) * BS, int(p) * BS + BS, device=DEV)
                      for p in top_pages.tolist()])
    if l2_mode == "q4":
        kq = quant4_rows(k_all[mid_lo:mid_hi])           # 逐 token 4bit
        # s_q 本就是 mid 段（长度 n_mid），候选/参考均为 mid 相对坐标
        mid_sel_s = torch.einsum("hd,td->t", q_s, kq) * scale
    else:
        mid_sel_s = mid_s
    k2 = min(K2MID, cand.numel())
    sel = cand[mid_sel_s[cand].topk(k2).indices]
    ov = float(torch.isin(sel, ref).float().mean())
    ret_mass = float(mid_p[sel].sum() / ref_mass)
    return ov, ret_mass, page_cov


def main():
    torch.cuda.set_device(DEV)
    print(f"=== E140g L1 dilution probe {DEV} ===", flush=True)
    tok, model = load_model()
    assert getattr(model.config, "rope_scaling", None) is None, \
        "rope_scaling 存在——手写 RoPE 需适配，fail loudly"
    theta_base = float(model.config.rope_theta)
    d = 128
    inv = 1.0 / (theta_base ** (torch.arange(0, d // 2, device=DEV).float()
                                / (d // 2)))
    pos = torch.arange(MAXLEN, device=DEV).float()
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    results = {m: {str(k): {"hotpotqa": [], "musique": []} for k in K1_SWEEP}
               for m in L2_MODES}
    for task in TASKS:
        path = os.path.join(DATA, f"{task}.jsonl")
        samples = [json.loads(l) for l in open(path)][:N_SAMPLE]
        for si, s in enumerate(samples):
            prompt = s["context"] if "context" in s else s.get("input", "")
            ids = tok(prompt, truncation=True,
                      max_length=MAXLEN - 64, return_tensors="pt").input_ids
            ids = ids.to(DEV)
            outs = get_qk(model, cos, sin, ids)
            Tq = ids.shape[1]
            g = torch.Generator().manual_seed(20261002 + si)
            qpos = torch.randperm(Tq - SINK - SWA - 2, generator=g)[:N_QUERY] \
                + SINK + 1
            for li in LAYERS:
                q_l, k_l = outs[li]
                for qp in qpos.tolist():
                    k_all = k_l[: qp + 1]                    # (T',8,128)
                    scale = d ** -0.5
                    P_all = group_mean_probs(
                        q_l[qp], k_all, scale)               # (8,T')
                    for grp in range(8):
                        P = P_all[grp]                       # (T',)
                        k_sub = k_all[:, grp][:, SUB_IDX]    # (T',32)
                        q_grp = q_l[qp][grp * 4: (grp + 1) * 4]  # (4,128)
                        for k1 in K1_SWEEP:
                            for m in L2_MODES:
                                ov, rm, pc = probe_group(
                                    P, q_grp, k_sub, scale, k1, m)
                                results[m][str(k1)][task].append(
                                    {"ov": ov, "rm": rm, "pc": pc})
            print(f"  {task} sample {si} layer done", flush=True)

    summary = {}
    for m in L2_MODES:
        summary[m] = {}
        for k1 in K1_SWEEP:
            row = {}
            for task in TASKS:
                cells = results[m][str(k1)][task]
                row[task] = {
                    "overlap": round(sum(c["ov"] for c in cells) / len(cells), 4),
                    "retained_mass": round(
                        sum(c["rm"] for c in cells) / len(cells), 4),
                    "page_recall": round(
                        sum(c["pc"] for c in cells) / len(cells), 4),
                    "n": len(cells),
                }
            summary[m][str(k1)] = row
    out = {"experiment": "E140g L1 dilution probe",
           "k1_sweep": K1_SWEEP, "l2_modes": L2_MODES,
           "k2mid": K2MID, "subspace": "tail32",
           "quant_note": "L2 q4 = per-token min-max 16 级均匀（cmp4 量级仿真，非逐位复刻）",
           "summary": summary, "raw": results}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
