# -*- coding: utf-8 -*-
"""E141 v2 collector correctness repair; diagnostic only, explicit --execute.

Expanded per-head NoPE coordinates are not latent coordinates. full192 is a
logit proxy, not the head-mean probability mass oracle. Variance arms use future
keys offline. No algorithm, speedup, layer-skip or scientific acceptance claim.
Historical outputs must never be overwritten.
"""
import json
import os
import statistics as st
import argparse
import hashlib
import inspect
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("e141_e140_accounting", Path(__file__).with_name("e140_o_trace.py"))
e140 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e140)
UPSTREAM_SHA256 = "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"
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
    "~/two-level-attention/exp/trace/results/e141_mla_nope_trace_v2.json"))
MODEL = os.environ.get("MLA_MODEL", os.path.expanduser(
    "~/models/huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat"))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True,
                                        revision=os.environ["E141_TOKENIZER_REVISION"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEV,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
        revision=os.environ["E141_MODEL_REVISION"])
    model.eval()
    source = Path(inspect.getfile(type(model.model.layers[0].self_attn)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("loaded attention source differs from pinned verified implementation")
    return tok, model


@torch.no_grad()
def get_mla_qk(model, input_ids):
    """Unpadded B=1, no-cache, zero-origin reconstruction of pinned eager MLA.

    Apply decoder input RMSNorm and the official even/odd rotary rearrangement.
    FP32 analysis of native projections is not native bf16 probability parity.
    """
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]
            or input_ids.dtype not in (torch.int32, torch.int64) or getattr(model, 'training', False)):
        raise ValueError("need eval model and one nonempty integer input sequence")
    if (not LAYERS or any(type(i) is not int or i<0 or i>=len(model.model.layers) for i in LAYERS)
            or len(set(LAYERS)) != len(LAYERS)):
        raise ValueError("need unique valid layers")
    for layer in LAYERS:
        attn = model.model.layers[layer].self_attn
        if (getattr(attn, 'q_lora_rank', None) is not None
                or getattr(attn, 'qk_nope_head_dim', 128) != 128
                or getattr(attn, 'qk_rope_head_dim', 64) != 64
                or getattr(attn, 'kv_lora_rank', 512) != 512
                or getattr(attn, 'v_head_dim', 128) != 128):
            raise ValueError("unsupported MLA geometry")
    H = model.config.num_attention_heads
    with torch.no_grad():
        out = model(input_ids=input_ids.to(DEV), output_hidden_states=True,
                    use_cache=False, position_ids=torch.arange(input_ids.shape[1], device=DEV).unsqueeze(0))
    T = input_ids.shape[1]
    qn, qr, kn, kr, scales = {}, {}, {}, {}, {}
    for L in LAYERS:
        hs = model.model.layers[L].input_layernorm(out.hidden_states[L])
        attn = model.model.layers[L].self_attn
        # ---- q：nope 128 | rope 64（q_lora_rank=None → q_proj 直连）----
        q = attn.q_proj(hs).view(1, T, H, NOPE_D + ROPE_D)
        q_nope = q[..., :NOPE_D][0]                     # (T,H,128)
        q_pe = q[..., NOPE_D:][0]                       # (T,H,64)
        # ---- kv：latent c_kv 512 + 共享 rope 64；kv_b_proj 上投影 → per-head nope ----
        ckv_kpe = attn.kv_a_proj_with_mqa(hs)
        c_kv, k_pe = ckv_kpe[..., :512], ckv_kpe[..., 512:]  # (1,T,512)/(1,T,64)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(c_kv))
        kv = kv.view(1, T, H, NOPE_D + 128)
        k_nope = kv[..., :NOPE_D][0]                    # (T,H,128)
        # ---- yarn RoPE（走模型自带模块，cos/sin 与生产逐位一致）----
        cos, sin = attn.rotary_emb(k_pe, seq_len=T)     # (T,64) bf16
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)   # (T,1,64) 广播 head
        k_pe = k_pe[0].unsqueeze(1)                     # (T,1,64) MQA 共享
        # Official layout: interleaved even/odd coordinates -> split halves.
        q_pe = q_pe.reshape(T, H, ROPE_D // 2, 2).transpose(-1, -2).reshape(T, H, ROPE_D)
        k_pe = k_pe.reshape(T, 1, ROPE_D // 2, 2).transpose(-1, -2).reshape(T, 1, ROPE_D)
        q_pe_r = q_pe * cos + rotate_half(q_pe) * sin
        k_pe_r = k_pe * cos + rotate_half(k_pe) * sin
        qn[L], qr[L] = q_nope.float(), q_pe_r.float()
        kn[L], kr[L] = k_nope.float(), k_pe_r.float()
        if any(not torch.isfinite(v).all() for v in (qn[L], qr[L], kn[L], kr[L])):
            raise ValueError("nonfinite reconstructed Q/K")
        scales[L] = float(attn.softmax_scale)           # 192^-0.5 * mscale²
    del out
    return qn, qr, kn, kr, scales


def head_mean_probs(q_row, k_nope_pre, k_pe_pre, scale):
    """因果前缀 → 逐 head softmax(full 192d score) 概率的 16-head 均值 (T',)。
    q_row: nope (H,128) / rope (H,64)；k_nope (T',H,128)；k_pe (T',1,64) MQA 共享。
    逐 head 算分后各自 softmax 再均值（O1 组均值口径的 head 均值版）。"""
    qn, qr = q_row["nope"], q_row["rope"]
    if (qn.ndim!=2 or not qn.shape[0] or qn.shape[1]!=128 or qr.shape!=(qn.shape[0],64)
            or k_nope_pre.ndim!=3 or k_nope_pre.shape[1:]!=qn.shape
            or not k_nope_pre.shape[0] or k_pe_pre.shape!=(k_nope_pre.shape[0],1,64)):
        raise ValueError("invalid MLA Q/K schema")
    if type(scale) not in (int,float) or not 0<scale<float('inf'):
        raise ValueError("invalid attention scale")
    if any(not v.is_floating_point() or v.dtype!=qn.dtype or v.device!=qn.device
           or not torch.isfinite(v).all() for v in (qn,qr,k_nope_pre,k_pe_pre)):
        raise ValueError("invalid Q/K dtype/device/values")
    s_h = (torch.einsum("hd,thd->ht", q_row["nope"], k_nope_pre) +
           torch.einsum("hd,td->ht", q_row["rope"], k_pe_pre.squeeze(1))) * scale
    if not torch.isfinite(s_h).all(): raise ValueError("nonfinite logits")
    p_h = torch.softmax(s_h, dim=-1)                    # (H,T')
    return p_h.mean(dim=0)                              # (T',)


def arm_score(q_row, k_nope_pre, k_pe_pre, arm, idx=None):
    """选择器臂分数（head 均值原始分，无 softmax——排序用，单调不变量）。"""
    if arm == "nope":
        return torch.einsum("hd,thd->t", q_row["nope"], k_nope_pre)
    if arm == "rope":
        return torch.einsum("hd,td->t", q_row["rope"], k_pe_pre.squeeze(1))
    if arm == "full":                                   # Full-dimensional logit proxy, not mass oracle
        return (torch.einsum("hd,thd->t", q_row["nope"], k_nope_pre) +
                torch.einsum("hd,td->t", q_row["rope"], k_pe_pre.squeeze(1)))
    if arm == "nope_sub":                               # idx: 选定 nope 维坐标
        if (not isinstance(idx,torch.Tensor) or idx.ndim!=1 or not idx.numel()
                or idx.dtype!=torch.long or idx.device!=k_nope_pre.device
                or (idx<0).any() or (idx>=NOPE_D).any() or idx.unique().numel()!=idx.numel()):
            raise ValueError("invalid NoPE subspace coordinates")
        return torch.einsum("hd,thd->t", q_row["nope"][:, idx],
                            k_nope_pre[:, :, idx])
    raise ValueError(arm)


def two_level_select(s_tok, mid_lo, mid_hi):
    """L1 页级 avg top-K1 → L2 候选页内 token top-768（O3 同规则）。
    返回绝对坐标（sel/cand 都 +mid_lo，与 ref 同坐标系）。"""
    if (s_tok.ndim!=1 or not torch.isfinite(s_tok).all() or type(mid_lo) is not int
            or type(mid_hi) is not int or not 0<=mid_lo<=mid_hi<=s_tok.numel()
            or any(type(v) is not int or v<=0 for v in (BS,K1,K2MID))):
        raise ValueError("invalid selection scores, bounds or budgets")
    mid_s = s_tok[mid_lo:mid_hi]
    starts = list(range(0, mid_s.numel(), BS))
    if not starts:
        empty = torch.empty(0, dtype=torch.long, device=s_tok.device)
        return empty, empty
    pg = torch.stack([mid_s[i:i+BS].mean() for i in starts])
    top_pages = pg.topk(min(K1,len(starts))).indices
    cand = torch.cat([torch.arange(starts[i],min(starts[i]+BS,mid_s.numel()),device=s_tok.device)
                      for i in top_pages.tolist()]) + mid_lo
    sel = cand[s_tok[cand].topk(min(K2MID,cand.numel())).indices]
    return sel, cand


def probe_cell(q_row, k_nope_pre, k_pe_pre, scale, sub_spaces):
    """Preserve every query, including below-threshold and saturated support."""
    T = k_nope_pre.shape[0]
    P = head_mean_probs(q_row, k_nope_pre, k_pe_pre, scale)
    # Reuse E140's disjoint token accounting and right-near partition.
    o1 = e140.o1_trace(P.unsqueeze(0), T-1)
    r = o1['region_bounds']
    mid_lo, mid_hi = r['mid_lo'], r['mid_hi']
    n_mid = mid_hi-mid_lo
    ref = P[mid_lo:mid_hi].topk(min(K2MID,n_mid)).indices + mid_lo
    ref_mass = float(P[ref].sum())
    reason = 'empty_mid' if not n_mid else ('reference_mass_below_threshold' if ref_mass<1e-9 else None)
    record = dict(query_position=T-1, status='skipped' if reason else 'used', skip_reason=reason,
                  reference_topk_mass=ref_mass, skip_threshold=1e-9,
                  mid_mass=float(P[mid_lo:mid_hi].sum()), probabilities_finite=True,
                  reference_count=len(ref), reference_ids=ref.tolist(), mid_count=n_mid,
                  budget_saturated=n_mid<=K2MID, o1=o1, arms={})
    if reason: return record
    for name, (arm, idx) in sub_spaces.items():
        scores = arm_score(q_row,k_nope_pre,k_pe_pre,arm,idx)
        sel,cand = two_level_select(scores,mid_lo,mid_hi)
        record['arms'][name] = dict(
            reference_token_page_coverage=float(torch.isin(ref,cand).float().mean()),
            reference_token_overlap_recall=float(torch.isin(ref,sel).float().mean()),
            selected_mass_over_reference_topk_mass=float(P[sel].sum())/ref_mass,
            selected_full_attention_mass=float(P[sel].sum()),
            selected_count=len(sel), candidate_count=len(cand),
            selected_ids=sel.tolist(), candidate_ids=cand.tolist())
    return record


def summarize_cells(cells, names):
    used=[c for c in cells if c['status']=='used']
    metrics=('reference_token_page_coverage','reference_token_overlap_recall',
             'selected_mass_over_reference_topk_mass','selected_full_attention_mass')
    def summary(rows):
        return {name:{m:dict(mean=st.mean(c['arms'][name][m] for c in rows),
                            worst=min(c['arms'][name][m] for c in rows)) for m in metrics}
                if rows else None for name in names}
    strata={label:[c for c in used if c['budget_saturated']==flag]
            for label,flag in [('saturated',True),('budget_constrained',False)]}
    return dict(n_query_planned=len(cells), n_query_used=len(used),
                n_query_skipped=len(cells)-len(used),
                n_budget_saturated=sum(c['budget_saturated'] for c in cells),
                summary_condition='used queries only; not independent documents',
                arms=summary(used),
                strata={label:dict(n_query_used=len(rows),arms=summary(rows))
                        for label,rows in strata.items()}, raw_queries=cells)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='plan-only',gpu_started=False)))
        return
    if Path(OUT).exists(): raise ValueError("output exists; historical results cannot be overwritten")
    required=('E141_MODEL_REVISION','E141_TOKENIZER_REVISION','E141_DATASET_REVISION')
    if not all(os.environ.get(k) for k in required): raise ValueError("explicit identity declarations required")
    local_source=Path(MODEL)/'modeling_deepseek.py'
    if not local_source.is_file() or hashlib.sha256(local_source.read_bytes()).hexdigest()!=UPSTREAM_SHA256:
        raise ValueError("local pinned attention source required before model execution")
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
            qpos = e140.query_schedule(T)
            qn, qr, kn, kr, scales = get_mla_qk(model, ids)
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
                cells=[]
                for qp in qpos:
                    q_row={"nope":qn[L][qp],"rope":qr[L][qp]}
                    cells.append(probe_cell(q_row,kn[L][:qp+1],kr[L][:qp+1],scale,sub_spaces))
                summary=summarize_cells(cells,sub_spaces)
                results.append(dict(task=task,sample=si,layer=L,T=T,
                    input_ids_sha256=hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest(),
                    subspaces={name:dict(arm=arm,indices=idx.tolist() if idx is not None else None)
                               for name,(arm,idx) in sub_spaces.items()}, **summary))
            del qn, qr, kn, kr
            torch.cuda.empty_cache()
    Path(OUT).parent.mkdir(parents=True,exist_ok=True)
    meta = {
        "schema":"e141-observation-v2", "status":"diagnostic-unverified-weight-data-identity",
        "declared_revisions":{k:os.environ[k] for k in required},
        "model_source_sha256":UPSTREAM_SHA256,
        "producer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "accounting_source_sha256":hashlib.sha256(Path(e140.__file__).read_bytes()).hexdigest(),
        "variance_coordinates":"offline full-sequence keys, including future positions",
        "independent_unit":"source sequence/document; queries and layers are repeated measurements",
        "full192":"logit proxy, not mass oracle; no output quality or speed claim",
        "model": MODEL, "arch": "MLA (DeepSeek-V2-Lite)", "tasks": TASKS,
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
    with open(OUT, "x") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
