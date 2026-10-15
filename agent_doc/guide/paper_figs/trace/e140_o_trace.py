# -*- coding: utf-8 -*-
"""E140 observation v2: diagnostic-only collection, explicit --execute required.

Near is the right end of nonprotected mid; SWA contains exactly SWA positions.
O3 uses uniform page average, one unpaired random seed and KV-group metrics;
it is not the production near-average/far-minmax selector. Raw per-query/group
records and optional Q/K snapshots support reanalysis. No scientific acceptance.
"""
import json
import os
import sys
import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DEV = os.environ.get("E140_DEVICE", "cuda:0")
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
    "~/two-level-attention_git/exp/trace/results/e140_o_trace_v2.json"))
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
    theta = float(model.config.rope_theta)
    return tok, model, theta


def get_qk(model, theta, input_ids):
    """Capture actual normalized projections and supplied RoPE during forward.

    Qwen3 v4.51.3 applies input_layernorm before self_attn; hooks avoid reconstructing
    hidden_states and therefore do not omit that normalization. Unexpected APIs
    fail closed. This is not a GPU parity certificate for an untested HF version.
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
            qs[layer]=(q*cos+rotate_half(q)*sin)[0].float()
            ks[layer]=(k*cos+rotate_half(k)*sin)[0].float()
        return qs,ks
    finally:
        for handle in handles: handle.remove()


def group_mean_probs(q_row, k_all, scale):
    """q_row (32,128) k_all (T,8,128) → 组内逐 head softmax 概率均值 (8,T)。
    scale = 1/sqrt(128)（原始全维口径）。GQA：32 Q-head 按 8 组 × 4 映射。"""
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:
        raise ValueError("scale must be finite and positive")
    if q_row.ndim!=2 or k_all.ndim!=3 or not q_row.shape[-1] or q_row.shape[-1]!=k_all.shape[-1] or not k_all.shape[0]:
        raise ValueError("invalid Q/K schema")
    if not q_row.is_floating_point() or not k_all.is_floating_point() or q_row.dtype!=k_all.dtype or q_row.device!=k_all.device:
        raise ValueError("Q/K must have matching floating dtype and device")
    Hkv = k_all.shape[1]
    Hq = q_row.shape[0]
    if Hkv<1 or Hq<1 or Hq%Hkv or not torch.isfinite(q_row).all() or not torch.isfinite(k_all).all():
        raise ValueError("invalid GQA heads or nonfinite Q/K")
    g = Hq // Hkv
    q_g = q_row.reshape(Hkv, g, -1)                     # (8,4,128)
    scores = torch.einsum("ghd,tgd->ght", q_g, k_all) * scale  # (8,4,T)
    probs = torch.softmax(scores, dim=-1)               # 逐 head softmax
    return probs.mean(dim=1)                            # 组内均值 (8,T)


def regions(q_pos, *, sink=SINK, swa=SWA, alpha=ALPHA, bs=BS):
    """Token diagnostic partition; near-left uses production floor alignment.

    Clip to nonprotected mid: unlike production candidate pools, diagnostic mass
    does not double-count sub-block sink/SWA overlap. q_pos is zero-based; S=q+1.
    """
    if any(type(v) is not int or v<0 for v in (q_pos,sink,swa)) or type(bs) is not int or bs<=0:
        raise ValueError("invalid region geometry")
    if type(alpha) not in (int,float) or not 0<alpha<=1:
        raise ValueError("alpha must be in (0,1]")
    length=q_pos+1; sink_hi=min(sink,length); swa_lo=max(0,length-swa)
    mid_lo,mid_hi=sink_hi,max(sink_hi,swa_lo)
    mid_len=mid_hi-mid_lo
    requested=max(bs,int(alpha*mid_len)) if mid_len else 0
    near_lo=max(mid_lo,min(mid_hi,((swa_lo-requested)//bs)*bs))
    return dict(length=length,sink_hi=sink_hi,swa_lo=swa_lo,
                mid_lo=mid_lo,mid_hi=mid_hi,near_lo=near_lo)


def near_len_for(q_pos):
    r=regions(q_pos);return r['mid_hi']-r['near_lo']


def o1_trace(P, q_pos):
    """Full-reference KV-group probabilities; maximum is over groups, not Q heads."""
    if type(q_pos) is not int or q_pos<0 or P.ndim!=2 or not P.shape[0] or P.shape[1]!=q_pos+1:
        raise ValueError("causal probability shape mismatch")
    if not torch.isfinite(P).all() or (P<0).any() or not torch.allclose(P.sum(-1),torch.ones(P.shape[0],device=P.device),atol=1e-5,rtol=1e-5):
        raise ValueError("need normalized finite full-reference probabilities")
    r=regions(q_pos);mass=P.mean(0);lo,hi,nl=r['mid_lo'],r['mid_hi'],r['near_lo']
    sink_m=float(mass[:r['sink_hi']].sum());swa_m=float(mass[max(r['sink_hi'],r['swa_lo']):].sum())
    bins=[]
    for b0 in range(lo,hi,BS):
        b1=min(b0+BS,hi)
        bins.append(dict(d_lo=q_pos-b1+1,d_hi=q_pos-b0,mass=float(mass[b0:b1].sum()),n_tok=b1-b0))
    far=P[:,lo:nl].sum(-1)
    return dict(schema='e140-o1-v2',mass_sink=sink_m,mass_swa_excluding_sink=swa_m,
                mass_protected_union=sink_m+swa_m,mass_near=float(mass[nl:hi].sum()),
                mass_far=float(mass[lo:nl].sum()),near_len=hi-nl,q_pos=q_pos,
                far_max_kv_group_mass=float(far.max()),region_bounds=r,bins=bins)


def o3_trace(q_row, k_all, scale, sub_idx):
    """Diagnostic uniform-page-average selector; all partial pages participate.

    Metrics name their denominator explicitly. Observations are query × KV group,
    never independent documents. This does not reproduce production L1/L2.
    """
    P = group_mean_probs(q_row, k_all, scale)
    if q_row.shape[-1] != 128 or scale != 128 ** -0.5:
        raise ValueError("O3 requires frozen 128-dimensional reference scale")
    T = k_all.shape[0]
    g = torch.Generator().manual_seed(20261002)
    if sub_idx is None:
        sub_idx = torch.randperm(128, generator=g)[:32].sort().values.tolist()
    if (not isinstance(sub_idx,(list,tuple)) or not sub_idx
            or any(type(i) is not int or not 0<=i<128 for i in sub_idx)
            or len(set(sub_idx))!=len(sub_idx)):
        raise ValueError("subspace requires unique integer coordinates in [0,128)")
    idx = torch.tensor(sub_idx, device=q_row.device)
    r=regions(T-1)
    mid_lo,mid_hi=r["mid_lo"],r["mid_hi"]
    n_mid = mid_hi - mid_lo
    metrics = []
    hkv=k_all.shape[1];group_size=q_row.shape[0]//hkv
    if n_mid<=0: raise ValueError("O3 requires nonempty nonprotected mid")
    for grp in range(hkv):
        mid_p = P[grp, mid_lo:mid_hi]                # (n_mid,)
        k_sel = min(K2MID, n_mid)
        ref = mid_p.topk(k_sel).indices              # mid 坐标
        ref_mass = mid_p[ref].sum()
        # ---- selector 臂（32 维，组内 4 Q-head 均值分）----
        q_s = q_row[grp * group_size: (grp + 1) * group_size][:, idx]         # (4,32)
        k_s = k_all[:, grp, :][:, idx]                        # (T',32)
        s_tok = torch.einsum("hd,td->t", q_s, k_s) * (128 ** -0.5)
        mid_s = s_tok[mid_lo:mid_hi]
        # L1：64-token 页 avg 分 → top-K1 页（非保护池）
        starts=list(range(0,n_mid,BS))
        pg=torch.stack([mid_s[start:min(start+BS,n_mid)].mean() for start in starts])
        top_pages=pg.topk(min(K1,len(starts))).indices
        cand=torch.cat([torch.arange(starts[i],min(starts[i]+BS,n_mid),device=q_row.device)
                        for i in top_pages.tolist()])
        page_cov = float(torch.isin(ref, cand).float().mean())
        # L2：候选 token 32 维分 top-768
        k2 = min(K2MID, cand.numel())
        sel = cand[mid_s[cand].topk(k2).indices]
        ov = float(torch.isin(ref, sel).float().mean())       # overlap recall
        if not torch.isfinite(ref_mass) or ref_mass<=0: raise ValueError("zero/nonfinite reference mass")
        metrics.append(dict(kv_group=grp,reference_token_page_coverage=page_cov,
            reference_token_overlap_recall=ov,selected_mass_over_reference_topk_mass=float(mid_p[sel].sum()/ref_mass),
            selected_full_attention_mass=float(mid_p[sel].sum()),reference_topk_full_attention_mass=float(ref_mass),
            reference_count=k_sel,selected_count=int(sel.numel()),mid_count=n_mid,
            selected_ids=(sel+mid_lo).tolist(),reference_ids=(ref+mid_lo).tolist()))
    return metrics


def query_schedule(length):
    if type(length) is not int or length<=SINK+SWA+320:
        raise ValueError("sample too short for frozen query schedule")
    positions=torch.linspace(SINK+SWA+256,length-64,N_QUERY).long().tolist()
    if len(set(positions))!=N_QUERY: raise ValueError("query schedule requires distinct positions")
    return positions


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--save-qk',action='store_true')
    args=ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='plan-only',gpu_started=False,requires=['explicit --execute','model/data identity','fresh E140_OUT'],historical_results='not repaired by relabeling')))
        return
    if Path(OUT).exists(): raise ValueError("output exists; historical observations cannot be overwritten")
    if not all(os.environ.get(k) for k in ('E140_MODEL_REVISION','E140_TOKENIZER_REVISION','E140_DATASET_REVISION')):
        raise ValueError("explicit model/tokenizer/dataset identity declarations required")
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
            qpos=query_schedule(T)
            qs, ks = get_qk(model, theta, ids)
            for L in LAYERS:
                qL, kL = qs[L], ks[L]
                scale = 128 ** -0.5
                o1_cells, o3_cells = [], {k_: [] for k_ in SUBSPACES}
                for qp in qpos:
                    q_row = qL[qp]                     # (32,128)
                    k_all = kL[: qp + 1]               # 因果前缀 (T',8,128)
                    P = group_mean_probs(q_row, k_all, scale)
                    o1_cells.append(o1_trace(P, qp))
                    # O3 shares corrected nonprotected mid [SINK, qp+1-SWA).
                    k_pre = k_all
                    for name, sub in SUBSPACES.items():
                        o3_cells[name].append(o3_trace(q_row, k_pre, scale, sub))
                # Keep raw observations; aggregates cannot recover different partitions.
                if args.save_qk:
                    import numpy as np
                    raw_path=Path(OUT).with_name(Path(OUT).stem+f'.{task}.{si}.L{L}.npz')
                    raw_path.parent.mkdir(parents=True,exist_ok=True)
                    with raw_path.open('xb') as raw:
                        np.savez_compressed(raw,q=qL[qpos].cpu().numpy(),k=kL.cpu().numpy(),query_positions=qpos)
                    qk_receipt=dict(path=raw_path.name,sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
                else: qk_receipt=None
                # Across query × KV-group descriptive summaries, not independent n.
                o3_sum = {}
                for name, cells in o3_cells.items():
                    import statistics as st
                    flat = [m for cell in cells for m in cell]
                    o3_sum[name] = {
                        k2: {"mean": st.mean(m[k2] for m in flat),
                             "worst": min(m[k2] for m in flat)}
                        for k2 in ("reference_token_page_coverage", "reference_token_overlap_recall", "selected_mass_over_reference_topk_mass", "selected_full_attention_mass")}
                results.append({
                    "task": task, "sample": si, "layer": L, "T": int(T),
                    "input_ids_sha256": hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest(),
                    "q_positions":qpos,"qk_receipt":qk_receipt,
                    "o1": o1_cells, "o3": o3_sum,"o3_raw":o3_cells})
                print(f"  layer {L}: O1 {len(o1_cells)} queries, O3 3 arms done",
                      flush=True)
            del qs, ks
            torch.cuda.empty_cache()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    import transformers,inspect
    model_source=Path(inspect.getfile(type(model.model.layers[0].self_attn)))
    meta = {"schema":"e140-observation-v2","status":"diagnostic-unverified-identity",
            "independent_unit":"source sequence/document; queries and heads are repeated observations",
            "n_source_sequences":len({(c['task'],c['sample']) for c in results}),
            "qk_capture":"actual q_norm/k_norm outputs + actual attention position_embeddings",
            "transformers_version":transformers.__version__,"model_source_sha256":hashlib.sha256(model_source.read_bytes()).hexdigest(),
            "declared_revisions":{k:os.environ[k] for k in ('E140_MODEL_REVISION','E140_TOKENIZER_REVISION','E140_DATASET_REVISION')},
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tasks": TASKS, "n_sample": N_SAMPLE,
            "layers": LAYERS, "maxlen": MAXLEN, "alpha": ALPHA, "bs": BS,
            "K1": K1, "K2_mid": K2MID, "sink": SINK, "swa": SWA,
            "subspaces": {k: (v if v else "random-unpaired seed20261002")
                          for k, v in SUBSPACES.items()},
            "o3_rule": "L1 page-avg top-K1 -> L2 token top-768, per-group, "
                       "same rule across arms (production far-minmax/near-avg "
                       "typing NOT replicated; diagnostic frequency contrast)",
            "o1_rule": "GQA per-head softmax -> group mean prob; sink/swa "
                       "separate; 64-token distance bins on mid"}
    with open(OUT, "x") as f:
        json.dump({"meta": meta, "cells": results}, f, ensure_ascii=False, indent=1)
    print(f"=== saved {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
