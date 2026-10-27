# -*- coding: utf-8 -*-
"""E178: LoDA-SA K1=128 full production vs FlashAttention2 — kernel-decode
speedup microbench. bs∈{1,4,6} × ctx∈{32K,64K,128K,256K}.

Protocol:
  - Deterministic synthetic needle prompt (E152 build_prompt_tokens, fixed seed)
  - Qwen3-8B bf16, H20, CUDA_VISIBLE_DEVICES-isolated
  - Three arms:
      fa2:    dense SDPA (H20 flash path, PyTorch 2.8 cu128)
      tli:    K1=128 + tail32 + sink4 + swa128 + cmp_ratio=4, layer_skip OFF
      tli+fs: same + layer_skip ON (far skip)
  - Per-step wall+device sync; warmup then reps; median TPOT
  - Peak GPU memory per arm

Dispatch:
  CUDA_VISIBLE_DEVICES=0 python -u exp/bench/e178_fa2_speedup.py \\
    --method tli+fs --ctx 32768 --bs 1 --repeats 5 --warmup 2 \\
    --model-path ~/Qwen3-8B --out exp/trace/results/e178/tli_fs_b1_L32k.json
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from exp.bench.e152_e2e_latency import build_prompt_tokens
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch as tli_patch
from sparse_attn.info import resolve_treatment_snapshot


def build_tli_args(ns):
    """Build a complete sparse_attn Args namespace from parsed CLI args.
    Strategy: use add_sparse_attn_args to populate defaults, then override."""
    import sys as _sys
    from io import StringIO as _StringIO
    # Build a throwaway Namespace using add_sparse_attn_args
    _cp = _sys.argv[:]
    _sys.argv = [_cp[0]]
    _p = argparse.ArgumentParser()
    add_sparse_attn_args(_p)
    # parse_known_args 避免参数不识别报错
    a, _ = _p.parse_known_args([])
    _sys.argv = _cp
    # Override with K1=128 生产口径
    a.method = "tli"
    a.tli_subspace = "tail"
    a.tia_level2_cmp_ratio = 4
    a.tli_far_method = "minmax"
    a.tli_near_method = "avg"
    a.tia_level1_topk = 128      # K1=128
    a.tia_level2_topk = 1024
    a.tli_near_pages = 32
    a.tli_near_tokens = "off"
    a.tli_near_len = 4096
    a.tli_sink_tokens = 4
    a.tli_swa_size = 128
    a.tli_enable_layer_skip = (ns.method == "tli+fs")
    a.tli_far_skip_backend = "compact"
    a.tli_dyn_dgate = False
    a.tli_sparse_prefill = False
    a.tli_enable_kmeans = False
    a.tli_layer_skip_path = None
    a.tli_alpha, a.tli_beta, a.tli_gamma = None, None, None
    return a


def load_model_fa2(model_path):
    """Dense baseline via SDPA (PyTorch 2.8 cu128, H20 = flash/cudnn path)."""
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa")


def load_model_tli(model_path, method):
    args_ns = build_tli_args(
        argparse.Namespace(method=method))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa")
    # Inject sparse patch (with treatment snapshot, production path)
    snapshot = resolve_treatment_snapshot(args_ns)
    tli_patch(model, args_ns, snapshot=snapshot)
    return model


def bench_step(model, input_ids, past_key_values):
    """One decode step; returns (output, elapsed_ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past_key_values,
                    use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def bench_decode(model, tokenizer, method, a):
    token_ids_batch = [build_prompt_tokens(tokenizer, a.ctx)
                       for _ in range(a.bs)]
    # 直接转 tensor（encode-only 路线避免二次 tokenize）
    max_len = max(len(t) for t in token_ids_batch)
    input_ids = torch.zeros(a.bs, max_len, dtype=torch.long)
    attn_mask = torch.zeros(a.bs, max_len, dtype=torch.long)
    for i, t in enumerate(token_ids_batch):
        input_ids[i, :len(t)] = torch.tensor(t)
        attn_mask[i, :len(t)] = 1
    enc = {"input_ids": input_ids.to("cuda"),
           "attention_mask": attn_mask.to("cuda")}
    actual_ctx = int(enc["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()

    # Prefill (dense, both arms)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    # Warmup
    for _ in range(a.warmup):
        out, _ = bench_step(model, next_tok, past)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gc.collect()

    # Timed
    records = []
    for _ in range(a.repeats):
        out, dt = bench_step(model, next_tok, past)
        records.append(dt)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    records.sort()
    tpot = records[len(records) // 2]
    peak = torch.cuda.max_memory_allocated()

    return {
        "method": method, "bs": a.bs,
        "ctx_len": a.ctx, "actual_ctx": actual_ctx,
        "tpot_ms": round(tpot, 3),
        "peak_mem_miB": round(peak / 1048576, 1),
        "n_timed_steps": len(records),
        "repeats": a.repeats, "warmup": a.warmup,
        "all_records_ms": [round(r, 3) for r in records],
    }


def main():
    p = argparse.ArgumentParser(description="E178 FA2 speedup")
    p.add_argument("--method", required=True,
                   choices=["fa2", "tli", "tli+fs"])
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-gen", type=int, default=32)
    p.add_argument("--model-path", default="~/Qwen3-8B")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    model_path = os.path.expanduser(a.model_path)
    method = a.method
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if method == "fa2":
        model = load_model_fa2(model_path)
    else:
        model = load_model_tli(model_path, method)
    model.eval()

    result = bench_decode(model, tokenizer, method, a)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[E178] {method:8s} L{a.ctx//1024}K bs={a.bs} "
          f"TPOT={result['tpot_ms']:.2f}ms "
          f"peak={result['peak_mem_miB']:.0f}MiB "
          f"ctx_actual={result['actual_ctx']}")


if __name__ == "__main__":
    main()
