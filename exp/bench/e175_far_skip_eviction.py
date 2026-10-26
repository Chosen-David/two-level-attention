"""E175 e2e far-skip-kv-eviction decode TPOT+memory benchmark.

pc40 GPU1 (H20-3e). Eager-only decode (graph=eager baseline). Far-skip is
eager-only by design and rejected under CUDA Graph capture.

Usage:
  python exp/bench/e175_far_skip_eviction.py \
      --arm legacy --lengths 32768 --out out_legacy.json
  python exp/bench/e175_far_skip_eviction.py \
      --arm far_skip_evict --lengths 32768 --out out_evict.json
"""
import argparse
import json
import os
import sys
import time

import torch


os.environ.setdefault("TLI_SPARSE_PREFILL", "0")
os.environ.setdefault("TLI_DECODE_DENSE", "0")


def build_argv(arm, L):
    base = [
        "prog",
        "--method", "tli",
        "--tli_far_method", "minmax",
        "--tli_near_method", "avg",
        "--tli_near_pages", "32",
        "--tli_near_tokens", "off",
        "--tli_near_len", "4096",
        "--tli_subspace", "tail",
        "--tia_level2_cmp_ratio", "4",
        "--tli_sink_tokens", "4",
        "--tli_swa_size", "128",
        "--tia_level1_topk", "64",
        "--tia_level2_topk", "1024",
        "--tli_enable_layer_skip", "false",
        "--tli_boundary_bypass", "true",
        "--tli_far_select", "4bit",
        "--tli_near_select", "4bit",
    ]
    if arm == "far_skip_evict":
        base += [
            "--tli_far_skip_backend", "compact",
            "--tli_far_skip_cache", "evict",
        ]
    return base, L


def run_one(arm, L, warmup_steps, max_gen, model_path):
    argv, L = build_argv(arm, L)
    sys.argv = argv

    from sparse_attn.arguments import add_sparse_attn_args
    import argparse as _ap
    p = _ap.ArgumentParser()
    add_sparse_attn_args(p)
    args = p.parse_args(argv[1:])

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    # Synthetic prompt: L tokens of fixed vocab id 3 (IDX_VALID in tokenizers)
    ids = torch.full((1, L), 3, dtype=torch.long, device="cuda:0")

    from sparse_attn.patches.patch import register_patch
    register_patch(model, args, snapshot=None)

    with torch.inference_mode():
        # Prefill
        t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3

        # Reset memory counter post-prefill for decode-only peak
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()

        # Decode loop
        past = out.past_key_values
        cursor = L - 1 + 1  # last prompt position used
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_ms = []
        for step in range(warmup_steps + max_gen):
            t1 = time.perf_counter()
            out = model(
                next_tok, past_key_values=past,
                cache_position=torch.tensor([cursor + step], device="cuda:0"),
                use_cache=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t1) * 1e3
            if step >= warmup_steps:
                step_ms.append(elapsed)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = out.past_key_values

        mem_after = torch.cuda.max_memory_allocated()

    avg_tpot = sum(step_ms) / len(step_ms) if step_ms else None
    return {
        "arm": arm,
        "context_length": L,
        "prefill_ms": prefill_ms,
        "decode_tpot_ms": avg_tpot,
        "n_measured_steps": len(step_ms),
        "decode_mem_before_gib": mem_before / 2**30,
        "decode_mem_peak_gib": mem_after / 2**30,
        "decode_mem_delta_gib": (mem_after - mem_before) / 2**30,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",
                        choices=["legacy", "far_skip_evict", "far_skip_retain"],
                        default="legacy")
    parser.add_argument("--lengths", "--L", "-L", dest="lengths",
                        type=int, nargs="+", required=True)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--max-gen", type=int, default=8)
    parser.add_argument("--model-path",
                        default=os.path.expanduser("~/Qwen3-8B"))
    parser.add_argument("--shadow",
                        default=os.path.expanduser("~/tli_ako/solution_iter3"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["TLI_SHADOW_PKG"] = args.shadow

    results = []
    for L in args.lengths:
        rec = run_one(args.arm, L, args.warmup_steps, args.max_gen, args.model_path)
        results.append(rec)
        tpot = f"{rec['decode_tpot_ms']:.3f}" if rec['decode_tpot_ms'] else "N/A"
        print(f"[{args.arm}] L={L} TPOT={tpot}ms "
              f"decode_mem_peak={rec['decode_mem_peak_gib']:.3f}GiB")
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results,
                   "config": {"arm": args.arm, "shadow": args.shadow,
                              "model": args.model_path,
                              "warmup": args.warmup_steps, "max_gen": args.max_gen}},
                  f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
