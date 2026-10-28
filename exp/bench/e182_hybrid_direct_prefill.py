# -*- coding: utf-8 -*-
"""E182：hybrid direct/conversion 的真实分配、峰值与 decode 对拍采集器。"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.RULER.yarn_receipt import resolve_yarn_config  # noqa: E402
from exp.bench.e152v4_graph_latency import (  # noqa: E402
    QWEN3_NATIVE_MPE, YARN_AUTO, _dynamic_storage_snapshot, _memory_snapshot,
    _storage_snapshot, build_prompt_tokens, sha)
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.info import resolve_treatment_snapshot  # noqa: E402
from sparse_attn.ops.cuda_graph_state import disable_graph  # noqa: E402
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes  # noqa: E402
from sparse_attn.ops.hybrid_batch_cache import HybridBatchCacheBuilder  # noqa: E402
from sparse_attn.ops.hybrid_batch_runner import HybridBatchRunner  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--max-gen", type=int, default=16)
    parser.add_argument("--model-path", default=os.path.expanduser("~/Qwen3-8B"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefill-mode", choices=["direct", "conversion"], required=True)
    parser.add_argument("--graph-mode", choices=["graph", "eager"], default="graph")
    add_sparse_attn_args(parser)
    args = parser.parse_args()
    if (args.method != "tli" or args.max_gen < 2
            or any(type(length) is not int or length < 16384 for length in args.lengths)):
        parser.error("E182 requires method=tli, max_gen>=2 and every length>=16384")
    os.environ.setdefault("TLI_SPARSE_PREFILL", "0")
    if os.environ["TLI_SPARSE_PREFILL"] != "0":
        parser.error("E182 requires dense prefill")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    max_length = max(args.lengths)
    factor, rope_scaling = resolve_yarn_config(
        max_length > QWEN3_NATIVE_MPE, None, max_length,
        YARN_AUTO, QWEN3_NATIVE_MPE)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if max_length > QWEN3_NATIVE_MPE:
        config.rope_scaling = rope_scaling
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    resolve_far_skip_modes(args, model=model)
    snapshot = resolve_treatment_snapshot(args)
    if register_patch(model, args, snapshot) == 0:
        raise SystemExit("E182 patched zero attention modules")

    prompts = [build_prompt_tokens(tokenizer, length) for length in args.lengths]
    lengths = [len(ids) for ids in prompts]
    capacity = max(lengths) + args.max_gen + 8
    direct = args.prefill_mode == "direct"
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    rows, generated = [], [[] for _ in lengths]
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    memory_before = _memory_snapshot()
    with torch.no_grad():
        builder = HybridBatchCacheBuilder(model, capacity, len(lengths))
        prefill_t0 = time.perf_counter()
        finalize_ms = 0.0
        for row, ids in enumerate(prompts):
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            row_cache = builder.prepare_row(len(ids)) if direct else None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, past_key_values=row_cache,
                        use_cache=True, logits_to_keep=1)
            dyn = None if direct else out.past_key_values
            first = int(out.logits[0, -1].argmax().item())
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1e3
            del out
            memory = _memory_snapshot()
            storage = {
                "static": _storage_snapshot(row_cache, model) if direct else None,
                "dynamic_source": _dynamic_storage_snapshot(dyn),
            }
            t1 = time.perf_counter()
            if direct:
                builder.finalize_row(row_cache, len(ids))
            else:
                builder.add(dyn, len(ids))
            torch.cuda.synchronize()
            elapsed_finalize = (time.perf_counter() - t1) * 1e3
            finalize_ms += elapsed_finalize
            rows.append({
                "row": row, "length": len(ids), "prefill_ms": prefill_ms,
                "finalize_or_add_ms": elapsed_finalize,
                "after_prefill_memory": memory, "prefill_storage": storage,
            })
            generated[row].append(first)
            del dyn, row_cache, input_ids
            gc.collect()
        prefill_wall_ms = (time.perf_counter() - prefill_t0) * 1e3
        prefill_ms = sum(row["prefill_ms"] for row in rows)
        runner = HybridBatchRunner(model, builder.finish(), pad_id=pad_id)
        after_finalize = _memory_snapshot()
        final_storage = _storage_snapshot(runner.cache, model)
        if final_storage["total_bytes"] != runner.kv_storage_bytes():
            raise RuntimeError("hybrid physical storage accounting mismatch")

        current = [length + 1 for length in lengths]
        runner.inject([tokens[-1] for tokens in generated], current)
        capture_t0 = time.perf_counter()
        if args.graph_mode == "graph":
            runner.capture()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - capture_t0) * 1e3
        after_capture = _memory_snapshot()
        step_ms = []
        for step in range(args.max_gen - 1):
            if step:
                current = [length + step + 1 for length in lengths]
                runner.inject([tokens[-1] for tokens in generated], current)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = runner.run()
            next_tokens = out.logits[:, -1].argmax(dim=-1).tolist()
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            for tokens, token in zip(generated, next_tokens):
                tokens.append(int(token))
            if args.graph_mode == "eager":
                del out
        steady = _memory_snapshot()
    disable_graph()
    record = {
        "schema": "e182-hybrid-direct-prefill",
        "caliber": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "model": args.model_path,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "prefill_mode": args.prefill_mode,
            "graph_mode": args.graph_mode,
            "lengths": lengths,
            "batch": len(lengths),
            "capacity": capacity,
            "max_gen": args.max_gen,
            "yarn_factor": factor,
            "treatment_manifest": json.loads(snapshot.manifest_json),
            "input_ids_sha256_16": [sha(torch.tensor([ids]).numpy().tobytes())[:16]
                                    for ids in prompts],
        },
        "rows": rows,
        "build_wall_ms": prefill_wall_ms,
        "prefill_ms": prefill_ms,
        "finalize_or_add_ms": finalize_ms,
        "capture_ms": capture_ms,
        "step_ms": step_ms,
        "tpot_ms": sum(step_ms) / len(step_ms),
        "generated": generated,
        "kv_storage_bytes": final_storage["total_bytes"],
        "kv_storage": final_storage,
        "memory": {
            "before_prefill": memory_before,
            "after_finalize": after_finalize,
            "after_capture": after_capture,
            "steady_decode": steady,
        },
        "prefill_peak_delta_gib": (max(row["after_prefill_memory"]["peak_allocated_bytes"]
                                       for row in rows) - base_bytes) / 2**30,
        "request_peak_delta_gib": (steady["peak_allocated_bytes"] - base_bytes) / 2**30,
    }
    with open(args.out, "x") as stream:
        json.dump(record, stream, indent=1)
    print(f"[E182] mode={args.prefill_mode} lengths={lengths} "
          f"prefill={prefill_ms:.1f}ms finalize={finalize_ms:.1f}ms "
          f"tpot={record['tpot_ms']:.2f}ms "
          f"KV={final_storage['total_bytes']/2**30:.3f}GiB "
          f"prefill_peak_delta={record['prefill_peak_delta_gib']:.3f}GiB", flush=True)


if __name__ == "__main__":
    main()
