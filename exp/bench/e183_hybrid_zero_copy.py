# -*- coding: utf-8 -*-
"""E183：hybrid final-storage direct prefill 的所有权、峰值与 decode 对拍。"""
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
    QWEN3_NATIVE_MPE, YARN_AUTO, _memory_snapshot, build_prompt_tokens, sha)
from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.info import resolve_treatment_snapshot  # noqa: E402
from sparse_attn.ops.cuda_graph_state import disable_graph  # noqa: E402
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes  # noqa: E402
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer  # noqa: E402
from sparse_attn.ops.hybrid_batch_cache import HybridBatchCacheBuilder  # noqa: E402
from sparse_attn.ops.hybrid_batch_runner import HybridBatchRunner  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402


def _ownership_snapshot(cache, staging_cache=None, dyn=None):
    """Deduplicate physical storages while retaining each tensor's view bytes."""
    entries, seen = [], set()
    physical, views = {}, {}

    def add(owner, role, layer, row):
        if owner is None:
            return
        for kind in ("keys", "values"):
            tensor = getattr(owner, kind, None)
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            view_bytes = tensor.numel() * tensor.element_size()
            owner_bytes = storage.nbytes()
            entries.append({
                "layer": layer, "row": row, "role": role, "kind": kind,
                "view_bytes": view_bytes, "physical_owner_bytes": owner_bytes,
                "view_data_ptr": tensor.data_ptr(), "storage_data_ptr": storage.data_ptr(),
                "physical_owner_counted": key not in seen,
            })
            views[role] = views.get(role, 0) + view_bytes
            if key not in seen:
                seen.add(key)
                physical[role] = physical.get(role, 0) + owner_bytes

    for li, layer in enumerate(cache.layers):
        rows = getattr(layer, "rows", None)
        if rows is None:
            add(layer, "linear", li, None)
        else:
            for row, owner in enumerate(rows):
                add(owner, "ring" if isinstance(owner, FarSkipStaticLayer) else "linear", li, row)
    staging = getattr(staging_cache, "_direct_prefill", None)
    if staging is not None:
        for li, state in enumerate(staging.states):
            add(state.linear, "staging_linear", li, None)
            add(state.ring, "staging_ring", li, None)
    if dyn is not None:
        for li, layer in enumerate(dyn.layers):
            add(layer, "dynamic_source", li, None)
    return {
        "total_bytes": sum(physical.values()),
        "physical_bytes_by_role": physical,
        "view_bytes_by_role": views,
        "entries": entries,
    }


def _addresses(cache):
    return [
        (li, row, kind, tensor.data_ptr(), tensor.untyped_storage().data_ptr())
        for li, layer in enumerate(cache.layers)
        for ri, owner in enumerate(getattr(layer, "rows", None) or [layer])
        for row, kind, tensor in [
            (None if getattr(layer, "rows", None) is None else ri, "keys", owner.keys),
            (None if getattr(layer, "rows", None) is None else ri, "values", owner.values),
        ] if tensor is not None
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--max-gen", type=int, default=16)
    parser.add_argument("--model-path", default=os.path.expanduser("~/Qwen3-8B"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefill-mode", choices=["direct", "conversion"], required=True)
    parser.add_argument("--graph-mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--eos-after", type=int, default=0,
                        help="0 disables EOS; otherwise rows 0,3,... freeze at this decode step")
    add_sparse_attn_args(parser)
    args = parser.parse_args()
    if (args.method != "tli" or args.max_gen < 2
            or any(type(length) is not int or length < 16384 for length in args.lengths)):
        parser.error("E183 requires method=tli, max_gen>=2 and every length>=16384")
    os.environ.setdefault("TLI_SPARSE_PREFILL", "0")
    if os.environ["TLI_SPARSE_PREFILL"] != "0":
        parser.error("E183 requires dense prefill")

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
        raise SystemExit("E183 patched zero attention modules")

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
            after_prefill = _memory_snapshot()
            ownership = _ownership_snapshot(builder.cache, row_cache, dyn)
            states = (list(row_cache._direct_prefill.states)
                      if direct else None)
            plans = [state.mode for state in states] if direct else None
            t1 = time.perf_counter()
            if direct:
                builder.finalize_row(row_cache, len(ids))
            else:
                builder.add(dyn, len(ids))
            torch.cuda.synchronize()
            elapsed_finalize = (time.perf_counter() - t1) * 1e3
            finalize_ms += elapsed_finalize
            compaction_storage = compaction_live = 0
            if direct:
                for li, state in enumerate(states):
                    layer = builder.cache.layers[li]
                    if state.mode != "LINEAR_PROVISIONAL" or not layer.indexers[-1].skip_far:
                        continue
                    owner = layer.rows[-1]
                    compaction_storage += sum(
                        getattr(owner, kind).untyped_storage().nbytes()
                        for kind in ("keys", "values"))
                    live = min(owner.config.sink_tokens, len(ids)) + len(ids) - owner.config.tail_start(len(ids))
                    compaction_live += (live * owner.keys.shape[1] * owner.keys.shape[-1]
                                        * owner.keys.element_size() * 2)
            after_finalize_row = _memory_snapshot()
            rows.append({
                "row": row, "length": len(ids), "prefill_ms": prefill_ms,
                "finalize_or_add_ms": elapsed_finalize,
                "storage_plans": plans,
                "finalize_compaction_storage_bytes": compaction_storage,
                "finalize_compaction_live_bytes": compaction_live,
                "after_prefill_memory": after_prefill,
                "after_finalize_memory": after_finalize_row,
                "prefill_ownership": ownership,
            })
            generated[row].append(first)
            del dyn, row_cache, input_ids
            gc.collect()
        prefill_wall_ms = (time.perf_counter() - prefill_t0) * 1e3
        prefill_ms = sum(row["prefill_ms"] for row in rows)
        runner = HybridBatchRunner(model, builder.finish(), pad_id=pad_id)
        after_finalize = _memory_snapshot()
        final_storage = _ownership_snapshot(runner.cache)
        if final_storage["total_bytes"] != runner.kv_storage_bytes():
            raise RuntimeError("hybrid physical storage accounting mismatch")
        addresses_after_finalize = _addresses(runner.cache)

        current = [length + 1 for length in lengths]
        runner.inject([tokens[-1] for tokens in generated], current)
        capture_t0 = time.perf_counter()
        if args.graph_mode == "graph":
            runner.capture()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - capture_t0) * 1e3
        after_capture = _memory_snapshot()
        addresses_after_capture = _addresses(runner.cache)
        step_ms = []
        alive = [True] * len(lengths)
        for step in range(args.max_gen - 1):
            if step:
                alive = [not (args.eos_after and step >= args.eos_after and row % 3 == 0)
                         for row in range(len(lengths))]
                current = [length + int(active) for length, active in zip(current, alive)]
                runner.inject([tokens[-1] for tokens in generated], current, alive)
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
        addresses_steady = _addresses(runner.cache)
    disable_graph()
    address_stable = addresses_after_finalize == addresses_after_capture == addresses_steady
    record = {
        "schema": "e183-hybrid-zero-copy",
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
            "eos_after": args.eos_after,
            "final_alive": alive,
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
        "kv_addresses_stable": address_stable,
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
    if not address_stable:
        raise RuntimeError("KV addresses changed across finalize/capture/replay")
    with open(args.out, "x") as stream:
        json.dump(record, stream, indent=1)
    print(f"[E183] mode={args.prefill_mode} lengths={lengths} "
          f"prefill={prefill_ms:.1f}ms finalize={finalize_ms:.1f}ms "
          f"tpot={record['tpot_ms']:.2f}ms "
          f"KV={final_storage['total_bytes']/2**30:.3f}GiB "
          f"prefill_peak_delta={record['prefill_peak_delta_gib']:.3f}GiB", flush=True)


if __name__ == "__main__":
    main()
