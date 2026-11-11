"""JSSA official-vLLM runner: execute the unmodified JSSA vLLM backend.

This is the faithful JSSA path: it drives the patched vLLM tree
(`kv_cache_dtype="jssa"` + `attention_config={"backend": "JSSA", ...}`) exactly
like the official `benchmarks/decode_throughput.py`, and produces RULER-style
prediction JSONL for the existing scorers. The HF `qwen3_jssa_patch.py` remains
only as an eager semantics reference; this runner is the real upstream backend.

Requires the vLLM tree at pinned commit `39910f2b25aacc09f5e7f166cdf0030b19f8b9e8`
with `jssa_vllm.patch` applied (see upstream/vllm_patch/setup_vllm.sh). Import of
vllm is deferred so this file can be read/compiled before vLLM is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _ruler_max_new_tokens(task: str) -> int:
    """Same default as benchmark/RULER/pred_ruler.py for the 11 RULER tasks."""
    if task.startswith("niah_"):
        return 128
    if task == "cwe":
        return 120
    if task == "fwe":
        return 50
    if task == "vt":
        return 30
    return 128


def _yarn_overrides(model: str) -> dict:
    """Qwen3 YaRN factor 4, mirroring upstream decode_throughput.yarn_overrides."""
    if "qwen3" not in model.lower():
        return {}
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rp.get("rope_type", "default") != "default":
        return {}
    theta = rp.get("rope_theta", getattr(cfg, "rope_theta", None) or 1000000)
    return {"rope_parameters": {"rope_theta": theta, "rope_type": "yarn",
                                "factor": 4.0, "original_max_position_embeddings": 32768}}


def build_llm(args):
    from vllm import LLM
    kw = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=args.seed,
    )
    ov = _yarn_overrides(args.model)
    if ov:
        kw["hf_overrides"] = ov
    kw["kv_cache_dtype"] = "jssa"
    kw["attention_config"] = {
        "backend": "JSSA",
        "jssa_basis_path": args.basis,
        "jssa_budget": args.budget,
        "jssa_rank": args.rank,
    }
    kw["kv_cache_dtype_skip_layers"] = ["sliding_window"]
    return LLM(**kw)


def read_rows(args):
    data_file = Path(args.data_root) / str(args.context_length) / f"{args.task}.jsonl"
    rows = [json.loads(line) for line in open(data_file)]
    if args.skip:
        rows = rows[args.skip:]
    if args.max_num and len(rows) > args.max_num:
        rows = rows[:args.max_num]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run JSSA through the official patched vLLM backend")
    ap.add_argument("--model", required=True)
    ap.add_argument("--basis", required=True, help="JSSA basis .pt from jssa_calibrate.py")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--max-num", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    max_new = args.max_new_tokens or _ruler_max_new_tokens(args.task)
    if args.max_model_len is None:
        args.max_model_len = args.context_length + max_new + 16

    from vllm import SamplingParams
    llm = build_llm(args)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=True)
    rows = read_rows(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fout:
        prompts = [row["input"] for row in rows]
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        for row, gen in zip(rows, outputs):
            pred = gen.outputs[0].text
            answers = row.get("outputs")
            fout.write(json.dumps({
                "pred": pred,
                "answers": answers,
                "length": row.get("length"),
                "budget": args.budget,
                "execution": {"backend": "vllm_jssa", "official_patch": True},
                "_id": f"{args.task}:{row.get('index')}",
                "_answers_sha": hashlib.sha256(
                    json.dumps(answers, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16],
            }, ensure_ascii=False) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
