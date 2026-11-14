"""Qwen3-8B 256K/512K 纯官方 Quest/JSSA 性能入口。

只复用 NFA 历史矩阵的扩展 RULER 输入文件与两点计时函数，不导入 NFA、
旧 vendor 或 Torch 稀疏注意力实现。RoPE 按统一 near-far 政策显式配置并验证，
不要求跨臂输出逐位一致。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from benchmark.official_baselines.longbench_official import (
    OfficialEngine,
    atomic_json,
    perf_summary,
    sha256_file,
    sha256_ids,
    timed_pair,
)
from benchmark.rope_policy import configure_rope, verify_runtime_rope

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("quest", "jssa"), required=True)
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--data-root", default="/tmp/nfa_kernel_inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="niah_multikey_2")
    parser.add_argument("--context-length", type=int, required=True,
                        choices=(262144, 524288))
    parser.add_argument("--sample-index", type=int, default=2)
    parser.add_argument("--input-provenance", default="repeated-needle-body-extension",
                        choices=("repeated-needle-body-extension",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--quest-max-seq-len", type=int, default=40960)
    parser.add_argument("--jssa-rank", type=int, default=32)
    parser.add_argument("--jssa-basis-path", default=str(REPO / "bases/qwen3_8b.pt"))
    parser.add_argument("--jssa-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    return parser.parse_args()


def load_performance_input(args, tokenizer):
    path = Path(args.data_root) / str(args.context_length) / f"{args.task}.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    file_row_index = 0
    row = rows[file_row_index]
    raw_ids = tokenizer(row["input"], truncation=False, return_tensors="pt").input_ids[0]
    ids = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
    actual = len(ids)
    if not 0.95 * args.context_length <= actual <= args.context_length:
        raise ValueError(
            f"扩展输入长度 {actual} 不在 {args.context_length} 的 95%--100% 合同内"
        )
    return ids, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "input_ids_sha256": sha256_ids(ids),
        "task": args.task,
        "sample_index": args.sample_index,
        "sample_id": row.get("index"),
        "file_row_index": file_row_index,
        "actual_prompt_tokens": actual,
        "provenance": args.input_provenance,
        "performance_only": True,
    }


def configure_long_context_rope(args, config):
    """在构造官方引擎前应用并预检 256K/512K 的统一 YaRN 政策。"""
    if config.model_type != "qwen3":
        raise ValueError("长上下文官方性能入口仅验收 Qwen3-8B")
    policy = configure_rope(
        config, args.context_length, args.max_new_tokens, yarn=True,
    )
    preflight = verify_runtime_rope(config)
    args.model_config = config
    args.quest_rope_mode = "hf-position-embeddings"
    args.hf_overrides = {
        "rope_scaling": getattr(config, "rope_scaling", None),
        "max_position_embeddings": int(config.max_position_embeddings),
    }
    return policy, preflight


def verify_engine_rope(engine, method):
    """验证实际构造后的配置，而不是只复述命令行或原始 config.json。"""
    if method == "quest":
        return verify_runtime_rope(engine.model)
    runtime_config = engine.model.llm_engine.vllm_config.model_config.hf_config
    return verify_runtime_rope(runtime_config)


def source_receipt():
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "head": head,
        "harness": str(Path(__file__).resolve()),
        "harness_sha256": sha256_file(__file__),
    }


def main():
    args = parse_args()
    if args.max_new_tokens < 2 or args.repeats <= 0 or args.warmup_requests < 0:
        raise ValueError("要求 max-new-tokens>=2、repeats>0、warmup-requests>=0")
    if args.budget <= 0 or args.budget % args.quest_page_size:
        raise ValueError("Quest budget 必须为正且整除 page-size")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_policy, rope_preflight = configure_long_context_rope(args, config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids, input_receipt = load_performance_input(args, tokenizer)
    engine = OfficialEngine(args, len(ids) + args.max_new_tokens + 16)
    runtime_rope = verify_engine_rope(engine, args.method)
    engine.runtime["long_context_rope_policy"] = {
        "quest": "HF Qwen3 YaRN position_embeddings exactly once before official Quest KV/attention",
        "jssa": "patched-vLLM effective HF YaRN config; no hidden HF/Torch attention fallback",
        "nfa_flashinfer_reference": "near-far-rope-32768-v1 with YaRN factor 8 at 256K and 16 at 512K",
        "cross_arm_output_identity_required": False,
        "reason": "performance-only comparison with explicitly configured and runtime-verified RoPE",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qwen-official-long-context-performance-v1",
        "verification": "pending-independent-verification",
        "status": "in_progress",
        "method": args.method,
        "model_path": str(Path(args.model_path).resolve()),
        "source": source_receipt(),
        "context_length": args.context_length,
        "task": args.task,
        "sample_index": args.sample_index,
        "input": input_receipt,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "measurement_mode": "clean-e2e-no-profiler",
        "timing_protocol": "T(33)-T(1) two-point complete request host wall; EOS ignored",
        "rope_policy": rope_policy,
        "rope_preflight": rope_preflight,
        "runtime_rope_verification": runtime_rope,
        "runtime": engine.runtime,
        "cells": [],
    }
    for _ in range(args.warmup_requests):
        timed_pair(engine, ids, args.max_new_tokens, require_positive=False)
    for repeat in range(args.repeats):
        generated, timing = timed_pair(engine, ids, args.max_new_tokens)
        payload["cells"].append({
            "task": args.task,
            "sample_index": args.sample_index,
            "repeat": repeat,
            "request_timing": timing,
            "output_ids_sha256": sha256_ids(generated),
        })
        payload["status"] = "complete" if repeat + 1 == args.repeats else "in_progress"
        payload["summary"] = perf_summary(payload["cells"])
        if args.method == "quest":
            payload["quest_call_counts"] = dict(engine.model.quest_runtime_stats)
            if payload["quest_call_counts"]["dense_or_torch_fallback"]:
                raise RuntimeError("Quest 触发禁止的 dense/Torch fallback")
        atomic_json(output, payload)
        print(
            f"[LONG PERF] {args.method} L={args.context_length} repeat={repeat} "
            f"TPOT={timing['tpot_ms']:.3f}ms", flush=True,
        )
    print(f"[LONG PERF DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
