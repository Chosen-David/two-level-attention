"""Qwen3-8B × LongBench v1 英文16任务 NFA/FlashInfer 两点法性能入口。

与官方 Quest 共用原始数据、prompt 构建、固定33 token、EOS不提前停止、
1 warmup pair + 3 repeats 和 T(33)-T(1) complete-request host-wall 协议。
每个任务使用独立 NFA cache arena / Graph session，避免跨输入长度复用固定图。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from benchmark.efficiency.benchmark_long_context_two_point import (
    RequestTwoPointEngine,
    source_receipt as long_context_source_receipt,
)
from benchmark.efficiency.benchmark_nfa_end_to_end import _load_model
from benchmark.official_baselines.longbench_official import (
    ENGLISH16,
    atomic_json,
    build_input,
    load_json,
    load_rows,
    perf_summary,
    sha256_file,
    sha256_ids,
    timed_pair,
)

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("nfa", "flashinfer"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-path", default=str(REPO / "benchmark/LongBench/config"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=ENGLISH16)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visible-len", type=int, default=8192)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    return parser.parse_args()


def source_receipt():
    receipt = long_context_source_receipt()
    receipt["longbench_harness"] = str(Path(__file__).resolve())
    receipt["longbench_harness_sha256"] = sha256_file(__file__)
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "HEAD", "--binary", "--",
         "benchmark/efficiency/benchmark_longbench_two_point.py"])
    receipt["longbench_source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    return receipt


def _release_task_engine(model, engine, method):
    """任务边界释放固定arena和Graph pool；不得在计时请求内调用。"""
    if method == "nfa" and getattr(model, "_nfa_graph_session_pool", None) is not None:
        model._nfa_graph_session_pool = None
    engine.request_receipts.clear()
    torch.cuda.empty_cache()


def run_performance(args, model, tokenizer, eos_ids, prompt_cfg, max_length,
                    *, engine_factory=RequestTwoPointEngine, pair=timed_pair):
    samples = args.sample_indices or [0]
    output = Path(args.output_dir) / f"{args.method}_performance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qwen-nfa-flashinfer-longbench-two-point-v1",
        "verification": "pending-independent-verification",
        "status": "in_progress",
        "method": args.method,
        "execution": args.execution,
        "batch_size": 1,
        "model_path": str(Path(args.model_path).resolve()),
        "source": source_receipt(),
        "max_length": max_length,
        "tasks": args.tasks,
        "sample_indices": samples,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "measurement_mode": "clean-e2e-no-profiler-no-per-step-cuda-events",
        "timing_protocol": "T(33)-T(1) two-point complete request host wall; EOS ignored",
        "task_runtime": {},
        "request_receipts": {},
        "cells": [],
    }
    expected = {
        (task, sample, repeat)
        for task in args.tasks for sample in samples for repeat in range(args.repeats)
    }
    with torch.inference_mode():
        for task in args.tasks:
            rows, data_sha = load_rows(args.data_root, task)
            engine = engine_factory(args, model, tokenizer, eos_ids)
            try:
                for sample in samples:
                    row = rows[sample]
                    ids, identity = build_input(
                        row, task, prompt_cfg[task], max_length, tokenizer)
                    for _ in range(args.warmup_requests):
                        pair(engine, ids, args.max_new_tokens, require_positive=False)
                    for repeat in range(args.repeats):
                        generated, timing = pair(engine, ids, args.max_new_tokens)
                        payload["cells"].append({
                            "task": task,
                            "sample_index": sample,
                            "repeat": repeat,
                            "input": identity,
                            "data_sha256": data_sha,
                            "request_timing": timing,
                            "output_ids_sha256": sha256_ids(generated),
                        })
                        complete = {
                            (c["task"], c["sample_index"], c["repeat"])
                            for c in payload["cells"]
                        } == expected
                        payload["status"] = "complete" if complete else "in_progress"
                        payload["summary"] = perf_summary(payload["cells"])
                        atomic_json(output, payload)
                        print(
                            f"[PERF] {args.method}-{args.execution} {task} "
                            f"sample={sample} repeat={repeat} "
                            f"TPOT={timing['tpot_ms']:.3f}ms", flush=True)
                payload["task_runtime"][task] = copy.deepcopy(engine.runtime)
                payload["request_receipts"][task] = copy.deepcopy(engine.request_receipts)
            finally:
                _release_task_engine(model, engine, args.method)
            atomic_json(output, payload)
    print(f"[PERF DONE] {output}", flush=True)
    return payload


def main():
    args = parse_args()
    if len(set(args.tasks)) != len(args.tasks) or any(t not in ENGLISH16 for t in args.tasks):
        raise ValueError("tasks 必须是唯一的 LongBench 英文16任务")
    if args.model_name != "Qwen3-8B":
        raise ValueError("当前入口仅验收 Qwen3-8B")
    if args.max_new_tokens < 2 or args.repeats <= 0 or args.warmup_requests < 0:
        raise ValueError("要求 max-new-tokens>=2、repeats>0、warmup-requests>=0")
    if args.visible_len <= 0:
        raise ValueError("visible-len 必须为正")
    prompt_cfg = load_json(Path(args.config_path) / "dataset2prompt.json")
    model_len_cfg = load_json(Path(args.config_path) / "model2maxlen.json")
    max_length = int(model_len_cfg[args.model_name])
    if max_length != 31500:
        raise ValueError("LongBench 对齐口径要求 max_length=31500")

    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_args = SimpleNamespace(
        method=args.method, model_path=args.model_path, context_length=max_length)
    model = _load_model(load_args, args.max_new_tokens)
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)
    run_performance(args, model, tokenizer, eos_ids, prompt_cfg, max_length)


if __name__ == "__main__":
    main()
