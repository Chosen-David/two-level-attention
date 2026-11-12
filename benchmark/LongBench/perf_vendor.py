"""Qwen3-8B × LongBench v1 英文16任务的 vendor 完整请求计时。

复用 pred.py 的真实 prompt、截断、chat、question split 与生产生成循环。
默认 clean-e2e 不启用 profiler；operator-profile 只供外部 nsys 单独运行，
两种产物使用不同输出路径，不能相减推导 host 开销。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from benchmark.LongBench.pred import (
    get_pred,
    load_model_and_tokenizer,
    parse_args as parse_pred_args,
    seed_everything,
)
from benchmark.LongBench.run_cap_manifest import (
    build_lbv1_input_identity,
    resolve_producer_identity,
)


REPO = Path(__file__).resolve().parents[2]
ENGLISH16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
METHODS = ("quest_vendor_kernel", "jssa_official", "asynctls_vendor")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=ENGLISH16)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    parser.add_argument(
        "--measurement-mode", choices=("clean-e2e", "operator-profile"),
        default="clean-e2e")
    return parser.parse_args(args)


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class HostWallRequestTimer:
    """生产 eager 循环的 host-wall 边界；逐 token .item()/int() 回传包含在内。"""

    def __init__(self, eos_token_id):
        if isinstance(eos_token_id, (list, tuple, set, frozenset)):
            self.eos_ids = frozenset(eos_token_id)
        else:
            self.eos_ids = frozenset((eos_token_id,))
        self.started = None
        self.first = None
        self.last = None
        self.tokens = []
        self.receipt = None

    def start(self):
        if self.started is not None:
            raise RuntimeError("request timer started twice")
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def token_ready(self, token):
        if self.started is None or self.receipt is not None:
            raise RuntimeError("request timer token outside active request")
        self.tokens.append(int(token))
        if self.first is None:
            torch.cuda.synchronize()
        # 生产路径的 token.item()/int(token) 已完成本步回传；首 token 上面
        # 再做一次边界同步，后续直接记录 host 时间，不新增逐步 CUDA sync。
        self.last = time.perf_counter()
        if self.first is None:
            self.first = self.last

    def finish(self):
        if self.started is None or self.first is None or self.last is None:
            raise RuntimeError("request timer has no generated token")
        if self.receipt is not None:
            raise RuntimeError("request timer finished twice")
        torch.cuda.synchronize()
        ended = time.perf_counter()
        count = len(self.tokens)
        decode_ms = (self.last - self.first) * 1000.0
        self.receipt = {
            "schema": "longbench-v1-request-wall-v1",
            "ttft_ms": (self.first - self.started) * 1000.0,
            "decode_after_first_wall_ms": decode_ms,
            "tpot_ms": decode_ms / (count - 1) if count > 1 else None,
            "request_wall_ms": (ended - self.started) * 1000.0,
            "actual_generated_tokens": count,
            "generated_token_ids": list(self.tokens),
            "eos_positions": [
                index for index, token in enumerate(self.tokens)
                if token in self.eos_ids
            ],
            "eos_early_stop": False,
            "decode_protocol": "fixed length; EOS recorded but not used for early stop",
            "timing_scope": (
                "generation context including cache/index work, greedy argmax, "
                "per-token host readback and boundary synchronization"),
            "timing_metric": "complete request host wall, not kernel-only latency",
            "host_readback_in_decode_loop": True,
        }


def _pred_argv(args):
    argv = [
        "--task", args.tasks[0], "--model", "Qwen3-8B",
        "--model_path", args.model_path,
        "--config-path", args.config_path,
        "--dataset-path", args.data_root,
        "--method", args.method,
        "--nfa_enable_layer_skip", "false",
    ]
    if args.method == "quest_vendor_kernel":
        argv += [
            "--quest_token_budget", "1024", "--quest_chunk_size", "16",
            "--quest_page_size", "16", "--quest_max_seq_len", "40960",
        ]
    elif args.method == "jssa_official":
        argv += [
            "--jssa_budget", "1024", "--jssa_rank", "32",
            "--jssa_basis_path", str(REPO / "bases/qwen3_8b.pt"),
        ]
    else:
        argv += [
            "--asynctls_token_budget", "1024", "--asynctls_block_size", "64",
            "--asynctls_top_kb", "128", "--asynctls_channel_file",
            str(REPO / "dimension_channels/qwen3-8b/qasper_channels_per_group.json"),
        ]
    return argv


def _source_files(method):
    files = [
        Path(__file__),
        REPO / "benchmark/LongBench/pred.py",
    ]
    if method == "quest_vendor_kernel":
        files.append(REPO / "exp/vendor_baselines/quest_vendor/qwen3_quest_kernel_patch.py")
    elif method == "jssa_official":
        files.append(REPO / "exp/vendor_baselines/jssa_vendor/qwen3_jssa_official.py")
    else:
        files.append(REPO / "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py")
    return {str(path.relative_to(REPO)): _sha256(path) for path in files}


def _identity(args, max_length):
    treatment = {
        "method": args.method,
        "model": "Qwen3-8B",
        "effective_cap": max_length,
        "max_new_tokens": args.max_new_tokens,
        "execution": "eager",
        "batch_size": 1,
        "decode_protocol": "fixed length; EOS recorded but not used for early stop",
    }
    if args.method == "quest_vendor_kernel":
        treatment.update(
            token_budget=1024, chunk_size=16, page_size=16, max_seq_len=40960,
            implementation="official Quest CUDA operators")
        extension = next((REPO / "exp/vendor_baselines/quest_vendor/upstream/quest").glob(
            "_kernels*.so"), None)
        if extension is None:
            raise FileNotFoundError("compiled official Quest _kernels extension is missing")
        asset = {"quest_kernel_extension_sha256": _sha256(extension)}
    elif args.method == "jssa_official":
        basis = REPO / "bases/qwen3_8b.pt"
        treatment.update(
            budget=1024, rank=32, page_size=64,
            implementation="official vendored Triton store/score/indexed-attend")
        asset = {"jssa_basis_sha256": _sha256(basis)}
    else:
        channels = REPO / "dimension_channels/qwen3-8b/qasper_channels_per_group.json"
        treatment.update(
            token_budget=1024, block_size=64, top_kb=128,
            implementation="AsyncTLS paper-semantic reproduction")
        asset = {"asynctls_channel_file_sha256": _sha256(channels)}
    return {
        "schema": "longbench-v1-vendor-performance-identity-v1",
        "treatment": treatment,
        "model_path": str(Path(args.model_path).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "tasks": list(args.tasks),
        "sample_indices": list(args.sample_indices),
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "measurement_mode": args.measurement_mode,
        "producer": resolve_producer_identity(__file__),
        "source_files_sha256": _source_files(args.method),
        "calibration": asset,
    }


def _load_task(data_root, task):
    path = Path(data_root) / f"{task}.jsonl"
    with open(path, "rb") as handle:
        raw = handle.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    return rows, hashlib.sha256(raw).hexdigest()


def _run_one(model, tokenizer, row, args, task, max_length, prompt_format,
             data_fp, timed):
    run_stats = {}
    timers = []

    def timing_factory():
        timer = HostWallRequestTimer(tokenizer.eos_token_id)
        timers.append(timer)
        return timer

    records = get_pred(
        model, tokenizer, [row], max_length, args.max_new_tokens,
        prompt_format, task, torch.device("cuda"), "Qwen3-8B",
        data_fp=data_fp, run_stats=run_stats,
        request_timing_factory=timing_factory if timed else None,
        stop_on_eos=False,
    )
    if len(records) != 1:
        raise RuntimeError("performance request must produce exactly one record")
    identity = {
        "input_tokens": run_stats["input_tokens"][0],
        "input_ids_sha256": run_stats["input_ids_sha256"][0],
        "truncated": run_stats["truncated"][0],
    }
    result = {
        "input": identity,
        "jssa_runtime": records[0].get("jssa_runtime"),
    }
    if timed:
        timing = records[0].get("request_timing")
        if timing is None or timing["actual_generated_tokens"] != args.max_new_tokens:
            raise RuntimeError("fixed-length request timing is missing or incomplete")
        result["request_timing"] = timing
    return result


def _median(values):
    return statistics.median(values) if values else None


def _summary(cells):
    metrics = ("ttft_ms", "tpot_ms", "request_wall_ms")
    per_task = {}
    for task in sorted({cell["task"] for cell in cells}):
        selected = [cell for cell in cells if cell["task"] == task]
        per_task[task] = {
            metric: _median([
                cell["request_timing"][metric] for cell in selected
                if cell["request_timing"][metric] is not None
            ])
            for metric in metrics
        }
        per_task[task]["requests"] = len(selected)
    return {
        "per_task": per_task,
        "macro_task_mean": {
            metric: statistics.mean([
                values[metric] for values in per_task.values()
                if values[metric] is not None
            ]) if per_task else None
            for metric in metrics
        },
        "all_request_median": {
            metric: _median([
                cell["request_timing"][metric] for cell in cells
                if cell["request_timing"][metric] is not None
            ])
            for metric in metrics
        },
    }


def _expected_keys(args):
    return {
        (task, sample, repeat)
        for task in args.tasks for sample in args.sample_indices
        for repeat in range(args.repeats)
    }


def _write(path, payload, args):
    cells = payload["cells"]
    complete = {(c["task"], c["sample_index"], c["repeat"])
                for c in cells} == _expected_keys(args)
    payload["status"] = "complete" if complete else "in_progress"
    payload["verification"] = "pending-independent-verification"
    payload["summary"] = _summary(cells)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main(args=None):
    args = parse_args(args)
    if len(set(args.tasks)) != len(args.tasks) or any(
            task not in ENGLISH16 for task in args.tasks):
        raise ValueError("tasks must be unique members of the LongBench English16 set")
    if (args.warmup_requests < 0 or args.repeats < 1
            or args.max_new_tokens < 2
            or any(index < 0 for index in args.sample_indices)):
        raise ValueError("require warmup>=0, repeats>=1, max-new>=2 and sample indices>=0")
    with open(Path(args.config_path) / "model2maxlen.json", encoding="utf-8") as handle:
        max_length = int(json.load(handle)["Qwen3-8B"])
    if max_length != 31500:
        raise ValueError(f"aligned LongBench cap must be 31500, got {max_length}")
    with open(Path(args.config_path) / "dataset2prompt.json", encoding="utf-8") as handle:
        prompts = json.load(handle)

    output = Path(args.output)
    identity = _identity(args, max_length)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise RuntimeError("existing performance output has a different identity")
    else:
        payload = {"identity": identity, "cells": []}
    if {(c["task"], c["sample_index"], c["repeat"])
            for c in payload["cells"]} == _expected_keys(args):
        _write(output, payload, args)
        print(f"SKIP complete performance matrix: {output}", flush=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    pred_args = parse_pred_args(_pred_argv(args))
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, "Qwen3-8B", torch.device("cuda"), pred_args)
    payload["device"] = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
    }
    payload["profiler_clean_e2e_isolated"] = True
    payload["host_overhead_derivable_by_subtraction"] = False
    _write(output, payload, args)

    completed = {(c["task"], c["sample_index"], c["repeat"])
                 for c in payload["cells"]}
    for task in args.tasks:
        rows, source_sha256 = _load_task(args.data_root, task)
        input_identity = build_lbv1_input_identity(rows, prompts[task])
        data_fp = input_identity["source_fingerprint"]
        for sample_index in args.sample_indices:
            if sample_index >= len(rows):
                raise IndexError(f"{task} sample {sample_index} outside {len(rows)} rows")
            missing = [repeat for repeat in range(args.repeats)
                       if (task, sample_index, repeat) not in completed]
            if not missing:
                continue
            expected_input = None
            for _ in range(args.warmup_requests):
                warmup = _run_one(
                    model, tokenizer, rows[sample_index], args, task, max_length,
                    prompts[task], data_fp, timed=False)
                if expected_input is None:
                    expected_input = warmup["input"]
                elif warmup["input"] != expected_input:
                    raise RuntimeError("warmup input identity changed")
            for repeat in missing:
                measured = _run_one(
                    model, tokenizer, rows[sample_index], args, task, max_length,
                    prompts[task], data_fp, timed=True)
                if expected_input is not None and measured["input"] != expected_input:
                    raise RuntimeError("measured input differs from warmup input")
                cell = {
                    "task": task,
                    "sample_index": sample_index,
                    "repeat": repeat,
                    "source_sha256": source_sha256,
                    "source_fingerprint": data_fp,
                    **measured,
                }
                payload["cells"].append(cell)
                completed.add((task, sample_index, repeat))
                _write(output, payload, args)
                timing = measured["request_timing"]
                print(json.dumps({
                    "method": args.method, "task": task,
                    "sample_index": sample_index, "repeat": repeat,
                    "input_tokens": measured["input"]["input_tokens"],
                    "ttft_ms": timing["ttft_ms"], "tpot_ms": timing["tpot_ms"],
                    "request_wall_ms": timing["request_wall_ms"],
                }), flush=True)
    _write(output, payload, args)
    print(f"COMPLETE pending-independent-verification: {output}", flush=True)


if __name__ == "__main__":
    main()
