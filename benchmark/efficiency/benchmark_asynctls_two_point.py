"""Qwen3-8B AsyncTLS 论文语义复现的 32K–512K 干净两点性能入口。

固定 K2=1024、block64、K1=128 和 per-KV-group channel；逐行 prefill 到同一
固定 arena，随后执行真实 B1/2/4 batched Graph decode。该实现不是官方 TileLang，
也没有真实异步 host offload。计时主体不创建逐步 CUDA Event、不读取 GPU token、
不运行 profiler；输出只在 request-wall 停止后一次 D2H。
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmark.efficiency.benchmark_nfa_end_to_end import _load_model
from benchmark.official_baselines.long_context_official import load_performance_input
from benchmark.official_baselines.longbench_official import (
    atomic_json,
    perf_summary,
    sha256_file,
    sha256_ids,
    timed_pair,
)
from benchmark.rope_policy import verify_runtime_rope
from exp.vendor_baselines.asynctls_vendor.qwen3_asynctls_graph import (
    AsyncTLSQwen3GraphRuntime,
    bind_asynctls_graph_runtime,
    register_asynctls_graph_patch,
)
from sparse_attn.scheduler.scheduler import greedy_next_tokens

REPO = Path(__file__).resolve().parents[2]
ALIGNED_BUDGET = (1024, 64, 128)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--data-root", default="/tmp/nfa_kernel_inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="niah_multikey_2")
    parser.add_argument("--context-length", type=int, required=True,
                        choices=(32768, 65536, 131072, 262144, 524288))
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--execution", choices=("graph",), default="graph")
    parser.add_argument("--sample-index", type=int, default=2)
    parser.add_argument("--input-provenance", default="repeated-needle-body-extension",
                        choices=("repeated-needle-body-extension",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    parser.add_argument("--asynctls-channel-file", required=True)
    parser.add_argument("--asynctls-budget", type=int, default=1024)
    parser.add_argument("--asynctls-block-size", type=int, default=64)
    parser.add_argument("--asynctls-top-kb", type=int, default=128)
    return parser.parse_args()


def validate_aligned_budget(args):
    actual = (args.asynctls_budget, args.asynctls_block_size, args.asynctls_top_kb)
    if actual != ALIGNED_BUDGET:
        raise ValueError(
            f"AsyncTLS 对齐入口要求 token_budget/block_size/top_kb={ALIGNED_BUDGET}，"
            f"实际 {actual}"
        )


def source_receipt():
    paths = (
        "benchmark/efficiency/benchmark_asynctls_two_point.py",
        "benchmark/efficiency/benchmark_nfa_end_to_end.py",
        "benchmark/official_baselines/long_context_official.py",
        "benchmark/official_baselines/longbench_official.py",
        "benchmark/rope_policy.py",
        "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_graph.py",
        "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py",
        "sparse_attn/patches/qwen3_flashinfer_patch.py",
        "sparse_attn/scheduler/scheduler.py",
    )
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "HEAD", "--binary", "--", *paths])
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "harness": str(Path(__file__).resolve()),
        "harness_sha256": sha256_file(__file__),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "asynctls_runtime_sha256": sha256_file(
            REPO / "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_graph.py"),
        "asynctls_channel_loader_sha256": sha256_file(
            REPO / "exp/vendor_baselines/asynctls_vendor/qwen3_asynctls_patch.py"),
    }


def validate_request_receipt(num_layers, max_new_tokens, receipt, *,
                             allow_initial_capture=False):
    """核验逐行 token、真实 prefill、冷 capture 与 Graph replay 计数。"""
    steps = max_new_tokens - 1
    batch = receipt["batch_size"]
    if batch not in (1, 2, 4):
        raise RuntimeError(f"AsyncTLS 回执 batch_size 非法: {batch}")
    if receipt["generated_tokens"] != max_new_tokens * batch:
        raise RuntimeError("AsyncTLS 固定长度生成总数与回执不一致")
    if receipt["generated_tokens_per_request"] != [max_new_tokens] * batch:
        raise RuntimeError("AsyncTLS 逐请求固定长度生成计数与回执不一致")
    if receipt["prefill_calls"] != batch * num_layers:
        raise RuntimeError("AsyncTLS prefill 未逐请求覆盖全部层")
    expected_capture = int(steps > 0 and allow_initial_capture)
    actual_graph = (
        receipt["graph_capture_count"], receipt["graph_replays"],
        receipt["graph_eager_fallbacks"],
    )
    expected_graph = (expected_capture, steps, 0)
    if actual_graph != expected_graph:
        raise RuntimeError(
            f"AsyncTLS Graph 计数 mismatch: {actual_graph} != {expected_graph}")
    expected_decode = 2 * num_layers if expected_capture else 0
    if receipt["python_decode_calls_warmup_and_capture"] != expected_decode:
        raise RuntimeError(
            "AsyncTLS Python decode 调用只应出现在首个请求的 warmup/capture，"
            "Graph replay 不得冒充 Python backend 调用"
        )


class AsyncTLSTwoPointEngine:
    """同一模型和固定 AsyncTLS arena 上的 T(1)/T(33) 请求引擎。"""

    def __init__(self, args, model, tokenizer, prompt_length):
        validate_aligned_budget(args)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.prompt_length = prompt_length
        self.request_receipts = []
        self.graph = None
        self.static_logits = None
        self.graph_capture_count = 0
        self.graph_replay_count = 0
        self.runtime_impl = AsyncTLSQwen3GraphRuntime(
            model, args.batch_size, prompt_length, args.max_new_tokens,
            device=args.device,
            channel_file=args.asynctls_channel_file,
            token_budget=args.asynctls_budget,
            block_size=args.asynctls_block_size,
            top_kb=args.asynctls_top_kb,
        )
        bind_asynctls_graph_runtime(model, self.runtime_impl)
        self.input_buf = torch.empty(
            (args.batch_size, 1), dtype=torch.long, device=args.device)
        self.position_buf = torch.empty(
            (args.batch_size, 1), dtype=torch.long, device=args.device)
        self.cache_position_buf = torch.empty(
            (1,), dtype=torch.long, device=args.device)
        descriptor = self.runtime_impl.receipt()
        descriptor.pop("prefill_calls", None)
        descriptor.pop("decode_calls_warmup_and_capture", None)
        self.runtime = {
            **descriptor,
            "execution": "graph",
            "batch_size": args.batch_size,
            "prefill_mode": "serial B1 prefill per row into one fixed cache arena",
            "fixed_cache_arena": True,
            "graph_reuse_across_requests": True,
            "dense_or_torch_decode_fallback": False,
            "official_tilelang": False,
            "real_async_offload": False,
        }

    def _forward(self):
        return self.model(
            input_ids=self.input_buf,
            position_ids=self.position_buf,
            cache_position=self.cache_position_buf,
            past_key_values=self.runtime_impl.mask_metadata,
            use_cache=True,
            logits_to_keep=1,
        )

    def generate(self, ids, max_new_tokens, ignore_eos):
        if not ignore_eos:
            raise ValueError("该性能入口只支持固定长度 ignore_eos=True")
        if max_new_tokens < 1 or max_new_tokens > self.args.max_new_tokens:
            raise ValueError("请求生成长度超出固定 AsyncTLS arena 容量")
        args = self.args
        batch = self.batch_size
        runtime = self.runtime_impl
        prompt = torch.tensor(ids, dtype=torch.long, device=args.device)
        if prompt.numel() != self.prompt_length:
            raise RuntimeError("固定 Graph arena 不允许跨输入长度复用")
        decode_steps = max_new_tokens - 1
        request_index = len(self.request_receipts)
        graph_capture_before = self.graph_capture_count
        graph_replay_before = self.graph_replay_count
        graph_prepare_seconds = 0.0

        torch.cuda.synchronize()
        started = time.perf_counter()
        runtime.prefill_calls = 0
        runtime.decode_calls = 0
        runtime.initial_blocks = None
        input_ids = prompt.unsqueeze(0)
        cache_position = torch.arange(self.prompt_length, device=args.device)
        position_ids = cache_position.unsqueeze(0)
        first_token_rows = []
        for _ in range(batch):
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=runtime.mask_metadata,
                use_cache=True,
                logits_to_keep=1,
            )
            first_token_rows.append(greedy_next_tokens(
                output.logits, (), generated_counts=0, min_new_tokens=1))
            del output
        first_tokens = torch.cat(first_token_rows)
        del input_ids, position_ids, cache_position

        runtime.prepare_decode(args.execution, decode_steps)
        token_history = torch.empty(
            (decode_steps, batch), dtype=torch.long, device=args.device)
        if decode_steps:
            self.input_buf.copy_(first_tokens[:, None])
            self.position_buf.fill_(self.prompt_length)
            self.cache_position_buf.fill_(self.prompt_length)
            if self.graph is None:
                capture_started = time.perf_counter()
                runtime.prepare_decode_step(0, reset_pipeline=True)
                self._forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                runtime.prepare_decode_step(0, reset_pipeline=True)
                with torch.cuda.graph(graph):
                    self.static_logits = self._forward()
                self.graph = graph
                self.graph_capture_count += 1
                graph_prepare_seconds = time.perf_counter() - capture_started

            for step in range(decode_steps):
                self.input_buf.copy_(first_tokens[:, None] if step == 0 else next_tokens[:, None])
                self.position_buf.fill_(self.prompt_length + step)
                self.cache_position_buf.fill_(self.prompt_length + step)
                # capture/warmup 后恢复 M_(t-1)，首个正式 replay 不重复使用 capture 状态。
                runtime.prepare_decode_step(step, reset_pipeline=(step == 0))
                self.graph.replay()
                self.graph_replay_count += 1
                next_tokens = greedy_next_tokens(
                    self.static_logits.logits, (), generated_counts=step + 1,
                    min_new_tokens=1)
                token_history[step].copy_(next_tokens)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        all_tokens = torch.cat(
            (first_tokens[:, None], token_history.transpose(0, 1)), dim=1)
        generated_rows = all_tokens.cpu().tolist()
        generated = generated_rows[0] if batch == 1 else generated_rows
        receipt = {
            "request_index": request_index,
            "request_role": (
                "warmup" if request_index < 2 * args.warmup_requests else "measured"),
            "cache_reused": request_index > 0,
            "batch_size": batch,
            "generated_tokens": sum(map(len, generated_rows)),
            "generated_tokens_per_request": list(map(len, generated_rows)),
            "decode_steps": decode_steps,
            "prefill_calls": runtime.prefill_calls,
            "python_decode_calls_warmup_and_capture": runtime.decode_calls,
            "graph_capture_count": self.graph_capture_count - graph_capture_before,
            "graph_replays": self.graph_replay_count - graph_replay_before,
            "graph_eager_fallbacks": 0,
            "graph_prepare_ms": graph_prepare_seconds * 1000.0,
        }
        validate_request_receipt(
            self.model.config.num_hidden_layers, max_new_tokens, receipt,
            allow_initial_capture=(graph_capture_before == 0),
        )
        self.request_receipts.append(receipt)
        self.runtime["graph_capture_count"] = self.graph_capture_count
        self.runtime["graph_replay_count"] = self.graph_replay_count
        return generated, elapsed


def main():
    args = parse_args()
    validate_aligned_budget(args)
    if args.max_new_tokens < 2 or args.repeats <= 0 or args.warmup_requests < 0:
        raise ValueError("要求 max-new-tokens>=2、repeats>0、warmup-requests>=0")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids, input_receipt = load_performance_input(args, tokenizer)
    args.method = "asynctls-semantic-pytorch"
    model = _load_model(args, args.max_new_tokens)
    patched = register_asynctls_graph_patch(model)
    if patched != model.config.num_hidden_layers:
        raise RuntimeError(
            f"AsyncTLS patched {patched}/{model.config.num_hidden_layers} layers")
    engine = AsyncTLSTwoPointEngine(args, model, tokenizer, len(ids))
    rope_receipt = verify_runtime_rope(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qwen-asynctls-semantic-long-context-two-point-v1",
        "verification": "pending-independent-verification",
        "status": "in_progress",
        "method": "asynctls-semantic-pytorch",
        "execution": "graph",
        "batch_size": args.batch_size,
        "model_path": str(Path(args.model_path).resolve()),
        "source": source_receipt(),
        "context_length": args.context_length,
        "task": args.task,
        "sample_index": args.sample_index,
        "input": input_receipt,
        "channel_file": str(Path(args.asynctls_channel_file).resolve()),
        "channel_file_sha256": sha256_file(args.asynctls_channel_file),
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "measurement_mode": "clean-e2e-no-profiler-no-per-step-cuda-events",
        "timing_protocol": (
            "T(33)-T(1) two-point complete batched-request host wall; EOS ignored; "
            "TPOT is per batched decode step and is not divided by batch"),
        "batch_input_mode": "identical input IDs replicated across batch rows",
        "host_readback_in_decode_loop": False,
        "final_token_readback": "one D2H after request-wall timer stops",
        "runtime_rope_verification": rope_receipt,
        "runtime": engine.runtime,
        "request_receipts": engine.request_receipts,
        "gpu_hygiene": {"status": "requires external process monitor"},
        "cells": [],
    }
    with torch.inference_mode():
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
            atomic_json(output, payload)
            print(
                f"[LONG PERF] asynctls-semantic-graph L={args.context_length} "
                f"B={args.batch_size} repeat={repeat} TPOT={timing['tpot_ms']:.3f}ms",
                flush=True)
    print(f"[LONG PERF DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
