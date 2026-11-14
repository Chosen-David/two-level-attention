"""Qwen3-8B 32K–512K NFA/FlashInfer 完整请求两点法性能入口。

与官方 Quest/JSSA 共用输入、warmup、固定33 token 和 T(33)-T(1) host-wall
协议；不创建逐步 CUDA Event，不在 decode 循环中读取 GPU token，不运行 profiler。
每个进程只测一个方法、长度、batch 和执行模式。NFA 支持真实 B1/2/4 batched
decode；FlashInfer 当前仅接受 B1，不以串行 B1 冒充 batch。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmark.efficiency.benchmark_nfa_end_to_end import (
    _load_model,
    _validate_flashinfer_runtime,
)
from benchmark.official_baselines.long_context_official import load_performance_input
from benchmark.official_baselines.longbench_official import (
    atomic_json,
    perf_summary,
    sha256_file,
    sha256_ids,
    timed_pair,
)
from benchmark.rope_policy import verify_runtime_rope
from sparse_attn.engine.frozen_runner import build_frozen_cache, prefill_frozen_request
from sparse_attn.engine.graph_runner import session_graph_runner
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches.qwen3_flashinfer_patch import (
    FlashInferQwen3Runtime,
    bind_flashinfer_runtime,
)
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler, greedy_next_tokens

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("nfa", "flashinfer"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--data-root", default="/tmp/nfa_kernel_inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="niah_multikey_2")
    parser.add_argument("--context-length", type=int, required=True,
                        choices=(32768, 65536, 131072, 262144, 524288))
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--sample-index", type=int, default=2)
    parser.add_argument("--input-provenance", default="repeated-needle-body-extension",
                        choices=("repeated-needle-body-extension",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visible-len", type=int, default=8192)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33)
    return parser.parse_args()


def source_receipt():
    paths = (
        "sparse_attn",
        "benchmark/rope_policy.py",
        "benchmark/efficiency/benchmark_nfa_end_to_end.py",
        "benchmark/efficiency/benchmark_long_context_two_point.py",
        "benchmark/official_baselines/long_context_official.py",
        "benchmark/official_baselines/longbench_official.py",
    )
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "HEAD", "--binary", "--", *paths])
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "harness": str(Path(__file__).resolve()),
        "harness_sha256": sha256_file(__file__),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def validate_request_receipt(method, execution, num_layers, max_new_tokens, receipt,
                             *, allow_initial_nfa_capture=False):
    """在请求边界核验真实 decode/Graph/backend 计数，失败则不发布计时格。"""
    steps = max_new_tokens - 1
    batch_size = receipt.get("batch_size", 1)
    if batch_size not in (1, 2, 4):
        raise RuntimeError(f"回执 batch_size 非法: {batch_size}")
    if receipt["generated_tokens"] != max_new_tokens * batch_size:
        raise RuntimeError("固定长度生成总数与回执不一致")
    per_request = receipt.get("generated_tokens_per_request")
    if per_request is not None and per_request != [max_new_tokens] * batch_size:
        raise RuntimeError("逐请求固定长度生成计数与回执不一致")
    if method == "nfa":
        if receipt["nfa_decode_steps"] != steps:
            raise RuntimeError("NFA 实际 decode 步数与固定长度协议不一致")
        if execution == "graph":
            expected_capture = int(steps > 0 and allow_initial_nfa_capture)
            expected = (expected_capture, steps, 0)
        else:
            expected = (0, 0, steps)
        actual = (receipt["graph_capture_count"], receipt["graph_replays"],
                  receipt["graph_eager_fallbacks"])
        if actual != expected:
            raise RuntimeError(f"NFA Graph 计数 mismatch: {actual} != {expected}")
        return

    if receipt["prefill_append_calls"] != num_layers:
        raise RuntimeError("FlashInfer prefill append 未覆盖全部层")
    if steps == 0:
        expected_decode = 0
    elif execution == "graph":
        # 每个 Graph 请求在计时内执行一次 warmup forward 和一次 capture forward。
        expected_decode = 2 * num_layers
    else:
        expected_decode = steps * num_layers
    actual_decode = (receipt["python_decode_append_calls"],
                     receipt["python_decode_wrapper_calls"])
    if actual_decode != (expected_decode, expected_decode):
        raise RuntimeError(
            f"FlashInfer decode API 计数 mismatch: {actual_decode} != "
            f"{(expected_decode, expected_decode)}")
    expected_graph = (int(execution == "graph" and steps > 0),
                      steps if execution == "graph" else 0)
    actual_graph = (receipt["graph_capture_count"], receipt["graph_replays"])
    if actual_graph != expected_graph or receipt["graph_eager_fallbacks"]:
        raise RuntimeError("FlashInfer Graph 执行回执与命令行不一致")


class RequestTwoPointEngine:
    def __init__(self, args, model, tokenizer, eos_ids):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = tuple(eos_ids)
        self.batch_size = getattr(args, "batch_size", 1)
        if args.method != "nfa" and self.batch_size != 1:
            raise ValueError(
                f"{args.method} 两点入口当前仅支持 B1；拒绝用串行 B1 冒充 batch")
        self.request_receipts = []
        self._nfa_cache = None
        self._nfa_geometry = None
        self._nfa_decode_requests = 0
        self._nfa_graph_counters = {"capture": 0, "replay": 0, "fallback": 0}
        self.runtime = {
            "implementation": (
                "NFA GraphReadyPagedRegionCache and production selector"
                if args.method == "nfa" else
                "HF SDPA prefill + FlashInfer page64 paged decode"),
            "execution": args.execution,
            "batch_size": self.batch_size,
            "prefill_mode": (
                "serial B1 prefill per row into one fixed cache arena"
                if self.batch_size > 1 else "single-request prefill"),
            "dtype": str(model.dtype),
            "num_layers": model.config.num_hidden_layers,
            "dense_or_torch_decode_fallback": False,
        }
        if args.method == "nfa":
            selector = nfa_args()
            self.runtime["nfa_selector_config"] = {
                "subspace": selector.nfa_subspace,
                "subspace_cmp_ratio": selector.tia_level2_cmp_ratio,
                "level1_topk": selector.tia_level1_topk,
                "level2_topk": selector.tia_level2_topk,
                "page_size": selector.tia_block_size,
                "near_pages": selector.nfa_near_pages,
                "near_tokens": selector.nfa_near_tokens,
                "near_len": selector.nfa_near_len,
                "enable_layer_skip": selector.nfa_enable_layer_skip,
                "sink_tokens": selector.nfa_sink_tokens,
                "sliding_window_size": selector.nfa_swa_size,
            }
        else:
            self.runtime["flashinfer_version"] = __import__("flashinfer").__version__
            self.runtime["page_size"] = nfa_args().tia_block_size

    def generate(self, ids, max_new_tokens, ignore_eos):
        if not ignore_eos:
            raise ValueError("该性能入口只支持固定长度 ignore_eos=True")
        prompt = torch.tensor(ids, dtype=torch.long, device=self.args.device)
        if self.args.method == "nfa":
            return self._generate_nfa(prompt, max_new_tokens)
        return self._generate_flashinfer(prompt, max_new_tokens)

    def _generate_nfa(self, prompt, max_new_tokens):
        args = self.args
        patch_args = nfa_args()
        batch = self.batch_size
        torch.cuda.synchronize()
        started = time.perf_counter()
        if self._nfa_cache is None:
            # T(1) 与 T(33) 使用同一物理容量和固定地址；只有 decode 步数不同。
            self._nfa_geometry = RegionGeometry(
                sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
                window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
                max_new_tokens=args.max_new_tokens)
            self._nfa_cache = build_frozen_cache(
                self.model, self._nfa_geometry, prompt.numel(), max_requests=batch,
                max_visible_len=args.visible_len)
            self._nfa_prompt_length = prompt.numel()
        cache = self._nfa_cache
        geometry = self._nfa_geometry
        if prompt.numel() != self._nfa_prompt_length:
            raise RuntimeError("固定 Graph arena 不允许跨输入长度复用")
        if cache.states:
            raise RuntimeError("上一 NFA 请求尚未释放，拒绝复用固定 arena")

        request_ids = []
        first_tokens = {}
        runner = None
        request_index = len(self.request_receipts)
        cache_steps_before = cache.nfa_decode_steps
        graph_before = dict(self._nfa_graph_counters)
        pool = getattr(self.model, "_nfa_graph_session_pool", None)
        pool_hits_before = pool.hit_count if pool is not None else 0
        pool_misses_before = pool.miss_count if pool is not None else 0
        pool_warmup_before = pool.warmup_seconds if pool is not None else 0.0
        try:
            # 生产 prefill 是 B1 语义；B>1 逐行写入同一 arena，decode 才真实合批。
            for request_id in range(batch):
                output, audit = prefill_frozen_request(
                    self.model, prompt.unsqueeze(0), GateConfig(), geometry,
                    cache=cache, request_id=request_id)
                if audit["request_id"] != request_id:
                    raise RuntimeError("NFA prefill 返回了非预期 request_id")
                request_ids.append(request_id)
                first = greedy_next_tokens(
                    output.logits, (), generated_counts=0, min_new_tokens=1)
                first_tokens[request_id] = int(first[0])
                cache.transition_request(request_id, "GENERATE")
                del output

            decode_steps = max_new_tokens - 1
            token_history = torch.empty(
                (decode_steps, batch), dtype=torch.long, device=prompt.device)
            if decode_steps:
                scheduler = ContinuousBatchScheduler(
                    cache=cache, max_batch=batch, eos_ids=(), pad_token_id=0)
                for request_id in request_ids:
                    scheduler.add_request(request_id)
                    scheduler.next_tokens[request_id] = first_tokens[request_id]
                runner = session_graph_runner(
                    self.model, cache, scheduler, buckets=(batch,),
                    visible_buckets=(args.visible_len,), device=prompt.device,
                    capture=args.execution == "graph")
                for step in range(decode_steps):
                    _, tokens = runner.step(
                        self.model, cache,
                        generated_counts=[step + 1] * batch,
                        min_new_tokens=1, return_next_tokens=True, stop_on_eos=False)
                    token_history[step].copy_(tokens)
                scheduler.prepare_step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            history = token_history.cpu().tolist()
            generated_rows = [
                [first_tokens[request_id]]
                + [history[step][row] for step in range(decode_steps)]
                for row, request_id in enumerate(request_ids)
            ]
            generated = generated_rows[0] if batch == 1 else generated_rows
            if runner is None:
                current_graph = dict(graph_before)
            else:
                current_graph = {
                    "capture": runner.capture_count,
                    "replay": runner.replay_count,
                    "fallback": runner.eager_fallback_count,
                }
            pool = getattr(self.model, "_nfa_graph_session_pool", None)
            receipt = {
                "request_index": request_index,
                "request_role": (
                    "warmup" if request_index < 2 * args.warmup_requests
                    else "measured"),
                "cache_reused": request_index > 0,
                "batch_size": batch,
                "request_ids": list(request_ids),
                "generated_tokens": sum(map(len, generated_rows)),
                "generated_tokens_per_request": list(map(len, generated_rows)),
                # 一次 runner.step 是整个 batch 的一次 forward，不能乘以 B。
                "nfa_decode_steps": cache.nfa_decode_steps - cache_steps_before,
                "graph_capture_count": (
                    current_graph["capture"] - graph_before.get("capture", 0)),
                "graph_replays": (
                    current_graph["replay"] - graph_before.get("replay", 0)),
                "graph_eager_fallbacks": (
                    current_graph["fallback"] - graph_before.get("fallback", 0)),
                "graph_pool_hits": (
                    pool.hit_count - pool_hits_before if pool is not None else 0),
                "graph_pool_misses": (
                    pool.miss_count - pool_misses_before if pool is not None else 0),
                "graph_prepare_ms": (
                    (pool.warmup_seconds - pool_warmup_before) * 1000.0
                    if pool is not None else 0.0),
            }
            validate_request_receipt(
                "nfa", args.execution, self.model.config.num_hidden_layers,
                max_new_tokens, receipt,
                allow_initial_nfa_capture=self._nfa_decode_requests == 0)
            self._nfa_graph_counters = current_graph
            if decode_steps:
                self._nfa_decode_requests += 1
            if pool is not None:
                self.runtime["graph_session"] = {
                    "hit_count": pool.hit_count,
                    "miss_count": pool.miss_count,
                    "warmup_seconds": pool.warmup_seconds,
                    "fixed_cache_arena": True,
                }
            self.request_receipts.append(receipt)
            return generated, elapsed
        finally:
            if runner is not None:
                pool = getattr(self.model, "_nfa_graph_session_pool", None)
                if pool is not None:
                    pool.release(runner)
                else:
                    runner.scheduler = None
            for request_id in range(batch):
                cache.remove_request(request_id)

    def _generate_flashinfer(self, prompt, max_new_tokens):
        args = self.args
        length = prompt.numel()
        device = prompt.device
        decode_steps = max_new_tokens - 1
        page_size = nfa_args().tia_block_size
        if page_size != 64:
            raise RuntimeError(f"公平比较要求 FlashInfer page64，实际 {page_size}")
        torch.cuda.synchronize()
        started = time.perf_counter()
        runtime = FlashInferQwen3Runtime(
            self.model, 1, length, max_new_tokens, device=device, page_size=page_size)
        bind_flashinfer_runtime(self.model, runtime)
        input_ids = prompt.unsqueeze(0)
        cache_position = torch.arange(length, device=device)
        position_ids = cache_position.unsqueeze(0)
        output = self.model(
            input_ids=input_ids, position_ids=position_ids,
            cache_position=cache_position, past_key_values=runtime.mask_metadata,
            use_cache=True, logits_to_keep=1)
        next_tokens = greedy_next_tokens(
            output.logits, (), generated_counts=0, min_new_tokens=1)
        first_token = int(next_tokens[0])
        del output

        runtime.prepare_decode(args.execution, decode_steps)
        _validate_flashinfer_runtime(runtime, args.execution, page_size)
        input_buf = torch.empty((1, 1), dtype=torch.long, device=device)
        position_buf = torch.empty((1, 1), dtype=torch.long, device=device)
        cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)
        graph = None
        static_logits = None

        def forward():
            return self.model(
                input_ids=input_buf, position_ids=position_buf,
                cache_position=cache_position_buf,
                past_key_values=runtime.mask_metadata, use_cache=True,
                logits_to_keep=1)

        if args.execution == "graph" and decode_steps:
            input_buf.copy_(next_tokens[:, None])
            position_buf.fill_(length)
            cache_position_buf.fill_(length)
            runtime.prepare_decode_step(0)
            forward()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            runtime.prepare_decode_step(0)
            with torch.cuda.graph(graph):
                static_logits = forward()

        token_history = torch.empty(
            (decode_steps, 1), dtype=torch.long, device=device)
        for step in range(decode_steps):
            input_buf.copy_(next_tokens[:, None])
            position_buf.fill_(length + step)
            cache_position_buf.fill_(length + step)
            runtime.prepare_decode_step(step)
            if graph is None:
                logits = forward().logits
            else:
                graph.replay()
                logits = static_logits.logits
            next_tokens = greedy_next_tokens(
                logits, (), generated_counts=step + 1, min_new_tokens=1)
            token_history[step].copy_(next_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        generated = [first_token]
        generated.extend(row[0] for row in token_history.cpu().tolist())
        receipt = {
            "request_index": len(self.request_receipts),
            "generated_tokens": len(generated),
            "prefill_append_calls": runtime.prefill_append_calls,
            "python_decode_append_calls": runtime.decode_append_calls,
            "python_decode_wrapper_calls": runtime.decode_wrapper_calls,
            "graph_capture_count": int(graph is not None),
            "graph_replays": decode_steps if graph is not None else 0,
            "graph_eager_fallbacks": 0,
            "wrapper": type(runtime.wrapper).__name__,
            "page_size": runtime.page_size,
            "fixed_split_size_pages": runtime.fixed_split_size,
        }
        validate_request_receipt(
            "flashinfer", args.execution, self.model.config.num_hidden_layers,
            max_new_tokens, receipt)
        self.request_receipts.append(receipt)
        for module in self.model.modules():
            if hasattr(module, "_flashinfer_runtime"):
                module._flashinfer_runtime = None
        return generated, elapsed


def main():
    args = parse_args()
    if args.max_new_tokens < 2 or args.repeats <= 0 or args.warmup_requests < 0:
        raise ValueError("要求 max-new-tokens>=2、repeats>0、warmup-requests>=0")
    if args.visible_len <= 0:
        raise ValueError("visible-len 必须为正")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids, input_receipt = load_performance_input(args, tokenizer)
    model = _load_model(args, args.max_new_tokens)
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)
    engine = RequestTwoPointEngine(args, model, tokenizer, eos_ids)
    rope_receipt = verify_runtime_rope(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qwen-nfa-flashinfer-long-context-two-point-v2",
        "verification": "pending-independent-verification",
        "status": "in_progress",
        "method": args.method,
        "execution": args.execution,
        "batch_size": args.batch_size,
        "model_path": str(Path(args.model_path).resolve()),
        "source": source_receipt(),
        "context_length": args.context_length,
        "task": args.task,
        "sample_index": args.sample_index,
        "input": input_receipt,
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
                f"[LONG PERF] {args.method}-{args.execution} "
                f"L={args.context_length} B={args.batch_size} repeat={repeat} "
                f"TPOT={timing['tpot_ms']:.3f}ms", flush=True)
    print(f"[LONG PERF DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
