"""Qwen3-8B × LongBench v1 英文16任务 NFA/FlashInfer 真实质量入口。

复用官方数据、prompt、runcap、EOS/min-new/stop-string 与评分输入格式；不复用
固定33 token性能回执。NFA 每请求使用独立 Cache 并在边界释放 Graph session，
FlashInfer 每请求创建独立 page64 runtime。该入口只生成质量预测，不报告 clean TPOT。
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from benchmark.LongBench.context_capability_gate import enforce_context_capability_gate
from benchmark.efficiency.benchmark_long_context_two_point import (
    RequestTwoPointEngine,
    source_receipt as long_context_source_receipt,
)
from benchmark.efficiency.benchmark_nfa_end_to_end import (
    _load_model,
    _validate_flashinfer_runtime,
)
from benchmark.official_baselines.longbench_official import (
    ENGLISH16,
    build_input,
    load_json,
    run_quality,
    sha256_file,
)
from sparse_attn.engine.frozen_runner import (
    build_frozen_cache,
    decode_frozen_request,
    prefill_frozen_request,
)
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches.qwen3_flashinfer_patch import (
    FlashInferQwen3Runtime,
    bind_flashinfer_runtime,
)
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import StopStringMatcher, greedy_next_tokens

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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visible-len", type=int, default=8192)
    return parser.parse_args()


def source_receipt():
    receipt = long_context_source_receipt()
    receipt["quality_harness"] = str(Path(__file__).resolve())
    receipt["quality_harness_sha256"] = sha256_file(__file__)
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "HEAD", "--binary", "--",
         "benchmark/efficiency/benchmark_longbench_quality.py",
         "benchmark/official_baselines/longbench_official.py"])
    receipt["quality_source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    return receipt


def _validate_quality_receipt(method, execution, num_layers, max_new_tokens, receipt,
                              *, nfa_visible_len=8192):
    generated = receipt["generated_tokens"]
    if not 1 <= generated <= max_new_tokens:
        raise RuntimeError("质量生成长度越界")
    if receipt["stop_reason"] not in {"eos", "stop_string", "max_new_tokens"}:
        raise RuntimeError("质量请求缺少真实停止原因")
    steps = generated - 1
    if receipt["logical_decode_steps"] != steps:
        raise RuntimeError("质量请求逻辑 decode 步数不一致")
    if method == "nfa":
        if receipt["nfa_decode_steps"] != steps:
            raise RuntimeError("NFA 质量请求逻辑 decode 步数不一致")
        physical_steps = receipt["submitted_decode_forwards"]
        if physical_steps < steps:
            raise RuntimeError("NFA 质量请求实际提交数小于逻辑 decode 步数")
        if execution == "graph":
            buckets = (512, 1024, 2048, 4096)
            valid = tuple(width for width in buckets if width <= nfa_visible_len)
            captures = len(valid) + int(nfa_visible_len not in valid)
            expected = (captures if physical_steps > 0 else 0, physical_steps, 0)
        else:
            expected = (0, 0, physical_steps)
        actual = (receipt["graph_capture_count"], receipt["graph_replays"],
                  receipt["graph_eager_fallbacks"])
        if actual != expected:
            raise RuntimeError(f"NFA 质量 Graph 计数 mismatch: {actual} != {expected}")
        return

    if receipt["prefill_append_calls"] != num_layers:
        raise RuntimeError("FlashInfer 质量 prefill 未覆盖全部层")
    if steps == 0:
        expected_decode = 0
    elif execution == "graph":
        expected_decode = 2 * num_layers
    else:
        expected_decode = steps * num_layers
    actual_decode = (receipt["python_decode_append_calls"],
                     receipt["python_decode_wrapper_calls"])
    if actual_decode != (expected_decode, expected_decode):
        raise RuntimeError(
            f"FlashInfer 质量 API 计数 mismatch: {actual_decode} != "
            f"{(expected_decode, expected_decode)}")
    expected_graph = (int(execution == "graph" and steps > 0),
                      steps if execution == "graph" else 0)
    actual_graph = (receipt["graph_capture_count"], receipt["graph_replays"])
    if actual_graph != expected_graph or receipt["graph_eager_fallbacks"]:
        raise RuntimeError("FlashInfer 质量 Graph 回执与执行模式不一致")


class RequestQualityEngine(RequestTwoPointEngine):
    """逐请求独立资源的真实停止生成；返回形状与 OfficialEngine 一致。"""

    def generate(self, ids, max_new_tokens, ignore_eos, stop_token_ids=None,
                 min_new_tokens=0, stop_strings=None):
        if max_new_tokens < 1:
            raise ValueError("质量入口要求 max_new_tokens>=1")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens 必须位于 [0, max_new_tokens]")
        if stop_strings and ignore_eos:
            raise ValueError("固定长度性能语义不得混入质量 stop-string")
        eos_ids = ([] if ignore_eos else list(self.eos_ids)) + list(stop_token_ids or ())
        eos_ids = tuple(dict.fromkeys(
            int(token) for token in eos_ids
            if isinstance(token, int) and not isinstance(token, bool) and token >= 0))
        prompt = torch.tensor(ids, dtype=torch.long, device=self.args.device)
        if self.args.method == "nfa":
            return self._generate_nfa_quality(
                prompt, max_new_tokens, eos_ids, min_new_tokens, stop_strings)
        return self._generate_flashinfer_quality(
            prompt, max_new_tokens, eos_ids, min_new_tokens, stop_strings)

    def _generate_nfa_quality(self, prompt, max_new_tokens, eos_ids,
                              min_new_tokens, stop_strings):
        args = self.args
        patch_args = nfa_args()
        geometry = RegionGeometry(
            sink=patch_args.nfa_sink_tokens, near=patch_args.nfa_near_len,
            window=patch_args.nfa_swa_size, page=patch_args.tia_block_size,
            max_new_tokens=max_new_tokens)
        cache = build_frozen_cache(
            self.model, geometry, prompt.numel(), max_requests=1,
            max_visible_len=args.visible_len)
        request_id = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            output, audit = prefill_frozen_request(
                self.model, prompt.unsqueeze(0), GateConfig(), geometry, cache=cache)
            request_id = audit["request_id"]
            receipt = {}
            generated, _ = decode_frozen_request(
                self.model, output, max_new_tokens=max_new_tokens, eos_ids=eos_ids,
                capture_graph=args.execution == "graph", min_new_tokens=min_new_tokens,
                stop_strings=stop_strings, tokenizer=self.tokenizer,
                prompt_ids=prompt.tolist(), receipt=receipt)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            _validate_quality_receipt(
                "nfa", args.execution, self.model.config.num_hidden_layers,
                max_new_tokens, receipt, nfa_visible_len=args.visible_len)
            self.request_receipts.append(receipt)
            return generated, elapsed
        finally:
            # 每个变长质量请求独占 cache/graph；禁止池强引用上一请求 KV arena。
            self.model._nfa_graph_session_pool = None
            if request_id is not None:
                cache.remove_request(request_id)
            torch.cuda.empty_cache()

    def _generate_flashinfer_quality(self, prompt, max_new_tokens, eos_ids,
                                     min_new_tokens, stop_strings):
        args = self.args
        length = prompt.numel()
        device = prompt.device
        decode_capacity = max_new_tokens - 1
        page_size = nfa_args().tia_block_size
        if page_size != 64:
            raise RuntimeError(f"公平比较要求 FlashInfer page64，实际 {page_size}")
        runtime = FlashInferQwen3Runtime(
            self.model, 1, length, max_new_tokens, device=device, page_size=page_size)
        bind_flashinfer_runtime(self.model, runtime)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            input_ids = prompt.unsqueeze(0)
            cache_position = torch.arange(length, device=device)
            output = self.model(
                input_ids=input_ids, position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position, past_key_values=runtime.mask_metadata,
                use_cache=True, logits_to_keep=1)
            next_tokens = greedy_next_tokens(
                output.logits, eos_ids, generated_counts=0,
                min_new_tokens=min_new_tokens)
            del output

            generated = []
            matcher = (StopStringMatcher(self.tokenizer, stop_strings, prompt.tolist())
                       if stop_strings else None)

            def observe(token):
                generated.append(int(token))
                string_hit = (matcher.append_and_match(token)
                              if matcher is not None else False)
                if len(generated) >= min_new_tokens and generated[-1] in eos_ids:
                    return "eos"
                if string_hit:
                    return "stop_string"
                if len(generated) >= max_new_tokens:
                    return "max_new_tokens"
                return None

            stop_reason = observe(next_tokens[0])
            graph = None
            static_logits = None
            replay_count = 0
            if stop_reason is None:
                runtime.prepare_decode(args.execution, decode_capacity)
                _validate_flashinfer_runtime(runtime, args.execution, page_size)
                input_buf = torch.empty((1, 1), dtype=torch.long, device=device)
                position_buf = torch.empty((1, 1), dtype=torch.long, device=device)
                cache_position_buf = torch.empty((1,), dtype=torch.long, device=device)

                def forward():
                    return self.model(
                        input_ids=input_buf, position_ids=position_buf,
                        cache_position=cache_position_buf,
                        past_key_values=runtime.mask_metadata, use_cache=True,
                        logits_to_keep=1)

                if args.execution == "graph":
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

                for step in range(decode_capacity):
                    input_buf.copy_(next_tokens[:, None])
                    position_buf.fill_(length + step)
                    cache_position_buf.fill_(length + step)
                    runtime.prepare_decode_step(step)
                    if graph is None:
                        logits = forward().logits
                    else:
                        graph.replay()
                        replay_count += 1
                        logits = static_logits.logits
                    next_tokens = greedy_next_tokens(
                        logits, eos_ids, generated_counts=step + 1,
                        min_new_tokens=min_new_tokens)
                    stop_reason = observe(next_tokens[0])
                    if stop_reason is not None:
                        break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if stop_reason is None:
                raise RuntimeError("FlashInfer 质量生成未产生停止原因")
            steps = len(generated) - 1
            receipt = {
                "generated_tokens": len(generated),
                "stop_reason": stop_reason,
                "logical_decode_steps": steps,
                "submitted_decode_forwards": steps,
                "prefill_append_calls": runtime.prefill_append_calls,
                "python_decode_append_calls": runtime.decode_append_calls,
                "python_decode_wrapper_calls": runtime.decode_wrapper_calls,
                "graph_capture_count": int(graph is not None),
                "graph_replays": replay_count,
                "graph_eager_fallbacks": 0,
                "wrapper": type(runtime.wrapper).__name__ if runtime.wrapper else None,
                "page_size": runtime.page_size,
                "fixed_split_size_pages": runtime.fixed_split_size,
            }
            _validate_quality_receipt(
                "flashinfer", args.execution, self.model.config.num_hidden_layers,
                max_new_tokens, receipt)
            self.request_receipts.append(receipt)
            return generated, elapsed
        finally:
            for module in self.model.modules():
                if hasattr(module, "_flashinfer_runtime"):
                    module._flashinfer_runtime = None
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    if len(set(args.tasks)) != len(args.tasks) or any(t not in ENGLISH16 for t in args.tasks):
        raise ValueError("tasks 必须是唯一的 LongBench 英文16任务")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("要求 0<=shard-index<num-shards")
    if args.model_name != "Qwen3-8B" or args.visible_len <= 0:
        raise ValueError("当前入口仅验收 Qwen3-8B 且 visible-len 必须为正")
    prompt_cfg = load_json(Path(args.config_path) / "dataset2prompt.json")
    maxgen_cfg = load_json(Path(args.config_path) / "dataset2maxlen.json")
    model_len_cfg = load_json(Path(args.config_path) / "model2maxlen.json")
    max_length = int(model_len_cfg[args.model_name])
    if max_length != 31500:
        raise ValueError("LongBench 对齐口径要求 max_length=31500")
    cap_gate_info = enforce_context_capability_gate(
        args.model_name, args.model_path, max_length, args.config_path)
    args.quality_label = f"{args.method}_{args.execution}"
    args.quality_implementation = args.quality_label
    args.quality_producer_file = __file__

    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_args = SimpleNamespace(
        method=args.method, model_path=args.model_path, context_length=max_length)
    model = _load_model(load_args, max(maxgen_cfg[task] for task in args.tasks))
    from benchmark.RULER.pred_ruler import _ruler_eos_ids
    eos_ids = _ruler_eos_ids(model, tokenizer)
    engine = RequestQualityEngine(args, model, tokenizer, eos_ids)
    engine.runtime["source"] = source_receipt()
    engine.runtime["quality_protocol"] = {
        "fixed_length_performance_receipts_reused": False,
        "real_eos_min_new_stop_token_stop_string": True,
        "per_request_independent_cache": True,
        "graph_session_released_at_request_boundary": args.method == "nfa",
    }
    with torch.inference_mode():
        run_quality(args, engine, prompt_cfg, maxgen_cfg, max_length, cap_gate_info)


if __name__ == "__main__":
    main()
