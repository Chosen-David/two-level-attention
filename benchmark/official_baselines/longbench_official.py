"""LongBench v1 英文16任务与 v2 的纯官方 Quest/JSSA 独立入口。

Quest 使用本地官方克隆的 CUDA ``_kernels``；JSSA 使用指定提交加官方补丁的
vLLM/Triton backend。本文件不导入 NFA、``sparse_attn`` 或旧 vendor runner。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from benchmark.LongBench.context_capability_gate import (
    enforce_context_capability_gate,
)
from benchmark.LongBench.lbv2_choice import (
    LBV2_PARSER_VERSION,
    extract_choice_official,
)
from benchmark.LongBench.run_cap_manifest import (
    build_lbv1_input_identity,
    build_run_cap_core,
    compute_result_binding,
    finalize_run_cap_manifest,
    gate_run_cap_identity,
    input_ids_sha256 as runcap_input_ids_sha256,
    resolve_producer_identity,
    runcap_gate_lock,
    write_run_cap_manifest,
)

REPO = Path(__file__).resolve().parents[2]
ENGLISH16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
RAW_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
QUEST_COMMIT = "01c1623bf9395009520874e989e29f683203b357"
JSSA_COMMIT = "34ce5abd82453b16ca6d5f16a1a8ae50fac5b920"
VLLM_COMMIT = "39910f2b25aacc09f5e7f166cdf0030b19f8b9e8"
SUPPORTED_QWEN3_MODELS = {"Qwen3-8B", "Qwen3-32B"}


def validate_model_contract(model_name, max_length):
    if model_name not in SUPPORTED_QWEN3_MODELS or max_length != 31500:
        raise ValueError(
            "当前独立入口仅验收 Qwen3-8B/Qwen3-32B 的 31500 对齐口径"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("quest", "jssa"), required=True)
    parser.add_argument("--mode", choices=("quality", "perf"), default="quality")
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
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--quest-max-seq-len", type=int, default=40960)
    parser.add_argument("--jssa-rank", type=int, default=32)
    parser.add_argument("--jssa-basis-path", default=str(REPO / "bases/qwen3_8b.pt"))
    parser.add_argument("--jssa-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=33,
                        help="perf 固定生成长度；quality 默认读 dataset2maxlen")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(data_root, task):
    if task == "lbv2":
        path = Path(data_root) / "lbv2.json"
        if not path.is_file():
            path = Path(data_root) / "data.json"
        if not path.is_file():
            raise FileNotFoundError(f"本地 LongBench v2 数据不存在: {path}")
        raw = path.read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"LongBench v2 必须是 JSON 对象数组: {path}")
        sample_ids = [str(row.get("_id", "")) for row in rows]
        if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(rows):
            raise ValueError(f"LongBench v2 官方 _id 必须非空且唯一: {path}")
    else:
        path = Path(data_root) / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"本地 LongBench 数据不存在: {path}")
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    return rows, hashlib.sha256(raw).hexdigest()


def quality_sample_id(task, source_idx, row):
    return str(row["_id"]) if task == "lbv2" else f"{task}:{source_idx}"


def sharding_sample_id(task, source_idx, source_fingerprint, row):
    if task == "lbv2":
        return quality_sample_id(task, source_idx, row)
    return f"{task}:{source_fingerprint}:{source_idx}"


def quality_row_metadata(task, row):
    if task == "lbv2":
        return {
            "answers": [row["answer"]],
            "all_classes": ["A", "B", "C", "D"],
            "length": row["length"],
        }
    return {
        "answers": row["answers"],
        "all_classes": row["all_classes"],
        "length": row["length"],
    }


def quality_source_manifest_path(prediction_path):
    return Path(str(prediction_path) + ".source_manifest.json")


def build_quality_source_manifest(task, selected):
    if task != "lbv2":
        return None
    ids = [quality_sample_id(task, source_idx, row) for source_idx, row in selected]
    answers_sha = {}
    for source_idx, row in selected:
        answers = quality_row_metadata(task, row)["answers"]
        answers_sha[quality_sample_id(task, source_idx, row)] = hashlib.sha256(
            json.dumps(answers, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return {task: {"ids": ids, "answers_sha": answers_sha}}


def qwen3_chat_prompt(prompt):
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_input(row, task, prompt_format, max_length, tokenizer):
    prompt = prompt_format.format(**row)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    truncated = len(raw_ids) > max_length
    if truncated:
        half = max_length // 2
        prompt = (
            tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
            + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
        )
    if task not in RAW_TASKS:
        prompt = qwen3_chat_prompt(prompt)
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    if len(ids) > max_length + 64:
        raise RuntimeError(f"chat 后输入异常超长: {len(ids)} > {max_length}+64")
    return ids, {
        "input_tokens": len(ids),
        "input_ids_sha256": sha256_ids(ids),
        "truncated": truncated,
    }


def quality_generation_options(task, tokenizer):
    """LongBench 质量协议：首 token 屏蔽 EOS；samsum 额外在换行停止。"""
    stop_token_ids = []
    options = {"stop_token_ids": stop_token_ids, "min_new_tokens": 1}
    if task == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if not newline_ids:
            raise RuntimeError("tokenizer 无法产生 samsum 换行停止 token")
        stop_token_ids.append(newline_ids[-1])
        # 换行常与前后字符合并为同一 BPE token（如 ".\n"），仅匹配独立
        # token 198 会漏停；字符串停止同时覆盖合并 token 与独立 token。
        options["stop_strings"] = ["\n"]
    return options


def line_count(path):
    if not path.exists():
        return 0
    raw = path.read_bytes()
    return raw.count(b"\n") + int(bool(raw) and not raw.endswith(b"\n"))


def git_identity(path, expected_commit):
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"{path} HEAD={actual}，预期 {expected_commit}")
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary"])
    return {
        "path": str(path),
        "commit": actual,
        "uncommitted_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def validate_official_batch_size(args):
    """官方 Quest/JSSA 当前均为真实 B1 路径，禁止用串行请求冒充 batch。"""
    batch_size = getattr(args, "batch_size", 1)
    if batch_size != 1:
        raise ValueError(
            f"官方 {args.method} 当前仅支持 batch_size=1；拒绝用串行 B1 冒充 batch"
        )
    return batch_size


class OfficialEngine:
    def __init__(self, args, max_model_len):
        self.args = args
        self.batch_size = validate_official_batch_size(args)
        self.runtime = {"batch_size": self.batch_size}
        if args.method == "quest":
            self._init_quest(max_model_len)
        else:
            self._init_jssa(max_model_len)

    def _init_quest(self, max_model_len):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from benchmark.official_baselines.qwen3_quest_kernel import (
            enable_quest,
            quest_kernels,
        )

        args = self.args
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # 所有 36 层随后都由官方 Quest prefill/decode 接管。这里只让 HF 的
        # mask interface 走 SDPA 的 is-causal 快路，避免 eager 在 256K/512K
        # 先物化 [1,1,L,L] 的 64/256GiB 布尔 mask；不执行 SDPA attention。
        model_config = getattr(args, "model_config", None)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=model_config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to(args.device).eval()
        rope_mode = getattr(args, "quest_rope_mode", "native")
        n_patched = enable_quest(
            self.model,
            token_budget=args.budget,
            chunk_size=args.quest_page_size,
            page_size=args.quest_page_size,
            max_seq_len=max(args.quest_max_seq_len, max_model_len),
            rope_mode=rope_mode,
        )
        extension = Path(quest_kernels.__file__).resolve()
        self.runtime = {
            "batch_size": self.batch_size,
            "implementation": "official Quest CUDA _kernels and paged runtime",
            "patched_layers": n_patched,
            "budget": args.budget,
            "page_size": args.quest_page_size,
            "page_budget": args.budget // args.quest_page_size,
            "query_and_metadata_heads": int(self.model.config.num_attention_heads),
            "physical_kv_heads": int(self.model.config.num_key_value_heads),
            "rope_mode": rope_mode,
            "rope_execution": (
                "official quest.apply_rope_in_place"
                if rope_mode == "native"
                else "HF position_embeddings exactly once before official Quest KV/attention"
            ),
            "hf_length_cache_layers": 1,
            "prefill_tokenwise_chunk": 65536,
            "quest": git_identity(REPO / "third-party/Quest", QUEST_COMMIT),
            "kernel_extension": str(extension),
            "kernel_extension_sha256": sha256_file(extension),
            "dense_or_torch_fallback": False,
        }

    def _init_jssa(self, max_model_len):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        import torch
        from transformers import AutoTokenizer
        import vllm
        from vllm import LLM, SamplingParams

        args = self.args
        vllm_root = REPO / "third-party/JSSA/vllm-jssa"
        vllm_file = Path(vllm.__file__).resolve()
        if not vllm_file.is_relative_to(vllm_root.resolve()):
            raise RuntimeError(f"vLLM 并非来自 JSSA 官方补丁树: {vllm_file}")
        backend = vllm_root / "vllm/v1/attention/backends/jssa_attn.py"
        if not backend.is_file():
            raise RuntimeError(f"JSSA 官方 backend 缺失: {backend}")
        if not Path(args.jssa_basis_path).is_file():
            raise FileNotFoundError(args.jssa_basis_path)

        import vllm.v1.attention.backends.jssa_attn as jssa_backend
        import vllm.v1.attention.ops.triton_jssa_indexed_attend as indexed_ops
        from vllm.v1.attention.ops.triton_jssa_score_topk import _radix_topk_available

        if not _radix_topk_available():
            raise RuntimeError("JSSA 官方 radix Top-K binding 缺失，禁止退回 torch.topk")

        # 只包装官方函数入口做验收计数，不替换任何算子实现。CUDA Graph 下计数
        # 表示 capture 时的真实分支证据；eager 下还表示逐次 Python 调用。
        self._jssa_calls = {
            "radix_score_topk": 0,
            "fp8_projected_store": 0,
            "indexed_attend": 0,
            "gather_fallback": 0,
            "bf16_projected_store_fallback": 0,
            "unprojected_store_fallback": 0,
        }

        def counted_score_topk(*call_args, **call_kwargs):
            if not _radix_topk_available():
                raise RuntimeError("JSSA radix Top-K 在运行中失效")
            self._jssa_calls["radix_score_topk"] += 1
            return official_score_topk(*call_args, **call_kwargs)

        def counted_fp8_store(*call_args, **call_kwargs):
            self._jssa_calls["fp8_projected_store"] += 1
            return official_fp8_store(*call_args, **call_kwargs)

        def counted_indexed_attend(*call_args, **call_kwargs):
            self._jssa_calls["indexed_attend"] += 1
            return official_indexed_attend(*call_args, **call_kwargs)

        def forbidden_fallback(name):
            def reject(*call_args, **call_kwargs):
                self._jssa_calls[name] += 1
                raise RuntimeError(f"JSSA 触发禁止的低性能/非 FP8 路径: {name}")
            return reject

        official_score_topk = jssa_backend.jssa_score_topk
        official_fp8_store = jssa_backend.jssa_project_and_store_contig_fp8
        official_indexed_attend = indexed_ops.jssa_indexed_attend
        jssa_backend.jssa_score_topk = counted_score_topk
        jssa_backend.jssa_project_and_store_contig_fp8 = counted_fp8_store
        jssa_backend.jssa_gather = forbidden_fallback("gather_fallback")
        jssa_backend.jssa_project_and_store = forbidden_fallback(
            "bf16_projected_store_fallback"
        )
        jssa_backend.jssa_store = forbidden_fallback("unprojected_store_fallback")
        indexed_ops.jssa_indexed_attend = counted_indexed_attend

        self.torch = torch
        self.SamplingParams = SamplingParams
        from vllm.inputs import TokensPrompt
        self.TokensPrompt = TokensPrompt
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = LLM(
            model=args.model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.jssa_gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_num_seqs=1,
            trust_remote_code=True,
            hf_overrides=getattr(args, "hf_overrides", None),
            kv_cache_dtype="jssa",
            attention_config={
                "backend": "JSSA",
                "jssa_basis_path": args.jssa_basis_path,
                "jssa_budget": args.budget,
                "jssa_rank": args.jssa_rank,
                "jssa_fp8_projk": True,
                "jssa_indexed_attend": True,
            },
        )
        self.runtime = {
            "batch_size": self.batch_size,
            "implementation": "official JSSA patched-vLLM Triton FP8/indexed-attend",
            "budget": args.budget,
            "rank": args.jssa_rank,
            "fp8_projk": True,
            "indexed_attend": True,
            "enforce_eager": args.enforce_eager,
            "jssa": git_identity(REPO / "third-party/JSSA", JSSA_COMMIT),
            "patched_vllm": git_identity(vllm_root, VLLM_COMMIT),
            "backend_file": str(backend),
            "backend_file_sha256": sha256_file(backend),
            "basis_sha256": sha256_file(args.jssa_basis_path),
            "radix_topk_binding": "torch.ops._C.top_k_per_row_decode",
            "fallback_policy": "fail_closed",
            "jssa_call_counts": self._jssa_calls,
            "call_counts_scope": "eager calls plus CUDA Graph capture branches",
            "torch_or_hf_fallback": False,
        }

    def generate(
        self, ids, max_new_tokens, ignore_eos, stop_token_ids=None,
        min_new_tokens=0, stop_strings=None,
    ):
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens 必须位于 [0, max_new_tokens]")
        if self.args.method == "quest":
            return self._generate_quest(
                ids, max_new_tokens, ignore_eos, stop_token_ids, min_new_tokens,
                stop_strings,
            )
        return self._generate_jssa(
            ids, max_new_tokens, ignore_eos, stop_token_ids, min_new_tokens,
            stop_strings,
        )

    def _generate_quest(
        self, ids, max_new_tokens, ignore_eos, stop_token_ids=None,
        min_new_tokens=0, stop_strings=None,
    ):
        torch = self.torch
        inputs = torch.tensor([ids], dtype=torch.long, device=self.args.device)
        configured = getattr(self.model.generation_config, "eos_token_id", None)
        if configured is None:
            configured = ()
        elif not isinstance(configured, (list, tuple)):
            configured = (configured,)
        eos_ids = ([] if ignore_eos else [
            *configured, self.tokenizer.eos_token_id,
        ]) + list(stop_token_ids or ())
        eos_ids = list(dict.fromkeys(
            token for token in eos_ids
            if isinstance(token, int) and not isinstance(token, bool) and token >= 0
        ))
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=max_new_tokens,
                    min_length=inputs.shape[-1] + min_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=eos_ids or None,
                    stop_strings=stop_strings or None,
                    tokenizer=self.tokenizer if stop_strings else None,
                )[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        finally:
            self.model.reset_quest_controller()
        generated = output[len(ids):].tolist()
        return generated, elapsed

    def _generate_jssa(
        self, ids, max_new_tokens, ignore_eos, stop_token_ids=None,
        min_new_tokens=0, stop_strings=None,
    ):
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            min_tokens=min_new_tokens,
            ignore_eos=ignore_eos,
            stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
            stop=list(stop_strings) if stop_strings else None,
            include_stop_str_in_output=bool(stop_strings),
        )
        prompt = self.TokensPrompt(prompt_token_ids=ids)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = self.model.generate([prompt], params, use_tqdm=False)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("JSSA 单请求生成返回了异常输出数")
        if self._jssa_calls["fp8_projected_store"] == 0:
            raise RuntimeError("JSSA 未执行官方 FP8 projected-key store")
        if max_new_tokens > 1 and (
            self._jssa_calls["radix_score_topk"] == 0
            or self._jssa_calls["indexed_attend"] == 0
        ):
            raise RuntimeError("JSSA 未捕获/调用官方 radix Top-K 与 indexed-attend")
        return list(outputs[0].outputs[0].token_ids), elapsed


def _generated_rows(value, expected_length, label):
    """统一核验 B1 扁平输出与 B>1 batch-major 输出，拒绝混合/参差行。"""
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} 必须是非空 token 列表")
    batched = isinstance(value[0], list)
    if batched and not all(isinstance(row, list) for row in value):
        raise RuntimeError(f"{label} 不允许混合扁平和 batch-major 行")
    rows = value if batched else [value]
    lengths = [len(row) for row in rows]
    if any(length != expected_length for length in lengths):
        raise RuntimeError(
            f"固定长度生成失败: {label} 每行应为 {expected_length}，实际 {lengths}")
    if any(not all(isinstance(token, int) for token in row) for row in rows):
        raise RuntimeError(f"{label} 含非整数 token")
    return rows, batched


def timed_pair(engine, ids, max_new_tokens, require_positive=True):
    first_ids, ttft = engine.generate(ids, 1, ignore_eos=True)
    generated, request_wall = engine.generate(ids, max_new_tokens, ignore_eos=True)
    first_rows, first_batched = _generated_rows(first_ids, 1, "T(1)")
    generated_rows, generated_batched = _generated_rows(
        generated, max_new_tokens, f"T({max_new_tokens})")
    if first_batched != generated_batched or len(first_rows) != len(generated_rows):
        raise RuntimeError("T(1) 与 T(N) 的 batch 布局或行数不一致")
    decode_wall = request_wall - ttft
    if require_positive and decode_wall <= 0:
        raise RuntimeError(
            f"两点法 decode 时间非正: {decode_wall}; "
            f"T(1)={ttft}, T({max_new_tokens})={request_wall}"
        )
    eos_rows = [
        [i for i, token in enumerate(row)
         if token == engine.tokenizer.eos_token_id]
        for row in generated_rows
    ]
    return generated, {
        "schema": "longbench-official-request-two-point-v1",
        "ttft_ms": ttft * 1000.0,
        "decode_after_first_wall_ms": decode_wall * 1000.0,
        # 每个 batched decode step 的 host-wall；不按 B 归一化，也不冒充吞吐。
        "tpot_ms": decode_wall * 1000.0 / (max_new_tokens - 1),
        "request_wall_ms": request_wall * 1000.0,
        "batch_size": len(generated_rows),
        "output_layout": "batch-major" if generated_batched else "flat-b1",
        "actual_generated_tokens": sum(map(len, generated_rows)),
        "actual_generated_tokens_per_request": list(map(len, generated_rows)),
        "generated_token_ids": generated,
        "eos_positions": eos_rows if generated_batched else eos_rows[0],
        "eos_early_stop": False,
        "decode_protocol": "T(1+D)-T(1), fixed 1+D tokens per request; EOS ignored",
        "timing_metric": "complete request host wall including official runtime scheduling",
    }


def median(values):
    return statistics.median(values) if values else None


def perf_summary(cells):
    metrics = ("ttft_ms", "tpot_ms", "request_wall_ms")
    per_task = {}
    for task in sorted({cell["task"] for cell in cells}):
        selected = [cell for cell in cells if cell["task"] == task]
        per_task[task] = {
            metric: median([c["request_timing"][metric] for c in selected])
            for metric in metrics
        }
        per_task[task]["requests"] = len(selected)
    return {
        "per_task": per_task,
        "macro_task_mean": {
            metric: statistics.mean([v[metric] for v in per_task.values()])
            if per_task else None
            for metric in metrics
        },
        "all_request_median": {
            metric: median([c["request_timing"][metric] for c in cells])
            for metric in metrics
        },
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def quality_manifest_path(output_dir, shard_index, num_shards):
    return Path(output_dir) / f"quality_manifest_s{shard_index}of{num_shards}.json"


def quality_receipts_path(prediction_path):
    return Path(prediction_path).with_suffix(".receipts.json")


def _restore_quality_receipts(path, task, prepared):
    if not path.is_file():
        raise RuntimeError(f"完整预测缺少逐请求回执，拒绝 SKIP: {path}")
    receipts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipts, list) or len(receipts) != len(prepared):
        raise RuntimeError(f"逐请求回执人口与完整预测不一致: {path}")
    for receipt, (source_idx, _, _, identity) in zip(receipts, prepared):
        if (
            not isinstance(receipt, dict)
            or receipt.get("task") != task
            or receipt.get("source_index") != source_idx
            or receipt.get("input") != identity
        ):
            raise RuntimeError(f"逐请求回执与当前输入身份不一致: {path} {source_idx}")
    return receipts


def run_quality(
    args, engine, prompt_cfg, maxgen_cfg, max_length, cap_gate_info=None
):
    out_dir = Path(args.output_dir) / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    implementation = getattr(
        args, "quality_implementation", args.method + "_official")
    result_label = getattr(args, "quality_label", None)
    if result_label is None:
        result_label = (
            f"quest_official_b{args.budget}_p{args.quest_page_size}"
            if args.method == "quest"
            else f"jssa_official_vllm_b{args.budget}_r{args.jssa_rank}"
        )
    producer_file = getattr(args, "quality_producer_file", __file__)
    track_receipts = hasattr(engine, "request_receipts")
    manifest_receipts = []
    manifest = {
        "schema": "longbench-official-quality-manifest-v1",
        "verification": "pending-independent-verification",
        "method": args.method,
        "model_path": str(Path(args.model_path).resolve()),
        "max_length": max_length,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "runtime": engine.runtime,
        "tasks": {},
    }
    for task in args.tasks:
        rows, data_sha = load_rows(args.data_root, task)
        selected = [
            (source_idx, row)
            for source_idx, row in enumerate(rows)
            if source_idx % args.num_shards == args.shard_index
            and (args.sample_indices is None or source_idx in args.sample_indices)
        ]
        prefix = task.split("-")[0]
        out_path = out_dir / (
            f"{prefix}-{result_label}_s{args.shard_index}of{args.num_shards}.jsonl"
        )

        source_identity = build_lbv1_input_identity(rows, prompt_cfg[task])
        # 标准 sample_sharding 只能描述完整的 source_order[rank::world] 人口。
        # 额外 sample_indices 过滤时，把过滤后的有序行作为本次有效人口，同时
        # 用扩展字段保留完整源身份和源行号，绝不把空/子集分片伪装成完整分片。
        input_identity = (
            source_identity if args.sample_indices is None
            else build_lbv1_input_identity(
                [row for _, row in selected], prompt_cfg[task]
            )
        )
        run_timestamp = f"s{args.shard_index}of{args.num_shards}"
        if args.sample_indices is not None:
            selection_sha = hashlib.sha256(json.dumps(
                args.sample_indices, separators=(",", ":")
            ).encode("utf-8")).hexdigest()[:10]
            run_timestamp += f"-filter{selection_sha}"
        runcap_core = build_run_cap_core(
            model_name=getattr(args, "model_name", "Qwen3-8B"),
            model_path=args.model_path,
            method=implementation,
            dataset=task,
            dataset_fingerprint=input_identity["source_fingerprint"],
            run_timestamp=run_timestamp,
            effective_cap=max_length,
            cap_gate_info=cap_gate_info,
            max_new_tokens=maxgen_cfg[task],
            producer=resolve_producer_identity(producer_file),
            input_identity=input_identity,
        )
        if task == "lbv2":
            # runcap 的 sample_sharding.source_sha256 必须继续绑定规范化源身份；
            # 原始 JSON 字节哈希单独记录，二者不可互相替代。
            runcap_core["source_file_sha256"] = data_sha
            runcap_core["choice_parser"] = LBV2_PARSER_VERSION
        if args.sample_indices is None:
            selected_ids = [
                sharding_sample_id(
                    task, source_idx, source_identity["source_fingerprint"],
                    rows[source_idx],
                )
                for source_idx in range(
                    args.shard_index, len(rows), args.num_shards
                )
            ]
            runcap_core["sample_sharding"] = {
                "algorithm": "source_order[rank::world_size]",
                "rank": args.shard_index,
                "world_size": args.num_shards,
                "source_samples": len(rows),
                "selected_samples": len(selected),
                "source_sha256": source_identity["source_sha256"],
                "selected_ids_sha256": hashlib.sha256(
                    "\n".join(selected_ids).encode("utf-8")
                ).hexdigest(),
            }
        else:
            runcap_core["source_input_identity"] = source_identity
            runcap_core["sample_selection"] = {
                "algorithm": "explicit_source_indices_after_source_order_shard",
                "source_indices": [source_idx for source_idx, _ in selected],
                "source_samples": len(rows),
                "source_sha256": source_identity["source_sha256"],
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }

        prepared = []
        for source_idx, row in selected:
            ids, identity = build_input(
                row, task, prompt_cfg[task], max_length, engine.tokenizer
            )
            prepared.append((source_idx, row, ids, identity))
        input_tokens = [len(ids) for _, _, ids, _ in prepared]
        truncated = [identity["truncated"] for _, _, _, identity in prepared]
        input_hashes = [
            runcap_input_ids_sha256(ids) for _, _, ids, _ in prepared
        ]

        with runcap_gate_lock(str(out_path)):
            runcap_sidecar = gate_run_cap_identity(str(out_path), runcap_core)
            write_run_cap_manifest(
                runcap_sidecar,
                finalize_run_cap_manifest(
                    runcap_core, status="generating", n_samples=len(prepared),
                    input_tokens=input_tokens, truncated_flags=truncated,
                    result_file=out_path.name, result_sha256=None,
                    result_lines=None, input_ids_sha256=input_hashes,
                ),
            )
            receipt_path = quality_receipts_path(out_path)
            task_receipts = []
            if not prepared:
                out_path.write_text("", encoding="utf-8")
                if track_receipts:
                    atomic_json(receipt_path, [])
            elif line_count(out_path) == len(prepared):
                records = [
                    json.loads(line)
                    for line in out_path.read_text(encoding="utf-8").splitlines()
                ]
                for record, (source_idx, row, _, identity) in zip(records, prepared):
                    metadata = quality_row_metadata(task, row)
                    if (
                        record.get("_id") != quality_sample_id(task, source_idx, row)
                        or record.get("input") != identity
                        or record.get("answers") != metadata["answers"]
                        or record.get("implementation") != implementation
                        or (task == "lbv2" and (
                            record.get("all_classes") != metadata["all_classes"]
                            or record.get("length") != metadata["length"]
                        ))
                    ):
                        raise RuntimeError(
                            f"既有预测与当前输入身份不一致: {out_path} {source_idx}"
                        )
                    if task == "lbv2" and record.get("pred_choice") != extract_choice_official(
                            record.get("pred")):
                        raise RuntimeError(
                            f"既有 LBV2 预测的 strict choice 不一致: {out_path} {source_idx}"
                        )
                if track_receipts:
                    task_receipts = _restore_quality_receipts(
                        receipt_path, task, prepared)
                    engine.request_receipts.extend(task_receipts)
                    manifest_receipts.extend(task_receipts)
                print(f"[SKIP] {task}: {len(prepared)} 条已完成", flush=True)
            else:
                generation_options = quality_generation_options(task, engine.tokenizer)
                if track_receipts:
                    # 部分预测总是整任务重跑，旧回执不得混入新一轮。
                    atomic_json(receipt_path, [])
                with out_path.open("w", encoding="utf-8") as handle:
                    for source_idx, row, ids, identity in prepared:
                        receipts_before = (
                            len(engine.request_receipts) if track_receipts else 0)
                        generated, _ = engine.generate(
                            ids, maxgen_cfg[task], ignore_eos=False,
                            **generation_options,
                        )
                        if track_receipts and len(
                                engine.request_receipts) != receipts_before + 1:
                            raise RuntimeError(
                                f"质量生成未恰好产生一个逐请求回执: {task} {source_idx}")
                        pred = engine.tokenizer.decode(
                            generated, skip_special_tokens=True
                        )
                        record = {
                            "pred": pred,
                            **quality_row_metadata(task, row),
                            "_id": quality_sample_id(task, source_idx, row),
                            "input": identity,
                            "implementation": implementation,
                        }
                        if task == "lbv2":
                            record["pred_choice"] = extract_choice_official(pred)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if track_receipts:
                            receipt = engine.request_receipts[-1]
                            receipt.update(
                                task=task, source_index=source_idx, input=identity)
                            task_receipts.append(receipt)
                            manifest_receipts.append(receipt)
                            atomic_json(receipt_path, task_receipts)
                        handle.flush()
                        print(
                            f"[{args.method}:{task}] {source_idx} "
                            f"in={len(ids)} out={len(generated)}", flush=True
                        )
            result_sha, result_lines = compute_result_binding(str(out_path))
            write_run_cap_manifest(
                runcap_sidecar,
                finalize_run_cap_manifest(
                    runcap_core, status="complete", n_samples=len(prepared),
                    input_tokens=input_tokens, truncated_flags=truncated,
                    result_file=out_path.name, result_sha256=result_sha,
                    result_lines=result_lines, input_ids_sha256=input_hashes,
                ),
            )
            source_manifest = build_quality_source_manifest(task, selected)
            source_manifest_path = quality_source_manifest_path(out_path)
            if source_manifest is not None:
                atomic_json(source_manifest_path, source_manifest)
        manifest["tasks"][task] = {
            "path": str(out_path.relative_to(Path(args.output_dir))),
            "runcap": str(Path(runcap_sidecar).relative_to(Path(args.output_dir))),
            "rows": len(selected),
            "data_sha256": data_sha,
            "prediction_sha256": sha256_file(out_path),
        }
        if source_manifest is not None:
            manifest["tasks"][task].update(
                source_manifest=str(source_manifest_path.relative_to(Path(args.output_dir))),
                source_manifest_sha256=sha256_file(source_manifest_path),
                choice_parser=LBV2_PARSER_VERSION,
            )
        if track_receipts:
            manifest["tasks"][task].update(
                receipts=str(receipt_path.relative_to(Path(args.output_dir))),
                receipts_sha256=sha256_file(receipt_path),
                receipt_rows=len(task_receipts),
            )
    if args.method == "quest":
        manifest["quest_call_counts"] = engine.model.quest_runtime_stats
        if manifest["quest_call_counts"]["decode_sparse_attn"] == 0:
            raise RuntimeError("Quest 未执行官方 decode_sparse_attn")
    if track_receipts:
        manifest["request_receipts"] = manifest_receipts
    atomic_json(
        quality_manifest_path(args.output_dir, args.shard_index, args.num_shards),
        manifest,
    )
    print(f"[ALL DONE] {args.output_dir}", flush=True)


def run_perf(args, engine, prompt_cfg, max_length):
    if args.max_new_tokens < 2:
        raise ValueError("perf 要求 max-new-tokens>=2")
    samples = args.sample_indices or [0]
    output = Path(args.output_dir) / f"{args.method}_performance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "longbench-official-performance-v1",
        "verification": "pending-independent-verification",
        "method": args.method,
        "model_path": str(Path(args.model_path).resolve()),
        "max_length": max_length,
        "tasks": args.tasks,
        "sample_indices": samples,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "runtime": engine.runtime,
        "cells": [],
    }
    expected = {
        (task, sample, repeat)
        for task in args.tasks for sample in samples for repeat in range(args.repeats)
    }
    for task in args.tasks:
        rows, data_sha = load_rows(args.data_root, task)
        for sample in samples:
            row = rows[sample]
            ids, identity = build_input(
                row, task, prompt_cfg[task], max_length, engine.tokenizer
            )
            for _ in range(args.warmup_requests):
                timed_pair(
                    engine, ids, args.max_new_tokens, require_positive=False
                )
            for repeat in range(args.repeats):
                generated, timing = timed_pair(engine, ids, args.max_new_tokens)
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
                if args.method == "quest":
                    payload["quest_call_counts"] = engine.model.quest_runtime_stats
                atomic_json(output, payload)
                print(
                    f"[PERF] {args.method} {task} sample={sample} repeat={repeat} "
                    f"TPOT={timing['tpot_ms']:.3f}ms", flush=True
                )
    print(f"[PERF DONE] {output}", flush=True)


def main():
    args = parse_args()
    if (len(set(args.tasks)) != len(args.tasks)
            or any(t not in {*ENGLISH16, "lbv2"} for t in args.tasks)):
        raise ValueError("tasks 必须是唯一的 LongBench 英文16任务或 lbv2")
    if "lbv2" in args.tasks and args.tasks != ["lbv2"]:
        raise ValueError("lbv2 必须作为唯一任务运行，不能与 LongBench v1 混合")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("要求 0<=shard-index<num-shards")
    if args.budget <= 0 or args.jssa_rank <= 0 or args.repeats <= 0:
        raise ValueError("budget/rank/repeats 必须为正")
    if args.method == "quest" and args.budget % args.quest_page_size:
        raise ValueError("Quest budget 必须整除 page-size")
    prompt_cfg = load_json(Path(args.config_path) / "dataset2prompt.json")
    maxgen_cfg = load_json(Path(args.config_path) / "dataset2maxlen.json")
    model_len_cfg = load_json(Path(args.config_path) / "model2maxlen.json")
    max_length = int(model_len_cfg[args.model_name])
    validate_model_contract(args.model_name, max_length)
    max_new = args.max_new_tokens if args.mode == "perf" else max(maxgen_cfg[t] for t in args.tasks)
    cap_gate_info = enforce_context_capability_gate(
        args.model_name, args.model_path, max_length, args.config_path
    )
    engine = OfficialEngine(args, max_length + max_new + 16)
    if args.mode == "quality":
        run_quality(
            args, engine, prompt_cfg, maxgen_cfg, max_length, cap_gate_info
        )
    else:
        run_perf(args, engine, prompt_cfg, max_length)


if __name__ == "__main__":
    main()
