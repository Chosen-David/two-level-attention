"""Qwen3-8B × RULER NFA/FlashInfer 真实质量入口。

复用 LongBench 质量引擎、正式 RULER 输入与 string_match_all 评分。NIAH 使用
max-new=128、min-new=1 和真实 EOS；不截断源提示，不消费固定长度性能回执。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from benchmark.LongBench.run_cap_manifest import (
    build_lbv1_input_identity,
    build_run_cap_core,
    compute_result_binding,
    finalize_run_cap_manifest,
    gate_run_cap_identity,
    input_ids_sha256,
    read_run_cap_manifest,
    resolve_producer_identity,
    runcap_gate_lock,
    write_run_cap_manifest,
)
from benchmark.RULER.pred_ruler import (
    MIN_NEW_TOKENS,
    RULER_TASKS,
    _ruler_eos_ids,
    _ruler_max_new_tokens,
)
from benchmark.RULER.score_ruler import _answers_sha16, string_match_all
from benchmark.efficiency.benchmark_longbench_quality import (
    RequestQualityEngine,
    resolve_nfa_quality_config,
)
from benchmark.efficiency.benchmark_long_context_two_point import (
    source_receipt as long_context_source_receipt,
)
from benchmark.efficiency.benchmark_nfa_end_to_end import _load_model
from benchmark.official_baselines.longbench_official import (
    atomic_json,
    line_count,
    quality_receipts_path,
    sha256_file,
    sharding_sample_id,
)

REPO = Path(__file__).resolve().parents[2]
RULER_INPUT_CONTRACT = (
    "row.input; AutoTokenizer(truncation=False); no external prompt template"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("nfa", "flashinfer"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--nfa-subspace", choices=("full", "tail"), default="tail")
    parser.add_argument("--nfa-subspace-dim", type=int, choices=(16, 32, 64, 128), default=None)
    parser.add_argument("--nfa-gate", choices=("off", "on"), default="off")
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--task", choices=RULER_TASKS, required=True)
    parser.add_argument("--context-length", type=int,
                        choices=(32768, 65536, 131072), required=True)
    parser.add_argument("--data-root", required=True,
                        help="KVCache-Factory data/RULER 目录")
    parser.add_argument("--output-dir", required=True)
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
    paths = (
        "benchmark/efficiency/benchmark_ruler_quality.py",
        "benchmark/efficiency/benchmark_longbench_quality.py",
        "benchmark/RULER/pred_ruler.py",
        "benchmark/RULER/score_ruler.py",
        "benchmark/LongBench/run_cap_manifest.py",
    )
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "HEAD", "--binary", "--", *paths])
    receipt["quality_source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    return receipt


def load_ruler_rows(data_root, context_length, task):
    path = Path(data_root) / str(context_length) / f"{task}.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    for source_idx, row in enumerate(rows):
        if row.get("index") != source_idx:
            raise ValueError(f"RULER 源行 index 必须等于有序行号: {path} {source_idx}")
        if not isinstance(row.get("input"), str) or not isinstance(row.get("outputs"), list):
            raise ValueError(f"RULER 源行缺少 input/outputs: {path} {source_idx}")
    return rows, hashlib.sha256(raw).hexdigest(), path


def select_ruler_rows(rows, shard_index, num_shards, sample_indices=None):
    if not 0 <= shard_index < num_shards:
        raise ValueError("要求 0<=shard-index<num-shards")
    if sample_indices is not None:
        if len(set(sample_indices)) != len(sample_indices):
            raise ValueError("sample-indices 不得重复")
        if any(idx < 0 or idx >= len(rows) for idx in sample_indices):
            raise ValueError("sample-indices 超出 RULER 源人口")
    selected = [
        (source_idx, row)
        for source_idx, row in enumerate(rows)
        if source_idx % num_shards == shard_index
        and (sample_indices is None or source_idx in sample_indices)
    ]
    if sample_indices is not None and not selected:
        raise ValueError("显式 sample-indices 在当前分片中为空")
    return selected


def ruler_sample_id(task, source_idx, row):
    if row["index"] != source_idx:
        raise ValueError("RULER 样本身份与源行号不一致")
    return f"{task}:{source_idx}"


def ruler_official_score(pred, answers):
    """单个样本的官方 string_match_all 分数，取值 0–100。"""
    return string_match_all([pred], [answers])


def _answers_sha256(answers):
    return hashlib.sha256(json.dumps(
        answers, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _quality_scores_path(out_path):
    return Path(out_path).with_suffix(".scores.json")


def _quality_source_manifest_path(out_path):
    return Path(str(out_path) + ".source_manifest.json")


def _expected_score_records(prepared, records):
    expected = []
    for (source_idx, row), record in zip(prepared, records):
        expected.append({
            "_id": ruler_sample_id(row["task"], source_idx, row),
            "source_index": source_idx,
            "answers_sha256": _answers_sha256(row["outputs"]),
            "official_score": ruler_official_score(record["pred"], row["outputs"]),
            "scorer": "ruler-official-string_match_all",
        })
    return expected


def _restore_complete_outputs(out_path, prepared, implementation, task, context_length):
    records = [json.loads(line) for line in out_path.read_text(
        encoding="utf-8").splitlines()]
    if len(records) != len(prepared):
        raise RuntimeError(f"完整预测人口不一致: {out_path}")
    for record, (source_idx, row, ids, identity) in zip(records, prepared):
        sample_id = ruler_sample_id(task, source_idx, row)
        expected = {
            "_id": sample_id,
            "source_index": source_idx,
            "answers": row["outputs"],
            "_answers_sha": _answers_sha16(row["outputs"]),
            "answers_sha256": _answers_sha256(row["outputs"]),
            "length": row["length"],
            "input": identity,
            "implementation": implementation,
            "max_new_tokens": _ruler_max_new_tokens(task),
            "min_new_tokens": MIN_NEW_TOKENS,
            "official_score": ruler_official_score(record["pred"], row["outputs"]),
            "scorer": "ruler-official-string_match_all",
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"既有预测与当前 RULER 身份/评分不一致: {out_path} {source_idx}")
        generated = record.get("generated_token_ids")
        if (not isinstance(generated, list)
                or record.get("generated_tokens") != len(generated)
                or record.get("pred") is None):
            raise RuntimeError(f"既有预测缺少真实生成闭包: {out_path} {source_idx}")

    receipt_path = quality_receipts_path(out_path)
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipts, list) or len(receipts) != len(prepared):
        raise RuntimeError(f"RULER 回执人口与完整预测不一致: {receipt_path}")
    for receipt, record, (source_idx, row, _, identity) in zip(
            receipts, records, prepared):
        if (receipt.get("task") != task
                or receipt.get("context_length") != context_length
                or receipt.get("source_index") != source_idx
                or receipt.get("sample_id") != record["_id"]
                or receipt.get("input") != identity
                or receipt.get("generated_tokens") != record["generated_tokens"]
                or receipt.get("stop_reason") not in {"eos", "max_new_tokens"}
                or record["output_truncated"] != (
                    receipt["stop_reason"] == "max_new_tokens")):
            raise RuntimeError(f"RULER 回执与预测停止/输入身份不一致: {receipt_path} {source_idx}")
    score_path = _quality_scores_path(out_path)
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    if scores != _expected_score_records(
            [(idx, {**row, "task": task}) for idx, row, _, _ in prepared], records):
        raise RuntimeError(f"RULER 逐样本评分文件与预测不一致: {score_path}")
    return records, receipts, scores


def run_quality(args, engine, rows, data_sha256):
    """生成或严格核验一个 RULER task×length×arm×shard 的完整闭集。"""
    task = args.task
    max_new_tokens = _ruler_max_new_tokens(task)
    selected = select_ruler_rows(
        rows, args.shard_index, args.num_shards, args.sample_indices)
    out_dir = Path(args.output_dir) / f"L{args.context_length}" / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{task}-{args.quality_label}_s{args.shard_index}of{args.num_shards}.jsonl")

    source_identity = build_lbv1_input_identity(rows, RULER_INPUT_CONTRACT)
    input_identity = (
        source_identity if args.sample_indices is None
        else build_lbv1_input_identity(
            [row for _, row in selected], RULER_INPUT_CONTRACT)
    )
    run_timestamp = f"L{args.context_length}-s{args.shard_index}of{args.num_shards}"
    if args.sample_indices is not None:
        selection_sha = hashlib.sha256(json.dumps(
            args.sample_indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        run_timestamp += f"-filter{selection_sha}"
    cap_gate_info = {
        "registry_protocol_cap": args.context_length,
        "registry_restricted_context_treatment": None,
        "capability": engine.model.config.max_position_embeddings,
        "max_position_embeddings": engine.model.config.max_position_embeddings,
    }
    runcap_core = build_run_cap_core(
        model_name=args.model_name, model_path=args.model_path,
        method=args.quality_implementation, dataset=f"{task}@L{args.context_length}",
        dataset_fingerprint=input_identity["source_fingerprint"],
        run_timestamp=run_timestamp, effective_cap=args.context_length,
        cap_gate_info=cap_gate_info, max_new_tokens=max_new_tokens,
        producer=resolve_producer_identity(args.quality_producer_file),
        input_identity=input_identity,
    )
    runcap_core["effective_cap_source"] = "RULER nominal context; source prompt not truncated"
    runcap_core["source_file_sha256"] = data_sha256
    runcap_core["scorer"] = "ruler-official-string_match_all"
    if args.sample_indices is None:
        selected_ids = [
            sharding_sample_id(task, source_idx, source_identity["source_fingerprint"],
                               rows[source_idx])
            for source_idx in range(args.shard_index, len(rows), args.num_shards)
        ]
        runcap_core["sample_sharding"] = {
            "algorithm": "source_order[rank::world_size]",
            "rank": args.shard_index,
            "world_size": args.num_shards,
            "source_samples": len(rows),
            "selected_samples": len(selected),
            "source_sha256": source_identity["source_sha256"],
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")).hexdigest(),
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
        ids = engine.tokenizer(row["input"], truncation=False)["input_ids"]
        if len(ids) + max_new_tokens > engine.model.config.max_position_embeddings:
            raise ValueError(
                f"实际输入加输出超过已验证位置容量: {task} {source_idx} "
                f"{len(ids)}+{max_new_tokens}")
        identity = {
            "input_tokens": len(ids),
            "input_ids_sha256": input_ids_sha256(ids),
            "truncated": False,
        }
        prepared.append((source_idx, row, ids, identity))
    input_tokens = [len(ids) for _, _, ids, _ in prepared]
    truncated = [False] * len(prepared)
    input_hashes = [identity["input_ids_sha256"] for _, _, _, identity in prepared]

    manifest_receipts = []
    with runcap_gate_lock(str(out_path)):
        runcap_sidecar = gate_run_cap_identity(str(out_path), runcap_core)
        write_run_cap_manifest(
            runcap_sidecar,
            finalize_run_cap_manifest(
                runcap_core, status="generating", n_samples=len(prepared),
                input_tokens=input_tokens, truncated_flags=truncated,
                result_file=out_path.name, result_sha256=None, result_lines=None,
                input_ids_sha256=input_hashes,
            ),
        )
        receipt_path = quality_receipts_path(out_path)
        score_path = _quality_scores_path(out_path)
        if not prepared:
            out_path.write_text("", encoding="utf-8")
            records, task_receipts, scores = [], [], []
            atomic_json(receipt_path, [])
            atomic_json(score_path, [])
        elif line_count(out_path) == len(prepared):
            records, task_receipts, scores = _restore_complete_outputs(
                out_path, prepared, args.quality_implementation, task,
                args.context_length)
            engine.request_receipts.extend(task_receipts)
            manifest_receipts.extend(task_receipts)
            print(f"[SKIP] {task} L{args.context_length}: {len(prepared)} 条已完成",
                  flush=True)
        else:
            # 部分文件整任务重跑；旧预测、回执和分数不得混入新一轮。
            atomic_json(receipt_path, [])
            atomic_json(score_path, [])
            records, task_receipts, scores = [], [], []
            with out_path.open("w", encoding="utf-8") as handle:
                for source_idx, row, ids, identity in prepared:
                    receipts_before = len(engine.request_receipts)
                    generated, _ = engine.generate(
                        ids, max_new_tokens, ignore_eos=False,
                        min_new_tokens=MIN_NEW_TOKENS,
                    )
                    if len(engine.request_receipts) != receipts_before + 1:
                        raise RuntimeError(
                            f"RULER 生成未恰好产生一个逐请求回执: {task} {source_idx}")
                    receipt = engine.request_receipts[-1]
                    pred = engine.tokenizer.decode(generated, skip_special_tokens=True)
                    sample_id = ruler_sample_id(task, source_idx, row)
                    stop_reason = receipt["stop_reason"]
                    record = {
                        "pred": pred,
                        "answers": row["outputs"],
                        "generated_token_ids": generated,
                        "generated_tokens": len(generated),
                        "stop_reason": stop_reason,
                        "output_truncated": stop_reason == "max_new_tokens",
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": MIN_NEW_TOKENS,
                        "length": row["length"],
                        "_id": sample_id,
                        "source_index": source_idx,
                        "_answers_sha": _answers_sha16(row["outputs"]),
                        "answers_sha256": _answers_sha256(row["outputs"]),
                        "input": identity,
                        "implementation": args.quality_implementation,
                        "official_score": ruler_official_score(pred, row["outputs"]),
                        "scorer": "ruler-official-string_match_all",
                    }
                    # EOS 与 max-new 同时落在最后一个 token 时，observe() 优先
                    # 报告 EOS；长度达到预算本身不等于被 max-new 截断。
                    if stop_reason == "max_new_tokens" and len(generated) != max_new_tokens:
                        raise RuntimeError("max-new 停止原因与实际生成长度不一致")
                    score_record = {
                        "_id": sample_id,
                        "source_index": source_idx,
                        "answers_sha256": record["answers_sha256"],
                        "official_score": record["official_score"],
                        "scorer": record["scorer"],
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    receipt.update(
                        task=task, context_length=args.context_length,
                        source_index=source_idx, sample_id=sample_id, input=identity)
                    task_receipts.append(receipt)
                    manifest_receipts.append(receipt)
                    records.append(record)
                    scores.append(score_record)
                    atomic_json(receipt_path, task_receipts)
                    atomic_json(score_path, scores)
                    print(
                        f"[{args.method}:{task}:L{args.context_length}] {source_idx} "
                        f"in={len(ids)} out={len(generated)} "
                        f"score={record['official_score']:.6f}", flush=True,
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
        source_manifest = {task: {
            "ids": [record["_id"] for record in records],
            "answers_sha": {
                record["_id"]: record["answers_sha256"] for record in records},
        }}
        source_manifest_path = _quality_source_manifest_path(out_path)
        atomic_json(source_manifest_path, source_manifest)

    output_root = Path(args.output_dir)
    manifest = {
        "schema": "ruler-official-quality-manifest-v1",
        "verification": "pending-independent-verification",
        "method": args.method,
        "model_path": str(Path(args.model_path).resolve()),
        "task": task,
        "context_length": args.context_length,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": MIN_NEW_TOKENS,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "runtime": engine.runtime,
        "rows": len(records),
        "official_score": string_match_all(
            [record["pred"] for record in records],
            [record["answers"] for record in records]),
        "prediction": str(out_path.relative_to(output_root)),
        "prediction_sha256": sha256_file(out_path),
        "runcap": str(Path(runcap_sidecar).relative_to(output_root)),
        "receipts": str(receipt_path.relative_to(output_root)),
        "receipts_sha256": sha256_file(receipt_path),
        "scores": str(score_path.relative_to(output_root)),
        "scores_sha256": sha256_file(score_path),
        "source_manifest": str(source_manifest_path.relative_to(output_root)),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "data_sha256": data_sha256,
        "request_receipts": manifest_receipts,
    }
    manifest_path = output_root / (
        f"quality_manifest_L{args.context_length}_s{args.shard_index}of{args.num_shards}.json")
    atomic_json(manifest_path, manifest)
    print(f"[ALL DONE] {manifest_path}", flush=True)
    return manifest


def main():
    args = parse_args()
    if args.model_name != "Qwen3-8B" or args.visible_len <= 0:
        raise ValueError("当前入口仅验收 Qwen3-8B 且 visible-len 必须为正")
    # 提前验证分片和显式样本，避免模型加载后才发现空人口。
    rows, data_sha256, _ = load_ruler_rows(
        args.data_root, args.context_length, args.task)
    select_ruler_rows(rows, args.shard_index, args.num_shards, args.sample_indices)
    if args.method == "nfa":
        args.nfa_patch_args, args.nfa_gate_config = resolve_nfa_quality_config(
            args.nfa_subspace, args.nfa_gate, args.nfa_subspace_dim)
        arm = (
            f"{args.nfa_subspace}_gate_{args.nfa_gate}"
            if args.nfa_subspace_dim in (None, 32, 128)
            else f"tail{args.nfa_subspace_dim}_gate_{args.nfa_gate}")
    else:
        args.nfa_patch_args = args.nfa_gate_config = None
        arm = "fullkv"
    args.quality_label = f"{args.method}_{arm}_{args.execution}"
    args.quality_implementation = args.quality_label
    args.quality_producer_file = __file__

    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    max_new_tokens = _ruler_max_new_tokens(args.task)
    load_args = SimpleNamespace(
        method=args.method, model_path=args.model_path,
        context_length=args.context_length)
    model = _load_model(load_args, max_new_tokens, patch_args=args.nfa_patch_args)
    eos_ids = _ruler_eos_ids(model, tokenizer)
    engine = RequestQualityEngine(args, model, tokenizer, eos_ids)
    engine.runtime["source"] = source_receipt()
    engine.runtime["quality_protocol"] = {
        "fixed_length_performance_receipts_reused": False,
        "ruler_source_prompt_truncation": False,
        "real_eos_min_new": True,
        "official_scorer": "string_match_all",
        "per_request_independent_cache": True,
        "graph_session_released_at_request_boundary": args.method == "nfa",
    }
    with torch.inference_mode():
        run_quality(args, engine, rows, data_sha256)


if __name__ == "__main__":
    main()
