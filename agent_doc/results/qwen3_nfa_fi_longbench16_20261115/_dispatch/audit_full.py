"""闭集后重算 NFA/FlashInfer 九路 LongBench 质量人口、回执和 backend 计数。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NFA_REPO = Path("/tmp/qwen3-nfa-fi-quality-cbfd70da-20261114")
DATA_ROOT = Path("/home/wangyuanshuo02/datasets/LongBench/data")
EXPECTED_HEADS = {
    "nfa": "cbfd70da607eb5afed4343d6d055073f450e4162",
    "flashinfer": "7fac274fda9df13e82caadd95f3cca5c8374a1a3",
}
EXPECTED_HARNESS_SHA = "b312e9947e5a9b1451e0ef36d3112a515b6ecee912598b04f58c3a08f46e00ef"
EXPECTED_SOURCE_DIRS = {
    "nfa": NFA_REPO,
    "flashinfer": Path("/tmp/qwen3-nfa-fi-quality-7fac274f-20261114"),
}
EXPECTED_RANK_ROWS = [428, 428, 415, 415, 415, 413, 412, 412, 412]
ENGLISH16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_source_identity(record, method, path, *, runcap=False):
    source = record["producer"] if runcap else record["runtime"]["source"]
    expected_harness = str(
        EXPECTED_SOURCE_DIRS[method] /
        "benchmark/efficiency/benchmark_longbench_quality.py")
    head_key = "git_commit" if runcap else "head"
    sha_key = "script_sha256" if runcap else "quality_harness_sha256"
    path_key = "script_path" if runcap else "quality_harness"
    require(source[head_key] == EXPECTED_HEADS[method], path)
    require(source[sha_key] == EXPECTED_HARNESS_SHA, path)
    require(source[path_key] == expected_harness, path)


maxgen = load(NFA_REPO / "benchmark/LongBench/config/dataset2maxlen.json")
source_counts = {
    task: sum(1 for line in (DATA_ROOT / f"{task}.jsonl").read_text(
        encoding="utf-8").splitlines() if line)
    for task in ENGLISH16
}
arm_inputs = {}
summary = {
    "schema": "qwen3-nfa-fi-longbench-quality-rank-audit-v2",
    "verification": "pending-independent-verification",
    "source_heads": EXPECTED_HEADS,
    "world_size": 9,
    "arms": {},
}

for method in ("nfa", "flashinfer"):
    implementation = f"{method}_eager"
    expected_head = EXPECTED_HEADS[method]
    all_inputs = {}
    stop_reasons = Counter()
    generated_total = 0
    physical_total = 0
    rank_summaries = []
    for rank, expected_rank_rows in enumerate(EXPECTED_RANK_ROWS):
        rank_root = ROOT / method / f"rank{rank}"
        manifest_path = rank_root / f"quality_manifest_s{rank}of9.json"
        manifest = load(manifest_path)
        require(manifest["verification"] == "pending-independent-verification",
                manifest_path)
        require(manifest["method"] == method, manifest_path)
        require(manifest["runtime"]["execution"] == "eager", manifest_path)
        require(manifest["runtime"]["batch_size"] == 1, manifest_path)
        require(manifest["runtime"]["dense_or_torch_decode_fallback"] is False,
                manifest_path)
        validate_source_identity(manifest, method, manifest_path)
        protocol = manifest["runtime"]["quality_protocol"]
        require(protocol["fixed_length_performance_receipts_reused"] is False,
                manifest_path)
        require(protocol["real_eos_min_new_stop_token_stop_string"] is True,
                manifest_path)
        require(protocol["per_request_independent_cache"] is True, manifest_path)
        require(protocol["graph_session_released_at_request_boundary"] ==
                (method == "nfa"), manifest_path)
        require(list(manifest["tasks"]) == ENGLISH16, manifest_path)
        require(len(manifest["request_receipts"]) == expected_rank_rows,
                manifest_path)
        if method == "flashinfer" and rank >= 6:
            reuse_path = ROOT / "_status" / f"rank{rank}_flashinfer.reused.json"
            reuse = load(reuse_path)
            require(reuse["source_head"] == expected_head, reuse_path)
            require(reuse["rows"] == expected_rank_rows, reuse_path)
            require(reuse["source_manifest_sha256"] == sha256(manifest_path),
                    reuse_path)

        rank_rows = 0
        for task in ENGLISH16:
            entry = manifest["tasks"][task]
            pred_path = rank_root / entry["path"]
            runcap_path = rank_root / entry["runcap"]
            receipts_path = rank_root / entry["receipts"]
            expected_rows = len(range(rank, source_counts[task], 9))
            require(entry["rows"] == entry["receipt_rows"] == expected_rows,
                    (rank, task, entry))
            require(entry["prediction_sha256"] == sha256(pred_path), pred_path)
            require(entry["receipts_sha256"] == sha256(receipts_path), receipts_path)
            rows = [json.loads(line) for line in pred_path.read_text(
                encoding="utf-8").splitlines()]
            receipts = load(receipts_path)
            require(len(rows) == len(receipts) == expected_rows,
                    (rank, task, len(rows), len(receipts), expected_rows))

            runcap = load(runcap_path)
            validate_source_identity(runcap, method, runcap_path, runcap=True)
            require(runcap["status"] == "complete", runcap_path)
            require(runcap["method"] == implementation, runcap_path)
            require(runcap["effective_cap"] == 31500, runcap_path)
            require(runcap["max_new_tokens"] == maxgen[task], runcap_path)
            require(runcap["result_file"] == pred_path.name, runcap_path)
            require(runcap["result_sha256"] == sha256(pred_path), runcap_path)
            require(runcap["result_lines"] == runcap["n_samples"] == expected_rows,
                    runcap_path)
            sharding = runcap["sample_sharding"]
            require(sharding["algorithm"] == "source_order[rank::world_size]",
                    runcap_path)
            require(sharding["rank"] == rank and sharding["world_size"] == 9,
                    runcap_path)
            require(sharding["source_samples"] == source_counts[task], runcap_path)
            require(sharding["selected_samples"] == expected_rows, runcap_path)
            identity = runcap["input_identity"]
            require(identity["schema"] == "longbench-input-identity-v2", runcap_path)
            require(identity["sample_count"] == source_counts[task], runcap_path)
            require(identity["actual_input_ids"]["count"] == expected_rows,
                    runcap_path)
            require(len(identity["actual_input_ids"]["per_sample_sha256"]) ==
                    expected_rows, runcap_path)

            for position, (row, receipt) in enumerate(zip(rows, receipts)):
                source_index = rank + position * 9
                require(row["_id"] == f"{task}:{source_index}", row["_id"])
                require(row["implementation"] == implementation, row["_id"])
                require(receipt["task"] == task and
                        receipt["source_index"] == source_index, row["_id"])
                require(receipt["input"] == row["input"], row["_id"])
                generated = receipt["generated_tokens"]
                require(1 <= generated <= maxgen[task], row["_id"])
                require(receipt["stop_reason"] in
                        {"eos", "stop_string", "max_new_tokens"}, row["_id"])
                steps = generated - 1
                require(receipt["logical_decode_steps"] == steps, row["_id"])
                physical = receipt["submitted_decode_forwards"]
                if method == "nfa":
                    require(receipt["nfa_decode_steps"] == steps, row["_id"])
                    require(physical >= steps, row["_id"])
                    require(receipt["graph_capture_count"] == 0, row["_id"])
                    require(receipt["graph_replays"] == 0, row["_id"])
                    require(receipt["graph_eager_fallbacks"] == physical, row["_id"])
                else:
                    require(physical == steps, row["_id"])
                    require(receipt["prefill_append_calls"] == 36, row["_id"])
                    require(receipt["python_decode_append_calls"] == steps * 36,
                            row["_id"])
                    require(receipt["python_decode_wrapper_calls"] == steps * 36,
                            row["_id"])
                    require(receipt["graph_capture_count"] == 0, row["_id"])
                    require(receipt["graph_replays"] == 0, row["_id"])
                    require(receipt["graph_eager_fallbacks"] == 0, row["_id"])
                    require(receipt["wrapper"] ==
                            "BatchDecodeWithPagedKVCacheWrapper", row["_id"])
                    require(receipt["page_size"] == 64, row["_id"])
                stop_reasons[receipt["stop_reason"]] += 1
                generated_total += generated
                physical_total += physical
                key = f"{task}:{source_index}"
                require(key not in all_inputs, key)
                all_inputs[key] = {
                    "input": row["input"],
                    "raw_ids_sha256": identity["actual_input_ids"][
                        "per_sample_sha256"][position],
                }
            rank_rows += expected_rows
        require(rank_rows == expected_rank_rows, (rank, rank_rows))
        rank_summaries.append({"rank": rank, "rows": rank_rows,
                               "manifest_sha256": sha256(manifest_path)})
    require(len(all_inputs) == 3750, (method, len(all_inputs)))
    arm_inputs[method] = all_inputs
    summary["arms"][method] = {
        "source_head": expected_head,
        "rows": len(all_inputs),
        "generated_tokens": generated_total,
        "submitted_decode_forwards": physical_total,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "ranks": rank_summaries,
    }

require(arm_inputs["nfa"] == arm_inputs["flashinfer"],
        "NFA/FlashInfer 输入身份或逐样本 raw input IDs SHA 不一致")
summary["cross_arm_input_identity"] = "PASS"
summary["status"] = "PASS"
output = ROOT / "rank_audit.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
print(json.dumps({"status": "PASS", "arms": {
    method: {"rows": data["rows"], "stop_reasons": data["stop_reasons"]}
    for method, data in summary["arms"].items()
}}, ensure_ascii=False))
