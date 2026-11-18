import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig

from benchmark.efficiency.benchmark_ruler_quality import ruler_official_score
from benchmark.LongBench.run_cap_manifest import _ordered_digest_hash
from benchmark.rope_policy import configure_rope, verify_runtime_rope

EXPECTED_HEADS = {
    "67f3b099073dd2d27fc0176b631eb9b4bd6a8214",
    "cdb83688c06f882b874e5aa20505057f1c13235d",
    "b70d8c08dcac64b5bfb4b86c78715a6956640e65",
    "2ddfd7416aedc03f0e43c6ea115c25c7212fa974",
    "233d09afe535a697dc30038b3fe59f06b6304e00",
    "9f3e8057acd528a904aa8883b830d541b389e39f",
}
A_ARMS = {
    "nfa_full_gate_off": ("nfa", "full", "off"),
    "nfa_full_gate_on": ("nfa", "full", "on"),
    "nfa_tail_gate_off": ("nfa", "tail", "off"),
    "nfa_tail_gate_on": ("nfa", "tail", "on"),
    "flashinfer_fullkv": ("flashinfer", None, None),
}
ARMS = {
    **A_ARMS,
    "nfa_tail16_gate_off": ("nfa", "tail", "off"),
    "nfa_tail64_gate_off": ("nfa", "tail", "off"),
    "tail16_gate_off": ("nfa", "tail", "off"),
    "tail64_gate_off": ("nfa", "tail", "off"),
    "nfa_full_gate_off_far_avg_top2": ("nfa", "full", "off"),
    "nfa_full_gate_on_far_avg_top2": ("nfa", "full", "on"),
    "nfa_tail_gate_off_far_avg_top2": ("nfa", "tail", "off"),
    "nfa_tail_gate_on_far_avg_top2": ("nfa", "tail", "on"),
    "nfa_tail_gate_off_near_minmax": ("nfa", "tail", "off"),
    "nfa_tail_gate_off_far_avg": ("nfa", "tail", "off"),
}
NEAR_METHODS = {
    "nfa_tail_gate_off_near_minmax": "minmax",
}
FAR_METHODS = {
    "nfa_tail_gate_off_far_avg": "avg",
    "nfa_full_gate_off_far_avg_top2": "avg_top2",
    "nfa_full_gate_on_far_avg_top2": "avg_top2",
    "nfa_tail_gate_off_far_avg_top2": "avg_top2",
    "nfa_tail_gate_on_far_avg_top2": "avg_top2",
}
SUBSPACE_DIMS = {
    "nfa_full_gate_off": 128,
    "nfa_full_gate_on": 128,
    "nfa_tail_gate_off": 32,
    "nfa_tail_gate_on": 32,
    "nfa_tail16_gate_off": 16,
    "nfa_tail64_gate_off": 64,
    "tail16_gate_off": 16,
    "tail64_gate_off": 64,
    "nfa_full_gate_off_far_avg_top2": 128,
    "nfa_full_gate_on_far_avg_top2": 128,
    "nfa_tail_gate_off_far_avg_top2": 32,
    "nfa_tail_gate_on_far_avg_top2": 32,
    "nfa_tail_gate_off_near_minmax": 32,
    "nfa_tail_gate_off_far_avg": 32,
}
LENGTHS = (32768, 65536, 131072)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_cell(manifest_path, expected_rows):
    arm = next(
        (parent.name for parent in manifest_path.parents if parent.name in ARMS),
        manifest_path.parent.parent.name,
    )
    assert arm in ARMS, (arm, manifest_path)
    method, subspace, gate = ARMS[arm]
    manifest = load_json(manifest_path)
    length = manifest["context_length"]
    assert length in LENGTHS
    assert manifest["schema"] == "ruler-official-quality-manifest-v1"
    assert manifest["verification"] == "pending-independent-verification"
    shard_index = manifest["shard_index"]
    num_shards = manifest["num_shards"]
    assert 0 <= shard_index < num_shards
    expected_shard_rows = len(range(shard_index, expected_rows, num_shards))
    assert manifest["method"] == method and manifest["rows"] == expected_shard_rows
    assert manifest["max_new_tokens"] == 128 and manifest["min_new_tokens"] == 1
    runtime = manifest["runtime"]
    assert runtime["source"]["head"] in EXPECTED_HEADS
    assert runtime["source"]["source_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert runtime["source"]["quality_source_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert runtime["execution"] == "eager" and runtime["num_layers"] == 36
    assert runtime["dense_or_torch_decode_fallback"] is False
    base = manifest_path.parent
    prediction_path = base / manifest["prediction"]
    receipts_path = base / manifest["receipts"]
    scores_path = base / manifest["scores"]
    runcap_path = base / manifest["runcap"]
    source_manifest_path = base / manifest["source_manifest"]
    assert sha256(prediction_path) == manifest["prediction_sha256"]
    assert sha256(receipts_path) == manifest["receipts_sha256"]
    assert sha256(scores_path) == manifest["scores_sha256"]
    assert sha256(source_manifest_path) == manifest["source_manifest_sha256"]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    receipts = load_json(receipts_path)
    scores = load_json(scores_path)
    runcap = load_json(runcap_path)
    count = len(predictions)
    assert count == len(receipts) == len(scores) == expected_shard_rows
    assert runcap["status"] == "complete" and runcap["result_lines"] == expected_shard_rows
    assert runcap["result_sha256"] == manifest["prediction_sha256"]
    assert runcap["n_samples"] == expected_shard_rows
    sharding = runcap.get("sample_sharding")
    if sharding is not None:
        assert sharding["algorithm"] == "source_order[rank::world_size]"
        assert sharding["rank"] == shard_index and sharding["world_size"] == num_shards
        assert sharding["source_samples"] == expected_rows
        assert sharding["selected_samples"] == expected_shard_rows
    else:
        assert num_shards == 1 and "sample_selection" in runcap
    position_capacity = runcap["cap_treatment"]["capability"]
    assert position_capacity == max(length + 128, 40960)
    max_input_tokens = runcap["input_tokens"]["max"]
    assert max_input_tokens + manifest["max_new_tokens"] <= position_capacity
    assert runcap["truncated"]["count"] == 0
    input_hashes = set()
    input_hashes_in_order = []
    source_indices = []
    gate_counts = {"keep": 0, "near": 0, "far": 0, "both": 0}
    stop_reasons = {}
    for prediction, receipt, score in zip(predictions, receipts, scores):
        source_idx = prediction["source_index"]
        source_indices.append(source_idx)
        sample_id = f"niah_multikey_2:{source_idx}"
        assert prediction["_id"] == receipt["sample_id"] == score["_id"] == sample_id
        assert receipt["source_index"] == score["source_index"] == source_idx
        assert prediction["answers_sha256"] == score["answers_sha256"]
        assert prediction["official_score"] == score["official_score"]
        assert ruler_official_score(prediction["pred"], prediction["answers"]) == score["official_score"]
        assert prediction["generated_tokens"] == receipt["generated_tokens"]
        assert prediction["stop_reason"] == receipt["stop_reason"]
        assert prediction["output_truncated"] is (receipt["stop_reason"] == "max_new_tokens")
        assert prediction["input"] == receipt["input"]
        assert prediction["input"]["truncated"] is False
        assert prediction["max_new_tokens"] == 128 and prediction["min_new_tokens"] == 1
        input_hashes.add(prediction["input"]["input_ids_sha256"])
        input_hashes_in_order.append(prediction["input"]["input_ids_sha256"])
        stop_reasons[receipt["stop_reason"]] = stop_reasons.get(receipt["stop_reason"], 0) + 1
        if method == "nfa":
            decisions = receipt["prefill_decisions"]
            assert len(decisions) == 36 and set(decisions) == {str(i) for i in range(36)}
            counts = {"keep": 0, "near": 0, "far": 0, "both": 0}
            for decision in decisions.values():
                key = ("both" if decision["skip_near"] and decision["skip_far"]
                       else "near" if decision["skip_near"]
                       else "far" if decision["skip_far"] else "keep")
                counts[key] += 1
            assert counts == receipt["gate_counts"]
            for key in counts:
                gate_counts[key] += counts[key]
            if gate == "off":
                assert receipt["gate_epsilon"] == 0.0
                assert counts == {"keep": 36, "near": 0, "far": 0, "both": 0}
            else:
                assert receipt["gate_epsilon"] == 0.1
            assert receipt["gate_observation_rows"] == 256
            assert receipt["logical_decode_steps"] == receipt["submitted_decode_forwards"] == receipt["nfa_decode_steps"]
            assert receipt["graph_replays"] == 0
        else:
            assert receipt["page_size"] == 64 and receipt["prefill_append_calls"] == 36
            assert receipt["python_decode_append_calls"] == 36 * receipt["logical_decode_steps"]
    if sharding is not None:
        expected_source_indices = list(range(shard_index, expected_rows, num_shards))
    else:
        expected_source_indices = runcap["sample_selection"]["source_indices"]
    assert sorted(source_indices) == sorted(expected_source_indices)
    assert len(set(source_indices)) == count
    recomputed_mean = sum(row["official_score"] for row in scores) / count
    assert abs(manifest["official_score"] - recomputed_mean) <= 1e-12
    if method == "nfa":
        config = runtime["nfa_selector_config"]
        assert config["subspace"] == subspace
        assert config["level1_topk"] == 128 and config["level2_topk"] == 1024
        assert config["page_size"] == 64 and config["near_pages"] == 32
        assert config["near_tokens"] == "off" and config["near_len"] == 4096
        assert config["near_method"] == NEAR_METHODS.get(arm, "avg")
        assert config["far_method"] == FAR_METHODS.get(arm, "minmax")
        assert config["sink_tokens"] == 4 and config["sliding_window_size"] == 128
        expected_dim = SUBSPACE_DIMS[arm]
        expected_ratio = 4 if subspace == "full" else 128 // expected_dim
        assert config["subspace_cmp_ratio"] == expected_ratio
        if "subspace_dim" in config:
            assert config["subspace_dim"] == expected_dim
            assert config["enable_subspace"] is (subspace != "full")
        else:
            assert runtime["source"]["head"] != "b70d8c08dcac64b5bfb4b86c78715a6956640e65"
            assert expected_dim in {32, 128}
        assert config["level1_topk"] - config["near_pages"] == 96
        assert config["dynamic_gate_epsilon"] == 0.1
        assert config["dynamic_gate_rows"] == 256
        assert config["boundary_bypass"] is True
        assert config["enable_layer_skip"] is (gate == "on")
    else:
        assert runtime["page_size"] == 64 and runtime["flashinfer_version"] == "0.5.2"
    return (arm, length), {
        "arm": arm,
        "source_head": runtime["source"]["head"],
        "context_length": length,
        "rows": count,
        "official_score": manifest["official_score"],
        "recomputed_mean_score": recomputed_mean,
        "data_sha256": manifest["data_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "scores_sha256": manifest["scores_sha256"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_indices": source_indices,
        "per_sample_input_sha256": input_hashes_in_order,
        "input_ids_sha256": sorted(input_hashes),
        "ordered_input_sha256": runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
        "max_input_tokens": max_input_tokens,
        "max_input_plus_new_tokens": max_input_tokens + manifest["max_new_tokens"],
        "position_capacity": position_capacity,
        "source_truncated_rows": runcap["truncated"]["count"],
        "stop_reasons": stop_reasons,
        "gate_counts": gate_counts if method == "nfa" else None,
        "runcap": "complete",
        "hashes": "pass",
        "manifest_path": str(manifest_path),
    }


def merge_audited_shards(reports, expected_rows):
    """把逐分片审计结果按 source_index 重排为一个完整逻辑单元。"""
    assert reports
    num_shards_values = {report["num_shards"] for report in reports}
    assert len(num_shards_values) == 1
    num_shards = num_shards_values.pop()
    if num_shards <= 1 or len(reports) != num_shards:
        raise ValueError(
            f"incomplete sharded cell: expected {num_shards} shards, got {len(reports)}")
    reports = sorted(reports, key=lambda report: report["shard_index"])
    assert [report["shard_index"] for report in reports] == list(range(num_shards))
    for field in ("arm", "source_head", "context_length", "data_sha256",
                  "position_capacity"):
        assert len({report[field] for report in reports}) == 1, field
    assert sum(report["rows"] for report in reports) == expected_rows

    source_order = sorted(
        (source_idx, input_sha)
        for report in reports
        for source_idx, input_sha in zip(
            report["source_indices"], report["per_sample_input_sha256"])
    )
    assert [source_idx for source_idx, _ in source_order] == list(range(expected_rows))
    ordered_hashes = [input_sha for _, input_sha in source_order]
    stop_reasons = {}
    for report in reports:
        for reason, count in report["stop_reasons"].items():
            stop_reasons[reason] = stop_reasons.get(reason, 0) + count
    if reports[0]["gate_counts"] is None:
        assert all(report["gate_counts"] is None for report in reports)
        gate_counts = None
    else:
        gate_counts = {
            key: sum(report["gate_counts"][key] for report in reports)
            for key in ("keep", "near", "far", "both")
        }
    recomputed_mean = sum(
        report["recomputed_mean_score"] * report["rows"] for report in reports
    ) / expected_rows
    manifest_mean = sum(
        report["official_score"] * report["rows"] for report in reports
    ) / expected_rows
    assert abs(manifest_mean - recomputed_mean) <= 1e-12
    max_input_tokens = max(report["max_input_tokens"] for report in reports)
    return {
        "arm": reports[0]["arm"],
        "source_head": reports[0]["source_head"],
        "context_length": reports[0]["context_length"],
        "rows": expected_rows,
        "official_score": manifest_mean,
        "recomputed_mean_score": recomputed_mean,
        "data_sha256": reports[0]["data_sha256"],
        "prediction_sha256": [report["prediction_sha256"] for report in reports],
        "scores_sha256": [report["scores_sha256"] for report in reports],
        "num_shards": num_shards,
        "source_indices": [source_idx for source_idx, _ in source_order],
        "per_sample_input_sha256": ordered_hashes,
        "input_ids_sha256": sorted(set(ordered_hashes)),
        "ordered_input_sha256": _ordered_digest_hash(ordered_hashes),
        "max_input_tokens": max_input_tokens,
        "max_input_plus_new_tokens": max_input_tokens + 128,
        "position_capacity": reports[0]["position_capacity"],
        "source_truncated_rows": sum(report["source_truncated_rows"] for report in reports),
        "stop_reasons": stop_reasons,
        "gate_counts": gate_counts,
        "runcap": "complete",
        "hashes": "pass",
        "manifest_path": [report["manifest_path"] for report in reports],
        "shards": [{
            "shard_index": report["shard_index"],
            "rows": report["rows"],
            "official_score": report["official_score"],
            "prediction_sha256": report["prediction_sha256"],
            "scores_sha256": report["scores_sha256"],
            "ordered_input_sha256": report["ordered_input_sha256"],
            "manifest_path": report["manifest_path"],
        } for report in reports],
    }


def rope_receipts(model_path):
    receipts = {}
    for length in LENGTHS:
        config = AutoConfig.from_pretrained(model_path)
        configure_rope(config, length, 128, yarn=length > 32768,
                       factor={65536: 2.0, 131072: 4.0}.get(length))
        receipt = verify_runtime_rope(config)
        assert receipt["status"] == "verified"
        assert receipt["requested_factor"] == {32768: None, 65536: 2.0, 131072: 4.0}[length]
        assert receipt["runtime_max_position_embeddings"] == max(length + 128, 40960)
        assert receipt["max_inv_freq_error"] <= 2e-6
        assert max(receipt["max_cos_error"], receipt["max_sin_error"]) <= 5e-4
        receipts[str(length)] = receipt
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--model-path", default=str(Path.home() / "Qwen3-8B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    manifests = []
    for raw_root in args.roots:
        manifests.extend(Path(raw_root).rglob("quality_manifest_L*_s*of*.json"))
    assert manifests, "no quality manifests found"
    groups = {}
    for manifest_path in sorted(set(manifests)):
        key, report = audit_cell(manifest_path, args.rows)
        groups.setdefault(key, []).append(report)
    cells = {}
    for key, reports in groups.items():
        if len(reports) == 1:
            assert reports[0]["num_shards"] == 1, (
                f"incomplete sharded cell {key}: {reports[0]['manifest_path']}")
            cells[key] = reports[0]
        else:
            cells[key] = merge_audited_shards(reports, args.rows)
    expected = {(arm, length) for arm in A_ARMS for length in LENGTHS}
    if args.cells == 15:
        assert set(cells) == expected, (sorted(expected - set(cells)), sorted(set(cells) - expected))
    else:
        assert len(cells) == args.cells
    for length in LENGTHS:
        length_cells = [report for (arm, cell_length), report in cells.items() if cell_length == length]
        if len(length_cells) > 1:
            assert len({report["data_sha256"] for report in length_cells}) == 1
            assert len({tuple(report["input_ids_sha256"]) for report in length_cells}) == 1
            assert len({report["ordered_input_sha256"] for report in length_cells}) == 1
    report = {
        "schema": "nfa-dac-ruler-matrix-production-audit-v1",
        "source_heads": sorted({row["source_head"] for row in cells.values()}),
        "status": "pass-production-side",
        "verification": "pending-independent-verification",
        "formal_result": args.formal,
        "rows_per_cell": args.rows,
        "cells": len(cells),
        "requests": args.rows * len(cells),
        "paired_input_identity": "pass",
        "rope": rope_receipts(args.model_path),
        "results": sorted(cells.values(), key=lambda row: (row["context_length"], row["arm"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"results", "rope"}}, indent=2))
    for row in report["results"]:
        print(row["context_length"], row["arm"], row["rows"], row["official_score"], row["gate_counts"])


if __name__ == "__main__":
    main()
