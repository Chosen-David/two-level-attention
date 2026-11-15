import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig

from benchmark.efficiency.benchmark_ruler_quality import ruler_official_score
from benchmark.rope_policy import configure_rope, verify_runtime_rope

EXPECTED_HEAD = "67f3b099073dd2d27fc0176b631eb9b4bd6a8214"
ARMS = {
    "nfa_full_gate_off": ("nfa", "full", "off"),
    "nfa_full_gate_on": ("nfa", "full", "on"),
    "nfa_tail_gate_off": ("nfa", "tail", "off"),
    "nfa_tail_gate_on": ("nfa", "tail", "on"),
    "flashinfer_fullkv": ("flashinfer", None, None),
}
LENGTHS = (32768, 65536, 131072)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_cell(manifest_path, expected_rows):
    arm = manifest_path.parent.parent.name
    assert arm in ARMS, (arm, manifest_path)
    method, subspace, gate = ARMS[arm]
    manifest = load_json(manifest_path)
    length = manifest["context_length"]
    assert length in LENGTHS
    assert manifest["schema"] == "ruler-official-quality-manifest-v1"
    assert manifest["verification"] == "pending-independent-verification"
    assert manifest["method"] == method and manifest["rows"] == expected_rows
    assert manifest["max_new_tokens"] == 128 and manifest["min_new_tokens"] == 1
    runtime = manifest["runtime"]
    assert runtime["source"]["head"] == EXPECTED_HEAD
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
    assert count == len(receipts) == len(scores) == expected_rows
    assert runcap["status"] == "complete" and runcap["result_lines"] == expected_rows
    assert runcap["result_sha256"] == manifest["prediction_sha256"]
    assert runcap["n_samples"] == expected_rows
    assert runcap["cap_treatment"]["capability"] == max(length + 128, 40960)
    input_hashes = set()
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
    assert sorted(source_indices) == list(range(expected_rows))
    if method == "nfa":
        config = runtime["nfa_selector_config"]
        assert config["subspace"] == subspace
        assert config["level1_topk"] == 128 and config["level2_topk"] == 1024
        assert config["page_size"] == 64 and config["near_pages"] == 32
        assert config["near_tokens"] == "off" and config["near_len"] == 4096
        assert config["near_method"] == "avg" and config["far_method"] == "minmax"
        assert config["sink_tokens"] == 4 and config["sliding_window_size"] == 128
        assert config["subspace_cmp_ratio"] == 4
        assert config["level1_topk"] - config["near_pages"] == 96
        assert config["dynamic_gate_epsilon"] == 0.1
        assert config["dynamic_gate_rows"] == 256
        assert config["boundary_bypass"] is True
        assert config["enable_layer_skip"] is (gate == "on")
    else:
        assert runtime["page_size"] == 64 and runtime["flashinfer_version"] == "0.5.2"
    return (arm, length), {
        "arm": arm,
        "context_length": length,
        "rows": count,
        "official_score": manifest["official_score"],
        "recomputed_mean_score": sum(row["official_score"] for row in scores) / count,
        "data_sha256": manifest["data_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "input_ids_sha256": sorted(input_hashes),
        "ordered_input_sha256": runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
        "stop_reasons": stop_reasons,
        "gate_counts": gate_counts if method == "nfa" else None,
        "runcap": "complete",
        "hashes": "pass",
        "manifest_path": str(manifest_path),
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
        manifests.extend(Path(raw_root).rglob("quality_manifest_L*_s0of1.json"))
    assert manifests, "no quality manifests found"
    cells = {}
    for manifest_path in sorted(set(manifests)):
        key, report = audit_cell(manifest_path, args.rows)
        assert key not in cells, f"duplicate cell {key}: {manifest_path} and {cells[key]['manifest_path']}"
        cells[key] = report
    expected = {(arm, length) for arm in ARMS for length in LENGTHS}
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
        "source_head": EXPECTED_HEAD,
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
