"""NFA DAC LongBench 两臂任务正式结果的生产侧完整性与官方评分审计。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark.LongBench.eval import sample_score
from benchmark.rope_policy import configure_rope
from transformers import AutoConfig

EXPECTED_HEAD = "dfb750d71e2b8167ac31a97d5e7e3e87c0f7be5a"
ARMS = {
    "full_gate_off": ("full", "off"),
    "tail_gate_off": ("tail", "off"),
    "full_gate_on": ("full", "on"),
    "tail_gate_on": ("tail", "on"),
}
TASKS = ("hotpotqa", "gov_report")
MAX_NEW = {"hotpotqa": 32, "gov_report": 512}
REPO = Path("/tmp/nfa-dac-quality-20261115")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_arm(manifest_path, repo, position_capacity):
    arm = manifest_path.parent.name
    require(arm in ARMS, (arm, manifest_path))
    subspace, gate = ARMS[arm]
    base = manifest_path.parent
    manifest = load(manifest_path)
    require(manifest["schema"] == "longbench-official-quality-manifest-v1", manifest_path)
    require(manifest["verification"] == "pending-independent-verification", manifest_path)
    require(manifest["method"] == "nfa", manifest_path)
    require(list(manifest["tasks"]) == list(TASKS), manifest_path)
    runtime = manifest["runtime"]
    require(runtime["source"]["head"] == EXPECTED_HEAD, manifest_path)
    empty_diff = hashlib.sha256(b"").hexdigest()
    require(runtime["source"]["source_diff_sha256"] == empty_diff, manifest_path)
    require(runtime["source"]["quality_source_diff_sha256"] == empty_diff, manifest_path)
    require(runtime["execution"] == "eager" and runtime["num_layers"] == 36, manifest_path)
    require(runtime["dense_or_torch_decode_fallback"] is False, manifest_path)
    protocol = runtime["quality_protocol"]
    require(protocol["fixed_length_performance_receipts_reused"] is False, manifest_path)
    require(protocol["real_eos_min_new_stop_token_stop_string"] is True, manifest_path)
    require(protocol["per_request_independent_cache"] is True, manifest_path)
    config = runtime["nfa_selector_config"]
    require(config["subspace"] == subspace, manifest_path)
    require(config["level1_topk"] == 128 and config["level2_topk"] == 1024, manifest_path)
    require(config["page_size"] == 64 and config["near_pages"] == 32, manifest_path)
    require(config["level1_topk"] - config["near_pages"] == 96, manifest_path)
    require(config["near_tokens"] == "off" and config["near_len"] == 4096, manifest_path)
    require(config["near_method"] == "avg" and config["far_method"] == "minmax", manifest_path)
    require(config["sink_tokens"] == 4 and config["sliding_window_size"] == 128, manifest_path)
    require(config["subspace_cmp_ratio"] == 4, manifest_path)
    require(config["dynamic_gate_epsilon"] == 0.1 and config["dynamic_gate_rows"] == 256, manifest_path)
    require(config["boundary_bypass"] is True, manifest_path)
    require(config["enable_layer_skip"] is (gate == "on"), manifest_path)

    audit_dir = base.parent / "_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([
        str(repo),
        "/home/wangyuanshuo02/local_pylibs/venv/lib/python3.11/site-packages",
    ])
    with (audit_dir / f"official_eval_{arm}.log").open("w", encoding="utf-8") as log:
        subprocess.run([
            "/home/fluentllmenv/bin/python", "-m", "benchmark.LongBench.eval",
            "--output-dir", str(base), "--expect-count", "200",
        ], cwd=repo, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    official_result_path = base / "pred" / "result.json"
    official_result = load(official_result_path)
    require(official_result["_meta"]["runcap"]["strict_validated"] == 2, official_result_path)
    require(official_result["_meta"]["runcap"]["legacy_no_sidecar"] == 0, official_result_path)

    report = {
        "arm": arm,
        "source_head": EXPECTED_HEAD,
        "rows": 0,
        "tasks": {},
        "gate_counts": {"keep": 0, "near": 0, "far": 0, "both": 0},
        "manifest_path": str(manifest_path),
    }
    for task in TASKS:
        entry = manifest["tasks"][task]
        prediction_path = base / entry["path"]
        receipts_path = base / entry["receipts"]
        runcap_path = base / entry["runcap"]
        require(entry["rows"] == entry["receipt_rows"] == 200, manifest_path)
        require(entry["prediction_sha256"] == sha256(prediction_path), prediction_path)
        require(entry["receipts_sha256"] == sha256(receipts_path), receipts_path)
        predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        receipts = load(receipts_path)
        runcap = load(runcap_path)
        require(len(predictions) == len(receipts) == 200, prediction_path)
        require(runcap["status"] == "complete", runcap_path)
        require(runcap["result_lines"] == runcap["n_samples"] == 200, runcap_path)
        require(runcap["result_sha256"] == entry["prediction_sha256"], runcap_path)
        require(runcap["max_new_tokens"] == MAX_NEW[task], runcap_path)
        require(runcap["effective_cap"] == 31500, runcap_path)
        require(runcap["method"] == f"nfa_{arm}_eager", runcap_path)
        require(runcap["producer"]["git_commit"] == EXPECTED_HEAD, runcap_path)
        max_input_tokens = runcap["input_tokens"]["max"]
        require(max_input_tokens + MAX_NEW[task] <= position_capacity, runcap_path)
        require(runcap["truncated"]["count"] == sum(
            bool(flag) for flag in runcap["truncated"]["flags"]), runcap_path)
        scores = []
        for source_index, (prediction, receipt) in enumerate(zip(predictions, receipts)):
            sample_id = f"{task}:{source_index}"
            require(prediction["_id"] == sample_id, prediction_path)
            require(receipt["task"] == task and receipt["source_index"] == source_index, receipts_path)
            require(receipt["input"] == prediction["input"], receipts_path)
            require(prediction["input"]["input_tokens"] ==
                    runcap["input_tokens"]["per_sample"][source_index], prediction_path)
            require(prediction["input"]["truncated"] is
                    runcap["truncated"]["flags"][source_index], prediction_path)
            require(prediction["implementation"] == f"nfa_{arm}_eager", prediction_path)
            generated = receipt["generated_tokens"]
            require(1 <= generated <= MAX_NEW[task], receipts_path)
            require(receipt["stop_reason"] in {"eos", "stop_string", "max_new_tokens"}, receipts_path)
            require(receipt["logical_decode_steps"] == generated - 1, receipts_path)
            require(receipt["submitted_decode_forwards"] >= generated - 1, receipts_path)
            require(receipt["nfa_decode_steps"] == generated - 1, receipts_path)
            require(receipt["graph_capture_count"] == receipt["graph_replays"] == 0, receipts_path)
            require(receipt["graph_eager_fallbacks"] == receipt["submitted_decode_forwards"], receipts_path)
            decisions = receipt["prefill_decisions"]
            require(len(decisions) == 36 and set(decisions) == {str(i) for i in range(36)}, receipts_path)
            counts = {"keep": 0, "near": 0, "far": 0, "both": 0}
            for decision in decisions.values():
                key = ("both" if decision["skip_near"] and decision["skip_far"]
                       else "near" if decision["skip_near"]
                       else "far" if decision["skip_far"] else "keep")
                counts[key] += 1
            require(counts == receipt["gate_counts"], receipts_path)
            require(receipt["gate_observation_rows"] == 256, receipts_path)
            require(receipt["gate_epsilon"] == (0.1 if gate == "on" else 0.0), receipts_path)
            if gate == "off":
                require(counts == {"keep": 36, "near": 0, "far": 0, "both": 0}, receipts_path)
            for key in counts:
                report["gate_counts"][key] += counts[key]
            score = 100.0 * sample_score(
                task, prediction["pred"], prediction["answers"], prediction["all_classes"])
            scores.append({
                "_id": sample_id,
                "source_index": source_index,
                "official_score": score,
                "scorer": "LongBench-official-sample_score",
            })
        scores_path = prediction_path.with_suffix(".official_scores.json")
        scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        filename = prediction_path.name
        official_score = official_result[filename]["score"]
        recomputed = sum(row["official_score"] for row in scores) / len(scores)
        require(official_score == round(recomputed, 2), (official_score, recomputed))
        report["tasks"][task] = {
            "rows": len(scores),
            "official_score": official_score,
            "recomputed_mean_score": recomputed,
            "prediction_sha256": entry["prediction_sha256"],
            "official_scores": str(scores_path.relative_to(base)),
            "official_scores_sha256": sha256(scores_path),
            "ordered_input_sha256": runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
            "per_sample_input_sha256": runcap["input_identity"]["actual_input_ids"]["per_sample_sha256"],
            "runcap": "complete",
            "max_input_tokens": max_input_tokens,
            "max_input_plus_new_tokens": max_input_tokens + MAX_NEW[task],
            "position_capacity": position_capacity,
            "source_truncated_rows": runcap["truncated"]["count"],
            "hashes": "pass",
        }
        report["rows"] += len(scores)
    require(report["rows"] == 400, manifest_path)
    report["official_result_sha256"] = sha256(official_result_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--model-path", default=str(Path.home() / "Qwen3-8B"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    model_config = AutoConfig.from_pretrained(args.model_path)
    length_policy = configure_rope(model_config, 31500, max(MAX_NEW.values()))
    position_capacity = length_policy["runtime_max_position_embeddings"]
    require(position_capacity == 40960, length_policy)
    manifests = sorted(root.glob("*/quality_manifest_s0of1.json"))
    reports = [audit_arm(path, Path(args.repo), position_capacity) for path in manifests]
    require(len(reports) == args.cells, (len(reports), args.cells, manifests))
    by_arm = {report["arm"]: report for report in reports}
    require(len(by_arm) == len(reports), "duplicate LongBench arm")
    if len(reports) > 1:
        for task in TASKS:
            reference = reports[0]["tasks"][task]
            for report in reports[1:]:
                current = report["tasks"][task]
                require(current["ordered_input_sha256"] == reference["ordered_input_sha256"], task)
                require(current["per_sample_input_sha256"] == reference["per_sample_input_sha256"], task)
    for report in reports:
        for task in report["tasks"].values():
            del task["per_sample_input_sha256"]
    summary = {
        "schema": "nfa-dac-longbench-production-audit-v1",
        "source_head": EXPECTED_HEAD,
        "status": "pass-production-side",
        "verification": "pending-independent-verification",
        "formal_result": True,
        "rows_per_arm": 400,
        "cells": len(reports),
        "requests": 400 * len(reports),
        "paired_input_identity": "pass" if len(reports) > 1 else "single-arm",
        "length_policy": {
            "input_cap": 31500,
            "max_new_by_task": MAX_NEW,
            "runtime_position_capacity": position_capacity,
            "truncation": "official LongBench cap; per-row flags retained",
        },
        "results": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    for report in reports:
        print(report["arm"], {task: values["official_score"] for task, values in report["tasks"].items()}, report["gate_counts"])


if __name__ == "__main__":
    main()
