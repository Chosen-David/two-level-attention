#!/usr/bin/env python3
"""Qwen3-8B NFA K2 四档 Gov/Hot 质量与 L1+L2 算子生产侧闭集审计。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark.LongBench.eval import sample_score
from benchmark.rope_policy import configure_rope
from transformers import AutoConfig

PRODUCER = "ebcae298297c18f68ec224de611dc8b922ab9e95"
QUALITY_HARNESS_SHA256 = "d05703fb50bc106e560f88588c020418eca82dbcc51989a8450a031d73f03fde"
OPERATOR_HARNESS_SHA256 = "f64a35b43e258f33617c5156439a66ed4a8672a03ec80cb22d86af902e7c9cb1"
K2_VALUES = (256, 512, 1024, 2048)
EPS_VALUES = (0.01, 0.05, 0.1, 0.2, 0.5)
TASKS = ("hotpotqa", "gov_report")
MAX_NEW = {"hotpotqa": 32, "gov_report": 512}
EMPTY_DIFF = hashlib.sha256(b"").hexdigest()


def arm_name(k2, epsilon):
    arm = "tail_gate_on"
    if epsilon != 0.1:
        arm += f"_eps{epsilon:g}"
    if k2 != 1024:
        arm += f"_k2{k2}"
    return arm


def operator_cell_dir(root, task, k2, epsilon):
    if epsilon == 0.1:
        return root / "operators" / task / f"k2{k2}"
    return root / "operators" / task / f"eps{epsilon:g}" / f"k2{k2}"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_quality_cell(manifest_path, repo, position_capacity, task, k2, epsilon):
    arm = arm_name(k2, epsilon)
    require(manifest_path.parent.name == arm, manifest_path)
    base = manifest_path.parent
    manifest = load(manifest_path)
    require(manifest["schema"] == "longbench-official-quality-manifest-v1", manifest_path)
    require(manifest["verification"] == "pending-independent-verification", manifest_path)
    require(manifest["method"] == "nfa", manifest_path)
    require(list(manifest["tasks"]) == [task], manifest_path)
    runtime = manifest["runtime"]
    source = runtime["source"]
    require(source["head"] == PRODUCER, manifest_path)
    require(source["source_diff_sha256"] == EMPTY_DIFF, manifest_path)
    require(source["quality_source_diff_sha256"] == EMPTY_DIFF, manifest_path)
    require(source["quality_harness_sha256"] == QUALITY_HARNESS_SHA256, manifest_path)
    require(runtime["execution"] == "eager" and runtime["num_layers"] == 36, manifest_path)
    require(runtime["dense_or_torch_decode_fallback"] is False, manifest_path)
    protocol = runtime["quality_protocol"]
    require(protocol["fixed_length_performance_receipts_reused"] is False, manifest_path)
    require(protocol["real_eos_min_new_stop_token_stop_string"] is True, manifest_path)
    require(protocol["per_request_independent_cache"] is True, manifest_path)
    config = runtime["nfa_selector_config"]
    require(config["subspace"] == "tail", manifest_path)
    require(config["subspace_dim"] == 32 and config["subspace_cmp_ratio"] == 4, manifest_path)
    require(config["enable_subspace"] is True, manifest_path)
    require(config["level1_topk"] == 128 and config["level2_topk"] == k2, manifest_path)
    require(config["page_size"] == 64 and config["near_pages"] == 32, manifest_path)
    require(config["level1_topk"] - config["near_pages"] == 96, manifest_path)
    require(config["near_tokens"] == "off" and config["near_len"] == 4096, manifest_path)
    require(config["near_method"] == "avg" and config["far_method"] == "minmax", manifest_path)
    require(config["sink_tokens"] == 4 and config["sliding_window_size"] == 128, manifest_path)
    require(config["dynamic_gate_epsilon"] == epsilon and config["dynamic_gate_rows"] == 256, manifest_path)
    require(config["boundary_bypass"] is True and config["enable_layer_skip"] is True, manifest_path)
    require(runtime["nfa_quality_arm"] == arm, manifest_path)

    audit_dir = base.parent.parent / "_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([
        str(repo),
        "/home/wangyuanshuo02/local_pylibs/venv/lib/python3.11/site-packages",
    ])
    with (audit_dir / f"official_eval_{task}_eps{epsilon:g}_k2{k2}.log").open("w", encoding="utf-8") as log:
        subprocess.run([
            "/home/fluentllmenv/bin/python", "-m", "benchmark.LongBench.eval",
            "--output-dir", str(base), "--expect-count", "200",
        ], cwd=repo, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    official_result_path = base / "pred/result.json"
    official_result = load(official_result_path)
    require(official_result["_meta"]["runcap"]["strict_validated"] == 1, official_result_path)
    require(official_result["_meta"]["runcap"]["legacy_no_sidecar"] == 0, official_result_path)

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
    require(runcap["producer"]["git_commit"] == PRODUCER, runcap_path)
    max_input_tokens = runcap["input_tokens"]["max"]
    require(max_input_tokens + MAX_NEW[task] <= position_capacity, runcap_path)
    require(runcap["truncated"]["count"] == sum(
        bool(flag) for flag in runcap["truncated"]["flags"]), runcap_path)

    scores = []
    gate_counts = {"keep": 0, "near": 0, "far": 0, "both": 0}
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
        require(receipt["gate_epsilon"] == epsilon, receipts_path)
        for key in counts:
            gate_counts[key] += counts[key]
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
    official_score = official_result[prediction_path.name]["score"]
    recomputed = sum(row["official_score"] for row in scores) / len(scores)
    require(official_score == round(recomputed, 2), (official_score, recomputed))
    return {
        "arm": arm,
        "level2_topk": k2,
        "task": task,
        "source_head": PRODUCER,
        "gate_epsilon": epsilon,
        "rows": len(scores),
        "official_score": official_score,
        "recomputed_mean_score": recomputed,
        "prediction_sha256": entry["prediction_sha256"],
        "official_scores": str(scores_path.relative_to(base)),
        "official_scores_sha256": sha256(scores_path),
        "official_result_sha256": sha256(official_result_path),
        "ordered_input_sha256": runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
        "per_sample_input_sha256": runcap["input_identity"]["actual_input_ids"]["per_sample_sha256"],
        "runcap": "complete",
        "max_input_tokens": max_input_tokens,
        "max_input_plus_new_tokens": max_input_tokens + MAX_NEW[task],
        "position_capacity": position_capacity,
        "source_truncated_rows": runcap["truncated"]["count"],
        "gate_counts": gate_counts,
        "cell_dir": str(base),
        "hashes": "pass",
    }


def audit_operator_cell(root, repo, task, k2, epsilon):
    cell = operator_cell_dir(root, task, k2, epsilon)
    path = cell / "nfa-operator-profile/nfa-operator-profile_l32768_b1_eager.json"
    require(path.is_file(), path)
    row = load(path)
    summary_path = cell / "summary.json"
    summary = load(summary_path)
    require(row["schema"] == "nfa-selector-and-e2e-v2", path)
    source = row["source"]
    require(source["head"] == PRODUCER and source["diff_sha256"] == EMPTY_DIFF, path)
    require(source["harness_sha256"] == OPERATOR_HARNESS_SHA256, path)
    require(row["gpu_hygiene"] == "clean", path)
    expected = {
        "level2_topk": k2, "task": task, "sample_index": 0,
        "context_length": 32768, "batch_size": 1, "execution": "eager",
        "measurement_mode": "operator-profile", "subspace": "tail",
        "l2_subspace": "same", "gate_epsilon": epsilon,
        "force_skip_mode": "natural", "operator_mode": "off", "visible_len": 8192,
        "warmup_steps": 3, "profile_repeat": 5, "decode_steps": 32,
        "profiled_decode_steps": 5,
        "input_provenance": "deterministic-longbench-context-repetition",
    }
    for key, value in expected.items():
        require(row[key] == value, (path, key, row[key], value))
    require(row["graph_capture_count"] == 0, path)
    require(row["nsys_profile_finalized"] is True, path)
    require(row["region_kernel_audit"]["passed"] is True, path)
    for key in ("report", "sqlite"):
        artifact = Path(row["nsys_receipt"][f"{key}_path"])
        require(artifact.is_file(), artifact)
        require(row["nsys_receipt"][f"{key}_sha256"] == sha256(artifact), artifact)
    for key in ("l1_ms", "l2_ms", "main_attention_ms", "nfa_operator_total_ms"):
        require(row["operator_summary"][key]["mean_ms_per_step"] > 0, (path, key))
    require(summary["validation_status"] == "pending-independent-verification", summary_path)
    require(summary["arguments"]["level2_topk"] == k2, summary_path)
    require(summary["arguments"]["gate_epsilon"] == epsilon, summary_path)
    require(summary["source_by_arm"]["nfa-operator-profile"] == source, summary_path)
    require(summary["matrix_driver_sha256"] == sha256(
        repo / "benchmark/efficiency/run_nfa_perf_matrix.py"), summary_path)
    require(len(summary["records"]) == 1, summary_path)
    return {
        "arm": arm_name(k2, epsilon), "level2_topk": k2, "task": task,
        "source_head": PRODUCER, "gate_epsilon": epsilon, "gpu_hygiene": "clean",
        "operator_summary": row["operator_summary"],
        "region_kernel_audit": row["region_kernel_audit"],
        "skip_validation": row["skip_validation"],
        "input": row["input"], "nsys_receipt": row["nsys_receipt"],
        "receipt_path": str(path), "receipt_sha256": sha256(path),
        "summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-path", default=str(Path.home() / "Qwen3-8B"))
    parser.add_argument("--longbench-data", default=str(Path.home() / "datasets/LongBench/data"))
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve() if args.output else root / "audit.json"
    evidence_path = root / "evidence_manifest.json"
    model_config = AutoConfig.from_pretrained(args.model_path)
    length_policy = configure_rope(model_config, 31500, max(MAX_NEW.values()))
    position_capacity = length_policy["runtime_max_position_embeddings"]
    require(position_capacity == 40960, length_policy)

    quality = []
    operator = []
    for epsilon in EPS_VALUES:
        for k2 in K2_VALUES:
            for task in TASKS:
                manifest = (root / "cells" / task / arm_name(k2, epsilon)
                            / "quality_manifest_s0of1.json")
                require(manifest.is_file(), manifest)
                quality.append(audit_quality_cell(
                    manifest, repo, position_capacity, task, k2, epsilon))
                operator.append(audit_operator_cell(root, repo, task, k2, epsilon))
    require(len(quality) == len(operator) == 40, (len(quality), len(operator)))
    for task in TASKS:
        selected = [row for row in quality if row["task"] == task]
        require(len({row["ordered_input_sha256"] for row in selected}) == 1, task)
        require(len({json.dumps(row["per_sample_input_sha256"], sort_keys=True)
                     for row in selected}) == 1, task)
        selected_operator = [row for row in operator if row["task"] == task]
        identities = {(row["input"]["sha256"], row["input"]["material_sha256"],
                       row["input"]["suffix_sha256"], row["input"]["actual_prompt_tokens"])
                      for row in selected_operator}
        require(len(identities) == 1, (task, identities))
        source_path = Path(args.longbench_data) / f"{task}.jsonl"
        for row in selected_operator:
            require(row["input"]["sha256"] == sha256(source_path), source_path)

    arms = []
    for epsilon in EPS_VALUES:
        for k2 in K2_VALUES:
            quality_rows = [row for row in quality
                            if row["level2_topk"] == k2 and row["gate_epsilon"] == epsilon]
            operator_rows = [row for row in operator
                             if row["level2_topk"] == k2 and row["gate_epsilon"] == epsilon]
            gate_counts = {key: sum(row["gate_counts"][key] for row in quality_rows)
                           for key in ("keep", "near", "far", "both")}
            arms.append({
                "level2_topk": k2,
                "gate_epsilon": epsilon,
                "arm": arm_name(k2, epsilon),
                "quality": {row["task"]: {
                    "official_score": row["official_score"],
                    "recomputed_mean_score": row["recomputed_mean_score"],
                    "rows": row["rows"],
                    "gate_counts": row["gate_counts"],
                } for row in quality_rows},
                "gate_counts": gate_counts,
                "operator": {row["task"]: {
                    key: row["operator_summary"][key]["mean_ms_per_step"]
                    for key in ("l1_ms", "l2_ms", "main_attention_ms", "nfa_operator_total_ms")
                } for row in operator_rows},
            })

    excluded = {output, evidence_path}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in excluded or path.suffix == ".lock":
            continue
        files.append({"path": str(path.relative_to(root)),
                      "bytes": path.stat().st_size, "sha256": sha256(path)})
    evidence = {
        "schema": "nfa-gate-epsilon-k2-grid-evidence-manifest-v1",
        "producer": PRODUCER,
        "verification": "pending-independent-verification",
        "files": files,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "nfa-gate-epsilon-k2-grid-production-audit-v1",
        "status": "pass-production-side",
        "verification": "pending-independent-verification",
        "formal_result": True,
        "producer": PRODUCER,
        "model": "Qwen3-8B",
        "configuration": {
            "subspace": "tail32", "gate": "on", "gate_epsilon": list(EPS_VALUES),
            "observation_rows": 256, "level1_topk": 128,
            "level2_topk": list(K2_VALUES), "page_size": 64,
            "near_pages": 32, "far_pages": 96, "near_tokens": "off",
            "near_method": "avg", "far_method": "minmax",
            "quality_execution": "eager", "quality_input_cap": 31500,
            "quality_max_new_tokens": MAX_NEW,
            "operator_execution": "eager", "operator_context_length": 32768,
            "operator_batch_size": 1, "operator_sample_index": 0,
            "operator_warmup_steps": 3, "operator_profile_repeat": 5,
            "operator_decode_steps": 32,
        },
        "scope": {
            "quality": "LongBench 官方 evaluator 与逐样本 sample_score，200源请求/格",
            "operator": "独立 nsys eager L1+L2 与主 attention；不是 clean TPOT",
            "host_overhead_derivable_by_subtraction": False,
        },
        "quality_cells": quality,
        "operator_cells": operator,
        "arms": arms,
        "quality_requests": sum(row["rows"] for row in quality),
        "paired_quality_input_identity": "pass",
        "paired_operator_input_identity": "pass",
        "reused_previous_k2_1024": False,
        "reuse_note": "新增真实 epsilon×K2 配置链改变 harness 身份；20 组合在同一 producer 下闭集，不跨身份拼曲线。",
        "evidence_manifest": str(evidence_path.relative_to(root)),
        "evidence_manifest_sha256": sha256(evidence_path),
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"quality_cells", "operator_cells", "arms"}},
                     ensure_ascii=False, indent=2))
    for arm in arms:
        print(arm["gate_epsilon"], arm["level2_topk"], arm["gate_counts"],
              {task: row["official_score"] for task, row in arm["quality"].items()},
              arm["operator"])
    print(output)


if __name__ == "__main__":
    main()
