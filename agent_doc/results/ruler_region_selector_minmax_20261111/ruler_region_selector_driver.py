#!/usr/bin/env python3
"""单卡串行执行冻结的 RULER cell；只跳过身份闭合的完整 generation。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK_BUDGET = {
    "niah_single_1": 128, "niah_single_2": 128, "niah_single_3": 128,
    "niah_multikey_1": 128, "niah_multikey_2": 128, "niah_multikey_3": 128,
    "niah_multiquery": 128, "niah_multivalue": 128,
    "cwe": 120, "fwe": 50, "vt": 30,
}
ARMS = {"fullkv", "single_avg", "single_minmax", "nfa"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def gpu_used_mib(gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    return int(output.strip().splitlines()[0])


def arm_flags(arm):
    if arm == "fullkv":
        return ["--method", "none"]
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    flags = [
        "--method", "nfa",
        "--tia_block_size", "64", "--tia_level1_topk", "128",
        "--tia_level2_topk", "1024", "--tia_level2_cmp_ratio", "4",
        "--nfa_subspace", "tail", "--nfa_l2_subspace", "same",
        "--nfa_sink_tokens", "4", "--nfa_swa_size", "128",
        "--nfa_enable_subspace", "true", "--nfa_enable_kmeans", "false",
        "--nfa_enable_layer_skip", "false", "--nfa_per_q_head", "false",
        "--nfa_far_select", "4bit", "--nfa_near_select", "4bit",
        "--nfa_boundary_bypass", "true", "--nfa_l2_quant", "false",
        "--nfa_far_skip_backend", "legacy", "--nfa_far_skip_cache", "retain",
    ]
    if arm == "nfa":
        flags += [
            "--nfa_far_method", "minmax", "--nfa_near_method", "avg",
            "--nfa_near_pages", "32", "--nfa_near_tokens", "off",
            "--nfa_near_len", "4096",
        ]
    else:
        flags += [
            "--nfa_far_method", "avg" if arm == "single_avg" else "minmax",
            "--nfa_near_method", "avg", "--nfa_near_pages", "0",
            "--nfa_near_tokens", "off", "--nfa_near_len", "0",
        ]
    flags += ["--nfa_prefill_frozen_regions",
              "--nfa_frozen_region_mode", "full"]
    return flags


def expected_treatment(arm):
    common = {
        "tia_block_size": 64, "tia_level1_topk": 128,
        "tia_level2_topk": 1024, "tia_level2_cmp_ratio": 4,
        "nfa_subspace": "tail", "nfa_l2_subspace": "same",
        "nfa_sink_tokens": 4, "nfa_swa_size": 128,
        "nfa_enable_subspace": True, "nfa_enable_kmeans": False,
        "nfa_enable_layer_skip": False, "nfa_dyn_dgate": False,
        "nfa_per_q_head": False, "nfa_far_select": "4bit",
        "nfa_near_select": "4bit", "nfa_boundary_bypass": True,
        "nfa_l2_quant": False, "nfa_far_skip_backend": "legacy",
        "nfa_far_skip_cache": "retain", "nfa_layer_skip_path": None,
    }
    if arm == "nfa":
        common.update(nfa_far_method="minmax", nfa_near_method="avg",
                      nfa_near_pages=32, nfa_near_tokens="off", nfa_near_len=4096)
    elif arm != "fullkv":
        common.update(
            nfa_far_method="avg" if arm == "single_avg" else "minmax",
            nfa_near_method="avg", nfa_near_pages=0,
            nfa_near_tokens="off", nfa_near_len=0)
    return common


def validate_complete(job, paths, run_config_sha, probe_module, receipt_module):
    out_dir = paths["out_dir"]
    probe = probe_module.probe_task(str(out_dir), job["task"], job["max_num"])
    if probe["state"] != "complete":
        return probe
    candidates = probe_module._discover_candidates(str(out_dir), job["task"])
    info = candidates[probe["base"]]
    if not info["pointer"]:
        raise RuntimeError(f"legacy output is not accepted: {info['logical']}")
    generation = receipt_module.resolve_generation_pointer(info["logical"])
    receipt = json.loads(Path(generation["rcp_path"]).read_text(encoding="utf-8"))
    if receipt.get("run_config_sha256") != run_config_sha:
        raise RuntimeError(f"run config identity mismatch: {generation['rcp_path']}")
    params = receipt["generation_params"]
    expected = {
        "max_gen": TASK_BUDGET[job["task"]], "min_new_tokens": 1,
        "max_num": job["max_num"], "seed": 42,
        "method": "none" if job["arm"] == "fullkv" else "nfa",
        "pred_postfix": paths["postfix"], "t": paths["t"],
        "graph_policy": "off" if job["arm"] == "fullkv" else "auto",
        "operator_mode": "off",
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise RuntimeError(f"generation {key} mismatch: {params.get(key)!r} != {value!r}")
    if receipt.get("context_length") != job["length"] or receipt.get("task") != job["task"]:
        raise RuntimeError("receipt cell identity mismatch")
    expected_yarn = job["length"] > 40960
    if receipt.get("yarn_enabled") is not expected_yarn:
        raise RuntimeError("YaRN enable mismatch")
    if expected_yarn and receipt.get("effective_yarn_factor") != (2.0 if job["length"] == 65536 else 4.0):
        raise RuntimeError("YaRN factor mismatch")
    if job["arm"] != "fullkv":
        treatment = json.loads(receipt["treatment_manifest"]["json"])
        for key, value in expected_treatment(job["arm"]).items():
            # info.py intentionally omits default far-skip modes to preserve
            # historical treatment identities; omission means the default.
            if key not in treatment and key in (
                    "nfa_far_skip_backend", "nfa_far_skip_cache"):
                actual = value
            else:
                actual = treatment.get(key)
            if actual != value:
                raise RuntimeError(f"treatment {key} mismatch: {actual!r} != {value!r}")
        frozen = treatment.get("nfa_prefill_frozen_regions")
        if not isinstance(frozen, dict) or frozen.get("gate", {}).get("mode") != "full":
            raise RuntimeError("sparse arm is not bound to full/no-skip gate")
        expected_near = 4096 if job["arm"] == "nfa" else 64
        if frozen.get("geometry", {}).get("near") != expected_near:
            raise RuntimeError("frozen cache near geometry mismatch")
    with Path(generation["pred_path"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != job["max_num"]:
        raise RuntimeError("prediction population mismatch")
    for row in rows:
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= TASK_BUDGET[job["task"]]:
            raise RuntimeError(f"invalid generated token budget: {row.get('_id')}")
        execution = row.get("execution", {})
        expected_execution = "eager" if job["arm"] == "fullkv" else "graph"
        if (execution.get("requested") != expected_execution
                or execution.get("actual") != expected_execution):
            raise RuntimeError(f"execution path mismatch: {row.get('_id')}")
        if job["arm"] != "fullkv":
            audit = row.get("_frozen_region")
            if not isinstance(audit, dict) or audit.get("request_phase") != "FINISHED":
                raise RuntimeError(f"missing finished frozen audit: {row.get('_id')}")
            counts = audit.get("gate_counts", {})
            decisions = audit.get("decisions", {})
            if (not decisions or counts.get("keep") != len(decisions)
                    or any(counts.get(name) != 0 for name in ("near", "far", "both"))):
                raise RuntimeError(f"non-keep gate decision: {row.get('_id')}: {counts}")
            if any(decision.get("skip_near") or decision.get("skip_far")
                   for decision in decisions.values()):
                raise RuntimeError(f"skip flag enabled: {row.get('_id')}")
    probe["receipt_path"] = generation["rcp_path"]
    probe["pred_path"] = generation["pred_path"]
    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--t", required=True)
    parser.add_argument("--postfix", required=True)
    parser.add_argument("--max-used-mib", type=int, default=2048)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    from benchmark.RULER import gen_completion_probe as probe_module
    from benchmark.RULER import yarn_receipt as receipt_module

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list")
    run_config_sha = sha256_file(args.run_config)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status = {
        "schema": "ruler-region-selector-driver-status-v1",
        "state": "running", "driver_pid": os.getpid(),
        "physical_gpu": args.physical_gpu, "jobs_total": len(jobs),
        "jobs_complete": 0, "current": None, "cells": [],
        "run_config_sha256": run_config_sha,
    }
    atomic_json(args.status, status)
    for index, job in enumerate(jobs):
        if set(job) != {"length", "arm", "task", "max_num"}:
            raise ValueError(f"bad job schema: {job}")
        if job["arm"] not in ARMS or job["task"] not in TASK_BUDGET:
            raise ValueError(f"bad job: {job}")
        paths = {"postfix": args.postfix, "t": args.t}
        paths["out_dir"] = (args.output_root / job["arm"] / f"L{job['length']}" /
                            f"pred{args.postfix}")
        probe = validate_complete(job, paths, run_config_sha, probe_module, receipt_module)
        if probe["state"] == "complete":
            record = {**job, "state": "complete", "action": "skip",
                      "prediction": probe["pred_path"], "receipt": probe["receipt_path"]}
            status["cells"].append(record)
            status["jobs_complete"] += 1
            atomic_json(args.status, status)
            print(f"CELL_COMPLETE action=skip {job}", flush=True)
            continue
        used = gpu_used_mib(args.physical_gpu)
        if used > args.max_used_mib:
            raise RuntimeError(f"GPU{args.physical_gpu} used={used} MiB > {args.max_used_mib}; refuse to share")
        log_path = args.log_dir / f"{index:03d}_{job['length']}_{job['arm']}_{job['task']}.log"
        module = ("benchmark.RULER.pred_ruler" if job["arm"] == "fullkv"
                  else "benchmark.RULER.pred_ruler_region_selector")
        command = [
            args.python, "-u", "-m", module,
            "--model", "Qwen3-8B", "--model_path", str(args.model_path),
            "--task", job["task"], "--context_length", str(job["length"]),
            "--t", args.t, "--data-root", str(args.data_root),
            "--output-dir", str(args.output_root / job["arm"]),
            "--pred_postfix", args.postfix, "--skip", "0",
            "--max-num", str(job["max_num"]),
        ] + arm_flags(job["arm"])
        if job["length"] > 40960:
            command += ["--yarn", "--yarn_factor", "2.0" if job["length"] == 65536 else "4.0"]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "PYTHONPATH": str(Path.home() / "local_pylibs") + os.pathsep + str(source_root),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NFA_RECEIPT_RUN_CONFIG": str(args.run_config.resolve()),
            "NFA_CUDA_GRAPH": "off", "NFA_OPERATOR_MODE": "off",
            "NFA_FINE_GATHER": "0", "NFA_SPARSE_PREFILL": "0",
            "NFA_DECODE_DENSE": "0", "NFA_FROZEN_EAGER_REFERENCE": "0",
        })
        current = {**job, "state": "running", "log": str(log_path),
                   "command": command, "started_at": datetime.datetime.now().isoformat(),
                   "gpu_used_mib_before": used}
        status["current"] = current
        atomic_json(args.status, status)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=source_root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            current["producer_pid"] = process.pid
            atomic_json(args.status, status)
            return_code = process.wait()
        current["return_code"] = return_code
        current["finished_at"] = datetime.datetime.now().isoformat()
        current["gpu_used_mib_after"] = gpu_used_mib(args.physical_gpu)
        if return_code != 0:
            current["state"] = "failed"
            status["state"] = "failed"
            status["current"] = current
            atomic_json(args.status, status)
            print(f"CELL_FAILED {job} rc={return_code} log={log_path}", flush=True)
            raise SystemExit(return_code)
        probe = validate_complete(job, paths, run_config_sha, probe_module, receipt_module)
        if probe["state"] != "complete":
            raise RuntimeError(f"producer returned zero but cell incomplete: {job}: {probe}")
        record = {**job, "state": "complete", "action": "produced",
                  "prediction": probe["pred_path"], "receipt": probe["receipt_path"],
                  "log": str(log_path), "started_at": current["started_at"],
                  "finished_at": current["finished_at"]}
        status["cells"].append(record)
        status["jobs_complete"] += 1
        status["current"] = None
        atomic_json(args.status, status)
        print(f"CELL_COMPLETE action=produced {job}", flush=True)
    status["state"] = "complete"
    status["finished_at"] = datetime.datetime.now().isoformat()
    atomic_json(args.status, status)
    print(f"DRIVER_COMPLETE jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
