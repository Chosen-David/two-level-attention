"""把已审计的 33 格结果绑定为 experiment-result/v1 manifest 并复核哈希。"""
import argparse
import hashlib
import json
from pathlib import Path

HEAD = "7798ddb6569071d1238ca21e7ddf925a18d8dcd5"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def ref(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    check_path = root / "validation" / "independent_check.json"
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if check["status"] != "usable-with-scope":
        raise RuntimeError("refusing to package a result that has not passed independent audit")
    summary = json.loads((root / "derived" / "summary.json").read_text(encoding="utf-8"))
    raw = sorted((root / "raw").glob("*.json"))
    if len(raw) != 33:
        raise RuntimeError(f"expected 33 raw files, found {len(raw)}")
    crosscheck_path = root / "validation" / "freeze_mode_crosscheck.json"
    crosscheck = json.loads(crosscheck_path.read_text(encoding="utf-8"))
    if (crosscheck["status"] != "pass" or crosscheck["compared_cells"] != 22
            or crosscheck["skip_mismatches"]
            or crosscheck["max_abs_profile_mean_delta"] != 0.0):
        raise RuntimeError("freeze-mode cross-check did not pass")
    plan_path = root / "validation" / "validation_plan.json"
    execution = {
        "commands": [
            "ssh pc02; CUDA_VISIBLE_DEVICES=<0|1|2|3> bash /tmp/run_ruler_dynamic_gate_decision_only_20261110.sh <same-gpu>",
            "python3 scripts/audit_and_summarize.py --root agent_doc/results/ruler_dynamic_gate_scan_20261110",
            "ssh pc02 'python3 -' < scripts/compare_freeze_modes.py",
            "python3 scripts/package_result.py --root agent_doc/results/ruler_dynamic_gate_scan_20261110",
        ],
        "environment_description": "pc02; four idle NVIDIA H20-3e GPUs; one request per Python process; deterministic task index modulo 4; lengths 131072 then 65536 then 32768; torch=2.8.0+cu128; transformers=4.56.2; details and hashes in derived/environment.json",
        "seeds": [20261102],
        "repeats": 1,
        "freeze_mode": "decision-only-no-reclaim",
        "completed_cells": 33,
    }
    artifacts = {
        "code": [ref(root, root / "scripts" / name) for name in (
            "scan_dynamic_gate.py", "run_shards.sh", "audit_and_summarize.py",
            "package_result.py", "compare_freeze_modes.py")],
        "inputs": [ref(root, root / "derived" / "input_manifest.json")],
        "config": [ref(root, plan_path), ref(root, root / "derived" / "environment.json")],
        "raw_data": [ref(root, path) for path in raw],
        "outputs": [ref(root, root / "derived" / name) for name in (
            "summary.json", "summary.tsv")],
        "environment": [ref(root, root / "derived" / "environment.json")],
        "validation": [ref(root, check_path), ref(root, crosscheck_path)],
    }
    manifest = {
        "schema_version": "experiment-result/v1",
        "result_id": "ruler-dynamic-gate-scan-20261110-v1",
        "run_id": "ruler_dynamic_gate_scan_20261110",
        "producer_task_id": "TL-RULER-DYNAMIC-GATE-SCAN-20261110",
        "producer_actor": "claude-single-agent",
        "code_revision": HEAD,
        "scope": summary["scope"],
        "validation_plan": ref(root, plan_path),
        "execution": execution,
        "artifacts": artifacts,
        "metrics": [
            {"name": "completed_cells", "value": 33, "unit": "cells"},
            {"name": "layer_profiles", "value": 1188, "unit": "layer_profiles"},
            {"name": "tasks_per_length", "value": 11, "unit": "tasks"},
        ],
        "limitations": check["limitations"],
    }
    manifest_path = root / "manifest.json"
    atomic_json(manifest_path, manifest)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = []
    for category, items in loaded["artifacts"].items():
        for item in items:
            path = root / item["path"]
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"hash mismatch: {item['path']}")
            observations.append({"category": category, **item})
    for item in (loaded["validation_plan"],):
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"hash mismatch: {item['path']}")
    manifest_check = {
        "schema": "ruler-dynamic-gate-manifest-check-v1",
        "status": "pass",
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_hash_count": len(observations),
        "artifact_hashes": observations,
        "validation_plan_sha256": sha256_file(plan_path),
        "independent_check_sha256": sha256_file(check_path),
        "freeze_mode_crosscheck_sha256": sha256_file(crosscheck_path),
        "limitations": check["limitations"],
    }
    atomic_json(root / "validation" / "manifest_check.json", manifest_check)
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and path.name not in ("sha256_manifest.tsv", "record.json"))
    with (root / "derived" / "sha256_manifest.tsv").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}\t{path.relative_to(root)}\n")
    atomic_json(root / "record.json", {
        "schema": "ruler-dynamic-gate-result-record-v1",
        "run_id": loaded["run_id"],
        "status": "usable-with-scope",
        "manifest": ref(root, manifest_path),
        "independent_check": ref(root, check_path),
        "manifest_check": ref(root, root / "validation" / "manifest_check.json"),
        "sha256_manifest": ref(root, root / "derived" / "sha256_manifest.tsv"),
        "scope": loaded["scope"],
    })
    print(json.dumps({"status": "usable-with-scope", "raw_files": len(raw),
                      "artifact_hashes": len(observations),
                      "manifest_sha256": sha256_file(manifest_path)}))


if __name__ == "__main__":
    main()
