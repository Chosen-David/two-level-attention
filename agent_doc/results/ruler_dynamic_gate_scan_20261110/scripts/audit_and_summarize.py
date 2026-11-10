"""独立只读审计 33 格 quick scan，并从原始 profiles/decisions 重建汇总。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

TASKS = [
    "cwe", "fwe", "niah_multikey_1", "niah_multikey_2",
    "niah_multikey_3", "niah_multiquery", "niah_multivalue",
    "niah_single_1", "niah_single_2", "niah_single_3", "vt",
]
LENGTHS = [32768, 65536, 131072]
HEAD = "7798ddb6569071d1238ca21e7ddf925a18d8dcd5"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
GATE = {"epsilon": 0.05, "rows": 256, "statistic": "mean", "mode": "dual"}
SELECTOR = {
    "nfa_far_method": "minmax", "nfa_near_method": "avg",
    "nfa_near_pages": 32, "nfa_near_tokens": "off", "nfa_near_len": 4096,
    "nfa_subspace": "tail", "tia_level2_cmp_ratio": 4,
    "tia_block_size": 64, "tia_level1_topk": 128, "tia_level2_topk": 1024,
    "nfa_enable_layer_skip": False, "nfa_far_skip_backend": "legacy",
    "nfa_far_skip_cache": "retain",
}


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"nonfinite value at {path}")


def legacy_compare(length, rows, legacy_path):
    if not legacy_path.exists():
        return {"path": str(legacy_path), "status": "not-found"}
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    old = {(row["task"], row["sample_index"]): row for row in legacy}
    new = {(row["task"], row["sample_index"]): row for row in rows}
    require(set(old) == set(new), f"legacy {length} cell set mismatch")
    skip_mismatches = []
    identity_mismatches = []
    max_abs_mean_delta = 0.0
    for key in sorted(new):
        before, after = old[key], new[key]
        for field in ("repo_head", "context_length", "input_sha256", "prompt_tokens"):
            if before[field] != after[field]:
                identity_mismatches.append({"cell": list(key), "field": field})
        for region in ("far", "near"):
            field = f"skip_{region}_layers"
            if before[field] != after[field]:
                skip_mismatches.append({
                    "cell": list(key), "region": region,
                    "legacy": before[field], "current": after[field],
                })
        for layer in range(36):
            for region in ("far", "near"):
                delta = abs(float(before["profiles"][str(layer)][region]["mean"])
                            - float(after["profiles"][str(layer)][region]["mean"]))
                max_abs_mean_delta = max(max_abs_mean_delta, delta)
    return {
        "path": str(legacy_path), "sha256": sha256_file(legacy_path),
        "status": "compared", "cells": len(new),
        "identity_mismatches": identity_mismatches,
        "skip_decision_mismatches": skip_mismatches,
        "max_abs_profile_mean_delta": max_abs_mean_delta,
        "note": "legacy files lack runtime source-diff identity and are cross-checks, not the primary dataset",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(
        "/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER"))
    parser.add_argument("--model", type=Path, default=Path("/home/wangyuanshuo02/Qwen3-8B"))
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = root / "raw"
    derived = root / "derived"
    validation = root / "validation"
    derived.mkdir(exist_ok=True)
    validation.mkdir(exist_ok=True)
    errors = []
    all_rows = []
    raw_hashes = {}
    try:
        require(sha256_file(args.model / "config.json") is not None, "model config unreadable")
        expected_model_config = sha256_file(args.model / "config.json")
        for length in LENGTHS:
            length_rows = []
            expected_inputs = [
                (raw_root / f"ruler_{length}_task_{task}.json", [task])
                for task in TASKS
            ]
            for path, expected_tasks in expected_inputs:
                require(path.is_file(), f"missing {path.name}")
                raw_hashes[path.name] = sha256_file(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                require(payload["schema"] == "ruler-dynamic-gate-quick-scan-v2", "schema mismatch")
                require(payload["complete"] is True, f"{path.name} incomplete")
                require(payload["expected_cells"] == len(expected_tasks), "expected count mismatch")
                require(payload["realized_cells"] == len(expected_tasks), "realized count mismatch")
                require(payload["assigned_tasks"] == expected_tasks, "shard assignment mismatch")
                require(payload["context_length"] == length, "context mismatch")
                require(payload["sample_indices"] == [0], "sample protocol mismatch")
                require(payload["gate"] == GATE, "gate mismatch")
                require(payload["freeze_mode"] == "decision-only-no-reclaim", "freeze mode mismatch")
                require(payload["selector"] == SELECTOR, "selector mismatch")
                require(payload["model_config_sha256"] == expected_model_config, "model config mismatch")
                source = payload["source"]
                require(source["repo_head"] == HEAD, "source HEAD mismatch")
                require(source["worktree_diff_sha256"] == EMPTY_SHA256, "dirty worktree")
                require(source["index_diff_sha256"] == EMPTY_SHA256, "dirty index")
                require(source["worktree_diff_bytes"] == 0, "nonempty worktree diff")
                require(source["index_diff_bytes"] == 0, "nonempty index diff")
                require(source["untracked_files"] == [], "untracked source files")
                geometry = payload["geometry"]
                for key, expected in (("sink", 4), ("near", 4096), ("window", 128),
                                      ("page", 64), ("max_new_tokens", 128)):
                    require(geometry[key] == expected, f"geometry {key} mismatch")
                rows = payload["results"]
                require(len(rows) == len(expected_tasks), "row count mismatch")
                require({row["task"] for row in rows} == set(expected_tasks), "task set mismatch")
                length_rows.extend(rows)
            require(len(length_rows) == 11, f"length {length} does not have 11 rows")
            require({row["task"] for row in length_rows} == set(TASKS), "duplicate/missing task")
            for row in length_rows:
                task = row["task"]
                require(row["sample_index"] == 0, "nonzero sample index")
                require(row["repo_head"] == HEAD and row["context_length"] == length, "row identity mismatch")
                require(row["freeze_mode"] == "decision-only-no-reclaim", "row freeze mode mismatch")
                require(row["gate"] == GATE and row["selector"] == SELECTOR, "row config mismatch")
                input_path = args.data_root / str(length) / f"{task}.jsonl"
                require(sha256_file(input_path) == row["input_file_sha256"], f"{length}/{task} input file hash mismatch")
                with input_path.open(encoding="utf-8") as handle:
                    line = handle.readline()
                source_row = json.loads(line)
                require(hashlib.sha256(line.encode("utf-8")).hexdigest() == row["raw_line_sha256"], "raw line hash mismatch")
                require(hashlib.sha256(source_row["input"].encode("utf-8")).hexdigest() == row["input_sha256"], "input hash mismatch")
                require(source_row.get("id", 0) == row["sample_id"], "sample id mismatch")
                require(0 < row["prompt_tokens"] <= length, "prompt token bound mismatch")
                require(set(row["profiles"]) == {str(layer) for layer in range(36)}, "profile layers mismatch")
                require(set(row["decisions"]) == {str(layer) for layer in range(36)}, "decision layers mismatch")
                finite_tree(row["profiles"], f"{length}/{task}.profiles")
                far, near = [], []
                for layer in range(36):
                    profile = row["profiles"][str(layer)]
                    decision = row["decisions"][str(layer)]
                    require(set(decision) == {"skip_near", "skip_far"}, "decision schema mismatch")
                    expected_near = float(profile["near"]["mean"]) < GATE["epsilon"]
                    expected_far = float(profile["far"]["mean"]) < GATE["epsilon"]
                    require(decision["skip_near"] is expected_near, f"{length}/{task}/L{layer} near decision mismatch")
                    require(decision["skip_far"] is expected_far, f"{length}/{task}/L{layer} far decision mismatch")
                    if decision["skip_near"]:
                        near.append(layer)
                    if decision["skip_far"]:
                        far.append(layer)
                require(row["skip_near_layers"] == near, "near layer list mismatch")
                require(row["skip_far_layers"] == far, "far layer list mismatch")
                require(row["skip_both_layers"] == sorted(set(near) & set(far)), "both layer list mismatch")
                for region in ("near", "far"):
                    means = [float(row["profiles"][str(layer)][region]["mean"]) for layer in range(36)]
                    require(row[f"{region}_mean_min"] == min(means), f"{region} min mismatch")
                    require(row[f"{region}_mean_max"] == max(means), f"{region} max mismatch")
                    require(int(row[f"{region}_mean_min_layer"]) == means.index(min(means)), f"{region} min layer mismatch")
                all_rows.append(row)
        require(len(all_rows) == 33, "not exactly 33 cells")
        require(len({(row["context_length"], row["task"]) for row in all_rows}) == 33, "duplicate cells")
        require(sum(len(row["profiles"]) for row in all_rows) == 1188, "not exactly 1188 layer profiles")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    cells = []
    totals = {}
    for length in LENGTHS:
        rows = sorted((row for row in all_rows if row["context_length"] == length),
                      key=lambda row: TASKS.index(row["task"]))
        histogram = {region: {str(layer): 0 for layer in range(36)}
                     for region in ("far", "near", "both")}
        for row in rows:
            both = sorted(set(row["skip_far_layers"]) & set(row["skip_near_layers"]))
            cells.append({
                "context_length": length, "task": row["task"], "sample_index": 0,
                "prompt_tokens": row["prompt_tokens"], "input_sha256": row["input_sha256"],
                "skip_far_layers": row["skip_far_layers"],
                "skip_near_layers": row["skip_near_layers"],
                "skip_both_layers": both,
                "far_count": len(row["skip_far_layers"]),
                "near_count": len(row["skip_near_layers"]), "both_count": len(both),
                "near_mean_min": row["near_mean_min"],
                "near_mean_min_layer": row["near_mean_min_layer"],
                "far_mean_min": row["far_mean_min"],
                "far_mean_min_layer": row["far_mean_min_layer"],
            })
            for region, key in (("far", "skip_far_layers"), ("near", "skip_near_layers")):
                for layer in row[key]:
                    histogram[region][str(layer)] += 1
            for layer in both:
                histogram["both"][str(layer)] += 1
        totals[str(length)] = {
            "task_count": len(rows),
            "tasks_with_far_skip": sum(bool(row["skip_far_layers"]) for row in rows),
            "tasks_with_near_skip": sum(bool(row["skip_near_layers"]) for row in rows),
            "tasks_with_both_skip": sum(bool(set(row["skip_far_layers"]) & set(row["skip_near_layers"])) for row in rows),
            "far_skip_layer_decisions": sum(len(row["skip_far_layers"]) for row in rows),
            "near_skip_layer_decisions": sum(len(row["skip_near_layers"]) for row in rows),
            "both_skip_layer_decisions": sum(len(set(row["skip_far_layers"]) & set(row["skip_near_layers"])) for row in rows),
            "layer_histogram": histogram,
        }
    legacy = {}
    if not errors:
        for length, name in ((32768, "/tmp/ruler32k_dynamic_gate_scan_current.json"),
                             (65536, "/tmp/ruler64k_dynamic_gate_scan_current.json")):
            legacy[str(length)] = legacy_compare(
                length, [row for row in all_rows if row["context_length"] == length], Path(name))
    summary = {
        "schema": "ruler-dynamic-gate-quick-scan-summary-v1",
        "status": "invalid" if errors else "usable-with-scope",
        "scope": "Qwen3-8B; 11 RULER tasks; sample_index=0 only; first dynamic-gate decision after dense prefill; decision-only freeze without reclaim/decode; no generation and no performance claim",
        "source_head": HEAD, "gate": GATE, "selector": SELECTOR,
        "cell_count": len(cells), "layer_profile_count": sum(
            len(row["profiles"]) for row in all_rows),
        "cells": cells, "totals_by_length": totals,
        "legacy_crosscheck": legacy,
    }
    if not errors:
        atomic_json(derived / "summary.json", summary)
        input_manifest = {
            "schema": "ruler-dynamic-gate-input-manifest-v1",
            "model_path": str(args.model),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "inputs": [{
                "context_length": row["context_length"], "task": row["task"],
                "sample_index": row["sample_index"], "sample_id": row["sample_id"],
                "path": row["input_path"], "file_sha256": row["input_file_sha256"],
                "raw_line_sha256": row["raw_line_sha256"],
                "input_sha256": row["input_sha256"],
                "prompt_tokens": row["prompt_tokens"],
            } for row in sorted(all_rows, key=lambda row: (
                row["context_length"], TASKS.index(row["task"])))],
        }
        atomic_json(derived / "input_manifest.json", input_manifest)
        environments = {}
        for length in LENGTHS:
            for task in TASKS:
                payload = json.loads((raw_root / f"ruler_{length}_task_{task}.json").read_text(
                    encoding="utf-8"))
                identity = {
                    "python": payload["python"], "torch": payload["torch"],
                    "transformers": payload["transformers"],
                    "cuda_device": payload["cuda_device"],
                    "source": payload["source"],
                }
                environments.setdefault(json.dumps(identity, sort_keys=True), {
                    **identity, "cells": [],
                })["cells"].append([length, task])
        atomic_json(derived / "environment.json", {
            "schema": "ruler-dynamic-gate-environment-v1",
            "seed": 20261102, "repeats": 1,
            "freeze_mode": "decision-only-no-reclaim",
            "environments": list(environments.values()),
        })
        with (derived / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["context_length", "task", "sample_index", "prompt_tokens",
                             "far_count", "skip_far_layers", "near_count",
                             "skip_near_layers", "both_count", "skip_both_layers",
                             "near_mean_min", "far_mean_min"])
            for cell in cells:
                writer.writerow([
                    cell["context_length"], cell["task"], 0, cell["prompt_tokens"],
                    cell["far_count"], json.dumps(cell["skip_far_layers"], separators=(",", ":")),
                    cell["near_count"], json.dumps(cell["skip_near_layers"], separators=(",", ":")),
                    cell["both_count"], json.dumps(cell["skip_both_layers"], separators=(",", ":")),
                    cell["near_mean_min"], cell["far_mean_min"],
                ])
    report = {
        "schema": "ruler-dynamic-gate-independent-check-v1",
        "status": "invalid" if errors else "usable-with-scope",
        "verifier": {
            "actor": "single-agent-separate-readonly-audit-process",
            "method": "independent JSON/data/hash parser; no producer helper imported; no subagent used",
        },
        "scope": summary["scope"],
        "checks": {
            "implementation": {"verdict": "fail" if errors else "pass", "reason": "checked source/config identities and recomputed every decision from raw profiles without importing producer code"},
            "reference_boundary": {"verdict": "fail" if errors else "pass", "reason": "recomputed near/far threshold decisions and intersections for all 1188 layers"},
            "data_integrity": {"verdict": "fail" if errors else "pass", "reason": "rehashed 33 source files and raw first lines; checked deterministic shard assignment, duplicates and missing cells"},
            "numerical_sanity": {"verdict": "fail" if errors else "pass", "reason": "checked finite profiles, 36 layers per cell, extrema and summary counts"},
            "measurement_validity": {"verdict": "not_applicable", "reason": "mechanism scan only; no latency, throughput, generation-quality or full-sample statistical claim"},
            "reproducibility": {"verdict": "fail" if errors else "pass", "reason": "fixed clean detached HEAD with empty source/index diffs; legacy 32K/64K cross-checks recorded separately"},
        },
        "raw_sha256": raw_hashes,
        "summary_sha256": sha256_file(derived / "summary.json") if not errors else None,
        "errors": errors,
        "limitations": [
            "Each task contributes only sample_index=0; counts are not full-dataset means or quality scores.",
            "Only the first post-prefill gate state is measured; no generated tokens or later decode updates are included.",
            "The scan writes decisions into cache state but deliberately skips physical freeze/reclaim; it does not validate the fixed revision's freeze accounting or decode behavior.",
            "The audit is a separate read-only implementation/process under the project's single-agent constraint, not a second human reviewer.",
            "Legacy 32K/64K files lack runtime source-diff identity and are not accepted as primary evidence.",
        ],
    }
    atomic_json(validation / "independent_check.json", report)
    print(json.dumps({"status": report["status"], "cells": len(cells),
                      "errors": errors, "totals": totals}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
