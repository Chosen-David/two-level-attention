#!/usr/bin/env python3
"""GLM LongBench 英文 16 项九分片独立正式审计。"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

AUDIT_MODULE = Path("/tmp/audit_glm_subspace_full_20261106.py")
spec = importlib.util.spec_from_file_location("glm_audit_common", AUDIT_MODULE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
EXPECTED_COUNTS = {
    "narrativeqa": 200, "qasper": 200, "multifieldqa_en": 150,
    "hotpotqa": 200, "2wikimqa": 200, "musique": 200,
    "gov_report": 200, "qmsum": 200, "multi_news": 200,
    "trec": 200, "triviaqa": 200, "samsum": 200,
    "passage_count": 200, "passage_retrieval_en": 200,
    "lcc": 500, "repobench-p": 500,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=9)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--expected-subspace", default="glm_nope_random32")
    parser.add_argument("--expected-subspace-seed", type=int, default=1)
    parser.add_argument("--data-root", default="/home/wangyuanshuo02/datasets/LongBench/data")
    parser.add_argument("--repo", default="/tmp/tli-glm-jssa-prod-20261106")
    args = parser.parse_args()

    root = Path(args.run_dir)
    failures = []
    require = common.require
    manifests = []
    for rank in range(args.world_size):
        path = root / f"manifest_rank{rank}.json"
        require(path.is_file(), f"missing {path.name}", failures)
        if path.is_file():
            try:
                manifest = json.loads(path.read_text())
                require(manifest.get("rank") == rank, f"manifest rank mismatch {rank}", failures)
                manifests.append(manifest)
            except Exception as exc:
                failures.append(f"manifest {rank}: {type(exc).__name__}: {exc}")
        done = root / f"worker_done_rank{rank}.json"
        require(done.is_file(), f"missing {done.name}", failures)
    require(len(manifests) == args.world_size, "not all manifests readable", failures)
    if not manifests:
        result = {"status": "FAIL", "failures": failures}
        common.write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1

    identity = manifests[0].get("identity")
    protocol = manifests[0].get("protocol", {})
    require(identity == args.expected_identity, "identity != expected", failures)
    require(all(m.get("identity") == identity for m in manifests), "rank identity mismatch", failures)
    require(all(m.get("protocol") == protocol for m in manifests), "rank protocol mismatch", failures)
    for rank in range(args.world_size):
        done = root / f"worker_done_rank{rank}.json"
        if done.is_file():
            try:
                require(json.loads(done.read_text()).get("identity") == identity,
                        f"worker done identity mismatch rank={rank}", failures)
            except Exception as exc:
                failures.append(f"worker done {rank}: {type(exc).__name__}: {exc}")

    require(protocol.get("schema") == "frozen-near-far-screen-v4-material-suffix",
            "protocol schema mismatch", failures)
    require(protocol.get("seed") == 42, "seed != 42", failures)
    require(protocol.get("samples_per_task") is None, "samples_per_task is not full", failures)
    require(protocol.get("full_dataset") is True, "full_dataset flag absent", failures)
    require(protocol.get("context_cap") == 31500, "context cap != 31500", failures)
    require(protocol.get("thresholds") == [], "thresholds not empty", failures)
    require(protocol.get("datasets") == DATASETS, "dataset order/protocol mismatch", failures)
    require(protocol.get("selected_counts") == EXPECTED_COUNTS, "selected counts mismatch", failures)
    require(protocol.get("world_size") == args.world_size, "world size mismatch", failures)
    require(protocol.get("generation_protocol", "").startswith("material prefill"),
            "material/suffix protocol missing", failures)
    nfa = protocol.get("nfa_args", {})
    require(nfa.get("nfa_subspace") == args.expected_subspace, "subspace mismatch", failures)
    require(nfa.get("nfa_subspace_seed") == args.expected_subspace_seed, "subspace seed mismatch", failures)
    require(protocol.get("nfa_environment", {}).get("NFA_DECODE_DENSE") == "0",
            "NFA decode dense fallback enabled", failures)

    dataset_sha = {}
    for dataset in DATASETS:
        source = Path(args.data_root) / f"{dataset}.jsonl"
        require(source.is_file(), f"missing dataset {source}", failures)
        if source.is_file():
            dataset_sha[dataset] = common.sha_bytes(source)
            require(protocol.get("dataset_sha256", {}).get(dataset) == dataset_sha[dataset],
                    f"dataset SHA mismatch {dataset}", failures)
        expected_rows = protocol.get("selected_rows", {}).get(dataset, [])
        require(sorted(expected_rows) == list(range(EXPECTED_COUNTS[dataset])),
                f"full rows mismatch {dataset}", failures)

    source_sha = {}
    for relative, expected_sha in protocol.get("source_sha256", {}).items():
        path = Path(args.repo) / relative
        require(path.is_file(), f"missing source {relative}", failures)
        if path.is_file():
            actual_sha = common.sha_bytes(path)
            source_sha[relative] = actual_sha
            require(actual_sha == expected_sha, f"source SHA mismatch {relative}", failures)

    tasks = {}
    rank_request_counts = {str(rank): 0 for rank in range(args.world_size)}
    all_full = []
    all_native = []
    dataset_offset = 0
    for dataset in DATASETS:
        expected_rows = protocol.get("selected_rows", {}).get(dataset, [])
        expected_files = {f"{row:04d}.json" for row in expected_rows}
        request_dir = root / "requests" / dataset
        actual_files = {path.name for path in request_dir.glob("*.json")} if request_dir.is_dir() else set()
        require(actual_files == expected_files,
                f"request file set mismatch {dataset}: missing={len(expected_files-actual_files)} extra={len(actual_files-expected_files)}",
                failures)
        rows = []
        for position, row_index in enumerate(expected_rows):
            expected_rank = (dataset_offset + position) % args.world_size
            path = request_dir / f"{row_index:04d}.json"
            if not path.is_file():
                failures.append(f"missing request {dataset}:{row_index}")
                continue
            try:
                row = json.loads(path.read_text())
                require(row.get("identity") == identity, f"identity {dataset}:{row_index}", failures)
                require(row.get("status") == "complete", f"status {dataset}:{row_index}", failures)
                require(row.get("row_index") == row_index, f"row index {dataset}:{row_index}", failures)
                require(set(row.get("arms", {})) == {"full", "native_full"},
                        f"arms {dataset}:{row_index}", failures)
                info = row["input"]
                require(len(info.get("input_sha256", "")) == 64,
                        f"input SHA {dataset}:{row_index}", failures)
                require(info.get("input_tokens") == info.get("material_tokens", 0) + info.get("suffix_tokens", 0),
                        f"material/suffix token sum {dataset}:{row_index}", failures)
                full = row["arms"]["full"]
                native = row["arms"]["native_full"]
                require(full.get("request_phase") == "FINISHED",
                        f"request phase {dataset}:{row_index}", failures)
                require(full.get("material_length") == info.get("material_tokens"),
                        f"material length {dataset}:{row_index}", failures)
                require(full.get("suffix_length") == info.get("suffix_tokens"),
                        f"suffix length {dataset}:{row_index}", failures)
                require(full.get("gate_counts") == {"keep": 47, "near": 0, "far": 0, "both": 0},
                        f"47-layer keep gates {dataset}:{row_index}", failures)
                require(len(full.get("nfa_receipts", {})) == 47,
                        f"NFA receipts {dataset}:{row_index}", failures)
                require(len(full.get("prefill_full_kv", {}).get("layers", [])) == 47,
                        f"prefill layers {dataset}:{row_index}", failures)
                require(len(full.get("final_kv", {}).get("layers", [])) == 47,
                        f"final layers {dataset}:{row_index}", failures)
                decisions = full.get("decisions", {})
                require(len(decisions) == 47 and all(
                    not decision.get("skip_far") and not decision.get("skip_near")
                    for decision in decisions.values()),
                    f"full-region decisions {dataset}:{row_index}", failures)
                require(native.get("native_full") is True,
                        f"native flag {dataset}:{row_index}", failures)
                for arm in (full, native):
                    require(isinstance(arm.get("generated_ids"), list),
                            f"generated IDs {dataset}:{row_index}", failures)
                    require(isinstance(arm.get("prediction"), str),
                            f"prediction {dataset}:{row_index}", failures)
                    require(isinstance(arm.get("score"), (int, float)) and math.isfinite(arm["score"]),
                            f"score {dataset}:{row_index}", failures)
                rows.append(row)
                rank_request_counts[str(expected_rank)] += 1
            except Exception as exc:
                failures.append(f"request {dataset}:{row_index}: {type(exc).__name__}: {exc}")
        rows.sort(key=lambda row: row["row_index"])
        require(len(rows) == len(expected_rows), f"incomplete {dataset}", failures)
        full_scores = [row["arms"]["full"]["score"] for row in rows]
        native_scores = [row["arms"]["native_full"]["score"] for row in rows]
        all_full.extend(full_scores)
        all_native.extend(native_scores)
        tasks[dataset] = {
            "n": len(rows),
            "full": sum(full_scores) / len(full_scores) if full_scores else None,
            "native_full": sum(native_scores) / len(native_scores) if native_scores else None,
            "full_prediction_sha256": common.sha_payload([
                [row["row_index"], row["arms"]["full"]["prediction"]] for row in rows]),
            "full_generated_ids_sha256": common.sha_payload([
                [row["row_index"], row["arms"]["full"]["generated_ids"]] for row in rows]),
            "native_prediction_sha256": common.sha_payload([
                [row["row_index"], row["arms"]["native_full"]["prediction"]] for row in rows]),
            "native_generated_ids_sha256": common.sha_payload([
                [row["row_index"], row["arms"]["native_full"]["generated_ids"]] for row in rows]),
        }
        dataset_offset += len(expected_rows)

    expected_rank_counts = {str(rank): 0 for rank in range(args.world_size)}
    dataset_offset = 0
    for dataset in DATASETS:
        rows = protocol.get("selected_rows", {}).get(dataset, [])
        for position in range(len(rows)):
            expected_rank = (dataset_offset + position) % args.world_size
            expected_rank_counts[str(expected_rank)] += 1
        dataset_offset += len(rows)
    require(rank_request_counts == expected_rank_counts,
            f"rank request counts mismatch: actual={rank_request_counts} expected={expected_rank_counts}", failures)
    require(sum(rank_request_counts.values()) == 3750, "total requests != 3750", failures)
    complete_scores = all(task["full"] is not None and task["native_full"] is not None for task in tasks.values())
    result = {
        "schema": "glm_longbench_16en_full_audit_v1",
        "run_dir": str(root),
        "identity": identity,
        "expected_subspace": args.expected_subspace,
        "expected_subspace_seed": args.expected_subspace_seed,
        "dataset_sha256": dataset_sha,
        "source_sha256": source_sha,
        "rank_request_counts": rank_request_counts,
        "tasks": tasks,
        "macro_full": sum(task["full"] for task in tasks.values()) / len(DATASETS) if complete_scores else None,
        "macro_native_full": sum(task["native_full"] for task in tasks.values()) / len(DATASETS) if complete_scores else None,
        "micro_full": sum(all_full) / len(all_full) if len(all_full) == 3750 else None,
        "micro_native_full": sum(all_native) / len(all_native) if len(all_native) == 3750 else None,
        "native_sha_order": "ascending row_index; JSON array of [row_index,prediction]",
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    common.write_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "tasks": tasks,
        "macro_full": result["macro_full"], "macro_native_full": result["macro_native_full"],
        "rank_request_counts": rank_request_counts, "failures": failures,
    }, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
