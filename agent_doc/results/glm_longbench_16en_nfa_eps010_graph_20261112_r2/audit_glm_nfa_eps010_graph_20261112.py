#!/usr/bin/env python3
"""GLM 完整 NFA LongBench16 八分片只读审计；重算输入身份与官方分数。"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATASETS = (
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
)
EXPECTED_COUNTS = {
    **{name: 200 for name in DATASETS},
    "multifieldqa_en": 150,
    "lcc": 500,
    "repobench-p": 500,
}
EXPECTED_RANK_COUNTS = {str(rank): 469 if rank < 6 else 468 for rank in range(8)}
EXPECTED_GIT_HEAD = "7a0f6de8fa4eea1d0e216aaef7b5a8c9275a44fb"
EXPECTED_DIFF_SHA256 = "cd3d83b1e9c02ebd9b8176e478ae57a2dac640137c8380297b4846c0da3c5b85"
ARM = "dual_0.1"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_payload(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n")
    temporary.replace(path)


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def gate_category(decision):
    near = bool(decision.get("skip_near"))
    far = bool(decision.get("skip_far"))
    if near and far:
        return "both"
    if near:
        return "near"
    if far:
        return "far"
    return "keep"


def audit_record(record, source_row, dataset, prompt, tokenizer, sample_score,
                 prepare_ids, identity, failures):
    label = f"{dataset}:{record.get('row_index')}"
    require(record.get("identity") == identity, f"identity {label}", failures)
    require(record.get("status") == "complete", f"status {label}", failures)
    require(record.get("dataset") == dataset, f"dataset {label}", failures)
    require(record.get("row_index") is not None, f"missing row {label}", failures)
    require(record.get("answers") == source_row["answers"], f"answers {label}", failures)
    require(set(record.get("arms", {})) == {ARM}, f"arms {label}", failures)

    material_ids, suffix_ids, expected_info = prepare_ids(
        tokenizer, source_row, dataset, prompt, 31500)
    require(record.get("input") == expected_info, f"tokenizer input identity {label}", failures)
    require(len(material_ids) + len(suffix_ids) == expected_info["input_tokens"],
            f"token sum {label}", failures)

    arm = record["arms"][ARM]
    info = record["input"]
    require(arm.get("request_phase") == "FINISHED", f"request phase {label}", failures)
    require(arm.get("material_length") == info["material_tokens"],
            f"material length {label}", failures)
    require(arm.get("suffix_length") == info["suffix_tokens"],
            f"suffix length {label}", failures)
    generated = arm.get("generated_ids")
    require(isinstance(generated, list) and all(
        isinstance(token, int) and not isinstance(token, bool) for token in generated),
        f"generated IDs {label}", failures)
    require(arm.get("generated_tokens") == len(generated),
            f"generated token count {label}", failures)
    prediction = tokenizer.decode(generated, skip_special_tokens=True)
    require(arm.get("prediction") == prediction, f"prediction decode {label}", failures)
    metric_dataset = "repobench" if dataset == "repobench-p" else dataset
    expected_score = sample_score(metric_dataset, prediction, source_row["answers"],
                                  source_row.get("all_classes", []))
    require(isinstance(arm.get("score"), (int, float))
            and not isinstance(arm.get("score"), bool)
            and math.isfinite(arm["score"])
            and math.isclose(arm["score"], expected_score, rel_tol=0.0, abs_tol=1e-12),
            f"official score {label}: stored={arm.get('score')} recomputed={expected_score}",
            failures)

    decisions = arm.get("decisions", {})
    receipts = arm.get("nfa_receipts", {})
    expected_layers = {str(layer) for layer in range(47)}
    require(set(decisions) == expected_layers, f"47 decisions {label}", failures)
    require(set(receipts) == expected_layers, f"47 NFA receipts {label}", failures)
    require(all(receipt.get("route") == "graph_ready_sparse"
                for receipt in receipts.values()), f"NFA route {label}", failures)
    counts = Counter(gate_category(decision) for decision in decisions.values())
    expected_gates = {name: counts[name] for name in ("keep", "near", "far", "both")}
    require(sum(expected_gates.values()) == 47, f"gate sum {label}", failures)
    require(arm.get("gate_counts") == expected_gates,
            f"gate/decision mismatch {label}: stored={arm.get('gate_counts')} "
            f"recomputed={expected_gates}", failures)
    final_kv = arm.get("final_kv", {})
    require(final_kv.get("requests") == 1, f"BS=1 final KV {label}", failures)
    require(final_kv.get("active_requests") == [0], f"active request {label}", failures)
    require(final_kv.get("n_layers") == 47 and len(final_kv.get("layers", [])) == 47,
            f"47 final KV layers {label}", failures)

    graph = arm.get("cuda_graph", {})
    expected_steps = info["suffix_tokens"] + max(0, len(generated) - 1)
    require(graph.get("requested") is True, f"graph not requested {label}", failures)
    require(graph.get("eager_fallback_count") == 0, f"eager fallback {label}", failures)
    require(isinstance(graph.get("capture_count"), int) and graph["capture_count"] >= 0,
            f"graph capture count {label}", failures)
    require(graph.get("replay_count") == expected_steps,
            f"graph replay {label}: actual={graph.get('replay_count')} "
            f"expected={expected_steps}", failures)
    return {
        "row_index": record["row_index"],
        "score": expected_score,
        "prediction": prediction,
        "generated_ids": generated,
        "gates": expected_gates,
        "graph": graph,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--repo", default="/tmp/tli-glm-nfa-eps010-graph-20261112")
    parser.add_argument("--data-root", default="/home/wangyuanshuo02/datasets/LongBench/data")
    args = parser.parse_args()

    root = Path(args.run_dir)
    repo = Path(args.repo)
    data_root = Path(args.data_root)
    failures = []
    manifests = []
    for rank in range(8):
        manifest_path = root / f"manifest_rank{rank}.json"
        done_path = root / f"worker_done_rank{rank}.json"
        heartbeat_path = root / f"heartbeat_rank{rank}.json"
        require(manifest_path.is_file(), f"missing {manifest_path.name}", failures)
        require(done_path.is_file(), f"missing {done_path.name}", failures)
        require(heartbeat_path.is_file(), f"missing {heartbeat_path.name}", failures)
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                require(manifest.get("rank") == rank, f"manifest rank {rank}", failures)
                require(manifest.get("identity") == args.expected_identity,
                        f"manifest identity {rank}", failures)
                manifests.append(manifest)
            except Exception as exc:
                failures.append(f"manifest {rank}: {type(exc).__name__}: {exc}")
        if done_path.is_file():
            try:
                require(json.loads(done_path.read_text()).get("identity") == args.expected_identity,
                        f"worker done identity {rank}", failures)
            except Exception as exc:
                failures.append(f"worker done {rank}: {type(exc).__name__}: {exc}")
        if heartbeat_path.is_file():
            try:
                require(json.loads(heartbeat_path.read_text()).get("status") == "complete",
                        f"heartbeat not complete {rank}", failures)
            except Exception as exc:
                failures.append(f"heartbeat {rank}: {type(exc).__name__}: {exc}")
    require(len(manifests) == 8, "not all eight manifests readable", failures)
    if not manifests:
        write_json(args.output, {"status": "FAIL", "failures": failures})
        print(json.dumps({"status": "FAIL", "failures": failures}, ensure_ascii=False))
        return 1

    identity = manifests[0]["identity"]
    protocol = manifests[0]["protocol"]
    require(all(manifest.get("protocol") == protocol for manifest in manifests),
            "rank protocol mismatch", failures)
    require(protocol.get("schema") == "frozen-near-far-screen-v4-material-suffix",
            "protocol schema", failures)
    require(protocol.get("seed") == 42, "seed != 42", failures)
    require(protocol.get("samples_per_task") is None and protocol.get("full_dataset") is True,
            "not full dataset", failures)
    require(protocol.get("context_cap") == 31500, "context cap != 31500", failures)
    require(protocol.get("thresholds") == [0.1], "thresholds != [0.1]", failures)
    require(protocol.get("gate_modes") == ["dual"], "gate modes != [dual]", failures)
    require(protocol.get("row_index") is None, "row-index filter present", failures)
    require(protocol.get("arms") == [ARM], "not dual_0.1 single arm", failures)
    require(protocol.get("batch_size") == 1, "batch size != 1", failures)
    require(protocol.get("datasets") == list(DATASETS), "dataset order/protocol", failures)
    require(protocol.get("dataset_counts") == EXPECTED_COUNTS, "dataset counts", failures)
    require(protocol.get("selected_counts") == EXPECTED_COUNTS, "selected counts", failures)
    require(protocol.get("world_size") == 8, "world size != 8", failures)
    require(protocol.get("input_provenance") == "native-longbench-with-cap",
            "input provenance", failures)
    require(protocol.get("attention") == "nfa-glm_nope_last32-frozen-regions-reference",
            "attention treatment", failures)
    nfa = protocol.get("nfa_args", {})
    require(nfa.get("nfa_subspace") == "glm_nope_last32", "subspace", failures)
    require(nfa.get("nfa_subspace_seed") == 0, "subspace seed", failures)
    require(nfa.get("nfa_enable_subspace") is True, "subspace disabled", failures)
    require(protocol.get("nfa_environment", {}).get("NFA_DECODE_DENSE") == "0",
            "dense decode fallback enabled", failures)
    require(protocol.get("git_head") == EXPECTED_GIT_HEAD, "git HEAD", failures)
    require(protocol.get("worktree_diff_sha256") == EXPECTED_DIFF_SHA256,
            "worktree diff SHA", failures)
    try:
        actual_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        actual_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
        require(actual_head == EXPECTED_GIT_HEAD, "snapshot git HEAD", failures)
        require(hashlib.sha256(actual_diff).hexdigest() == EXPECTED_DIFF_SHA256,
                "snapshot diff SHA", failures)
    except Exception as exc:
        failures.append(f"snapshot identity: {type(exc).__name__}: {exc}")

    dataset_sha = {}
    source_rows = {}
    for dataset in DATASETS:
        path = data_root / f"{dataset}.jsonl"
        require(path.is_file(), f"missing dataset {dataset}", failures)
        if path.is_file():
            dataset_sha[dataset] = sha_file(path)
            require(protocol.get("dataset_sha256", {}).get(dataset) == dataset_sha[dataset],
                    f"dataset SHA {dataset}", failures)
            source_rows[dataset] = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(source_rows[dataset]) == EXPECTED_COUNTS[dataset],
                    f"source rows {dataset}", failures)
        selected = protocol.get("selected_rows", {}).get(dataset, [])
        require(sorted(selected) == list(range(EXPECTED_COUNTS[dataset])),
                f"selected closed set {dataset}", failures)

    source_sha = {}
    for relative, expected_sha in protocol.get("source_sha256", {}).items():
        path = repo / relative
        require(path.is_file(), f"missing source {relative}", failures)
        if path.is_file():
            source_sha[relative] = sha_file(path)
            require(source_sha[relative] == expected_sha, f"source SHA {relative}", failures)
    model_root = Path(protocol["model"])
    for key, filename in (("model_config_sha256", "config.json"),
                          ("tokenizer_config_sha256", "tokenizer_config.json")):
        actual = sha_file(model_root / filename)
        require(protocol.get(key) == actual, f"{key}", failures)

    sys.path.insert(0, str(repo))
    from transformers import AutoTokenizer
    from benchmark.LongBench.eval import sample_score
    from benchmark.LongBench.prefill_frozen_skip import prepare_ids
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    prompts = json.loads((repo / "benchmark/LongBench/config/dataset2prompt.json").read_text())

    tasks = {}
    all_scores = []
    rank_request_counts = {str(rank): 0 for rank in range(8)}
    gate_totals = Counter()
    graph_totals = Counter()
    dataset_offset = 0
    for dataset in DATASETS:
        selected = protocol.get("selected_rows", {}).get(dataset, [])
        expected_files = {f"{row:04d}.json" for row in selected}
        request_dir = root / "requests" / dataset
        actual_files = ({path.name for path in request_dir.glob("*.json")}
                        if request_dir.is_dir() else set())
        require(actual_files == expected_files,
                f"request file set {dataset}: missing={len(expected_files - actual_files)} "
                f"extra={len(actual_files - expected_files)}", failures)
        rows = []
        for position, row_index in enumerate(selected):
            path = request_dir / f"{row_index:04d}.json"
            expected_rank = (dataset_offset + position) % 8
            if not path.is_file():
                failures.append(f"missing request {dataset}:{row_index}")
                continue
            try:
                record = json.loads(path.read_text())
                require(record.get("row_index") == row_index,
                        f"row index {dataset}:{row_index}", failures)
                audited = audit_record(record, source_rows[dataset][row_index], dataset,
                                       prompts[dataset], tokenizer, sample_score, prepare_ids,
                                       identity, failures)
                rows.append(audited)
                rank_request_counts[str(expected_rank)] += 1
                gate_totals.update(audited["gates"])
                graph_totals.update(audited["graph"])
            except Exception as exc:
                failures.append(f"request {dataset}:{row_index}: {type(exc).__name__}: {exc}")
        rows.sort(key=lambda row: row["row_index"])
        require(len(rows) == EXPECTED_COUNTS[dataset], f"incomplete {dataset}", failures)
        scores = [row["score"] for row in rows]
        all_scores.extend(scores)
        task_gates = Counter()
        task_graph = Counter()
        for row in rows:
            task_gates.update(row["gates"])
            task_graph.update(row["graph"])
        tasks[dataset] = {
            "n": len(rows),
            "dual_0.1": sum(scores) / len(scores) if scores else None,
            "gate_counts": {name: task_gates[name] for name in ("keep", "near", "far", "both")},
            "cuda_graph": {name: task_graph[name] for name in
                           ("requested", "capture_count", "replay_count", "eager_fallback_count")},
            "prediction_sha256": sha_payload([
                [row["row_index"], row["prediction"]] for row in rows]),
            "generated_ids_sha256": sha_payload([
                [row["row_index"], row["generated_ids"]] for row in rows]),
        }
        dataset_offset += len(selected)

    require(rank_request_counts == EXPECTED_RANK_COUNTS,
            f"rank counts actual={rank_request_counts} expected={EXPECTED_RANK_COUNTS}", failures)
    require(sum(rank_request_counts.values()) == 3750, "total requests != 3750", failures)
    require(sum(gate_totals.values()) == 3750 * 47, "global gate total", failures)
    require(gate_totals["near"] + gate_totals["both"] > 0,
            "no real near skip in full run", failures)
    require(gate_totals["far"] + gate_totals["both"] > 0,
            "no real far skip in full run", failures)
    require(graph_totals["eager_fallback_count"] == 0, "global eager fallback", failures)
    complete = not failures and len(all_scores) == 3750
    result = {
        "schema": "glm_nfa_eps010_graph_lbv1_16en_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "validation_scope": (
            "Producer-side read-only recomputation; different-owner verification remains pending. "
            "Tokenizer identities and official per-sample scores are recomputed. "
            "The arm emits decisions/NFA receipts/final KV but no prefill_full_kv field."),
        "run_dir": str(root),
        "identity": identity,
        "git_head": EXPECTED_GIT_HEAD,
        "worktree_diff_sha256": EXPECTED_DIFF_SHA256,
        "dataset_sha256": dataset_sha,
        "source_sha256": source_sha,
        "rank_request_counts": rank_request_counts,
        "tasks": tasks,
        "gate_totals": {name: gate_totals[name] for name in ("keep", "near", "far", "both")},
        "graph_totals": {name: graph_totals[name] for name in
                         ("requested", "capture_count", "replay_count", "eager_fallback_count")},
        "tokenizer_inputs_recomputed": len(all_scores),
        "official_scores_recomputed": len(all_scores),
        "macro_dual_0.1": (sum(task["dual_0.1"] for task in tasks.values()) / 16
                           if complete else None),
        "micro_dual_0.1": sum(all_scores) / 3750 if complete else None,
        "failures": failures,
    }
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "macro_dual_0.1": result["macro_dual_0.1"],
        "micro_dual_0.1": result["micro_dual_0.1"],
        "rank_request_counts": rank_request_counts,
        "gate_totals": result["gate_totals"],
        "graph_totals": result["graph_totals"],
        "failures": failures,
    }, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
