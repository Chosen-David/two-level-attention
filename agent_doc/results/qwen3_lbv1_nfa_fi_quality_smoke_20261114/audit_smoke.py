"""重算四臂 LongBench 真实质量 smoke 的身份、回执、评分与文件哈希。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {
    "nfa_graph": {
        "method": "nfa", "execution": "graph", "implementation": "nfa_graph",
        "graph_capture_count": 5, "graph_replays": 7,
        "graph_eager_fallbacks": 0, "graph_pool_misses": 1,
    },
    "nfa_eager": {
        "method": "nfa", "execution": "eager", "implementation": "nfa_eager",
        "graph_capture_count": 0, "graph_replays": 0,
        "graph_eager_fallbacks": 7, "graph_pool_misses": 1,
    },
    "flashinfer_graph": {
        "method": "flashinfer", "execution": "graph",
        "implementation": "flashinfer_graph", "prefill_append_calls": 36,
        "python_decode_append_calls": 72, "python_decode_wrapper_calls": 72,
        "graph_capture_count": 1, "graph_replays": 7,
        "graph_eager_fallbacks": 0,
        "wrapper": "CUDAGraphBatchDecodeWithPagedKVCacheWrapper",
    },
    "flashinfer_eager": {
        "method": "flashinfer", "execution": "eager",
        "implementation": "flashinfer_eager", "prefill_append_calls": 36,
        "python_decode_append_calls": 252, "python_decode_wrapper_calls": 252,
        "graph_capture_count": 0, "graph_replays": 0,
        "graph_eager_fallbacks": 0, "wrapper": "BatchDecodeWithPagedKVCacheWrapper",
    },
}

summaries = {}
predictions = []
input_hashes = []
for arm, contract in expected.items():
    arm_root = ROOT / arm
    manifest_path = arm_root / "quality_manifest_s0of1.json"
    manifest = load(manifest_path)
    method = contract["method"]
    execution = contract["execution"]
    implementation = contract["implementation"]
    assert manifest["verification"] == "pending-independent-verification"
    assert manifest["method"] == method
    assert manifest["runtime"]["execution"] == execution
    assert manifest["runtime"]["batch_size"] == 1
    assert manifest["runtime"]["dense_or_torch_decode_fallback"] is False
    assert manifest["runtime"]["quality_protocol"] == {
        "fixed_length_performance_receipts_reused": False,
        "real_eos_min_new_stop_token_stop_string": True,
        "per_request_independent_cache": True,
        "graph_session_released_at_request_boundary": method == "nfa",
    }
    harness = REPO / "benchmark/efficiency/benchmark_longbench_quality.py"
    assert manifest["runtime"]["source"]["quality_harness_sha256"] == sha256(harness)
    assert manifest["runtime"]["source"]["head"] == "5eb43ddc428eb15deba5c30433966e50b55cb7a0"

    task = manifest["tasks"]["samsum"]
    pred_path = arm_root / task["path"]
    runcap_path = arm_root / task["runcap"]
    receipts_path = arm_root / task["receipts"]
    result_path = pred_path.parent / "result.json"
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == task["rows"] == 1
    row = rows[0]
    assert row["_id"] == "samsum:0"
    assert row["implementation"] == implementation
    assert row["input"]["input_tokens"] == 9508
    assert row["input"]["input_ids_sha256"] == "df7dbfe0b6402dfbfec00fccbe6b2a48b75f3132cf1bd281a526d302fe49b6b3"
    assert row["input"]["truncated"] is False
    assert task["prediction_sha256"] == sha256(pred_path)
    assert task["receipts_sha256"] == sha256(receipts_path)
    assert task["receipt_rows"] == 1

    runcap = load(runcap_path)
    assert runcap["status"] == "complete"
    assert runcap["method"] == implementation
    assert runcap["effective_cap"] == 31500
    assert runcap["max_new_tokens"] == 128
    assert runcap["result_file"] == pred_path.name
    assert runcap["result_lines"] == runcap["n_samples"] == 1
    assert runcap["result_sha256"] == sha256(pred_path)
    assert runcap["producer"]["script_sha256"] == sha256(harness)
    identity = runcap["input_identity"]
    assert identity["schema"] == "longbench-input-identity-v2"
    assert identity["sample_count"] == 1
    assert identity["actual_input_ids"]["count"] == 1
    assert identity["actual_input_ids"]["ordered_sha256"] == "b00a02c31419eebccc808260df70ae3962eb81d2a336105af863dcaf6349c4e2"
    assert identity["prompt_template_sha256"] == "a68d884234907eefcaa0eacf7b2997f60fa46603988dceae6a44f9b49f045b3f"

    receipts = load(receipts_path)
    assert receipts == manifest["request_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["task"] == "samsum" and receipt["source_index"] == 0
    assert receipt["input"] == row["input"]
    assert receipt["generated_tokens"] == 8
    assert receipt["stop_reason"] == "stop_string"
    assert receipt["logical_decode_steps"] == 7
    assert receipt["submitted_decode_forwards"] == 7
    if method == "nfa":
        assert receipt["nfa_decode_steps"] == 7
    else:
        assert receipt["page_size"] == 64
        assert receipt["fixed_split_size_pages"] == 8
    for key, value in contract.items():
        if key not in {"method", "execution", "implementation"}:
            assert receipt[key] == value, (arm, key, receipt[key], value)

    result = load(result_path)
    assert result[pred_path.name]["score"] == 80.0
    meta = result["_meta"]["runcap"]
    assert meta["strict_validated"] == 1
    assert meta["legacy_no_sidecar"] == 0
    assert meta["cap_effective"] == 31500
    assert meta["input_identity"]["v2_validated"] == 1
    assert meta["input_identity"]["legacy_strict"] == 0

    predictions.append(row["pred"])
    input_hashes.append(row["input"]["input_ids_sha256"])
    summaries[arm] = {
        "generated_tokens": receipt["generated_tokens"],
        "stop_reason": receipt["stop_reason"],
        "logical_decode_steps": receipt["logical_decode_steps"],
        "submitted_decode_forwards": receipt["submitted_decode_forwards"],
        "score": result[pred_path.name]["score"],
        "prediction_sha256": sha256(pred_path),
        "receipts_sha256": sha256(receipts_path),
        "runcap_sha256": sha256(runcap_path),
        "result_sha256": sha256(result_path),
        "manifest_sha256": sha256(manifest_path),
        "quality_harness_sha256": sha256(harness),
        "receipt": receipt,
    }

assert len(set(predictions)) == 1
assert len(set(input_hashes)) == 1
summary = {
    "schema": "qwen3-longbench-real-quality-smoke-audit-v1",
    "verification": "pending-independent-verification",
    "status": "PASS",
    "source_head": "5eb43ddc428eb15deba5c30433966e50b55cb7a0",
    "task": "samsum",
    "source_index": 0,
    "input_tokens": 9508,
    "input_ids_sha256": input_hashes[0],
    "prediction": predictions[0],
    "arms": summaries,
}
(ROOT / "smoke_audit.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

files = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name == "manifest.json":
        continue
    if path.suffix == ".lock" or "__pycache__" in path.parts:
        continue
    files[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
(ROOT / "manifest.json").write_text(json.dumps({
    "schema": "sha256-manifest-v1",
    "verification": "pending-independent-verification",
    "files": files,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "arms": list(summaries), "files": len(files)}, ensure_ascii=False))
