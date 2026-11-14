"""复核并评分 2026-11-14 Qwen32B/Llama/GLM LongBench v2 FullKV 单分片结果。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.LongBench.eval import validate_runcap_sidecar
from benchmark.LongBench.lbv2_choice import (
    LBV2_PARSER_VERSION,
    extract_choice_official,
)

RESULTS = REPO / "agent_doc/results"
SOURCE = Path("/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json")
ARMS = {
    "qwen32b": {
        "root": RESULTS / "qwen32b_fullkv_lbv2_20261114",
        "model": "Qwen3-32B",
        "producer": "4381d92edf1df9a16b13872621ea81bbb6fe9cf8",
    },
    "llama": {
        "root": RESULTS / "llama_fullkv_lbv2_20261114",
        "model": "Llama-3.1-8B",
        "producer": "4381d92edf1df9a16b13872621ea81bbb6fe9cf8",
    },
    "glm": {
        "root": RESULTS / "glm_fullkv_lbv2_20261114",
        "model": "GLM-4.7-Flash",
        "producer": "409f4b86e65952b1bb8602db49aaf9ffdb087dc7",
    },
}
EXPECTED_SCRIPT_SHA = "8be2ee032422ac8d0099a19f8ebc1ab66454d125892602b7fb7bc31dcf7ed81d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answers_sha(answers) -> str:
    return hashlib.sha256(json.dumps(
        answers, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def collect(name: str, spec: dict, source_bytes: bytes) -> None:
    root = spec["root"]
    pred_dir = root / "full/pred_r0of1"
    predictions = sorted(pred_dir.glob("*.jsonl"))
    if len(predictions) != 1:
        raise RuntimeError(f"{name}: 期望唯一正式 JSONL，实际 {predictions}")
    pred = predictions[0]
    runcap_path = Path(str(pred) + ".runcap.json")
    binding = validate_runcap_sidecar(str(pred), legacy_allowed=False)
    runcap = json.loads(runcap_path.read_text(encoding="utf-8"))
    source = json.loads(source_bytes)
    source_ids = [row["_id"] for row in source]
    if len(source) != 503 or len(set(source_ids)) != 503:
        raise RuntimeError("LongBench v2 源集不是 503 个唯一样本")
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    selected_ids_sha = hashlib.sha256(
        "\n".join(source_ids).encode()).hexdigest()
    expected_sharding = {
        "algorithm": "source_order[rank::world_size]",
        "rank": 0,
        "selected_ids_sha256": selected_ids_sha,
        "selected_samples": 503,
        "source_samples": 503,
        "source_sha256": source_sha,
        "world_size": 1,
    }
    if runcap["sample_sharding"] != expected_sharding:
        raise RuntimeError(f"{name}: 分片或源输入身份不一致")
    if runcap["producer"] != {
            "git_commit": spec["producer"],
            "script_path": runcap["producer"]["script_path"],
            "script_sha256": EXPECTED_SCRIPT_SHA}:
        raise RuntimeError(f"{name}: producer 身份不一致")
    if not runcap["producer"]["script_path"].endswith("/benchmark/LongBench/pred.py"):
        raise RuntimeError(f"{name}: producer 脚本路径异常")
    expected_common = {
        "dataset": "lbv2", "method": "none", "model_name": spec["model"],
        "effective_cap": 31500, "max_new_tokens": 128,
        "n_samples": 503, "result_lines": 503, "status": "complete",
    }
    for key, expected in expected_common.items():
        if runcap[key] != expected:
            raise RuntimeError(f"{name}: runcap {key}={runcap[key]!r} != {expected!r}")
    rows = [json.loads(line) for line in pred.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if [row["_id"] for row in rows] != source_ids:
        raise RuntimeError(f"{name}: 预测不是完整、唯一、源顺序的 503 样本")
    correct = 0
    parsed = 0
    per_sample = []
    for row, source_row in zip(rows, source):
        if row["answers"] != [source_row["answer"]]:
            raise RuntimeError(f"{name}: {row['_id']} 答案 metadata 不一致")
        choice = extract_choice_official(row["pred"])
        if row.get("pred_choice") != choice:
            raise RuntimeError(f"{name}: {row['_id']} 生成端 parser 不一致")
        parsed += choice is not None
        hit = choice == source_row["answer"]
        correct += hit
        per_sample.append({"_id": row["_id"], "pred_choice": choice,
                           "answer": source_row["answer"], "correct": hit})

    source_manifest = root / "source_manifest.json"
    source_manifest.write_text(json.dumps({"lbv2": {
        "ids": source_ids,
        "answers_sha": {row["_id"]: answers_sha(row["answers"])
                        for row in rows},
    }}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "benchmark.LongBench.eval",
        "--output-path", str(pred_dir), "--expect-count", "503",
        "--manifest", str(source_manifest),
    ], cwd=REPO, check=True)
    official_result_path = pred_dir / "result.json"
    official_result = json.loads(official_result_path.read_text(encoding="utf-8"))
    official_entry = official_result[pred.name]
    expected_score = round(100 * correct / 503, 2)
    if official_entry["score"] != expected_score:
        raise RuntimeError(
            f"{name}: 官方 eval={official_entry['score']} 与逐样本重算={expected_score} 不一致")

    per_sample_path = root / "official_strict_per_sample.json"
    per_sample_path.write_text(json.dumps({
        "schema": "lbv2-official-strict-per-sample-v1",
        "verification": "pending-independent-verification",
        "parser": LBV2_PARSER_VERSION,
        "samples": per_sample,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "longbench-v2-fullkv-closed-set-v1",
        "status": "complete",
        "verification": "pending-independent-verification",
        "scope": "FullKV HF manual generation only; no sparse method or performance claim",
        "arm": name,
        "model": spec["model"],
        "method": "none",
        "decoding": "greedy",
        "effective_cap": 31500,
        "max_new_tokens": 128,
        "samples": 503,
        "correct": correct,
        "accuracy": correct / 503,
        "accuracy_percent": 100 * correct / 503,
        "parsed": parsed,
        "unparsed": 503 - parsed,
        "parser": LBV2_PARSER_VERSION,
        "source": str(SOURCE),
        "source_sha256": source_sha,
        "selected_ids_sha256": selected_ids_sha,
        "producer": runcap["producer"],
        "prediction_file": str(pred.relative_to(root)),
        "prediction_sha256": sha256(pred),
        "runcap_sha256": sha256(runcap_path),
        "cap_identity_hash": binding["cap_identity_hash"],
        "official_eval_result": str(official_result_path.relative_to(root)),
        "official_eval_score_percent": official_entry["score"],
        "official_eval_result_sha256": sha256(official_result_path),
        "input_tokens": runcap["input_tokens"]["count"],
        "truncated_samples": runcap["truncated"]["count"],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema": "longbench-v2-fullkv-file-manifest-v1",
        "verification": "pending-independent-verification",
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {
        "correct": correct, "accuracy": summary["accuracy"],
        "parsed": parsed, "unparsed": 503 - parsed,
        "official_eval_score_percent": official_entry["score"],
        "files": len(files),
    }}, ensure_ascii=False))


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    for name, spec in ARMS.items():
        collect(name, spec, source_bytes)


if __name__ == "__main__":
    main()
