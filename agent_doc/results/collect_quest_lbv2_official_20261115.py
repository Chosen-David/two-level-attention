"""复核并汇总 2026-11-15 官方 Quest LongBench v2 六分片结果。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.LongBench.eval import validate_runcap_sidecar
from benchmark.LongBench.lbv2_choice import LBV2_PARSER_VERSION, extract_choice_official

ROOT = REPO / "agent_doc/results/quest_lbv2_official_8ec4e66d_20261115"
SOURCE = Path("/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json")
EXPECTED_SOURCE_SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
EXPECTED_COMMIT = "8ec4e66de07a4f84432da0f8461e45cbfbc05040"
EXPECTED_SCRIPT_SHA = "e3ea36cbbbee6795f8a8a26b85011d5c7968f2d53305a531064d9c8307f12786"
EXPECTED_QUEST_COMMIT = "01c1623bf9395009520874e989e29f683203b357"
EXPECTED_DIFF_SHA = "edb9e2d579687781a9b293ed87dc2445664db25118811bc5a8227b5ea6792b8f"
EXPECTED_KERNEL_SHA = "de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA:
        raise RuntimeError("LongBench v2 原始文件 SHA 不一致")
    source = json.loads(source_bytes)
    source_ids = [row["_id"] for row in source]
    if len(source) != 503 or len(set(source_ids)) != 503:
        raise RuntimeError("LongBench v2 源集不是 503 个唯一样本")

    pred_dir = ROOT / "full/pred_r0of1"
    predictions = sorted(pred_dir.glob("*.jsonl"))
    if len(predictions) != 1:
        raise RuntimeError(f"期望唯一 canonical JSONL，实际 {predictions}")
    pred = predictions[0]
    runcap_path = Path(str(pred) + ".runcap.json")
    binding = validate_runcap_sidecar(str(pred), legacy_allowed=False)
    runcap = json.loads(runcap_path.read_text(encoding="utf-8"))
    if runcap["status"] != "complete" or runcap["result_lines"] != 503:
        raise RuntimeError("canonical runcap 不是完整 503 题")
    if runcap["producer"]["git_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("canonical producer commit 不一致")
    if runcap["producer"]["script_sha256"] != EXPECTED_SCRIPT_SHA:
        raise RuntimeError("canonical producer script SHA 不一致")
    if runcap["source_file_sha256"] != EXPECTED_SOURCE_SHA:
        raise RuntimeError("canonical 原始数据 SHA 不一致")
    if runcap["method"] != "quest_official" or runcap["model_name"] != "Qwen3-8B":
        raise RuntimeError("canonical 方法或模型身份不一致")
    if runcap["effective_cap"] != 31500 or runcap["max_new_tokens"] != 128:
        raise RuntimeError("canonical cap 或 max-new 不一致")
    merge = runcap.get("canonical_merge", {})
    if merge.get("world_size") != 6 or [s["rank"] for s in merge["shards"]] != list(range(6)):
        raise RuntimeError("canonical 不是完整六分片合并")

    rows = read_jsonl(pred)
    if [row["_id"] for row in rows] != source_ids:
        raise RuntimeError("canonical 预测不是完整、唯一、源顺序的 503 样本")
    correct = 0
    parsed = 0
    per_sample = []
    for row, source_row in zip(rows, source):
        if row["answers"] != [source_row["answer"]]:
            raise RuntimeError(f"{row['_id']} 答案 metadata 不一致")
        if row["all_classes"] != ["A", "B", "C", "D"]:
            raise RuntimeError(f"{row['_id']} 选项 metadata 不一致")
        if row["length"] != source_row["length"]:
            raise RuntimeError(f"{row['_id']} length metadata 不一致")
        if row["implementation"] != "quest_official":
            raise RuntimeError(f"{row['_id']} implementation 不一致")
        choice = extract_choice_official(row["pred"])
        if row.get("pred_choice") != choice:
            raise RuntimeError(f"{row['_id']} strict parser 不一致")
        parsed += choice is not None
        hit = choice == source_row["answer"]
        correct += hit
        per_sample.append({"_id": row["_id"], "pred_choice": choice,
                           "answer": source_row["answer"], "correct": hit})

    official_result_path = pred_dir / "result.json"
    official_result = json.loads(official_result_path.read_text(encoding="utf-8"))
    official_entry = official_result[pred.name]
    expected_score = round(100 * correct / 503, 2)
    if official_entry["score"] != expected_score:
        raise RuntimeError("官方 eval 与逐样本重算不一致")

    totals = Counter()
    shard_audits = []
    for rank in range(6):
        rank_root = ROOT / f"raw/rank{rank}"
        manifest_path = rank_root / f"quality_manifest_s{rank}of6.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        expected_runtime = {
            "batch_size": 1, "patched_layers": 36, "budget": 1024,
            "page_size": 16, "page_budget": 64,
            "query_and_metadata_heads": 32, "physical_kv_heads": 8,
            "dense_or_torch_fallback": False,
        }
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                raise RuntimeError(f"rank{rank} runtime {key} 不一致")
        if runtime["quest"]["commit"] != EXPECTED_QUEST_COMMIT:
            raise RuntimeError(f"rank{rank} Quest commit 不一致")
        if runtime["quest"]["uncommitted_diff_sha256"] != EXPECTED_DIFF_SHA:
            raise RuntimeError(f"rank{rank} Quest 适配 diff SHA 不一致")
        if runtime["kernel_extension_sha256"] != EXPECTED_KERNEL_SHA:
            raise RuntimeError(f"rank{rank} Quest kernel SHA 不一致")
        task = manifest["tasks"]["lbv2"]
        shard_pred = rank_root / task["path"]
        shard_runcap_path = rank_root / task["runcap"]
        shard_source_manifest = rank_root / task["source_manifest"]
        if sha256(shard_pred) != task["prediction_sha256"]:
            raise RuntimeError(f"rank{rank} 预测 SHA 与 quality manifest 不一致")
        if sha256(shard_source_manifest) != task["source_manifest_sha256"]:
            raise RuntimeError(f"rank{rank} source manifest SHA 不一致")
        if task["data_sha256"] != EXPECTED_SOURCE_SHA:
            raise RuntimeError(f"rank{rank} 原始数据 SHA 不一致")
        shard_runcap = json.loads(shard_runcap_path.read_text(encoding="utf-8"))
        if shard_runcap["status"] != "complete":
            raise RuntimeError(f"rank{rank} runcap 未 complete")
        if shard_runcap["producer"] != runcap["producer"]:
            raise RuntimeError(f"rank{rank} producer 与 canonical 不一致")
        shard_rows = read_jsonl(shard_pred)
        expected_ids = source_ids[rank::6]
        if [row["_id"] for row in shard_rows] != expected_ids:
            raise RuntimeError(f"rank{rank} 官方 ID 或源顺序不一致")
        if task["rows"] != len(shard_rows) or shard_runcap["n_samples"] != len(shard_rows):
            raise RuntimeError(f"rank{rank} 人口不一致")
        for row in shard_rows:
            if row["pred_choice"] != extract_choice_official(row["pred"]):
                raise RuntimeError(f"rank{rank} {row['_id']} strict parser 不一致")
        counts = manifest["quest_call_counts"]
        if counts["dense_or_torch_fallback"] != 0:
            raise RuntimeError(f"rank{rank} 出现 dense/Torch fallback")
        if counts["prefill_forward"] != 36 * len(shard_rows):
            raise RuntimeError(f"rank{rank} prefill 调用不是 36 层×样本数")
        if counts["decode_estimate"] != counts["decode_topk"]:
            raise RuntimeError(f"rank{rank} estimate/topk 计数不一致")
        if counts["decode_estimate"] <= 0 or counts["decode_sparse_attn"] <= 0:
            raise RuntimeError(f"rank{rank} 官方 decode kernel 未实际调用")
        totals.update(counts)
        shard_audits.append({
            "rank": rank, "rows": len(shard_rows),
            "prediction_sha256": task["prediction_sha256"],
            "runcap_sha256": sha256(shard_runcap_path),
            "source_manifest_sha256": task["source_manifest_sha256"],
            "quality_manifest_sha256": sha256(manifest_path),
            "quest_call_counts": counts,
        })
    if totals["prefill_forward"] != 36 * 503:
        raise RuntimeError("全量 prefill 调用不是 36 层×503")

    write_json(ROOT / "official_strict_per_sample.json", {
        "schema": "lbv2-official-strict-per-sample-v1",
        "verification": "pending-independent-verification",
        "parser": LBV2_PARSER_VERSION,
        "samples": per_sample,
    })
    write_json(ROOT / "shard_audit.json", {
        "schema": "quest-lbv2-six-shard-audit-v1",
        "status": "PASS",
        "verification": "pending-independent-verification",
        "scope": "producer-side structural and official backend audit only",
        "quest_commit": EXPECTED_QUEST_COMMIT,
        "adaptation_diff_sha256": EXPECTED_DIFF_SHA,
        "kernel_extension_sha256": EXPECTED_KERNEL_SHA,
        "quest_call_counts": dict(sorted(totals.items())),
        "shards": shard_audits,
    })
    summary = {
        "schema": "longbench-v2-official-quest-closed-set-v1",
        "status": "complete",
        "verification": "pending-independent-verification",
        "scope": "official Quest CUDA kernels with local Qwen3 GQA/interface adaptation; quality only, no performance claim",
        "model": "Qwen3-8B",
        "method": "quest_official",
        "decoding": "greedy",
        "effective_cap": 31500,
        "max_new_tokens": 128,
        "budget": 1024,
        "page_size": 16,
        "page_budget": 64,
        "batch_size": 1,
        "samples": 503,
        "correct": correct,
        "accuracy": correct / 503,
        "accuracy_percent": 100 * correct / 503,
        "parsed": parsed,
        "unparsed": 503 - parsed,
        "parser": LBV2_PARSER_VERSION,
        "source": str(SOURCE),
        "source_sha256": EXPECTED_SOURCE_SHA,
        "producer": runcap["producer"],
        "canonical_merge_tool": merge["tool"],
        "prediction_file": str(pred.relative_to(ROOT)),
        "prediction_sha256": sha256(pred),
        "runcap_sha256": sha256(runcap_path),
        "cap_identity_hash": binding["cap_identity_hash"],
        "official_eval_result": str(official_result_path.relative_to(ROOT)),
        "official_eval_score_percent": official_entry["score"],
        "official_eval_result_sha256": sha256(official_result_path),
        "input_tokens": runcap["input_tokens"]["count"],
        "truncated_samples": runcap["truncated"]["count"],
        "quest_commit": EXPECTED_QUEST_COMMIT,
        "adaptation_diff_sha256": EXPECTED_DIFF_SHA,
        "kernel_extension_sha256": EXPECTED_KERNEL_SHA,
        "quest_call_counts": dict(sorted(totals.items())),
        "dense_or_torch_fallback": 0,
        "shard_audit": "shard_audit.json",
    }
    write_json(ROOT / "summary.json", summary)
    files = sorted(path for path in ROOT.rglob("*")
                   if path.is_file() and path.name != "manifest.json")
    write_json(ROOT / "manifest.json", {
        "schema": "longbench-v2-official-quest-file-manifest-v1",
        "verification": "pending-independent-verification",
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    })
    print(json.dumps({
        "samples": 503, "correct": correct, "accuracy": summary["accuracy"],
        "parsed": parsed, "unparsed": 503 - parsed,
        "official_eval_score_percent": official_entry["score"],
        "quest_call_counts": dict(sorted(totals.items())),
        "files": len(files),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
