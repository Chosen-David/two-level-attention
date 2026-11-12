"""校验并合并 LongBench v2 三臂分片；只接受完整、绑定的 503 样本闭集。"""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark.LongBench.eval import validate_runcap_sidecar
from benchmark.LongBench.lbv2_choice import (
    LBV2_PARSER_VERSION,
    extract_choice_official,
)


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json")
ARM_DIRS = {
    "quest": [
        "quest_retry1/pred_r0of3", "quest_retry1/pred_r1of3", "quest/pred_r2of3"],
    "jssa": ["jssa/pred_r0of3", "jssa/pred_r1of3", "jssa/pred_r2of3"],
    "asynctls": ["asynctls/pred_r0of3", "asynctls/pred_r1of3", "asynctls/pred_r2of3"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_arm(arm, source, source_ids, source_sha):
    rows_by_id = {}
    shard_receipts = []
    for rank, relative in enumerate(ARM_DIRS[arm]):
        shard_dir = ROOT / "raw" / relative
        jsonls = list(shard_dir.glob("*.jsonl"))
        if len(jsonls) != 1:
            raise ValueError(f"{arm} rank{rank}: expected one JSONL, got {jsonls}")
        path = jsonls[0]
        binding = validate_runcap_sidecar(str(path), legacy_allowed=False)
        manifest = json.loads(path.with_name(path.name + ".runcap.json").read_text())
        expected = source[rank::3]
        expected_ids = [row["_id"] for row in expected]
        sharding = manifest["sample_sharding"]
        expected_id_sha = hashlib.sha256(
            "\n".join(expected_ids).encode("utf-8")).hexdigest()
        if sharding != {
                "algorithm": "source_order[rank::world_size]", "rank": rank,
                "world_size": 3, "source_samples": 503,
                "selected_samples": len(expected), "source_sha256": source_sha,
                "selected_ids_sha256": expected_id_sha}:
            raise ValueError(f"{arm} rank{rank}: sharding identity mismatch")
        if manifest["producer"]["git_commit"] != (
                "482911be08df9e4e2cc9bf2f9025fb52f9d9b547"):
            raise ValueError(f"{arm} rank{rank}: unexpected producer commit")
        rows = read_jsonl(path)
        if [row["_id"] for row in rows] != expected_ids:
            raise ValueError(f"{arm} rank{rank}: source order or closed set mismatch")
        for row, source_row in zip(rows, expected):
            if row["_id"] in rows_by_id:
                raise ValueError(f"{arm}: duplicate _id {row['_id']}")
            if row["answers"] != [source_row["answer"]]:
                raise ValueError(f"{arm} {row['_id']}: answer mismatch")
            parsed = extract_choice_official(row["pred"])
            if row.get("pred_choice") != parsed:
                raise ValueError(f"{arm} {row['_id']}: parser mismatch")
            rows_by_id[row["_id"]] = row
        shard_receipts.append({
            "path": str(path.relative_to(ROOT)), "rank": rank,
            "rows": len(rows), "result_sha256": sha256(path),
            "cap_identity_hash": binding["cap_identity_hash"],
        })
    if set(rows_by_id) != set(source_ids) or len(rows_by_id) != 503:
        raise ValueError(f"{arm}: merged result is not the 503-sample closed set")
    ordered = [rows_by_id[sample_id] for sample_id in source_ids]
    merged = ROOT / f"{arm}_503.jsonl"
    merged.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8")
    correct = sum(
        extract_choice_official(row["pred"]) == source_row["answer"]
        for row, source_row in zip(ordered, source))
    parsed = sum(extract_choice_official(row["pred"]) is not None for row in ordered)
    return {
        "samples": 503, "correct": correct, "accuracy": correct / 503,
        "parsed": parsed, "unparsed": 503 - parsed,
        "parser": LBV2_PARSER_VERSION, "merged_file": merged.name,
        "merged_sha256": sha256(merged), "shards": shard_receipts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=sorted(ARM_DIRS),
                        default=list(ARM_DIRS))
    parser.add_argument("--output", default="summary.json")
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    source_ids = [row["_id"] for row in source]
    if len(source) != 503 or len(set(source_ids)) != 503:
        raise ValueError("LongBench v2 source is not 503 unique samples")
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    arms = {arm: collect_arm(arm, source, source_ids, source_sha) for arm in args.arms}
    payload = {
        "schema": "qwen3-lbv2-vendor-accuracy-v1",
        "status": "complete" if len(arms) == 3 else "partial",
        "verification": "pending-independent-verification",
        "model": "Qwen3-8B", "effective_cap": 31500, "max_new_tokens": 128,
        "decoding": "greedy", "samples": 503, "source": str(SOURCE),
        "source_sha256": source_sha, "parser": LBV2_PARSER_VERSION,
        "arms": arms,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and path.name != "manifest.json")
    manifest = {
        "schema": "qwen3-lbv2-vendor-file-manifest-v1",
        "verification": "pending-independent-verification",
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({arm: {"accuracy": values["accuracy"],
                            "correct": values["correct"], "parsed": values["parsed"]}
                      for arm, values in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
