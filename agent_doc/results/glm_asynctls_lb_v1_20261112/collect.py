"""校验 GLM AsyncTLS LongBench v1 双分片并用官方 scorer 重算。"""
import hashlib
import json
from pathlib import Path

from benchmark.LongBench.eval import sample_score, validate_runcap_sidecar
from benchmark.LongBench.run_cap_manifest import build_lbv1_input_identity


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE_DIR = Path("/home/wangyuanshuo02/datasets/LongBench/data")
PROMPTS_PATH = REPO / "benchmark/LongBench/config/dataset2prompt.json"
MAXLEN_PATH = REPO / "benchmark/LongBench/config/dataset2maxlen.json"
TASK_SIZES = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "lcc": 500,
    "repobench-p": 500,
}
PRODUCER_COMMIT = "a0b086850ba7d290fda47c1c27833e7057e774bb"
PRED_SHA256 = "cd155d8591591576c3723db65d1a31034ee9a3968dec5d73ed3a39a59e115f14"
CHANNEL_SHA256 = "87b28849a4ec034c6c52df4a0b1d35ef58c04d88a8557327d967140c1dc8058a"


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def read_jsonl_bytes(raw):
    text = raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def selected_id(task, fingerprint, source_idx):
    return f"{task}:{fingerprint}:{source_idx}"


def collect_task(task, source_count, prompts, maxlens):
    source_path = SOURCE_DIR / f"{task}.jsonl"
    source = read_jsonl_bytes(source_path.read_bytes())
    if len(source) != source_count:
        raise ValueError(f"{task}: source population {len(source)} != {source_count}")
    identity = build_lbv1_input_identity(source, prompts[task])
    fingerprint = identity["source_fingerprint"]
    expected_ids = [selected_id(task, fingerprint, i) for i in range(source_count)]
    rows_by_id = {}
    receipts = []
    for rank in range(2):
        shard_dir = ROOT / "raw" / task / f"pred_r{rank}of2"
        paths = sorted(shard_dir.glob("*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"{task} rank{rank}: expected one JSONL, got {paths}")
        path = paths[0]
        raw = path.read_bytes()
        rows = read_jsonl_bytes(raw)
        binding = validate_runcap_sidecar(
            str(path), legacy_allowed=False,
            actual_sha=sha256_bytes(raw), actual_lines=len(rows))
        cap_path = path.with_name(path.name + ".runcap.json")
        cap = json.loads(cap_path.read_text(encoding="utf-8"))
        if cap["dataset"] != task or cap["model_name"] != "GLM-4.7-Flash":
            raise ValueError(f"{task} rank{rank}: model/dataset mismatch")
        if cap["method"] != "asynctls_vendor" or cap["run_timestamp"] != "a0b08685":
            raise ValueError(f"{task} rank{rank}: method/run identity mismatch")
        if cap["effective_cap"] != 31500 or cap["max_new_tokens"] != maxlens[task]:
            raise ValueError(f"{task} rank{rank}: cap/generation budget mismatch")
        producer = cap["producer"]
        if (producer["git_commit"] != PRODUCER_COMMIT
                or producer["script_sha256"] != PRED_SHA256):
            raise ValueError(f"{task} rank{rank}: producer identity mismatch")
        cap_identity = cap["input_identity"]
        for key in ("schema", "canonicalization", "source_sha256",
                    "source_fingerprint", "sample_count", "prompt_template_sha256"):
            if cap_identity[key] != identity[key]:
                raise ValueError(f"{task} rank{rank}: full-source {key} mismatch")
        expected_shard_ids = expected_ids[rank::2]
        sharding = {
            "algorithm": "source_order[rank::world_size]",
            "rank": rank,
            "world_size": 2,
            "source_samples": source_count,
            "selected_samples": len(expected_shard_ids),
            "source_sha256": identity["source_sha256"],
            "selected_ids_sha256": sha256_bytes(
                "\n".join(expected_shard_ids).encode("utf-8")),
        }
        if cap["sample_sharding"] != sharding:
            raise ValueError(f"{task} rank{rank}: sharding mismatch")
        if [row["_id"] for row in rows] != expected_shard_ids:
            raise ValueError(f"{task} rank{rank}: output IDs/order mismatch")
        actual = cap_identity["actual_input_ids"]
        if actual["count"] != len(rows) or cap["input_tokens"]["count"] != len(rows):
            raise ValueError(f"{task} rank{rank}: realized input count mismatch")

        treatment_path = path.with_name(path.name + ".nfa_manifest.json")
        treatment_raw = treatment_path.read_bytes()
        treatment = json.loads(treatment_raw)
        expected_vendor = {
            "method": "asynctls_vendor",
            "asynctls_token_budget": 1024,
            "asynctls_block_size": 64,
            "asynctls_top_kb": 128,
        }
        for key, value in expected_vendor.items():
            if treatment.get(key) != value:
                raise ValueError(f"{task} rank{rank}: treatment {key} mismatch")
        channel = treatment.get("asynctls_channel_file")
        if (not isinstance(channel, dict)
                or channel.get("sha256") != CHANNEL_SHA256
                or channel.get("basename") != "glm-4.7-flash_qasper_channels.json"):
            raise ValueError(f"{task} rank{rank}: calibration identity mismatch")
        if treatment.get("nfa_layer_skip_path") is not None:
            raise ValueError(f"{task} rank{rank}: NFA layer skip was not disabled")
        treatment_hash = sha256_bytes(treatment_raw)[:10]
        if (f"_h{treatment_hash}-a0b08685.jsonl" not in path.name
                or not path.name.startswith(task.split("-")[0] + "-")):
            raise ValueError(f"{task} rank{rank}: treatment/result filename mismatch")

        for row, source_row in zip(rows, source[rank::2]):
            sample_id = row["_id"]
            if sample_id in rows_by_id:
                raise ValueError(f"{task}: duplicate {sample_id}")
            if (row["answers"] != source_row["answers"]
                    or row["all_classes"] != source_row["all_classes"]
                    or row["length"] != source_row["length"]):
                raise ValueError(f"{task} {sample_id}: source answer/metadata mismatch")
            if not isinstance(row["pred"], str):
                raise ValueError(f"{task} {sample_id}: prediction is not text")
            rows_by_id[sample_id] = row
        receipts.append({
            "rank": rank,
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "result_sha256": sha256_bytes(raw),
            "runcap_sha256": sha256_file(cap_path),
            "treatment_sha256": sha256_bytes(treatment_raw),
            "cap_identity_hash": binding["cap_identity_hash"],
            "actual_input_ids_ordered_sha256": actual["ordered_sha256"],
        })
    if len(rows_by_id) != source_count or set(rows_by_id) != set(expected_ids):
        raise ValueError(f"{task}: merged output is not the full closed set")
    ordered = [rows_by_id[sample_id] for sample_id in expected_ids]
    merged_path = ROOT / "merged" / f"{task}.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_raw = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in ordered).encode("utf-8")
    merged_path.write_bytes(merged_raw)
    metric_task = "repobench" if task == "repobench-p" else task
    scores = [
        sample_score(metric_task, row["pred"], row["answers"], row["all_classes"])
        for row in ordered
    ]
    return {
        "samples": source_count,
        "score_100": 100.0 * sum(scores) / source_count,
        "source_path": str(source_path),
        "source_sha256": identity["source_sha256"],
        "source_fingerprint": fingerprint,
        "merged_path": str(merged_path.relative_to(ROOT)),
        "merged_sha256": sha256_bytes(merged_raw),
        "shards": receipts,
    }, scores


def main():
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    maxlens = json.loads(MAXLEN_PATH.read_text(encoding="utf-8"))
    tasks = {}
    all_scores = []
    for task, source_count in TASK_SIZES.items():
        tasks[task], scores = collect_task(task, source_count, prompts, maxlens)
        all_scores.extend(scores)
        print(f"{task}: {tasks[task]['score_100']:.6f} ({source_count})", flush=True)
    total = sum(TASK_SIZES.values())
    if total != 3750 or len(all_scores) != total:
        raise ValueError("LongBench v1 English population is not 3750")
    payload = {
        "schema": "glm-asynctls-longbench-v1-accuracy-v1",
        "status": "complete",
        "verification": "pending-independent-verification",
        "model": "GLM-4.7-Flash",
        "method": "asynctls_vendor",
        "producer_commit": PRODUCER_COMMIT,
        "token_budget": 1024,
        "block_size": 64,
        "top_kb": 128,
        "channel_file_sha256": CHANNEL_SHA256,
        "effective_cap": 31500,
        "decoding": "greedy",
        "prefill": "full-prompt-dense",
        "decode_block_selection": "previous-step-blocks",
        "tasks": tasks,
        "samples": total,
        "macro_avg16_100": sum(t["score_100"] for t in tasks.values()) / len(tasks),
        "micro_avg3750_100": 100.0 * sum(all_scores) / total,
        "scorer": "benchmark.LongBench.eval.sample_score",
        "note": "生产侧重算与闭集校验，不冒充不同 owner 独立验收。",
    }
    output = ROOT / "summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and path.name != "manifest.json")
    manifest = {
        "schema": "glm-asynctls-longbench-v1-file-manifest-v1",
        "verification": "pending-independent-verification",
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({
        "samples": total,
        "macro_avg16_100": payload["macro_avg16_100"],
        "micro_avg3750_100": payload["micro_avg3750_100"],
        "manifest_files": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
