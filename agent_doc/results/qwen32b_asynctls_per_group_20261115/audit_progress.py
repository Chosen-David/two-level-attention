"""只读复算 Qwen3-32B AsyncTLS LongBench v1 已完成分片。"""
import datetime
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.LongBench.eval import validate_runcap_sidecar
from benchmark.LongBench.pred import (
    _drop_leading_bos,
    _split_prompt_question,
    build_chat,
    get_method_name_with_info,
    parse_args,
)
from benchmark.LongBench.run_cap_manifest import (
    _ordered_digest_hash,
    build_lbv1_input_identity,
    input_ids_sha256,
)
from sparse_attn.info import get_treatment_manifest_json


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = Path("/home/wangyuanshuo02/ak_verify_results/qwen32b_asynctls_lb_v1_per_group_20261115")
SOURCE_DIR = Path("/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench")
MODEL = Path("/home/wangyuanshuo02/models/huggingface.co/Qwen/Qwen3-32B")
CHANNEL = ROOT / "calibration/qwen3_32b_qasper_per_group.json"
PROMPTS_PATH = REPO / "benchmark/LongBench/config/dataset2prompt.json"
MAXLEN_PATH = REPO / "benchmark/LongBench/config/dataset2maxlen.json"
RUNNER = REPO / "benchmark/LongBench/run_longbench_asynctls_qwen32b_per_group.sh"
PRED = REPO / "benchmark/LongBench/pred.py"
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
WORLD = 2
TIMESTAMP = "20261115T01"
PRODUCER_COMMIT = "1cfed8481e87a62b3a705cd981049e8e699a1f81"
PRED_SHA256 = "8be2ee032422ac8d0099a19f8ebc1ab66454d125892602b7fb7bc31dcf7ed81d"
CHANNEL_SHA256 = "0fc1c57319cebf6c1f6674f9c7239b33bd3d81d82211e5d73be1d5eaf6cb3cbb"
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def read_jsonl_bytes(raw):
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_treatment():
    args = parse_args([
        "--model", "Qwen3-32B", "--model_path", str(MODEL),
        "--task", "narrativeqa", "--method", "asynctls_vendor",
        "--t", TIMESTAMP, "--config-path", str(REPO / "benchmark/LongBench/config"),
        "--dataset-path", str(SOURCE_DIR), "--output-dir", str(OUT),
        "--pred_postfix", "_r0", "--sample-rank", "0",
        "--sample-world-size", str(WORLD),
        "--asynctls_channel_file", str(CHANNEL),
        "--asynctls_token_budget", "1024", "--asynctls_block_size", "64",
        "--asynctls_top_kb", "128", "--nfa_enable_layer_skip", "false",
    ])
    raw = get_treatment_manifest_json(args).encode("utf-8")
    treatment = json.loads(raw)
    require(sha256_file(CHANNEL) == CHANNEL_SHA256, "calibration bytes changed")
    require(treatment["asynctls_channel_file"]["sha256"] == CHANNEL_SHA256,
            "treatment calibration identity mismatch")
    require(treatment["asynctls_channel_file"]["basename"] == CHANNEL.name,
            "treatment calibration basename mismatch")
    require(treatment["asynctls_token_budget"] == 1024
            and treatment["asynctls_block_size"] == 64
            and treatment["asynctls_top_kb"] == 128,
            "AsyncTLS budget mismatch")
    require(treatment["nfa_enable_layer_skip"] is False,
            "NFA layer skip must be disabled")
    return raw, get_method_name_with_info(args)


def realize_input(tokenizer, source_row, prompt_format, task, max_length):
    prompt = prompt_format.format(**source_row)
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        truncated = True
        half = int(max_length / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    else:
        truncated = False
    if task not in NO_CHAT_TASKS:
        prompt = build_chat(tokenizer, prompt, "Qwen3-32B")
    prompt, question = _split_prompt_question(prompt, task)
    input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
    question_ids = tokenizer(question, truncation=False, return_tensors="pt").input_ids
    question_ids = _drop_leading_bos(question_ids, tokenizer.bos_token_id)
    realized_ids = input_ids[0].tolist() + question_ids[0].tolist()
    return len(realized_ids), input_ids_sha256(realized_ids), truncated


def audit_shard(task, rank, path, source, identity, prompts, maxlens, tokenizer,
                treatment_raw, treatment_hash):
    raw = path.read_bytes()
    rows = read_jsonl_bytes(raw)
    binding = validate_runcap_sidecar(
        str(path), legacy_allowed=False,
        actual_sha=sha256_bytes(raw), actual_lines=len(rows))
    cap_path = path.with_name(path.name + ".runcap.json")
    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    prefix = f"{task} rank{rank}"
    require(cap["status"] == "complete", f"{prefix}: non-complete shard")
    require(cap["dataset"] == task and cap["model_name"] == "Qwen3-32B",
            f"{prefix}: dataset/model mismatch")
    require(cap["method"] == "asynctls_vendor" and cap["run_timestamp"] == TIMESTAMP,
            f"{prefix}: method/run identity mismatch")
    require(cap["effective_cap"] == 31500 and cap["max_new_tokens"] == maxlens[task],
            f"{prefix}: cap/max-new mismatch")
    require(cap["model_path"] == str(MODEL.resolve()), f"{prefix}: model path mismatch")
    require(cap["dataset_fingerprint"] == identity["source_fingerprint"],
            f"{prefix}: dataset fingerprint mismatch")
    producer = cap["producer"]
    require(producer == {
        "git_commit": PRODUCER_COMMIT,
        "script_path": str(PRED),
        "script_sha256": PRED_SHA256,
    }, f"{prefix}: producer mismatch")
    cap_identity = cap["input_identity"]
    for key in ("schema", "canonicalization", "source_sha256",
                "source_fingerprint", "sample_count", "prompt_template_sha256"):
        require(cap_identity[key] == identity[key], f"{prefix}: source {key} mismatch")
    expected_ids = [
        f"{task}:{identity['source_fingerprint']}:{i}"
        for i in range(len(source))
    ][rank::WORLD]
    expected_sharding = {
        "algorithm": "source_order[rank::world_size]",
        "rank": rank,
        "world_size": WORLD,
        "source_samples": len(source),
        "selected_samples": len(expected_ids),
        "source_sha256": identity["source_sha256"],
        "selected_ids_sha256": sha256_bytes(
            "\n".join(expected_ids).encode("utf-8")),
    }
    require(cap["sample_sharding"] == expected_sharding, f"{prefix}: sharding mismatch")
    require(cap["n_samples"] == len(expected_ids) == len(rows),
            f"{prefix}: population mismatch")
    require([row["_id"] for row in rows] == expected_ids,
            f"{prefix}: output IDs/order mismatch")

    tokens = []
    hashes = []
    flags = []
    for row, source_row in zip(rows, source[rank::WORLD]):
        sample_id = row["_id"]
        require(row["answers"] == source_row["answers"]
                and row["all_classes"] == source_row["all_classes"]
                and row["length"] == source_row["length"],
                f"{prefix} {sample_id}: source metadata mismatch")
        require(isinstance(row["pred"], str), f"{prefix} {sample_id}: prediction not text")
        require(row["execution"] == {"requested": "eager", "actual": "eager"},
                f"{prefix} {sample_id}: execution mismatch")
        require(row["budget"] is None and row["score_sum"] is None,
                f"{prefix} {sample_id}: unexpected populated metric fields")
        token_count, token_hash, truncated = realize_input(
            tokenizer, source_row, prompts[task], task, 31500)
        tokens.append(token_count)
        hashes.append(token_hash)
        flags.append(truncated)
    require(cap["input_tokens"] == {
        "count": len(tokens), "min": min(tokens), "max": max(tokens),
        "per_sample": tokens,
    }, f"{prefix}: realized input token counts mismatch")
    require(cap["truncated"] == {"count": sum(flags), "flags": flags},
            f"{prefix}: truncation flags mismatch")
    actual = cap_identity["actual_input_ids"]
    require(actual == {
        "encoding": "signed-int64-little-endian-v1",
        "count": len(hashes),
        "per_sample_sha256": hashes,
        "ordered_sha256": _ordered_digest_hash(hashes),
    }, f"{prefix}: realized input hashes mismatch")
    require(f"_h{treatment_hash}-{TIMESTAMP}.jsonl" in path.name,
            f"{prefix}: treatment filename mismatch")
    require(not path.with_name(path.name + ".nfa_manifest.json").exists(),
            f"{prefix}: unexpected treatment sidecar; audit must remain read-only")
    return {
        "task": task,
        "rank": rank,
        "world_size": WORLD,
        "path": str(path),
        "rows": len(rows),
        "result_sha256": sha256_bytes(raw),
        "runcap_sha256": sha256_file(cap_path),
        "cap_identity_hash": binding["cap_identity_hash"],
        "source_sha256": identity["source_sha256"],
        "actual_input_ids_ordered_sha256": actual["ordered_sha256"],
        "input_tokens_min": min(tokens),
        "input_tokens_max": max(tokens),
        "truncated": sum(flags),
        "max_new_tokens": maxlens[task],
        "treatment_sha256": sha256_bytes(treatment_raw),
        "ids": [row["_id"] for row in rows],
        "checks": [
            "runcap-schema-result-binding",
            "full-source-identity-and-round-robin-ids",
            "producer-and-fixed-treatment",
            "per-sample-tokenizer-input-token-counts-hashes-and-truncation",
            "source-answer-metadata-and-eager-execution",
        ],
    }


def main():
    require(sha256_file(PRED) == PRED_SHA256, "current pred.py differs from producer")
    treatment_raw, method_name = fixed_treatment()
    treatment_hash = sha256_bytes(treatment_raw)[:10]
    require(treatment_hash == "932711dc4e", "canonical treatment hash mismatch")
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    maxlens = json.loads(MAXLEN_PATH.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    shards = []
    pending = []
    task_rows = {task: {} for task in TASK_SIZES}
    for task, source_count in TASK_SIZES.items():
        source_path = SOURCE_DIR / f"{task}.jsonl"
        source = read_jsonl_bytes(source_path.read_bytes())
        require(len(source) == source_count, f"{task}: source population mismatch")
        identity = build_lbv1_input_identity(source, prompts[task])
        for rank in range(WORLD):
            path = OUT / f"pred_r{rank}" / f"{task.split('-')[0]}-{method_name}-{TIMESTAMP}.jsonl"
            cap_path = path.with_name(path.name + ".runcap.json")
            if not cap_path.is_file():
                pending.append({"task": task, "rank": rank, "state": "no-runcap"})
                continue
            cap = json.loads(cap_path.read_text(encoding="utf-8"))
            if cap.get("status") != "complete":
                pending.append({"task": task, "rank": rank,
                                "state": cap.get("status", "unknown")})
                continue
            receipt = audit_shard(
                task, rank, path, source, identity, prompts, maxlens, tokenizer,
                treatment_raw, treatment_hash)
            ids = receipt.pop("ids")
            for sample_id in ids:
                require(sample_id not in task_rows[task], f"{task}: duplicate {sample_id}")
                task_rows[task][sample_id] = rank
            shards.append(receipt)
            print(f"AUDIT PASS {task} rank{rank}: {receipt['rows']} rows", flush=True)
    closed_tasks = []
    for task, source_count in TASK_SIZES.items():
        expected = {
            f"{task}:v2-{build_lbv1_input_identity(read_jsonl_bytes((SOURCE_DIR / f'{task}.jsonl').read_bytes()), prompts[task])['source_sha256']}:{i}"
            for i in range(source_count)
        }
        if set(task_rows[task]) == expected:
            closed_tasks.append(task)
        elif task_rows[task]:
            require(len(task_rows[task]) < source_count, f"{task}: invalid merged population")
    shards.sort(key=lambda item: (list(TASK_SIZES).index(item["task"]), item["rank"]))
    payload = {
        "schema": "qwen3-32b-asynctls-longbench-v1-progress-audit-v1",
        "status": "partial" if len(shards) < 2 * len(TASK_SIZES) else "complete",
        "audit": "PASS",
        "verification": "pending-independent-verification",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "仅审计快照发现时已 complete 的分片；不评分、不合并、不修改正式输出。",
        "note": "生产侧独立进程复算，不冒充不同 owner 独立验收。",
        "model": "Qwen3-32B",
        "method": "asynctls_vendor",
        "producer_commit": PRODUCER_COMMIT,
        "pred_sha256": PRED_SHA256,
        "runner_sha256": sha256_file(RUNNER),
        "auditor_sha256": sha256_file(Path(__file__)),
        "treatment_sha256": sha256_bytes(treatment_raw),
        "treatment_provenance": "由固定 runner CLI 只读重建；正式目录无 treatment sidecar，未补写。",
        "complete_shards": len(shards),
        "expected_shards": 2 * len(TASK_SIZES),
        "finalized_predictions": sum(item["rows"] for item in shards),
        "expected_predictions": sum(TASK_SIZES.values()),
        "closed_tasks": closed_tasks,
        "pending_shards": pending,
        "shards": shards,
        "limitations": [
            "当前 JSONL 不含逐请求生成 token/stop-reason/backend-call receipt，本审计不虚构这些字段。",
            "局部 PASS 不代表 LongBench v1 16 任务质量完成，不产生 macro/micro 分数。",
        ],
    }
    output = ROOT / "progress_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "audit": payload["audit"],
        "complete_shards": payload["complete_shards"],
        "finalized_predictions": payload["finalized_predictions"],
        "closed_tasks": closed_tasks,
        "progress_audit": str(output),
        "progress_audit_sha256": sha256_file(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
