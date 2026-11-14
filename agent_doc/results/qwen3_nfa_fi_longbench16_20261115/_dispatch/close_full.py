"""闭集审计通过后，扁平化九路分片、canonical 合并并调用官方严格评分。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/tmp/qwen3-nfa-fi-quality-cbfd70da-20261114")
DATA_ROOT = Path("/home/wangyuanshuo02/datasets/LongBench/data")
PYTHON = "/home/fluentllmenv/bin/python"
LOCAL_DEPS = Path(
    "/home/wangyuanshuo02/local_pylibs/venv/lib/python3.11/site-packages")
EXPECTED_HEADS = {
    "nfa": "cbfd70da607eb5afed4343d6d055073f450e4162",
    "flashinfer": "7fac274fda9df13e82caadd95f3cca5c8374a1a3",
}
ENGLISH16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def copy_verified(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(source) == sha256(destination),
                f"既有扁平分片与新来源不一致: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    require(sha256(source) == sha256(temporary), f"复制后 SHA 不一致: {source}")
    os.replace(temporary, destination)


def run_logged(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        subprocess.run(command, check=True, env=env, cwd=REPO, stdout=handle,
                       stderr=subprocess.STDOUT)


audit_path = ROOT / "rank_audit.json"
audit = load(audit_path)
require(audit["status"] == "PASS", f"九路闭集审计尚未 PASS: {audit_path}")
require(audit["source_heads"] == EXPECTED_HEADS, audit_path)
require(audit["cross_arm_input_identity"] == "PASS", audit_path)
require(audit["verification"] == "pending-independent-verification", audit_path)
for method in ("nfa", "flashinfer"):
    require(audit["arms"][method]["rows"] == 3750, (method, audit["arms"][method]))
    require(audit["arms"][method]["source_head"] == EXPECTED_HEADS[method], audit_path)

merge_env = dict(os.environ)
merge_env["PYTHONPATH"] = str(REPO)
eval_env = dict(os.environ)
eval_env["PYTHONPATH"] = f"{REPO}:{LOCAL_DEPS}"
summary = {
    "schema": "qwen3-nfa-fi-longbench-quality-canonical-v2",
    "verification": "pending-independent-verification",
    "source_heads": EXPECTED_HEADS,
    "rank_audit_sha256": sha256(audit_path),
    "arms": {},
}

for method in ("nfa", "flashinfer"):
    label = f"{method}_eager"
    shard_pred = ROOT / "shards" / method / "pred"
    canonical_root = ROOT / "canonical" / method
    canonical_pred = canonical_root / "pred"
    for task in ENGLISH16:
        prefix = task.split("-")[0]
        shards = []
        for rank in range(9):
            source = ROOT / method / f"rank{rank}" / "pred" / (
                f"{prefix}-{label}_s{rank}of9.jsonl")
            destination = shard_pred / source.name
            copy_verified(source, destination)
            copy_verified(Path(str(source) + ".runcap.json"),
                          Path(str(destination) + ".runcap.json"))
            copy_verified(source.with_suffix(".receipts.json"),
                          destination.with_suffix(".receipts.json"))
            shards.append(str(destination))
        output = canonical_pred / f"{prefix}-{label}.jsonl"
        # 合并器要求 canonical 与所有分片同目录；先在 shards 完成严格合并，
        # 再将预测和 canonical runcap 一并复制到官方评分消费目录。
        merged_output = shard_pred / output.name
        run_logged([
            PYTHON, "-m", "benchmark.official_baselines.merge_longbench_quality",
            "--dataset", task, "--shard-files", *shards,
            "--output", str(merged_output), "--data-root", str(DATA_ROOT),
            "--config-path", str(REPO / "benchmark/LongBench/config"),
        ], ROOT / "_logs" / f"canonical_{method}_{task}.log", merge_env)
        copy_verified(merged_output, output)
        copy_verified(Path(str(merged_output) + ".runcap.json"),
                      Path(str(output) + ".runcap.json"))

    run_logged([
        PYTHON, "-m", "benchmark.LongBench.eval",
        "--output-dir", str(canonical_root), "--expect-count", "150",
    ], ROOT / "_logs" / f"official_eval_{method}.log", eval_env)
    result_path = canonical_pred / "result.json"
    result = load(result_path)
    expected_keys = {f"{task.split('-')[0]}-{label}.jsonl" for task in ENGLISH16}
    require(set(result) == expected_keys | {"_meta"},
            (method, sorted(set(result) ^ (expected_keys | {"_meta"}))))
    require(result["_meta"]["runcap"]["strict_validated"] == 16, result_path)
    require(result["_meta"]["runcap"]["legacy_no_sidecar"] == 0, result_path)
    require(result["_meta"]["runcap"]["cap_effective"] == 31500, result_path)
    require(result["_meta"]["runcap"]["input_identity"]["v2_validated"] == 16,
            result_path)
    scores = {task: result[f"{task.split('-')[0]}-{label}.jsonl"]["score"]
              for task in ENGLISH16}
    summary["arms"][method] = {
        "source_head": EXPECTED_HEADS[method],
        "rows": 3750,
        "tasks": 16,
        "scores": scores,
        "macro_avg16": sum(scores.values()) / len(scores),
        "official_result": str(result_path.relative_to(ROOT)),
        "official_result_sha256": sha256(result_path),
        "canonical_predictions": {
            task: {
                "path": str((canonical_pred /
                             f"{task.split('-')[0]}-{label}.jsonl").relative_to(ROOT)),
                "sha256": sha256(canonical_pred /
                                 f"{task.split('-')[0]}-{label}.jsonl"),
            }
            for task in ENGLISH16
        },
    }

summary["status"] = "PASS"
output = ROOT / "quality_summary.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
print(json.dumps({
    "status": "PASS",
    "arms": {method: {"macro_avg16": arm["macro_avg16"], "rows": arm["rows"]}
             for method, arm in summary["arms"].items()},
}, ensure_ascii=False))
