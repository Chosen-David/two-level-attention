#!/usr/bin/env python3
"""独立验收 Qwen3-8B LongBench Hot/Gov 的 K2 token budget 消融。

只读审计本地 raw/<task>_k2_<k>/quality_manifest_s0of1.json，调用官方
benchmark.LongBench.eval 重算逐样本分数，并与 FlashInfer FullKV 基线做
配对 bootstrap。不修改原始预测文件；输出 independent_verify.json。
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PY = "/home/fluentllmenv/bin/python"
TASKS = ("hotpotqa", "gov_report")
BUDGETS = (256, 512, 1024, 2048)
EXPECTED_HEAD = "8736a965e903fd461ef5591347933205e18adee0"
BASELINE_ROOT = (
    REPO / "agent_doc/results/qwen3_nfa_fi_longbench16_20261115/"
    "canonical/flashinfer/pred"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def empty_sha() -> str:
    return hashlib.sha256(b"").hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap_ci(a, b, *, b_repeats=10000, seed=20261117):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    n = diffs.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(b_repeats, dtype=float)
    for i in range(b_repeats):
        idx = rng.integers(0, n, size=n)
        means[i] = float(diffs[idx].mean())
    lo, hi = float(np.quantile(means, [0.025, 0.975])[0]), float(np.quantile(means, [0.025, 0.975])[1])
    return float(diffs.mean()), lo, hi


def run_official_eval(cell: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        "/home/wangyuanshuo02/local_pylibs/venv/lib/python3.11/site-packages:"
        + str(REPO)
    )
    log_path = cell / "_independent_eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [PY, "-m", "benchmark.LongBench.eval", "--output-dir", str(cell),
             "--expect-count", "200"],
            cwd=REPO, env=env, check=True, stdout=log, stderr=subprocess.STDOUT,
        )
    return load_json(cell / "pred" / "result.json")


def check(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def verify_cell(root: Path, task: str, k: int):
    cell = root / "raw" / f"{task}_k2_{k}"
    manifest = load_json(cell / "quality_manifest_s0of1.json")
    check(manifest["schema"] == "longbench-official-quality-manifest-v1", "schema")
    check(manifest["verification"] == "pending-independent-verification", "verification")
    check(manifest["method"] == "nfa", "method")
    runtime = manifest["runtime"]
    check(runtime["execution"] == "eager", "execution")
    check(runtime["num_layers"] == 36, "num_layers")
    check(runtime["dense_or_torch_decode_fallback"] is False, "dense fallback")
    source = runtime["source"]
    check(source["head"] == EXPECTED_HEAD, f"source head {source['head']}")
    check(source["source_diff_sha256"] == empty_sha(), "source diff not empty")
    check(source["quality_source_diff_sha256"] == empty_sha(), "quality source diff not empty")
    protocol = runtime["quality_protocol"]
    check(protocol["fixed_length_performance_receipts_reused"] is False, "perf receipt reused")
    check(protocol["real_eos_min_new_stop_token_stop_string"] is True, "stop protocol")
    check(protocol["per_request_independent_cache"] is True, "cache protocol")
    cfg = runtime["nfa_selector_config"]
    check(cfg["subspace"] == "full", "subspace")
    check(cfg["level1_topk"] == 128, "level1_topk")
    check(cfg["level2_topk"] == k, "level2_topk")
    check(cfg["page_size"] == 64, "page_size")
    check(cfg["near_pages"] == 32, "near_pages")
    check(cfg["near_tokens"] == "off", "near_tokens")
    check(cfg["near_len"] == 4096, "near_len")
    check(cfg["sink_tokens"] == 4, "sink")
    check(cfg["sliding_window_size"] == 128, "swa")
    check(cfg["boundary_bypass"] is True, "boundary")
    check(cfg["enable_layer_skip"] is False, "layer_skip")
    check(cfg["dynamic_gate_epsilon"] == 0.1, "dgate eps")
    check(cfg["dynamic_gate_rows"] == 256, "dgate rows")
    check(cfg["near_method"] == "avg", "near_method")
    check(cfg["far_method"] == "minmax", "far_method")
    check(runtime["nfa_quality_arm"] == f"full_gate_off_k2{k}", "arm")
    check(set(manifest["tasks"]) == {task}, "tasks")
    task_info = manifest["tasks"][task]
    check(task_info["rows"] == 200, "rows")
    check(task_info["receipt_rows"] == 200, "receipt_rows")
    pred_rel = task_info["path"]
    pred_path = cell / pred_rel
    receipts_rel = task_info["receipts"]
    receipts_path = cell / receipts_rel
    preds = load_jsonl(pred_path)
    receipts = load_json(receipts_path)
    check(len(preds) == 200, "pred lines")
    check(len(receipts) == 200, "receipt lines")
    check(sha256_bytes(pred_path.read_bytes()) == task_info["prediction_sha256"], "pred sha")
    check(sha256_bytes(receipts_path.read_bytes()) == task_info["receipts_sha256"], "receipt sha")
    baseline_path = BASELINE_ROOT / f"{task}-flashinfer_eager.jsonl"
    baseline = {r["_id"]: r for r in load_jsonl(baseline_path)}
    check(len(baseline) >= 200, "baseline rows")

    result = run_official_eval(cell)
    official_entries = [v for k, v in result.items() if k != "_meta"]
    check(len(official_entries) == 1, "official result entries")
    official_score = official_entries[0]["score"]
    check(result["_meta"]["runcap"]["strict_validated"] == 1, "runcap strict_validated")

    from benchmark.LongBench.eval import sample_score
    scores = []
    paired_scores = []
    for pred, receipt in zip(preds, receipts):
        check(receipt.get("task") == task, "receipt task")
        check(receipt.get("input") == pred["input"], "receipt input identity")
        check(pred["input"]["input_ids_sha256"] == baseline[pred["_id"]]["input"]["input_ids_sha256"], "input identity")
        score = 100.0 * sample_score(task, pred["pred"], pred["answers"], pred["all_classes"])
        base_score = 100.0 * sample_score(task, baseline[pred["_id"]]["pred"], baseline[pred["_id"]]["answers"], baseline[pred["_id"]]["all_classes"])
        scores.append(score)
        paired_scores.append((score, base_score))
    mean = float(np.mean(scores))
    check(abs(mean - official_score) < 0.011, (mean, official_score))
    diff, lo, hi = paired_bootstrap_ci(
        [p[0] for p in paired_scores], [p[1] for p in paired_scores])
    return {
        "task": task,
        "k2": k,
        "rows": 200,
        "official_score": official_score,
        "recomputed_mean_score": mean,
        "fullkv_baseline_score": float(np.mean([p[1] for p in paired_scores])),
        "delta_points": diff,
        "paired_bootstrap_ci95": [lo, hi],
        "prediction_sha256": task_info["prediction_sha256"],
        "receipts_sha256": task_info["receipts_sha256"],
        "source_head": source["head"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[0]))
    ap.add_argument("--repeats", type=int, default=10000)
    args = ap.parse_args()
    root = Path(args.root)
    cells = []
    for task in TASKS:
        for k in BUDGETS:
            cells.append(verify_cell(root, task, k))
    with (root / "k2_ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)
    payload = {
        "schema": "qwen3-lb-k2-ablation-independent-verify-v1",
        "verification": "pending-independent-verification",
        "source_head": EXPECTED_HEAD,
        "protocol": "benchmark.efficiency.benchmark_longbench_quality --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off --nfa-level2-topk 256|512|1024|2048 --visible-len 8192",
        "baseline": {
            "source": "agent_doc/results/qwen3_nfa_fi_longbench16_20261115",
            "flashinfer_source_head": "7fac274fda9df13e82caadd95f3cca5c8374a1a3",
        },
        "cells": cells,
        "paired_bootstrap_repeats": args.repeats,
        "paired_bootstrap_seed": 20261117,
        "scorer": "benchmark.LongBench.eval.sample_score + official eval",
    }
    (root / "independent_verify.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("PASS", len(cells), "cells")


if __name__ == "__main__":
    main()
