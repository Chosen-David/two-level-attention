#!/usr/bin/env python3
"""冻结 Q0 的 500 个输入，并审计既有三臂 raw 的可复用边界。"""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark.LongBench.eval import sample_score  # noqa: E402
from benchmark.RULER.score_ruler import string_match_all  # noqa: E402

LONGBENCH_ROOT = Path("/home/wangyuanshuo02/datasets/LongBench/data")
RULER_ROOT = Path(
    "/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER")
B_FINAL = (ROOT / "agent_doc/results/nfa_ablation_20261030/B_final/quality"
           / "2d2a466b4e9a")
FULLKV_LB = ROOT / "exp/results_longbench/Qwen3-8B/pred_1024"
FULLKV_RULER = (ROOT / "exp/trace/results"
                / "e116b_ruler65536k_formal_fullkv.json.run-20261011200053-2710075-5192"
                / "pred_root/L65536/pred_1024/niah_multikey_2-none-merged.jsonl")
STRICT_COMPARISON = (ROOT / "exp/trace/results/nfa_strict_skip_seed_removal"
                     / "longbench_comparison.json")
MAX_NEW_TOKENS = {"hotpotqa": 32, "gov_report": 512, "niah_multikey_2": 128}
EXPECTED_DATASET_SHA = {
    "hotpotqa": "a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f",
    "gov_report": "d28112beb3a9b41d80aa390837fa1a31c9e3da84a5262009c97585cc49f597c4",
    "niah_multikey_2": "52b587a924defa07d36091824e3128310f657bca34a4f011c062879e757c5063",
}


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(compact(row) + "\n")


def repo_path(path):
    return path.relative_to(ROOT).as_posix()


def main():
    dataset_paths = {
        "hotpotqa": LONGBENCH_ROOT / "hotpotqa.jsonl",
        "gov_report": LONGBENCH_ROOT / "gov_report.jsonl",
        "niah_multikey_2": RULER_ROOT / "65536/niah_multikey_2.jsonl",
    }
    for dataset, path in dataset_paths.items():
        assert sha_file(path) == EXPECTED_DATASET_SHA[dataset], path
    source_rows = {name: read_jsonl(path) for name, path in dataset_paths.items()}
    assert len(source_rows["hotpotqa"]) == 200
    assert len(source_rows["gov_report"]) == 200
    assert len(source_rows["niah_multikey_2"]) == 100

    fullkv_paths = {
        "hotpotqa": FULLKV_LB / "hotpotqa-none-01161806.jsonl",
        "gov_report": FULLKV_LB / "gov_report-none-01161806.jsonl",
        "niah_multikey_2": FULLKV_RULER,
    }
    fullkv_rows = {name: read_jsonl(path) for name, path in fullkv_paths.items()}
    for dataset in fullkv_rows:
        assert len(fullkv_rows[dataset]) == len(source_rows[dataset])

    targets = []
    quality = []
    identity_counts = Counter()
    for dataset in ("hotpotqa", "gov_report", "niah_multikey_2"):
        for row_index, (source, prediction) in enumerate(
                zip(source_rows[dataset], fullkv_rows[dataset])):
            answers = source["outputs"] if dataset == "niah_multikey_2" else source["answers"]
            assert prediction["answers"] == answers
            assert prediction["length"] == source["length"]
            if dataset == "niah_multikey_2":
                expected_id = f"niah_multikey_2:{source['index']}"
                assert source["index"] == row_index
                assert prediction["_id"] == expected_id
                assert prediction["_answers_sha"] == sha_bytes(
                    json.dumps(answers, ensure_ascii=False, sort_keys=True).encode()
                )[:16]
                input_fields = {
                    "input_text_sha256": sha_bytes(source["input"].encode()),
                }
                identity_counts["ruler_id_answers_length"] += 1
            else:
                raw_request = B_FINAL / dataset / "requests" / f"{row_index:04d}.json"
                raw = json.loads(raw_request.read_text())
                assert raw["status"] == "complete"
                assert raw["row_index"] == row_index and raw["dataset"] == dataset
                assert raw["answers"] == answers
                input_fields = {
                    "b_final_input_sha256": raw["input"]["input_sha256"],
                    "b_final_raw_tokens": raw["input"]["raw_tokens"],
                    "b_final_input_tokens": raw["input"]["input_tokens"],
                    "b_final_truncated": raw["input"]["truncated"],
                    "b_final_raw_request": repo_path(raw_request),
                }
                identity_counts["longbench_answers_length"] += 1
            target_id = f"{dataset}:{row_index}"
            target = {
                "target_id": target_id,
                "dataset": dataset,
                "row_index": row_index,
                "context_length": 65536 if dataset == "niah_multikey_2" else None,
                "max_new_tokens": MAX_NEW_TOKENS[dataset],
                "dataset_sha256": EXPECTED_DATASET_SHA[dataset],
                "source_record_sha256": sha_bytes(compact(source).encode()),
                "answers_sha256": sha_bytes(compact(answers).encode()),
                "source_length": source["length"],
                **input_fields,
            }
            targets.append(target)
            if dataset == "niah_multikey_2":
                score_percent = string_match_all([prediction["pred"]], [answers])
                score = score_percent / 100.0
            else:
                score = sample_score(
                    dataset, prediction["pred"], answers, source.get("all_classes"))
                score_percent = 100.0 * score
            quality.append({
                "target_id": target_id,
                "dataset": dataset,
                "row_index": row_index,
                "arm": "dense_fullkv",
                "prediction": prediction["pred"],
                "answers": answers,
                "score": score,
                "score_percent": score_percent,
                "generated_token_ids": None,
                "generated_token_count": None,
                "declared_task_max_new_tokens": MAX_NEW_TOKENS[dataset],
                "effective_max_new_tokens": None,
                "generation_cap_provenance": "legacy raw without producer run-cap receipt",
                "identity_mode": "legacy-partial: row order, answers and length"
                                 + (" plus _id" if dataset == "niah_multikey_2" else ""),
                "raw_prediction_file": repo_path(fullkv_paths[dataset]),
                "raw_prediction_sha256": sha_file(fullkv_paths[dataset]),
            })

    assert len(targets) == 500 and len({row["target_id"] for row in targets}) == 500
    write_jsonl(OUT / "target_inputs.jsonl", targets)
    write_jsonl(OUT / "fullkv_quality.jsonl", quality)

    means = {}
    for dataset in MAX_NEW_TOKENS:
        scores = [row["score_percent"] for row in quality if row["dataset"] == dataset]
        means[dataset] = math.fsum(scores) / len(scores)
    assert round(means["hotpotqa"], 2) == 53.48
    assert round(means["gov_report"], 2) == 33.17
    assert means["niah_multikey_2"] == 50.0

    comparison = json.loads(STRICT_COMPARISON.read_text())
    assert comparison["valid"] is True
    assert comparison["baseline_head"] == "f5f3df3c4fe4c4af39c9074db0baaa5e00770eab"
    assert comparison["candidate_head"] == "2f702c222186bdf17de996e981c52af467957ee5"
    smoke_by_id = comparison["rows"]
    target_by_id = {row["target_id"]: row for row in targets}
    for target_id, smoke in smoke_by_id.items():
        assert target_id in target_by_id
        assert smoke["input_sha256"] == target_by_id[target_id]["b_final_input_sha256"]
        for arm in ("full", "far_0.05", "dual_0.05"):
            assert smoke["arms"][arm]["baseline_generated_ids"] is not None
            assert smoke["arms"][arm]["candidate_generated_ids"] is not None

    decisions = []
    for target in targets:
        target_id = target["target_id"]
        arms = {
            "dense_fullkv": {
                "quality": "rescore",
                "token_bridge": "rerun",
                "reason": "prediction/answers/length reusable with legacy-partial scope; "
                          "full generated token IDs and effective producer cap/model revision absent",
                "quality_source": "fullkv_quality.jsonl",
            },
        }
        for arm, head_key in (("seed_on_nfa", "baseline_head"),
                              ("strict_skip_nfa", "candidate_head")):
            if target_id in smoke_by_id:
                arms[arm] = {
                    "quality": "reuse",
                    "token_bridge": "reuse",
                    "gate_summary": "reuse-smoke-only",
                    "source_arm": "dual_0.05",
                    "source_git_head": comparison[head_key],
                    "source": repo_path(STRICT_COMPARISON),
                    "reason": "same target input SHA and 32-token HotpotQA protocol",
                }
            else:
                arms[arm] = {
                    "quality": "rerun",
                    "token_bridge": "rerun",
                    "gate_summary": "rerun",
                    "reason": "no raw for this target dataset/row under the required arm",
                }
        decisions.append({"target_id": target_id, "arms": arms})
    write_jsonl(OUT / "reuse_decisions.jsonl", decisions)

    summary = {
        "schema": "nfa-q0-reuse-audit-v1",
        "status": "reuse-audit-complete; target three-arm token bridge incomplete",
        "target_count": len(targets),
        "targets_per_dataset": {name: len(source_rows[name]) for name in source_rows},
        "target_max_new_tokens": MAX_NEW_TOKENS,
        "fullkv_official_rescore_means_percent": means,
        "fullkv_identity_checks": dict(identity_counts),
        "arm_actions": {
            "dense_fullkv": {
                "quality_rescore": 500,
                "token_bridge_reuse": 0,
                "token_bridge_rerun": 500,
            },
            "seed_on_nfa": {
                "quality_token_reuse": len(smoke_by_id),
                "quality_token_rerun": 500 - len(smoke_by_id),
                "source_git_head": comparison["baseline_head"],
            },
            "strict_skip_nfa": {
                "quality_token_reuse": len(smoke_by_id),
                "quality_token_rerun": 500 - len(smoke_by_id),
                "source_git_head": comparison["candidate_head"],
            },
        },
        "limitations": [
            "FullKV quality reuse is legacy-partial, not proof of exact source prompt/token identity or effective producer cap/model revision.",
            "FullKV raw has no generated token IDs; text predictions cannot reconstruct token-level first divergence.",
            "Five HotpotQA seed-on/strict-skip rows are smoke evidence, not a full 500-sample quality conclusion.",
            "B_final supplies target input identity only; its gate=full, reclaim=true and GovReport 32-token outputs are not reused as current strict-skip or official GovReport quality results.",
            "No GPU execution, paper, figure or Overleaf modification is part of this audit.",
        ],
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    source_files = [
        *dataset_paths.values(), *fullkv_paths.values(), STRICT_COMPARISON,
        ROOT / "benchmark/LongBench/eval.py",
        ROOT / "benchmark/LongBench/metrics.py",
        ROOT / "benchmark/LongBench/config/dataset2maxlen.json",
        ROOT / "benchmark/RULER/score_ruler.py",
        ROOT / "benchmark/RULER/pred_ruler.py",
        ROOT / "exp/trace/results/e109_lbv1_manifest.json",
        ROOT / "exp/trace/results/e116b_ruler65536k_formal_fullkv.json.manifest.json",
        ROOT / "exp/trace/results/e116b_ruler65536k_formal_fullkv.json.receipt.json",
    ]
    manifest = {
        "schema": "nfa-q0-target-manifest-v1",
        "target_protocol": {
            "model": "Qwen3-8B",
            "arms": ["dense_fullkv", "seed_on_nfa", "strict_skip_nfa"],
            "datasets": {"hotpotqa": 200, "gov_report": 200, "ruler_65536_niah_multikey_2": 100},
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation": "greedy; complete generated token IDs, prediction, score and first divergence required",
            "strict_skip": "sink4/SWA128/K1=128/K2=1024/near_pages32/near4096/page16; gate active and frozen from first decode forward; KV retain; no first-step seed/core-token compensation",
            "seed_on": "reproduce old first-step seed/core-token compensation at f5f3df3c; do not relabel old cache semantics as current strict-skip",
        },
        "source_sha256": {
            (repo_path(path) if path.is_relative_to(ROOT) else str(path)): sha_file(path)
            for path in source_files
        },
        "outputs": {
            "target_inputs": "target_inputs.jsonl",
            "fullkv_quality": "fullkv_quality.jsonl",
            "reuse_decisions": "reuse_decisions.jsonl",
            "audit_summary": "audit_summary.json",
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("Q0_REUSE_AUDIT_OK targets=500 fullkv_rescore=500 "
          f"seed_on_reuse={len(smoke_by_id)} strict_skip_reuse={len(smoke_by_id)}")


if __name__ == "__main__":
    main()
