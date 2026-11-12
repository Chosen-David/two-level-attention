"""校验 Qwen3-8B LongBench v1 三臂性能闭集并生成文件哈希。"""
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmark.LongBench.run_cap_manifest import build_lbv1_input_identity


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATA_ROOT = Path(
    "/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench")
CONFIG = REPO / "benchmark/LongBench/config/dataset2prompt.json"
IMPLEMENTATION_COMMIT = "cc1eca728aaf10e02553cb4b7efa1c1eab753718"
RECORDED_PRODUCER_COMMIT = "482911be08df9e4e2cc9bf2f9025fb52f9d9b547"
PROFILE_PRODUCER_COMMITS = {
    "quest": IMPLEMENTATION_COMMIT,
    "jssa": RECORDED_PRODUCER_COMMIT,
    "asynctls": IMPLEMENTATION_COMMIT,
}
ENGLISH16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
PROFILE_TASKS = ["narrativeqa", "qasper", "samsum"]
METHODS = {
    "quest": "quest_vendor_kernel",
    "jssa": "jssa_official",
    "asynctls": "asynctls_vendor",
}
EXPECTED_TREATMENT = {
    "quest": {
        "token_budget": 1024, "chunk_size": 16, "page_size": 16,
        "max_seq_len": 40960, "implementation": "official Quest CUDA operators",
    },
    "jssa": {
        "budget": 1024, "rank": 32, "page_size": 64,
        "implementation": "official vendored Triton store/score/indexed-attend",
    },
    "asynctls": {
        "token_budget": 1024, "block_size": 64, "top_kb": 128,
        "implementation": "AsyncTLS paper-semantic reproduction",
    },
}
PROFILE_KERNEL_MARKERS = {
    "quest": [
        "MaxPossibleSampleWithPagedKVCacheKernel",
        "BatchDecodeWithPagedKVCacheKernel",
        "radix_topk_one_block_kernel",
    ],
    "jssa": [
        "_jssa_project_store_contig_fp8_kernel",
        "_jssa_score_contig_kernel",
        "_split_attend_kernel",
    ],
    "asynctls": [],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(cells):
    metrics = ("ttft_ms", "tpot_ms", "request_wall_ms")
    per_task = {}
    for task in sorted({cell["task"] for cell in cells}):
        selected = [cell for cell in cells if cell["task"] == task]
        per_task[task] = {
            metric: statistics.median(
                cell["request_timing"][metric] for cell in selected)
            for metric in metrics
        }
        per_task[task]["requests"] = len(selected)
    return {
        "per_task": per_task,
        "macro_task_mean": {
            metric: statistics.mean(values[metric] for values in per_task.values())
            for metric in metrics
        },
        "all_request_median": {
            metric: statistics.median(
                cell["request_timing"][metric] for cell in cells)
            for metric in metrics
        },
    }


def check_identity(arm, payload, tasks, repeats, max_new, mode, producer_commit):
    identity = payload["identity"]
    treatment = identity["treatment"]
    assert identity["schema"] == "longbench-v1-vendor-performance-identity-v1"
    assert treatment["method"] == METHODS[arm]
    assert treatment["model"] == "Qwen3-8B"
    assert treatment["effective_cap"] == 31500
    assert treatment["max_new_tokens"] == max_new
    assert treatment["execution"] == "eager"
    assert treatment["batch_size"] == 1
    assert treatment["decode_protocol"] == (
        "fixed length; EOS recorded but not used for early stop")
    for key, value in EXPECTED_TREATMENT[arm].items():
        assert treatment[key] == value, (arm, key, treatment[key], value)
    assert identity["tasks"] == tasks
    assert identity["sample_indices"] == [0]
    assert identity["warmup_requests"] == 1
    assert identity["repeats"] == repeats
    assert identity["measurement_mode"] == mode
    assert identity["producer"]["git_commit"] == producer_commit
    for relative, recorded_sha in identity["source_files_sha256"].items():
        assert sha256(REPO / relative) == recorded_sha, (arm, relative)
    assert payload["verification"] == "pending-independent-verification"
    assert payload["profiler_clean_e2e_isolated"] is True
    assert payload["host_overhead_derivable_by_subtraction"] is False


def check_cells(payload, tasks, repeats, max_new):
    expected = {(task, 0, repeat) for task in tasks for repeat in range(repeats)}
    cells = payload["cells"]
    keys = [(cell["task"], cell["sample_index"], cell["repeat"]) for cell in cells]
    assert len(keys) == len(set(keys)) == len(expected)
    assert set(keys) == expected
    assert payload["status"] == "complete"
    prompts = load(CONFIG)
    for cell in cells:
        task = cell["task"]
        source_path = DATA_ROOT / f"{task}.jsonl"
        source_bytes = source_path.read_bytes()
        assert cell["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line]
        fingerprint = build_lbv1_input_identity(rows, prompts[task])["source_fingerprint"]
        assert cell["source_fingerprint"] == fingerprint
        timing = cell["request_timing"]
        assert timing["schema"] == "longbench-v1-request-wall-v1"
        assert timing["actual_generated_tokens"] == max_new
        assert len(timing["generated_token_ids"]) == max_new
        assert timing["eos_early_stop"] is False
        assert timing["host_readback_in_decode_loop"] is True
        for metric in ("ttft_ms", "decode_after_first_wall_ms", "tpot_ms",
                       "request_wall_ms"):
            assert math.isfinite(timing[metric]) and timing[metric] > 0
        assert math.isclose(
            timing["decode_after_first_wall_ms"], timing["tpot_ms"] * (max_new - 1),
            rel_tol=1e-12, abs_tol=1e-9)
    if max_new == 33:
        assert payload["summary"] == summarize(cells)


def main():
    arms = {}
    inputs = {}
    source_hashes = {}
    for arm in METHODS:
        clean_path = ROOT / arm / "clean-e2e.json"
        clean = load(clean_path)
        check_identity(
            arm, clean, ENGLISH16, 3, 33, "clean-e2e",
            RECORDED_PRODUCER_COMMIT)
        check_cells(clean, ENGLISH16, 3, 33)
        for cell in clean["cells"]:
            key = (cell["task"], cell["sample_index"])
            identity = (cell["source_sha256"], cell["source_fingerprint"], cell["input"])
            if key in inputs:
                assert inputs[key] == identity, (arm, key, "cross-arm input mismatch")
            else:
                inputs[key] = identity
        assert len(inputs) >= 16

        profile_path = ROOT / arm / "operator-profile.json"
        profile = load(profile_path)
        check_identity(
            arm, profile, PROFILE_TASKS, 1, 9, "operator-profile",
            PROFILE_PRODUCER_COMMITS[arm])
        check_cells(profile, PROFILE_TASKS, 1, 9)
        assert profile["identity"]["source_files_sha256"] == (
            clean["identity"]["source_files_sha256"])
        assert profile["identity"]["calibration"] == clean["identity"]["calibration"]
        clean_inputs = {
            cell["task"]: cell["input"] for cell in clean["cells"]}
        for cell in profile["cells"]:
            assert cell["input"] == clean_inputs[cell["task"]]
        kernel_path = ROOT / arm / "operator-profile.kernels.txt"
        report = kernel_path.read_text(encoding="utf-8")
        for marker in PROFILE_KERNEL_MARKERS[arm]:
            assert marker in report, (arm, marker)
        nsys_path = ROOT / arm / "operator-profile.nsys-rep"
        assert nsys_path.stat().st_size > 0

        source_hashes[arm] = clean["identity"]["source_files_sha256"]
        arms[arm] = {
            "method": METHODS[arm],
            "clean_e2e": {
                "path": str(clean_path.relative_to(ROOT)),
                "sha256": sha256(clean_path), "cells": 48,
                "summary": clean["summary"],
                "device": clean["device"],
            },
            "operator_profile": {
                "path": str(profile_path.relative_to(ROOT)),
                "sha256": sha256(profile_path), "cells": 3,
                "generated_tokens_per_cell": 9,
                "nsys_receipt": str(nsys_path.relative_to(ROOT)),
                "kernel_report": str(kernel_path.relative_to(ROOT)),
                "kernel_markers_observed": PROFILE_KERNEL_MARKERS[arm],
                "note": "profiler observations are separate; no host-time subtraction",
            },
        }
    assert len(inputs) == 16
    payload = {
        "schema": "qwen3-lbv1-vendor-performance-v1",
        "status": "complete",
        "verification": "pending-independent-verification",
        "scope": (
            "B1 eager English16 sample0; one warmup and three measured requests; "
            "33 generated tokens per clean request; macro task mean and all-request median"),
        "not_inferred": [
            "not a 3750-request distribution",
            "no kernel-only to complete-request substitution",
            "no clean-e2e versus profiler subtraction",
        ],
        "cross_arm_input_identity": "16/16 matched",
        "clean_cells": 144,
        "provenance": {
            "recorded_producer_commit": RECORDED_PRODUCER_COMMIT,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "profile_producer_commits": PROFILE_PRODUCER_COMMITS,
            "mapping": "recorded old HEAD plus verified source hashes identifies the measured code",
            "current_source_hashes_match_receipts": True,
            "source_files_sha256": source_hashes,
        },
        "arms": arms,
    }
    output = ROOT / "summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
        and path.suffix != ".sqlite" and "__pycache__" not in path.parts)
    manifest = {
        "schema": "qwen3-lbv1-vendor-performance-file-manifest-v1",
        "verification": "pending-independent-verification",
        "excluded_derived_files": "nsys SQLite caches are regenerable from archived .nsys-rep",
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "complete", "clean_cells": 144,
        "cross_arm_inputs": len(inputs),
        "macro_task_mean": {
            arm: values["clean_e2e"]["summary"]["macro_task_mean"]
            for arm, values in arms.items()},
        "manifest_files": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
