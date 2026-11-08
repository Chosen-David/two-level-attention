#!/usr/bin/env python3
"""只读重建 P0 decode TPOT 覆盖表；不执行模型或 GPU 测量。"""
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
AUDIT_BASE = "41e7c38d42cd69b94d8e0cf4747fafb4c2032315"
CURRENT = "4bc6aa98730e060489f742bfe1875d687237e422"
RAGGED = "c5be4f69262a7eb2287e396589e18d8a91fd79ff"
ITER7 = "cc7d6aa45c791d610e4d8365131962f1e30622c0"
STRICT = "2f702c222186bdf17de996e981c52af467957ee5"
GRAPH_READY = "40756b6c925ad0caa5741ccbc305299fb8189567"
MAIN_VERDICT = ROOT / "exp/trace/results/nfa_decode_main_20261102/nfa_decode_main_verdict.json"
FLASHINFER = ROOT / "benchmark/efficiency/results/nfa_flashinfer_20261104/e2e/ruler32k.json"
PRODUCTION_PATHS = [
    "sparse_attn",
    "benchmark/efficiency/benchmark_nfa_end_to_end.py",
    "benchmark/efficiency/benchmark_nfa_vs_fa2.py",
]
LENGTHS = (32768, 65536, 131072)
BATCHES = (1, 4)
EXECUTIONS = ("eager", "graph")
ITER7_DIRS = {
    (32768, 1): "nfa-iter7-decode-32k-b1",
    (65536, 1): "nfa-iter7-decode-64k-b1",
    (131072, 1): "nfa-iter7-decode-128k-b1",
    (32768, 4): "nfa-iter7-decode-32k-b4",
    (65536, 4): "nfa-iter7-decode-64k-b4",
}
CURRENT_DIRS = {
    (32768, 1): ["nfa-ragged-decode-32k-b1"],
    (65536, 4): ["nfa-ragged-decode-64k-b4"],
    (131072, 4): ["nfa-ragged-decode-128k-b4"],
}
EXPECTED_SELECTOR = {
    "subspace": "tail",
    "subspace_cmp_ratio": 4,
    "level1_topk": 128,
    "level2_topk": 1024,
    "near_pages": 32,
    "near_tokens": "off",
    "near_len": 4096,
    "enable_layer_skip": False,
    "sink_tokens": 4,
    "sliding_window_size": 128,
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


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(compact(row) + "\n")


def rel(path):
    return path.relative_to(OUT).as_posix()


def repo_rel(path):
    return path.relative_to(ROOT).as_posix()


def git(*args, check=True):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result


def is_ancestor(ancestor, descendant):
    return git("merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0


def changed_files(start, end, paths):
    result = git("diff", "--name-only", f"{start}..{end}", "--", *paths)
    return sorted(filter(None, result.stdout.splitlines()))


def load_manifests():
    expected = set(ITER7_DIRS.values()) | {
        "nfa-ragged-decode-32k-b1",
        "nfa-ragged-decode-32k-b1-eager-repeat",
        "nfa-ragged-decode-64k-b4",
        "nfa-ragged-decode-128k-b4",
    }
    actual = {path.name for path in RAW.iterdir() if path.is_dir()}
    assert actual == expected, (actual, expected)
    manifests = {}
    inventory = []
    for name in sorted(expected):
        directory = RAW / name
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        assert manifest["status"] == "complete"
        assert manifest["gpu_hygiene"] == "clean"
        assert config["max_new_tokens"] == 512 and config["warmup_steps"] == 3
        assert config["sample_index"] == 0 and config["task"] == "niah_multikey_1"
        expected_executions = list(config["executions"])
        assert len(manifest["runs"]) == 4 * len(expected_executions)
        monitor_path = directory / "gpu_monitor.jsonl"
        monitor_rows = [json.loads(line) for line in monitor_path.read_text().splitlines()]
        assert monitor_rows and all(not row["unknown_apps"] for row in monitor_rows)
        per_execution = {}
        for execution in expected_executions:
            runs = [row for row in manifest["runs"] if row["execution"] == execution]
            assert [(row["method"], row["leg"]) for row in runs] == [
                ("baseline", "a"), ("candidate", "a"),
                ("candidate", "b"), ("baseline", "b"),
            ]
            generated = {}
            for run in runs:
                assert run["returncode"] == 0
                json_path = directory / Path(run["json"]).name
                log_path = directory / Path(run["log"]).name
                assert json_path.is_file() and log_path.is_file()
                result = json.loads(json_path.read_text())
                assert result["method"] == "nfa"
                assert result["execution"] == execution
                assert result["context_length"] == config["context_length"]
                assert result["batch_size"] == config["batch_size"]
                assert result["max_new_tokens"] == 512
                assert result["warmup_steps_excluded_from_steady"] == 3
                assert result["measurement_mode"] == "latency"
                assert result["timing_quality"] == "clean-device-history"
                assert result["host_readback_in_decode_loop"] is False
                assert result["input"]["sha256"] == manifest["summary"][execution]["input_sha256"]
                assert result["input"]["sample_index"] == 0
                assert result["nfa_selector_config"] == EXPECTED_SELECTOR
                assert result["gate_mode"] == "dynamic" and result["gate_epsilon"] == 0.05
                tokens = result["generated_token_ids"]
                assert len(tokens) == config["batch_size"]
                assert all(len(row) == 512 for row in tokens)
                steps = result["tpot_ms_all"]
                assert len(steps) == result["decode_steps"]
                steady = statistics.median(steps[3:])
                assert math.isclose(steady, result["steady_tpot_ms_median"], abs_tol=1e-12)
                arm = run["method"]
                generated.setdefault(arm, {})[run["leg"]] = tokens
                source = manifest["sources"][arm]
                inventory.append({
                    "run_key": f"{name}/{execution}/{arm}/{run['leg']}",
                    "directory": name,
                    "context_length": config["context_length"],
                    "batch_size": config["batch_size"],
                    "execution": execution,
                    "method": "nfa",
                    "comparison_arm": arm,
                    "abba_leg": run["leg"],
                    "source_git_head": source["head"],
                    "source_diff_sha256": source["diff_sha256"],
                    "input_sha256": result["input"]["sha256"],
                    "input_sample_index": 0,
                    "max_new_tokens": 512,
                    "warmup_steps": 3,
                    "generated_token_ids_sha256": sha_bytes(compact(tokens).encode()),
                    "steady_tpot_ms": result["steady_tpot_ms_median"],
                    "gpu_hygiene": "clean",
                    "raw_json": rel(json_path),
                    "raw_json_sha256": sha_file(json_path),
                    "raw_log": rel(log_path),
                    "raw_log_sha256": sha_file(log_path),
                    "gpu_monitor": rel(monitor_path),
                    "gpu_monitor_sha256": sha_file(monitor_path),
                })
            for arm in ("baseline", "candidate"):
                assert generated[arm]["a"] == generated[arm]["b"]
            assert generated["baseline"]["a"] == generated["candidate"]["a"]
            summary = manifest["summary"][execution]
            assert summary["baseline_repeat_token_identity"] is True
            assert summary["candidate_repeat_token_identity"] is True
            assert summary["baseline_candidate_token_identity"] is True
            for arm in ("baseline", "candidate"):
                runs_for_arm = [row for row in inventory
                                if row["directory"] == name
                                and row["execution"] == execution
                                and row["comparison_arm"] == arm]
                median = statistics.median(row["steady_tpot_ms"] for row in runs_for_arm)
                assert math.isclose(median, summary[f"{arm}_median_ms"], abs_tol=1e-12)
            per_execution[execution] = summary
        manifests[name] = manifest
    assert len(inventory) == 68
    return manifests, inventory


def build_lineage():
    relations = [
        (GRAPH_READY, CURRENT), (CURRENT, GRAPH_READY),
        (ITER7, RAGGED), (RAGGED, CURRENT), (STRICT, ITER7),
        (CURRENT, AUDIT_BASE), (RAGGED, AUDIT_BASE), (ITER7, AUDIT_BASE),
        (STRICT, AUDIT_BASE), (GRAPH_READY, AUDIT_BASE),
    ]
    ancestry = [{"ancestor": a, "descendant": b, "result": is_ancestor(a, b)}
                for a, b in relations]
    expected = [True, False, True, True, True, True, True, True, True, True]
    assert [row["result"] for row in ancestry] == expected
    ragged_to_current = changed_files(RAGGED, CURRENT, PRODUCTION_PATHS)
    current_to_base = changed_files(CURRENT, AUDIT_BASE, PRODUCTION_PATHS)
    iter7_to_ragged = changed_files(ITER7, RAGGED, PRODUCTION_PATHS)
    assert ragged_to_current == [] and current_to_base == []
    assert "sparse_attn/indexer/nfa_layer_batch_selector.py" in iter7_to_ragged
    return {
        "audit_base": AUDIT_BASE,
        "production_paths": PRODUCTION_PATHS,
        "ancestry": ancestry,
        "ragged_to_current_production_diff": ragged_to_current,
        "current_to_audit_base_production_diff": current_to_base,
        "iter7_to_ragged_production_diff": iter7_to_ragged,
        "scope": "c5be4f69/4bc6aa98 与审计基线的 NFA 生产和 e2e 入口一致；cc7d6aa4 的 selector 不同",
    }


def raw_refs(rows):
    refs = []
    package_root = OUT.relative_to(ROOT).as_posix()
    for row in rows:
        refs.append({"path": f"{package_root}/{row['raw_json']}",
                     "sha256": row["raw_json_sha256"]})
        refs.append({"path": f"{package_root}/{row['raw_log']}",
                     "sha256": row["raw_log_sha256"]})
        refs.append({"path": f"{package_root}/{row['gpu_monitor']}",
                     "sha256": row["gpu_monitor_sha256"]})
    unique = {(row["path"], row["sha256"]) for row in refs}
    return [{"path": path, "sha256": sha} for path, sha in sorted(unique)]


def build_coverage(inventory, manifests):
    main = json.loads(MAIN_VERDICT.read_text())
    assert main["correctness"]["fixed_128_tokens_per_row"] is True
    assert main["gpu_hygiene"]["status"] == "clean"
    main_rows = {(row["context_length"], row["batch_size"], row["execution"]): row
                 for row in main["rows"]}
    assert len(main_rows) == 18  # 含不在本次目标内的 B2。
    coverage = []
    for method in ("nfa", "fa2"):
        for length in LENGTHS:
            for batch in BATCHES:
                for execution in EXECUTIONS:
                    key = (length, batch, execution)
                    old = main_rows[key]
                    unit = f"{method}/L{length}/B{batch}/{execution}"
                    refs = []
                    heads = []
                    existing_scope = "historical-128token-summary"
                    evidence_class = "B-historical-summary"
                    existing_tokens = 128
                    existing_inputs = None
                    existing_repeats = None
                    existing_warmup = None
                    token_scope = "eager/Graph; no target repeat identity"
                    pair_method = "fa2" if method == "nfa" else "nfa"
                    current_equivalent = False
                    reason = "只有 128-token 历史汇总；不是 512-token、5 输入×3 重复的正式单元"
                    if method == "nfa":
                        cell = (length, batch)
                        current_dirs = list(CURRENT_DIRS.get(cell, []))
                        if cell == (32768, 1) and execution == "eager":
                            current_dirs.append("nfa-ragged-decode-32k-b1-eager-repeat")
                        current_runs = [row for row in inventory
                                        if row["directory"] in current_dirs
                                        and row["execution"] == execution
                                        and row["comparison_arm"] == "candidate"]
                        iter7_dir = ITER7_DIRS.get(cell)
                        iter7_runs = [row for row in inventory
                                      if row["directory"] == iter7_dir
                                      and row["execution"] == execution
                                      and row["comparison_arm"] == "candidate"]
                        if current_runs:
                            refs = raw_refs(current_runs)
                            heads = sorted({row["source_git_head"] for row in current_runs})
                            existing_scope = "current-production-single-input-sanity"
                            evidence_class = "B-current-sanity"
                            existing_tokens = 512
                            existing_inputs = 1
                            existing_repeats = len(current_runs)
                            existing_warmup = 3
                            token_scope = "NFA repeat + old/new NFA; eager/Graph checked in Iter8"
                            pair_method = "nfa"
                            current_equivalent = True
                            reason = ("当前 NFA 仅有 sample0 的 512-token sanity；对照臂是旧 NFA，"
                                      "缺 dense FA2、其余 4 输入和目标独立重复")
                        elif iter7_runs:
                            refs = raw_refs(iter7_runs)
                            heads = sorted({row["source_git_head"] for row in iter7_runs})
                            existing_scope = "iter7-historical-single-input-sanity"
                            evidence_class = "B-historical-sanity"
                            existing_tokens = 512
                            existing_inputs = 1
                            existing_repeats = len(iter7_runs)
                            existing_warmup = 3
                            token_scope = "NFA repeat + old/new NFA; not current selector"
                            pair_method = "nfa"
                            reason = ("iter7 NFA raw 的 selector 与当前生产不同，且只有 sample0、"
                                      "NFA-vs-NFA；不能升级为当前 NFA/FA2 正式单元")
                        else:
                            refs = [{"path": repo_rel(MAIN_VERDICT),
                                     "sha256": sha_file(MAIN_VERDICT)}]
                    else:
                        refs = [{"path": repo_rel(MAIN_VERDICT),
                                 "sha256": sha_file(MAIN_VERDICT)}]
                    coverage.append({
                        "unit_id": unit,
                        "method": method,
                        "context_length": length,
                        "batch_size": batch,
                        "execution": execution,
                        "classification": "C",
                        "action": "rerun",
                        "formal_reuse": False,
                        "target_max_new_tokens": 512,
                        "target_fixed_inputs": 5,
                        "target_independent_repeats": 3,
                        "target_warmup_steps": 3,
                        "target_pairing": "same-input NFA/FA2 ABBA; independent processes",
                        "existing_evidence_class": evidence_class,
                        "existing_scope": existing_scope,
                        "existing_max_new_tokens": existing_tokens,
                        "existing_fixed_inputs": existing_inputs,
                        "existing_measured_repeats": existing_repeats,
                        "existing_warmup_steps": existing_warmup,
                        "existing_pair_method": pair_method,
                        "existing_source_heads": ";".join(heads),
                        "current_production_equivalent": current_equivalent,
                        "existing_gpu_hygiene": "clean",
                        "existing_token_identity_scope": token_scope,
                        "historical_nfa_tpot_ms": old["nfa_steady_tpot_ms_median"],
                        "historical_fa2_tpot_ms": old["fa2_steady_tpot_ms_median"],
                        "raw_references": refs,
                        "reason": reason,
                    })
    assert len(coverage) == 24 and len({row["unit_id"] for row in coverage}) == 24
    assert sum(row["current_production_equivalent"] for row in coverage) == 6
    assert sum(row["method"] == "nfa" and row["existing_max_new_tokens"] == 512
               for row in coverage) == 12
    assert not any(row["formal_reuse"] for row in coverage)
    return coverage


def main():
    lineage = build_lineage()
    manifests, inventory = load_manifests()
    coverage = build_coverage(inventory, manifests)
    write_json(OUT / "source_lineage.json", lineage)
    write_jsonl(OUT / "run_inventory.jsonl", inventory)
    write_jsonl(OUT / "coverage.jsonl", coverage)
    csv_rows = []
    for row in coverage:
        flat = dict(row)
        flat["raw_paths"] = ";".join(ref["path"] for ref in row["raw_references"])
        flat["raw_sha256"] = ";".join(ref["sha256"] for ref in row["raw_references"])
        del flat["raw_references"]
        csv_rows.append(flat)
    with (OUT / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    flashinfer = json.loads(FLASHINFER.read_text())
    assert flashinfer["arguments"]["max_new_tokens"] == 512
    assert flashinfer["arguments"]["methods"] == ["nfa", "flashinfer"]
    summary = {
        "schema": "nfa-p0-performance-coverage-v1",
        "status": "coverage-audit-complete; formal 5x3 NFA/FA2 matrix incomplete",
        "target_method_units": 24,
        "target_comparison_cells": 12,
        "formal_reuse_units": 0,
        "rerun_units": 24,
        "nfa_512_single_input_units": 12,
        "current_nfa_512_sanity_units": 6,
        "iter7_only_nfa_512_sanity_units": 6,
        "archived_abba_manifests": len(manifests),
        "archived_runs": len(inventory),
        "archived_run_returncodes": dict(Counter(
            "success" for _ in inventory)),
        "classification_counts": dict(Counter(row["classification"] for row in coverage)),
        "excluded_method_mismatch": {
            "path": repo_rel(FLASHINFER),
            "sha256": sha_file(FLASHINFER),
            "reason": "FlashInfer 不是 dense FA2；32K 单输入结果不能填入本表",
        },
        "limitations": [
            "所有正式单元都要求同一输入的 NFA/FA2 ABBA、5 个固定输入和 3 次独立重复；现有 NFA-vs-NFA 不折抵正式配对。",
            "c5be4f69/4bc6aa98 只在限定生产路径和 e2e 入口上与审计基线一致，不表示整个 Git 树逐字节相同。",
            "旧 128-token FA2/NFA 汇总保留为历史参考，不与 512-token 数字拼接成新倍率。",
            "本审计只复制和校验既有 raw；没有执行 GPU、修改生产代码、论文、图或 Overleaf。",
        ],
    }
    write_json(OUT / "audit_summary.json", summary)
    manifest = {
        "schema": "nfa-p0-performance-manifest-v1",
        "audit_base": AUDIT_BASE,
        "target_protocol": {
            "model": "Qwen3-8B",
            "methods": ["nfa", "fa2"],
            "lengths": LENGTHS,
            "batch_sizes": BATCHES,
            "executions": EXECUTIONS,
            "max_new_tokens": 512,
            "fixed_inputs": 5,
            "independent_repeats": 3,
            "warmup_steps": 3,
            "pairing": "same-input ABBA; independent subprocesses",
            "required_receipts": ["per-step TPOT", "complete generated token IDs",
                                  "input/model/source identity", "gpu_hygiene"],
        },
        "source_anchors": {
            "current_production": CURRENT,
            "ragged_candidate": RAGGED,
            "iter7_candidate": ITER7,
            "strict_skip_functional": STRICT,
            "graph_ready": GRAPH_READY,
        },
        "source_evidence_sha256": {
            repo_rel(MAIN_VERDICT): sha_file(MAIN_VERDICT),
            repo_rel(FLASHINFER): sha_file(FLASHINFER),
            "ITERATIONS.md": sha_file(ROOT / "ITERATIONS.md"),
        },
        "raw_origin": {
            "pc40": ["nfa-iter7-decode-32k-b1", "nfa-iter7-decode-64k-b1",
                     "nfa-iter7-decode-128k-b1", "nfa-iter7-decode-32k-b4"],
            "pc31": ["nfa-iter7-decode-64k-b4", "nfa-ragged-decode-32k-b1",
                     "nfa-ragged-decode-32k-b1-eager-repeat",
                     "nfa-ragged-decode-64k-b4", "nfa-ragged-decode-128k-b4"],
            "note": "原目录下 manifest、逐 run JSON/log 和 gpu_monitor 已原样复制到 raw/；路径字符串保留生成时绝对路径",
        },
        "outputs": ["coverage.csv", "coverage.jsonl", "run_inventory.jsonl",
                    "source_lineage.json", "audit_summary.json", "metadata.json"],
    }
    write_json(OUT / "manifest.json", manifest)
    metadata = {
        "units": "TPOT ms/decode token；历史数值仅供定位，不重新计算跨协议加速比",
        "classification": "C 表示正式目标单元必须补测；B-current/B-historical 只描述已有 raw 的有限复用范围",
        "repeat_semantics": "目标是每个固定输入 3 次独立重复；ABBA 两腿和重复轮次必须分别落盘，不得混为单一均值",
        "token_identity": "NFA 优化前后身份不证明 NFA 与 FA2 token 相同；稀疏近似分叉须单列",
        "statistics": "不从不同源码、不同 decode 长度或不同输入的结果拼接正式均值",
        "gpu_hygiene": "manifest=clean 且 gpu_monitor 的 unknown_apps 全为空；新实验仍需重新锁卡和旁路监控",
    }
    write_json(OUT / "metadata.json", metadata)
    files = sorted(path for path in OUT.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(
        f"{sha_file(path)}  {rel(path)}\n" for path in files))
    print("P0_PERFORMANCE_COVERAGE_OK units=24 rerun=24 current_nfa_sanity=6 "
          f"manifests={len(manifests)} runs={len(inventory)}")


if __name__ == "__main__":
    main()
