#!/usr/bin/env python3
"""从既有 formal 文件与来源记录生成 baseline 可复现性索引；不运行模型或 GPU。"""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "exp/trace/results"
REPORT = ROOT / "research/docs/论文baseline矩阵补齐报告_20261031_by_claude.md"
ACQUISITION = ROOT / "exp/paper/evidence/baseline-followup/source_acquisition.json"
PAPERS = ROOT / "agent_doc/advice/other_paper_pdf/papers.json"
VENDOR_README = ROOT / "exp/vendor_baselines/README.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_record(relative):
    path = ROOT / relative
    present = path.is_file()
    return {
        "path": relative,
        "present": present,
        "sha256": sha256(path) if present else None,
    }


def source_row(method, upstream, integration, records, scope, execution):
    return {
        "method": method,
        "upstream": upstream,
        "integration": integration,
        "source_records": [file_record(path) for path in records],
        "scope": scope,
        "qwen3_8b_execution_status": execution,
    }


def result_task_stats(data):
    scores = data.get("scores", {})
    if scores and all(isinstance(value, dict) for value in scores.values()):
        tasks = sorted({task for cell in scores.values() for task in cell})
        cells = sorted(scores)
    else:
        tasks = sorted(scores)
        cells = []
    counts = data.get("n", {})
    if counts and all(isinstance(value, dict) for value in counts.values()):
        sample_counts = {
            task: sorted({cell[task] for cell in counts.values() if task in cell})
            for task in tasks
        }
    else:
        sample_counts = {task: [counts[task]] for task in tasks if task in counts}
    return tasks, cells, sample_counts


# 路径来自当前快照；缺失文件保留为 recover 项，不据此判定实验未做。
EVIDENCE_SPECS = [
    ("fullkv-ruler-32k", "FullKV", "RULER", 32768, "e116e_ruler32_formal_fullkv.json", 11, "archived"),
    ("fullkv-ruler-64k", "FullKV", "RULER", 65536, "e119_ruler64k_formal_fullkv.json", 11, "archived"),
    ("fullkv-ruler-128k", "FullKV", "RULER", 131072, "e119_ruler128k_formal_fullkv.json", 11, "archived"),
    ("quest-ruler-32k", "Quest", "RULER", 32768, "e119_ruler32k_formal_quest_vendor.json", 11, "archived"),
    ("quest-ruler-64k", "Quest", "RULER", 65536, "e119_ruler64k_formal_quest_vendor.json", 11, "archived"),
    ("quest-ruler-128k", "Quest", "RULER", 131072, "e119_ruler128k_formal_quest_vendor.json", 11, "archived"),
    ("asynctls-ruler-32k", "AsyncTLS", "RULER", 32768, "asynctls_r32k_formal.json", 11, "archived"),
    ("jssa-ruler-32k", "JSSA", "RULER", 32768, "jssa_r32k_formal.json", 11, "archived"),
    ("fasa-ruler-32k-legacy", "FASA", "RULER", 32768, "fasa_32k.json", 11, "legacy-partial"),
    ("fasa-longbench-v1-legacy", "FASA", "LongBench-v1", None, "fasa_lb_16.json", 16, "legacy-partial"),
    ("hisa-ruler-32k-reported", "HISA", "RULER", 32768, "../results_ruler/ruler32k_all_table.json", 11, "reported"),
    ("fullkv-longbench-v1-reported", "FullKV", "LongBench-v1", None, "qwen3_8b_longbench_fullkv.json", 16, "reported"),
    ("quest-longbench-v1-reported", "Quest", "LongBench-v1", None, "qwen3_8b_longbench_quest_vendor.json", 16, "reported"),
    ("fasa-longbench-v1-reported", "FASA", "LongBench-v1", None, "qwen3_8b_longbench_fasa_vendor.json", 16, "reported"),
    ("jssa-longbench-v1-reported", "JSSA", "LongBench-v1", None, "qwen3_8b_longbench_jssa_vendor.json", 16, "reported"),
    ("asynctls-longbench-v1-reported", "AsyncTLS", "LongBench-v1", None, "qwen3_8b_longbench_asynctls_vendor.json", 16, "reported"),
    ("hisa-longbench-v1-not-run", "HISA", "LongBench-v1", None, None, 16, "not-run"),
]


def build_evidence(spec):
    evidence_id, method, dataset, length, name, expected_tasks, kind = spec
    if name is None:
        return {
            "evidence_id": evidence_id,
            "method": method,
            "dataset": dataset,
            "model": "Qwen3-8B",
            "context_length": length,
            "status": "not-run",
            "expected_tasks": expected_tasks,
            "evidence_origin": "research report",
            "report": file_record(str(REPORT.relative_to(ROOT))),
            "action": "none; absence is an explicit experimental gap, not an indexing failure",
        }
    relative = name if name.startswith("exp/") else f"exp/trace/results/{name}"
    # HISA 的共享汇总位于 exp/results_ruler，而非 exp/trace/results。
    if name.startswith("../"):
        relative = f"exp/trace/results/{name}"
    result_path = (ROOT / relative).resolve()
    relative = str(result_path.relative_to(ROOT))
    record = file_record(relative)
    row = {
        "evidence_id": evidence_id,
        "method": method,
        "dataset": dataset,
        "model": "Qwen3-8B",
        "context_length": length,
        "expected_tasks": expected_tasks,
        "result": record,
        "manifest": file_record(relative + ".manifest.json"),
        "receipt": file_record(relative + ".receipt.json"),
        "evidence_origin": "archived file" if kind != "reported" else "research report",
        "report": file_record(str(REPORT.relative_to(ROOT))),
    }
    if not record["present"]:
        row["status"] = "reported-not-archived" if kind == "reported" else "referenced-missing"
        row["action"] = "recover or synchronize the referenced artifact; do not rerun solely because this snapshot lacks it"
        return row

    data = read_json(result_path)
    tasks, cells, sample_counts = result_task_stats(data)
    row.update({
        "task_count": len(tasks),
        "tasks": tasks,
        "cells": cells,
        "sample_counts": sample_counts,
        "incomplete_cells": data.get("incomplete_cells"),
        "result_structure_sha256": canonical_sha({
            "tasks": tasks,
            "cells": cells,
            "sample_counts": sample_counts,
        }),
    })
    manifest = None
    receipt = None
    if row["manifest"]["present"]:
        manifest = read_json(ROOT / row["manifest"]["path"])
        identity = manifest.get("run_identity", {})
        source_data = manifest.get("source_data_sha256", {})
        row.update({
            "run_id": manifest.get("run_id"),
            "manifest_version": manifest.get("manifest_version"),
            "model_identity": {
                "declared_local_path": identity.get("model_path"),
                "hf_revision": None,
                "status": "HF revision not closed by this sidecar",
            },
            "generation_command": manifest.get("command"),
            "generation_command_status": "recorded" if manifest.get("command") else "not recorded in formal sidecar",
            "extra_params": identity.get("extra_params", {}),
            "source_data_sha256": source_data,
            "source_data_identity_sha256": canonical_sha(source_data),
            "formal_script_sha256": identity.get("formal_script_sha256"),
            "scorer_sha256": identity.get("scorer_sha256"),
            "identity_modes": sorted({
                cell.get("identity_mode")
                for cell in manifest.get("cells", {}).values()
                if cell.get("identity_mode") is not None
            }),
        })
    if row["receipt"]["present"]:
        receipt = read_json(ROOT / row["receipt"]["path"])
        row.update({
            "receipt_status": receipt.get("status"),
            "receipt_result_sha_matches": receipt.get("result_sha256") == record["sha256"],
            "receipt_manifest_sha_matches": receipt.get("manifest_sha256") == row["manifest"]["sha256"],
        })
    for role, current in (
        ("scorer", "benchmark/RULER/score_ruler.py"),
        ("formal_script", "benchmark/RULER/score_ruler_formal.py"),
    ):
        current_record = file_record(current)
        row[f"current_{role}"] = current_record
        historical = row.get(f"{role}_sha256")
        row[f"{role}_matches_current_snapshot"] = (
            historical == current_record["sha256"] if historical else None
        )
    complete = (
        kind == "archived"
        and len(tasks) == expected_tasks
        and not data.get("incomplete_cells")
        and row["manifest"]["present"]
        and row["receipt"]["present"]
        and row.get("receipt_status") == "success"
        and row.get("receipt_result_sha_matches") is True
        and row.get("receipt_manifest_sha_matches") is True
    )
    if complete:
        row["status"] = "archived-complete"
        row["action"] = "reuse within its recorded protocol; retain source and scorer boundaries"
    else:
        row["status"] = "legacy-partial" if kind == "legacy-partial" else "archived-partial"
        missing = []
        if len(tasks) != expected_tasks:
            missing.append(f"tasks={len(tasks)}/{expected_tasks}")
        for sidecar in ("manifest", "receipt"):
            if not row[sidecar]["present"]:
                missing.append(sidecar)
        if row.get("receipt_result_sha_matches") is False:
            missing.append("receipt-result-sha-mismatch")
        if row.get("receipt_manifest_sha_matches") is False:
            missing.append("receipt-manifest-sha-mismatch")
        row["partial_reasons"] = missing
        row["action"] = "recover missing identity/sidecars or rescore from validated raw; do not treat this row as a complete formal unit"
    return row


def main():
    acquisition = read_json(ACQUISITION)
    acquired = {row["name"]: row for row in acquisition["sources"]}
    papers = read_json(PAPERS)
    jssa_paper = next(row for row in papers if row.get("method") == "JSSA")
    jssa_url = jssa_paper["code_source"]
    jssa_repo, jssa_commit = jssa_url.split("/tree/")

    def checked(name, repository, version=None):
        path = f"exp/vendor_baselines/{name}_vendor/SOURCE_CHECK.json"
        data = read_json(ROOT / path)
        return {
            "repository": repository,
            "commit": data["checked_commit"],
            "version": version,
            "matched_files": data["matched_files"],
            "identity_scope": data["scope"],
        }, path

    quest, quest_record = checked("quest", "https://github.com/mit-han-lab/Quest")
    fasa, fasa_record = checked("fasa", "https://github.com/wangyifei0047/FASA-ICLR2026")
    minference, minference_record = checked(
        "minference", "https://github.com/microsoft/MInference", "v0.1.6"
    )
    sparq_path = "exp/vendor_baselines/sparq_vendor/UPSTREAM.json"
    sparq_data = read_json(ROOT / sparq_path)
    ffd_path = "exp/vendor_baselines/ffd_vendor/UPSTREAM.json"
    ffd_data = read_json(ROOT / ffd_path)

    sources = [
        source_row("FullKV", [], "native-dense", ["benchmark/LongBench/pred.py", "benchmark/RULER/pred_ruler.py"], "原生 Qwen3 dense FullKV；不是 TLI gates-off，模型 HF revision 仍需按 run 闭合", "executed"),
        source_row("Quest", [quest], "modified-qwen3-quality-adapter", [quest_record, "exp/vendor_baselines/quest_vendor/README.md"], "Qwen3 适配不是未修改官方文件；质量路径 full-QK+mask 不是高效 sparse-KV runtime", "executed"),
        source_row("FASA", [fasa], "qwen3-calibration-dependent-adapter", [fasa_record, "exp/vendor_baselines/fasa_vendor/README.md"], "仅登记的 matched files 与上游逐字节一致；adapter 和校准产物身份需分别保留", "executed"),
        source_row("MInference", [minference], "qwen3-sparse-prefill-native-dense-decode", [minference_record, "exp/vendor_baselines/README.md"], "官方核心限定文件匹配；当前路径是稀疏 prefill、原生 dense decode，不能填 decode 稀疏质量或性能结论", "integration-only-in-this-index"),
        source_row("SparQ", [{"repository": sparq_data["repository"], "commit": sparq_data["commit"], "files": sparq_data["files"]}], "qwen3-quality-adapter-candidate", [sparq_path, "exp/vendor_baselines/README.md"], "完整驻留 KV 与 mean-V 重算边界保留；本索引没有正式质量结果", "integration-only-in-this-index"),
        source_row("FFD", [{"repository": ffd_data["repository"], "commit": ffd_data["commit"], "files": ffd_data["files"]}], "native-llama-only", [ffd_path, "exp/vendor_baselines/README.md"], "上游 Qwen2/3 不支持；只能作为 Llama 候选，不是 Qwen3-8B 可执行臂", "unsupported-for-qwen3"),
        source_row("JSSA", [{"repository": jssa_repo, "commit": jssa_commit, "paper": jssa_paper["source"]}], "qwen3-calibrated-adapter", ["agent_doc/advice/other_paper_pdf/papers.json", "exp/trace/results/jssa_r32k_verdict.json"], "固定上游 commit；basis 文件与运行配置仍需随各 run 的身份记录闭合", "executed"),
        source_row("ClusterKV", [{"repository": acquired["ClusterKV"]["url"], "commit": acquired["ClusterKV"]["commit"], "submodules": acquired["ClusterKV"].get("submodules", []), "source_files": acquired["ClusterKV"].get("source_files", {})}], "source-inventory-only", ["exp/paper/evidence/baseline-followup/source_acquisition.json", "exp/vendor_baselines/README.md"], "没有本地 faithful runtime adapter；不得以其他聚类算法替代", "source-only"),
        source_row("HISA", [{"repository": acquired[name]["url"], "commit": acquired[name]["commit"], "source_files": acquired[name].get("source_files", {})} for name in ("HISA_author_entry", "HISA_TileLang")], "source-resolved-torch-semantic-port", ["exp/paper/evidence/baseline-followup/source_acquisition.json", "exp/vendor_baselines/README.md"], "TransArch/TileLang 身份已固定；现有 Qwen 结果是语义复现，不是官方 HISA kernel 执行", "semantic-port-executed"),
        source_row("AsyncTLS", [], "faithful-paper-protocol-port", ["exp/trace/results/asynctls_r32k_reverify_verdict.json", "exp/trace/results/baseline4_status_verdict.json"], "官方代码未开源；当前为论文协议语义复现，不是官方代码直跑，预算 kt=512 与 NFA K2=1024 不同", "semantic-port-executed"),
    ]

    evidence = [build_evidence(spec) for spec in EVIDENCE_SPECS]
    by_method = {}
    for row in evidence:
        by_method.setdefault(row["method"], []).append(row["evidence_id"])
    for row in sources:
        row["evidence_ids"] = by_method.get(row["method"], [])
        assert all(record["present"] for record in row["source_records"])

    with (OUT / "baselines.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sources:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "evidence.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    status_counts = Counter(row["status"] for row in evidence)
    complete = [row for row in evidence if row["status"] == "archived-complete"]
    summary = {
        "schema": "nfa-baseline-reproducibility-index-v1",
        "status": "index-complete; recovery and experimental gaps remain explicit",
        "classification": "B",
        "model_scope": "Qwen3-8B baseline quality evidence; source-only candidates retained with limits",
        "methods": len(sources),
        "evidence_units": len(evidence),
        "status_counts": dict(sorted(status_counts.items())),
        "archived_complete_by_method": dict(sorted(Counter(row["method"] for row in complete).items())),
        "integration_only_methods": [row["method"] for row in sources if row["qwen3_8b_execution_status"] in {"integration-only-in-this-index", "source-only"}],
        "unsupported_qwen3_methods": [row["method"] for row in sources if row["qwen3_8b_execution_status"] == "unsupported-for-qwen3"],
        "official_source_unavailable_methods": [row["method"] for row in sources if not row["upstream"] and row["method"] != "FullKV"],
        "complete_units_with_closed_hf_revision": sum(row.get("model_identity", {}).get("hf_revision") is not None for row in complete),
        "complete_units_with_generation_command": sum(row.get("generation_command") is not None for row in complete),
        "complete_units_scorer_matches_current_snapshot": sum(row.get("scorer_matches_current_snapshot") is True for row in complete),
        "recovery_units": [
            {"evidence_id": row["evidence_id"], "status": row["status"], "action": row["action"]}
            for row in evidence if row["status"] in {"reported-not-archived", "referenced-missing", "archived-partial", "legacy-partial"}
        ],
        "limitations": [
            "报告中的完成状态与当前快照的文件存在性分开记录；reported-not-archived 不表示实验未执行。",
            "formal sidecar 的本地模型路径不等于 Hugging Face revision；未闭合字段保持 null。",
            "历史 scorer/formal script SHA 与当前快照不同只能表示需找回运行时 blob，不能改写成当前哈希。",
            "语义复现、修改版 quality adapter、官方 kernel 直跑是不同证据等级，不在本索引中混称。",
            "本包只建立索引和文件身份，不重评分、不运行生成、不启动 GPU，也不修改论文、图或 Overleaf。",
        ],
    }
    write_json(OUT / "summary.json", summary)

    source_files = sorted({
        record["path"]
        for row in sources for record in row["source_records"]
    } | {str(REPORT.relative_to(ROOT)), str(ACQUISITION.relative_to(ROOT)), str(PAPERS.relative_to(ROOT))})
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "nfa-baseline-reproducibility-index-manifest-v1",
        "package_date": "2026-11-08",
        "source_snapshot_head": head,
        "source_snapshot_note": "HEAD only identifies this release snapshot; each historical run retains its own sidecar identities",
        "source_records": [file_record(path) for path in source_files],
        "build_script": file_record("agent_doc/results/nfa_baseline_reproducibility_index_20261108/build_baseline_index.py"),
        "command": "python3 agent_doc/results/nfa_baseline_reproducibility_index_20261108/build_baseline_index.py",
        "gpu_used": False,
        "generation_rerun": False,
        "rescoring_rerun": False,
        "paper_or_figures_modified": False,
    }
    write_json(OUT / "manifest.json", manifest)
    metadata = {
        "status_definitions": {
            "archived-complete": "result/manifest/receipt 均在当前快照，任务数完整，receipt success 且结果与 manifest SHA 回指一致",
            "archived-partial": "结果存在，但任务、sidecar 或 SHA 回指未全部闭合",
            "legacy-partial": "历史结果可登记，但不满足当前 formal 完整证据契约",
            "reported-not-archived": "研究报告引用了该结果，但当前发布快照没有对应文件；动作是找回同步，不是直接重跑",
            "not-run": "报告明确的实验缺口，与文件未同步不同",
        },
        "identity_rule": "所有 SHA 均由本脚本从当前文件实时计算；历史 run 的 SHA 仅从对应 sidecar 读取，不以当前脚本或当前 scorer 替代",
        "score_policy": "本包不生成新的质量均值，也不把不同 generation、预算或模型口径拼接成新表格",
        "coverage_note": "RULER 使用 11 任务；LongBench v1 使用 16 任务。样本数从结果 JSON 的 n 字段提取，不从报告文字复制",
    }
    write_json(OUT / "metadata.json", metadata)

    files = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    )
    with (OUT / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(OUT)}\n")
    print(
        "NFA_BASELINE_INDEX_OK",
        f"methods={len(sources)}",
        f"evidence={len(evidence)}",
        f"archived_complete={status_counts['archived-complete']}",
        f"recovery={len(summary['recovery_units'])}",
    )


if __name__ == "__main__":
    main()
