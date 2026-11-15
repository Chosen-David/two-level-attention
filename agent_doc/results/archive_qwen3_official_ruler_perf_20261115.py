#!/usr/bin/env python3
"""归档并复核 Qwen3 RULER MultiKey2 官方 baseline/NFA 性能矩阵。"""

import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path("/home/wangyuanshuo02/ak_verify_results/qwen3_official_ruler_perf_19edfebd_20261115")
RESULTS = Path(__file__).resolve().parent
DEST = RESULTS / "qwen3_official_ruler_multikey2_perf_20261115"
REUSED = {
    "nfa_256k_two_point_performance.json": RESULTS / "qwen3_official_long_context_20261113/nfa_256k_two_point_performance.json",
    "quest_256k_workspace_fix_performance.json": RESULTS / "qwen3_official_long_context_20261113/quest_256k_workspace_fix_performance.json",
}
FIXED_FILES = [
    "audit_perf.py",
    "performance_audit.json",
    "run_asynctls.sh",
    "run_nfa.sh",
    "run_quest.sh",
    "logs/asynctls_graph_gpu1.log",
    "logs/asynctls_graph_gpu1_256k_b4_oom.log",
    "logs/nfa_graph_gpu0.log",
    "logs/nfa_graph_gpu0_256k_b4_oom.log",
    "logs/nfa_graph_gpu0_attempt1_negative_pair.log",
    "logs/quest_gpu0.log",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    audit = json.loads((SOURCE / "performance_audit.json").read_text())
    require(audit["status"] == "PASS", "最终审计不是 PASS")
    require(audit["completed_cells"] == 26 and audit["pending_cells"] == 0, "可测格未闭集")
    require(audit["resource_blocked_cells"] == 2 and len(audit["matrix"]) == 48, "矩阵人口不符")
    require(audit["verification"] == "pending-independent-verification", "验收状态被改变")

    raw_files = sorted(
        path.relative_to(SOURCE).as_posix()
        for method in ("quest", "asynctls", "nfa")
        for path in (SOURCE / method).glob("*.json")
    )
    require(len(raw_files) == 24, f"新原始回执数量应为24，实际{len(raw_files)}")
    copy_files = FIXED_FILES + raw_files
    require(len(copy_files) == 35 and len(set(copy_files)) == 35, "归档文件清单不符")

    DEST.mkdir(parents=True, exist_ok=True)
    for relative in copy_files:
        source = SOURCE / relative
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha256(source) == sha256(target), f"复制后哈希不一致: {relative}")

    archived_audit = json.loads((DEST / "performance_audit.json").read_text())
    row_by_key = {
        (row["method"], row["context_length"], row["batch_size"]): row
        for row in archived_audit["rows"]
    }
    for row in archived_audit["rows"]:
        if row["reused_closed_result"]:
            continue
        relative = f"{row['method']}/{Path(row['path']).name}"
        require((DEST / relative).is_file(), f"缺少原始回执: {relative}")
        require(sha256(DEST / relative) == row["sha256"], f"回执哈希与审计不符: {relative}")

    cells = []
    for cell in archived_audit["matrix"]:
        key = (cell["method"], cell["context_length"], cell["batch_size"])
        item = dict(cell)
        if key in row_by_key:
            row = row_by_key[key]
            item["repeat_tpot_ms"] = row["repeat_tpot_ms"]
            item["reused_closed_result"] = row["reused_closed_result"]
            item["input_ids_sha256"] = row["input_ids_sha256"]
            item["actual_prompt_tokens"] = row["actual_prompt_tokens"]
        cells.append(item)

    summary = {
        "schema": "qwen3-official-ruler-multikey2-performance-summary-v1",
        "status": "complete",
        "verification": "pending-independent-verification",
        "created_date": "2026-11-15",
        "model": "Qwen3-8B",
        "task": "niah_multikey_2",
        "sample_index": 2,
        "source": {
            "production_head": "19edfebd404cfffadeadc30de7378f9fbe3e0069",
            "reused_closed_head": "41006cf8298167eed4b9bac386dfaa2a6c24bc85",
            "raw_root": str(SOURCE),
            "audit_sha256": sha256(DEST / "performance_audit.json"),
        },
        "protocol": {
            "lengths": [32768, 65536, 131072, 262144],
            "batch_sizes": [1, 2, 4],
            "warmup_requests": 1,
            "repeats": 3,
            "max_new_tokens": 33,
            "timing": "two-point host-wall TPOT=(T(33)-T(1))/32; B>1 is batched-step latency, not divided by batch",
            "rope_factors": {"32768": None, "65536": 2.0, "131072": 4.0, "262144": 8.0},
        },
        "scope": archived_audit["scope"],
        "completed_cells": 26,
        "resource_blocked_cells": 2,
        "blocked_jssa_b1_cells": 4,
        "unsupported_batched_quest_jssa_cells": 16,
        "pending_cells": 0,
        "implementation_labels": {
            "quest": "official Quest CUDA kernels and paged runtime with native GQA/interface adaptation; B1 only",
            "jssa": "fixed official cu129 wheel remains blocked; no version change or fallback",
            "asynctls": "paper-semantics PyTorch reproduction with CUDA Graph; not official TileLang and no real async host offload",
            "nfa": "NFA CUDA Graph production geometry K1=128, K2=1024, page64, tail32/cmp4, layer-skip disabled",
        },
        "cells": cells,
        "notes": [
            "All three repeats are retained, including NFA 128K B2=3.3572011161595583ms; no repeat was selected or removed.",
            "NFA and AsyncTLS 256K B4 are resource-blocked by real cache-allocation OOM and must not be retried as performance cells.",
            "Quest/JSSA B2/B4 are unsupported because no official high-performance batched path was validated; serial B1 is not substituted.",
            "Two 256K B1 cells reuse previously closed identical-protocol records; their original files and hashes are listed in manifest.json.",
            "All measurements remain pending independent verification; no cross-method ranking is promoted to an independently verified claim.",
        ],
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    files = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append({
            "path": path.relative_to(DEST).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    reused_files = []
    for name, path in sorted(REUSED.items()):
        require(path.is_file(), f"复用文件不存在: {path}")
        reused_files.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    archive_script = Path(__file__).resolve()
    manifest = {
        "schema": "qwen3-official-ruler-multikey2-performance-manifest-v1",
        "verification": "pending-independent-verification",
        "file_count": len(files),
        "files": files,
        "reused_closed_files": reused_files,
        "archive_script": {
            "path": str(archive_script),
            "sha256": sha256(archive_script),
        },
    }
    require(len(files) == 36, f"归档文件数应为36，实际{len(files)}")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "verification": summary["verification"],
        "completed_cells": summary["completed_cells"],
        "matrix_cells": len(cells),
        "files": len(files),
        "summary_sha256": sha256(DEST / "summary.json"),
        "manifest_sha256": sha256(DEST / "manifest.json"),
        "output": str(DEST),
    }, indent=2))


if __name__ == "__main__":
    main()
