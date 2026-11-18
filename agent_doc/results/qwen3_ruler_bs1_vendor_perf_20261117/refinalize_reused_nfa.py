#!/usr/bin/env python3
"""仅从既有 NFA SQLite 重算副本；不生成新 GPU 测量，不改旧证据。"""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.measure_nfa_selector_e2e import finalize_nsys_profile

ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    "/home/wangyuanshuo02/ak_verify_results/nfa_region_skip_full_20261111/"
    "tail_noskip_graph/nfa-operator-profile"
)
FINALIZER = REPO / "benchmark/efficiency/measure_nfa_selector_e2e.py"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    rows = []
    for length in (262144, 524288):
        stem = f"nfa-operator-profile_l{length}_b1_graph"
        source_json = SOURCE / f"{stem}.json"
        sqlite = SOURCE / f"{stem}.sqlite"
        nsys = SOURCE / f"{stem}.nsys-rep"
        output = ROOT / "profiles" / f"nfa_operator_b1_{length // 1024}k_reused.json"
        original_bytes = source_json.read_bytes()
        original = json.loads(original_bytes)
        original_summary = original["operator_summary"]
        result = json.loads(original_bytes)
        finalize_nsys_profile(result, sqlite)
        for key, value in original_summary.items():
            if result["operator_summary"].get(key) != value:
                raise RuntimeError(f"{source_json}: 重算改变了旧字段 {key}")
        result["verification"] = "pending-independent-verification"
        result["status"] = "complete"
        result["reuse_provenance"] = {
            "kind": "historical-external-reference-with-derived-refinalization",
            "original_json_path": str(source_json),
            "original_json_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "original_sqlite_path": str(sqlite),
            "original_sqlite_sha256": sha256(sqlite),
            "original_nsys_path": str(nsys),
            "original_nsys_sha256": sha256(nsys),
            "refinalizer_path": str(FINALIZER),
            "refinalizer_sha256": sha256(FINALIZER),
            "original_operator_fields_unchanged": True,
            "new_fields": ["pure_l1_score_ms", "pure_l2_score_ms"],
            "new_gpu_measurement": False,
            "current_binary_measurement": False,
            "note": "原始 5-step profile 与旧源码身份保留；仅用原 SQLite 在当前副本补充纯打分字段。",
        }
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary = result["operator_summary"]
        rows.append({
            "context_length": length,
            "output": str(output),
            "sha256": sha256(output),
            "profile_repeat": result["profile_repeat"],
            "pure_l1_ms_per_step": summary["pure_l1_score_ms"]["mean_ms_per_step"],
            "pure_l2_ms_per_step": summary["pure_l2_score_ms"]["mean_ms_per_step"],
            "wide_l1_ms_per_step": summary["l1_ms"]["mean_ms_per_step"],
            "wide_l2_ms_per_step": summary["l2_ms"]["mean_ms_per_step"],
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
