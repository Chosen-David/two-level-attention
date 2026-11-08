#!/usr/bin/env python3
"""从 B_final 原始请求生成区域评分的轻量、可追溯数据包。"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "B_final" / "quality"
OUT = Path(__file__).resolve().parent / "data_package"


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(cell, dataset, row_index):
    candidates = []
    for path in (QUALITY / cell / dataset).glob("manifest_*.json"):
        doc = json.loads(path.read_text())
        protocol = doc["protocol"]
        start = int(protocol.get("start", 0))
        end = int(protocol.get("end", 200))
        if start <= row_index < end:
            candidates.append((end - start, path, doc))
    if not candidates:
        raise ValueError(f"没有覆盖 {cell}/{dataset}/{row_index} 的 manifest")
    # 重叠续跑优先使用覆盖范围最大的正式 manifest。
    return max(candidates, key=lambda item: (item[0], str(item[1])))[2]


def main():
    requests = sorted(QUALITY.glob("*/*/requests/*.json"))
    if len(requests) != 2000:
        raise ValueError(f"预期 2000 个请求，实际 {len(requests)}")
    OUT.mkdir(exist_ok=True)
    rows = []
    source_commits = set()
    manifest_identities = set()
    specs = {}
    statuses = Counter()

    with (OUT / "quality.jsonl").open("w", encoding="utf-8") as quality_file:
        for request_path in requests:
            request = json.loads(request_path.read_text())
            cell, dataset = request_path.parts[-4], request_path.parts[-3]
            row_index = int(request["row_index"])
            manifest = load_manifest(cell, dataset, row_index)
            protocol = manifest["protocol"]
            spec = protocol["cell"]["spec"]
            specs[cell] = spec
            source_commits.add(protocol["git_head"])
            manifest_identities.add(manifest["identity"])
            result = request["result"]
            generated_ids = result["generated_ids"]
            if not isinstance(generated_ids, list):
                raise TypeError(f"generated_ids 不是数组: {request_path}")
            input_meta = request["input"]
            gates = result.get("gate_counts") or {}
            relative_raw = request_path.relative_to(ROOT).as_posix()
            raw_sha = sha256_file(request_path)
            record = {
                "cell": cell,
                "dataset": dataset,
                "row_index": row_index,
                "status": request["status"],
                "request_identity": request["identity"],
                "manifest_identity": manifest["identity"],
                "source_git_head": protocol["git_head"],
                "input_sha256": input_meta["input_sha256"],
                "input_tokens": input_meta["input_tokens"],
                "raw_tokens": input_meta["raw_tokens"],
                "truncated": input_meta["truncated"],
                "answers": request["answers"],
                "prediction": result["prediction"],
                "score": result["score"],
                "generated_token_count": len(generated_ids),
                "generated_ids_sha256": sha256_bytes(compact_json(generated_ids).encode()),
                "gate_keep": gates.get("keep", 0),
                "gate_near": gates.get("near", 0),
                "gate_far": gates.get("far", 0),
                "gate_both": gates.get("both", 0),
                "raw_request": f"../../{relative_raw}",
                "raw_request_sha256": raw_sha,
            }
            quality_file.write(compact_json(record) + "\n")
            statuses[request["status"]] += 1
            rows.append({
                "cell": cell,
                "dataset": dataset,
                "row_index": row_index,
                "subspace": spec["subspace"],
                "dimensions": spec["dimensions"],
                "sink": spec["sink"],
                "swa": spec["swa"],
                "near": spec["near"],
                "k1": spec["k1"],
                "k2": spec["k2"],
                "near_pages": spec["near_pages"],
                "partition": spec["partition"],
                "near_score": spec["near_score"],
                "far_score": spec["far_score"],
                "boundary": spec["boundary"],
                "gate_mode": spec["mode"],
                "epsilon": spec["epsilon"],
                "score": result["score"],
                "score_percent": 100.0 * result["score"],
                "input_tokens": input_meta["input_tokens"],
                "raw_tokens": input_meta["raw_tokens"],
                "truncated": input_meta["truncated"],
                "generated_token_count": len(generated_ids),
                "input_sha256": input_meta["input_sha256"],
                "raw_request_sha256": raw_sha,
            })

    fieldnames = list(rows[0])
    with (OUT / "plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest_doc = {
        "schema": "nfa-regional-scoring-data-package-v1",
        "source": "B_final raw requests; derived without GPU execution",
        "request_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "unique_cells": sorted(specs),
        "datasets": sorted({row["dataset"] for row in rows}),
        "rows_per_cell_dataset": 200,
        "source_git_heads": sorted(source_commits),
        "manifest_identities": sorted(manifest_identities),
        "cell_specs": specs,
        "full_generated_ids_location": "raw_request 指向的 JSON result.generated_ids；quality.jsonl 只另存其 SHA256，避免重复约 1 GiB token 数组",
        "independent_review": "../completeness_checks.json",
        "paired_statistics": "../paired_statistics.json",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    metadata = {
        "scope": "HotpotQA/GovReport 各 200 行、5 个区域选择配置；旧 GraphReady 之前的质量路径，gate mode=full（关闭 skip）",
        "near_far_order": "plot_data.csv 分列保存 near_score/far_score，论文 Near/Far 顺序不得反写",
        "baseline": "2d2a466b4e9a 是 NFA tail32 + Avg/MinMax，不是 dense FullKV",
        "statistics": "单次自然生成；独立 paired bootstrap B=10000、seed=20261031；未校正多重比较；CI 含 0 不证明等价",
        "reuse_boundary": "配置与当前 sink4/SWA128/K1=128/K2=1024/near4096/no-skip 工作点匹配；旧源码语义与 reclaim=true 原样保留，不改称当前 strict-skip 实测",
        "official_rescoring": "2000/2000，最大绝对误差 0；详见 ../completeness_checks.json",
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"DATA_PACKAGE_OK rows={len(rows)} cells={len(specs)} output={OUT}")


if __name__ == "__main__":
    main()
