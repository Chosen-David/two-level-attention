#!/usr/bin/env python3
"""从冻结 LongBench v2 输入重建 503 条实际长度与截断证据；不加载模型、不用 GPU。"""

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RAW = OUT / "raw"
SOURCE = Path(os.environ.get(
    "NFA_LBV2_SOURCE_JSON",
    "/home/wangyuanshuo02/datasets/LongBench/data/lbv2.json",
))
TOKENIZER = Path(os.environ.get(
    "NFA_QWEN3_TOKENIZER",
    "/home/wangyuanshuo02/Qwen3-8B",
))
CAP = 31500
MODEL_NAME = "Qwen3-8B"
EVIDENCE_BASE = "260f88b852e566e337d17abd182bfa40df71698b"
EXPECTED_SOURCE_SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
EXPECTED_RAW_SHA = {
    "FullKV": "67665479cd50a464d4f82ab7263ea64841d6326a09dc9868810c5e5ddf4c231c",
    "mavg": "da2719cd8350b88967c92f9f67449ae685d1cac10c3b64f009860b33b18dc141",
    "aavg": "1896ad19c04860351f06740e7d86790c6b94fb3aa02c36b967b36bdcd52da218",
}
RAW_FILES = {
    "FullKV": RAW / "fullkv.jsonl",
    "mavg": RAW / "mavg.jsonl",
    "aavg": RAW / "aavg.jsonl",
}
CODE_FILES = [
    ROOT / "benchmark/LongBench/pred.py",
    ROOT / "benchmark/LongBench/lbv2_choice.py",
    ROOT / "benchmark/LongBench/run_cap_manifest.py",
    ROOT / "benchmark/LongBench/config/dataset2prompt.json",
    ROOT / "benchmark/LongBench/config/model2maxlen.json",
    OUT / "build_length_audit.py",
]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
]


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


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def percentile(values, fraction):
    """线性插值分位数；输入已排序，定义固定以便跨环境重算。"""
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def distribution(values):
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
        "mean": sum(ordered) / len(ordered) if ordered else None,
    }


def main():
    import transformers
    from transformers import AutoTokenizer

    from benchmark.LongBench.lbv2_choice import (
        LBV2_PARSER_VERSION,
        extract_choice_official,
    )
    from benchmark.LongBench.pred import _split_prompt_question, build_chat
    from benchmark.LongBench.run_cap_manifest import input_ids_sha256

    assert sha256(SOURCE) == EXPECTED_SOURCE_SHA, "LongBench v2 源输入身份不符"
    with SOURCE.open(encoding="utf-8") as handle:
        source = json.load(handle)
    assert len(source) == 503
    source_ids = [row["_id"] for row in source]
    assert len(set(source_ids)) == 503
    prompt_format = json.loads(
        (ROOT / "benchmark/LongBench/config/dataset2prompt.json").read_text()
    )["lbv2"]
    model_caps = json.loads(
        (ROOT / "benchmark/LongBench/config/model2maxlen.json").read_text()
    )
    assert model_caps[MODEL_NAME] == CAP

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, local_files_only=True, use_fast=True
    )
    tokenizer_sha = {
        name: sha256(TOKENIZER / name) for name in TOKENIZER_FILES
    }

    # 三臂 raw 只用于校验样本身份与既有评分；长度完全由同一冻结源输入重建。
    quality = {}
    for arm, path in RAW_FILES.items():
        assert sha256(path) == EXPECTED_RAW_SHA[arm], f"{arm} raw 身份不符"
        rows = read_jsonl(path)
        assert len(rows) == 503
        assert [row["_id"] for row in rows] == source_ids
        answers = {row["_id"]: row["answer"] for row in source}
        correct = 0
        for row in rows:
            assert list(row["answers"]) == [answers[row["_id"]]]
            choice = extract_choice_official(row["pred"])
            correct += int(choice is not None and choice == answers[row["_id"]])
        quality[arm] = {
            "raw_path": str(path.relative_to(ROOT)),
            "raw_sha256": EXPECTED_RAW_SHA[arm],
            "rows": len(rows),
            "official_parser": LBV2_PARSER_VERSION,
            "official_correct": correct,
            "official_score": round(100.0 * correct / len(rows), 2),
        }
    assert quality["FullKV"]["official_score"] == 32.21
    assert quality["mavg"]["official_score"] == 32.01
    assert quality["aavg"]["official_score"] == 31.81

    length_rows = []
    for row_index, source_row in enumerate(source):
        obj = dict(source_row)
        obj["answers"] = [obj["answer"]]
        obj["all_classes"] = ["A", "B", "C", "D"]
        prompt = prompt_format.format(**obj)
        prompt_ids = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        raw_prompt_tokens = len(prompt_ids)
        truncated = raw_prompt_tokens > CAP
        if truncated:
            half = int(CAP / 2)
            prompt = tokenizer.decode(
                prompt_ids[:half], skip_special_tokens=True
            ) + tokenizer.decode(
                prompt_ids[-half:], skip_special_tokens=True
            )
        post_truncation_prompt_tokens = len(tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0])

        prompt = build_chat(tokenizer, prompt, MODEL_NAME)
        material, question = _split_prompt_question(prompt, "lbv2")
        material_ids = tokenizer(
            material, truncation=False, return_tensors="pt"
        ).input_ids[0].tolist()
        question_ids = tokenizer(
            question, truncation=False, return_tensors="pt"
        ).input_ids[0].tolist()
        if (
            question_ids
            and tokenizer.bos_token_id is not None
            and question_ids[0] == tokenizer.bos_token_id
        ):
            question_ids = question_ids[1:]
        realized_ids = material_ids + question_ids
        assert realized_ids
        length_rows.append({
            "row_index": row_index,
            "_id": source_row["_id"],
            "source_length_label": source_row.get("length"),
            "source_row_sha256": canonical_sha(source_row),
            "raw_prompt_tokens": raw_prompt_tokens,
            "cap_effective": CAP,
            "truncated": truncated,
            "kept_prefix_tokens": int(CAP / 2) if truncated else raw_prompt_tokens,
            "kept_suffix_tokens": int(CAP / 2) if truncated else 0,
            "post_truncation_prompt_tokens": post_truncation_prompt_tokens,
            "material_tokens": len(material_ids),
            "question_tokens": len(question_ids),
            "actual_input_tokens": len(realized_ids),
            "actual_input_ids_sha256": input_ids_sha256(realized_ids),
        })

    with (OUT / "lengths.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in length_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    columns = [
        "row_index", "_id", "source_length_label", "raw_prompt_tokens",
        "cap_effective", "truncated", "kept_prefix_tokens",
        "kept_suffix_tokens", "post_truncation_prompt_tokens",
        "material_tokens", "question_tokens", "actual_input_tokens",
        "actual_input_ids_sha256",
    ]
    with (OUT / "plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(length_rows)

    grouped = defaultdict(list)
    for row in length_rows:
        grouped[str(row["source_length_label"])].append(row)
    summary = {
        "schema": "nfa-lbv2-length-audit-v1",
        "status": "complete",
        "classification": "B",
        "rows": len(length_rows),
        "unique_ids": len({row["_id"] for row in length_rows}),
        "cap_effective": CAP,
        "truncation": "tokenized prompt mid-truncation: first 15750 + last 15750 tokens, decode with skip_special_tokens=True, then chat template and question split",
        "truncated_rows": sum(row["truncated"] for row in length_rows),
        "untruncated_rows": sum(not row["truncated"] for row in length_rows),
        "raw_prompt_tokens": distribution(row["raw_prompt_tokens"] for row in length_rows),
        "actual_input_tokens": distribution(row["actual_input_tokens"] for row in length_rows),
        "post_truncation_prompt_tokens": distribution(
            row["post_truncation_prompt_tokens"] for row in length_rows
        ),
        "by_source_length_label": {
            label: {
                "rows": len(rows),
                "truncated_rows": sum(row["truncated"] for row in rows),
                "raw_prompt_tokens": distribution(row["raw_prompt_tokens"] for row in rows),
                "actual_input_tokens": distribution(row["actual_input_tokens"] for row in rows),
            }
            for label, rows in sorted(grouped.items())
        },
        "quality_identity_check": quality,
    }
    write_json(OUT / "summary.json", summary)

    source_code_sha = {
        str(path.relative_to(ROOT)): sha256(path) for path in CODE_FILES
    }
    manifest = {
        "schema": "nfa-lbv2-length-audit-manifest-v1",
        "package_date": "2026-11-08",
        "classification": "B",
        "status": "complete",
        "scope": "CPU-only reconstruction of LongBench v2 input lengths and identities; no generation or GPU execution",
        "evidence_base_commit": EVIDENCE_BASE,
        "model_name": MODEL_NAME,
        "source_json": str(SOURCE),
        "source_sha256": EXPECTED_SOURCE_SHA,
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "transformers_version": transformers.__version__,
        "source_code_sha256": source_code_sha,
        "raw_predictions": quality,
        "sample_count": len(length_rows),
        "cap_effective": CAP,
        "max_new_tokens_historical": 128,
        "command": "python3 agent_doc/results/nfa_lbv2_length_audit_20261108/build_length_audit.py",
        "environment_overrides": {
            "NFA_LBV2_SOURCE_JSON": "optional path to the byte-identical lbv2.json",
            "NFA_QWEN3_TOKENIZER": "optional path to the byte-identical Qwen3-8B tokenizer/config snapshot",
        },
        "gpu_used": False,
        "generation_rerun": False,
    }
    write_json(OUT / "manifest.json", manifest)
    metadata = {
        "units": "all token counts are Qwen3 tokenizer token counts, not characters or words",
        "raw_prompt_tokens": "official 0-shot prompt before middle truncation and before Qwen3 chat wrapping",
        "post_truncation_prompt_tokens": "re-tokenized prompt after token-slice/decode middle truncation, before chat wrapping",
        "actual_input_tokens": "material + question token IDs after chat wrapping and the production Choices: split; this is the final model-input length",
        "actual_input_ids_sha256": "signed-int64 little-endian IDs, using benchmark/LongBench/run_cap_manifest.py:input_ids_sha256",
        "source_length_label": "dataset-provided coarse label; never used as a numeric token length",
        "limitations": [
            "Historical prediction JSONL did not store per-sample input IDs or token counts; this package deterministically reconstructs them from the frozen source, tokenizer and archived production functions.",
            "The historical model revision is not inferred from the local tokenizer directory; tokenizer/config file SHA256 values are the identity used here.",
            "This is not the official LongBench v2 120K protocol. The historical internal cap is 31500, so scores must not be compared directly with the official leaderboard.",
            "Length reconstruction does not rerun or alter any of the 503 generations and does not convert legacy predictions into token-level generation evidence.",
        ],
        "quality_scope": "E109 FullKV/mavg/aavg raw is archived only to close row identity and parser checks; E145d shares the 503-row source and 31500 cap but is not rescored or replaced here.",
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
        "LBV2_LENGTH_AUDIT_OK",
        f"rows={len(length_rows)}",
        f"truncated={summary['truncated_rows']}",
        f"raw_min={summary['raw_prompt_tokens']['min']}",
        f"raw_max={summary['raw_prompt_tokens']['max']}",
        f"actual_min={summary['actual_input_tokens']['min']}",
        f"actual_max={summary['actual_input_tokens']['max']}",
    )


if __name__ == "__main__":
    main()
