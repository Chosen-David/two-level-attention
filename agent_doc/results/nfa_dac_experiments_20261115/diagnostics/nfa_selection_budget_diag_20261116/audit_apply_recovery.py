#!/usr/bin/env python3
"""审计固定预算 L1 apply 的单样本恢复证据；不作正式闭集结论。"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_target_retention as base


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="current_apply_avg_top2_mean")
    parser.add_argument("--expected-far-method", default="minmax")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = {
        "schema": "nfa-single-sample-apply-recovery-audit-v1",
        "scope": "three single-sample diagnostics; not a formal closed-set result",
        "verification": "pending-independent-verification",
        "prefix": args.prefix,
        "samples": {},
        "evidence": {},
    }
    for sample in (13, 23, 32):
        os.environ["NFA_ANALYSIS_TARGETFIX"] = "1"
        os.environ["NFA_ANALYSIS_SAMPLE"] = str(sample)
        global base
        base = importlib.reload(base)
        name = f"{args.prefix}_s{sample}_targetfix"
        directory = ROOT / name
        log_path = directory.with_suffix(".log")
        trace, score = base.parse(log_path)
        variant_path = directory / "l1_variants.json"
        rows = json.loads(variant_path.read_text(encoding="utf-8"))
        if len(rows) != 108:
            raise ValueError(f"{name}: incomplete L1 rows={len(rows)}")
        expected_keys = {(layer, step) for layer in range(36) for step in range(3)}
        if {(row["layer"], row["step"]) for row in rows} != expected_keys:
            raise ValueError(f"{name}: invalid layer-step population")
        target_blocks = tuple(str(int(span.split("-")[0]) // 64)
                              for span in base.SPANS)
        applied_mismatches = []
        for row in rows:
            if row["far_quota"] != 96:
                raise ValueError(f"{name}: far quota={row['far_quota']}")
            for block, span in zip(target_blocks, base.SPANS):
                target = row["targets"][block]
                applied_heads = sorted(trace[(row["layer"], row["step"], span)]["candidate"])
                if applied_heads != target["avg_top2_mean"]["selected_heads"]:
                    applied_mismatches.append({
                        "layer": row["layer"], "step": row["step"], "block": block,
                        "applied_trace": applied_heads,
                        "avg_top2_mean": target["avg_top2_mean"]["selected_heads"],
                    })
        if applied_mismatches:
            raise ValueError(
                f"{name}: applied-trace/avg_top2 mismatch={applied_mismatches[:3]}")

        manifest_path = directory / "quality_manifest_L65536_s0of1.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["official_score"] != score["official_score"]:
            raise ValueError(f"{name}: log/manifest score mismatch")
        config = manifest["runtime"]["nfa_selector_config"]
        expected = {
            "level1_topk": 128, "level2_topk": 1024, "near_tokens": "off",
            "subspace": "full", "enable_subspace": False,
            "far_method": args.expected_far_method,
        }
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(f"{name}: config {key}={config[key]} expected={value}")
        if manifest["max_new_tokens"] != 128 or manifest["min_new_tokens"] != 1:
            raise ValueError(f"{name}: invalid generation protocol")
        receipt = manifest["request_receipts"][0]
        if (receipt["source_index"] != sample or receipt["generated_tokens"] != 128
                or receipt["input"]["input_tokens"] != base.INPUT_TOKENS):
            raise ValueError(f"{name}: invalid request receipt")
        if (receipt["gate_epsilon"] != 0.0
                or receipt["gate_counts"] != {"keep": 36, "near": 0, "far": 0, "both": 0}
                or any(decision["skip_near"] or decision["skip_far"]
                       for decision in receipt["prefill_decisions"].values())):
            raise ValueError(f"{name}: gate-off contract failed")

        file_hashes = {}
        for field, hash_field in (
                ("prediction", "prediction_sha256"),
                ("receipts", "receipts_sha256"),
                ("scores", "scores_sha256"),
                ("source_manifest", "source_manifest_sha256")):
            path = directory / manifest[field]
            digest = sha256(path)
            if digest != manifest[hash_field]:
                raise ValueError(f"{name}: {field} SHA256 mismatch")
            file_hashes[field] = digest
        for path in (log_path, variant_path, manifest_path,
                     directory / "target_truth.json", directory / manifest["runcap"]):
            relative = str(path.relative_to(ROOT))
            output["evidence"][relative] = {"bytes": path.stat().st_size,
                                            "sha256": sha256(path)}
        output["samples"][str(sample)] = {
            "official_score": score["official_score"], "trace_rows": len(trace),
            "l1_rows": len(rows), "applied_avg_top2_target_mismatches": 0,
            "file_hashes": file_hashes,
        }
        print(name, json.dumps(output["samples"][str(sample)], sort_keys=True))
    output_path = ROOT / (args.output or f"{args.prefix}_audit.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
