"""只读比较早期真实 freeze 与最终 decision-only 的同格输出。"""
import json
from pathlib import Path

old = Path("/tmp/ruler_dynamic_gate_scan_20261110")
new = Path("/tmp/ruler_dynamic_gate_decision_scan_20261110")
tasks = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
         "niah_single_3", "vt"]
gpu = {task: index % 4 for index, task in enumerate(tasks)}
max_delta = 0.0
mismatches = []
compared = []
for length in (131072, 65536):
    for task in tasks:
        before_path = old / f"ruler_{length}_shard{gpu[task]}.json"
        after_path = new / f"ruler_{length}_task_{task}.json"
        if not before_path.exists() or not after_path.exists():
            continue
        before_payload = json.loads(before_path.read_text())
        after_payload = json.loads(after_path.read_text())
        if not after_payload.get("complete"):
            continue
        before = next(row for row in before_payload["results"] if row["task"] == task)
        after = after_payload["results"][0]
        assert before["input_sha256"] == after["input_sha256"]
        for region in ("far", "near"):
            field = f"skip_{region}_layers"
            if before[field] != after[field]:
                mismatches.append([length, task, region])
            for layer in range(36):
                delta = abs(before["profiles"][str(layer)][region]["mean"]
                            - after["profiles"][str(layer)][region]["mean"])
                max_delta = max(max_delta, delta)
        compared.append([length, task])
print(json.dumps({"compared_cells": len(compared), "cells": compared,
                  "skip_mismatches": mismatches,
                  "max_abs_profile_mean_delta": max_delta}, indent=2))
