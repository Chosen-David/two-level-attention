"""从已审计正式结果生成质量消融表与同输入配对 bootstrap CI。"""
import csv
import json
from pathlib import Path

import numpy as np

from benchmark.LongBench.eval import sample_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
QWEN_LB = REPO / "agent_doc/results/qwen3_nfa_fi_longbench16_20261115"
FIELDS = [
    "group", "arm", "subspace", "gate", "task", "context_length", "rows",
    "official_score", "fullkv_score", "delta_points",
    "paired_bootstrap_ci95_low", "paired_bootstrap_ci95_high", "source_head",
    "verification", "result_location",
]
ARM_META = {
    "nfa_full_gate_off": ("A1_full_gate_off", "full", "off"),
    "nfa_tail_gate_off": ("A2_tail32_gate_off", "tail32", "off"),
    "nfa_full_gate_on": ("A3_full_gate_on", "full", "on"),
    "nfa_tail_gate_on": ("A4_tail32_gate_on", "tail32", "on"),
}
LB_ARMS = {
    "full_gate_off": "nfa_full_gate_off",
    "tail_gate_off": "nfa_tail_gate_off",
    "full_gate_on": "nfa_full_gate_on",
    "tail_gate_on": "nfa_tail_gate_on",
}
MAX_NEW = {"hotpotqa": 32, "gov_report": 512}
rows = []


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def paired_ci(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("paired score vectors must be nonempty and aligned")
    deltas = candidate - reference
    rng = np.random.default_rng(20261115)
    indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
    means = deltas[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(low), float(high)


def add(arm, subspace, gate, task, length, count, score, source_head, location,
        fullkv_score=None, delta=None, ci_low=None, ci_high=None):
    rows.append({
        "group": "A", "arm": arm, "subspace": subspace, "gate": gate,
        "task": task, "context_length": length or "", "rows": count,
        "official_score": fmt(score), "fullkv_score": fmt(fullkv_score),
        "delta_points": fmt(delta), "paired_bootstrap_ci95_low": fmt(ci_low),
        "paired_bootstrap_ci95_high": fmt(ci_high), "source_head": source_head,
        "verification": "pending-independent-verification", "result_location": location,
    })


qwen_lb = load(QWEN_LB / "quality_summary.json")
fullkv_lb = {}
for task in ("hotpotqa", "gov_report"):
    source = qwen_lb["arms"]["flashinfer"]
    entry = source["canonical_predictions"][task]
    prediction_path = QWEN_LB / entry["path"]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    scores = [100.0 * sample_score(task, row["pred"], row["answers"], row["all_classes"])
              for row in predictions]
    mean_score = sum(scores) / len(scores)
    if len(predictions) != 200 or round(mean_score, 2) != source["scores"][task]:
        raise ValueError(f"reused FullKV score mismatch: {task}")
    runcap = load(Path(str(prediction_path) + ".runcap.json"))
    fullkv_lb[task] = {
        "ids": [row["_id"] for row in predictions], "scores": scores,
        "mean": mean_score, "ordered_input_sha256":
            runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
        "source_head": source["source_head"],
    }
    add("FullKV", "", "", task, None, 200, source["scores"][task], source["source_head"],
        "../qwen3_nfa_fi_longbench16_20261115", source["scores"][task], 0.0)

lb_audit_path = ROOT / "longbench_formal/_audits/local_archive_reaudit.json"
if lb_audit_path.exists():
    audit = load(lb_audit_path)
    if audit["status"] != "pass-production-side":
        raise ValueError("LongBench audit did not pass")
    for report in audit["results"]:
        arm = LB_ARMS[report["arm"]]
        arm_name, subspace, gate = ARM_META[arm]
        arm_root = ROOT / "longbench_formal" / report["arm"]
        for task in ("hotpotqa", "gov_report"):
            produced = report["tasks"][task]
            reference = fullkv_lb[task]
            if produced["ordered_input_sha256"] != reference["ordered_input_sha256"]:
                raise ValueError(f"LongBench paired input mismatch: {report['arm']} {task}")
            candidate_scores = load(arm_root / produced["official_scores"])
            if [row["_id"] for row in candidate_scores] != reference["ids"]:
                raise ValueError(f"LongBench paired IDs mismatch: {report['arm']} {task}")
            candidate = [row["official_score"] for row in candidate_scores]
            delta, low, high = paired_ci(candidate, reference["scores"])
            add(arm_name, subspace, gate, task, None, produced["rows"],
                produced["official_score"], report["source_head"],
                f"longbench_formal/{report['arm']}", reference["mean"], delta, low, high)

ruler_audit_path = ROOT / "ruler_formal/_audits/local_archive_reaudit.json"
if ruler_audit_path.exists():
    audit = load(ruler_audit_path)
    if audit["status"] != "pass-production-side":
        raise ValueError("RULER audit did not pass")
    cells = {(row["arm"], row["context_length"]): row for row in audit["results"]}
    for (arm, length), produced in sorted(cells.items(), key=lambda item: (item[0][1], item[0][0])):
        cell_root = ROOT / "ruler_formal" / arm / f"L{length}"
        manifest = load(cell_root / f"quality_manifest_L{length}_s0of1.json")
        scores = load(cell_root / manifest["scores"])
        if arm == "flashinfer_fullkv":
            add("FullKV", "", "", "niah_multikey_2", length, produced["rows"],
                produced["official_score"], produced["source_head"],
                f"ruler_formal/{arm}/L{length}", produced["official_score"], 0.0)
            continue
        arm_name, subspace, gate = ARM_META[arm]
        reference = cells.get(("flashinfer_fullkv", length))
        fullkv_score = delta = low = high = None
        if reference is not None:
            if produced["ordered_input_sha256"] != reference["ordered_input_sha256"]:
                raise ValueError(f"RULER paired input mismatch: {arm} {length}")
            reference_root = ROOT / "ruler_formal/flashinfer_fullkv" / f"L{length}"
            reference_manifest = load(reference_root / f"quality_manifest_L{length}_s0of1.json")
            reference_scores = load(reference_root / reference_manifest["scores"])
            if [row["_id"] for row in scores] != [row["_id"] for row in reference_scores]:
                raise ValueError(f"RULER paired IDs mismatch: {arm} {length}")
            fullkv_score = reference["recomputed_mean_score"]
            delta, low, high = paired_ci(
                [row["official_score"] for row in scores],
                [row["official_score"] for row in reference_scores])
        add(arm_name, subspace, gate, "niah_multikey_2", length, produced["rows"],
            produced["official_score"], produced["source_head"],
            f"ruler_formal/{arm}/L{length}", fullkv_score, delta, low, high)

with (ROOT / "quality_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"rows": len(rows), "output": str(ROOT / "quality_ablation.csv")}, indent=2))
