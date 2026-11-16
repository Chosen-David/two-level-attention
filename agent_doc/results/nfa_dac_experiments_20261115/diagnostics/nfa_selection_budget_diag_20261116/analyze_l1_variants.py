#!/usr/bin/env python3
import hashlib
import json
import statistics
from pathlib import Path

import analyze_target_retention as base

ROOT = Path(__file__).resolve().parent
ARMS = (({
    "current_l2": "current_denseprob_globalquota_targetfix",
    "historical_l2": "hist_denseprob_globalquota_targetfix",
} if base.SAMPLE == 13 else {
    "current_l2": f"current_denseprob_globalquota_s{base.SAMPLE}_targetfix",
    "historical_l2": f"hist_denseprob_globalquota_s{base.SAMPLE}_targetfix",
}) if base.TARGETFIX else {
    "current_l2": "current_l1variants_mn128",
    "historical_l2": "hist_l1variants_mn128",
    "current_l2_exact_page": "current_exact_l1_mn128_v2",
    "historical_l2_exact_page": "hist_exact_l1_mn128_v2",
    "current_l2_dense_prob": "current_denseprob_l1_mn128",
    "historical_l2_dense_prob": "hist_denseprob_l1_mn128",
})
SPANS = {
    str(int(span.split("-")[0]) // 64): span
    for span in base.SPANS
}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": values[0],
        "median": statistics.median(values), "max": values[-1],
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
    }


def main():
    allpage_name = (
        ("allpage_histl2_mn128_targetfix.log" if base.SAMPLE == 13 else
         f"allpage_histl2_s{base.SAMPLE}_mn128_targetfix.log") if base.TARGETFIX
        else "allpage_histl2_mn128_v2.log")
    allpage_path = ROOT / allpage_name
    allpage, _ = base.parse(allpage_path)
    output = {
        "scope": "single-sample decision-only L1 variants; not a formal quality result",
        "verification": "pending-independent-verification",
        "arms": {},
        "evidence": {},
    }
    for arm, directory in ARMS.items():
        log_path = ROOT / f"{directory}.log"
        variant_path = ROOT / directory / "l1_variants.json"
        if not variant_path.exists():
            output["arms"][arm] = {"status": "pending"}
            continue
        trace, score = base.parse(log_path)
        rows = json.loads(variant_path.read_text(encoding="utf-8"))
        if len(rows) != 108:
            raise ValueError(f"incomplete {directory}: {len(rows)}")
        for path in (log_path, variant_path):
            output["evidence"][str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        names = list(rows[0]["targets"][next(iter(SPANS))])
        summaries = {
            name: {
                "selected_head_layer_step_hits": 0,
                "additional_vs_production": 0,
                "lost_vs_production": 0,
                "current_middle_recall": 0,
                "historical_middle_recall": 0,
                "allpage_historical_recall_cross_trajectory": 0,
                "ranks": [],
                "allpage_historical_ranks_cross_trajectory": [],
                "by_span": {
                    span: {"selected_head_layer_step_hits": 0,
                           "historical_middle_recall": 0,
                           "allpage_historical_recall_cross_trajectory": 0}
                    for span in SPANS.values()
                },
            }
            for name in names
        }
        production_mismatches = []
        totals = {"current_middle": 0, "historical_middle": 0,
                  "allpage_historical": 0}
        for row in rows:
            layer, step = row["layer"], row["step"]
            quota = row["far_quota"]
            if quota != 96:
                raise ValueError(f"unexpected far quota: {quota}")
            for block, span in SPANS.items():
                key = (layer, step, span)
                production_heads = frozenset(
                    row["targets"][block]["production"]["selected_heads"])
                if production_heads != trace[key]["candidate"]:
                    production_mismatches.append({
                        "layer": layer, "step": step, "span": span,
                        "production": sorted(production_heads),
                        "trace": sorted(trace[key]["candidate"]),
                    })
                totals["current_middle"] += len(trace[key]["middle"])
                totals["historical_middle"] += len(trace[key]["historical"] or ())
                totals["allpage_historical"] += len(allpage[key]["historical"])
                for name in names:
                    item = row["targets"][block][name]
                    selected = frozenset(item["selected_heads"])
                    summary = summaries[name]
                    summary["selected_head_layer_step_hits"] += len(selected)
                    summary["by_span"][span]["selected_head_layer_step_hits"] += len(selected)
                    summary["additional_vs_production"] += len(selected - production_heads)
                    summary["lost_vs_production"] += len(production_heads - selected)
                    summary["current_middle_recall"] += len(selected & trace[key]["middle"])
                    historical = trace[key]["historical"] or frozenset()
                    summary["historical_middle_recall"] += len(selected & historical)
                    summary["by_span"][span]["historical_middle_recall"] += len(
                        selected & historical)
                    successful = allpage[key]["historical"]
                    summary["allpage_historical_recall_cross_trajectory"] += len(
                        selected & successful)
                    summary["by_span"][span][
                        "allpage_historical_recall_cross_trajectory"] += len(
                            selected & successful)
                    summary["ranks"].extend(rank[0] for rank in item["far_rank"])
                    summary["allpage_historical_ranks_cross_trajectory"].extend(
                        item["far_rank"][head][0] for head in successful)
        if production_mismatches:
            raise ValueError(f"production variant mismatch: {production_mismatches[:3]}")
        for name, summary in summaries.items():
            summary["far_rank_distribution"] = distribution(summary.pop("ranks"))
            summary["allpage_historical_rank_distribution_cross_trajectory"] = (
                distribution(summary.pop("allpage_historical_ranks_cross_trajectory")))
        output["arms"][arm] = {
            "status": "complete", "score": score, "trace_denominators": totals,
            "production_trace_mismatches": production_mismatches,
            "variants": summaries,
        }
    suffix = ("_targetfix" if base.SAMPLE == 13 else f"_s{base.SAMPLE}_targetfix")
    out = ROOT / (f"l1_variant_summary{suffix}.json" if base.TARGETFIX
                  else "l1_variant_summary.json")
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(out)
    for arm, result in output["arms"].items():
        print(arm, result["status"], result.get("score"))
        if result["status"] != "complete":
            continue
        for name, summary in result["variants"].items():
            print(" ", name, {
                key: summary[key] for key in (
                    "selected_head_layer_step_hits", "additional_vs_production",
                    "lost_vs_production", "current_middle_recall",
                    "historical_middle_recall",
                    "allpage_historical_recall_cross_trajectory")},
                summary["far_rank_distribution"])


if __name__ == "__main__":
    main()
