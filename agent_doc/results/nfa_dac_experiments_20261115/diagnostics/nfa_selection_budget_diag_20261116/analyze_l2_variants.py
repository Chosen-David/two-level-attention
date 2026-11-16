#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import analyze_target_retention as base

ROOT = Path(__file__).resolve().parent
ARMS = ({
    "default_l1": "hist_l2variants_defaultl1_mn128_targetfix",
    "exact_lse_l1": "hist_l2variants_exactlse_mn128_targetfix",
} if base.TARGETFIX else {
    "default_l1": "hist_l2variants_defaultl1_mn128_v4",
    "exact_lse_l1": "hist_l2variants_exactlse_mn128_v4",
})
SPANS = {"249": base.SPANS[0], "749": base.SPANS[1]}


def main():
    output = {
        "scope": "single-sample decision-only L2 variants; not a formal quality result",
        "verification": "pending-independent-verification",
        "arms": {},
        "evidence": {},
    }
    for arm, directory in ARMS.items():
        log_path = ROOT / f"{directory}.log"
        variant_path = ROOT / directory / "l2_variants.json"
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
        names = list(rows[0]["targets"]["249"])
        summaries = {
            name: {
                "selected_head_layer_step_hits": 0,
                "by_span": {span: 0 for span in SPANS.values()},
                "by_layer": {str(layer): 0 for layer in range(36)},
            }
            for name in names
        }
        actual_trace_mismatches = []
        for row in rows:
            layer, step = row["layer"], row["step"]
            if row["applied_variant"] != "historical_mean":
                raise ValueError(f"unexpected applied variant: {row['applied_variant']}")
            for block, span in SPANS.items():
                key = (layer, step, span)
                for name in names:
                    selected = frozenset(row["targets"][block][name]["selected_heads"])
                    summaries[name]["selected_head_layer_step_hits"] += len(selected)
                    summaries[name]["by_span"][span] += len(selected)
                    summaries[name]["by_layer"][str(layer)] += len(selected)
                    if name == "historical_mean" and selected != trace[key]["historical"]:
                        actual_trace_mismatches.append({
                            "layer": layer, "step": step, "span": span,
                            "variant": sorted(selected),
                            "trace": sorted(trace[key]["historical"] or ()),
                        })
        if actual_trace_mismatches:
            raise ValueError(f"actual L2 trace mismatch: {actual_trace_mismatches[:3]}")
        output["arms"][arm] = {
            "status": "complete", "score": score,
            "actual_trace_mismatches": actual_trace_mismatches,
            "variants": summaries,
        }
    out = ROOT / ("l2_variant_summary_targetfix.json" if base.TARGETFIX
                  else "l2_variant_summary.json")
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(out)
    for arm, result in output["arms"].items():
        print(arm, result["status"], result.get("score"))
        if result["status"] != "complete":
            continue
        for name, summary in result["variants"].items():
            print(" ", name, summary["selected_head_layer_step_hits"],
                  summary["by_span"])


if __name__ == "__main__":
    main()
