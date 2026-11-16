#!/usr/bin/env python3
import hashlib
import json
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
    "current_l2": "current_globalquota_l1_mn128",
    "historical_l2": "hist_globalquota_l1_mn128",
})
SPANS = {
    str(int(span.split("-")[0]) // 64): span
    for span in base.SPANS
}
GLOBAL_QUOTA = 8 * 96
PER_HEAD_CAP = 256
CRITICAL = (((7, 0, "15951-15957", 5), (7, 2, "15951-15957", 1))
            if base.SAMPLE == 13 else ())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_for(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_manifest(directory, manifest, evidence):
    if (manifest["rows"] != 1
            or (not base.TARGETFIX and manifest["official_score"] != 0.0)):
        raise ValueError(f"unexpected diagnostic manifest: {directory}: {manifest}")
    request = manifest["request_receipts"][0]
    if (request["source_index"] != base.SAMPLE
            or request["input"]["input_tokens"] != base.INPUT_TOKENS
            or request["generated_tokens"] != 128
            or request["gate_counts"] != {"keep": 36, "near": 0, "far": 0, "both": 0}):
        raise ValueError(f"diagnostic protocol mismatch: {directory}: {request}")
    config = manifest["runtime"]["nfa_selector_config"]
    expected = {
        "level1_topk": 128, "level2_topk": 1024, "page_size": 64,
        "near_pages": 32, "near_tokens": "off", "subspace_dim": 128,
        "enable_subspace": False,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError(f"diagnostic geometry mismatch: {directory}: {config}")
    for field in ("prediction", "runcap", "receipts", "scores", "source_manifest"):
        path = directory / manifest[field]
        if not path.is_file():
            raise ValueError(f"missing {field}: {path}")
        evidence[str(path.relative_to(ROOT))] = evidence_for(path)
        digest_field = field + "_sha256"
        if digest_field in manifest and manifest[digest_field] != sha256(path):
            raise ValueError(f"{field} hash mismatch: {path}")


def main():
    allpage_name = (
        ("allpage_histl2_mn128_targetfix.log" if base.SAMPLE == 13 else
         f"allpage_histl2_s{base.SAMPLE}_mn128_targetfix.log") if base.TARGETFIX
        else "allpage_histl2_mn128_v2.log")
    allpage_path = ROOT / allpage_name
    allpage, _ = base.parse(allpage_path)
    successful = {
        (layer, step, span, head)
        for (layer, step, span), row in allpage.items()
        for head in (row["historical"] or frozenset())
    }
    if not base.TARGETFIX and len(successful) != 38:
        raise ValueError(f"unexpected successful-target denominator: {len(successful)}")
    if base.TARGETFIX and not successful:
        raise ValueError("corrected all-page trajectory has no successful targets")

    output = {
        "scope": "single-sample decision-only global L1 quota oracle; not a formal quality result",
        "verification": "pending-independent-verification",
        "global_quota": GLOBAL_QUOTA,
        "per_head_cap_intersection": PER_HEAD_CAP,
        "cap256_note": (
            "global_cap256 is the intersection of global top-768 and original per-head "
            "rank<256; it is not a simulation of backfilling after per-head truncation"),
        "cross_trajectory_note": (
            "The successful targets come from the all-page historical-L2 trajectory. "
            "Coverage in these default-L1 trajectories is descriptive, not causal proof."),
        "successful_target_count": len(successful),
        "arms": {},
        "evidence": {},
    }
    for arm, directory_name in ARMS.items():
        directory = ROOT / directory_name
        log_path = ROOT / f"{directory_name}.log"
        variant_path = directory / "l1_variants.json"
        manifest_path = directory / "quality_manifest_L65536_s0of1.json"
        for path in (log_path, variant_path, manifest_path):
            output["evidence"][str(path.relative_to(ROOT))] = evidence_for(path)
        trace, score = base.parse(log_path)
        rows = json.loads(variant_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(rows) != 108 or score["official_score"] != manifest["official_score"]:
            raise ValueError(f"incomplete or score-mismatched {directory_name}")
        validate_manifest(directory, manifest, output["evidence"])

        names = list(rows[0]["targets"][next(iter(SPANS))])
        summaries = {
            name: {
                "all_target_fixed_per_head_96": 0,
                "all_target_global_768": 0,
                "all_target_global_768_cap256_intersection": 0,
                "successful_target_fixed_per_head_96": 0,
                "successful_target_global_768": 0,
                "successful_target_global_768_cap256_intersection": 0,
                "successful_global_gain_vs_fixed": 0,
                "successful_global_loss_vs_fixed": 0,
                "successful_cap256_loss_vs_global": 0,
                "by_span": {
                    span: {
                        "successful_fixed": 0, "successful_global": 0,
                        "successful_global_cap256": 0,
                    }
                    for span in SPANS.values()
                },
            }
            for name in names
        }
        coverage_rows = []
        critical_rows = []
        production_mismatches = []
        rank_invariant_mismatches = []
        for row in rows:
            layer, step = row["layer"], row["step"]
            if row["far_quota"] != 96 or row["applied_variant"] is not None:
                raise ValueError(
                    "L1 variants must be decision-only with production L1 applied: "
                    f"{directory_name}: layer={layer} step={step} "
                    f"quota={row['far_quota']} applied={row['applied_variant']}")
            for block, span in SPANS.items():
                key = (layer, step, span)
                for name in names:
                    item = row["targets"][block][name]
                    fixed = frozenset(item["selected_heads"])
                    global_selected = frozenset(item["global_selected_heads"])
                    cap256 = frozenset(item["global_cap256_selected_heads"])
                    if name == "production" and fixed != trace[key]["candidate"]:
                        production_mismatches.append({
                            "layer": layer, "step": step, "span": span,
                            "variant": sorted(fixed),
                            "trace": sorted(trace[key]["candidate"]),
                        })
                    for head in range(8):
                        far_rank = item["far_rank"][head][0]
                        global_rank = item["global_rank"][head]
                        expected = (
                            (head in fixed) == (far_rank < 96)
                            and (head in global_selected) == (global_rank < GLOBAL_QUOTA)
                            and (head in cap256)
                            == (head in global_selected and far_rank < PER_HEAD_CAP)
                        )
                        if not expected:
                            rank_invariant_mismatches.append({
                                "layer": layer, "step": step, "span": span,
                                "variant": name, "head": head,
                                "far_rank": far_rank, "global_rank": global_rank,
                            })
                        target_key = (layer, step, span, head)
                        states = (head in fixed, head in global_selected, head in cap256)
                        for field, selected in zip((
                                "all_target_fixed_per_head_96",
                                "all_target_global_768",
                                "all_target_global_768_cap256_intersection"), states):
                            summaries[name][field] += selected
                        if target_key in successful:
                            summary = summaries[name]
                            summary["successful_target_fixed_per_head_96"] += states[0]
                            summary["successful_target_global_768"] += states[1]
                            summary["successful_target_global_768_cap256_intersection"] += states[2]
                            summary["successful_global_gain_vs_fixed"] += states[1] and not states[0]
                            summary["successful_global_loss_vs_fixed"] += states[0] and not states[1]
                            summary["successful_cap256_loss_vs_global"] += states[1] and not states[2]
                            by_span = summary["by_span"][span]
                            by_span["successful_fixed"] += states[0]
                            by_span["successful_global"] += states[1]
                            by_span["successful_global_cap256"] += states[2]
                            if name in ("production", "exact_dense_meanprob_max"):
                                coverage_rows.append({
                                    "layer": layer, "step": step, "span": span,
                                    "head": head, "variant": name,
                                    "far_rank": far_rank, "global_rank": global_rank,
                                    "fixed_selected": states[0],
                                    "global_selected": states[1],
                                    "global_cap256_selected": states[2],
                                })
                        if target_key in CRITICAL and name in (
                                "production", "exact_page_lse_mean",
                                "exact_dense_meanprob_max", "exact_dense_maxprob_max"):
                            critical_rows.append({
                                "layer": layer, "step": step, "span": span,
                                "head": head, "variant": name,
                                "far_rank": far_rank, "global_rank": global_rank,
                                "fixed_selected": states[0],
                                "global_selected": states[1],
                                "global_cap256_selected": states[2],
                            })
        if production_mismatches or rank_invariant_mismatches:
            raise ValueError(
                f"global-quota validation failed: {directory_name}: "
                f"production={production_mismatches[:3]} ranks={rank_invariant_mismatches[:3]}")
        output["arms"][arm] = {
            "status": "complete", "score": score,
            "prediction_sha256": manifest["prediction_sha256"],
            "receipts_sha256": manifest["receipts_sha256"],
            "scores_sha256": manifest["scores_sha256"],
            "production_trace_mismatches": production_mismatches,
            "rank_invariant_mismatches": rank_invariant_mismatches,
            "variants": summaries,
            "critical_layer7": critical_rows,
            "successful_target_coverage_rows": coverage_rows,
        }

    suffix = ("_targetfix" if base.SAMPLE == 13 else f"_s{base.SAMPLE}_targetfix")
    out = ROOT / (f"global_quota_summary{suffix}.json" if base.TARGETFIX
                  else "global_quota_summary.json")
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    for arm, result in output["arms"].items():
        print(arm, result["status"], result["score"], result["prediction_sha256"])
        for name, summary in result["variants"].items():
            print(" ", name, {
                key: summary[key] for key in (
                    "successful_target_fixed_per_head_96", "successful_target_global_768",
                    "successful_target_global_768_cap256_intersection",
                    "successful_global_gain_vs_fixed", "successful_global_loss_vs_fixed",
                    "successful_cap256_loss_vs_global")})
        print(" critical layer7")
        for row in result["critical_layer7"]:
            print("  ", row)


if __name__ == "__main__":
    main()
