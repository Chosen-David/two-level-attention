#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETFIX = os.environ.get("NFA_ANALYSIS_TARGETFIX", "0") == "1"
SAMPLE = int(os.environ.get("NFA_ANALYSIS_SAMPLE", "13"))
TRUTHS = {
    13: (65424, "9685956", ("15951-15957", "47939-47945")),
    23: (65469, "7824674", ("15607-15613", "47574-47580")),
    32: (65461, "5463011", ("24489-24495", "57003-57009")),
}
if TARGETFIX and SAMPLE not in TRUTHS:
    raise ValueError(f"没有已核验的自动目标真值: sample={SAMPLE}")
INPUT_TOKENS, ANSWER, CORRECTED_SPANS = TRUTHS.get(SAMPLE, TRUTHS[13])
ARMS = ({
    "default_k1_hist_l2": (
        "default_histl2_rank_oracle_mn128_targetfix.log" if SAMPLE == 13 else
        f"default_histl2_rank_oracle_s{SAMPLE}_mn128_targetfix.log"),
    "allpage_hist_l2": (
        "allpage_histl2_mn128_targetfix.log" if SAMPLE == 13 else
        f"allpage_histl2_s{SAMPLE}_mn128_targetfix.log"),
} if TARGETFIX else {
    "default_k1_hist_l2": "default_histl2_rank_oracle_mn128.log",
    "historical_l1_hist_l2": "hist_l1_hist_l2_mn128.log",
    "allpage_hist_l2": "allpage_histl2_mn128_v2.log",
    "exact_lse_hist_l2": "hist_apply_exact_lse_mean_mn128.log",
    "exact_max_hist_l2": "hist_apply_exact_max_mean_mn128.log",
})
SPANS = (CORRECTED_SPANS if TARGETFIX
         else ("15951-15957", "47951-47957"))
LINE = re.compile(
    r"TRACE_FILTER layer=(?P<layer>\d+) step=(?P<step>\d+) "
    r"span=(?P<span>\d+-\d+) "
    r"candidate\(any,all\)=(?P<candidate>\(\[.*?\], \[.*?\]\)) "
    r"middle\(any,all\)=(?P<middle>\(\[.*?\], \[.*?\]\))"
    r"(?: historical\(any,all\)=(?P<historical>\(\[.*?\], \[.*?\]\)))?"
)
SCORE = re.compile(
    rf"\[nfa:.*\] {SAMPLE} in=(\d+) out=(\d+) score=([0-9.]+)")


def any_heads(match, name):
    if match.group(name) is None:
        return None
    return frozenset(ast.literal_eval(match.group(name))[0])


def parse(path):
    rows = {}
    score = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if match:
            key = (int(match.group("layer")), int(match.group("step")),
                   match.group("span"))
            if key in rows:
                raise ValueError(f"duplicate {key} in {path}")
            rows[key] = {
                name: any_heads(match, name)
                for name in ("candidate", "middle", "historical")
            }
        score_match = SCORE.search(line)
        if score_match:
            score = {
                "input_tokens": int(score_match.group(1)),
                "generated_tokens": int(score_match.group(2)),
                "official_score": float(score_match.group(3)),
            }
    if len(rows) != 216 or score is None:
        raise ValueError(f"incomplete {path}: rows={len(rows)} score={score}")
    if TARGETFIX:
        truth_path = path.with_suffix("") / "target_truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        actual_spans = tuple(f"{row['lo']}-{row['hi']}" for row in truth["spans"])
        if (truth["source_index"] != SAMPLE or truth["input_tokens"] != INPUT_TOKENS
                or truth["answer"] != ANSWER or actual_spans != SPANS):
            raise ValueError(f"target truth mismatch: {truth_path}: {truth}")
        if set(row[2] for row in rows) != set(SPANS):
            raise ValueError(f"trace spans differ from target truth: {path}")
    return rows, score


def retention(rows, span, field):
    selected = [key for key in rows if key[2] == span]
    return {
        "layer_steps_with_any_head": sum(bool(rows[key][field]) for key in selected),
        "head_layer_step_hits": sum(len(rows[key][field]) for key in selected),
        "per_step_head_layer_hits": {
            str(step): sum(len(rows[(layer, step, span)][field])
                           for layer in range(36))
            for step in range(3)
        },
        "per_layer_head_step_hits": {
            str(layer): sum(len(rows[(layer, step, span)][field])
                            for step in range(3))
            for layer in range(36)
        },
    }


def main():
    parsed = {}
    evidence = {}
    for arm, filename in ARMS.items():
        path = ROOT / filename
        parsed[arm] = parse(path)
        evidence[filename] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if TARGETFIX:
            truth_path = path.with_suffix("") / "target_truth.json"
            evidence[str(truth_path.relative_to(ROOT))] = {
                "bytes": truth_path.stat().st_size,
                "sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
            }
    default = parsed["default_k1_hist_l2"][0]
    allpage = parsed["allpage_hist_l2"][0]
    transitions = Counter()
    examples = {name: [] for name in (
        "retained_only_with_allpage_candidate_missing",
        "retained_only_with_allpage_candidate_present",
        "default_historical_gain",
        "default_current_gain",
    )}
    for key in sorted(default):
        layer, step, span = key
        candidate = default[key]["candidate"]
        current = default[key]["middle"]
        historical = default[key]["historical"]
        successful = allpage[key]["historical"]
        for head in range(8):
            state = (
                head in candidate,
                head in current,
                head in historical,
                head in successful,
            )
            transitions[(span, state)] += 1
            example = {"layer": layer, "step": step, "span": span, "head": head}
            if head in successful and head not in historical:
                name = ("retained_only_with_allpage_candidate_present"
                        if head in candidate else
                        "retained_only_with_allpage_candidate_missing")
                if len(examples[name]) < 30:
                    examples[name].append(example)
            if head in historical and head not in successful:
                if len(examples["default_historical_gain"]) < 30:
                    examples["default_historical_gain"].append(example)
            if head in current and head not in historical:
                if len(examples["default_current_gain"]) < 30:
                    examples["default_current_gain"].append(example)
    serializable_transitions = {
        span: [
            {
                "candidate": state[0], "default_current": state[1],
                "default_historical": state[2], "allpage_historical": state[3],
                "head_layer_step_count": count,
            }
            for (row_span, state), count in sorted(
                transitions.items(), key=lambda item: (item[0][0], item[0][1]))
            if row_span == span
        ]
        for span in SPANS
    }
    same_run_name = (
        ("default_histl2_rank_oracle_mn128_targetfix" if SAMPLE == 13 else
         f"default_histl2_rank_oracle_s{SAMPLE}_mn128_targetfix") if TARGETFIX
        else "default_histl2_rank_oracle_mn128")
    same_run = ROOT / same_run_name
    l1_path = same_run / "l1_oracle.json"
    fine_path = same_run / "fine_oracle.json"
    l1_rows = json.loads(l1_path.read_text(encoding="utf-8"))
    fine_rows = json.loads(fine_path.read_text(encoding="utf-8"))
    if len(l1_rows) != 108 or len(fine_rows) != 108:
        raise ValueError(
            f"incomplete same-run oracle: L1={len(l1_rows)} fine={len(fine_rows)}")
    l1_by_key = {(row["layer"], row["step"]): row for row in l1_rows}
    fine_by_key = {(row["layer"], row["step"]): row for row in fine_rows}
    if l1_by_key.keys() != fine_by_key.keys():
        raise ValueError("same-run L1/fine layer-step keys differ")
    for relative in ("l1_oracle.json", "fine_oracle.json"):
        path = same_run / relative
        evidence[f"{same_run_name}/{relative}"] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for key, row in fine_by_key.items():
        if row["q_mismatch"] or row["finite_disagreement"]:
            raise ValueError(f"same-run fine validity mismatch at {key}")
        for item in row["targets"]:
            if (not item["reference_unique"]
                    or item["actual"] != item["reference"]):
                raise ValueError(f"same-run target fine mismatch at {key}: {item}")

    def span_for_token(token):
        for span in SPANS:
            lo, hi = (int(value) for value in span.split("-"))
            if lo <= token <= hi:
                return span
        raise ValueError(f"unexpected fine target token: {token}")

    fine_target_rows = []
    fine_target_keys = set()
    for (layer, step), fine in sorted(fine_by_key.items()):
        l1 = l1_by_key[(layer, step)]
        for item in fine["targets"]:
            token = item["token"]
            head = item["head"]
            unique_key = (layer, step, token, head)
            if unique_key in fine_target_keys:
                raise ValueError(f"duplicate same-run fine target: {unique_key}")
            fine_target_keys.add(unique_key)
            span = span_for_token(token)
            block = str(token // 64)
            far_rank = l1["targets"][block]["far_rank"][head][0]
            fine_target_rows.append({
                "layer": layer, "step": step, "span": span, **item,
                "l1_far_rank_zero_based": far_rank,
            })
    fine_targets_by_head = {}
    for row in fine_target_rows:
        key = (row["layer"], row["step"], row["span"], row["head"])
        fine_targets_by_head.setdefault(key, []).append(row)

    rank_rows = []
    candidate_crosscheck_mismatches = []
    fine_trace_selection_mismatches = []
    for (layer, step, span), row in sorted(default.items()):
        block = str(int(span.split("-")[0]) // 64)
        target = l1_by_key[(layer, step)]["targets"][block]
        actual_heads = frozenset(target["actual_far_heads"] + target["actual_near_heads"])
        if actual_heads != row["candidate"]:
            candidate_crosscheck_mismatches.append({
                "layer": layer, "step": step, "span": span,
                "l1_oracle_heads": sorted(actual_heads),
                "trace_candidate_heads": sorted(row["candidate"]),
            })
        for head in range(8):
            fine_targets = fine_targets_by_head.get((layer, step, span, head), [])
            for field, trace_field in (("middle", "middle"),
                                       ("historical_selected", "historical")):
                actual = any(item[field] for item in fine_targets)
                if actual != (head in row[trace_field]):
                    fine_trace_selection_mismatches.append({
                        "layer": layer, "step": step, "span": span, "head": head,
                        "field": field, "fine_selected": actual,
                        "trace_selected": head in row[trace_field],
                    })
            rank_rows.append({
                "layer": layer, "step": step, "span": span, "head": head,
                "far_rank_zero_based": target["far_rank"][head][0],
                "far_score": target["far_score"][head],
                "candidate": head in row["candidate"],
                "default_historical": head in row["historical"],
                "allpage_historical": head in allpage[(layer, step, span)]["historical"],
            })
    if candidate_crosscheck_mismatches:
        raise ValueError(f"L1/trace candidate mismatch: {candidate_crosscheck_mismatches[:3]}")
    if fine_trace_selection_mismatches:
        raise ValueError(
            f"fine/trace selection mismatch: {fine_trace_selection_mismatches[:3]}")
    successful_missing = [row["far_rank_zero_based"] for row in rank_rows
                          if row["allpage_historical"] and not row["candidate"]]

    def distribution(values):
        values = sorted(values)
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": values[0],
            "median": statistics.median(values),
            "max": values[-1],
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        }

    def fine_summary(rows):
        return {
            "candidate_target_token_entries": len(rows),
            "current_selected_entries": sum(row["middle"] for row in rows),
            "historical_selected_entries": sum(
                row["historical_selected"] for row in rows),
            "current_raw_rank_one_based": distribution(
                row["raw_rank"] for row in rows),
            "historical_strict_greater_rank_one_based": distribution(
                row["historical_raw_rank_strict_greater"] for row in rows),
            "historical_tie_count": distribution(
                row["historical_tie_count"] for row in rows),
            "historical_selected_rank": distribution(
                row["historical_raw_rank_strict_greater"] for row in rows
                if row["historical_selected"]),
            "historical_missed_rank": distribution(
                row["historical_raw_rank_strict_greater"] for row in rows
                if not row["historical_selected"]),
            "current_selected_raw_rank": distribution(
                row["raw_rank"] for row in rows if row["middle"]),
            "current_missed_raw_rank": distribution(
                row["raw_rank"] for row in rows if not row["middle"]),
        }

    output = {
        "scope": "single-sample diagnostic; not a formal quality result",
        "sample": {"source_index": SAMPLE, "task": "niah_multikey_2",
                   "context_length": 65536},
        "verification": "pending-independent-verification",
        "evidence": evidence,
        "arms": {
            arm: {
                "score": score,
                "spans": {
                    span: {
                        field: retention(rows, span, field)
                        for field in ("candidate", "middle", "historical")
                        if rows[(0, 0, span)][field] is not None
                    }
                    for span in SPANS
                },
            }
            for arm, (rows, score) in parsed.items()
        },
        "default_vs_allpage_head_transitions": serializable_transitions,
        "l1_rank_analysis": {
            "rank_convention": "zero-based full active far-page ranking; far quota is 96 pages/head",
            "candidate_crosscheck_mismatches": candidate_crosscheck_mismatches,
            "allpage_retained_but_default_candidate_missing": {
                "count": len(successful_missing),
                "rank_min": min(successful_missing),
                "rank_median": statistics.median(successful_missing),
                "rank_max": max(successful_missing),
                "rank_lt_96_count": sum(rank < 96 for rank in successful_missing),
                "rank_96_127_count": sum(96 <= rank < 128 for rank in successful_missing),
                "rank_128_255_count": sum(128 <= rank < 256 for rank in successful_missing),
                "rank_gte_256_count": sum(rank >= 256 for rank in successful_missing),
            },
            "allpage_retained_rows": [row for row in rank_rows
                                      if row["allpage_historical"]],
        },
        "same_trajectory_rank_analysis": {
            "scope": (
                "L1, fine scores, current L2 and historical L2 all come from "
                "default_histl2_rank_oracle_mn128; no cross-run L1 join"),
            "l1_far_mismatch_heads": sum(
                len(row["far_mismatch_heads"]) for row in l1_rows),
            "l1_near_mismatch_heads": sum(
                len(row["near_mismatch_heads"]) for row in l1_rows),
            "candidate_mismatch_heads": sum(
                len(row["candidate_mismatch_heads"]) for row in l1_rows),
            "fine_q_mismatch": sum(row["q_mismatch"] for row in fine_rows),
            "fine_finite_disagreement": sum(
                row["finite_disagreement"] for row in fine_rows),
            "fine_trace_selection_mismatches": fine_trace_selection_mismatches,
            "all_candidate_target_tokens": fine_summary(fine_target_rows),
            "by_span": {
                span: fine_summary([
                    row for row in fine_target_rows if row["span"] == span])
                for span in SPANS
            },
            "candidate_target_rows": fine_target_rows,
        },
        "examples": examples,
    }
    suffix = ("_targetfix" if SAMPLE == 13 else f"_s{SAMPLE}_targetfix")
    out = ROOT / (f"target_retention_summary{suffix}.json" if TARGETFIX
                  else "target_retention_summary.json")
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(out)
    for arm, (rows, score) in parsed.items():
        print(arm, score)
        for span in SPANS:
            values = output["arms"][arm]["spans"][span]
            print(" ", span, {
                field: (values[field]["layer_steps_with_any_head"],
                        values[field]["head_layer_step_hits"])
                for field in values
            })
    for span, rows in serializable_transitions.items():
        print("transitions", span)
        for row in rows:
            if row["allpage_historical"] or row["default_historical"]:
                print(" ", row)
    print("L1 successful-but-missing ranks",
          output["l1_rank_analysis"]["allpage_retained_but_default_candidate_missing"])
    same = output["same_trajectory_rank_analysis"]
    print("same-trajectory gates", {
        key: same[key] for key in (
            "l1_far_mismatch_heads", "l1_near_mismatch_heads",
            "candidate_mismatch_heads", "fine_q_mismatch",
            "fine_finite_disagreement", "fine_trace_selection_mismatches")})
    print("same-trajectory target tokens", same["all_candidate_target_tokens"])
    for span, summary in same["by_span"].items():
        print("same-trajectory", span, summary)


if __name__ == "__main__":
    main()
