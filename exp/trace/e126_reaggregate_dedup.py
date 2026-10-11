#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E126 重聚合：修复 TL-E126-STAGE-DOUBLECOUNT-001（S2 父子区间重复求和）。

== 缺陷（GPT 审计 confirmed，主 AI 独立复核）==
harness 旧版把 S2（父区间 S2_in→S2_out 全长）与其子段 S2_far/S2_near（S2 区间
的细分，父子关系非互斥阶段）一并累加：
  1) selector_total_ms_median_est = 全阶段中位数和（含 S2+S2_far+S2_near）；
  2) amortization.selector_ms 逐步和同样全键累加。
触发条件：cluster 臂（cavgsim/cclustersim）产出 S2_far/S2_near 子段样本；
aavg/mavg/mminmax 三臂无子段样本，数值不受此根因影响（须逐位不变）。

== 修复口径 ==
  selector_total_ms_median_est：从持久化 stage_ms_samples fresh 重算
      互斥阶段中位数和（白名单 EXCLUSIVE_STAGE_KEYS，排除子段）。
  amortization.selector_ms：逐步全 span 未持久化（仅 s2_ms/s2_far_ms/
      s2_near_ms/selector_ms 数组落盘），唯一可恢复路径 = 逐步原值
      − (s2_far_ms + s2_near_ms)（子段扣除；浮点误差 ≤ 1 ulp 级）。
  原始样本（stage_ms_samples、逐 span breakdown、wall/s8/e2e、s2 系数组）
      逐位不动；S2_far/S2_near 字段保留作 breakdown。
  语义边界：修正值 = 互斥阶段和（中位数和 / 逐步和），不是 wall time。

== 门禁（fail-closed）==
  G1 落盘 SHA：五臂 JSON + summary 的 .sha256 sidecar 与文件字节一致；
  G2 样本保真（首次修正路径）：旧口径（全键求和）fresh 重算 == 落盘旧值
     （逐位）——证明 stage_ms_samples 键序/median 复算链路无损；
  G3 非簇三臂逐位不变（30 格）：fresh 新口径 == 旧口径 == 落盘值（逐位），
     amortization.selector_ms 不改写；
  G4 簇两臂修正（20 格）：全部变化；守恒 old−new == 子段重复量（total 层
     = 子段中位数和；逐步层 = 该步 s2_far+s2_near，1e-9）；
  G5 GPT 对拍锚点（1e-9）：64K structured 两簇臂 total 旧/新 + 首步
     selector_ms 旧/新 共 8 数字；
  G6 计数断言：changed==20 且 unchanged==30；
  G7 幂等（已修正状态重跑）：total 层 fresh 互斥中位数和 == 落盘值（逐位，
     独立重算）；amortization 层落盘值 == 报告 after 数组（逐位）且
     after == before − 子段（逐位，报告内部对账）；GPT 锚点 after 数字复验。

== 用法 ==
  python3 exp/trace/e126_reaggregate_dedup.py           # 未修正则应用；已修正则校验幂等
  python3 exp/trace/e126_reaggregate_dedup.py --check   # 只校验不写（未修正状态即失败）
本机无 torch：聚合代码经 torch stub 加载（cuda.empty_cache no-op），
不触碰任何 GPU kernel。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

# ---- torch stub：仅允许加载聚合/计时代码，绝不让真实 GPU kernel 执行 ----
sys.modules.setdefault("torch", types.SimpleNamespace(
    cuda=types.SimpleNamespace(empty_cache=lambda: None)))
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CORRECTION_ID = "TL-E126-STAGE-DOUBLECOUNT-001"
ARMS = ("mavg", "aavg", "mminmax", "cavgsim", "cclustersim")
CLUSTER_ARMS = frozenset(("cavgsim", "cclustersim"))
REPORT_PATH = os.path.join(RESULTS, "e126_dedup_correction_report.json")
SUMMARY_PATH = os.path.join(RESULTS, "e126_summary.json")

# GPT 审计复现的受影响数字（对拍锚点，1e-9）：
# (arm, data, S) -> (total_before, total_after, step0_before, step0_after)
GPT_ANCHOR = {
    ("cavgsim", "structured", 65536):
        (5.483968052, 5.481056052, 3723.537647740, 1864.600391880),
    ("cclustersim", "structured", 65536):
        (5.732592100, 5.726768099, 3365.226106500, 1685.533197259),
}


def fail(msg, code=3):
    print("[GATE-FAIL] " + msg, flush=True)
    raise SystemExit(code)


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "e126_stage_timing", os.path.join(HERE, "e126_stage_timing.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["e126_stage_timing"] = m
    spec.loader.exec_module(m)
    return m


H = load_harness()
MEDIAN = H.median
SUB_KEYS = H.SUB_STAGE_KEYS


def old_stage_median_sum(samples):
    """旧口径（缺陷口径）：全键中位数和——仅用于守恒/对拍，不写回任何字段。"""
    return sum(MEDIAN(v) for v in samples.values() if v)


def new_stage_median_sum(samples):
    """新口径：互斥阶段中位数和（白名单）——从原始样本独立重算，与运行状态无关。"""
    return sum(MEDIAN(samples[k]) for k in samples
               if samples[k] and k not in SUB_KEYS)


def verify_sidecar(path):
    if not os.path.exists(path + ".sha256"):
        fail(f"缺 {path}.sha256 sidecar")
    with open(path, "rb") as f:
        body = f.read()
    claimed = open(path + ".sha256").read().strip()
    if hashlib.sha256(body).hexdigest() != claimed:
        fail(f"{path} 的 .sha256 sidecar 与文件字节不一致")


def compute_case(case, arm):
    """单 case 状态无关重算（不做任何减法修正——那是首次应用路径的动作）：
    返回落盘值 + 样本派生值 + 子段逐步重复量（s2_far/s2_near 为原始 breakdown，
    修正前后逐位相同，两种状态下都可作为守恒基准）。"""
    samples = case["stage_ms_samples"]
    am = case["amortization"]
    n_steps = len(am["selector_ms"])
    if len(am["s2_far_ms"]) != n_steps or len(am["s2_near_ms"]) != n_steps:
        fail(f"{arm} S={case['S']} {case['data']} amortization 数组长度不一致")
    dup_step = []
    for i in range(n_steps):
        if am["selector_ms"][i] is None:
            fail(f"{arm} S={case['S']} {case['data']} step{i} selector_ms 为 None")
        dup_step.append((am["s2_far_ms"][i] or 0.0)
                        + (am["s2_near_ms"][i] or 0.0))
    return {
        "arm": arm, "data": case["data"], "S": case["S"],
        "stored_total": case["selector_total_ms_median_est"],
        "old_total_fresh": old_stage_median_sum(samples),   # 全键口径（G2 用）
        "new_total": new_stage_median_sum(samples),         # 互斥口径（目标值）
        "sub_med_sum": sum(MEDIAN(samples[k]) for k in samples
                           if samples[k] and k in SUB_KEYS),
        "stored_selector_ms": list(am["selector_ms"]),
        "dup_step": dup_step,
    }


def corrected_selector_ms(r):
    """首次应用路径的逐步修正值：落盘原值 − 子段重复量。"""
    return [v - d for v, d in zip(r["stored_selector_ms"], r["dup_step"])]


def gpt_anchor_check(r, sel_ms_after):
    """G5：GPT 对拍锚点（total 旧/新 + 首步 旧/新，1e-9）。sel_ms_after 为
    修正后的首步逐步和（未修正状态 = corrected 值；已修正状态 = 落盘值）。"""
    key = (r["arm"], r["data"], r["S"])
    tb, ta, sb, sa = GPT_ANCHOR[key]
    ok = (abs(r["stored_total"] - tb) < 1e-9          # 落盘值 = 修正前 total
          and abs(r["new_total"] - ta) < 1e-9         # 互斥重算 = 修正后 total
          and abs(r["stored_selector_ms"][0] - sb) < 1e-9   # 修正前首步
          and abs(sel_ms_after[0] - sa) < 1e-9)      # 修正后首步
    if not ok:
        fail(f"G5 {key} GPT 对拍失配：total {r['stored_total']!r}"
             f"→{r['new_total']!r} step0 {r['stored_selector_ms'][0]!r}"
             f"→{sel_ms_after[0]!r}")
    print(f"[GPT对拍 PASS] {key}: total {tb:.9f}→{ta:.9f}, "
          f"step0 {sb:.9f}→{sa:.9f}", flush=True)


def correction_meta():
    return {
        "id": CORRECTION_ID,
        "applied": time.strftime("%F %T"),
        "description": (
            "S2 父子区间重复求和修复：selector_total_ms_median_est 与 "
            "amortization.selector_ms 原把 S2 与其子段 S2_far/S2_near 一并"
            "累加（父子关系非互斥阶段）导致 cluster 臂重复计时。修正 = 互斥"
            "阶段（EXCLUSIVE_STAGE_KEYS）中位数和/逐步和；S2_far/S2_near "
            "保留为 breakdown。修正值语义 = 互斥阶段和，非 wall time。"),
        "method": {
            "selector_total_ms_median_est":
                "stage_ms_samples 白名单互斥阶段中位数和（原始样本逐位不动，"
                "fresh 重算）",
            "amortization.selector_ms":
                "逐步原值 − (s2_far_ms + s2_near_ms)（逐步全 span 未持久化，"
                "子段扣除为唯一可恢复路径，浮点误差 ≤ 1 ulp 级）",
        },
        "affected_fields": [
            "cases[].selector_total_ms_median_est",
            "cases[].amortization.selector_ms（仅 cluster 臂）",
        ],
        "raw_untouched": [
            "cases[].stage_ms_samples",
            "cases[].wall_ns_selector_samples",
            "cases[].s8_ms_samples",
            "cases[].stage_ms_median",
            "cases[].amortization.step_S/s2_ms/s2_far_ms/s2_near_ms/s2_cum_ms",
            "cases[].e2e_128step_wall_ns_timing_off",
        ],
        "report": "exp/trace/results/e126_dedup_correction_report.json",
    }


def write_arm(doc, path):
    doc["output_content_sha256"] = H.output_content_sha256(doc)
    body = json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8")
    H.atomic_write_bytes(path, body)
    H.atomic_write_bytes(path + ".sha256",
                         (hashlib.sha256(body).hexdigest() + "\n").encode("utf-8"))
    if not H.verify_output_sha256(path):
        fail(f"{path} 发布后内容 SHA 自校验失败")


def write_report(rows, changed, corrected_sel):
    """重聚合报告：逐 case 记录 before/after（total + 逐步完整数组）——
    幂等校验（G7）与外部对账的数据依据。"""
    changed_set = set(changed)
    report = {
        "meta": {
            "id": CORRECTION_ID,
            "probe": "E126 去重重聚合报告（GPT 审计 TL-E126-STAGE-DOUBLECOUNT-001）",
            "generated": time.strftime("%F %T"),
            "gates": ["G1 sidecar", "G2 样本保真（旧口径 fresh==落盘，逐位）",
                      "G3 非簇三臂逐位不变（30 格）", "G4 簇两臂守恒（20 格）",
                      "G5 GPT 对拍锚点 8 数字（1e-9）", "G6 计数 20/30",
                      "G7 幂等"],
            "changed_cases": len(changed),
            "unchanged_cases": len(rows) - len(changed),
        },
        "cases": [],
    }
    for r in rows:
        key = (r["arm"], r["data"], r["S"])
        is_changed = key in changed_set
        before_sel = r["stored_selector_ms"] if is_changed else None
        after_sel = corrected_sel[key] if is_changed else r["stored_selector_ms"]
        report["cases"].append({
            "arm": r["arm"], "data": r["data"], "S": r["S"],
            "changed": is_changed,
            "total_before": r["stored_total"],
            "total_after": r["new_total"],
            "sub_med_sum": r["sub_med_sum"],
            "selector_ms_before": before_sel,
            "selector_ms_after": after_sel,
        })
    report["output_content_sha256"] = H.output_content_sha256(report)
    body = json.dumps(report, indent=1, ensure_ascii=False).encode("utf-8")
    H.atomic_write_bytes(REPORT_PATH, body)
    H.atomic_write_bytes(REPORT_PATH + ".sha256",
                         (hashlib.sha256(body).hexdigest() + "\n").encode("utf-8"))
    print(f"report -> {REPORT_PATH}", flush=True)


def rebuild_summary(corrected_arms):
    """重建 e126_summary.json（读修正后 per-arm JSON，内部再过 SHA 门）+ 登记 correction。"""
    H.build_summary()          # 复用 harness 汇总逻辑（重新校验五臂 SHA 后聚合）
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)
    summary["meta"]["correction"] = {
        "id": CORRECTION_ID,
        "applied": time.strftime("%F %T"),
        "source": ("五臂 per-arm JSON 重聚合（exp/trace/e126_reaggregate_dedup.py）；"
                   "amortization.selector_ms 已按互斥阶段口径去重，"
                   "见各臂 JSON correction 段与 e126_dedup_correction_report.json"),
        "corrected_arms": sorted(corrected_arms),
    }
    summary["output_content_sha256"] = H.output_content_sha256(summary)
    body = json.dumps(summary, indent=1, ensure_ascii=False).encode("utf-8")
    H.atomic_write_bytes(SUMMARY_PATH, body)
    H.atomic_write_bytes(SUMMARY_PATH + ".sha256",
                         (hashlib.sha256(body).hexdigest() + "\n").encode("utf-8"))
    print(f"summary -> {SUMMARY_PATH}（correction 已登记）", flush=True)


def apply_first_run(docs, rows):
    """首次修正：G2-G6 全套门禁 + 应用 + 报告 + summary 重建。"""
    changed, unchanged, corrected_sel = [], [], {}
    for r in rows:
        key = (r["arm"], r["data"], r["S"])
        is_cluster = r["arm"] in CLUSTER_ARMS
        # G2 样本保真：旧口径 fresh 重算 == 落盘值（逐位）
        if r["old_total_fresh"] != r["stored_total"]:
            fail(f"G2 {key} 旧口径 fresh 重算 {r['old_total_fresh']!r} != "
                 f"落盘 {r['stored_total']!r}")
        if not is_cluster:
            # G3 非簇臂：无子段样本；新旧口径相等（逐位）；子段重复量恒 0
            has_sub = any(v for k, v in
                          next(c for c in docs[r["arm"]]["cases"]
                               if c["data"] == r["data"] and c["S"] == r["S"]
                               )["stage_ms_samples"].items() if k in SUB_KEYS)
            if has_sub:
                fail(f"G3 {key} 非簇臂却含子段样本（身份异常）")
            if r["new_total"] != r["stored_total"]:
                fail(f"G3 {key} 非簇臂新旧口径不等（根因分析错误）")
            if any(d != 0.0 for d in r["dup_step"]):
                fail(f"G3 {key} 非簇臂逐步子段重复量非零")
            unchanged.append(key)
            corrected_sel[key] = r["stored_selector_ms"]  # 不改写
        else:
            # G4 簇臂：必须变化 + total/逐步两层守恒
            if r["new_total"] >= r["stored_total"]:
                fail(f"G4 {key} 簇臂修正后未变小（子段重复量缺失）")
            if abs((r["stored_total"] - r["new_total"])
                   - r["sub_med_sum"]) > 1e-9:
                fail(f"G4 {key} total 守恒破坏：diff="
                     f"{r['stored_total'] - r['new_total']!r} "
                     f"sub_med_sum={r['sub_med_sum']!r}")
            sel_after = corrected_selector_ms(r)
            for i in range(len(sel_after)):
                if abs((r["stored_selector_ms"][i] - sel_after[i])
                       - r["dup_step"][i]) > 1e-9:
                    fail(f"G4 {key} step{i} 逐步守恒破坏")
            corrected_sel[key] = sel_after
            changed.append(key)
        # G5 GPT 对拍锚点
        if key in GPT_ANCHOR:
            gpt_anchor_check(r, corrected_sel[key])
    # G6 计数断言
    if len(changed) != 20 or len(unchanged) != 30:
        fail(f"G6 计数失配：changed={len(changed)}（须 20）、"
             f"unchanged={len(unchanged)}（须 30）")
    print(f"[G6 PASS] changed={len(changed)} 格（两簇臂），"
          f"unchanged={len(unchanged)} 格（三非簇臂）逐位不变", flush=True)

    # ---- 应用写盘 ----
    for arm in ARMS:
        path = os.path.join(RESULTS, f"e126_stage_timing_{arm}.json")
        for case in docs[arm]["cases"]:
            r = next(x for x in rows if x["arm"] == arm
                     and x["data"] == case["data"] and x["S"] == case["S"])
            case["selector_total_ms_median_est"] = r["new_total"]
            if arm in CLUSTER_ARMS:
                case["amortization"]["selector_ms"] = corrected_sel[
                    (arm, case["data"], case["S"])]
            # 非簇臂 amortization.selector_ms 不改写（逐位保持）
        docs[arm]["correction"] = correction_meta()
        write_arm(docs[arm], path)
        print(f"corrected -> {path}", flush=True)
    write_report(rows, changed, corrected_sel)
    rebuild_summary(sorted({k[0] for k in changed}))
    print("[DONE] 重聚合完成：20 格修正、30 格逐位不变、summary 已重建",
          flush=True)


def check_idempotent(docs, rows):
    """G7：已修正状态校验——total 层独立重算对账 + amortization 层报告对账。"""
    verify_sidecar(REPORT_PATH)
    if not H.verify_output_sha256(REPORT_PATH):
        fail(f"{REPORT_PATH} 内容 SHA 自校验失败")
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    rep = {(c["arm"], c["data"], c["S"]): c for c in report["cases"]}
    if len(rep) != 50:
        fail(f"G7 报告 case 数 {len(rep)} != 50")
    for r in rows:
        key = (r["arm"], r["data"], r["S"])
        rc = rep.get(key)
        if rc is None:
            fail(f"G7 报告缺 case {key}")
        # total 层：fresh 互斥中位数和（独立重算）== 落盘值（逐位）
        if r["new_total"] != r["stored_total"]:
            fail(f"G7 {key} 落盘 total {r['stored_total']!r} != fresh 互斥"
                 f"重算 {r['new_total']!r}")
        # 报告对账：after == 落盘（逐位）
        if rc["total_after"] != r["stored_total"]:
            fail(f"G7 {key} 报告 total_after != 落盘值")
        if rc["selector_ms_after"] != r["stored_selector_ms"]:
            fail(f"G7 {key} 报告 selector_ms_after != 落盘值（逐位）")
        if rc["changed"]:
            # 簇臂：before == after + 子段（逐位）；落盘 == after
            for i in range(len(rc["selector_ms_after"])):
                if rc["selector_ms_after"][i] != \
                        rc["selector_ms_before"][i] - r["dup_step"][i]:
                    fail(f"G7 {key} step{i} 报告 after != before − 子段")
            if rc["total_before"] - rc["total_after"] != rc["sub_med_sum"] \
                    and abs((rc["total_before"] - rc["total_after"])
                            - rc["sub_med_sum"]) > 1e-9:
                fail(f"G7 {key} 报告 total 守恒破坏")
        else:
            # 非簇臂：before/after 相同且未改写
            if rc["selector_ms_before"] is not None:
                fail(f"G7 {key} 非簇臂报告不应有 before 数组")
            if rc["total_before"] != rc["total_after"]:
                fail(f"G7 {key} 非簇臂报告 before/after 不等")
        # GPT 锚点 after 数字复验（total_after + 首步 after，1e-9）
        if key in GPT_ANCHOR:
            _, ta, _, sa = GPT_ANCHOR[key]
            if abs(r["stored_total"] - ta) > 1e-9 or \
                    abs(r["stored_selector_ms"][0] - sa) > 1e-9:
                fail(f"G7 {key} GPT after 锚点复验失配")
    n_changed = sum(1 for c in report["cases"] if c["changed"])
    if n_changed != 20:
        fail(f"G7 报告 changed 计数 {n_changed} != 20")
    # summary 修正登记核验
    verify_sidecar(SUMMARY_PATH)
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("meta", {}).get("correction", {}).get("id") != CORRECTION_ID:
        fail("G7 summary 缺 correction 登记")
    print("[G7 PASS] 五臂已全部处于修正后状态：total 层 fresh 重算逐位对账、"
          "amortization 层报告 before/after 逐位对账、GPT after 锚点复验通过、"
          "零改写", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只校验不写；发现未修正文件即失败")
    args = ap.parse_args()

    docs, rows = {}, []
    already_corrected = []
    for arm in ARMS:
        path = os.path.join(RESULTS, f"e126_stage_timing_{arm}.json")
        if not os.path.exists(path):
            fail(f"缺 {path}")
        verify_sidecar(path)                      # G1
        if not H.verify_output_sha256(path):
            fail(f"{path} 内容 SHA 自校验失败")
        with open(path, encoding="utf-8") as f:
            docs[arm] = json.load(f)
        if "correction" in docs[arm]:
            already_corrected.append(arm)
        for case in docs[arm]["cases"]:
            rows.append(compute_case(case, arm))
    print(f"[G1 PASS] 五臂 sidecar + 内容 SHA 全部校验通过；"
          f"已含 correction：{already_corrected or '无'}", flush=True)

    if already_corrected:
        if len(already_corrected) != len(ARMS):
            fail(f"correction 状态不一致（部分臂已修正）：{already_corrected}")
        check_idempotent(docs, rows)
        return
    if args.check:
        fail("--check 模式：检测到未修正文件（存在无 correction 段的臂）")
    apply_first_run(docs, rows)


if __name__ == "__main__":
    main()
