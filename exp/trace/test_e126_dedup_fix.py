#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TL-E126-STAGE-DOUBLECOUNT-001 修复红绿测试（CPU-only，mock GPU producer）。

缺陷：e126_stage_timing.py 旧版把 S2（父区间）与子段 S2_far/S2_near（S2 的
细分，父子非互斥阶段）一并累加 → selector_total_ms_median_est 与
amortization.selector_ms 对 cluster 臂重复计时。

测试（全部显式 check，不依赖 assert 语义，python -O 下同样生效——045 口径）：
  T1 (CPU, mock)  父子区间合成：S2=5, S2_far=2, S2_near=3, 其余互斥段各 1
                  → bench_arm 聚合 selector_total 与逐步和都 = 15（缺陷口径 20），
                  S2 细分字段保留 2/3；cavgsim（无 near）/mavg（无子段）变体。
  T2 (CPU, mock)  守恒：全键和 − 互斥和 == 子段和（逐步逐段）。
  T3 (CPU, 真数据) 五臂数据重聚合对拍：50 格 fresh 互斥重算 == 落盘值（逐位）；
                  30 非簇格新旧口径逐位相等；20 簇格守恒；GPT 8 数字锚点
                  （1e-9）；重聚合报告 before/after 对账（逐位）。
  T4 (CPU, 真数据) 五臂 sidecar + correction 元数据 + summary 修正登记 +
                  cluster 臂 amortization 首步 GPT 对拍。

运行：python3 exp/trace/test_e126_dedup_fix.py（及 python3 -O 同文件）。
本机无 torch：经 torch stub 加载聚合代码（cuda.empty_cache no-op），
不触碰任何真实 GPU kernel——本套件不构成 GPU 验证。
"""
import hashlib
import importlib.util
import json
import os
import sys
import types

# ---- torch stub：仅允许加载聚合/计时代码，绝不让真实 GPU kernel 执行 ----
sys.modules.setdefault("torch", types.SimpleNamespace(
    cuda=types.SimpleNamespace(empty_cache=lambda: None)))
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CORRECTION_ID = "TL-E126-STAGE-DOUBLECOUNT-001"
CLUSTER_ARMS = ("cavgsim", "cclustersim")
NON_CLUSTER_ARMS = ("mavg", "aavg", "mminmax")

# GPT 审计复现的受影响数字（对拍锚点，1e-9）：
# (arm, data, S) -> (total_before, total_after, step0_before, step0_after)
GPT_ANCHOR = {
    ("cavgsim", "structured", 65536):
        (5.483968052, 5.481056052, 3723.537647740, 1864.600391880),
    ("cclustersim", "structured", 65536):
        (5.732592100, 5.726768099, 3365.226106500, 1685.533197259),
}

FAILURES = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}", flush=True)
    if not cond:
        FAILURES.append(msg)


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "e126_stage_timing", os.path.join(HERE, "e126_stage_timing.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["e126_stage_timing"] = m
    spec.loader.exec_module(m)
    return m


def load_arm_json(arm):
    with open(os.path.join(RESULTS, f"e126_stage_timing_{arm}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def sidecar_ok(path):
    with open(path, "rb") as f:
        body = f.read()
    sc = open(path + ".sha256").read().strip()
    return hashlib.sha256(body).hexdigest() == sc


# ===================================================================== #
# T1/T2：父子区间合成（mock GPU producer——run_sim 数据面全为罐头，
# 聚合表达式为 harness 真实代码路径）
# ===================================================================== #
BASE_SPANS = {          # 互斥段各 1（11 个互斥键）+ 父 S2=5 + 子段 2/3
    "S0": 1.0, "S1": 1.0, "S2": 5.0,
    "S3_coarse": 1.0, "S3_avg": 1.0, "S3_fine": 1.0,
    "mask_setup": 1.0, "S4": 1.0, "S5": 1.0, "S6": 1.0, "S7": 1.0,
}


def make_canned_run_sim(H):
    """按臂配置生成罐头 run_sim（模拟 GPU producer 输出；含冷建簇式子段）。"""
    def fake_run_sim(mod, eager_fn, arm_cfg, k_all, v_all, S, n_steps=128,
                     device="cuda", q_seed=20261011, timing=True,
                     collect_gate=False):
        spans = dict(BASE_SPANS)
        if arm_cfg["enable_kmeans"]:
            spans["S2_far"] = 2.0
            if arm_cfg["near_select"] in ("cluster", "sim_greedy"):
                spans["S2_near"] = 3.0
        samples = [dict(spans) for _ in range(n_steps)]
        gate = {"min_row_any": True, "max_row_any": True, "out_finite": True,
                "sel_tokens_min": 1024.0, "sel_tokens_max": 1024.0,
                "sel_tokens_mean": 1024.0, "mask_shape": [1, 32, 1]}
        return ([1000] * n_steps, [1.0] * n_steps, samples, samples,
                gate, {})
    return fake_run_sim


class _DummyKV:
    """罐头 KV 张量替身：只需支持 clone()（值不进任何计算路径）。"""
    def clone(self):
        return self


def run_bench_arm_mocked(H, arm):
    """monkeypatch 数据生成与 run_sim 后跑真实 bench_arm（聚合代码在测）。"""
    import argparse
    H.gen_structured_kv = lambda *a, **k: _DummyKV()
    H.gen_singleton_kv = lambda *a, **k: _DummyKV()
    H.run_sim = make_canned_run_sim(H)
    args = argparse.Namespace(sizes=[4096], out="/tmp/e126_t1_dummy_out")
    recs = H.bench_arm(arm, None, None, {}, args, "cpu")
    check(len(recs) == 2, f"T1 {arm}: bench_arm 产出 2 个 case（structured+singleton）")
    for rec in recs:
        med = rec["stage_ms_median"]
        # 互斥段中位数和 = S2(5) + 其余互斥段 10 段×1 = 15（缺陷口径 = 20/17）
        check(rec["selector_total_ms_median_est"] == 15.0,
              f"T1 {arm}/{rec['data']}: selector_total == 15.0（实测 "
              f"{rec['selector_total_ms_median_est']!r}）")
        # 缺陷口径复算（全键和）作对照：cclustersim=20、cavgsim=17、非簇=15
        am = rec["amortization"]
        check(all(v == 15.0 for v in am["selector_ms"]),
              f"T1 {arm}/{rec['data']}: amortization.selector_ms 逐步全 == 15.0")
        # S2 细分字段保留（breakdown 不删）
        check(med["S2"] == 5.0, f"T1 {arm}/{rec['data']}: S2 breakdown == 5.0")
        check(am["s2_ms"][0] == 5.0, f"T1 {arm}/{rec['data']}: s2_ms == 5.0")
        old_style = sum(v for v in med.values() if v is not None)
        if arm == "cclustersim":
            check(med["S2_far"] == 2.0 and med["S2_near"] == 3.0,
                  f"T1 {arm}: S2_far/S2_near breakdown 保留 2.0/3.0")
            check(am["s2_far_ms"][0] == 2.0 and am["s2_near_ms"][0] == 3.0,
                  f"T1 {arm}: amortization s2_far/s2_near == 2.0/3.0")
            check(old_style == 20.0,
                  f"T1 {arm}: 缺陷口径全键和 == 20.0（证明 15 非缺陷口径凑巧）")
            check(abs(sum(am["selector_ms"]) / len(am["selector_ms"]) - 15.0) == 0
                  and abs(old_style - 15.0) == 5.0,
                  f"T1 {arm}: 逐步和与中位数和都不含子段重复量 5.0")
        elif arm == "cavgsim":
            check(med["S2_far"] == 2.0 and med["S2_near"] is None,
                  f"T1 {arm}: S2_far == 2.0 且 S2_near == None（near 缺席）")
            check(old_style == 17.0,
                  f"T1 {arm}: 缺陷口径全键和 == 17.0（只重复 S2_far）")
        else:
            check(med.get("S2_far") is None and med.get("S2_near") is None,
                  f"T1 {arm}: 非簇臂无子段样本（median None）")
            check(old_style == 15.0,
                  f"T1 {arm}: 非簇臂新旧口径同值 15.0（根因不触发）")


def t1_parent_child_synthetic():
    print("--- T1 父子区间合成（mock GPU producer，bench_arm 真实聚合路径）---")
    H = load_harness()
    for arm in ("cclustersim", "cavgsim", "mavg"):
        run_bench_arm_mocked(H, arm)


def t2_timeline_conservation():
    print("--- T2 逐步 timeline 守恒（全键和 − 互斥和 == 子段和）---")
    H = load_harness()
    spans = dict(BASE_SPANS)
    spans["S2_far"] = 2.0
    spans["S2_near"] = 3.0
    all_sum = sum(v for v in spans.values() if v is not None)
    excl_sum = H.exclusive_stage_sum(spans)
    sub_sum = spans["S2_far"] + spans["S2_near"]
    check(all_sum == 20.0 and excl_sum == 15.0,
          f"T2: 全键和 {all_sum!r} / 互斥和 {excl_sum!r} == 20.0/15.0")
    check(abs((all_sum - excl_sum) - sub_sum) < 1e-12,
          f"T2: 守恒 差值 {all_sum - excl_sum!r} == 子段和 {sub_sum!r}")
    # near 缺席变体（cavgsim 形态）
    spans2 = dict(BASE_SPANS)
    spans2["S2_far"] = 2.0
    all2 = sum(v for v in spans2.values() if v is not None)
    check(all2 - H.exclusive_stage_sum(spans2) == 2.0,
          "T2: near 缺席时重复量 == S2_far 单段 2.0")


def t3_real_data_reaggregate():
    print("--- T3 真实五臂数据重聚合对拍（50 格）---")
    H = load_harness()
    n_changed = n_unchanged = 0
    for arm in NON_CLUSTER_ARMS + CLUSTER_ARMS:
        doc = load_arm_json(arm)
        check(doc.get("correction", {}).get("id") == CORRECTION_ID,
              f"T3 {arm}: correction 元数据段存在且 id 正确")
        check(sidecar_ok(os.path.join(RESULTS, f"e126_stage_timing_{arm}.json")),
              f"T3 {arm}: .sha256 sidecar 与文件字节一致")
        check(H.verify_output_sha256(
            os.path.join(RESULTS, f"e126_stage_timing_{arm}.json")),
              f"T3 {arm}: output_content_sha256 内部自校验通过")
        is_cluster = arm in CLUSTER_ARMS
        for c in doc["cases"]:
            key = (arm, c["data"], c["S"])
            samples = c["stage_ms_samples"]
            new_fresh = sum(H.median(samples[k]) for k in samples
                            if samples[k] and k not in H.SUB_STAGE_KEYS)
            old_fresh = sum(H.median(v) for v in samples.values() if v)
            sub_med = sum(H.median(samples[k]) for k in samples
                          if samples[k] and k in H.SUB_STAGE_KEYS)
            am = c["amortization"]
            dup0 = (am["s2_far_ms"][0] or 0.0) + (am["s2_near_ms"][0] or 0.0)
            if not is_cluster:
                # 三非簇臂：新旧口径逐位相等 + 落盘值逐位等于两口径（逐位不变断言）
                check(old_fresh == new_fresh == c["selector_total_ms_median_est"],
                      f"T3 {key}: 非簇臂 fresh 新旧口径与落盘值三方逐位相等")
                check(dup0 == 0.0, f"T3 {key}: 非簇臂子段重复量为 0")
                n_unchanged += 1
            else:
                n_changed += 1
                # 落盘值 == fresh 互斥重算（逐位）
                check(new_fresh == c["selector_total_ms_median_est"],
                      f"T3 {key}: 落盘 selector_total == fresh 互斥重算（逐位）")
                # 守恒：旧口径 fresh − 新口径 == 子段中位数和
                check(abs((old_fresh - new_fresh) - sub_med) < 1e-9,
                      f"T3 {key}: total 守恒 old−new == 子段中位数和")
                # GPT 锚点（1e-9）
                if key in GPT_ANCHOR:
                    tb, ta, sb, sa = GPT_ANCHOR[key]
                    check(abs(old_fresh - tb) < 1e-9
                          and abs(new_fresh - ta) < 1e-9,
                          f"T3 {key}: GPT total 对拍 {tb:.9f}→{ta:.9f}")
                    check(abs(am["selector_ms"][0] + dup0 - sb) < 1e-9
                          and abs(am["selector_ms"][0] - sa) < 1e-9,
                          f"T3 {key}: GPT 首步 selector_ms 对拍 {sb:.9f}→{sa:.9f}")
    check(n_changed == 20 and n_unchanged == 30,
          f"T3: 计数 changed={n_changed}/unchanged={n_unchanged} == 20/30")
    # ---- 重聚合报告对账（before/after 逐位）----
    rp = os.path.join(RESULTS, "e126_dedup_correction_report.json")
    check(os.path.exists(rp), "T3: 重聚合报告文件存在")
    with open(rp, encoding="utf-8") as f:
        report = json.load(f)
    check(sidecar_ok(rp), "T3: 报告 sidecar 一致")
    check(H.verify_output_sha256(rp), "T3: 报告 output_content_sha256 自校验通过")
    rep = {(x["arm"], x["data"], x["S"]): x for x in report["cases"]}
    n_rep_changed = 0
    for (arm, data, S), rc in rep.items():
        c = next(x for x in load_arm_json(arm)["cases"]
                 if x["data"] == data and x["S"] == S)
        check(rc["total_after"] == c["selector_total_ms_median_est"],
              f"T3 报告({arm}/{data}/S={S}): total_after == 落盘值（逐位）")
        check(rc["selector_ms_after"] == c["amortization"]["selector_ms"],
              f"T3 报告({arm}/{data}/S={S}): selector_ms_after == 落盘值（逐位）")
        if rc["changed"]:
            n_rep_changed += 1
            dup = [(c["amortization"]["s2_far_ms"][i] or 0.0)
                   + (c["amortization"]["s2_near_ms"][i] or 0.0)
                   for i in range(len(rc["selector_ms_after"]))]
            ok = all(rc["selector_ms_after"][i]
                     == rc["selector_ms_before"][i] - dup[i]
                     for i in range(len(rc["selector_ms_after"])))
            check(ok, f"T3 报告({arm}/{data}/S={S}): 簇臂逐步 "
                  f"after == before − 子段（128 步逐位对账）")
        else:
            check(rc["selector_ms_before"] is None
                  and rc["total_before"] == rc["total_after"],
                  f"T3 报告({arm}/{data}/S={S}): 非簇臂 before==after 未改写")
    check(n_rep_changed == 20, f"T3 报告: changed 计数 {n_rep_changed} == 20")


def t4_summary_and_correction():
    print("--- T4 summary 重建 + correction 登记 + sidecar 闸门---")
    H = load_harness()
    sp = os.path.join(RESULTS, "e126_summary.json")
    check(sidecar_ok(sp), "T4: summary sidecar 一致")
    check(H.verify_output_sha256(sp), "T4: summary output_content_sha256 自校验通过")
    with open(sp, encoding="utf-8") as f:
        s = json.load(f)
    corr = s.get("meta", {}).get("correction", {})
    check(corr.get("id") == CORRECTION_ID,
          "T4: summary meta.correction.id 登记")
    check(sorted(corr.get("corrected_arms", [])) == ["cavgsim", "cclustersim"],
          "T4: summary correction.corrected_arms == 两簇臂")
    # cluster 臂 64K structured amortization 首步 == GPT 修正值（summary 重建生效）
    for arm in CLUSTER_ARMS:
        am = s["arms"][arm]["amortization"]["structured/S65536"]
        _, ta, _, sa = GPT_ANCHOR[(arm, "structured", 65536)]
        check(abs(am["selector_ms"][0] - sa) < 1e-9,
              f"T4 {arm}: summary amortization 首步 == GPT 修正值 {sa:.9f}")
        row = next(r for r in s["arms"][arm]["rows"]
                   if r["data"] == "structured" and r["S"] == 65536)
        check(row.get("S2_far") is not None and row.get("S2_near") is not None
              or arm == "cavgsim",
              f"T4 {arm}: summary rows 子段 breakdown 保留")
    # 五臂 correction 段 + method 语义边界（不出现 wall time 表述）
    for arm in NON_CLUSTER_ARMS + CLUSTER_ARMS:
        doc = load_arm_json(arm)
        cm = doc.get("correction", {})
        check(cm.get("id") == CORRECTION_ID, f"T4 {arm}: correction.id")
        check("wall" not in json.dumps(cm.get("method", {}), ensure_ascii=False),
              f"T4 {arm}: correction.method 无 wall time 表述（语义边界）")


def main():
    for t in (t1_parent_child_synthetic, t2_timeline_conservation,
              t3_real_data_reaggregate, t4_summary_and_correction):
        t()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(len(FAILURES))}"
          f"（checks 全部显式，python -O 下同样生效）", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
