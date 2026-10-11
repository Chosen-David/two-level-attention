# -*- coding: utf-8 -*-
"""E128 五臂质量×性能综合权衡表（任务 #216，判卷风格克隆 E125 fail-closed 纪律）。

回答的问题：E125 质量海选（cclustersim 点估计冠军）× E126 修正后分阶段计时 ×
E127 共享热路径优化定案，三者合成后，论文主方法应锚定哪个 method 臂。

三源输入（全部只读，sha256 进 provenance）：
  1) exp/trace/results/e125_screen_verdict.json
     —— AVG5 质量分 + 配对差 bootstrap CI95（difflib:stdlib，B=10000 seed=42，
        任务级配对差 vs mavg 锚点，E123 锚点复用）。
  2) exp/trace/results/e126_stage_timing_{mavg,aavg,mminmax,cavgsim,cclustersim}.json
     —— 五臂选择器分阶段计时（H20-3e，structured/singleton × S 4K-64K，B=1
        q_len=1，D' off）。只消费修正后字段（correction 段存在性是硬门禁，
        TL-E126-STAGE-DOUBLECOUNT-001；selector_total_ms_median_est 与
        amortization.selector_ms 已排除 S2_far/S2_near 父子双计）。
  3) exp/results_efficiency/e127_stage_bench_baseline.json + _iter8.json
     （+ final_64k/final_8k 交叉核对）—— 共享热路径优化配对组：S=64K 每层
        每步 GPU 8.29→1.52ms（5.47×）/ wall 9.62→2.02ms（4.76×）。
  补充源（跨口径符号稳定性论证用）：
  4) exp/trace/results/e109_screen4_selection.json —— E109 旧口径五任务海选
     逐任务分数（B10 前，repobench-p 任务名）。

GPT f4139b5 四点窄更正（E128 消费硬约束，逐条落实）：
  ① E126 去重后 64K median 和 mavg 5.5516 vs cclustersim 5.7268 相对差
     3.16%（本脚本从数据重算并如实输出，绝不写 <2%）。
  ② e2e_128step_wall_ns_timing_off 是混合时钟微基准（selector host
     interval + S8 CUDA event），只允许 E126 内部横向可比，禁止与 E127
     配对组提速混算——输出中每个 e2e 字段都带 mixed_clock 标注。
  ③ aavg 配置 (α,β,γ)=(.125,.125,.375) 非 (0,0)——E125 arm_configs 字串
     与 E126 arm_config 数值双重门禁。
  ④ S8 = eager_decoding 全长 PV（非只读 K2 个 KV）——S8 语义注释如实。

统计诚实纪律（084 措辞）：质量差 CI 含 0 的臂间写「未证明差异」，
不写「持平/等价」。E125 全部臂 vs mavg 的 CI 均含 0（mminmax 除外，
显著更差）；cclustersim +0.34 [-0.20,+1.02] 未证明显著。

结论限定口径：structured/singleton 合成数据、B=1、q_len=1、D′ off、
单层 microbench（E126 计时插桩开 / E127 独立 harness），H20-3e。

python 与 -O 双跑安全（无 assert 门禁，全部 sys.exit fail-closed）。
v1 by Opus 派遣分析 agent，2026-10-12。
"""
import hashlib
import json
import os
import random
import sys

REPO = os.environ.get(
    "E128_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..")))
OUT_JSON = os.environ.get(
    "E128_OUT_JSON",
    os.path.join(REPO, "exp", "trace", "results", "e128_tradeoff.json"))

ARMS = ["mavg", "aavg", "mminmax", "cavgsim", "cclustersim"]
CLUSTER_ARMS = ["cavgsim", "cclustersim"]
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
SIZES = [4096, 8192, 16384, 32768, 65536]
DATA_KINDS = ["structured", "singleton"]
B, SEED = 10000, 42

# E125/E126 双侧冻结配置（GPT ③：(α,β,γ) 权威值；E125 arm_configs 字串中
# 的三元组也须逐臂匹配）
EXPECTED_ABG = {
    "mavg": (0.25, 0.125, 0.625),
    "aavg": (0.125, 0.125, 0.375),
    "mminmax": (0.875, 0.875, 0.125),
    "cavgsim": (0.0, 0.0, 0.0),
    "cclustersim": (0.125, 0.25, 0.125),
}
EXPECTED_ABG_STR = {  # E125 meta.arm_configs 内应出现的字面三元组
    "mavg": "(.25,.125,.625)",
    "aavg": "(.125,.125,.375)",
    "mminmax": "(.875,.875,.125)",
    "cavgsim": "(0,0,0)",
    "cclustersim": "(.125,.25,.125)",
}

E125_PATH = os.path.join(REPO, "exp", "trace", "results",
                        "e125_screen_verdict.json")
E126_PATH = {a: os.path.join(REPO, "exp", "trace", "results",
                             f"e126_stage_timing_{a}.json") for a in ARMS}
E127_DIR = os.path.join(REPO, "exp", "results_efficiency")
E127_BASE = os.path.join(E127_DIR, "e127_stage_bench_baseline.json")
E127_ITER8 = os.path.join(E127_DIR, "e127_stage_bench_iter8.json")
E127_FIN64 = os.path.join(E127_DIR, "e127_stage_bench_final_64k.json")
E127_FIN8 = os.path.join(E127_DIR, "e127_stage_bench_final_8k.json")
E109_PATH = os.path.join(REPO, "exp", "trace", "results",
                        "e109_screen4_selection.json")

E109_KEYS = {  # E109 旧口径各臂最佳配置（E125 重跑的就是这五个配置）
    "mavg": "E109_mavg_a0.25_b0.125_g0.625",
    "aavg": "E109_aavg_a0.125_b0.125_g0.375",
    "mminmax": "E109_mminmax_a0.875_b0.875_g0.125",
    "cavgsim": "E109_cavgsim_a0_b0_g0",
    "cclustersim": "E109_cclustersim_a0.125_b0.25_g0.125",
}

CORR_ID = "TL-E126-STAGE-DOUBLECOUNT-001"
TLI_INDEXER_MD5 = "bf5e26c4d6123875b2786b9ed886da5f"

# E126 互斥阶段白名单（与 e126_stage_timing.py EXCLUSIVE_STAGE_KEYS 同源：
# 全部 STAGE_SPANS 名去掉子段 S2_far/S2_near）
EXCLUSIVE_STAGE_KEYS = ["S0", "S1", "S2", "S3_coarse", "S3_avg", "S3_fine",
                       "mask_setup", "S4", "S5", "S6", "S7"]
# E127 fast 路径（TLIIndexerFast + eager_decoding 优化）覆盖的选择器阶段：
# S0/S1=prepare_index 增量缓存（iter4/5）；S3_*=compute_score Triton
# （iter6/8）。metrics 惰性化（iter1）不在选择器阶段内；S8 单列。
OPT_STAGE_KEYS = ["S0", "S1", "S3_coarse", "S3_avg", "S3_fine"]
# 未被 E127 结构优化的选择器阶段：S2 建簇（簇臂冷建簇，fast 路径原样调用
# _maybe_build_kmeans）、S4 簇内 greedy 打分（sim_greedy 臂）、
# S5/S6/S7+mask_setup（compute_mask 核心，iter9-11 负结果）。
UNOPT_STAGE_KEYS = [k for k in EXCLUSIVE_STAGE_KEYS if k not in OPT_STAGE_KEYS]


def fail(msg):
    sys.exit("[ABORT] " + msg)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median(samples):
    """与 e126_stage_timing.py L416 同构（偶数长度取中二均值）。"""
    s = sorted(samples)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def load_json(path, label):
    if not os.path.exists(path):
        fail(f"{label}: 输入文件不存在 {path}——先核对三源数据落盘")
    return json.load(open(path))


def paired_bootstrap(deltas, B=B, seed=SEED):
    """与 analyze_e125_screen_verdict.py 同协议（fresh seed / 任务级重采样）。"""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def r6(x):
    return None if x is None else round(x, 6)


# ------------------------------------------------------------------ E125
def load_e125(prov):
    d = load_json(E125_PATH, "E125")
    prov["e125_screen_verdict"] = {
        "path": os.path.relpath(E125_PATH, REPO),
        "sha256": sha256_of(E125_PATH),
        "consumed_fields": [
            "meta.scorer_backend", "meta.bootstrap", "meta.arm_configs",
            "input_manifest[]（20 格 rows==expect 闭包门禁）",
            "per_task.{arm}/{task}.{arm,mavg,fullkv,delta_vs_mavg,"
            "delta_vs_fullkv}",
            "{arm}_avg / {arm}_avg_delta_vs_mavg / {arm}_ci95 / "
            "{arm}_significant_vs_mavg（重算复核门禁）",
            "mavg_anchor_avg / fullkv_anchor_avg / candidates_avg5 / verdict"],
    }
    if (d.get("meta") or {}).get("scorer_backend") != "difflib:stdlib":
        fail("E125: meta.scorer_backend != difflib:stdlib——跨后端比较无意义")
    bs = d["meta"]["bootstrap"]
    if bs["B"] != B or bs["seed"] != SEED:
        fail(f"E125: bootstrap B/seed 与本脚本协议不一致（{bs['B']}/{bs['seed']}"
             f" vs {B}/{SEED}）——CI 复核无意义，fail closed")

    # input_manifest 闭包：20 格，rows==expect
    man = d.get("input_manifest") or []
    if len(man) != 20:
        fail(f"E125: input_manifest {len(man)} 格 != 20——闭包不完整")
    for e in man:
        if e["rows"] != e["expect"]:
            fail(f"E125: manifest {e['arm']}/{e['task']} rows={e['rows']} "
                 f"!= expect={e['expect']}")

    # GPT ③：aavg 等五臂 (α,β,γ) 字串门禁
    arm_cfg_str = d["meta"]["arm_configs"]
    for a in ARMS:
        if a == "mavg":
            continue  # mavg 是 E123 锚点，不在 E125 四候选 arm_configs 内
        if EXPECTED_ABG_STR[a] not in arm_cfg_str.get(a, ""):
            fail(f"E125: arm_configs[{a}]={arm_cfg_str.get(a)!r} 不含权威"
                 f"三元组 {EXPECTED_ABG_STR[a]!r}——GPT f4139b5 ③ 配置口径"
                 f"冲突（aavg 须为 (.125,.125,.375) 非 (0,0)），fail closed")

    pt = d["per_task"]
    for a in ARMS:
        if a == "mavg":
            continue
        for t in TASKS:
            if f"{a}/{t}" not in pt:
                fail(f"E125: per_task 缺 {a}/{t}")

    # 重算复核：AVG5 与 CI95 必须与落盘判决逐位吻合（1e-9）
    quality = {}
    for a in ARMS:
        if a == "mavg":
            scores = [pt[f"mavg-proxy/{t}"]["mavg"] if False else None
                      for t in TASKS]
        entry = {}
        if a == "mavg":
            # mavg 锚点逐任务分数嵌在四候选臂条目的 "mavg" 字段（E125 判决
            # 落盘结构，candidates_avg5 里有锚点 AVG5）。四臂交叉核对一致
            # 才消费——同一 E123 锚点复用，任一分歧 fail closed。
            cand_arms = [x for x in ARMS if x != "mavg"]
            mavg_scores = []
            for t in TASKS:
                vals = [pt[f"{c}/{t}"]["mavg"] for c in cand_arms]
                if len(set(vals)) != 1:
                    fail(f"E125: mavg 锚点分数四臂交叉核对不一致 task={t} "
                         f"{vals}")
                mavg_scores.append(vals[0])
            avg5 = sum(mavg_scores) / 5
            if abs(avg5 - d["mavg_anchor_avg"]) > 1e-9:
                fail(f"E125: mavg 锚点 AVG5 重算 {avg5} != 落盘 "
                     f"{d['mavg_anchor_avg']}")
            entry = {"avg5": avg5, "delta_vs_mavg": 0.0,
                     "ci95_vs_mavg": [0.0, 0.0], "significant_vs_mavg": None,
                     "delta_vs_fullkv_point": avg5 - d["fullkv_anchor_avg"]}
        else:
            scores = [pt[f"{a}/{t}"]["arm"] for t in TASKS]
            avg5 = sum(scores) / 5
            if abs(avg5 - d[f"{a}_avg"]) > 1e-9:
                fail(f"E125: {a} AVG5 重算 {avg5} != 落盘 {d[f'{a}_avg']}")
            deltas = [pt[f"{a}/{t}"]["delta_vs_mavg"] for t in TASKS]
            lo, hi = paired_bootstrap(deltas)
            if abs(lo - d[f"{a}_ci95"][0]) > 1e-9 or \
               abs(hi - d[f"{a}_ci95"][1]) > 1e-9:
                fail(f"E125: {a} CI95 重算 [{lo},{hi}] != 落盘 "
                     f"{d[f'{a}_ci95']}")
            sig = (lo > 0) or (hi < 0)
            if sig != d[f"{a}_significant_vs_mavg"]:
                fail(f"E125: {a} significant 重算 {sig} != 落盘 "
                     f"{d[f'{a}_significant_vs_mavg']}")
            # 补充计算（E128 新增，协议同 E125 B=10000 seed=42，非 E125 产物）：
            # vs fullkv 配对差 CI——供「点估计高于 FullKV」主张的显著性边界核查
            dfkv = [pt[f"{a}/{t}"]["delta_vs_fullkv"] for t in TASKS]
            flo, fhi = paired_bootstrap(dfkv)
            entry = {
                "avg5": avg5,
                "delta_vs_mavg": avg5 - d["mavg_anchor_avg"],
                "ci95_vs_mavg": [lo, hi],
                "significant_vs_mavg": sig,
                "delta_vs_fullkv_point": avg5 - d["fullkv_anchor_avg"],
                "ci95_vs_fullkv_recomputed_e128": [flo, fhi],
                "significant_vs_fullkv": (flo > 0) or (fhi < 0),
                "per_task_scores": {t: pt[f"{a}/{t}"]["arm"] for t in TASKS},
            }
        quality[a] = entry
    quality["fullkv_anchor_avg"] = d["fullkv_anchor_avg"]
    quality["verdict_e125"] = d["verdict"]
    return quality


# ------------------------------------------------------------------ E126
def load_e126(prov):
    """五臂修正后计时。只消费 correction 段存在的文件（任务硬约束 4：
    correction 缺失 = 修正前版本，立即 fail-closed 不猜测）。"""
    timing = {}
    for a in ARMS:
        d = load_json(E126_PATH[a], f"E126-{a}")
        corr = d.get("correction")
        if not corr or corr.get("id") != CORR_ID or not corr.get("applied"):
            fail(f"E126-{a}: correction 段缺失或非 {CORR_ID}——修正前版本"
                 f"（S2 父子双计未排除），拒绝消费，fail closed")
        ident = (d.get("meta") or {}).get("identity") or {}
        md5 = ((ident.get("tli_indexer") or {}).get("md5") or "")
        if md5 != TLI_INDEXER_MD5:
            fail(f"E126-{a}: tli_indexer md5={md5} != 权威锚点 "
                 f"{TLI_INDEXER_MD5}——源身份漂移，fail closed")
        if (d.get("meta") or {}).get("gpu") != "NVIDIA H20-3e":
            fail(f"E126-{a}: gpu != NVIDIA H20-3e——跨机混比无意义")
        proto = d["meta"]["protocol"]
        if proto["sizes"] != SIZES or proto["steps"] != 128 or \
           proto["reps"] != 10 or proto["B"] != 1 or proto["q_len"] != 1:
            fail(f"E126-{a}: protocol 与 E128 消费口径不符 {proto}")
        ac = d["meta"]["arm_config"]
        exp = EXPECTED_ABG[a]
        got_abg = (ac["alpha"], ac["beta"], ac["gamma"])
        if got_abg != exp:
            fail(f"E126-{a}: arm_config (α,β,γ)={got_abg} != 权威 {exp}"
                 f"——GPT f4139b5 ③ 配置口径冲突")
        prov[f"e126_timing_{a}"] = {
            "path": os.path.relpath(E126_PATH[a], REPO),
            "sha256": sha256_of(E126_PATH[a]),
            "correction_id": corr["id"], "correction_applied": corr["applied"],
            "consumed_fields": [
                "cases[].selector_total_ms_median_est（修正后互斥阶段中位数和，"
                "从 stage_ms_samples fresh 重算复核）",
                "cases[].wall_ns_selector_median（055 host wall 口径）",
                "cases[].s8_ms_median（S8=eager_decoding 全长 PV，GPT ④）",
                "cases[].e2e_128step_wall_ns_timing_off（混合时钟微基准，"
                "GPT ②：仅 E126 内部可比）",
                "cases[].stage_ms_median（分阶段中位数，E127 覆盖面测算）",
                "cases[].amortization.selector_ms/s2_far_ms/s2_near_ms"
                "（rep0 全 128 步摊销曲线：首步超额=冷建簇）",
                "cases[].gate_last_step（sel_tokens mean=min=max=1024 门禁）"],
        }

        cells = {}
        seen = set()
        for c in d["cases"]:
            data, S = c["data"], c["S"]
            if data not in DATA_KINDS or S not in SIZES:
                fail(f"E126-{a}: 非法格 {data}/S={S}")
            key = (data, S)
            if key in seen:
                fail(f"E126-{a}: 重复格 {key}")
            seen.add(key)
            g = c["gate_last_step"]
            if not g["out_finite"] or g["sel_tokens_mean"] != 1024 or \
               g["sel_tokens_min"] != 1024 or g["sel_tokens_max"] != 1024:
                fail(f"E126-{a}/{data}/S={S}: fail-closed 门未命中 "
                     f"sel_tokens mean/min/max=1024（实际 {g}）")
            # 修正值复核：从持久化 stage_ms_samples 按互斥白名单 fresh 重算
            samples = c["stage_ms_samples"]
            recomp = sum(median(samples[k]) for k in EXCLUSIVE_STAGE_KEYS
                         if samples.get(k))
            if abs(recomp - c["selector_total_ms_median_est"]) > 1e-6:
                fail(f"E126-{a}/{data}/S={S}: selector_total 重算 {recomp} != "
                     f"落盘 {c['selector_total_ms_median_est']}"
                     f"（1e-6 容差）——修正口径不可信，fail closed")
            sm = c["stage_ms_median"]
            missing = [k for k in EXCLUSIVE_STAGE_KEYS if sm.get(k) is None]
            if missing:
                fail(f"E126-{a}/{data}/S={S}: stage_ms_median 缺阶段 {missing}")
            am = c["amortization"]
            if len(am["selector_ms"]) != 128:
                fail(f"E126-{a}/{data}/S={S}: amortization 曲线长度 "
                     f"{len(am['selector_ms'])} != 128")
            steady_amort = median(am["selector_ms"][1:])
            first = am["selector_ms"][0]
            s2f0 = am["s2_far_ms"][0]
            s2n0 = am["s2_near_ms"][0]
            # ---- 建簇重建分析（E128 新增）：簇臂建簇并非严格一次性——
            #      rep0 摊销曲线中出现多个尖峰（步 0 初建 + 随 S 跨阈的
            #      重建）。尖峰判据 = 超过稳态中位数 2×；非簇臂曲线平稳
            #      无尖峰（max < 2×median 实测成立）。
            curve = am["selector_ms"]
            spikes = [(i, v) for i, v in enumerate(curve)
                      if v > 2.0 * steady_amort]
            rebuilds = [(i, v) for i, v in spikes if i > 0]
            rebuild_excess = sum(v - steady_amort for _, v in rebuilds)
            cells[f"{data}/S{S}"] = {
                "selector_total_ms_median_est": r6(
                    c["selector_total_ms_median_est"]),
                "wall_selector_ms": r6(c["wall_ns_selector_median"] / 1e6),
                "s8_ms_median": r6(c["s8_ms_median"]),
                "e2e_128step_ms_timing_off_mixed_clock": r6(
                    c["e2e_128step_wall_ns_timing_off"] / 1e6),
                "amort_first_step_selector_ms": r6(first),
                "amort_steady_selector_ms_median": r6(steady_amort),
                "amort_steady_selector_ms_max": r6(max(curve[1:])),
                "initial_build_excess_ms": r6(first - steady_amort),
                "s2_far_first_ms": r6(s2f0),
                "s2_near_first_ms": r6(s2n0),
                "rebuild_spike_n_in_128steps": len(rebuilds),
                "rebuild_steps": [i for i, _ in rebuilds],
                "rebuild_excess_ms_in_128steps": r6(rebuild_excess),
                "rebuild_max_ms": r6(max((v for _, v in rebuilds), default=0.0)),
                "amort_mean_selector_ms_128step_window": r6(sum(curve) / 128.0),
                "stage_ms_median": {k: r6(sm[k]) for k in EXCLUSIVE_STAGE_KEYS},
                "opt_stage_share_of_selector": r6(
                    sum(sm[k] for k in OPT_STAGE_KEYS)
                    / c["selector_total_ms_median_est"]),
            }
        if len(cells) != 10:
            fail(f"E126-{a}: 格数 {len(cells)} != 10（2 data_kinds × 5 S）")
        timing[a] = {
            "arm_config_e126": ac,
            "cells": cells,
            "is_cluster_arm": ac.get("enable_kmeans", False),
        }
    return timing


# ------------------------------------------------------------------ E127
def load_e127(prov):
    base = load_json(E127_BASE, "E127-baseline")
    it8 = load_json(E127_ITER8, "E127-iter8")
    f64 = load_json(E127_FIN64, "E127-final64k")
    f8 = load_json(E127_FIN8, "E127-final8k")
    for label, d in (("baseline", base), ("iter8", it8),
                     ("final_64k", f64), ("final_8k", f8)):
        cfg = d["config"]
        if (cfg["alpha"], cfg["beta"], cfg["gamma"]) != (0.25, 0.125, 0.625):
            fail(f"E127-{label}: bench config (α,β,γ) != mavg(.25,.125,.625)"
                 f"——配对组非 mavg 配置，覆盖面推断口径冲突")
        if d["gpu"] != "NVIDIA H20-3e":
            fail(f"E127-{label}: gpu != NVIDIA H20-3e——跨机混比无意义")
    prov["e127_stage_bench"] = {
        "baseline": {"path": os.path.relpath(E127_BASE, REPO),
                     "sha256": sha256_of(E127_BASE)},
        "iter8": {"path": os.path.relpath(E127_ITER8, REPO),
                  "sha256": sha256_of(E127_ITER8)},
        "final_64k": {"path": os.path.relpath(E127_FIN64, REPO),
                      "sha256": sha256_of(E127_FIN64)},
        "final_8k": {"path": os.path.relpath(E127_FIN8, REPO),
                     "sha256": sha256_of(E127_FIN8)},
        "consumed_fields": ["config(α,β,γ)/gpu", "runs[].step_gpu_sum_ms",
                             "runs[].step_cpu_wall_ms",
                             "runs[].{prepare_index,compute_score,"
                             "compute_mask,metrics,eager_decoding}"
                             ".gpu_ms_median", "profile.one_step_cuda_kernels"],
        "bench_config_is_mavg": True,
    }

    def index(d):
        out = {}
        for run in d["runs"]:
            out[(run["S0"], run["layer_kind"])] = run
        return out

    bi, ii = index(base), index(it8)
    if set(bi) != set(ii) or len(bi) != 8:
        fail(f"E127: baseline/iter8 run 集合不一致或非 8 组（{sorted(bi)}）")
    paired = {}
    for k in sorted(bi):
        b, i = bi[k], ii[k]
        if b["step_gpu_sum_ms"] <= 0 or i["step_gpu_sum_ms"] <= 0:
            fail(f"E127: {k} 非正计时——数据损坏")
        paired[f"S{k[0]}/{k[1]}"] = {
            "baseline_gpu_ms": r6(b["step_gpu_sum_ms"]),
            "iter8_gpu_ms": r6(i["step_gpu_sum_ms"]),
            "gpu_speedup": r6(b["step_gpu_sum_ms"] / i["step_gpu_sum_ms"]),
            "baseline_wall_ms": r6(b["step_cpu_wall_ms"]),
            "iter8_wall_ms": r6(i["step_cpu_wall_ms"]),
            "wall_speedup": r6(b["step_cpu_wall_ms"] / i["step_cpu_wall_ms"]),
            "stage_gpu_baseline": {
                st: r6(b[st]["gpu_ms_median"]) for st in
                ("prepare_index", "compute_score", "compute_mask",
                 "metrics", "eager_decoding")},
            "stage_gpu_iter8": {
                st: r6(i[st]["gpu_ms_median"]) for st in
                ("prepare_index", "compute_score", "compute_mask",
                 "metrics", "eager_decoding")},
        }
    # final 复核对（iter8 定案恢复逐字一致口径，2% 容差）
    fi64 = index(f64)
    fi8 = index(f8)
    for kind in ("normal", "skip"):
        k64 = (65536, kind)
        k8 = (8192, kind)
        if k64 not in fi64 or k8 not in fi8:
            fail("E127: final 文件缺 64K/8K 复核 run")
        for metric in ("step_gpu_sum_ms", "step_cpu_wall_ms"):
            if abs(fi64[k64][metric] - ii[k64][metric]) / ii[k64][metric] > 0.02:
                fail(f"E127: final_64k {kind} {metric} 偏离 iter8 超 2%"
                     f"（{fi64[k64][metric]} vs {ii[k64][metric]}）")
            if abs(fi8[k8][metric] - ii[k8][metric]) / ii[k8][metric] > 0.02:
                fail(f"E127: final_8k {kind} {metric} 偏离 iter8 超 2%"
                     f"（{fi8[k8][metric]} vs {ii[k8][metric]}）")
    headline = paired["S65536/normal"]
    return paired, headline, {
        "one_step_cuda_kernels": {"baseline": base["profile"]
                                  ["one_step_cuda_kernels"],
                                  "iter8": it8["profile"]
                                  ["one_step_cuda_kernels"]},
    }


# ------------------------------------------------------------------ E109
def load_e109(prov):
    d = load_json(E109_PATH, "E109")
    prov["e109_screen4_selection"] = {
        "path": os.path.relpath(E109_PATH, REPO),
        "sha256": sha256_of(E109_PATH),
        "consumed_fields": [f"tasks[{k}].AVG5（旧口径 B10 前，5 任务含 "
                            f"repobench-p；用途=跨口径符号稳定性论证）"
                            for k in E109_KEYS.values()],
    }
    out = {}
    for a, k in E109_KEYS.items():
        if k not in d["tasks"]:
            fail(f"E109: 缺臂最佳配置键 {k}")
        out[a] = d["tasks"][k]["AVG5"]
    return out


# ------------------------------------------------------------------ 分析
def pareto_analysis(quality, timing):
    """(AVG5 ↑, 稳态选择器成本 ↓) 点估计支配关系 + 冷建簇第三维。

    成本向量 = 10 格（2 data_kind × 5 S）selector_total_ms_median_est。
    支配 = 质量点估计更高 AND 成本向量逐格不高于（至少一格严格更低）。
    显著性边界如实标注：除 mminmax vs mavg 外全部臂间质量差 CI 含 0，
    支配关系只是点估计层面的排序主张。
    """
    cost = {}
    cold = {}
    for a in ARMS:
        cells = timing[a]["cells"]
        cost[a] = [cells[f"{dk}/S{s}"]["selector_total_ms_median_est"]
                   for dk in DATA_KINDS for s in SIZES]
        # 冷成本第三维 = 初建超额 + 128 步窗内重建尖峰超额（簇臂建簇非一次性）
        cold[a] = [cells[f"{dk}/S{s}"]["initial_build_excess_ms"]
                   + cells[f"{dk}/S{s}"]["rebuild_excess_ms_in_128steps"]
                   for dk in DATA_KINDS for s in SIZES]
    dom = {}
    for a in ARMS:
        dom[a] = {"dominates_point_estimate": [], "dominated_by_point_estimate": []}
    for a in ARMS:
        for b in ARMS:
            if a == b:
                continue
            q = quality[a]["avg5"] > quality[b]["avg5"]
            c = all(x <= y for x, y in zip(cost[a], cost[b])) and \
                any(x < y for x, y in zip(cost[a], cost[b]))
            k = all(x <= y for x, y in zip(cold[a], cold[b])) and \
                any(x < y for x, y in zip(cold[a], cold[b]))
            if q and c:
                dom[a]["dominates_point_estimate"].append(
                    {"arm": b, "with_cold_build_dimension": bool(k)})
                dom[b]["dominated_by_point_estimate"].append(
                    {"arm": a, "with_cold_build_dimension": bool(k)})
    any_dom = any(dom[a]["dominates_point_estimate"] for a in ARMS)
    return {
        "cost_unit": "selector_total_ms_median_est（ms，10 格 = "
                     "structured/singleton × S 4K-64K）",
        "cold_unit": "初建超额 initial_build_excess_ms + 128 步窗内重建尖峰"
                     "超额 rebuild_excess_ms_in_128steps（簇臂建簇非一次性）",
        "significance_note": "除 mminmax（vs mavg CI [-1.56,-0.08] 显著更差）"
                             "外，臂间质量差 bootstrap CI 均含 0——以下支配"
                             "关系是点估计排序，不构成统计证明",
        "dominance": dom,
        "frontier_summary": (
            "质量-稳态成本二维平面上无任何严格支配对：每对臂或质量点估计"
            "交叉或成本向量交叉（mavg vs aavg 在 S=16K 结构化成本交叉；簇"
            "臂质量点估计更高但成本全面更高；mminmax 成本最低但质量显著更"
            "差）——五臂构成真实权衡前沿。加入建簇+重建成本维后：cclustersim"
            "相对 mavg 稳态更贵且建簇不摊销（任何 decode 长度总成本更高的"
            "点估计支配关系成立）；cavgsim 64K 回本需 26,000+ 步（steady-only"
            "口径）且重建复发继续侵蚀——簇臂在总成本维被零建簇臂（mavg/"
            "aavg）点估计支配" if not any_dom else
            "存在点估计支配对，见 dominance"),
    }


def amortization_horizon(timing):
    """簇臂建簇相对 mavg 锚的摊销视野。

    三个口径如实分开：
      - steady_only：稳态中位数差回本步数（假设建簇永不再发生——对
        cclustersim 已证伪，见 rebuild 字段，仅供下界参考）；
      - window_effective_128：rep0 实测 128 步窗有效均值（含初建+全部
        重建尖峰）——簇臂在该窗内是否比 mavg 贵；
      - 重建复发事实：128 步窗内重建尖峰次数/超额。
    """
    out = {}
    for a in CLUSTER_ARMS:
        rows = {}
        for dk in DATA_KINDS:
            for s in SIZES:
                key = f"{dk}/S{s}"
                ca = timing[a]["cells"][key]
                cm = timing["mavg"]["cells"][key]
                diff = (cm["selector_total_ms_median_est"]
                        - ca["selector_total_ms_median_est"])
                initial = ca["initial_build_excess_ms"]
                eff_a = ca["amort_mean_selector_ms_128step_window"]
                eff_m = cm["amort_mean_selector_ms_128step_window"]
                if diff <= 0:
                    steady_status = ("never（簇臂稳态亦更贵，稳态口径下"
                                     "任何 decode 长度不摊销）")
                    steady_steps = None
                else:
                    steady_steps = r6(initial / diff)
                    steady_status = f"steady_only_amortize_after_steps≈{initial / diff:.0f}"
                rows[key] = {
                    "initial_build_excess_ms": r6(initial),
                    "rebuild_spike_n_in_128steps":
                        ca["rebuild_spike_n_in_128steps"],
                    "rebuild_excess_ms_in_128steps":
                        ca["rebuild_excess_ms_in_128steps"],
                    "steady_diff_vs_mavg_ms_per_step": r6(diff),
                    "steady_only_horizon": steady_status,
                    "steady_only_steps_to_amortize": steady_steps,
                    "window_effective_128_ms_arm": r6(eff_a),
                    "window_effective_128_ms_mavg": r6(eff_m),
                    "window_verdict": (
                        "arm 更贵（128 步窗有效均值含建簇+重建后仍高于 "
                        "mavg）" if eff_a >= eff_m else
                        "arm 略便宜（128 步窗有效均值低于 mavg——仅小 S "
                        "短窗可能，重建复发会侵蚀该优势）"),
                }
        out[a] = rows
    return out


def build_recommendation(quality, timing, e127_paired, headline, e109):
    """论文主方法推荐（分析支撑，最终裁决权在主 AI Opus）。"""
    cc = quality["cclustersim"]
    mm = quality["mminmax"]
    steady64 = {a: timing[a]["cells"]["structured/S65536"]
                ["selector_total_ms_median_est"] for a in ARMS}
    rel = steady64["cclustersim"] / steady64["mavg"] - 1.0
    c64 = timing["cclustersim"]["cells"]
    c64s = c64["structured/S65536"]
    c64i = c64["singleton/S65536"]
    v64s = timing["cavgsim"]["cells"]["structured/S65536"]
    v64i = timing["cavgsim"]["cells"]["singleton/S65536"]
    e2e64 = {a: timing[a]["cells"]["structured/S65536"]
             ["e2e_128step_ms_timing_off_mixed_clock"] for a in ARMS}
    reasons = [
        {"id": "R1-quality-unproven",
         "statement": "无臂被证明优于 mavg：E125 全部四候选 vs mavg 的 "
                      "bootstrap CI 均含 0（cclustersim +0.34 "
                      f"[{cc['ci95_vs_mavg'][0]:+.2f},{cc['ci95_vs_mavg'][1]:+.2f}]、"
                      "aavg +0.21 [-0.25,+0.86]、cavgsim -0.18 [-0.44,+0.09]）"
                      "——084 措辞纪律下只能写「未证明差异」，不足以支撑换锚。"
                      "唯一显著判决是 mminmax 显著更差"
                      f"（{mm['delta_vs_mavg']:+.2f} "
                      f"[{mm['ci95_vs_mavg'][0]:+.2f},{mm['ci95_vs_mavg'][1]:+.2f}]）。",
         "data_source": "e125_screen_verdict.json（本脚本重算复核通过）",
         "uncertainty": "cclustersim 点估计最高且高于 FullKV 锚点"
                        f"（+{cc['delta_vs_fullkv_point']:.2f}，vs fullkv "
                        "CI 补算含 0）——若 E116b 全量咬合后 CI 不含 0 可重议"},
        {"id": "R2-steady-cost",
         "statement": f"稳态选择器成本（修正后互斥阶段中位数和，64K "
                      f"structured）：mavg {steady64['mavg']:.4f}ms、"
                      f"aavg {steady64['aavg']:.4f}、cavgsim "
                      f"{steady64['cavgsim']:.4f}、mminmax "
                      f"{steady64['mminmax']:.4f}、cclustersim "
                      f"{steady64['cclustersim']:.4f}——cclustersim vs mavg "
                      f"相对差 {rel * 100:+.2f}%（GPT f4139b5 ①：3.16% 口径，"
                      "不得写 <2%），且 10 格成本向量无一格低于 mavg。",
         "data_source": "e126_stage_timing_*.json 修正后字段",
         "uncertainty": "单层 microbench（计时插桩开），launch-gap 口径下"
                        "主机侧差异可能低估；量级本身很小（<0.2ms/步）"},
        {"id": "R3-cold-build",
         "statement": f"cclustersim 初建簇首步超额 {c64s['initial_build_excess_ms']:.0f}ms"
                      f"/层/序列 @64K structured（cavgsim "
                      f"{v64s['initial_build_excess_ms']:.0f}ms；非簇臂首步"
                      "超额 ~0.06ms 量级），随 S 超线性（cclustersim "
                      "structured：4K 10.6ms → 16K 120.8 → 64K 1679.7）。"
                      "三条 E128 新增事实使成本面进一步恶化："
                      f"(a) 建簇非一次性——64K 128 步窗内观测到 "
                      f"{c64s['rebuild_spike_n_in_128steps']} 次重建尖峰"
                      f"（~{c64s['rebuild_max_ms']:.0f}ms/次，步 "
                      f"{c64s['rebuild_steps']}，随 S 跨阈复发）；"
                      f"(b) 建簇成本强数据依赖——singleton 64K 初建 "
                      f"{c64i['initial_build_excess_ms']:.0f}ms（cavgsim "
                      f"{v64i['initial_build_excess_ms']:.0f}ms），为 "
                      "structured 的 4.7×/5.7×——structured 是有利下界，"
                      "真实数据建簇成本落在两者之间且未知；"
                      "(c) E126 内部可比的 128 步 e2e 窗（混合时钟微基准，"
                      f"GPT ②）：cclustersim {e2e64['cclustersim']:.0f}ms "
                      f"vs mavg {e2e64['mavg']:.0f}ms（structured）/"
                      f"singleton {c64i['e2e_128step_ms_timing_off_mixed_clock']:.0f}ms"
                      f"（mavg {timing['mavg']['cells']['singleton/S65536']['e2e_128step_ms_timing_off_mixed_clock']:.0f}）"
                      "——与摊销曲线自洽（初建+重建+稳态差+S8 分解吻合）。"
                      "cclustersim 稳态亦更贵（R2），故建簇成本在任何 decode "
                      "长度下都不摊销；cavgsim 稳态略便宜但 64K 回本步数 "
                      "26,000+ 步（steady-only 口径，重建复发再侵蚀）。",
         "data_source": "e126 amortization rep0 首 128 步摊销曲线（重建尖峰"
                        "判据 = 超稳态中位数 2×）+ e2e_128step_wall_ns_"
                        "timing_off（仅 E126 内部可比，禁止与 E127 配对组"
                        "提速混算）",
         "uncertainty": "建簇为每序列每层发生（服务冷启动税）；重建尖峰"
                        "cadence（~30-70 步/次@64K）与单次量级随 S 与数据"
                        "分布变化，128 步窗只是采样；singleton/structured "
                        "为合成数据两界，真实 LongBench 分布未测"},
        {"id": "R4-e127-convergence",
         "statement": "E127 优化对象是共享热路径（S0/S1 增量缓存、S3 "
                      "compute_score Triton、S8 eager gather、metrics 惰性化），"
                      "激活门与 method 臂无关，结构上五臂共享；但实测只在 "
                      "mavg 配置下进行（64K normal GPU "
                      f"{headline['baseline_gpu_ms']:.2f}→"
                      f"{headline['iter8_gpu_ms']:.2f}ms "
                      f"{headline['gpu_speedup']:.2f}× / wall "
                      f"{headline['baseline_wall_ms']:.2f}→"
                      f"{headline['iter8_wall_ms']:.2f}ms "
                      f"{headline['wall_speedup']:.2f}×）。簇臂的 S2 冷建簇"
                      "（fast 路径原样调用 _maybe_build_kmeans）与 S4 簇内 "
                      "greedy 打分、compute_mask 核心（S5-S7，iter9-11 负结果）"
                      "未被优化——共享路径 5.47× 压缩后，臂间残余差异与冷建簇"
                      "在优化后系统里的相对占比上升，簇臂的相对成本位置不改善"
                      "反而恶化。切换主方法将使 5.47×/4.76× 主张从实测降级为"
                      "外推。",
         "data_source": "e127_stage_bench_baseline/iter8.json（config=mavg）+ "
                        "S-T024 §2/§4/§6.1 + tli_indexer_fast.py 门控条件"
                        "（行为门禁含 E125 尾巴 10/10 与 kmeans 守卫 5/5 "
                        "env=1 全绿——簇臂行为已验，性能未测）",
         "uncertainty": "五臂 fast 路径收益为结构外推（S0/S1/S3/S8 阶段五臂"
                        "逐位同路径，E126 佐证）；E127 单层 event 同步口径是"
                        "保守下界（S-T024 §6.4：36 层流水收益测不出）"},
        {"id": "R5-cross-caliber-stability",
         "statement": f"跨口径符号稳定性：E109 旧口径 mavg 最佳 "
                      f"{e109['mavg']:.2f}，cclustersim {e109['cclustersim']:.2f}"
                      f"（−{e109['mavg'] - e109['cclustersim']:.2f} 居后）、"
                      f"aavg {e109['aavg']:.2f}；E125 新口径符号翻转"
                      "（cclustersim +0.34 反超）——n=5 任务噪声主导，"
                      "不足以在两口径间换锚；mavg 两口径均居首/前列，"
                      "稳健性最好。",
         "data_source": "e109_screen4_selection.json（旧口径 B10 前）vs "
                        "e125_screen_verdict.json（新口径 E121 修复后）",
         "uncertainty": "两口径任务集/协议不同（repobench-p vs repobench），"
                        "只作符号稳定性参考，不作分数横比"},
        {"id": "R6-paper-story",
         "statement": "论文主张链已锚定 mavg：E100 13 任务 FullKV 持平"
                      "（+0.42，CI [-0.18,+1.02] 含 0）+ E127 同预算 kernel "
                      "加速（mavg 配置实测 5.47×）。换 cclustersim 需重跑"
                      "全量验证（E116b 式咬合）+ fast-path 臂级基准 + prefill "
                      "suffix 不变性验收，成本大而质量收益未证明。",
         "data_source": "e100_bootstrap_ci.json 结论（历史判决）+ "
                        "E127 配对组实测口径",
         "uncertainty": "E100 持平口径基于 mavg 臂 13 任务全量；"
                        "cclustersim 无对应全量数据"},
    ]
    rec = {
        "main_method_recommendation": "mavg（维持论文主方法）",
        "decision_authority": "主 AI Opus（10-11 授权自主决策权）；本推荐为"
                              "分析支撑，供裁决",
        "reasons": reasons,
        "alternative_disposition": {
            "cclustersim": "进消融/讨论段如实报告：E125 点估计海选冠军"
                           "（+0.34 vs mavg，未证明显著；点估计亦高于 "
                           "FullKV +0.23）——但冷建簇超线性 + E127 未覆盖，"
                           "不适合作主方法",
            "aavg": "零建簇成本、稳态与 mavg 同档（10 格内有格交叉），点估计"
                    "+0.21 未证明且跨口径符号翻转（E109 −0.52）——不换锚，"
                    "可作消融臂",
            "cavgsim": "被 mavg/aavg 点估计支配（质量更低 + 冷建簇 1859ms"
                       "@64K）；E123 cavg 线已 NO-GO 关闭",
            "mminmax": "唯一显著判决臂（vs mavg CI [-1.56,-0.08]）——剔除",
        },
        "inflight_conflict_note": "E129 子空间消融在飞以 cclustersim 为冠军"
                                  "臂——若本推荐被采纳，E129 结果仍有效"
                                  "（臂内子空间相对效应），但「冠军」称谓与"
                                  "论文主方法口径需主 AI 对齐；若 E116b 全量"
                                  "咬合后 cclustersim CI 不含 0，本推荐可"
                                  "重新审议（R1 不确定性边界）",
    }
    return rec


# ------------------------------------------------------------------ main
def main():
    prov = {}
    quality = load_e125(prov)
    timing = load_e126(prov)
    e127_paired, headline, e127_profile = load_e127(prov)
    e109 = load_e109(prov)
    pareto = pareto_analysis(quality, timing)
    amort = amortization_horizon(timing)

    # E127 覆盖面（静态事实，来源=代码门控条件 + S-T024 报告；如实区分）
    fast_file = os.path.join(REPO, "sparse_attn", "indexer",
                             "tli_indexer_fast.py")
    if not os.path.exists(fast_file):
        fail("sparse_attn/indexer/tli_indexer_fast.py 不存在——E127 交付物"
             "缺失，覆盖面事实无从锚定")
    e127_coverage = {
        "activation": "TLI_E127_FAST=1 时 patch.py 替换实例化为 TLIIndexerFast；"
                      "激活门（_basis None、非 moba、D=128、sub_key 稳定、"
                      "k_avg 按需缓存、triton_ok）与 method 臂无关",
        "optimized_shared_stages_all_arms": {
            "S0+S1": "prepare_index 增量缓存（iter4/5）——五臂同享",
            "S3_coarse/S3_avg/S3_fine": "compute_score Triton GQA K 复用"
                                        "（iter6/8）——五臂同路径",
            "S8": "eager gather + Triton 融合（iter2/3/7）——五臂同享"
                  "（E126 基线 S8 为全长 PV，GPT ④）",
            "metrics": "惰性化（iter1）",
        },
        "not_optimized": {
            "S2_kmeans_build": "fast 路径原样调用 _maybe_build_kmeans"
                               "（tli_indexer_fast.py L186-188）——簇臂冷建簇"
                               "成本 E127 后原样保留",
            "S4_in_cluster_greedy": "sim_greedy 臂簇内分数调用区未 Triton 化"
                                    "（cclustersim far+near 双侧、cavgsim "
                                    "far 侧）",
            "S5/S6/S7+mask_setup": "compute_mask 核心未结构优化"
                                   "（iter9-11 负结果；S-T024 §6.1 登记为"
                                   "遗留方向：35-launch 段 + launch 排队"
                                   "~800μs）",
        },
        "behavior_gate_coverage": "S-T024 §4：回归 7 套件含 E125 尾巴 10/10、"
                                  "E125 kmeans 守卫 5/5 env=1 全绿——簇臂"
                                  "行为已验证（性能未测）",
        "perf_measured_config": "仅 mavg(.25,.125,.625)（e127_stage_bench_*.json）"
                                "——其余臂 fast 收益为结构外推",
        "sources": ["agent_doc/task/task_details/S-T024.md §1/§2/§4/§6（主仓"
                    "untracked，只读参考）",
                    "sparse_attn/indexer/tli_indexer_fast.py（sha256="
                    + sha256_of(fast_file)[:16] + "…）",
                    "exp/trace/e126_stage_timing.py stage_boundaries"],
    }
    e127_applicability = {}
    for a in ARMS:
        cells = timing[a]["cells"]
        e127_applicability[a] = {
            "fast_path_stage_share_of_steady_selector": {
                dk: {f"S{s}": cells[f"{dk}/S{s}"]["opt_stage_share_of_selector"]
                     for s in SIZES} for dk in DATA_KINDS},
            "untouched_stage_keys": UNOPT_STAGE_KEYS + (
                ["S2_far/S2_near build", "S4 greedy"] if a in CLUSTER_ARMS
                else []),
            "post_e127_cost_status": (
                "实测（e127_stage_bench config=mavg；64K normal 每层每步 "
                f"GPU {headline['iter8_gpu_ms']:.2f}ms / wall "
                f"{headline['iter8_wall_ms']:.2f}ms）" if a == "mavg"
                else "结构外推（fast 激活门与臂无关、共享阶段五臂同路径；"
                     "但未在本臂配置下实测）"),
        }

    # 64K structured 汇总数字（GPT ① 复核：3.16%）
    steady64 = {a: timing[a]["cells"]["structured/S65536"]
                ["selector_total_ms_median_est"] for a in ARMS}
    rel_cc = steady64["cclustersim"] / steady64["mavg"] - 1.0

    payload = {
        "meta": {
            "probe": "E128 五臂质量×性能综合权衡表（#216）——E125 质量分 × "
                     "E126 修正后计时 × E127 优化定案",
            "generated": "2026-10-12",
            "worktree_branch": "e128-tradeoff",
            "base_commit": os.popen(
                "git -C " + REPO + " rev-parse --short HEAD").read().strip(),
            "scope_limits": "structured/singleton 合成数据、B=1、q_len=1、"
                            "D′ off（E126 layer_skip off / E127 normal 层）、"
                            "单层 microbench、H20-3e；E126 为优化前基线"
                            "（计时插桩开），E127 为独立 harness 配对组",
            "gpt_f4139b5_constraints": {
                "1_steady_rel_diff_64k": "cclustersim vs mavg 相对差 "
                                         f"{rel_cc * 100:.2f}%（重算核实，"
                                         "非 <2%）",
                "2_e2e_mixed_clock": "e2e_128step_*_mixed_clock 字段"
                                     "仅 E126 内部可比，禁止与 E127 配对组"
                                     "提速混算",
                "3_aavg_config": "(α,β,γ)=(.125,.125,.375)（E125 字串 + "
                                 "E126 数值双门禁通过）",
                "4_s8_semantics": "E126 基线 S8 = eager_decoding 全长 PV"
                                  "（非只读 K2 个 KV）；E127 iter2 后消费端"
                                  "gather 选中子集",
            },
            "statistical_discipline": "084 措辞：CI 含 0 写「未证明差异」"
                                      "不写「持平/等价」；支配关系为点估计"
                                      "排序主张",
            "consumed_only_corrected_e126_fields": True,
        },
        "provenance": prov,
        "quality_e125": quality,
        "steady_selector_cost_e126": {
            a: {"cells": timing[a]["cells"],
                "is_cluster_arm": timing[a]["is_cluster_arm"],
                "arm_config_e126": timing[a]["arm_config_e126"]}
            for a in ARMS},
        "cold_build_e126": {
            a: {f"{dk}/S{s}": {
                "initial_build_excess_ms":
                    timing[a]["cells"][f"{dk}/S{s}"]["initial_build_excess_ms"],
                "s2_far_first_ms":
                    timing[a]["cells"][f"{dk}/S{s}"]["s2_far_first_ms"],
                "s2_near_first_ms":
                    timing[a]["cells"][f"{dk}/S{s}"]["s2_near_first_ms"],
                "rebuild_spike_n_in_128steps":
                    timing[a]["cells"][f"{dk}/S{s}"]
                    ["rebuild_spike_n_in_128steps"],
                "rebuild_excess_ms_in_128steps":
                    timing[a]["cells"][f"{dk}/S{s}"]
                    ["rebuild_excess_ms_in_128steps"],
                "rebuild_max_ms":
                    timing[a]["cells"][f"{dk}/S{s}"]["rebuild_max_ms"],
                "amort_mean_selector_ms_128step_window":
                    timing[a]["cells"][f"{dk}/S{s}"]
                    ["amort_mean_selector_ms_128step_window"]}
                for dk in DATA_KINDS for s in SIZES} for a in ARMS},
        "amortization_horizon_vs_mavg": amort,
        "e2e_128step_ms_e126_internal_only": {
            a: {f"{dk}/S{s}":
                timing[a]["cells"][f"{dk}/S{s}"]
                ["e2e_128step_ms_timing_off_mixed_clock"]
                for dk in DATA_KINDS for s in SIZES} for a in ARMS},
        "e127_paired_baseline_vs_iter8": e127_paired,
        "e127_headline_64k_normal": headline,
        "e127_profile": e127_profile,
        "e127_coverage": e127_coverage,
        "e127_applicability_per_arm": e127_applicability,
        "e109_old_caliber_avg5_cross_check": e109,
        "pareto": pareto,
        "recommendation": build_recommendation(
            quality, timing, e127_paired, headline, e109),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    # ---------------- 终端表 ----------------
    print("=" * 78)
    print("E128 五臂质量×性能权衡表（#216）——AVG5 = E125 difflib 5 任务；"
          "成本 = E126 修正后")
    print("=" * 78)
    print(f"{'臂':<12}{'AVG5':>8}{'Δvs mavg':>10}{'CI95':>18}"
          f"{'Δvs FullKV':>12}{'CI95(fkv)':>18}")
    for a in ARMS:
        q = quality[a]
        if a == "mavg":
            print(f"{a:<12}{q['avg5']:>8.2f}{'—':>10}{'—':>18}"
                  f"{q['delta_vs_fullkv_point']:>+12.2f}{'—':>18}")
        else:
            ci = q["ci95_vs_mavg"]
            fci = q["ci95_vs_fullkv_recomputed_e128"]
            print(f"{a:<12}{q['avg5']:>8.2f}{q['delta_vs_mavg']:>+10.2f}"
                  f"[{ci[0]:+.2f},{ci[1]:+.2f}]{'':<4}"
                  f"{q['delta_vs_fullkv_point']:>+12.2f}"
                  f"[{fci[0]:+.2f},{fci[1]:+.2f}]")
    print(f"FullKV 锚点 AVG5 = {quality['fullkv_anchor_avg']:.2f}")
    print(f"E125 verdict: {quality['verdict_e125']}")
    print()
    print("--- 稳态选择器成本（修正后互斥阶段中位数和, ms）与建簇成本 ---")
    print(f"{'臂':<12}{'4K':>8}{'16K':>8}{'64K':>8}{'初建@64K(ms)':>14}"
          f"{'重建尖峰@64K':>14}{'e2e128步@64K(ms)':>18}")
    for a in ARMS:
        c = timing[a]["cells"]
        cell = c["structured/S65536"]
        rb = (f"n={cell['rebuild_spike_n_in_128steps']},"
              f"{cell['rebuild_excess_ms_in_128steps']:.0f}ms"
              if cell["rebuild_spike_n_in_128steps"] else "—")
        init_txt = ("≈0" if abs(cell["initial_build_excess_ms"]) < 0.1
                    else f"{cell['initial_build_excess_ms']:.1f}")
        print(f"{a:<12}"
              f"{c['structured/S4096']['selector_total_ms_median_est']:>8.3f}"
              f"{c['structured/S16384']['selector_total_ms_median_est']:>8.3f}"
              f"{c['structured/S65536']['selector_total_ms_median_est']:>8.3f}"
              f"{init_txt:>14}"
              f"{rb:>14}"
              f"{cell['e2e_128step_ms_timing_off_mixed_clock']:>18.0f}")
    print("（初建/重建为 structured 口径；singleton 初建：cclustersim "
          f"{timing['cclustersim']['cells']['singleton/S65536']['initial_build_excess_ms']:.0f}ms、"
          f"cavgsim {timing['cavgsim']['cells']['singleton/S65536']['initial_build_excess_ms']:.0f}ms"
          "——structured 是有利下界）")
    print(f"cclustersim vs mavg 稳态相对差 @64K: {rel_cc * 100:+.2f}%"
          "（GPT f4139b5 ① 口径）")
    print()
    print("--- E127 配对组（mavg 配置实测；GPU busy sum / CPU wall, ms/层/步）---")
    for k in sorted(e127_paired):
        p = e127_paired[k]
        print(f"{k:<16}GPU {p['baseline_gpu_ms']:>6.2f}→{p['iter8_gpu_ms']:>5.2f}"
              f" ({p['gpu_speedup']:.2f}×)   wall {p['baseline_wall_ms']:>6.2f}"
              f"→{p['iter8_wall_ms']:>5.2f} ({p['wall_speedup']:.2f}×)")
    print(f"launch/step: {e127_profile['one_step_cuda_kernels']['baseline']} → "
          f"{e127_profile['one_step_cuda_kernels']['iter8']}")
    print()
    print("--- Pareto（点估计；质量差 CI 除 mminmax 外均含 0）---")
    for a in ARMS:
        d = pareto["dominance"][a]
        doms = ", ".join(x["arm"] for x in d["dominates_point_estimate"]) or "—"
        dby = ", ".join(x["arm"] for x in d["dominated_by_point_estimate"]) or "—"
        print(f"{a:<12}支配 [{doms}]  被支配 [{dby}]")
    any_dom = any(pareto["dominance"][a]["dominates_point_estimate"]
                  for a in ARMS)
    print("（质量-稳态成本二维平面上无严格支配对：每对臂或质量交叉或成本"
          "交叉——真实权衡前沿；加建簇+重建成本维后簇臂在任何 decode 长度"
          "的总成本意义上劣于零建簇臂，见摊销视野）" if not any_dom else "")
    print()
    print("--- 簇臂摊销视野（vs mavg；steady-only / 128 步窗有效均值双口径）---")
    for a in CLUSTER_ARMS:
        row = amort[a]
        for dk in DATA_KINDS:
            cells_txt = "  ".join(
                f"S{s}={row[f'{dk}/S{s}']['steady_only_horizon']}"
                for s in SIZES)
            print(f"{a}/{dk}: {cells_txt}")
        print(f"{a} 128 步窗有效均值（含初建+重建）vs mavg："
              + ", ".join(
                  f"{dk}64K {row[f'{dk}/S65536']['window_effective_128_ms_arm']:.2f}"
                  f" vs {row[f'{dk}/S65536']['window_effective_128_ms_mavg']:.2f}ms"
                  f"（{row[f'{dk}/S65536']['window_verdict'][:12]}…）"
                  for dk in DATA_KINDS))
    print()
    rec = payload["recommendation"]
    print("=" * 78)
    print(f"[推荐] 论文主方法：{rec['main_method_recommendation']}")
    for rr in rec["reasons"]:
        print(f"  - {rr['id']}: {rr['statement'][:120]}…")
    print(f"[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
