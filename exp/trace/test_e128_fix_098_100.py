# -*- coding: utf-8 -*-
"""E128 修复 098/099/100/107/108/109/110 红绿套件（worktree e128-fix-107-110，
2026-10-12）。

覆盖 GPT 审计两轮（098/099/100：sglang agent_doc/advice/2026-10-12_0527_
twolevel_bug_audit_08819a7.md；107/108/109/110：2026-10-12_0739_twolevel_
bug_audit_3d5b1b9.md，主 AI 逐条 confirmed）：

  098（P1）Pareto 摘要与 dominance 表矛盾
    - 绿证：frontier_summary 由 dominance 机器生成；表空时摘要不得包含
      任何支配主张；旧版「任何 decode 长度总成本更高的点估计支配关系
      成立」「总成本维被零建簇臂（mavg/aavg）点估计支配」逐句不得复活。
    - 性质测试：任意二维点集加入一个最小化成本维后，新支配集合必须是
      旧支配集合的子集（新增维不可能创造 Pareto 支配）。
    - 分离对象：quality_steady_cost_pareto 与 cost_only_total_order_at_
      horizon（后者 statements 不得含「Pareto 支配」术语）。

  099（P2）统计全称命题为假 + 措辞纪律
    - 绿证：pairwise_ci_all 10 对全枚举 + 锚点逐位复算（aavg-cavgsim
      [+0.110,+0.762]、cavgsim-cclustersim [-0.948,-0.178]，与主 AI
      独立复算一致）；significance_note 不得含旧全称命题；R1 不得同时
      写「四候选 vs mavg CI 均含 0」与 mminmax 显著；R6 的 E100 含零 CI
      写「未证明差异」不写「持平」。

  100（P2）fail-closed 门禁缺口——六组篡改负例全部红证
    - 红证锚点纪律：缺陷基线用固定 SHA 提取（OLD_SHA git show），
      不引用可变分支名；修复前脚本对同一篡改输入 rc==0（GPT 六反例
      逐条复现），修复后脚本 rc!=0 且 [ABORT] 文案命中。

  107（P2）套件浮点逐位比较跨环境不稳定（GPT AMD EPYC 74 PASS/1 FAIL）
    - 缺陷机制：CPython 3.12+ sum() 改用 Neumaier 补偿求和，同一聚合
      算法在不同 CPU/编译环境下浮点末位漂移 ~1e-14（实测 48.12400000
      000001 vs 48.124），重算 vs 提交 JSON 的 dict 逐位比较在本环境
      生成侧全绿、异环境重算侧必红。
    - 修复：重算与提交 JSON 比较中，bool/int/str/None/键集/列表长度
      保持 exact，双方均为 float 的值用明确容差（FLOAT_ABS_TOL=1e-9
      或 FLOAT_REL_TOL=1e-12，常量注释理由）。
    - 红绿：±5e-14 合成漂移必须通过容差门且逐位比较必须判红（缺陷
      机制复现）；±1e-6 必须被容差门拒绝（门不空转）；字符串/bool/
      键集/列表长度篡改必须仍被拒绝（容差不吞结构差异）。
    - 红证锚点：OLD2_SHA（本套件修复前 HEAD）原比较为逐位 ==，
      源码级锚点断言 + 机制级判红双重复现。

  108（P2）E127 baseline/iter8 配对未绑定完整配置
    - 红绿（T7）：iter8 改 K2=2048/D=64/bs=1/torch=fake 保留 α/β/γ/
      GPU/run 键——OLD2_SHA 旧脚本 rc==0 仍发布 5.47× 为「同预算」
      速度比（GPT 反例逐条复现），新脚本 ABORT 且逐字段差异进文案。
    - 绿证：真实四份归档过全键绑定；config canonical sha256 进每组
      配对（e127_paired_* 与 headline 的 config_binding_sha256）+
      provenance.config_binding（canonical/bound_fields/未记录字段
      如实声明——dtype/CUDA/Triton/kernel 开关归档未记录，不作未
      记录字段的同配置事实断言）。

  109（P2）NaN/Inf 穿透 E125/E126/E100 消费路径
    - 红绿（T8/T9/T10）：三路独立 NaN 篡改——E126 selector_total_ms_
      median_est=NaN / E100 claim+main_ci+calibration 联立 NaN /
      E125 aavg 五任务 arm/delta/AVG/CI 协同 NaN——OLD2_SHA 旧脚本
      全部 rc==0（NaN 使 abs()>tol 与含 0 判定静默通过、json.dump
      默认放行），新脚本入口整树 isfinite fail-closed。
    - 绿证：最终 json.dump(allow_nan=False)（NaN 漏网到输出层时
      序列化即崩）；真实输入全树无 NaN（绿证基线隐式覆盖）。

  110（P2）机器表更新但叙事硬编码
    - 性质测试：构造内部自洽且通过全部联立门的合法输入（aavg 五任务
      各 +1，同步 delta/AVG/CI/significant），文字与结构化对象必须
      同步翻转——significance_note「三候选」→「两候选 + aavg 显著
      更好」、R1 出现「被证明优于 mavg 的候选：aavg +1.21
      [+0.75,+1.86]」（全部从结构化字段格式化生成并与之一致）。
    - 红证：同一输入在 OLD2_SHA 旧脚本 rc==0，但其自身结构化对象
      （quality_e125.aavg.significant_vs_mavg=True、pairwise
      mavg-aavg CI 不含 0）与手写叙事（「三候选 CI 均含 0」、
      R1「aavg +0.21 [-0.25,+0.86]」）直接矛盾——GPT 110 反例
      机器复现。

运行：python3 exp/trace/test_e128_fix_098_100.py
  （可用环境变量 E128_TEST_REPO_ROOT 覆盖被测仓根；三跑 + python -O
  双跑由调用方执行，套件自身无 assert、全部显式 check 收集。）
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

REPO = os.environ.get(
    "E128_TEST_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..")))
SCRIPT = os.path.join(REPO, "exp", "trace", "analyze_e128_tradeoff.py")
ARMS = ["mavg", "aavg", "mminmax", "cavgsim", "cclustersim"]
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
B, SEED = 10000, 42
# 修复前主仓 HEAD（固定 SHA 锚点——E128 聚合与旧 JSON 均在此版本；
# 不得改用 main: 等可变分支名，修复合入后红证基线随之失效是预期行为，
# 锚点本体永久有效）
OLD_SHA = "6e918c53d6353b5d7c289c5a712b2bb296cb6f45"
# 107-110 缺陷基线（本套件修复前 HEAD，固定 SHA 锚点同上纪律）：0739
# 审计四项缺口都在「已含 098/099/100 修复」的版本上发现——109 的 NaN
# 穿透与 108 的配置漂移放行、110 的手写叙事矛盾都必须对该版本复现
# 红证（对更早的 6e918c5 复现语义不同：彼时连 E100 联立门都不存在）。
OLD2_SHA = "ce6cde0b95d6ef2bee3b08647cf1e48f03bd1b93"
# 107 浮点容差常量：CPython 3.12+ sum() 改用 Neumaier 补偿求和，同
# 算法跨 CPU/编译环境浮点末位可漂移 ~1e-14（GPT AMD EPYC 实测
# 48.12400000000001 vs 48.124）。abs 1e-9 吸收求和/排序级联误差，且
# 比任何真实数据差异（篡改门禁 T 系最小步长 0.1 分数单位）小 8 个量
# 级；rel 1e-12 覆盖大数值相对误差。bool/int/str/None/键集/列表长度
# 不进容差、保持 exact（deep_close 只对双方均为 float 的值启用容差）。
FLOAT_ABS_TOL = 1e-9
FLOAT_REL_TOL = 1e-12

RESULTS = []


def check(cond, label, detail=""):
    RESULTS.append((bool(cond), label, str(detail)))
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {label}" + (f"  ({detail})" if detail and not cond else ""))
    return bool(cond)


def run_script(repo_root, out_json, opt_o=False):
    env = dict(os.environ)
    env["E128_REPO_ROOT"] = repo_root
    env["E128_OUT_JSON"] = out_json
    cmd = [sys.executable] + (["-O"] if opt_o else []) + [SCRIPT]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def git_show_old(path):
    p = subprocess.run(["git", "-C", REPO, "show", f"{OLD_SHA}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"[ABORT] 无法提取修复前锚点 {OLD_SHA}:{path}"
                 f"——锚点 commit 必须存在于被测仓历史\n{p.stderr}")
    return p.stdout


def git_show_at(sha, path):
    p = subprocess.run(["git", "-C", REPO, "show", f"{sha}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"[ABORT] 无法提取锚点 {sha}:{path}"
                 f"——锚点 commit 必须存在于被测仓历史\n{p.stderr}")
    return p.stdout


# ------------------------------------------------- 107 容差比较器
def deep_close(x, y):
    """重算 vs 提交 JSON 的递归比较（107）。

    双方均为 float 的值用明确容差（abs 1e-9 或 rel 1e-12）；bool/int/
    str/None 与结构（键集、列表长度、类型）一律 exact——身份/离散字段
    不得被容差吞掉。"""
    if isinstance(x, bool) or isinstance(y, bool):
        return x is y
    if isinstance(x, float) and isinstance(y, float):
        return (abs(x - y) <= FLOAT_ABS_TOL
                or abs(x - y) <= FLOAT_REL_TOL * max(abs(x), abs(y)))
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(deep_close(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return (len(x) == len(y)
                and all(deep_close(a, b) for a, b in zip(x, y)))
    return x == y


def drift_floats(obj, delta, parity=0):
    """递归复制 obj，给所有 float 加 ±|delta|（按路径奇偶交替正负——
    模拟 Neumaier 求和的跨环境末位漂移，非单调平移）。"""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj + delta * (1.0 if parity % 2 == 0 else -1.0)
    if isinstance(obj, dict):
        return {k: drift_floats(v, delta, parity + i)
                for i, (k, v) in enumerate(obj.items())}
    if isinstance(obj, list):
        return [drift_floats(v, delta, parity + i)
                for i, v in enumerate(obj)]
    return obj


# ---------------------------------------------------------------- 篡改工厂
INPUT_COPIES = {
    "exp/trace/results/e125_screen_verdict.json": "e125.json",
    "exp/trace/results/e109_screen4_selection.json": "e109.json",
    "exp/trace/results/e100_bootstrap_ci.json": "e100.json",
}
for _a in ARMS:
    INPUT_COPIES[f"exp/trace/results/e126_stage_timing_{_a}.json"] = \
        f"e126_{_a}.json"
for _f in ("baseline", "iter8", "final_64k", "final_8k"):
    INPUT_COPIES[f"exp/results_efficiency/e127_stage_bench_{_f}.json"] = \
        f"e127_{_f}.json"


def build_tamper_repo(tamper_fn):
    """克隆最小输入树到 tmp 仓（真实输入副本 + 篡改钩子）。"""
    tmp = tempfile.mkdtemp(prefix="e128_tamper_")
    for src, dst in INPUT_COPIES.items():
        d = os.path.join(tmp, "orig", dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(REPO, src), d)
    tamper_fn(os.path.join(tmp, "orig"))
    root = os.path.join(tmp, "repo")
    for src, dst in INPUT_COPIES.items():
        out = os.path.join(root, src)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(os.path.join(tmp, "orig", dst), out)
    # 【inplace 合并（2026-10-12）】E127 交付物锚点已从 tli_indexer_fast.py
    # 改为 TLIIndexer 基类内 fast path 实现标记——tamper 树须含基类文件
    # （新脚本锚点）；同时从 git HEAD 提取合并前的 tli_indexer_fast.py
    # 占位（OLD_SHA/OLD2_SHA 旧脚本的存在性锚点仍指向该文件——红证
    # 「缺陷基线接受篡改」要求旧脚本不在存在性检查处提前 fail）。
    tli = os.path.join(root, "sparse_attn", "indexer", "tli_indexer.py")
    os.makedirs(os.path.dirname(tli), exist_ok=True)
    shutil.copy2(os.path.join(REPO, "sparse_attn", "indexer",
                              "tli_indexer.py"), tli)
    fast = os.path.join(root, "sparse_attn", "indexer",
                        "tli_indexer_fast.py")
    _head_fast = subprocess.run(
        ["git", "-C", REPO, "show",
         "HEAD:sparse_attn/indexer/tli_indexer_fast.py"],
        capture_output=True, text=True)
    if _head_fast.returncode == 0:
        with open(fast, "w", encoding="utf-8") as _f:
            _f.write(_head_fast.stdout)
    return tmp, root


def load_orig(name):
    return json.load(open(os.path.join(REPO, INPUT_COPIES_INVERSE[name])))


def orig_path(name):
    for src, dst in INPUT_COPIES.items():
        if dst == name:
            return os.path.join(REPO, src)
    sys.exit(f"[ABORT] 未知输入 {name}")


# ------------------------------------------------- 六组篡改负例（100）
def tamper_manifest_dup(orig_dir):
    """① E125 manifest 第 1 项复制 20 次（只数长度==20 的旧门通过）。"""
    p = os.path.join(orig_dir, "e125.json")
    d = json.load(open(p))
    d["input_manifest"] = [d["input_manifest"][0]] * 20
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_quality_ci_decouple(orig_dir):
    """② aavg/qasper.arm += 5 且同步 aavg_avg += 1（点估计与 CI 来源脱钩）。"""
    p = os.path.join(orig_dir, "e125.json")
    d = json.load(open(p))
    d["per_task"]["aavg/qasper"]["arm"] += 5.0
    d["aavg_avg"] += 1.0
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e126_far_method(orig_dir):
    """③ E126 mavg far_method 改 NOT_MAVG（αβγ 不变，旧门只绑 αβγ）。"""
    p = os.path.join(orig_dir, "e126_mavg.json")
    d = json.load(open(p))
    d["meta"]["arm_config"]["far_method"] = "NOT_MAVG"
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e126_stage_median(orig_dir):
    """④ E126 一格 stage_ms_median.S3_fine=1000（samples/total 不联立）。"""
    p = os.path.join(orig_dir, "e126_mminmax.json")
    d = json.load(open(p))
    d["cases"][0]["stage_ms_median"]["S3_fine"] = 1000.0
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e127_dup_run(orig_dir):
    """⑤ E127 baseline 追加重复 run（字典静默去重后仍 8 键）。"""
    p = os.path.join(orig_dir, "e127_baseline.json")
    d = json.load(open(p))
    import copy
    d["runs"].append(copy.deepcopy(d["runs"][0]))
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e127_nan(orig_dir):
    """⑥ E127 配对 step_gpu_sum_ms 改 NaN（<=0 与偏差比较均 False）。"""
    for name in ("e127_baseline.json", "e127_iter8.json"):
        p = os.path.join(orig_dir, name)
        d = json.load(open(p))
        for r in d["runs"]:
            if r["S0"] == 16384 and r["layer_kind"] == "normal":
                r["step_gpu_sum_ms"] = float("nan")
        json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e127_iter8_config(orig_dir):
    """⑦ E127 iter8 配置漂移（108 GPT 反例）：K2=2048/D=64/bs=1/
    torch=fake，保留 α/β/γ/GPU/run 键——旧门只绑 αβγ+GPU，异配置
    计时仍发布为「同预算」配对速度比。"""
    p = os.path.join(orig_dir, "e127_iter8.json")
    d = json.load(open(p))
    d["config"]["K2"] = 2048
    d["config"]["D"] = 64
    d["config"]["bs"] = 1
    d["torch"] = "0.0-fake"
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e126_selector_nan(orig_dir):
    """⑧ E126 selector_total_ms_median_est=NaN（109 GPT 反例之一）：
    NaN 使稳态比较/支配判定静默通过，旧脚本无 isfinite 门。"""
    p = os.path.join(orig_dir, "e126_mavg.json")
    d = json.load(open(p))
    d["cases"][0]["selector_total_ms_median_est"] = float("nan")
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e100_coordinated_nan(orig_dir):
    """⑨ E100 claim/main_ci/calibration 联立 NaN（109 GPT 反例之二）：
    abs(NaN−NaN)>tol 恒 False，旧联立门全放行且输出「+nan CI 不含 0
    显著差异」。"""
    p = os.path.join(orig_dir, "e100.json")
    d = json.load(open(p))
    nan = float("nan")
    d["claim"] = {"psi_avg": nan, "fkv_avg": nan, "diff": nan}
    d["main_ci_13tasks"] = {"mean": nan, "ci95_lo": nan, "ci95_hi": nan,
                            "contains_zero": False}
    d["calibration_check"] = {"psi_avg_expected": nan,
                              "fkv_avg_expected": nan, "passed": True}
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e125_coordinated_nan(orig_dir):
    """⑩ E125 aavg 五任务 arm/delta/AVG/CI 协同 NaN（109 GPT 反例之
    三）：全部联立门对 NaN 恒 False 放行，pairwise 把 NaN 对计入
    「不含 0」。mavg/fullkv 锚点字段不动（四臂交叉核对门需锚点一致）。"""
    p = os.path.join(orig_dir, "e125.json")
    d = json.load(open(p))
    nan = float("nan")
    for t in TASKS:
        e = d["per_task"][f"aavg/{t}"]
        e["arm"] = nan
        e["delta_vs_mavg"] = nan
        e["delta_vs_fullkv"] = nan
    d["aavg_avg"] = nan
    d["aavg_avg_delta_vs_mavg"] = nan
    d["aavg_ci95"] = [nan, nan]
    d["aavg_significant_vs_mavg"] = False
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


def tamper_e125_aavg_plus1(orig_dir):
    """⑪ 110 性质测试：aavg 五任务各 +1 且全字段协同自洽（delta 重算、
    AVG/均值差 +1、CI 用同协议 paired_bootstrap 重算、significant 重算
    True、candidates_avg5 同步）——通过全部联立门的合法数据变化。
    文字与结构化对象必须一起翻转（旧脚本文字硬编码即矛盾）。"""
    p = os.path.join(orig_dir, "e125.json")
    d = json.load(open(p))
    for t in TASKS:
        e = d["per_task"][f"aavg/{t}"]
        e["arm"] = e["arm"] + 1.0
        e["delta_vs_mavg"] = e["arm"] - e["mavg"]
        e["delta_vs_fullkv"] = e["arm"] - e["fullkv"]
    deltas = [d["per_task"][f"aavg/{t}"]["delta_vs_mavg"] for t in TASKS]
    lo, hi = paired_bootstrap(deltas)
    d["aavg_avg"] = d["aavg_avg"] + 1.0
    d["aavg_avg_delta_vs_mavg"] = d["aavg_avg_delta_vs_mavg"] + 1.0
    d["aavg_ci95"] = [lo, hi]
    d["aavg_significant_vs_mavg"] = bool(lo > 0 or hi < 0)
    if isinstance(d.get("candidates_avg5"), dict) \
            and isinstance(d["candidates_avg5"].get("aavg"), (int, float)):
        d["candidates_avg5"]["aavg"] = d["candidates_avg5"]["aavg"] + 1.0
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)


# 第四元素 = 红证锚点："old"（6e918c5，098/099/100 缺陷基线）或
# "prev"（ce6cde0，107-110 缺陷基线——109/108 反例是对「已含 098/099/
# 100 修复」版本复现的，对 6e918c5 红证语义不同）。
TAMPERS = [
    ("T1-manifest-dup-100①", tamper_manifest_dup,
     ["重复 (arm,task)", "笛卡尔积"], "old"),
    ("T2-quality-ci-decouple-100②", tamper_quality_ci_decouple,
     ["delta_vs_mavg", "来源脱钩"], "old"),
    ("T3-e126-far-method-100③", tamper_e126_far_method,
     ["完整方法配置不符"], "old"),
    ("T4-e126-stage-median-100④", tamper_e126_stage_median,
     ["精确中位数", "脱钩"], "old"),
    ("T5-e127-dup-run-100⑤", tamper_e127_dup_run, ["重复键"], "old"),
    ("T6-e127-nan-100⑥", tamper_e127_nan, ["非有限正数"], "old"),
    ("T7-e127-iter8-config-108", tamper_e127_iter8_config,
     ["配置绑定不一致", "108"], "prev"),
    ("T8-e126-selector-nan-109", tamper_e126_selector_nan,
     ["非有限值", "109"], "prev"),
    ("T9-e100-coordinated-nan-109", tamper_e100_coordinated_nan,
     ["非有限值", "109"], "prev"),
    ("T10-e125-coordinated-nan-109", tamper_e125_coordinated_nan,
     ["非有限值", "109"], "prev"),
]


def run_tamper_cases(old_script_path, prev_script_path):
    print("=" * 78)
    print("A. 十组篡改负例（红绿双向：缺陷基线脚本 rc==0 复现 / 新脚本 ABORT）")
    print("=" * 78)
    for label, fn, expect_words, anchor in TAMPERS:
        tmp, root = build_tamper_repo(fn)
        try:
            # 红证：缺陷基线脚本（固定 SHA 锚点）接受同一篡改输入。
            # T1-T6 基线=6e918c5（098/099/100 前）；T7-T10 基线=ce6cde0
            #（107-110 前，即 098/099/100 修复后——108/109 缺口在该版本
            # 上被 GPT 审计发现，红证必须对同一版本复现）。
            base = old_script_path if anchor == "old" else prev_script_path
            rc_old, _, _ = run_script_with(root, base,
                                           os.path.join(tmp, "old_out.json"))
            check(rc_old == 0, f"{label} 红证（缺陷基线接受篡改，rc==0）",
                  f"rc={rc_old}")
            # 绿证：修复后脚本拒绝
            rc_new, _, err_new = run_script(root, os.path.join(tmp,
                                                               "new_out.json"))
            hit = any(w in err_new for w in expect_words)
            check(rc_new != 0 and "[ABORT]" in err_new and hit,
                  f"{label} 绿证（新脚本 ABORT + 文案命中 {expect_words}）",
                  f"rc={rc_new} stderr={err_new[:300]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def run_script_with(repo_root, script_path, out_json):
    env = dict(os.environ)
    env["E128_REPO_ROOT"] = repo_root
    env["E128_OUT_JSON"] = out_json
    p = subprocess.run([sys.executable, script_path], env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ------------------------------------------------- 性质测试（098）
def pareto_pairs(points, minimize_dims):
    """通用 Pareto 支配：所有维 weak 不劣且至少一维严格优。
    points: [(name, {dim: val})]；minimize_dims: set；其余维 maximize。"""
    pairs = set()
    for a in points:
        for b in points:
            if a[0] == b[0]:
                continue
            dims = set(a[1])
            weak = all((a[1][d] <= b[1][d]) if d in minimize_dims
                       else (a[1][d] >= b[1][d]) for d in dims)
            strict = any((a[1][d] < b[1][d]) if d in minimize_dims
                         else (a[1][d] > b[1][d]) for d in dims)
            if weak and strict:
                pairs.add((a[0], b[0]))
    return pairs


def property_subset_test():
    print("=" * 78)
    print("B. Pareto 性质：新增最小化维后新支配集 ⊆ 旧支配集（098）")
    print("=" * 78)
    rng = random.Random(20261012)
    all_ok = True
    n_shrunk = 0
    for trial in range(200):
        n = rng.randint(2, 6)
        pts2 = {}
        for i in range(n):
            pts2[f"p{i}"] = {"q": rng.uniform(-10, 10),
                              "c": rng.uniform(-10, 10)}
        cold = {k: rng.uniform(-10, 10) for k in pts2}
        pts3 = {k: dict(v, k=cold[k]) for k, v in pts2.items()}
        d2 = pareto_pairs(list(pts2.items()), {"c"})
        d3 = pareto_pairs(list(pts3.items()), {"c", "k"})
        if not (d3 <= d2):
            all_ok = False
            print(f"  trial {trial} 违反子集性质: d2={d2} d3={d3}")
            break
        if d3 < d2:
            n_shrunk += 1
    check(all_ok, "200 组随机点集全部满足 d3 ⊆ d2")
    check(n_shrunk > 0, f"存在 2D 支配对被新增维消除的实例（实测 {n_shrunk} 组）"
                        f"——新增维只减不增支配")
    # 构造反例自检：2D 表为空时，任何第三维取值都不应产生支配对
    crafted2 = {"a": {"q": 1.0, "c": 1.0}, "b": {"q": 2.0, "c": 2.0}}
    d2 = pareto_pairs(list(crafted2.items()), {"c"})
    check(d2 == set(), "构造例：质量-成本交叉的二维点集支配表为空")
    empty_stays = True
    for kk in ([0.0, 0.0], [5.0, 5.0], [-3.0, 9.0]):
        pts3 = {"a": dict(crafted2["a"], k=kk[0]),
                "b": dict(crafted2["b"], k=kk[1])}
        if pareto_pairs(list(pts3.items()), {"c", "k"}) != set():
            empty_stays = False
    check(empty_stays, "构造例：2D 支配为空时加入第三维后仍为空"
                       "（E128 frontier_summary 旧矛盾句的数学反例）")
    # 构造正例：2D 支配对在第三维相等时保留、第三维反转时消除
    dom_case = {"x": {"q": 3.0, "c": 1.0}, "y": {"q": 1.0, "c": 3.0}}
    d2 = pareto_pairs(list(dom_case.items()), {"c"})
    check(("x", "y") in d2, "构造例：x 2D 支配 y")
    keep = {"x": dict(dom_case["x"], k=1.0), "y": dict(dom_case["y"], k=1.0)}
    elim = {"x": dict(dom_case["x"], k=9.0), "y": dict(dom_case["y"], k=0.1)}
    check(pareto_pairs(list(keep.items()), {"c", "k"}) == {("x", "y")},
          "构造例：第三维相等 → 支配保留")
    check(pareto_pairs(list(elim.items()), {"c", "k"}) == set(),
          "构造例：第三维反转 → 支配消除（不可能反向创造）")


# ------------------------------------------------- 独立复算（099）
def paired_bootstrap(deltas, B=10000, seed=42):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def verify_pairwise(new_json):
    print("=" * 78)
    print("C. 099 pairwise_ci_all：10 对全枚举独立复算 + 主 AI 锚点")
    print("=" * 78)
    pw = new_json["pairwise_ci_all"]
    pairs = pw["pairs"]
    check(len(pairs) == 10, "10 对全枚举（C(5,2)=10）", f"n={len(pairs)}")
    # 独立复算（测试自带 bootstrap 实现，直接从 e125 原始 per_task 读）
    e125 = json.load(open(os.path.join(
        REPO, "exp/trace/results/e125_screen_verdict.json")))
    pt = e125["per_task"]
    scores = {"mavg": [pt[f"aavg/{t}"]["mavg"] for t in TASKS]}
    for a in ARMS[1:]:
        scores[a] = [pt[f"{a}/{t}"]["arm"] for t in TASKS]
    import itertools
    ok_recompute = True
    for x, y in itertools.combinations(ARMS, 2):
        key = f"{x}-{y}"
        deltas = [sx - sy for sx, sy in zip(scores[x], scores[y])]
        point = sum(deltas) / 5
        lo, hi = paired_bootstrap(deltas)
        v = pairs[key]
        if abs(v["point_est"] - point) > 1e-6 or \
           abs(v["ci95"][0] - lo) > 1e-6 or abs(v["ci95"][1] - hi) > 1e-6 or \
           v["contains_zero"] != (lo <= 0 <= hi):
            ok_recompute = False
            print(f"  MISMATCH {key}: json={v} 重算={[point, lo, hi]}")
    check(ok_recompute, "10 对逐对独立复算与 JSON 一致（1e-6）")
    # 主 AI 复算锚点（advice 08819a7 回应 + 本任务书给定）
    ac = pairs["aavg-cavgsim"]
    cc = pairs["cavgsim-cclustersim"]
    check(abs(ac["ci95"][0] - 0.110) < 1e-6 and abs(ac["ci95"][1] - 0.762) < 1e-6
          and abs(ac["point_est"] - 0.392) < 1e-6,
          "锚点 aavg-cavgsim [+0.110,+0.762] 点估计 +0.392（不含 0）",
          str(ac))
    check(abs(cc["ci95"][0] + 0.948) < 1e-6 and abs(cc["ci95"][1] + 0.178) < 1e-6
          and abs(cc["point_est"] + 0.518) < 1e-6,
          "锚点 cavgsim-cclustersim [-0.948,-0.178] 点估计 -0.518（不含 0）",
          str(cc))
    notc0 = sorted(k for k, v in pairs.items() if not v["contains_zero"])
    check(notc0 == ["aavg-cavgsim", "cavgsim-cclustersim", "mavg-mminmax"],
          "不含 0 的恰好 3 对（mminmax 仅 vs mavg 显著；vs aavg/cavgsim/"
          "cclustersim 三对 CI 均含 0——按实测，不做全称外推）", str(notc0))
    check(pw["multiple_comparison_correction"].startswith("none")
          and "多重比较校正" in new_json["meta"].get(
              "multiple_comparison_note", ""),
          "meta + pairwise 标注未做多重比较校正")


# ------------------------------------------------- 措辞与对象分离（098/099）
def verify_wording(new_json):
    print("=" * 78)
    print("D. 098/099 措辞门禁（旧矛盾句不得复活 + 对象分离 + 纪律措辞）")
    print("=" * 78)
    raw = json.dumps(new_json, ensure_ascii=False)
    check("任何 decode 长度总成本更高的点估计支配关系成立" not in raw,
          "旧矛盾句①（总成本更高的点估计支配关系成立）不复存在")
    check("总成本维被零建簇臂" not in raw,
          "旧矛盾句②（簇臂在总成本维被零建簇臂点估计支配）不复存在")
    check("臂间质量差 bootstrap CI 均含 0" not in raw and
          "臂间质量差 CI 均含 0" not in raw,
          "旧全称命题（除 mminmax 外臂间 CI 均含 0）不复存在")
    check("除 mminmax" not in raw or "三候选" in raw,
          "若提及 mminmax 例外必须限定范围措辞")

    par = new_json["quality_steady_cost_pareto"]
    dom = par["dominance"]
    all_empty = all(not dom[a]["dominates_point_estimate"]
                    and not dom[a]["dominated_by_point_estimate"]
                    for a in ARMS)
    check(all_empty, "当前数据 dominance 五臂两列表全空（与 GPT 审计一致）")
    fs = par["frontier_summary"]
    if all_empty:
        check("无任何点估计支配对" in fs,
              "表空 → 摘要写「无任何点估计支配对」（机器生成一致性）")
        check("支配关系成立" not in fs and "被零建簇臂" not in fs,
              "表空 → 摘要不含任何支配主张")
    else:
        for a, x in dom.items():
            for item in x["dominates_point_estimate"]:
                check(f"{a} 支配 {item['arm']}" in fs,
                      "表非空 → 摘要逐一列出支配对")
    sn = par["significance_note"]
    check("三候选" in sn and "mminmax 显著更差" in sn and "多重" in sn,
          "significance_note 范围严格化 + 全枚举引用", sn[:200])

    co = new_json["cost_only_total_order_at_horizon"]
    for field in ("semantics", "decode_horizon", "calibers", "data_scope",
                  "point_estimate_nature", "statements"):
        check(field in co, f"cost_only 对象含 {field}")
    stmt_raw = json.dumps(co["statements"], ensure_ascii=False)
    check("Pareto 支配" not in stmt_raw and "支配" not in stmt_raw,
          "cost_only statements 禁用支配术语（098-②）",
          stmt_raw[:300])
    check("structured/singleton" in co["data_scope"],
          "cost_only 显式声明合成数据口径")
    check("steady-only" in co["decode_horizon"] and "128 步" in
          co["decode_horizon"], "cost_only 显式 decode horizon 双口径")

    rec = new_json["recommendation"]
    reasons = {r["id"]: r for r in rec["reasons"]}
    r1 = reasons["R1-quality-unproven"]["statement"]
    check("三候选" in r1 and "四候选" not in r1,
          "R1 措辞：三候选 vs mavg 含 0 与 mminmax 显著分开陈述（无内部冲突）",
          r1[:200])
    check("aavg-cavgsim" in r1, "R1 引用臂间配对事实（机器生成）")
    r6 = reasons["R6-paper-story"]["statement"]
    check("持平" not in r6 and "未证明差异" in r6 and "+0.42" in r6
          and "[-0.18" in r6,
          "R6 措辞：E100 含零 CI 写「未证明差异」不写「持平」（084 纪律）",
          r6[:200])
    check("持平" not in json.dumps(new_json["recommendation"],
                                   ensure_ascii=False),
          "recommendation 全对象无「持平」")
    check("非 Pareto 支配主张" in rec["alternative_disposition"]["cavgsim"]
          and "二维支配表为空" in rec["alternative_disposition"]["cavgsim"],
          "cavgsim 处置改为非支配措辞 + 明示支配表为空")
    check(rec["main_method_recommendation"] == "mavg（维持论文主方法）",
          "推荐结论不变（维持 mavg）")

    meta = new_json["meta"]
    check("multiple_comparison_note" in meta and "098" in
          meta["statistical_discipline"],
          "meta 纪律条目更新（多重比较注记 + 支配术语使用边界）")


# ------------------------------------------------- 100-⑥ E100 provenance
def verify_e100_prov(new_json):
    print("=" * 78)
    print("E. 100-⑥：E100 进 provenance + R6 数字机器取值")
    print("=" * 78)
    prov = new_json["provenance"]
    check("e100_bootstrap_ci" in prov, "provenance 含 e100_bootstrap_ci")
    p = prov.get("e100_bootstrap_ci", {})
    real = os.path.join(REPO, "exp/trace/results/e100_bootstrap_ci.json")
    import hashlib
    h = hashlib.sha256(open(real, "rb").read()).hexdigest()
    check(p.get("sha256") == h, "E100 sha256 与实际文件一致")
    check("claim" in json.dumps(p.get("consumed_fields", [])),
          "consumed_fields 显式列出 R6 消费的 delta/CI 字段")
    r6 = [r for r in new_json["recommendation"]["reasons"]
          if r["id"] == "R6-paper-story"][0]
    check("+0.42" in r6["statement"] and "[-0.18,+1.02]" in r6["statement"],
          "R6 数字与 e100 main_ci_13tasks 一致（+0.42 [-0.18,+1.02]）")


# ------------------------------------------------- 108 配置绑定绿证
def verify_config_binding(new_json):
    print("=" * 78)
    print("E2. 108 绿证：E127 全键配置绑定 hash 进每组配对 + provenance")
    print("=" * 78)
    prov = new_json["provenance"]["e127_stage_bench"]["config_binding"]
    h = prov.get("sha256", "")
    check(len(h) == 64 and all(c in "0123456789abcdef" for c in h),
          "config canonical sha256 为 64 位 hex", h)
    paired = new_json["e127_paired_baseline_vs_iter8"]
    check(all(paired[k].get("config_binding_sha256") == h
              for k in paired),
          "8 组配对条目各携带 config_binding_sha256 且与 provenance 一致")
    check(new_json["e127_headline_64k_normal"].get("config_binding_sha256")
          == h, "headline 条目携带同一 config_binding_sha256")
    canon = prov["canonical"]
    check(canon["config"] == {"alpha": 0.25, "beta": 0.125, "gamma": 0.625,
                              "K1": 128, "K2": 1024, "bs": 64, "D": 128,
                              "H": 32, "Hkv": 8}
          and canon["gpu"] == "NVIDIA H20-3e"
          and canon["torch"] == "2.8.0+cu128",
          "canonical 与四份真实归档一致（K1/K2/bs/D/H/Hkv/αβγ/gpu/torch）",
          json.dumps(canon, ensure_ascii=False)[:200])
    check(len(canon["runs_identity"]) == 8,
          "runs_identity 为 8 组 (S0,layer_kind,steps,layer_idx)")
    check(prov.get("baseline_sha256") == h and prov.get("iter8_sha256") == h,
          "baseline/iter8 canonical hash 一致（绑定通过的落盘形式）")
    check("unrecorded_fields_declaration" in prov
          and "dtype" in prov["unrecorded_fields_declaration"],
          "未记录字段（dtype/CUDA/Triton/kernel 开关）如实声明、不作未"
          "记录字段的同配置事实断言")


# ------------------------------------------------- 107 浮点容差
def verify_float_tolerance(new_json):
    print("=" * 78)
    print("F0. 107 浮点字段容差比较（红绿 + 判别力）")
    print("=" * 78)
    # 源码级红锚点：OLD2_SHA 套件的重算比较是逐位 dict ==
    prev_suite_src = git_show_at(
        OLD2_SHA, "exp/trace/test_e128_fix_098_100.py")
    check('old["quality_e125"][a] == new_json["quality_e125"][a]'
          in prev_suite_src,
          "红锚点源码复现：OLD2_SHA 套件 quality_e125 为逐位 == 比较")
    orig = new_json["quality_e125"]
    drift14 = drift_floats(orig, 5e-14)
    check(drift14 != orig,
          "机制红证：±5e-14 漂移下逐位 dict 比较判红（GPT EPYC 环境 "
          "74 PASS/1 FAIL 的失败机制——本环境生成侧逐位一致，漂移由"
          "合成注入模拟跨环境 Neumaier 求和差异）")
    check(deep_close(orig, drift14),
          "绿证：±5e-14 漂移通过容差门（FLOAT_ABS_TOL=1e-9）")
    drift14b = drift_floats(orig, 1e-13)
    check(deep_close(orig, drift14b),
          "绿证：±1e-13 漂移（GPT 实测漂移量级上界）亦通过容差门")
    drift6 = drift_floats(orig, 1e-6)
    check(not deep_close(orig, drift6),
          "红证：±1e-6 浮点差必须被容差门拒绝（门不空转——1e-6 已是"
          "AVG5 第 6 位有效数字，非环境漂移量级）")
    # 判别力：容差不得吞掉离散/身份/结构差异
    tampered = json.loads(json.dumps(orig))
    tampered["aavg"]["avg5"] = 49.124
    check(not deep_close(orig, tampered),
          "判别力：浮点真差异（+1 分）仍拒绝")
    tampered = json.loads(json.dumps(orig))
    tampered["aavg"]["significant_vs_mavg"] = \
        not tampered["aavg"]["significant_vs_mavg"]
    check(not deep_close(orig, tampered),
          "判别力：bool 显著性翻转仍拒绝（exact）")
    tampered = json.loads(json.dumps(orig))
    tampered["verdict_e125"] = "TAMPERED"
    check(not deep_close(orig, tampered),
          "判别力：字符串/判决字段篡改仍拒绝（exact）")
    tampered = json.loads(json.dumps(orig))
    tampered["aavg"]["ci95_vs_mavg"] = tampered["aavg"]["ci95_vs_mavg"][:-1]
    check(not deep_close(orig, tampered),
          "判别力：列表长度变化仍拒绝（exact）")
    tampered = json.loads(json.dumps(orig))
    tampered["aavg"]["NEW_KEY"] = 1.0
    check(not deep_close(orig, tampered),
          "判别力：键集新增仍拒绝（exact）")
    # 逐位漂移例（GPT 审计原文形态）：48.124 vs 48.12400000000001
    check(deep_close(48.124, 48.12400000000001)
          and 48.124 != 48.12400000000001,
          "GPT 审计实例：48.124 vs 48.12400000000001——逐位红、容差绿")


# ------------------------------------------------- 110 叙事一致性
def verify_narrative_consistency(prev_script_path):
    print("=" * 78)
    print("H. 110 性质测试：aavg+1 协同合法输入——文字与结构化同步翻转")
    print("=" * 78)
    tmp, root = build_tamper_repo(tamper_e125_aavg_plus1)
    try:
        out_new = os.path.join(tmp, "new.json")
        rc_new, _, err_new = run_script(root, out_new)
        check(rc_new == 0,
              "aavg+1 自洽输入过新脚本全部联立门（合法数据变化 rc==0）",
              f"rc={rc_new} stderr={err_new[:300]}")
        out_prev = os.path.join(tmp, "prev.json")
        rc_prev, _, _ = run_script_with(root, prev_script_path, out_prev)
        check(rc_prev == 0, "红证：同一输入 OLD2_SHA 旧脚本亦 rc==0"
                            "（110 缺陷是文字硬编码非门禁拦截）",
              f"rc={rc_prev}")
        if rc_new != 0 or rc_prev != 0:
            return
        nw = json.load(open(out_new))
        pv = json.load(open(out_prev))
        # ---- 结构化对象翻转（两版本都重算，应一致）
        qa_n = nw["quality_e125"]["aavg"]
        exp_delta = 1.214
        check(abs(qa_n["delta_vs_mavg"] - exp_delta) < 1e-9
              and qa_n["significant_vs_mavg"] is True
              and abs(qa_n["ci95_vs_mavg"][0] - 0.75) < 1e-9
              and abs(qa_n["ci95_vs_mavg"][1] - 1.862) < 1e-9,
              "结构化：aavg Δ=+1.214 CI=[+0.75,+1.862] 显著翻转"
              "（GPT 110 反例机器结果逐位复现）",
              json.dumps(qa_n, ensure_ascii=False)[:200])
        pw_n = nw["pairwise_ci_all"]["pairs"]["mavg-aavg"]
        check(abs(pw_n["point_est"] + exp_delta) < 1e-6
              and pw_n["contains_zero"] is False,
              "结构化：pairwise mavg-aavg CI 不含 0（aavg 显著更优）",
              json.dumps(pw_n))
        # ---- 新脚本文字与结构化一致（110 修复核心断言）
        sn = nw["quality_steady_cost_pareto"]["significance_note"]
        r1 = [r for r in nw["recommendation"]["reasons"]
              if r["id"] == "R1-quality-unproven"][0]["statement"]
        triple = (f"aavg {qa_n['delta_vs_mavg']:+.2f} "
                  f"[{qa_n['ci95_vs_mavg'][0]:+.2f},"
                  f"{qa_n['ci95_vs_mavg'][1]:+.2f}]")
        check("三候选" not in sn and "两候选" in sn
              and f"aavg 显著更好（{triple}）" in sn,
              "significance_note 随结构化翻转：「三候选」→「两候选 + "
              "aavg 显著更好」，数字由结构化字段格式化生成",
              sn[:200])
        check(f"被证明优于 mavg 的候选：{triple}" in r1
              and "+0.21" not in r1,
              "R1 出现「被证明优于 mavg 的候选：aavg +1.21 [+0.75,+1.86]」"
              "，旧硬编码数字不复存在", r1[:250])
        disp_a = nw["recommendation"]["alternative_disposition"]["aavg"]
        check("已证明优于 mavg" in disp_a and "+1.21" in disp_a,
              "aavg 处置随结构化翻转（已证明优于 mavg）", disp_a[:200])
        r5 = [r for r in nw["recommendation"]["reasons"]
              if r["id"] == "R5-cross-caliber-stability"][0]["statement"]
        qc = nw["quality_e125"]["cclustersim"]
        check(f"cclustersim {qc['delta_vs_mavg']:+.2f} 反超" in r5,
              "R5 反超措辞与 cclustersim 结构化 delta 条件生成"
              "（+0.34>0 → 反超；本臂未受 aavg 篡改影响）", r5[:150])
        # ---- 红证：旧脚本文字硬编码与自身结构化矛盾
        sn_p = pv["quality_steady_cost_pareto"]["significance_note"]
        r1_p = [r for r in pv["recommendation"]["reasons"]
                if r["id"] == "R1-quality-unproven"][0]["statement"]
        qa_p = pv["quality_e125"]["aavg"]
        check("三候选（aavg/cavgsim/cclustersim）" in sn_p
              and "aavg +0.21" in r1_p,
              "红证机制复现：旧 significance_note 仍写「三候选（aavg/…）"
              "CI 均含 0」、R1 仍写 aavg +0.21（GPT 110 原文形态）",
              sn_p[:200])
        check(qa_p["significant_vs_mavg"] is True,
              "红证矛盾实体：旧脚本自身结构化对象 aavg.significant=True"
              "（重算路径无硬编码），与手写叙事直接冲突")
        pw_p = pv["pairwise_ci_all"]["pairs"]["mavg-aavg"]
        check(pw_p["contains_zero"] is False,
              "红证矛盾实体：旧 pairwise mavg-aavg CI 不含 0，叙事却称"
              "「aavg CI 均含 0」——机器表与手写文字脱钩（110 缺陷本质）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- 主数值不变 diff
def verify_invariance(new_json):
    print("=" * 78)
    print("F. 主数值不变：与修复前 JSON（固定锚点 SHA）字段级 diff（107："
          "浮点容差 + 离散 exact）")
    print("=" * 78)
    old = json.loads(git_show_old("exp/trace/results/e128_tradeoff.json"))
    same_keys = ["steady_selector_cost_e126", "cold_build_e126",
                 "amortization_horizon_vs_mavg",
                 "e2e_128step_ms_e126_internal_only",
                 "e127_paired_baseline_vs_iter8", "e127_headline_64k_normal",
                 "e127_profile", "e127_coverage",
                 "e127_applicability_per_arm",
                 "e109_old_caliber_avg5_cross_check"]

    def strip_cfg_bind(obj, per_entry):
        """剥离 108 允许新增的 config_binding_sha256 子键后再比较
        （per_entry=True：配对对象每个 run 条目一层；False：headline
        顶层一层）——允许集外任何键集/数值变化仍被 deep_close 拒绝。"""
        if per_entry:
            return {k: {kk: vv for kk, vv in v.items()
                        if kk != "config_binding_sha256"}
                    for k, v in obj.items()}
        return {k: v for k, v in obj.items()
                if k != "config_binding_sha256"}

    for k in same_keys:
        o, n = old[k], new_json[k]
        if k == "e127_paired_baseline_vs_iter8":
            o, n = strip_cfg_bind(o, True), strip_cfg_bind(n, True)
        elif k == "e127_headline_64k_normal":
            o, n = strip_cfg_bind(o, False), strip_cfg_bind(n, False)
        if k == "e127_coverage":
            # 【inplace 合并（用户 2026-10-12 指令）】三个字段按设计变更
            # （activation 机制描述 / S2_kmeans_build 文件引用 / sources
            # 锚点文件），其余子键仍 exact；变更字段做正向锚定断言
            # （缺锚点即拒），不放行任意漂移。
            _cov_changed = {"activation"}
            _o_cov = {kk: vv for kk, vv in o.items() if kk not in _cov_changed}
            _n_cov = {kk: vv for kk, vv in n.items() if kk not in _cov_changed}
            _o_cov["not_optimized"] = {
                kk: vv for kk, vv in o["not_optimized"].items()
                if kk != "S2_kmeans_build"}
            _n_cov["not_optimized"] = {
                kk: vv for kk, vv in n["not_optimized"].items()
                if kk != "S2_kmeans_build"}
            _o_cov["sources"] = [o["sources"][0], o["sources"][2]]
            _n_cov["sources"] = [n["sources"][0], n["sources"][2]]
            check(deep_close(_o_cov, _n_cov)
                  and n["activation"].startswith("【inplace 合并后】")
                  and "tli_indexer.py（inplace 合并后含 E127" in n["sources"][1]
                  and "_prepare_index_inc" in n["not_optimized"]["S2_kmeans_build"],
                  f"{k} 不变（inplace 合并三字段设计内变更正向锚定，"
                  f"其余 exact）")
            continue
        check(deep_close(o, n), f"{k} 不变（浮点容差 + 离散/结构 exact）")
    # quality_e125：五臂 AVG5/CI/per_task_scores 不变（107 容差口径）
    qok = all(deep_close(old["quality_e125"][a], new_json["quality_e125"][a])
              for a in ARMS)
    check(qok, "quality_e125 五臂条目（AVG5/CI/Δ/逐任务分）不变"
              "（float 容差、离散/结构 exact）")
    check(deep_close(old["quality_e125"]["fullkv_anchor_avg"],
                    new_json["quality_e125"]["fullkv_anchor_avg"])
          and old["quality_e125"]["verdict_e125"] ==
          new_json["quality_e125"]["verdict_e125"],
          "quality_e125 锚点字段不变（verdict 字符串 exact）")
    # pairwise / cost_only / 分离对象：099/098 修复新增，与提交 JSON 逐位一致
    cur = json.load(open(os.path.join(
        REPO, "exp/trace/results/e128_tradeoff.json")))
    check(deep_close(cur["pairwise_ci_all"], new_json["pairwise_ci_all"])
          and deep_close(cur["quality_steady_cost_pareto"],
                         new_json["quality_steady_cost_pareto"])
          and deep_close(cur["cost_only_total_order_at_horizon"],
                         new_json["cost_only_total_order_at_horizon"]),
          "pairwise_ci_all / quality_steady_cost_pareto / cost_only 三对象"
          "与提交 JSON 一致")
    # 顶层键变化恰好 = 允许集
    old_top = set(old)
    new_top = set(new_json)
    check(old_top - new_top == {"pareto"},
          "删除的顶层键恰为 pareto（拆分为两个新对象）", str(old_top - new_top))
    check(new_top - old_top == {"pairwise_ci_all",
                                "quality_steady_cost_pareto",
                                "cost_only_total_order_at_horizon"},
          "新增顶层键恰为 pairwise_ci_all + 两个分离对象",
          str(new_top - old_top))
    # meta 变化恰好 = 允许集（环境字段 + 措辞 + 新增注记 + 110 叙事生成声明）
    meta_diff = {k for k in set(old["meta"]) | set(new_json["meta"])
                 if old["meta"].get(k) != new_json["meta"].get(k)}
    allowed_meta = {"probe", "statistical_discipline", "base_commit",
                    "worktree_branch", "multiple_comparison_note",
                    "fix_provenance_gpt_audit", "narrative_generation"}
    check(meta_diff <= allowed_meta,
          "meta 变化 ⊆ 允许集（环境/措辞/新增注记/叙事生成声明）",
          str(meta_diff))
    # provenance 变化恰好 = 允许集（E100 新增 + E125 consumed_fields 措辞
    # + 108 e127 config_binding）
    prov_diff = {k for k in set(old["provenance"]) | set(new_json["provenance"])
                 if old["provenance"].get(k) != new_json["provenance"].get(k)}
    check(prov_diff == {"e100_bootstrap_ci", "e125_screen_verdict",
                        "e127_stage_bench"},
          "provenance 变化恰为 E100 新增 + E125 consumed_fields 措辞 + "
          "E127 config_binding（108）",
          str(prov_diff))


# ------------------------------------------------- 主流程
def main():
    if not os.path.exists(SCRIPT):
        sys.exit(f"[ABORT] 被测脚本不存在 {SCRIPT}")
    tmpdir = tempfile.mkdtemp(prefix="e128_suite_")
    # 修复前脚本固定锚点提取（098/099/100 基线=OLD_SHA；107-110 缺陷
    # 基线=OLD2_SHA——T7-T10 红证与 110 性质测试的旧脚本）
    old_script = os.path.join(tmpdir, "analyze_e128_tradeoff_old.py")
    with open(old_script, "w") as f:
        f.write(git_show_old("exp/trace/analyze_e128_tradeoff.py"))
    prev_script = os.path.join(tmpdir, "analyze_e128_tradeoff_prev.py")
    with open(prev_script, "w") as f:
        f.write(git_show_at(OLD2_SHA, "exp/trace/analyze_e128_tradeoff.py"))

    # ---- 绿证：真实仓跑通（python 与 -O 各一），产物用于后续断言
    print("=" * 78)
    print("0. 绿证基线：真实输入跑通（python / python -O）")
    print("=" * 78)
    out1 = os.path.join(tmpdir, "green1.json")
    rc, so, se = run_script(REPO, out1)
    check(rc == 0, "绿证 python rc==0", se[:300])
    out1o = os.path.join(tmpdir, "green1_O.json")
    rco, _, seo = run_script(REPO, out1o, opt_o=True)
    check(rco == 0, "绿证 python -O rc==0", seo[:300])
    if rc != 0 or rco != 0:
        summarize()
        return
    new_json = json.load(open(out1))

    run_tamper_cases(old_script, prev_script)
    property_subset_test()
    verify_pairwise(new_json)
    verify_wording(new_json)
    verify_e100_prov(new_json)
    verify_config_binding(new_json)
    verify_float_tolerance(new_json)
    verify_invariance(new_json)
    verify_narrative_consistency(prev_script)

    # ---- 确定性：三跑逐位一致（python）+ -O 与 python 产物主数值一致
    print("=" * 78)
    print("G. 确定性：三跑输出逐位一致 + ±O 一致")
    print("=" * 78)
    runs = [open(out1, "rb").read()]
    for i in (2, 3):
        outn = os.path.join(tmpdir, f"green{i}.json")
        rc, _, se = run_script(REPO, outn)
        check(rc == 0, f"第 {i} 跑 rc==0")
        runs.append(open(outn, "rb").read())
    check(runs[0] == runs[1] == runs[2], "三跑输出逐位一致（bootstrap seed 固定）")
    a = json.load(open(out1))
    b = json.load(open(out1o))
    drop = {"meta"}
    check({k: v for k, v in a.items() if k not in drop} ==
          {k: v for k, v in b.items() if k not in drop},
          "python 与 -O 产物除 meta 环境字段外逐位一致")

    summarize()


def summarize():
    n_pass = sum(1 for ok, _, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print("=" * 78)
    print(f"[套件] {n_pass} PASS / {n_fail} FAIL / 共 {len(RESULTS)}")
    if n_fail:
        print("失败项：")
        for ok, label, detail in RESULTS:
            if not ok:
                print(f"  - {label}: {detail[:400]}")
    print("红绿套件结论：" + ("全绿" if not n_fail else "存在失败，禁止合入"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
