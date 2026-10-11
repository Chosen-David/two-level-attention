# -*- coding: utf-8 -*-
"""E128 修复 098/099/100 红绿套件（worktree e128-fix-098-100，2026-10-12）。

覆盖 GPT 审计三项（sglang agent_doc/advice/2026-10-12_0527_twolevel_bug_
audit_08819a7.md，主 AI 逐条 confirmed）：

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
    fast = os.path.join(root, "sparse_attn", "indexer",
                        "tli_indexer_fast.py")
    os.makedirs(os.path.dirname(fast), exist_ok=True)
    shutil.copy2(os.path.join(REPO, "sparse_attn", "indexer",
                              "tli_indexer_fast.py"), fast)
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


TAMPERS = [
    ("T1-manifest-dup-100①", tamper_manifest_dup, ["重复 (arm,task)",
                                                    "笛卡尔积"]),
    ("T2-quality-ci-decouple-100②", tamper_quality_ci_decouple,
     ["delta_vs_mavg", "来源脱钩"]),
    ("T3-e126-far-method-100③", tamper_e126_far_method,
     ["完整方法配置不符"]),
    ("T4-e126-stage-median-100④", tamper_e126_stage_median,
     ["精确中位数", "脱钩"]),
    ("T5-e127-dup-run-100⑤", tamper_e127_dup_run, ["重复键"]),
    ("T6-e127-nan-100⑥", tamper_e127_nan, ["非有限正数"]),
]


def run_tamper_cases(old_script_path):
    print("=" * 78)
    print("A. 六组篡改负例（红绿双向：旧脚本 rc==0 复现缺陷 / 新脚本 ABORT）")
    print("=" * 78)
    for label, fn, expect_words in TAMPERS:
        tmp, root = build_tamper_repo(fn)
        try:
            # 红证：修复前脚本（OLD_SHA 固定锚点）接受同一篡改输入
            rc_old, _, _ = run_script_with(root, old_script_path,
                                           os.path.join(tmp, "old_out.json"))
            check(rc_old == 0, f"{label} 红证（旧脚本接受篡改，rc==0）",
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


# ------------------------------------------------- 主数值不变 diff
def verify_invariance(new_json):
    print("=" * 78)
    print("F. 主数值不变：与修复前 JSON（固定锚点 SHA）逐字段 diff")
    print("=" * 78)
    old = json.loads(git_show_old("exp/trace/results/e128_tradeoff.json"))
    same_keys = ["steady_selector_cost_e126", "cold_build_e126",
                 "amortization_horizon_vs_mavg",
                 "e2e_128step_ms_e126_internal_only",
                 "e127_paired_baseline_vs_iter8", "e127_headline_64k_normal",
                 "e127_profile", "e127_coverage",
                 "e127_applicability_per_arm",
                 "e109_old_caliber_avg5_cross_check"]
    for k in same_keys:
        check(old[k] == new_json[k], f"{k} 逐位不变")
    # quality_e125：五臂 AVG5/CI/per_task_scores 逐位不变
    qok = all(old["quality_e125"][a] == new_json["quality_e125"][a]
              for a in ARMS)
    check(qok, "quality_e125 五臂条目（AVG5/CI/Δ/逐任务分）逐位不变")
    check(old["quality_e125"]["fullkv_anchor_avg"] ==
          new_json["quality_e125"]["fullkv_anchor_avg"] and
          old["quality_e125"]["verdict_e125"] ==
          new_json["quality_e125"]["verdict_e125"],
          "quality_e125 锚点字段逐位不变")
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
    # meta 变化恰好 = 允许集（环境字段 + 措辞 + 新增注记）
    meta_diff = {k for k in set(old["meta"]) | set(new_json["meta"])
                 if old["meta"].get(k) != new_json["meta"].get(k)}
    allowed_meta = {"probe", "statistical_discipline", "base_commit",
                    "worktree_branch", "multiple_comparison_note",
                    "fix_provenance_gpt_audit"}
    check(meta_diff <= allowed_meta,
          "meta 变化 ⊆ 允许集（环境/措辞/新增注记）", str(meta_diff))
    # provenance 变化恰好 = 允许集（E100 新增 + E125 consumed_fields 措辞）
    prov_diff = {k for k in set(old["provenance"]) | set(new_json["provenance"])
                 if old["provenance"].get(k) != new_json["provenance"].get(k)}
    check(prov_diff == {"e100_bootstrap_ci", "e125_screen_verdict"},
          "provenance 变化恰为 E100 新增 + E125 consumed_fields 措辞",
          str(prov_diff))


# ------------------------------------------------- 主流程
def main():
    if not os.path.exists(SCRIPT):
        sys.exit(f"[ABORT] 被测脚本不存在 {SCRIPT}")
    tmpdir = tempfile.mkdtemp(prefix="e128_suite_")
    # 修复前脚本固定锚点提取
    old_script = os.path.join(tmpdir, "analyze_e128_tradeoff_old.py")
    with open(old_script, "w") as f:
        f.write(git_show_old("exp/trace/analyze_e128_tradeoff.py"))

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

    run_tamper_cases(old_script)
    property_subset_test()
    verify_pairwise(new_json)
    verify_wording(new_json)
    verify_e100_prov(new_json)
    verify_invariance(new_json)

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
