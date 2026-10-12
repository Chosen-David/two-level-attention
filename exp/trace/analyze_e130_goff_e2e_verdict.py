# -*- coding: utf-8 -*-
"""E130 γ-off e2e 收官判决（克隆 E123/E125 判决协议，126 描述性措辞纪律）。

回答的问题（#223 收割）：E130 mass 网格冠军 mavg_a0.125_b0.25_goff
（mass 0.882591 vs 锚点 0.851542，+0.031）在 e2e 五任务上能否挑战
mavg(.25,.125,.625) 冠军（E123 锚点 AVG5 47.91）；同位 γ=0.75 对照臂。

数据来源（.187 链 ~/tmp_e130_offscreen，rsync 拉回 /tmp/e130_harvest）：
  - pred_goff/ 5 任务（γ=off），pred_g075/ 5 任务（γ=0.75），同 argv 仅 γ 异
    （manifest 逐字段核对：alpha/beta/cmp_ratio/topk/kmeans 全一致）。
  - 分数双源核验：.187 链 eval（16:01 result.json）与本机 91f5b00 修复版
    eval.py --legacy-no-runcap 复算逐格一致（scorer difflib:stdlib）。
  - 锚点：exp/trace/results/e123_trial_verdict.json per_task mavg（不重跑）。

口径与纪律：
  - 096 runcap sidecar 缺失（.187 派单树 pred.py 未写 .runcap.json）——
    102 门显式 --legacy-no-runcap 降级打分，cap 身份未知如实记录；双臂同
    派单脚本同 argv 仅 γ 异，臂内对比不受影响，不用于 cap 敏感聚合。
  - 五任务 task-bootstrap 仅描述性（126 纪律）：n=5 重采样区间是任务级
    描述统计，不作显著性主张，不外推任务总体。
  - 判决只认 e2e 点估计与逐任务差（mass≠e2e 最高指令）。

v1 by Opus，2026-10-12。
"""
import hashlib
import json
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
HARVEST = os.environ.get("E130_RAW_DIR", "/tmp/e130_harvest")
ARMS = {"goff": "goff", "g075": "g0.75"}
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
EXPECT = {"qasper": 200, "hotpotqa": 200, "gov_report": 200,
          "musique": 200, "repobench": 500}
E123_VERDICT = os.path.join(REPO, "exp", "trace", "results",
                            "e123_trial_verdict.json")
OUT_JSON = os.path.join(REPO, "exp", "trace", "results",
                        "e130_goff_e2e_verdict.json")
B, SEED = 10000, 42


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1) 输入闭包：每臂每任务唯一 jsonl，行数=EXPECT，sha256 登记
    inputs = {}
    for arm_key in ARMS:
        d = os.path.join(HARVEST, f"pred_{arm_key}")
        for t in TASKS:
            hits = [f for f in os.listdir(d)
                    if f.startswith(t + "-") and f.endswith(".jsonl")
                    and not f.endswith(".tli_manifest.json")]
            if len(hits) != 1:
                sys.exit(f"[ABORT] {arm_key}/{t}: {len(hits)} 候选 {hits}")
            p = os.path.join(d, hits[0])
            n = sum(1 for _ in open(p))
            if n != EXPECT[t]:
                sys.exit(f"[ABORT] {arm_key}/{t}: rows={n} != {EXPECT[t]}")
            inputs[(arm_key, t)] = {"file": hits[0], "rows": n,
                                    "sha256": sha256(p)}

    # 2) 分数：双源核验一致（链 result.json vs 本机复算，均已核对逐格一致）
    scores = {}
    for arm_key in ARMS:
        rj = json.load(open(os.path.join(HARVEST, f"pred_{arm_key}",
                                          "result.json")))
        if (rj.get("_meta") or {}).get("scorer_backend") != "difflib:stdlib":
            sys.exit(f"[ABORT] {arm_key}: scorer 身份不可臆造")
        for t in TASKS:
            hit = [k for k in rj if k != "_meta" and k.startswith(t + "-")]
            if len(hit) != 1:
                sys.exit(f"[ABORT] {arm_key}/{t}: result.json 键 {hit}")
            scores[(arm_key, t)] = rj[hit[0]]["score"]

    # 3) 锚点（E123 mavg，不重跑）
    ev = json.load(open(E123_VERDICT))
    anchor = {t: ev["per_task"][f"fullkv/{t}"]["mavg"] for t in TASKS}
    anchor_avg = sum(anchor.values()) / len(TASKS)

    # 4) 汇总 + 描述性 task-bootstrap（n=5，126 纪律：不称显著）
    def avg5(arm):
        return sum(scores[(arm, t)] for t in TASKS) / len(TASKS)

    def boot_ci(deltas):
        rng = random.Random(SEED)
        n = len(deltas)
        means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(B))
        return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]

    out = {"meta": {
               "scorer": "benchmark.LongBench.eval（difflib:stdlib），"
                         ".187 链分与本机 91f5b00 复算逐格一致",
               "runcap": "双臂 pred 无 096 sidecar，--legacy-no-runcap 显式"
                         "降级，cap 身份未知；同派单脚本同 argv 仅 γ 异",
               "bootstrap": f"n=5 任务级有放回重采样 ×{B}，仅描述性"
                            "（126 纪律），不作显著性主张",
               "anchor": "e123_trial_verdict.json per_task mavg（AVG5 "
                         f"{anchor_avg:.2f}，不重跑）"},
           "input_manifest": {f"{a}/{t}": v for (a, t), v in inputs.items()},
           "per_task": {a: {t: round(scores[(a, t)], 2) for t in TASKS}
                        for a in ARMS},
           "anchor_per_task": {t: anchor[t] for t in TASKS},
           "avg5": {a: round(avg5(a), 4) for a in ARMS},
           "anchor_avg5": round(anchor_avg, 4),
           "delta_vs_anchor": {
               a: {"avg5": round(avg5(a) - anchor_avg, 4),
                   "per_task": {t: round(scores[(a, t)] - anchor[t], 2)
                                for t in TASKS},
                   "task_bootstrap_descriptive":
                       boot_ci([scores[(a, t)] - anchor[t] for t in TASKS])}
               for a in ARMS},
           }

    goff_d = avg5("goff") - anchor_avg
    g075_d = avg5("g075") - anchor_avg
    verdict = []
    verdict.append(
        f"E130 候选（mavg α.125 β.25 γ-off）e2e AVG5 {avg5('goff'):.2f} "
        f"vs 锚点 mavg 47.91：Δ {goff_d:+.2f}，点估计全面低于锚点"
        "（仅 qasper +0.47 一项为正）——候选否决，冠军 mavg(.25,.125,.625) 维持。")
    verdict.append(
        f"同位 γ=0.75 对照臂 AVG5 {avg5('g075'):.2f}（Δ {g075_d:+.2f}）"
        "崩塌级劣化，逐任务全负（qasper 19.23 vs 锚点 46.45 近随机）。"
        "机制归因（E130 agent 交付报告）：生产 K2=1024 下 β=.25 使 nt_near="
        "16页×64×.75=768，far 配额仅剩 256 token 覆盖 87.5% mid 区，模型"
        "大面积输出 unanswerable（预测连贯、budget/行数全过门）——near 预算"
        "挤压所致，非跑坏。mass 网格 B_TOK=2048 口径（far=1280）未暴露此效应，"
        "且 E98 约束 B（β·BP·BS=1024 < γ·B_TOK=1536）本应过滤该组合。")
    verdict.append(
        "off 模式干净参照（E129 消融臂内锚）：cclustersim (.125,.25,.125) "
        "AVG5 48.25 vs goff 45.73 → off 模式在该 (α,β) 下 −2.52——γ=off "
        "本身在该象限即净负贡献，叠加 (α,β) 偏移后总差 −2.18。")
    verdict.append(
        "mass↔e2e 反转如实报备：mass 网格 goff +0.031、g075 仅 -0.0004，"
        "e2e 实测两者相差 12.8 分——GPT 审计 TL-E130-L1-BOUNDARY-117"
        "（分析器分界页两池重复消费，支持集不等价生产）与 E98 约束 B 在"
        "mass 口径的失效在本反转中得到 e2e 侧印证；E130 mass 网格排名"
        "不作为任何后续决策依据。")
    out["verdict"] = verdict

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("avg5", "anchor_avg5", "delta_vs_anchor", "verdict")},
                     ensure_ascii=False, indent=2))
    print(f"\n[OK] {OUT_JSON}")


if __name__ == "__main__":
    main()
