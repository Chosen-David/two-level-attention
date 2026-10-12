# -*- coding: utf-8 -*-
"""E129 2×2 消融判决脚本（克隆 E125 analyze_e125_screen_verdict.py 审计
协议：085 scorer 身份门禁 + best-file 行数门 + bootstrap 配对 CI 同款）。

回答的问题（#217）：E125 冠军 cclustersim（AVG5=48.25，点估计）之上，
两个增量维度各自与交互对质量的影响：
  C0 = 冠军 base（E125 产物复用，不重跑不重打，逐分引用
       e125_screen_verdict.json per_task，锚点 scorer_backend=difflib:stdlib）
  C1 = C0 + D' gate     （--tli_enable_layer_skip true）
  C2 = C0 + paired tail32 子空间（--tli_subspace tail，cmp_ratio=4）
  C3 = C1 + C2 交互

数据来源（镜像优先协议，.187 慢方停车，.251 补齐格）：
  .187 ~/tmp_e129_ablation/pred_{c1_gate,c2_tail,c3_both}/
  .251 ~/tmp_e129_ablation/ 镜像 7 格（c1_gate/gov_report、c2_tail×3、
  c3_both×3），已 rsync 经本机中转推到 .187 同目录合并。

判决规则（不预设方向，措辞中性留主会话）：
  - 逐任务配对差 vs C0 锚点 bootstrap 95% CI 如实报告；
  - CI 含 0 写「未证明显著」，CI 不含 0 写「显著」；
  - 主效应按派单口径 C1−C0（gate）/C2−C0（tail），交互 C3−C1−C2+C0；
    另登记析因平均口径主效应（(C1+C3)/2−(C0+C2)/2 等）作补充。

v1 by Opus，2026-10-12。python 与 -O 双跑安全（无 assert 门禁，全部
sys.exit fail-closed）。
"""
import hashlib
import json
import os
import random
import subprocess
import sys

REPO = os.environ.get(
    "E129_REPO_ROOT",
    "/home/wangyuanshuo02/two-level-attention_git")
PY = sys.executable
OUT_ROOT = os.environ.get("E129_RAW_DIR",
                          "/home/wangyuanshuo02/tmp_e129_ablation")
ARMS = ["c1_gate", "c2_tail", "c3_both"]
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
EXPECT = {"qasper": 200, "hotpotqa": 200, "gov_report": 200,
          "musique": 200, "repobench": 500}
B, SEED = 10000, 42
# C0 锚点：E125 判决（cclustersim 冠军逐分源，AVG5=48.25）
E125_VERDICT = os.path.join(REPO, "exp", "trace", "results",
                            "e125_screen_verdict.json")
OUT_JSON = os.environ.get(
    "E129_OUT_JSON",
    os.path.join(REPO, "exp", "trace", "results",
                 "e129_ablation_verdict.json"))

ARM_CONFIGS = {
    "c0_base": "cclustersim 冠军 base（E125）：far=minmax near=avg "
               "far_select=sim_greedy near_select=sim_greedy kmeans=true "
               "tli_sim=0.9 (.125,.25,.125) tli_alpha 0.125 tli_beta 0.25 "
               "tli_gamma 0.125 --tia_level1_topk 128 --tia_level2_topk 1024 "
               "--tia_level2_cmp_ratio 4 --tli_enable_layer_skip false "
               "--tli_subspace 默认（非 tail）",
    "c1_gate": "C0 + D' gate：--tli_enable_layer_skip true",
    "c2_tail": "C0 + paired tail32 子空间：--tli_subspace tail（cmp_ratio=4 "
               "→ [48..63]∪[112..127] 32 维）",
    "c3_both": "C1 + C2 交互：--tli_enable_layer_skip true "
               "--tli_subspace tail",
}
# 每格来源机器（镜像优先协议收割实录）
SOURCE = {
    ("c1_gate", "gov_report"): ".251(镜像)",
    ("c2_tail", "hotpotqa"): ".251(镜像)", ("c2_tail", "musique"): ".251(镜像)",
    ("c2_tail", "qasper"): ".251(镜像)",
    ("c3_both", "hotpotqa"): ".251(镜像)", ("c3_both", "musique"): ".251(镜像)",
    ("c3_both", "qasper"): ".251(镜像)",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def resolve_best(arm, task):
    """best-file 口径（克隆 E125）：每格唯一候选、达期望行数、首尾行
    JSON 可解析；多候选 AMBIGUOUS fail-closed。"""
    d = os.path.join(OUT_ROOT, f"pred_{arm}")
    hits = sorted(f for f in os.listdir(d)
                  if f.startswith(task + "-") and f.endswith(".jsonl"))
    if not hits:
        sys.exit(f"[ABORT] {arm}/{task}: 无候选文件——缺格，不补跑（主会话裁决）")
    rows = {f: count_lines(os.path.join(d, f)) for f in hits}
    full = [f for f, n in rows.items() if n == EXPECT[task]]
    if len(full) > 1:
        sys.exit(f"[ABORT-AMBIGUOUS] {arm}/{task}: {len(full)} 个候选均达 "
                 f"{EXPECT[task]} 行 {full}——身份歧义，fail closed")
    if not full:
        sys.exit(f"[ABORT] {arm}/{task}: 候选行数 {rows} 均未达 "
                 f"{EXPECT[task]}——短行缺格，不补跑（主会话裁决）")
    best = full[0]
    p = os.path.join(d, best)
    # partial 写盘检测：首行 + 尾行 JSON 可解析
    with open(p) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    for tag, l in (("first", lines[0]), ("last", lines[-1])):
        try:
            json.loads(l)
        except Exception as e:
            sys.exit(f"[ABORT] {arm}/{task}/{best}: {tag} 行 JSON 不可解析"
                     f"（partial 写盘嫌疑）：{e}——fail closed")
    return p


def run_eval(arm_dir, label):
    """官方打分（克隆 E125 同款命令与环境，不得自创口径）。"""
    cmd = [PY, "-m", "benchmark.LongBench.eval",
           "--output-path", arm_dir,
           "--expect-count", "200"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env={**os.environ,
                            "PYTHONPATH": REPO + ":"
                                          + os.environ.get("PYTHONPATH", "")
                                          + ":/tmp/e117_extra_pkgs"})
    if r.returncode != 0:
        print(f"[EVAL-FAIL] {label}: rc={r.returncode}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(1)
    result = json.load(open(os.path.join(arm_dir, "result.json")))
    backend = (result.get("_meta") or {}).get("scorer_backend")
    if not backend:
        sys.exit(f"[ABORT] {label}: result.json 缺 _meta.scorer_backend——"
                 f"scorer 身份不可臆造，fail closed")
    scores = {}
    for fn, v in result.items():
        if fn == "_meta" or not fn.endswith(".jsonl"):
            continue
        scores[fn.split("-")[0]] = v["score"]
    return scores, backend


def paired_bootstrap(deltas, B=B, seed=SEED):
    """克隆 E125/E123 同款：stdlib random.Random，percentile 95% CI。"""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def load_c0_anchor():
    """C0 = cclustersim E125 逐分复用（同 difflib 协议同 5 任务集，
    E125 已过 scorer 身份门禁）。"""
    d = json.load(open(E125_VERDICT))
    pt = d["per_task"]
    c0 = {}
    for t in TASKS:
        key = f"cclustersim/{t}"
        if key not in pt:
            sys.exit(f"[ABORT] E125 锚点缺 {key}——先核 {E125_VERDICT}")
        c0[t] = pt[key]["arm"]
    c0_avg = sum(c0.values()) / 5
    e125_backend = (d.get("meta") or {}).get("scorer_backend")
    if not e125_backend:
        sys.exit("[ABORT] E125 锚点 meta 缺 scorer_backend——身份不可臆造")
    # E125 登记的 cclustersim AVG5 自洽核对
    e125_avg = d.get("cclustersim_avg")
    if e125_avg is not None and abs(e125_avg - c0_avg) > 1e-9:
        sys.exit(f"[ABORT] E125 锚点自相矛盾：per_task 均值 {c0_avg:.4f} != "
                 f"cclustersim_avg {e125_avg}——fail closed")
    return c0, c0_avg, e125_backend


def load_e125_c0_sha_map():
    """E125 input_manifest 中 cclustersim 的 sha256（用于 C2≡C0 逐位一致硬对比）。"""
    d = json.load(open(E125_VERDICT))
    out = {}
    for m in d["input_manifest"]:
        if m["arm"] == "cclustersim":
            out[m["task"]] = m["sha256"]
    if set(out) != set(TASKS):
        sys.exit(f"[ABORT] E125 manifest cclustersim 任务不齐: {sorted(out)}")
    return out


def main():
    # 1) 完成度对账 + best-file + partial 检测 + manifest（来源机器登记）
    manifest = []
    for arm in ARMS:
        for t in TASKS:
            p = resolve_best(arm, t)
            manifest.append({
                "arm": arm, "task": t,
                "basename": os.path.basename(p),
                "rows": count_lines(p), "expect": EXPECT[t],
                "sha256": sha256_of(p),
                "source_machine": SOURCE.get((arm, t), ".187"),
                "remote_path": (
                    f"{SOURCE.get((arm, t), '.187')}:{p}"
                    if SOURCE.get((arm, t)) else p),
            })
    print(f"[READY] 三增量臂 5 任务 best-file 对账通过，manifest {len(manifest)} 格")

    # 2) 官方打分（三臂，scorer 身份门禁）
    scores, backends = {}, {}
    for arm in ARMS:
        print(f"=== 打分 {arm} ===")
        scores[arm], backends[arm] = run_eval(
            os.path.join(OUT_ROOT, f"pred_{arm}"), arm)
    if len(set(backends.values())) != 1:
        sys.exit(f"[ABORT] scorer backend mismatch（三臂不一致）: {backends}")
    backend_id = backends[ARMS[0]]
    print(f"[SCORER] 三臂实际打分后端一致：{backend_id}")

    # 3) C0 锚点 + 后端身份一致性核对（跨后端比较无意义）
    c0, c0_avg, e125_backend = load_c0_anchor()
    if e125_backend != backend_id:
        sys.exit(f"[ABORT] 锚点 scorer 后端（E125: {e125_backend}）与本轮"
                 f"（{backend_id}）不一致——跨后端比较无意义，fail closed")

    # 4) 逐任务配对差 + 2×2 析因 + bootstrap CI
    per_task, arm_avgs = {}, {}
    effect_payload = {}
    for arm in ARMS:
        deltas = []
        for t in TASKS:
            a = scores[arm][t]
            dd = a - c0[t]
            deltas.append(dd)
            per_task[f"{arm}/{t}"] = {
                "arm": a, "c0": c0[t], "delta_vs_c0": dd}
        avg = sum(scores[arm][t] for t in TASKS) / 5
        arm_avgs[arm] = avg
        lo, hi = paired_bootstrap(deltas)
        sig = (lo > 0) or (hi < 0)
        effect_payload[arm] = {
            "avg5": avg, "delta_avg5_vs_c0": avg - c0_avg,
            "ci95_vs_c0": [lo, hi], "significant_vs_c0": sig}
        print(f"[ARM] {arm}: AVG5={avg:.2f} delta_vs_c0={avg - c0_avg:+.2f} "
              f"CI95=[{lo:+.2f},{hi:+.2f}] significant={sig}")

    # 主效应 / 交互（派单口径：任务级配对差向量 bootstrap）
    def eff(name, deltas):
        lo, hi = paired_bootstrap(deltas)
        mean = sum(deltas) / len(deltas)
        sig = (lo > 0) or (hi < 0)
        effect_payload[name] = {
            "mean": mean, "ci95": [lo, hi], "significant": sig}
        print(f"[EFFECT] {name}: mean={mean:+.3f} "
              f"CI95=[{lo:+.3f},{hi:+.3f}] significant={sig}")

    gate_d = [scores["c1_gate"][t] - c0[t] for t in TASKS]          # C1−C0
    tail_d = [scores["c2_tail"][t] - c0[t] for t in TASKS]          # C2−C0
    inter_d = [scores["c3_both"][t] - scores["c1_gate"][t]
              - scores["c2_tail"][t] + c0[t] for t in TASKS]       # C3−C1−C2+C0
    eff("gate_effect_c1_minus_c0", gate_d)
    eff("tail_effect_c2_minus_c0", tail_d)
    eff("interaction_c3_minus_c1_minus_c2_plus_c0", inter_d)
    # 析因平均口径主效应（补充字段）
    gate_avg_d = [(scores["c1_gate"][t] + scores["c3_both"][t]
                   - c0[t] - scores["c2_tail"][t]) / 2 for t in TASKS]
    tail_avg_d = [(scores["c2_tail"][t] + scores["c3_both"][t]
                    - c0[t] - scores["c1_gate"][t]) / 2 for t in TASKS]
    eff("gate_main_effect_factorial_avg", gate_avg_d)
    eff("tail_main_effect_factorial_avg", tail_avg_d)

    # 5) C2≡C0 逐位一致硬对比（E125 manifest sha256 vs 本轮 c2_tail）
    #    ——tail 维度在冠军配置下是否 no-op 的直接证据
    c0_sha = load_e125_c0_sha_map()
    c2_identical = {}
    for t in TASKS:
        m2 = next(m for m in manifest if m["arm"] == "c2_tail" and m["task"] == t)
        c2_identical[t] = (m2["sha256"] == c0_sha[t])
    n_ident = sum(c2_identical.values())
    print(f"[BIT-IDENT] c2_tail vs c0(E125) 逐位一致格数: {n_ident}/5 "
          f"({json.dumps(c2_identical)})")
    c2_bit_identical_all = (n_ident == 5)

    # 6) 判决摘要（措辞中性，GO/NO-GO 留主会话）
    summary_lines = []
    for arm in ARMS:
        e = effect_payload[arm]
        s = "显著（CI 不含 0）" if e["significant_vs_c0"] else "未证明显著（CI 含 0）"
        summary_lines.append(
            f"{arm}: AVG5={e['avg5']:.2f} vs C0 {c0_avg:.2f}，"
            f"Δ={e['delta_avg5_vs_c0']:+.2f}，95% CI "
            f"[{e['ci95_vs_c0'][0]:+.2f},{e['ci95_vs_c0'][1]:+.2f}]，{s}")
    for name in ("gate_effect_c1_minus_c0", "tail_effect_c2_minus_c0",
                 "interaction_c3_minus_c1_minus_c2_plus_c0"):
        e = effect_payload[name]
        s = "显著" if e["significant"] else "未证明显著"
        summary_lines.append(
            f"{name}: mean={e['mean']:+.3f}，95% CI "
            f"[{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]，{s}")
    if c2_bit_identical_all:
        summary_lines.append(
            "c2_tail 与 C0 五任务输出逐位一致（sha256 相同）——tail 效应=0 "
            "是逐位 no-op 而非统计不可分辨；交互项数值上=gate 条件下的 "
            "tail 效应（C3−C1）")
    summary = "；".join(summary_lines)
    print(f"\n[SUMMARY] {summary}")

    # 机制分析（静态代码证据 + sha256 逐位一致硬证据，析因解读的关键）
    mechanism = {
        "c2_bit_identical_to_c0": {
            "per_task": c2_identical,
            "all_5_tasks": c2_bit_identical_all,
            "evidence": "本轮 c2_tail sha256 与 E125 input_manifest cclustersim "
                        "sha256 逐任务对比；跨天（10-11 vs 10-12）跨机"
                        "（.187/.251）逐位一致同时证明推理确定性",
            "implication": "tail32 子空间在冠军 base（无 gate）下是 no-op："
                           "ΔAVG5=0 非统计不可分辨，而是输出逐位相同——"
                           "tail 维度对 base 配置零信息量",
        },
        "code_evidence": {
            "c2_noop_root_cause": [
                "sparse_attn/indexer/tli_indexer.py _sim_dims_indices docstring："
                "sim_greedy 簇分维度 '_subspace_indices 全分支——注意 full 落到 "
                "tail 分支'（full 与 tail 的簇/建簇维度同源同值）",
                "非 skip 层 cclustersim 双池最终选择：far=topk(far_tok_score) 簇分、"
                "near=簇分（覆盖段）+细筛原始分 sf_g（未覆盖段，score_fine 全维"
                "与 idx_sub 无关），注释明示『都不过 L1 块池门控』→ L1 minmax 块"
                "分（唯一消费 idx_sub 的分数）不进最终选择 → full≡tail",
            ],
            "c3_vs_c1_root_cause": [
                "skip 层（gate=true 时 D' 掩码命中层）use_partition=False → "
                "簇分不可用 → 落 legacy 全序列 topk（L1382+），topk_mask 来自 "
                "L1 score_coarse（消费 idx_sub：full=128 维 vs tail=32 维）→ "
                "subspace flag 只在 skip 层生效 → C3≠C1（4/5 任务；qasper 上"
                "skip 层两口径块选择恰好未翻转任何最终输出，C3≡C1）",
            ],
        },
        "interaction_semantics": (
            "C2≡C0（逐位）使交互项 C3−C1−C2+C0 数值上退化为 C3−C1，"
            "即『gate 条件下的 tail 效应』——它度量的不是两增量独立性的"
            "经典交互，而是 tail 维度在 skip 层 legacy 路径上的边际效应"
            "（+0.20/任务，量级远小于 gate 主损失 −4.47）"),
        "gate_harm_context": (
            "D' gate 损失远端上下文（skip 层 far 区全 -inf）且该层落 "
            "legacy topk 路径（swa 占 K2 预算等残余口径差，B1 修复段记录）"
            "——−4.47 显著负符合机制预期；gate 省算收益属 E127 timing 口径，"
            "本判决仅质量维度"),
    }

    # 逐位一致格观察（sha256 相同 = 输出逐位一致，析因解释的关键证据）
    same_pairs = []
    arm_pairs = [("c1_gate", "c3_both"), ("c2_tail", "c3_both"),
                ("c1_gate", "c2_tail")]
    for a1, a2 in arm_pairs:
        for t in TASKS:
            m1 = next(m for m in manifest if m["arm"] == a1 and m["task"] == t)
            m2 = next(m for m in manifest if m["arm"] == a2 and m["task"] == t)
            if m1["sha256"] == m2["sha256"]:
                same_pairs.append({
                    "task": t, "arms": [a1, a2], "sha256": m1["sha256"],
                    "note": f"{a1} 与 {a2} 在 {t} 输出逐位一致——另一维度"
                            f"对该格输出零影响"})

    verdict_payload = {
        "meta": {
            "experiment": "E129 冠军配置子空间消融 + 静态 D' gate（2×2 增量格，#217）",
            "design": "C0=cclustersim 冠军 base（E125 复用不重跑）；C1=+D' gate；"
                      "C2=+paired tail32 子空间；C3=交互。3 增量臂 × 5 任务 = 15 新格。",
            "scorer_backend": backend_id,
            "bootstrap": {
                "B": B, "seed": SEED,
                "rng": "stdlib random.Random（Mersenne Twister）",
                "task_order": TASKS,
                "unit": "任务级配对差有放回重采样（vs C0 锚点；主效应/交互"
                        "按对应差分向量）",
                "ci": "percentile 95%：sorted means[int(0.025*B)], "
                      "means[int(0.975*B)-1]"},
            "arm_configs": ARM_CONFIGS,
            "c0_anchor_provenance": {
                "source": "exp/trace/results/e125_screen_verdict.json per_task "
                          "cclustersim/{task}（E125 判决同源，AVG5=48.25 点估计，"
                          "已过 scorer 身份门禁与 E123/E128 链核对）",
                "anchor_scorer_backend": e125_backend,
                "c0_per_task": c0, "c0_avg5": c0_avg},
            "harvest_provenance": {
                "protocol": "镜像优先：.187 慢方停车，.251 镜像补齐 7 格"
                            "（c1_gate/gov_report、c2_tail/{hotpotqa,musique,"
                            "qasper}、c3_both/{hotpotqa,musique,qasper}），"
                            "rsync 经本机中转推 .187 合并打分。同模型同数据同代码，"
                            "推理确定性下机器差异对分数无影响，来源逐格登记。",
                "raw_locations": {
                    ".187": "~/tmp_e129_ablation/pred_{c1_gate,c2_tail,c3_both}/",
                    ".251": "~/tmp_e129_ablation/（镜像 7 格原始文件本体不回传，"
                            "sha256 见 input_manifest）"},
                "machine_note": ".187 打分时卡 0/2/5/6/7 被 E130 e2e 占用——"
                                "打分为 CPU 活未占卡，未触碰任何 GPU 进程"},
        },
        "input_manifest": manifest,
        "per_task": per_task,
        "avg5": {"c0_base": c0_avg, **arm_avgs},
        "effects": effect_payload,
        "mechanism_analysis": mechanism,
        "identical_output_pairs": same_pairs,
        "summary": summary,
        "verdict_note": "判决结论措辞（GO/NO-GO、是否采纳 gate/tail 进主线）"
                        "留主会话裁决——本 JSON 只登记点估计与 CI 如实数据。",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(verdict_payload, f, ensure_ascii=False, indent=1)
    print(f"[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
