# -*- coding: utf-8 -*-
"""E125 静态海选收官判决脚本（克隆 E123 v4 审计协议：083 manifest + 085
scorer 身份 + 087 移植性 + best-file 口径隔离）。

回答的问题（S-T022/#212）：新口径（E121 修复后）下，四个 method 组合冠军
配置（aavg / cavgsim / cclustersim / mminmax，E109 海选各自最佳）谁的
AVG5 最高，能否挑战 mavg(.25,.125,.625) 冠军地位（E123 新口径锚点）。

数据来源（收割预步骤把四臂统一拷到 OUT_ROOT/pred_{arm}/）：
  - aavg / mminmax / cclustersim：.187:~/tmp_e125_screen/pred_{arm}/
    （rsync 拉回，含 gov_report redo 新时间戳文件）
  - cavgsim：BeeGFS tla/results/e125_screen/pred_cavgsim/（d85af51 修复后
    新源身份重跑，隔离 buggytail 旧产出后的干净数据）
  - 锚点 mavg / fullkv：**不重跑不重打**，逐任务分数复用
    exp/trace/results/e123_trial_verdict.json（同一 difflib 协议、同一
    5 任务集、E123 已过 scorer 身份门禁；E123_RAW_DIR 锚点指认可查）。

判决规则（不预设方向）：
  - 冠军选举按 AVG5 点估计最高者；配对差（vs mavg 锚点）bootstrap 95% CI
    如实报告——CI 含 0 写「未证明优于」，不写「持平/等价」（084 措辞纪律）。
  - 显著优于 mavg 且 AVG5 更高 → 冠军换人；否则维持 mavg。
  - election 是 5 任务小筛口径，冠军冻结后进全量（E116b 咬合）再作正式
    主张——本判决只回答「选谁进全量」。

best-file 口径（playbook §2.4）：
  - 每臂每任务多候选时取行数最大者；达期望行数的候选 >1 个 = AMBIGUOUS
    fail-closed；
  - 未达期望行数的 partial 文件移入 pred_{arm}/partial_quarantine/（只
    隔离不删除，防 eval.py 误扫），manifest 登记其行数与 sha256。

v1 by Opus，2026-10-11。python 与 -O 双跑安全（无 assert 门禁，全部
sys.exit fail-closed）。
"""
import hashlib
import json
import os
import random
import subprocess
import sys

REPO = os.environ.get(
    "E125_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..")))
PY = sys.executable
OUT_ROOT = os.environ.get("E125_RAW_DIR", "/tmp/e125_harvest")
ARMS = ["aavg", "cavgsim", "cclustersim", "mminmax"]
TASKS = ["qasper", "hotpotqa", "gov_report", "musique", "repobench"]
EXPECT = {"qasper": 200, "hotpotqa": 200, "gov_report": 200,
          "musique": 200, "repobench": 500}
B, SEED = 10000, 42
# 锚点：E123 判决（新口径 mavg 47.91 / fullkv 48.02 的逐任务源）
E123_VERDICT = os.path.join(REPO, "exp", "trace", "results",
                            "e123_trial_verdict.json")
OUT_JSON = os.environ.get(
    "E125_OUT_JSON",
    os.path.join(REPO, "exp", "trace", "results", "e125_screen_verdict.json"))

# 四臂冻结配置（派单 flags 权威登记，来源 .187 ps 快照 + e125 派单脚本）
ARM_CONFIGS = {
    "aavg": "far=avg near=avg (.125,.125,.375) kmeans=false "
            "--tli_far_method avg --tli_near_method avg",
    "cavgsim": "far=minmax near=avg (0,0,0) far_select=sim_greedy "
               "kmeans=true tli_sim=0.9（d85af51 修复后新源身份重跑）",
    "cclustersim": "far=minmax near=avg far_select=sim_greedy "
                   "near_select=sim_greedy kmeans=true tli_sim=0.9 "
                   "(.125,.25,.125)",
    "mminmax": "far=minmax near=minmax (.875,.875,.125) kmeans=false",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    if not os.path.exists(path):
        return -1
    with open(path) as f:
        return sum(1 for _ in f)


def resolve_best(arm, task):
    """best-file 口径：多候选取行数最大；双达标 AMBIGUOUS；partial 隔离。

    返回 best 文件路径；隔离动作写 manifest。
    """
    d = os.path.join(OUT_ROOT, f"pred_{arm}")
    if not os.path.isdir(d):
        sys.exit(f"[ABORT] {arm}: 臂目录不存在 {d}——先完成收割拷贝步骤")
    hits = sorted(f for f in os.listdir(d)
                  if f.startswith(task + "-") and f.endswith(".jsonl"))
    if not hits:
        return None
    rows = {f: count_lines(os.path.join(d, f)) for f in hits}
    best = max(rows, key=rows.get)
    if rows[best] != EXPECT[task]:
        _fail_best(arm, task, rows, best)
    full = [f for f, n in rows.items() if n == EXPECT[task]]
    if len(full) > 1:
        sys.exit(f"[ABORT-AMBIGUOUS] {arm}/{task}: {len(full)} 个候选均达 "
                 f"{EXPECT[task]} 行 {full}——身份歧义，人工裁决，fail closed")
    qdir = os.path.join(d, "partial_quarantine")
    for f in hits:
        if f != best:
            os.makedirs(qdir, exist_ok=True)
            os.rename(os.path.join(d, f), os.path.join(qdir, f))
            print(f"[QUARANTINE] {arm}/{task}: {f} rows={rows[f]} → "
                  f"partial_quarantine/（best-file 口径下 partial 无害，"
                  f"隔离防 eval 误扫，不删除）")
    return os.path.join(d, best)


def _fail_best(arm, task, rows, best):
    sys.exit(f"[ABORT] {arm}/{task}: best rows={rows[best]} != "
             f"{EXPECT[task]}（候选行数 {rows}）——未达期望行数，不进打分")


def run_eval(arm_dir, label):
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
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def load_e123_anchor():
    """E123 判决逐任务分数复用（mavg 锚点 + fullkv 参照）。"""
    d = json.load(open(E123_VERDICT))
    pt = d["per_task"]
    # E123 per_task 键只有 {cavg_g,cavg_off,fullkv}/{task}——mavg 锚点分数
    # 嵌在每条的 "mavg" 字段里；从 fullkv 条目取，并与其它臂条目交叉核对
    mavg, fullkv = {}, {}
    for t in TASKS:
        key = f"fullkv/{t}"
        if key not in pt:
            continue
        mavg[t] = pt[key]["mavg"]
        fullkv[t] = pt[key]["arm"]
        for other in ("cavg_g", "cavg_off"):
            k2 = f"{other}/{t}"
            if k2 in pt and pt[k2]["mavg"] != mavg[t]:
                sys.exit(f"[ABORT] E123 锚点自相矛盾：{k2}.mavg="
                         f"{pt[k2]['mavg']} != {key}.mavg={mavg[t]}")
    if len(mavg) != 5 or len(fullkv) != 5:
        sys.exit(f"[ABORT] E123 锚点逐任务分数不齐 mavg={len(mavg)} "
                 f"fullkv={len(fullkv)}——先核 {E123_VERDICT}")
    mavg_avg = sum(mavg.values()) / 5
    fullkv_avg = sum(fullkv.values()) / 5
    e123_backend = (d.get("meta") or {}).get("scorer_backend")
    migration = None
    if not e123_backend:
        # TL-E125-ANCHOR-BACKEND-001 修复（GPT 2026-10-11 审计）：verdict 缺
        # meta.scorer_backend 时绝不猜默认值——唯一放行路径是显式迁移：
        # 从 E123 raw（083 manifest 同源闭包目录）逐臂读 _meta.scorer_backend，
        # 四臂全部存在且一致才登记为锚点 backend；任一缺失/不一致 fail closed。
        raw_dir = os.path.join(
            REPO, "exp", "trace", "results", "e123_trial_raw")
        arms123 = ["mavg", "cavg_g", "cavg_off", "fullkv"]
        raw_backends = []
        for arm123 in arms123:
            rp = os.path.join(raw_dir, f"pred_{arm123}", "result.json")
            if not os.path.exists(rp):
                sys.exit(f"[ABORT] E123 锚点缺 meta.scorer_backend 且 raw "
                         f"result.json 不存在：{rp}——身份不可臆造，fail closed")
            b123 = (json.load(open(rp)).get("_meta") or {}).get("scorer_backend")
            if not b123:
                sys.exit(f"[ABORT] E123 raw pred_{arm123}/result.json 缺 "
                         f"_meta.scorer_backend——身份不可臆造，fail closed")
            raw_backends.append(b123)
        if len(set(raw_backends)) != 1:
            sys.exit(f"[ABORT] E123 raw 四臂 scorer_backend 不一致 "
                     f"{dict(zip(arms123, raw_backends))}——来源冲突，fail closed")
        e123_backend = raw_backends[0]
        migration = {
            "source": ("exp/trace/results/e123_trial_raw/pred_{mavg,cavg_g,"
                       "cavg_off,fullkv}/result.json _meta.scorer_backend"),
            "resolved_backend": e123_backend,
            "reason": ("e123_trial_verdict.json meta 缺 scorer_backend（E123 "
                       "判决生成早于 085 身份字段协议）；四臂 raw 闭包一致，"
                       "显式迁移登记，不猜默认值"),
        }
        print(f"[MIGRATE] E123 锚点 scorer backend 显式迁移：{e123_backend} "
              f"（来源 {migration['source']}）")
    return mavg, fullkv, mavg_avg, fullkv_avg, e123_backend, migration


def main():
    # 1) 完成度对账 + best-file 隔离 + manifest
    manifest = []
    for arm in ARMS:
        for t in TASKS:
            p = resolve_best(arm, t)
            if p is None:
                sys.exit(f"[ABORT] {arm}/{t}: 无候选文件")
            manifest.append({
                "arm": arm, "task": t,
                "basename": os.path.basename(p),
                "rows": count_lines(p), "expect": EXPECT[t],
                "sha256": sha256_of(p)})
    print(f"[READY] 四臂 5 任务 best-file 对账通过，manifest {len(manifest)} 格")

    # 2) 官方打分（四臂，scorer 身份门禁）
    scores, backends = {}, {}
    for arm in ARMS:
        print(f"=== 打分 {arm} ===")
        scores[arm], backends[arm] = run_eval(
            os.path.join(OUT_ROOT, f"pred_{arm}"), arm)
    if len(set(backends.values())) != 1:
        sys.exit(f"[ABORT] scorer backend mismatch（实际打分后端四臂不一致）: "
                 f"{backends}")
    backend_id = backends[ARMS[0]]
    print(f"[SCORER] 四臂实际打分后端一致：{backend_id}")

    # 3) E123 锚点（mavg/fullkv 逐任务）+ 后端身份一致性核对
    mavg, fullkv, mavg_avg, fullkv_avg, e123_backend, migration = (
        load_e123_anchor())
    # 001 修复后 e123_backend 保证非 None（缺字段已 fail-closed 或显式迁移），
    # 无条件严格比较——跨后端比较无意义
    if e123_backend != backend_id:
        sys.exit(f"[ABORT] 锚点 scorer 后端（E123: {e123_backend}）与本轮 "
                 f"（{backend_id}）不一致——跨后端比较无意义，fail closed")

    # 4) 逐任务配对差（vs mavg 锚点）+ bootstrap CI + 冠军选举
    per_task, arm_avgs = {}, {}
    verdict_payload = {
        "meta": {
            "champion_ref": "mavg(.25,.125,.625) 新口径（E123 锚点，"
                            "AVG5=%.2f）" % mavg_avg,
            "fullkv_ref": "FullKV 新口径（E123 锚点，AVG5=%.2f）" % fullkv_avg,
            "scorer_backend": backend_id,
            "bootstrap": {
                "B": B, "seed": SEED,
                "rng": "stdlib random.Random（Mersenne Twister）",
                "task_order": TASKS,
                "unit": "任务级配对差（arm−mavg 锚点）有放回重采样",
                "ci": "percentile 95%：sorted means[int(0.025*B)], "
                      "means[int(0.975*B)-1]"},
            "arm_configs": ARM_CONFIGS,
            "anchor_provenance": {
                "source": "exp/trace/results/e123_trial_verdict.json",
                "anchor_scorer_backend": e123_backend,
                "anchor_backend_migration": migration,
                "note": "mavg/fullkv 锚点逐任务分数复用 E123（同 difflib "
                        "协议同 5 任务集，E123 已过 scorer 身份门禁与 "
                        "input_manifest 闭包）"},
            "cavgsim_provenance": "d85af51 修复后新源身份重跑（BeeGFS 直写，"
                                  "旧 buggytail 产出已隔离）；其余三臂旧树 "
                                  "d993d748（C 组 10 配置逐位一致证明与修复"
                                  "树同行为）",
        },
        "input_manifest": manifest,
        "per_task": per_task,
    }
    for arm in ARMS:
        deltas = []
        for t in TASKS:
            a = scores[arm][t]
            d = a - mavg[t]
            deltas.append(d)
            per_task[f"{arm}/{t}"] = {
                "arm": a, "mavg": mavg[t], "fullkv": fullkv[t],
                "delta_vs_mavg": d, "delta_vs_fullkv": a - fullkv[t]}
        avg = sum(scores[arm][t] for t in TASKS) / 5
        arm_avgs[arm] = avg
        lo, hi = paired_bootstrap(deltas)
        sig = (lo > 0) or (hi < 0)
        verdict_payload[f"{arm}_avg"] = avg
        verdict_payload[f"{arm}_avg_delta_vs_mavg"] = avg - mavg_avg
        verdict_payload[f"{arm}_ci95"] = [lo, hi]
        verdict_payload[f"{arm}_significant_vs_mavg"] = sig
        print(f"[ARM] {arm}: AVG5={avg:.2f} delta_vs_mavg="
              f"{avg - mavg_avg:+.2f} CI95=[{lo:+.2f},{hi:+.2f}] "
              f"significant={sig}")

    # 5) 冠军选举（AVG5 点估计最高；显著性如实报告）
    candidates = {"mavg": mavg_avg, **arm_avgs}
    champ = max(candidates, key=candidates.get)
    champ_delta = candidates[champ] - mavg_avg
    if champ == "mavg":
        verdict_txt = (f"维持 mavg：四 method 臂 AVG5 均未超过 mavg 锚点"
                       f"（{mavg_avg:.2f}）")
    else:
        sig = verdict_payload[f"{champ}_significant_vs_mavg"]
        verdict_txt = (
            f"冠军换人：{champ} AVG5={candidates[champ]:.2f} > mavg "
            f"{mavg_avg:.2f}（配对差 {champ_delta:+.2f}，"
            + ("bootstrap CI 不含 0，显著）" if sig else
               "CI 含 0，未证明显著优于——按点估计选举，正式主张待全量"
               "（E116b 咬合）确认）"))
    verdict_payload["mavg_anchor_avg"] = mavg_avg
    verdict_payload["fullkv_anchor_avg"] = fullkv_avg
    verdict_payload["candidates_avg5"] = candidates
    verdict_payload["verdict"] = verdict_txt
    print(f"\n[VERDICT] {verdict_txt}")
    print(f"[CANDIDATES] {json.dumps(candidates, ensure_ascii=False)}")

    with open(OUT_JSON, "w") as f:
        json.dump(verdict_payload, f, ensure_ascii=False, indent=1)
    print(f"[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
