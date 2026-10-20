#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E155d T2 打分判决器：三维网格（nb×nt×nl）AVG5 排序 + top-3 配对 CI。

输入：~/tmp_e155d_collect/<host>/pred_<tag>/（e155d_grid_collect.sh 快照，
幂等可重收）。判决协议（TASK.md T2 预登记）：
  - 完整性门：单机 5 任务齐（hotpotqa/musique/gov_report/qasper 各 ≥200、
    repobench ≥500 行）——**只收单主机完整副本，不跨机拼 partial**
    （transformers 口径/运行 provenance 单一性；partial-partial 互补格
    列入 needs_redispatch 由人工重派，不静默混合）。
  - 去重：多主机同格完整副本 → 逐文件 sha256 全等才放行（树 ecfbd4f/
    同种子 T=10191900/同数据下应确定；不等则记录 conflict 如实披露）。
  - 打分：benchmark.LongBench.eval（runcap strict 消费链）→ AVG5 =
    5 任务分均值。排序按 AVG5 降序。
  - top-3 配对 CI（预登记纪律「不按点估计拍板」）：分层配对 bootstrap
    （任务内按 _id 对齐逐例分差、任务内重采样、任务均值差的均值），
    B=10000、2.5/97.5 分位。
  - 预登记故事分支：nt 甜区赢 / ntoff 赢 / 原位赢（nb8nt320nl8192，
    即 β.0625 锚 (8,320,8192)）/ nl 单调。
  - 轴外目录（legacy nt160 / pred_legacy_ratio_*）显式排除不计数。

用法：python3 exp/trace/analyze_e155d_grid.py [--bootstrap 10000]
输出：exp/trace/results/e155d_grid_verdict.json
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

COLLECT_ROOT = os.path.expanduser("~/tmp_e155d_collect")
STAGING = os.path.expanduser("~/tmp_e155d_collect_cells")
OUT = os.path.join(REPO, "exp", "trace", "results", "e155d_grid_verdict.json")

HOSTS = ["local", "pc02", "pc8100", "pc31", "pc32", "dot187"]  # 去重优先序（确定性）

AVG5_TASKS = ["hotpotqa", "musique", "gov_report", "qasper", "repobench"]
MIN_ROWS = {"repobench": 500}  # 其余默认 200

NB_AXIS = {4, 8, 16, 32, 64}
NT_AXIS = {128, 256, 320, 448, 640, "off"}
NL_AXIS = {1024, 2048, 4096, 8192}

GEO_RE = re.compile(r"^nb(\d+)nt(\d+|off)nl(\d+)$")


def parse_tag(name):
    """pred_<tag> → (kind, nb, nt, nl)；轴外返回 None。"""
    tag = name[len("pred_"):]
    if tag == "ctrl_singlepool":
        return ("ctrl", None, None, None)
    m = GEO_RE.match(tag)
    if not m:
        return None
    nb, nl = int(m.group(1)), int(m.group(3))
    nt = m.group(2)
    if nt != "off":
        nt = int(nt)   # 先转 int 再查轴（字符串查 int 集合恒 False 的坑）
    if nb not in NB_AXIS or nt not in NT_AXIS or nl not in NL_AXIS:
        return None  # legacy 轴外（nt160 等）
    # 构造期约束复核（grid 派单已强制；此处防御性复核）
    if nb * 64 > nl or (nt != "off" and nt > nb * 64):
        return None
    return ("geo", nb, nt, nl)


def task_files(cell_dir):
    """cell 目录 → {task: (jsonl_path, rows)}（每任务恰一文件；多文件记歧义）。"""
    out, amb = {}, []
    for fn in sorted(os.listdir(cell_dir)):
        if not fn.endswith(".jsonl"):
            continue
        task = fn.split("-")[0]
        if task in AVG5_TASKS:
            if task in out:
                amb.append(task)
            with open(os.path.join(cell_dir, fn), "rb") as f:
                rows = sum(1 for _ in f)
            out[task] = (os.path.join(cell_dir, fn), rows)
    return out, amb


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_cells():
    """跨主机合并：单主机完整副本优先 + sha 去重。"""
    per_cell = defaultdict(dict)   # tag -> {host: {task: (path, rows)}}
    excluded = []
    for host in HOSTS:
        root = os.path.join(COLLECT_ROOT, host)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.startswith("pred_"):
                continue
            parsed = parse_tag(name)
            if parsed is None:
                excluded.append(f"{host}/{name}")
                continue
            tf, amb = task_files(os.path.join(root, name))
            if amb:
                excluded.append(f"{host}/{name}:ambiguous_tasks={amb}")
            per_cell[name][host] = tf
    complete, incomplete, needs_redispatch = {}, {}, []
    for name, hosts in per_cell.items():
        full_hosts = []
        for host, tf in hosts.items():
            missing = [t for t in AVG5_TASKS
                       if t not in tf or tf[t][1] < MIN_ROWS.get(t, 200)]
            if not missing:
                full_hosts.append(host)
        if full_hosts:
            # 去重：逐文件 sha 全等才放行多副本
            base, dups, conflicts = full_hosts[0], [], []
            base_shas = {t: file_sha(hosts[base][t][0]) for t in AVG5_TASKS}
            for h in full_hosts[1:]:
                shas = {t: file_sha(hosts[h][t][0]) for t in AVG5_TASKS}
                if shas == base_shas:
                    dups.append(h)
                else:
                    conflicts.append({t: [base_shas[t][:12], shas[t][:12]]
                                      for t in AVG5_TASKS if shas[t] != base_shas[t]})
            complete[name] = {"source_host": base, "dup_hosts": dups,
                               "sha_conflicts": conflicts,
                               "files": {t: hosts[base][t][0] for t in AVG5_TASKS}}
        else:
            # 无单主机完整副本：不跨机拼 partial（口径单一性）
            miss = {h: [t for t in AVG5_TASKS
                        if t not in tf or tf[t][1] < MIN_ROWS.get(t, 200)]
                    for h, tf in hosts.items()}
            incomplete[name] = miss
            needs_redispatch.append({"tag": name, "per_host_missing": miss})
    return complete, incomplete, needs_redispatch, excluded


def stage_and_score(complete):
    """完整格 → staging cells/<tag>/pred/ → eval.py 打分 → AVG5。"""
    if os.path.isdir(STAGING):
        shutil.rmtree(STAGING)
    results = {}
    for name, info in complete.items():
        cell_root = os.path.join(STAGING, name)
        pred_dir = os.path.join(cell_root, "pred")
        os.makedirs(pred_dir)
        for t, src in info["files"].items():
            # jsonl + runcap sidecar + tli_manifest 同名伴生文件一并拷贝
            for suffix in ("", ".runcap.json", ".tli_manifest.json"):
                s = src + suffix
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(pred_dir, os.path.basename(s)))
        r = subprocess.run(
            [sys.executable, "-m", "benchmark.LongBench.eval",
             "--output-dir", cell_root, "--expect-count", "200"],
            cwd=REPO, capture_output=True, text=True)
        rj = os.path.join(pred_dir, "result.json")
        if r.returncode != 0 or not os.path.exists(rj):
            results[name] = {"eval_error": r.stdout[-500:] + r.stderr[-500:]}
            continue
        scores = json.load(open(rj))
        task_scores, budgets = {}, []
        for t in AVG5_TASKS:
            hit = [k for k in scores if k.startswith(t + "-") or k == t]
            if not hit:
                results[name] = {"eval_error": f"task {t} missing in result.json"}
                break
            task_scores[t] = scores[hit[0]]["score"]
            budgets.append(scores[hit[0]].get("budget", -1))
        else:
            results[name] = {
                "avg5": round(sum(task_scores.values()) / len(task_scores), 4),
                "tasks": task_scores,
                "budget_avg": round(sum(budgets) / len(budgets), 2),
                "source_host": info["source_host"],
                "dup_hosts": info["dup_hosts"],
                "sha_conflicts": info["sha_conflicts"],
            }
    return results


def per_example_scores(cell_info):
    """逐例分数 {task: {id: score}}——配对 bootstrap 用（与 eval 同 scorer）。"""
    from benchmark.LongBench.eval import sample_score
    out = {}
    for t, path in cell_info["files"].items():
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                d[row.get("_id", len(d))] = sample_score(
                    t, row["pred"], row["answers"], row.get("all_classes"))
        out[t] = d
    return out


def paired_bootstrap(cell_a, cell_b, B=10000, seed=20261020):
    """分层配对 bootstrap：AVG5(A−B) 差的 95% CI（任务内 _id 对齐）。"""
    import random
    rng = random.Random(seed)
    per_task_diffs, per_task_ids = {}, {}
    for t in AVG5_TASKS:
        a, b = cell_a[t], cell_b[t]
        ids = sorted(set(a) & set(b))
        per_task_ids[t] = ids
        per_task_diffs[t] = [a[i] - b[i] for i in ids]
    stats = []
    for _ in range(B):
        acc = 0.0
        for t in AVG5_TASKS:
            diffs = per_task_diffs[t]
            n = len(diffs)
            acc += sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        stats.append(acc / len(AVG5_TASKS))
    stats.sort()
    lo, hi = stats[int(0.025 * B)], stats[int(0.975 * B) - 1]
    point = sum(sum(d) / len(d) for d in per_task_diffs.values()) / len(AVG5_TASKS)
    aligned = {t: len(per_task_ids[t]) for t in AVG5_TASKS}
    return {"point_diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": not (lo <= 0 <= hi), "aligned_n": aligned,
            "B": B}


def story_slices(ranked_geo):
    """预登记故事分支判定素材（判决解读用，不替用户拍板）。"""
    by_key = {(c["nb"], c["nt"], c["nl"]): c["avg5"] for c in ranked_geo}
    stories = {}
    # 原位赢：β.0625 锚 (8,320,8192) 是否榜首
    stories["in_place_top1"] = bool(ranked_geo) and \
        ranked_geo[0]["nb"] == 8 and ranked_geo[0]["nt"] == 320 \
        and ranked_geo[0]["nl"] == 8192
    # nt 甜区 vs ntoff（逐 (nb,nl) 配对）
    pairs = []
    for (nb, nt, nl), v in by_key.items():
        if nt == "off":
            continue
        off = by_key.get((nb, "off", nl))
        if off is not None:
            pairs.append({"nb": nb, "nl": nl, "best_nt": nt, "nt_avg": v,
                          "ntoff_avg": off, "delta": round(v - off, 4)})
    stories["nt_vs_ntoff_pairs"] = sorted(
        pairs, key=lambda p: -p["delta"])
    # nl 趋势：每 nl 档最优格
    best_per_nl = {}
    for c in ranked_geo:
        best_per_nl.setdefault(c["nl"], c)
    stories["best_per_nl"] = [
        {"nl": nl, "cell": f"nb{c['nb']}nt{c['nt']}", "avg5": c["avg5"]}
        for nl, c in sorted(best_per_nl.items())]
    # nb 趋势：每 nb 档最优格
    best_per_nb = {}
    for c in ranked_geo:
        best_per_nb.setdefault(c["nb"], c)
    stories["best_per_nb"] = [
        {"nb": nb, "cell": f"nt{c['nt']}nl{c['nl']}", "avg5": c["avg5"]}
        for nb, c in sorted(best_per_nb.items())]
    return stories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--ci-top", type=int, default=3,
                    help="前 N 名进配对 CI（预登记 N=3）")
    args = ap.parse_args()

    complete, incomplete, needs_redispatch, excluded = collect_cells()
    print(f"完整格 {len(complete)} / 在飞或不完整 {len(incomplete)} / "
          f"轴外排除 {len(excluded)}")

    results = stage_and_score(complete)
    n_err = sum(1 for v in results.values() if "eval_error" in v)
    print(f"打分完成 {len(results) - n_err}，eval 失败 {n_err}")

    ranked_geo = sorted(
        ({"tag": k, "nb": parse_tag(k)[1], "nt": parse_tag(k)[2],
          "nl": parse_tag(k)[3], **{kk: vv for kk, vv in v.items()
                                    if kk != "eval_error"}}
         for k, v in results.items()
         if "avg5" in v and parse_tag(k)[0] == "geo"),   # CTRL 不进 geo 排序
        key=lambda c: -c["avg5"])
    ctrl = {k: v for k, v in results.items() if parse_tag(k)[0] == "ctrl"}

    # top-N 配对 CI（预登记：前 3 名两两配对）
    paired = {}
    top = ranked_geo[:args.ci_top]
    if len(top) >= 2:
        pes = {}
        for c in top:
            pes[c["tag"]] = per_example_scores(complete[c["tag"]])
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                paired[f"{a['tag']} vs {b['tag']}"] = paired_bootstrap(
                    pes[a["tag"]], pes[b["tag"]], B=args.bootstrap)

    verdict = {
        "meta": {
            "generated": __import__("datetime").datetime.now().isoformat(),
            "protocol": "E155d 三维网格 T2 打分判决（TASK.md 预登记）",
            "collect_root": COLLECT_ROOT,
            "hosts": HOSTS,
            "avg5_tasks": AVG5_TASKS,
            "bootstrap_B": args.bootstrap,
            "notes": "单主机完整副本口径（不跨机拼 partial）；多副本 sha 全等去重；"
                     "排序按 AVG5 点估计、判决按 top-3 配对 CI（不按点估计拍板）；"
                     "incomplete_before_closure = 在飞格（收官后重跑自动消化）；"
                     "closure 后仍残留的 incomplete 才是真需重派",
        },
        "counts": {"complete": len(complete), "incomplete": len(incomplete),
                   "eval_errors": n_err, "excluded": excluded},
        "ranking": ranked_geo,
        "ctrl_singlepool": ctrl,
        "paired_ci": paired,
        "stories": story_slices(ranked_geo),
        "incomplete_cells": incomplete,
        "incomplete_before_closure": needs_redispatch,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"\n===== E155d 网格判决（部分快照亦可跑，收官后重跑即终版）=====")
    print(f"输出：{OUT}")
    for i, c in enumerate(ranked_geo[:10]):
        print(f"  #{i+1} {c['tag']}: AVG5={c['avg5']}")
    if needs_redispatch:
        print(f"  ◇ 在飞格 {len(needs_redispatch)}（收官后重收重跑即终版；"
              f"closure 后仍残留才是真需重派）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
