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
    B=10000、2.5/97.5 分位；点差/区间以 score points 输出，与 AVG5
    的 0–100 分制一致（逐例原始分数仍为 0–1）。
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
import stat
import subprocess
import sys
import uuid
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
    # A warning alone must not admit whichever filename sorts last.
    for task in set(amb):
        out.pop(task, None)
    return out, amb


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_treatment_hashes(name, files):
    """Bind the grid label to the frozen producer manifest and filename hash."""
    parsed = parse_tag(name)
    if parsed is None:
        raise ValueError('invalid grid label')
    _, nb, nt, nl = parsed
    expected = dict(tli_near_pages=nb, tli_near_tokens=nt, tli_near_len=nl,
                    tia_block_size=64)
    hashes = {}
    for task, path in files.items():
        with open(path + '.tli_manifest.json', 'rb') as f:
            raw = f.read()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError(f'{task}: treatment manifest must be an object')
        if (not os.path.basename(path).startswith(task + '-tli_')
                or manifest.get('method', 'tli') != 'tli'):
            raise ValueError(f'{task}: grid requires a TLI treatment')
        for key, value in expected.items():
            if key not in manifest or type(manifest[key]) is not type(value) or manifest[key] != value:
                raise ValueError(f'{task}: grid label disagrees with treatment {key}')
        if parsed[0] == 'ctrl' and any(
                type(manifest.get(key)) not in (int, float) or manifest[key] != 0
                for key in ('tli_alpha', 'tli_beta')):
            raise ValueError(f'{task}: single-pool control requires alpha=beta=0')
        canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:10]
        named = re.findall(r'_h([0-9a-f]{10})-', os.path.basename(path))
        if named != [digest]:
            raise ValueError(f'{task}: treatment manifest disagrees with filename hash')
        hashes[task] = hashlib.sha256(raw).hexdigest()
    return hashes


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
                    conflicts.append({"host": h, "task_sha256": {
                        t: [base_shas[t], shas[t]]
                        for t in AVG5_TASKS if shas[t] != base_shas[t]}})
            if conflicts:
                # Fail closed: host priority is only for identical replicas.
                miss = {h: [t for t in AVG5_TASKS
                            if t not in tf or tf[t][1] < MIN_ROWS.get(t, 200)]
                        for h, tf in hosts.items()}
                incomplete[name] = miss
                needs_redispatch.append({
                    "tag": name, "reason": "sha_conflict",
                    "source_host": base, "sha_conflicts": conflicts,
                    "per_host_missing": miss})
                continue
            complete[name] = {"source_host": base, "dup_hosts": dups,
                               "sha_conflicts": conflicts,
                               "file_sha256": base_shas,
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
        staged_files = {}
        for t, src in info["files"].items():
            # jsonl + runcap sidecar + tli_manifest 同名伴生文件一并拷贝
            for suffix in ("", ".runcap.json", ".tli_manifest.json"):
                s = src + suffix
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(pred_dir, os.path.basename(s)))
            staged_files[t] = os.path.join(pred_dir, os.path.basename(src))
        # Bind admission, scoring and CI to the same prediction bytes. The
        # collector may replace its source files while this analysis runs.
        expected_shas = info["file_sha256"]
        if any(not os.path.isfile(path) or file_sha(path) != expected_shas[t]
               for t, path in staged_files.items()):
            results[name] = {"eval_error": "prediction snapshot changed after admission"}
            continue
        try:
            treatment_shas = cell_treatment_hashes(name, staged_files)
        except (OSError, ValueError) as exc:
            results[name] = {"eval_error": f"treatment identity: {exc}"}
            continue
        r = subprocess.run(
            [sys.executable, "-m", "benchmark.LongBench.eval",
             "--output-dir", cell_root, "--expect-count", "200"],
            cwd=REPO, capture_output=True, text=True)
        rj = os.path.join(pred_dir, "result.json")
        if r.returncode != 0 or not os.path.exists(rj):
            results[name] = {"eval_error": r.stdout[-500:] + r.stderr[-500:]}
            continue
        if any(not os.path.isfile(path) or file_sha(path) != expected_shas[t]
               for t, path in staged_files.items()):
            results[name] = {"eval_error": "prediction snapshot changed during evaluation"}
            continue
        try:
            if cell_treatment_hashes(name, staged_files) != treatment_shas:
                raise ValueError('treatment snapshot changed during evaluation')
        except (OSError, ValueError) as exc:
            results[name] = {"eval_error": f"treatment identity: {exc}"}
            continue
        with open(rj, encoding="utf-8") as f:
            scores = json.load(f)
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
                "files": staged_files,
                "file_sha256": dict(expected_shas),
                "treatment_sha256": treatment_shas,
            }
    return results


def per_example_scores(cell_info):
    """逐例分数 {task: {id: score}}——配对 bootstrap 用（与 eval 同 scorer）。"""
    from benchmark.LongBench.eval import sample_score
    out = {}
    for t, path in cell_info["files"].items():
        d = {}
        digest = hashlib.sha256()
        # Universal line splitting with untranslated endings preserves both
        # legacy CR/CRLF inputs and the exact UTF-8 bytes used for the digest.
        with open(path, "r", encoding="utf-8", newline="") as f:
            for lineno, line in enumerate(f, 1):
                digest.update(line.encode("utf-8"))
                row = json.loads(line)
                sample_id = row.get("_id")
                if not ((isinstance(sample_id, str) and sample_id)
                        or type(sample_id) is int):
                    raise ValueError(
                        f"{t}: invalid/missing _id at line {lineno}; "
                        "paired CI requires explicit sample identities")
                if sample_id in d:
                    raise ValueError(
                        f"{t}: duplicate _id at line {lineno}; "
                        "paired CI cannot overwrite samples")
                d[sample_id] = sample_score(
                    t, row["pred"], row["answers"], row.get("all_classes"))
        expected = cell_info.get("file_sha256")
        if expected is not None and digest.hexdigest() != expected[t]:
            raise ValueError(f"{t}: prediction snapshot changed before/during CI")
        out[t] = d
    return out


def paired_bootstrap(cell_a, cell_b, B=10000, seed=20261020):
    """配对 0–1 逐例分数，返回 AVG5 分制的分差；要求相同非空 ID 集合。"""
    import random
    rng = random.Random(seed)
    per_task_diffs, per_task_ids = {}, {}
    for t in AVG5_TASKS:
        a, b = cell_a[t], cell_b[t]
        a_ids, b_ids = set(a), set(b)
        if not a_ids or a_ids != b_ids:
            raise ValueError(
                f"{t}: paired CI requires nonempty identical ID sets "
                f"(A={len(a_ids)}, B={len(b_ids)}, "
                f"A_only={len(a_ids - b_ids)}, B_only={len(b_ids - a_ids)})")
        ids = sorted(a_ids, key=lambda value: (type(value).__name__, value))
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
    # eval.scorer multiplies by 100; convert only at the reporting boundary,
    # before rounding, leaving sampling and the unrounded zero test unchanged.
    return {"point_diff": round(100 * point, 4),
            "ci95": [round(100 * lo, 4), round(100 * hi, 4)],
            "unit": "score_points",
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


def _publish_verdict(path, verdict):
    """Publish complete JSON; a failed write must preserve the prior verdict."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(path):
        raise ValueError('verdict destination must not be a symlink')
    mode = stat.S_IMODE(os.stat(path).st_mode) if os.path.exists(path) else None
    temporary = os.path.join(parent, f'.{os.path.basename(path)}.tmp-{uuid.uuid4().hex}')
    owned = False
    try:
        with open(temporary, 'x', encoding='utf-8') as stream:
            owned = True
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            json.dump(verdict, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if owned:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--ci-top", type=int, default=3,
                    help="前 N 名进配对 CI（预登记 N=3）")
    args = ap.parse_args()

    complete, incomplete, needs_redispatch, excluded = collect_cells()
    print(f"完整格 {len(complete)} / 在飞、不完整或冲突 {len(incomplete)} / "
          f"轴外或歧义记录 {len(excluded)}")

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
            pes[c["tag"]] = per_example_scores(results[c["tag"]])
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
            "score_unit": "points_0_100",
            "bootstrap_B": args.bootstrap,
            "notes": "单主机完整副本口径（不跨机拼 partial）；多副本 sha 全等去重；"
                     "排序按 AVG5 点估计、判决按 top-3 配对 CI（不按点估计拍板）；"
                     "incomplete_before_closure 含在飞、不完整和SHA冲突格；"
                     "SHA冲突须核对原始副本，歧义文件须核对来源；"
                     "未解决项不得用于排名或CI，不自动重派作业",
        },
        "counts": {"complete": len(complete), "incomplete": len(incomplete),
                   "eval_errors": n_err, "excluded": excluded},
        "ranking": ranked_geo,
        "ctrl_singlepool": ctrl,
        "paired_ci": paired,
        "evaluation_errors": {k: v['eval_error'] for k, v in results.items() if 'eval_error' in v},
        "stories": story_slices(ranked_geo),
        "incomplete_cells": incomplete,
        "incomplete_before_closure": needs_redispatch,
    }
    _publish_verdict(OUT, verdict)
    print(f"\n===== E155d 网格判决（部分快照亦可跑，收官后重跑即终版）=====")
    print(f"输出：{OUT}")
    for i, c in enumerate(ranked_geo[:10]):
        print(f"  #{i+1} {c['tag']}: AVG5={c['avg5']}")
    if needs_redispatch:
        print(f"  ◇ 待核查格 {len(needs_redispatch)}（含在飞/不完整/SHA冲突；"
              f"冲突须核对来源，重收重跑不会自动解决）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
