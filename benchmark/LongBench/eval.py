import os
import re
import json
import argparse
import hashlib
import sys
import numpy as np

from .lbv2_choice import extract_choice_official, LBV2_PARSER_VERSION
from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    SCORER_BACKEND_ID,
)
# 102（TL-E096-CAP-CONSUMER-CLOSURE-102，GPT 审计 2026-10-12_0739，P1）：
# 消费端闭包——打分前强制消费 producer 侧的 run-cap sidecar（096 修复
# 写了 generating→JSONL→complete 两阶段 manifest，但修复前 eval.py 完全
# 不读取它：generating/损坏/异 cap/结果 SHA 不匹配的 JSONL 照样打分并
# 写 result.json，且结果元数据不携带 cap 身份）。run_cap_manifest 只依赖
# 标准库，eval.py 侧引入零新依赖。
from .run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX,
    read_run_cap_manifest,
)


def _runcap_file_sha256(path):
    """结果文件字节 sha256——与 producer compute_result_binding 同口径。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runcap_file_lines(path):
    """行数——与 producer compute_result_binding 同口径（按行迭代计数）。"""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def validate_runcap_sidecar(jsonl_path, legacy_allowed):
    """102：打分前对单个 *.jsonl 强制校验 run-cap sidecar。

    返回 cap 身份 dict（strict 通过）或 None（legacy 降级放行）；
    违规一律 SystemExit（[RUNCAP-FAIL]/[CAP-SCHEMA-FAIL]），调用方不写
    result.json。校验链（fail-closed）：
      ① sidecar 存在性：缺失时默认拒绝（新产物必须有 inline manifest，
         096 producer 侧）；--legacy-no-runcap 显式降级放行并打
         [RUNCAP-LEGACY] 警告（旧 Qwen E100/E109 复用产物无 sidecar，
         存量打分路径不炸死，但 cap 身份未知如实记 null）。
      ② schema：read_run_cap_manifest（损坏/缺键 [CAP-SCHEMA-FAIL]）。
      ③ status：必须 complete（generating = 生成中断/未回填，拒绝）。
      ④ 绑定：结果文件字节 sha256 重算 == manifest.result_sha256
         （schema 通过 ≠ 绑定通过——TOCTOU/篡改在此拦截）；行数 ==
         manifest.n_samples（且 manifest.result_lines 非 null 时须一致）。
      ⑤ cap 身份提取（effective/protocol/identity hash/provenance）供
         聚合 uniformity 检查与 result.json _meta 落盘。
    """
    sidecar = jsonl_path + RUN_CAP_SIDECAR_SUFFIX
    if not os.path.lexists(sidecar):
        if legacy_allowed:
            print(f"[RUNCAP-LEGACY] 102: {os.path.basename(jsonl_path)} 无 "
                  f"run-cap sidecar（096 前存量产物）——--legacy-no-runcap "
                  f"显式降级打分，cap 身份未知（_meta 如实记 null），不得"
                  f"用于 cap 敏感对比聚合", file=sys.stderr)
            return None
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 无 run-cap sidecar——新产物"
            f"必须由 096 producer 侧生成 inline run-cap manifest；旧产物"
            f"（096 前 Qwen E100/E109 复用格）请显式传 --legacy-no-runcap "
            f"降级打分（cap 身份未知，不入 cap 敏感聚合）。fail loudly，"
            f"不写 result.json")
    manifest = read_run_cap_manifest(sidecar)   # ② schema fail-closed
    if manifest["status"] != "complete":
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 的 run-cap manifest "
            f"status={manifest['status']!r} 非 complete（generating = 生"
            f"成中断/complete 回填未发生）——半残产物拒绝打分，不写 "
            f"result.json")
    actual_sha = _runcap_file_sha256(jsonl_path)
    if manifest["result_sha256"] is None or actual_sha != manifest["result_sha256"]:
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 结果文件 sha256 与 manifest "
            f"绑定不符——manifest={str(manifest['result_sha256'])[:16]}… "
            f"实际={actual_sha[:16]}…（篡改/TOCTOU 交错/写错文件），拒绝"
            f"打分，不写 result.json")
    actual_lines = _runcap_file_lines(jsonl_path)
    if actual_lines != manifest["n_samples"] or (
            manifest["result_lines"] is not None
            and actual_lines != manifest["result_lines"]):
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 行数与 manifest 绑定不符——"
            f"实际 {actual_lines} 行 vs n_samples={manifest['n_samples']}"
            f" / result_lines={manifest['result_lines']}（部分写入/混行），"
            f"拒绝打分，不写 result.json")
    return {
        "cap_effective": manifest["effective_cap"],
        "cap_protocol": manifest["protocol_cap"],
        "cap_identity_hash": manifest["cap_identity"]["hash"],
        "provenance": manifest["provenance"],
        "result_sha256": actual_sha,
        "sidecar": os.path.basename(sidecar),
    }

def lbv2_choice_score(prediction, ground_truth, all_classes=None):
    """LongBench-v2（lbv2）四选一 accuracy：官方口径（THUDM/LongBench）两条
    大小写敏感模式解析 pred 字母，无匹配记 0 分，与真值字母比对。

    解析统一在 lbv2_choice.py（E117b，GPT TL-LBV2-PARSER-037）：历史实现
    的 IGNORECASE + 独立字母兜底会把冠词 "a" 判成选项 A（503×3 重放实测
    aavg 4 个假阳性，三臂排序反转），已删除。
    """
    pred_choice = extract_choice_official(prediction)
    gt = str(ground_truth).strip().upper()
    return 1.0 if (pred_choice is not None and pred_choice == gt) else 0.0


dataset2metric = {
    "lbv2": lbv2_choice_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", type=int, default=0, help="Evaluate on LongBench-E")
    parser.add_argument("--t", type=str, default="benchmark launch time")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--pred_postfix", type=str, default="")
    # E116a（GPT TL-LBV1-SAMPLE-GATE-024）fail-closed 完整性门禁：
    #   --manifest <json>：{task: {"ids": [...], "answers_sha": {id: sha}}} 预期清单，
    #     实际 _id 集合必须与预期完全相等（无缺行/重复/混行），answers hash 一致
    #   --expect-count <int>：每文件行数下限（不满足即非零退出）
    #   门禁失败 → SystemExit(2)，不写 result.json
    parser.add_argument("--manifest", type=str, default="",
                        help="样本身份 manifest（JSON），给定时做集合闭包校验")
    parser.add_argument("--expect-count", type=int, default=0,
                        help="每个 jsonl 的最小行数（>0 时启用）")
    # 102（TL-E096-CAP-CONSUMER-CLOSURE-102）：run-cap sidecar 消费开关。
    # --require-runcap 默认开：目录里有 sidecar 的文件强制校验（schema/
    # status/sha/行数），无 sidecar 的文件默认拒绝（fail-closed，不静默
    # 打分 cap 未知产物）。
    parser.add_argument("--require-runcap",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="打分前强制校验 run-cap sidecar（默认开）")
    # 旧产物降级开关（096 前 Qwen E100/E109 复用产物无 sidecar）：显式
    # 降级 + [RUNCAP-LEGACY] 警告 + _meta cap 字段如实记 null——存量打分
    # 路径不炸死，但 cap 敏感聚合不得混入 legacy 格。
    parser.add_argument("--legacy-no-runcap", action="store_true",
                        help="对无 run-cap sidecar 的旧产物显式降级打分"
                             "（打 [RUNCAP-LEGACY] 警告，cap 身份记 null）")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    if len(args.output_path) > 0:
        dir = f"{args.output_path}"
        path = f"{dir}/"
    elif args.e:
        dir = f"{args.output_dir}/pred_e{args.pred_postfix}"
        path = f"{dir}/"
    else:
        dir = f"{args.output_dir}/pred{args.pred_postfix}"
        path = f"{dir}/"

    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    # 102：降级口径 = --legacy-no-runcap 显式降级 或 --no-require-runcap
    # 显式关闭（两者都只作用于「sidecar 缺失」的文件——有 sidecar 的文件
    # 无论如何都强制校验，降级开关不得绕过既有 provenance）。
    legacy_allowed = args.legacy_no_runcap or (not args.require_runcap)
    runcap_metas = {}    # filename -> cap meta（strict）/ None（legacy）
    # E116a：fail-closed 完整性门禁（GPT TL-LBV1-SAMPLE-GATE-024）
    manifest = json.load(open(args.manifest)) if args.manifest else None
    import hashlib as _hashlib

    def _answers_sha(ans):
        return _hashlib.sha256(
            json.dumps(ans, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    for filename in sorted(all_files):
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths, budgets = [], [], [], []
        ids = []
        dataset = filename.split("-")[0]
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "_id" in data:
                    ids.append(data["_id"])
                if "length" in data:
                    lengths.append(data["length"])
                if "budget" in data:
                    budgets.append(data["budget"])
        # ---- 门禁 1：行数下限（--expect-count）----
        if args.expect_count > 0 and len(predictions) < args.expect_count:
            raise SystemExit(
                f"[GATE-FAIL] {filename}: {len(predictions)} 行 < expect-count "
                f"{args.expect_count}——fail closed，不写 result.json"
            )
        # ---- 门禁 2：_id 无重复（新数据全带 _id；旧 v1 文件无 _id 时跳过）----
        if ids:
            if len(ids) != len(set(ids)):
                dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 检测到重复 _id（示例 {dup}）——"
                    f"fail closed，不写 result.json"
                )
            if len(ids) != len(predictions):
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 部分 行缺失 _id（{len(ids)}/"
                    f"{len(predictions)}）——fail closed，不写 result.json"
                )
        # ---- 门禁 3：manifest 集合闭包 + answers hash 一致 ----
        if manifest is not None:
            m_task = manifest.get(dataset)
            if m_task is None:
                print(f"[GATE-WARN] {filename}: 任务 {dataset} 不在 manifest——跳过该文件")
                continue
            exp_ids = set(m_task["ids"])
            got_ids = set(ids) if ids else None
            if got_ids is None:
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: 预测缺 _id 字段，无法对 manifest 校验——"
                    f"fail closed，不写 result.json"
                )
            if got_ids != exp_ids:
                missing = sorted(exp_ids - got_ids)[:5]
                extra = sorted(got_ids - exp_ids)[:5]
                raise SystemExit(
                    f"[GATE-FAIL] {filename}: _id 集合与 manifest 不等"
                    f"（缺 {len(exp_ids-got_ids)} 例 {missing}；多 "
                    f"{len(got_ids-exp_ids)} 例 {extra}）——fail closed，不写 result.json"
                )
            exp_sha = m_task.get("answers_sha", {})
            for i, a in zip(ids, answers):
                if i in exp_sha and _answers_sha(a) != exp_sha[i]:
                    raise SystemExit(
                        f"[GATE-FAIL] {filename}: _id={i} 的 answers hash 与 manifest "
                        f"不一致（答案错配/混行）——fail closed，不写 result.json"
                    )
        # ---- 门禁 4（102）：run-cap sidecar 强制消费（任何打分之前）----
        # manifest 集合门跳过的文件（不在 manifest）不进打分集合，也不做
        # runcap 校验；进入打分集合的文件必须有合格 sidecar（或显式 legacy
        # 降级）。schema/status/sha/行数全链 fail-closed，见
        # validate_runcap_sidecar docstring。
        runcap_metas[filename] = validate_runcap_sidecar(
            f"{path}/{filename}", legacy_allowed)
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        budgets = list(filter(lambda x: x is not None, budgets))
        budget = sum(budgets) / len(budgets) if len(budgets) > 0 else -1
        scores[filename] = {"score": score, "budget": budget}
    if args.e:
        out_path = f"{dir}/result.json"
    else:
        out_path = f"{dir}/result.json"
    # 102：聚合集合 cap identity 单一性——同一 result.json 聚合的所有
    # strict 文件必须同 cap identity（16K treatment 与 31.5K 协议口径的
    # 分数不得混进同一张表被当同一口径均值）。异 cap 混挂目录拒绝打分
    # （fail-closed），需要分组时按 cap 分目录分别打分（既有派单口径）。
    strict_metas = [m for m in runcap_metas.values() if m is not None]
    cap_hashes = sorted({m["cap_identity_hash"] for m in strict_metas})
    if len(cap_hashes) > 1:
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: 聚合目录混挂 {len(cap_hashes)} 个不同 "
            f"cap identity（{cap_hashes}）——异 cap 产物不得混入同一聚合"
            f"（16K 受限 treatment 与 31.5K 协议口径必须分目录分别打分），"
            f"拒绝写 result.json，fail loudly")
    # E116a：scorer 后端身份随 result.json 落盘（跨环境复现性锚点）
    # 102：cap 身份（effective/protocol/identity hash/provenance）随
    # result.json _meta 落盘——分数与打分时验证过的 cap 口径绑定，跨目录
    # 聚合时可比对 cap identity 拒混挂。legacy 格（无 sidecar）如实记
    # null + 计数，不冒称。
    runcap_meta = {
        "strict_validated": len(strict_metas),
        "legacy_no_sidecar": len(runcap_metas) - len(strict_metas),
        "cap_effective": strict_metas[0]["cap_effective"] if strict_metas else None,
        "cap_protocol": strict_metas[0]["cap_protocol"] if strict_metas else None,
        "cap_identity_hash": cap_hashes[0] if cap_hashes else None,
        "provenance": sorted({m["provenance"] for m in strict_metas})
        if strict_metas else None,
    }
    scores["_meta"] = {"scorer_backend": SCORER_BACKEND_ID,
                       "runcap": runcap_meta}
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
