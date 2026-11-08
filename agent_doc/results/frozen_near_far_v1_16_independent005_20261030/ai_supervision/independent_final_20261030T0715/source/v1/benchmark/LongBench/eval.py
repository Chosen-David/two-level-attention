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
    runcap_gate_lock,
    cap_comparability_key,
    validate_input_identity_uniformity,
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


def validate_runcap_sidecar(jsonl_path, legacy_allowed,
                            actual_sha=None, actual_lines=None):
    """102/111/115：打分前对单个 *.jsonl 强制校验 run-cap sidecar。

    返回 cap 身份 dict（strict 通过）或 None（legacy 降级放行）；
    违规一律 SystemExit（[RUNCAP-FAIL]/[CAP-SCHEMA-FAIL]），调用方不写
    result.json。校验链（fail-closed）：
      ① sidecar 存在性：缺失时默认拒绝（新产物必须有 inline manifest，
         096 producer 侧）；--legacy-no-runcap 显式降级放行并打
         [RUNCAP-LEGACY] 警告（旧 Qwen E100/E109 复用产物无 sidecar，
         存量打分路径不炸死，但 cap 身份未知如实记 null）。
      ② schema：read_run_cap_manifest（损坏/缺键 [CAP-SCHEMA-FAIL]）。
      ③ status：必须 complete（generating = 生成中断/未回填，拒绝）。
      ④ result_file 闭包（115）：manifest.result_file 必须就是当前
         JSONL 的 basename——误拷/错配 backfill 的 sidecar（cap 身份属
         于另一结果文件）拒绝。
      ⑤ 绑定：结果文件字节 sha256 == manifest.result_sha256（schema 通过
         ≠ 绑定通过——篡改在此拦截）；行数 == manifest.n_samples（且
         manifest.result_lines 非 null 时须一致）。
      ⑥ cap 身份提取（effective/protocol/identity hash/comparability
         key/provenance）供聚合 uniformity 检查与 result.json _meta 落盘。
    【111】actual_sha/actual_lines：调用方在同一 runcap_gate_lock 锁窗口
    内从**同一字节快照**预计算的 sha256/行数——评分、SHA、行数和
    provenance 必须绑定同一不可变快照（修复前 eval 先 parse A、后重开同
    路径校验重跑后的 B：分数来自 A、元数据 SHA 来自 B 的 TOCTOU）。缺省
    时本函数自行重读（向后兼容，仅供非 TOCTOU 敏感调用方）。
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
    # pred.py uses dataset.split("-")[0] as the output prefix (repobench-p
    # becomes repobench). Bind that prefix to the scorer's dataset selection.
    dataset = manifest["dataset"]
    file_dataset = os.path.basename(jsonl_path).split("-")[0]
    if not isinstance(dataset, str) or dataset.split("-")[0] != file_dataset:
        raise SystemExit(
            f"[RUNCAP-FAIL] DATASET-BINDING: {jsonl_path} 文件名前缀 "
            f"{file_dataset!r} 与 manifest.dataset={dataset!r} 不一致，"
            "拒绝用错误数据集评分或分组，不写 result.json")
    if manifest["status"] != "complete":
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 的 run-cap manifest "
            f"status={manifest['status']!r} 非 complete（generating = 生"
            f"成中断/complete 回填未发生）——半残产物拒绝打分，不写 "
            f"result.json")
    # ④【115】sidecar 的 result_file 必须就是当前 JSONL 的 basename
    #（validate 层不知 jsonl 路径，此闭包只能在消费侧核对）。
    if manifest.get("result_file") != os.path.basename(jsonl_path):
        raise SystemExit(
            f"[RUNCAP-FAIL] 115: {jsonl_path} 的 run-cap manifest "
            f"result_file={manifest.get('result_file')!r} != 当前结果文件"
            f" basename {os.path.basename(jsonl_path)!r}——sidecar 误拷/"
            f"错配 backfill（cap 身份属于另一结果文件），拒绝打分，不写 "
            f"result.json")
    if actual_sha is None:
        actual_sha = _runcap_file_sha256(jsonl_path)
    if manifest["result_sha256"] is None or actual_sha != manifest["result_sha256"]:
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 结果文件 sha256 与 manifest "
            f"绑定不符——manifest={str(manifest['result_sha256'])[:16]}… "
            f"实际={actual_sha[:16]}…（篡改/TOCTOU 交错/写错文件），拒绝"
            f"打分，不写 result.json")
    if actual_lines is None:
        actual_lines = _runcap_file_lines(jsonl_path)
    if actual_lines != manifest["n_samples"] or (
            manifest["result_lines"] is not None
            and actual_lines != manifest["result_lines"]):
        raise SystemExit(
            f"[RUNCAP-FAIL] 102: {jsonl_path} 行数与 manifest 绑定不符——"
            f"实际 {actual_lines} 行 vs n_samples={manifest['n_samples']}"
            f" / result_lines={manifest['result_lines']}（部分写入/混行），"
            f"拒绝打分，不写 result.json")
    ct = manifest["cap_treatment"] \
        if isinstance(manifest["cap_treatment"], dict) else {}
    input_identity = manifest.get("input_identity") \
        if isinstance(manifest.get("input_identity"), dict) else {}
    actual_input_ids = input_identity.get("actual_input_ids") \
        if isinstance(input_identity.get("actual_input_ids"), dict) else {}
    return {
        "dataset": manifest["dataset"],
        "cap_effective": manifest["effective_cap"],
        "cap_protocol": manifest["protocol_cap"],
        "cap_identity_hash": manifest["cap_identity"]["hash"],
        # 【112】跨任务 cap 等价 key（只含 cap 语义字段）——聚合单一性
        # 检查用它；完整 identity hash 仅供单文件防覆盖口径记录。
        "cap_comparability_key": cap_comparability_key(
            effective_cap=manifest["effective_cap"],
            protocol_cap=manifest["protocol_cap"],
            cap_treatment_type=ct.get("type"),
            cap_treatment_divergence=ct.get("divergence")),
        "provenance": manifest["provenance"],
        # LBv1 v2 输入闭包的聚合元数据。旧 sidecar 明确返回 null，不把
        # answers+length 旧指纹冒充完整 source/template/token 身份。
        "input_identity_schema": input_identity.get("schema"),
        "source_fingerprint": input_identity.get("source_fingerprint"),
        "prompt_template_sha256": input_identity.get(
            "prompt_template_sha256"),
        "actual_input_ids_ordered_sha256": actual_input_ids.get(
            "ordered_sha256"),
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


def sample_score(dataset, prediction, ground_truths, all_classes):
    """Official unrounded per-example score, shared by aggregate and paired reports."""
    score = 0.0
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip("\n").split("\n")[0]
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](
            prediction, ground_truth, all_classes=all_classes))
    return score


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        total_score += sample_score(dataset, prediction, ground_truths, all_classes)
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
        jsonl_full = f"{path}/{filename}"
        # ----【111 修复（GPT 审计 2026-10-12_0933 TL-E096-CAP-CONSUMER-
        #      TOCTOU-111，P1）】consumer 侧与 producer 同一把结果文件级
        # 锁（runcap_gate_lock，7367367 同 key：out_path+".runcap_gate.
        # lock"）——修复前 eval 先把 JSONL A parse 进内存、随后 validate
        # 才重开同路径算 SHA/行数：producer（pred.py open(out_path,"w")
        # 覆盖写）可交错完成 B+sidecar B，得到「分数来自 A、SHA/元数据来
        # 自 B」。现在读字节→parse→SHA→行数→sidecar 校验→评分输入在
        # 同一锁窗口一次完成（同一不可变字节快照），锁外只做 scorer。
        with runcap_gate_lock(jsonl_full):
            with open(jsonl_full, "rb") as f:
                raw = f.read()
            actual_sha = hashlib.sha256(raw).hexdigest()
            # 行数口径 = producer compute_result_binding 的二进制按行迭代
            actual_lines = raw.count(b"\n") + (
                1 if raw and not raw.endswith(b"\n") else 0)
            for line in raw.decode("utf-8").splitlines():
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
            # ---- 门禁 4（102/111/115）：run-cap sidecar 强制消费（任何
            #      打分之前）——manifest 集合门跳过的文件（不在 manifest）
            #      不进打分集合，也不做 runcap 校验；进入打分集合的文件必须
            #      有合格 sidecar（或显式 legacy 降级）。schema/status/
            #      result_file 闭包/sha/行数全链 fail-closed，且 sha/行数
            #      用同一锁窗口快照值（111），见 validate_runcap_sidecar
            #      docstring。
            runcap_metas[filename] = validate_runcap_sidecar(
                jsonl_full, legacy_allowed,
                actual_sha=actual_sha, actual_lines=actual_lines)
        # 锁外 scorer（评分输入已从快照字节物化为 Python 对象）
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
    # 102/112/113：聚合集合 cap 单一性——同一 result.json 聚合的所有
    # strict 文件必须同 cap 语义（16K treatment 与 31.5K 协议口径的
    # 分数不得混进同一张表被当同一口径均值）。异 cap 混挂目录拒绝打分
    # （fail-closed），需要分组时按 cap 分目录分别打分（既有派单口径）。
    strict_metas = [m for m in runcap_metas.values() if m is not None]
    legacy_count = len(runcap_metas) - len(strict_metas)
    # ----【113 修复（GPT 审计 2026-10-12_0933 TL-E096-CAP-LEGACY-MIX-113，
    #      P2）】strict 与 legacy 混合表 fail-closed 拒绝发布——修复前
    #      显式 legacy 模式允许「有 sidecar + 无 sidecar」混合同表发布，
    #      且把 strict 子集的 cap 标到整张表（_meta 同时
    #      strict_validated=1 + legacy_no_sidecar=1 + 非空 cap——legacy 格
    #      的 cap 身份未知却被标上 strict 的 cap 标签，同一张表两种
    #      provenance 口径）。全 legacy 表仅在显式模式（--legacy-no-
    #      runcap / --no-require-runcap）下放行且 cap 字段全 null（下方
    #      _meta 如实记），或先补齐 sidecar 再聚合。
    if strict_metas and legacy_count:
        raise SystemExit(
            f"[RUNCAP-FAIL] 113: 聚合目录混入 {len(strict_metas)} 个 "
            f"strict（sidecar 验证）+ {legacy_count} 个 legacy（无 "
            f"sidecar，cap 身份未知）产物——同一 result.json 不得混合两种"
            f"provenance 口径（legacy 格不得被标上 strict 的 cap 标签）。"
            f"请拆分目录分别打分，或先用 backfill 工具为 legacy 产物补制"
            f"sidecar 再聚合，拒绝写 result.json，fail loudly")
    # ----【112 修复（GPT 审计 2026-10-12_0933 TL-E096-CAP-UNIFORMITY-112，
    #      P1 可用性阻断）】跨任务 cap 单一性改用独立的 cap-comparability
    #      key（只含可跨任务统一的 cap 语义：effective/protocol/
    #      treatment 级）——修复前用完整 run identity hash（含 dataset/
    #      时间戳等 run 字段）做跨任务等价判断，不同数据集同 cap 的正常
    #      多任务聚合必被误拒（审计实测 strict 正常目录无法生成
    #      result.json）。完整 run identity 仍保留在 _meta（逐文件防
    #      覆盖口径记录），不参与跨任务等价判断。
    cap_keys = sorted({m["cap_comparability_key"] for m in strict_metas})
    if len(cap_keys) > 1:
        raise SystemExit(
            f"[RUNCAP-FAIL] 112: 聚合目录混挂 {len(cap_keys)} 个不同 "
            f"cap identity（cap-comparability key {cap_keys}——跨任务只"
            f"统一 cap 语义 effective/protocol/treatment，异 cap 产物不得"
            f"混入同一聚合被当同一口径均值，16K 受限 treatment 与 31.5K "
            f"协议口径必须分目录分别打分），拒绝写 result.json，"
            f"fail loudly")
    # TL-LBV1-INPUT-UNIFORMITY-20261021：逐文件拥有 v2 provenance 仍不
    # 足以形成公平 method 对比；同一 dataset 的 source/template/实际 token
    # 身份必须一致。门禁位于任何 result.json 发布之前。
    validate_input_identity_uniformity(runcap_metas)
    cap_hashes = sorted({m["cap_identity_hash"] for m in strict_metas})
    input_identity_by_file = {
        filename: {
            "schema": meta["input_identity_schema"],
            "source_fingerprint": meta["source_fingerprint"],
            "prompt_template_sha256": meta["prompt_template_sha256"],
            "actual_input_ids_ordered_sha256": meta[
                "actual_input_ids_ordered_sha256"],
        }
        for filename, meta in sorted(runcap_metas.items())
        if meta is not None and meta["input_identity_schema"] is not None
    }
    legacy_strict_identity_count = sum(
        1 for meta in strict_metas
        if meta["input_identity_schema"] is None)
    # E116a：scorer 后端身份随 result.json 落盘（跨环境复现性锚点）
    # 102/112：cap 身份（effective/protocol/comparability key/identity
    # hash/provenance）随 result.json _meta 落盘——分数与打分时验证过的
    # cap 口径绑定，跨目录聚合时可比对 cap key 拒混挂。legacy 格（无
    # sidecar）如实记 null + 计数，不冒称。
    runcap_meta = {
        "strict_validated": len(strict_metas),
        "legacy_no_sidecar": legacy_count,
        "cap_effective": strict_metas[0]["cap_effective"] if strict_metas else None,
        "cap_protocol": strict_metas[0]["cap_protocol"] if strict_metas else None,
        # 【112】跨任务可比 key（本表全部 strict 文件同 key）
        "cap_comparability_key": cap_keys[0] if cap_keys else None,
        # 完整 run identity：单 strict 文件时保留（防覆盖口径锚点）；
        # 多 strict 文件（异 dataset 同 cap，112 放行）时逐文件不同，
        # 不冒称单一——完整列表见 cap_identity_hashes。
        "cap_identity_hash": cap_hashes[0] if len(cap_hashes) == 1 else None,
        "cap_identity_hashes": cap_hashes,
        "provenance": sorted({m["provenance"] for m in strict_metas})
        if strict_metas else None,
        # 各 task 的 source/template/token hash 天然不同，不压成伪造的
        # “单一值”；逐文件留证，并把通过旧 run-cap schema 的 strict
        # sidecar 单独计数为 legacy（其 cap 仍可验证，输入闭包不可验证）。
        "input_identity": {
            "v2_validated": len(input_identity_by_file),
            "legacy_strict": legacy_strict_identity_count,
            "by_file": input_identity_by_file,
        },
    }
    scores["_meta"] = {"scorer_backend": SCORER_BACKEND_ID,
                       "runcap": runcap_meta}
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
