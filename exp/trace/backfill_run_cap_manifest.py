# -*- coding: utf-8 -*-
"""TL-DSV2-CAP-PROVENANCE-096 存量产物补制工具（096 审计建议第 4 条）。

096 修复前（.24 在飞批之前/之中的产物）结果 JSONL 自身不绑定 cap：
同 task/method/时间戳先后用 16K 与 31.5K 跑，路径与记录结构完全相同，
只有启动 stdout 打印 [EXPERIMENT-CAP-DIVERGENCE] / [CONTEXT-CAP-GATE]
能证明当时口径。本工具给这类存量产物补制 run-cap manifest：

  provenance = "backfilled-from-stdout"  —— 启动日志能解析出 cap（gate
      打印行是 093 修复后生产者原生证据，可信度低于 inline 但有据可查）；
  provenance = "provenance_incomplete"   —— 日志缺失/解析不出 cap（连
      stdout 线索都没有），是否入论文表由主 AI 裁决（如实降格，不冒称）。

用法：
  python3 exp/trace/backfill_run_cap_manifest.py \
      --result /path/to/hotpotqa-none-1012024000.jsonl \
      --log /path/to/that_run.stdout.log \
      --model DeepSeek-V2-Lite-Chat --method none --task hotpotqa \
      --t 1012024000 [--model-path ...] [--config-dir ...] [--overwrite-backfill]

规则（fail-closed）：
  - 目标 sidecar 已存在且 provenance=inline → 拒绝（真实生产落盘不允
    许 backfill 覆盖）；
  - 已存在且为 backfilled → 默认拒绝，显式 --overwrite-backfill 才重写
    （幂等重跑）；
  - 补制的 manifest 经 validate_run_cap_manifest 全量门（缺字段
    fail-closed）；
  - 每样本实际输入 token 数 / 截断标记在 JSONL 里不存在（只有字符数
    length），如实记 null + note，不伪造。

只依赖标准库 + benchmark.LongBench.run_cap_manifest（无 torch/GPU）。
"""
import argparse
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmark.LongBench.run_cap_manifest import (  # noqa: E402
    RUN_CAP_SIDECAR_SUFFIX, build_run_cap_core, finalize_run_cap_manifest,
    read_run_cap_manifest, resolve_producer_identity,
    validate_run_cap_manifest, write_run_cap_manifest)

# gate 打印行的解析模式（context_capability_gate.py 的输出格式，
# 093 修复后生产者原生证据）：
#   [CONTEXT-CAP-GATE]   experiment_input_cap (live)  = 16384   <- ...
#   [CONTEXT-CAP-GATE]   experiment_input_cap (registry protocol) = 31500
#   [EXPERIMENT-CAP-DIVERGENCE] 本次实验 cap=16384 与注册表 protocol 值 31500 不同
_RE_LIVE = re.compile(
    r"experiment_input_cap \(live\)\s*=\s*(\d+)")
_RE_PROTO = re.compile(
    r"experiment_input_cap \(registry protocol\)\s*=\s*(\d+)")
_RE_DIVERG_CAP = re.compile(
    r"\[EXPERIMENT-CAP-DIVERGENCE\] 本次实验 cap=(\d+) 与注册表 protocol 值 (\d+)")


def parse_cap_from_log(log_text):
    """从启动 stdout 解析 (effective_cap, protocol_cap)。

    优先 [EXPERIMENT-CAP-DIVERGENCE] 行（effective + protocol 一行双证），
    回退两条 [CONTEXT-CAP-GATE] 打印。任一 effective 证据都没有 →
    (None, None)（provenance_incomplete）。
    """
    eff, proto = None, None
    m = _RE_DIVERG_CAP.search(log_text)
    if m:
        eff, proto = int(m.group(1)), int(m.group(2))
        return eff, proto
    m = _RE_LIVE.search(log_text)
    if m:
        eff = int(m.group(1))
    m = _RE_PROTO.search(log_text)
    if m:
        proto = int(m.group(1))
    return eff, proto


def recompute_dataset_fingerprint(result_path, task):
    """按 pred.py E116a 同口径重算数据指纹（answers+length 逐行 canonical
    JSON 的 SHA256 前 12 位）；lbv2（行内含官方 _id）返回 ""。"""
    rows = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if task == "lbv2" or (rows and "_id" in rows[0]
                          and isinstance(rows[0]["_id"], str)
                          and ":" not in rows[0]["_id"]):
        # lbv2 官方 _id（非 "{dataset}:{fp}:{idx}" 复合格式）→ 与生产口径
        # 一致记 ""（生产侧 dataset == "lbv2" 时 data_fp=""）
        return ""
    src = json.dumps([[r.get("answers"), r.get("length")] for r in rows],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--result", required=True, help="存量结果 JSONL 路径")
    p.add_argument("--log", default=None,
                   help="该次运行的启动 stdout 日志（含 [CONTEXT-CAP-GATE]/"
                        "[EXPERIMENT-CAP-DIVERGENCE] 打印）；缺省即 "
                        "provenance_incomplete")
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--task", required=True, help="数据集名（生产口径全名）")
    p.add_argument("--t", required=True, help="运行时间戳（文件名尾段）")
    p.add_argument("--model-path", default=None)
    p.add_argument("--config-dir", default=None,
                   help="benchmark/LongBench/config 目录（可选：从 "
                        "dataset2maxlen.json 补 max_new_tokens）")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="显式声明 max_new_tokens（缺省查 config-dir 的 "
                        "dataset2maxlen.json，再缺省记 null）")
    p.add_argument("--overwrite-backfill", action="store_true",
                   help="已存在 backfilled manifest 时允许重写（幂等重跑）")
    args = p.parse_args()

    result_path = os.path.realpath(args.result)
    if not os.path.isfile(result_path):
        raise SystemExit(f"[GATE-FAIL] 096 backfill: 结果文件不存在"
                         f"（{result_path}），拒绝补制")
    sidecar = result_path + RUN_CAP_SIDECAR_SUFFIX
    if os.path.lexists(sidecar):
        existing = read_run_cap_manifest(sidecar)
        if existing["provenance"] == "inline":
            raise SystemExit(
                f"[GATE-FAIL] 096 backfill: {sidecar} 已存在且 provenance="
                f"inline（生产时同步落盘的真实 provenance）——backfill 不"
                f"得覆盖，fail loudly")
        if not args.overwrite_backfill:
            raise SystemExit(
                f"[GATE-FAIL] 096 backfill: {sidecar} 已存在"
                f"（provenance={existing['provenance']}）——重写须显式 "
                f"--overwrite-backfill（幂等重跑口径），fail loudly")

    # ---- cap 证据解析（provenance 三态分派）----
    log_text = None
    if args.log:
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    eff, proto = (None, None) if log_text is None \
        else parse_cap_from_log(log_text)
    if eff is None:
        provenance = "provenance_incomplete"
        proto = None
        note = ("启动日志缺失或解析不出 cap 打印——无任何 run 口径证据，"
                "是否入论文表由主 AI 裁决（096 审计建议第 4 条）")
    else:
        provenance = "backfilled-from-stdout"
        note = ("cap 证据来自启动 stdout 的 gate 打印（093 修复后生产者"
                "原生证据），非生产时同步落盘——弱于 inline，强于无证据")

    # ---- 从结果产物可复原的字段 ----
    result_sha = hashlib.sha256(open(result_path, "rb").read()).hexdigest()
    result_lines = sum(1 for _ in open(result_path, "rb"))
    data_fp = recompute_dataset_fingerprint(result_path, args.task)
    max_new = args.max_new_tokens
    if max_new is None and args.config_dir:
        d2m_path = os.path.join(args.config_dir, "dataset2maxlen.json")
        if os.path.isfile(d2m_path):
            d2m = json.load(open(d2m_path, encoding="utf-8"))
            max_new = d2m.get(args.task)

    # ---- 组装（effective_cap 证据分级：日志 > null）----
    core = build_run_cap_core(
        model_name=args.model,
        model_path=args.model_path,
        method=args.method,
        dataset=args.task,
        dataset_fingerprint=data_fp,
        run_timestamp=args.t,
        effective_cap=eff,   # None 时 cap_identity 亦如实含 null
        cap_gate_info=(
            {"registry_protocol_cap": proto, "capability": None,
             "max_position_embeddings": None} if eff is not None else None),
        max_new_tokens=max_new,
        producer=None,
        # provenance 分级如实记类型：连 effective 证据都没有 → unknown
        # （不得冒称 unregistered）；effective 有据但 protocol 打印缺失
        # → protocol-unknown（restricted 与 protocol 无法区分）；双证齐
        # → summarize 正常分派 restricted_context_treatment / protocol。
        treatment_type_hint=(
            "unknown" if eff is None
            else ("protocol-unknown" if proto is None else None)),
    )
    # backfill 的 producer = 补制工具自身身份 + note（生产 producer 已
    # 不可考，不冒称）
    core["producer"] = resolve_producer_identity(os.path.abspath(__file__))
    core["producer"]["note"] = ("backfill 工具身份——生产 producer 身份"
                                "生产时未落盘，已不可考，如实记补制者")
    core["effective_cap_source"] = (
        f"backfilled-from-stdout:{os.path.basename(args.log)}"
        if eff is not None else "unavailable (provenance_incomplete)")
    core["cap_treatment"]["backfill_note"] = note
    manifest = finalize_run_cap_manifest(
        core, status="complete", n_samples=result_lines,
        input_tokens=None,          # JSONL 只有字符数 length，token 数不可复原
        truncated_flags=None,
        result_file=os.path.basename(result_path),
        result_sha256=result_sha, result_lines=result_lines,
        provenance=provenance)
    validate_run_cap_manifest(manifest)
    write_run_cap_manifest(sidecar, manifest)
    print(f"[096-backfill] provenance={provenance} effective_cap={eff} "
          f"protocol_cap={proto} -> {sidecar}")


if __name__ == "__main__":
    main()
