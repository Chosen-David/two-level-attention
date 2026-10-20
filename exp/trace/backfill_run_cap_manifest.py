# -*- coding: utf-8 -*-
"""TL-DSV2-CAP-PROVENANCE-096 存量产物补制工具（096 审计建议第 4 条）。

096 修复前（.24 在飞批之前/之中的产物）结果 JSONL 自身不绑定 cap：
同 task/method/时间戳先后用 16K 与 31.5K 跑，路径与记录结构完全相同，
只有启动 stdout 打印 [EXPERIMENT-CAP-DIVERGENCE] / [CONTEXT-CAP-GATE]
能证明当时口径。本工具给这类存量产物补制 run-cap manifest：

  provenance = "backfilled-from-stdout"  —— 启动日志能解析出 cap（gate
      打印行是 093 修复后生产者原生证据，可信度低于 inline 但有据可查）；
  provenance = "provenance_incomplete"   —— 日志缺失/解析不出 cap（连
      stdout 线索都没有），或样本全集无法证明完整（105④），是否入论文表
      由主 AI 裁决（如实降格，不冒称）。

【105 修复（GPT 审计 2026-10-12_0739 TL-E096-BACKFILL-BINDING-105，P1）】
修复前本工具从整份日志 `.search()` 永远取第一条 cap、不绑定 run 身份、
不校验样本全集——跨模型日志可把一行 partial JSONL 声明成另一模型的
`complete` 产物（GPT CPU 反例复现）。修复后五重绑定（全部 fail-closed）：

  ① 日志分块 + run 身份绑定：按 `[CONTEXT-CAP-GATE] model=` 行把日志
     分块；块内以生产者打印的 `[CONTEXT-CAP-GATE] run-bind task=… t=…
     output=…` 行（105 修复起 pred.py 在 gate 后打印）绑定 task/时间戳/
     输出 basename。CLI --model/--task/--t/--result 与日志块不一致 →
     [GATE-FAIL] 拒绝（跨模型反例在此拦截）。
  ② 唯一候选：多个 gate 块与目标匹配（同 model/task/t/output 重复出现）
     → 拒绝，不再永远取第一条。旧格式日志（096/105 前生产者无 run-bind
     行——.24 存量批日志即此格式）走 model-only 回退：同 model 块恰 1 个
     → 用之；多个且 cap 证据全部一致 → 用之并记 consensus 注记；多个
     且 cap 不一致 → 无法归因 → 拒绝。
  ③ 日志 SHA256 + 匹配字节范围进 manifest（log_binding 字段）——补制
     证据可独立复核到字节级。
  ④ 样本全集完整性：JSONL 行数 != 数据集预期全集（EXPECTED_SAMPLE_
     COUNTS 表 / --expect-count 显式声明）时不得写 backfilled-from-
     stdout 的 complete 口径，只能降格 provenance_incomplete + note
     （隔离出论文聚合）。
  ⑤ 行内 _id 数据指纹绑定：E116a 起生产 _id = "{dataset}:{全量数据指纹}:"
     "{行号}"——partial 文件重算指纹 != _id 内嵌全量指纹 → [GATE-FAIL]
     直接拒绝（不降格——指纹冲突是「这份文件不是那次全集运行」的硬证据，
     降格放行会让操作者继续错配）；行号序列非 0..n-1（缺行/重行）同样拒
     绝。_id 通过 = 全集证明（优先于 ④ 的计数比对）；无 _id 的旧 v1 文件
     走 ④ 计数口径。

用法：
  python3 exp/trace/backfill_run_cap_manifest.py \
      --result /path/to/hotpotqa-none-1012024000.jsonl \
      --log /path/to/that_run.stdout.log \
      --model DeepSeek-V2-Lite-Chat --method none --task hotpotqa \
      --t 1012024000 [--model-path ...] [--config-dir ...] \
      [--expect-count N] [--overwrite-backfill]

规则（fail-closed，除 105 新增五重绑定外沿用 096 版）：
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
#   [EXPERIMENT-CAP-DIVERGENCE] 本次实验 cap=16384 与注册表 protocol 值 31500
# 105：gate 块锚点（分块）+ run-bind 绑定行（105 修复起 pred.py 打印）：
#   [CONTEXT-CAP-GATE] model=DeepSeek-V2-Lite-Chat
#   [CONTEXT-CAP-GATE] run-bind task=hotpotqa t=1012024000 output=hotpotqa-none-1012024000.jsonl
_RE_LIVE = re.compile(
    r"experiment_input_cap \(live\)\s*=\s*(\d+)")
_RE_PROTO = re.compile(
    r"experiment_input_cap \(registry protocol\)\s*=\s*(\d+)")
_RE_DIVERG_CAP = re.compile(
    r"\[EXPERIMENT-CAP-DIVERGENCE\] 本次实验 cap=(\d+) 与注册表 protocol 值 (\d+)")
_RE_GATE_HEADER = re.compile(r"\[CONTEXT-CAP-GATE\] model=(\S+)")
_RE_RUN_BIND = re.compile(
    r"\[CONTEXT-CAP-GATE\] run-bind task=(\S+) t=(\S+) output=(\S+)")

# 105④：数据集预期样本全集（test split 行数）。来源 = 仓内 E116b 全量产物
# 实测（wc -l，与 LongBench 官方 test split 一致；multifieldqa_en=150 系
# 官方 dev/test 切分，勿凭论文印象写 200）。DeepSeek .24 线任务全在表内；
# 表外任务须显式 --expect-count 声明，否则无法证明全集 → 降格
# provenance_incomplete。
EXPECTED_SAMPLE_COUNTS = {
    "2wikimqa": 200, "gov_report": 200, "hotpotqa": 200,
    "lbv2": 503, "lcc": 500, "multifieldqa_en": 150,
    "multi_news": 200, "musique": 200, "narrativeqa": 200,
    "passage_count": 200, "passage_retrieval_en": 200, "qasper": 200,
    "qmsum": 200, "repobench-p": 500, "samsum": 200, "trec": 200,
    "triviaqa": 200,
}


def parse_cap_from_block(block_text):
    """从单个 gate 块解析 (effective_cap, protocol_cap)。

    优先 [EXPERIMENT-CAP-DIVERGENCE] 行（effective + protocol 一行双证），
    回退两条 [CONTEXT-CAP-GATE] 打印。块内任一 effective 证据都没有 →
    (None, None)。
    """
    eff, proto = None, None
    m = _RE_DIVERG_CAP.search(block_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _RE_LIVE.search(block_text)
    if m:
        eff = int(m.group(1))
    m = _RE_PROTO.search(block_text)
    if m:
        proto = int(m.group(1))
    return eff, proto


def parse_cap_from_log(log_text):
    """【096 兼容入口，保留给既有调用方】整份日志取首条 cap。

    105 修复后主路径用 resolve_cap_evidence（分块 + 绑定 + 唯一候选）；
    本函数仅维持旧签名可导入（test_e096 套件静态引用），不再被 main
    消费。
    """
    return parse_cap_from_block(log_text)


def split_gate_blocks(log_bytes):
    """105①：按 `[CONTEXT-CAP-GATE] model=` 行分块。

    返回 block 列表：{"model", "byte_start", "byte_end", "text"}。块边界
    = 锚点行首字节 → 下一锚点行首字节（或 EOF）；字节范围供 manifest
    ③ 级证据复核。锚点行之前的日志属于 model=None 的前导区（无法绑定
    模型身份，不参与候选匹配）。
    """
    blocks = []
    cur = None
    off = 0
    for raw_line in log_bytes.splitlines(keepends=True):
        line = raw_line.decode("utf-8", errors="replace")
        m = _RE_GATE_HEADER.search(line)
        if m:
            if cur is not None:
                cur["byte_end"] = off
                blocks.append(cur)
            cur = {"model": m.group(1), "byte_start": off,
                   "byte_end": None, "text": ""}
        if cur is not None:
            cur["text"] += line
        off += len(raw_line)
    if cur is not None:
        cur["byte_end"] = off
        blocks.append(cur)
    return blocks


def resolve_cap_evidence(log_path, model, task, t, output_basename):
    """105①②③：日志 → 本 run 的 (eff, proto, binding) 或 [GATE-FAIL]。

    匹配链（全部 fail-closed，返回前不再有「取第一条」语义）：
      1. 无任何 model= 锚点 → 拒绝（日志与目标 run 无可绑定关系）。
      2. 无 model 匹配块 → 拒绝（跨模型日志反例在此拦截）。
      3. 匹配块含 run-bind 行（105 修复后生产者日志）：候选 = model 且
         (task,t,output) 三元组一致的块；恰 1 → 用；0 → 拒绝（日志表明
         该模型有 runs 但无与 --task/--t/--result 匹配者）；>1 → 拒绝
         （不唯一匹配）。
      4. 匹配块均无 run-bind（096/105 前旧格式日志，.24 存量）：
         恰 1 块 → 用之（model-only 绑定 + 结果文件名一致性由调用方
         单独校验）；多块 → cap 证据 (eff,proto) 全一致才用（consensus），
         不一致 → 拒绝（无法归因到本 run）。
      5. 选定块的 cap 证据解析不出 effective → 交由调用方降格
        provenance_incomplete（模型已锚定但无 cap 打印——旧门禁跳过路径）。
    """
    with open(log_path, "rb") as f:
        log_bytes = f.read()
    log_sha = hashlib.sha256(log_bytes).hexdigest()
    blocks = split_gate_blocks(log_bytes)
    if not blocks:
        raise SystemExit(
            f"[GATE-FAIL] 105 backfill: 日志 {log_path} 无任何 "
            f"[CONTEXT-CAP-GATE] model= 锚点——与目标 run 无可绑定关系，"
            f"拒绝补制（缺 cap 证据请省略 --log 直接落 "
            f"provenance_incomplete，fail loudly）")
    model_blocks = [b for b in blocks if b["model"] == model]
    if not model_blocks:
        models = sorted({b["model"] for b in blocks})
        raise SystemExit(
            f"[GATE-FAIL] 105 backfill: 日志中 gate 块的 model "
            f"({models}) 与 --model={model} 不一致——跨模型日志不得给 "
            f"目标产物补制 cap provenance，拒绝（GPT 105 反例：无关日志"
            f"声明另一模型的产物）")
    bind_blocks = [b for b in model_blocks if _RE_RUN_BIND.search(b["text"])]
    binding = {
        "log_file": os.path.basename(log_path),
        "log_sha256": log_sha,
        "match_mode": None,
        "match_byte_ranges": [],
    }
    if bind_blocks:
        cands = []
        for b in bind_blocks:
            m = _RE_RUN_BIND.search(b["text"])
            if (m.group(1) == task and m.group(2) == t
                    and m.group(3) == output_basename):
                cands.append(b)
        if not cands:
            raise SystemExit(
                f"[GATE-FAIL] 105 backfill: 日志中 model={model} 的 run-bind"
                f" 行与 --task={task} --t={t} --result={output_basename} "
                f"均不一致（日志属该模型的其它 run）——CLI 与日志块必须"
                f"一致，拒绝补制，fail loudly")
        if len(cands) > 1:
            raise SystemExit(
                f"[GATE-FAIL] 105 backfill: 日志中 {len(cands)} 个 gate 块"
                f"与目标 (model={model}, task={task}, t={t}, "
                f"output={output_basename}) 重复匹配——候选不唯一（不得"
                f"取第一条），拒绝补制，fail loudly")
        blk = cands[0]
        binding["match_mode"] = "run-bind"
        binding["match_byte_ranges"] = [[blk["byte_start"], blk["byte_end"]]]
        eff, proto = parse_cap_from_block(blk["text"])
    else:
        # 旧格式日志（096/105 前生产者无 run-bind 行）：model-only 回退
        if len(model_blocks) == 1:
            blk = model_blocks[0]
            binding["match_mode"] = "model-only (legacy log, no run-bind)"
            binding["match_byte_ranges"] = [[blk["byte_start"],
                                              blk["byte_end"]]]
            eff, proto = parse_cap_from_block(blk["text"])
        else:
            caps = {parse_cap_from_block(b["text"]) for b in model_blocks}
            if len(caps) == 1:
                blk = model_blocks[0]
                binding["match_mode"] = (
                    f"model-consensus over {len(model_blocks)} legacy "
                    f"blocks (cap evidence identical)")
                binding["match_byte_ranges"] = [
                    [b["byte_start"], b["byte_end"]] for b in model_blocks]
                eff, proto = caps.pop()
            else:
                raise SystemExit(
                    f"[GATE-FAIL] 105 backfill: 日志含 {len(model_blocks)} "
                    f"个 model={model} 的 gate 块且 cap 证据不一致 "
                    f"（{sorted(caps)}）——旧格式日志无 run-bind 绑定、"
                    f"无法归因到本 run，拒绝补制（请提供单次 run 的日志"
                    f"切片），fail loudly")
    return eff, proto, binding


def load_result_rows(result_path):
    """读结果 JSONL 全行（供指纹/行号/行数校验，一次读入复用）。"""
    rows = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recompute_dataset_fingerprint_rows(rows, task):
    """按 pred.py E116a 同口径重算数据指纹（answers+length 逐行 canonical
    JSON 的 SHA256 前 12 位）；lbv2（行内含官方 _id）返回 ""。"""
    if task == "lbv2" or (rows and "_id" in rows[0]
                          and isinstance(rows[0]["_id"], str)
                          and ":" not in rows[0]["_id"]):
        # lbv2 官方 _id（非 "{dataset}:{fp}:{idx}" 复合格式）→ 与生产口径
        # 一致记 ""（生产侧 dataset == "lbv2" 时 data_fp=""）
        return ""
    src = json.dumps([[r.get("answers"), r.get("length")] for r in rows],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def recompute_dataset_fingerprint(result_path, task):
    """【096 兼容入口，保留给既有调用方】按文件重算数据指纹。"""
    return recompute_dataset_fingerprint_rows(load_result_rows(result_path),
                                              task)


def check_id_fingerprint(rows, task, recomputed_fp):
    """105⑤：行内 _id 全量数据指纹 + 行号序列校验。

    返回 (embedded_fp, proof_ok)：
      embedded_fp 非 None = 全量数据指纹（E116a 生产 _id 内嵌），proof_ok
      = 指纹重算一致且行号恰为 0..n-1（全集证明，优先于 ④ 计数口径）。
      _id 非 composit 格（lbv2 官方 id / 旧 v1 无 _id）→ (None, False)，
      走 ④ 计数口径。
    指纹冲突 / 行号缺重 / 前缀错配 / 行间 _id 格式不一致 → [GATE-FAIL]
    直接拒绝（不降格——这是「这份文件不是那次全集运行产物」的硬证据）。
    """
    n = len(rows)
    pat = re.compile(r"^" + re.escape(task) + r":([0-9a-f]{12}):(\d+)$")
    if n == 0:
        return None, False
    first_m = pat.match(str(rows[0].get("_id", "")))
    if not first_m:
        if "_id" in rows[0]:
            sid = str(rows[0]["_id"])
            if ":" in sid and not sid.startswith(task + ":"):
                raise SystemExit(
                    f"[GATE-FAIL] 105 backfill: 行内 _id={sid!r} 的任务前缀"
                    f"与 --task={task} 不符——结果文件与 CLI 声明不属于同"
                    f"一任务，拒绝补制，fail loudly")
            if ":" in sid:
                raise SystemExit(
                    f"[GATE-FAIL] 105 backfill: 行内 _id={sid!r} 不符合 "
                    f"{{dataset}}:{{指纹12}}:{{行号}} 生产格式——无法验证"
                    f"全集绑定，拒绝补制（该文件可能非 E116a 后生产产物），"
                    f"fail loudly")
        # A legacy first row cannot waive identity checks for later rows.
        has_id = "_id" in rows[0]
        for lineno, row in enumerate(rows, 1):
            if ("_id" in row) != has_id or ":" in str(row.get("_id", "")):
                raise SystemExit(
                    f"[GATE-FAIL] backfill: 第 {lineno} 行 _id 格式与旧格式"
                    f"回退不一致——缺失/官方 ID 不得与复合 ID 混行，拒绝补制")
        return None, False   # lbv2 官方 id / 旧 v1 无 _id → ④ 计数口径
    embedded_fp = first_m.group(1)
    idxs = []
    for r in rows:
        sid = str(r.get("_id", ""))
        m = pat.match(sid)
        if not m or m.group(1) != embedded_fp:
            raise SystemExit(
                f"[GATE-FAIL] 105 backfill: 行内 _id 指纹不一致（首行指纹"
                f" {embedded_fp} vs 本行 {sid!r}）——混行/篡改/手工拼接，"
                f"拒绝补制，fail loudly")
        idxs.append(int(m.group(2)))
    if idxs != list(range(n)):
        raise SystemExit(
            f"[GATE-FAIL] 105 backfill: 行内 _id 行号序列非 0..{n - 1}"
            f"（缺行/重行/乱序）——partial 产物不得冒充全集，拒绝补制，"
            f"fail loudly")
    if embedded_fp != recomputed_fp:
        raise SystemExit(
            f"[GATE-FAIL] 105 backfill: 行内 _id 内嵌全量数据指纹 "
            f"{embedded_fp} != 本文件重算指纹 {recomputed_fp}——本文件"
            f"不是该指纹对应全集的完整产物（GPT 105 反例：一行 partial "
            f"文件冒充全集），拒绝补制，fail loudly")
    return embedded_fp, True


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
    p.add_argument("--expect-count", type=int, default=None,
                   help="105④：显式声明数据集预期全集行数（表外任务必"
                        "须给；与 EXPECTED_SAMPLE_COUNTS 表冲突时以显式"
                        "声明为准并打 [BACKFILL-WARN]）")
    p.add_argument("--overwrite-backfill", action="store_true",
                   help="已存在 backfilled manifest 时允许重写（幂等重跑）")
    args = p.parse_args()

    result_path = os.path.realpath(args.result)
    if not os.path.isfile(result_path):
        raise SystemExit(f"[GATE-FAIL] 096 backfill: 结果文件不存在"
                         f"（{result_path}），拒绝补制")
    output_basename = os.path.basename(result_path)

    # 105①前哨：CLI --task/--t 与结果文件名自身的一致性（文件名 = 生产者
    # 生成的 {task_prefix}-{method}-{t} 结构——CLI 声明与产物身份对不上
    # 直接拒，不等到日志匹配才发现）。
    task_prefix = args.task.split("-")[0]
    if args.t not in output_basename \
            or not output_basename.startswith(task_prefix + "-"):
        raise SystemExit(
            f"[GATE-FAIL] 105 backfill: 结果文件名 {output_basename!r} 与 "
            f"--task={args.task} / --t={args.t} 不一致（文件名须为 "
            f"'{task_prefix}-<method>-{args.t}.jsonl' 结构）——CLI 声明与"
            f"产物身份不符，拒绝补制，fail loudly")

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

    # ---- cap 证据解析（105①②③：分块 + 绑定 + 唯一候选 + 字节范围）----
    log_binding = None
    if args.log:
        eff, proto, log_binding = resolve_cap_evidence(
            args.log, args.model, args.task, args.t, output_basename)
    else:
        eff, proto = None, None
    if eff is None:
        provenance = "provenance_incomplete"
        proto = None
        note = ("启动日志缺失或该 run 的 gate 块解析不出 cap 打印——无"
                "任何 run 口径证据，是否入论文表由主 AI 裁决（096 审计"
                "建议第 4 条）")
    else:
        provenance = "backfilled-from-stdout"
        note = ("cap 证据来自启动 stdout 的 gate 打印（093 修复后生产者"
                "原生证据），非生产时同步落盘——弱于 inline，强于无证据")

    # ---- 105④⑤：样本全集完整性（先读全行，一次复用）----
    rows = load_result_rows(result_path)
    n_samples = len(rows)
    recomputed_fp = recompute_dataset_fingerprint_rows(rows, args.task)
    embedded_fp, id_proof_ok = check_id_fingerprint(
        rows, args.task, recomputed_fp)
    expected = args.expect_count
    if expected is not None and expected != EXPECTED_SAMPLE_COUNTS.get(
            args.task):
        print(f"[BACKFILL-WARN] 105: --expect-count={expected} 与表值 "
              f"{EXPECTED_SAMPLE_COUNTS.get(args.task)} 不同——以显式声明"
              f"为准，操作者自负其责", file=sys.stderr)
    if expected is None:
        expected = EXPECTED_SAMPLE_COUNTS.get(args.task)
    if id_proof_ok:
        sample_set_proof = "id-fingerprint (embedded == recomputed, idx 0..n-1)"
    elif expected is not None:
        sample_set_proof = (f"count ({n_samples} == 预期全集 {expected})"
                            if n_samples == expected
                            else f"count-mismatch ({n_samples} != 预期全集 "
                                 f"{expected})")
    else:
        sample_set_proof = None   # 无法证明（表外任务且未显式声明）
    sample_set_complete = (id_proof_ok
                           or (expected is not None
                               and n_samples == expected))
    # ④：全集无法证明 → 不得写 complete 口径的 backfilled provenance，
    # 降格 provenance_incomplete（cap 证据保留——诚实记录「有 cap 证据但
    # 样本集不完整」，隔离出论文聚合由 note 声明）。
    if not sample_set_complete and provenance == "backfilled-from-stdout":
        provenance = "provenance_incomplete"
        note += ("；【105④】样本全集无法证明完整（" +
                 (sample_set_proof or "表外任务且未 --expect-count 声明") +
                 "）——降格 provenance_incomplete，隔离出论文聚合")

    # ---- 从结果产物可复原的字段 ----
    result_sha = hashlib.sha256(open(result_path, "rb").read()).hexdigest()
    # ⑤：_id 指纹通过时用内嵌全量指纹（强于文件内重算值——文件行只是全集
    # 的投影，内嵌值绑定的是全集）；否则沿用文件内重算值（旧 v1 口径）。
    data_fp = embedded_fp if id_proof_ok else recomputed_fp
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
    # 105③：日志证据绑定（log SHA256 + 匹配字节范围 + 匹配模式）进
    # manifest——补制证据可独立复核到字节级。
    if log_binding is not None:
        core["log_binding"] = log_binding
    # 105④：样本全集完整性显式落盘（complete 与否的判据可复核）。
    core["sample_set"] = {
        "complete": bool(sample_set_complete),
        "n_samples": n_samples,
        "expected": expected,
        "proof": sample_set_proof,
        "id_fingerprint_verified": bool(id_proof_ok),
    }
    manifest = finalize_run_cap_manifest(
        core, status="complete", n_samples=n_samples,
        input_tokens=None,          # JSONL 只有字符数 length，token 数不可复原
        truncated_flags=None,
        result_file=output_basename,
        result_sha256=result_sha, result_lines=n_samples,
        provenance=provenance)
    validate_run_cap_manifest(manifest)
    write_run_cap_manifest(sidecar, manifest)
    print(f"[096-backfill] provenance={provenance} effective_cap={eff} "
          f"protocol_cap={proto} sample_set_complete={sample_set_complete} "
          f"-> {sidecar}")


if __name__ == "__main__":
    main()
