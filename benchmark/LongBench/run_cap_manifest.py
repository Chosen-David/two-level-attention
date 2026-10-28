# -*- coding: utf-8 -*-
"""TL-DSV2-CAP-PROVENANCE-096 修复（GPT 审计 2026-10-12_0527_08819a7 §096，
P1）：run 级 cap provenance manifest——结果产物自身绑定实验输入 cap 与
处理口径，不再依赖启动 stdout 人工追踪。

093 修复把 16K 正确改称受限上下文 treatment（注册表
restricted_context_treatment=16384、protocol 31500 保留、gate 打
[EXPERIMENT-CAP-DIVERGENCE]），但落盘闭包缺口（096 审计原话）：
  - pred.py 调 enforce_context_capability_gate() 不接收返回值（含
    experiment_input_cap 的字典被丢弃）；
  - 每行 JSONL 只记数据集原字段 length，不记本次 max_length、模板后实际
    输入 token 数、是否截断；
  - 输出文件名只含 dataset/method/时间戳，cap 不在身份中；
  - sidecar 仅 method==tli 时写，fullkv（method=none）无任何 provenance
    旁挂；
  → 同一 task/method/时间戳先后用 16K 与 31.5K 跑，路径与记录结构完全
  相同——可覆盖或被当同一 treatment，产物自身无法证明处理口径。

本模块提供（全部 method 含 none/fullkv 统一接入，pred.py 生成开始时调用）：
  1. build_run_cap_core       —— run 级 manifest 的核心字段（模型/producer
     身份/protocol cap/effective cap/cap 类型与理由/max_new_tokens/数据
     指纹/时间戳），其中 cap_identity 是 cap 相关字段排序键 JSON→sha256
     前 10 hex 的 canonical hash（cap 进输出身份）。
  2. gate_run_cap_identity    —— fail-closed 写门：同一路径已存在不同 cap
     （或 cap_identity 不同）的 run-cap manifest 时打 [CAP-CLASH] 并
     SystemExit 非零退出，拒绝覆盖；同 cap 幂等放行（SKIP 语义）；目标
     manifest 缺失/损坏 → [CAP-SCHEMA-FAIL] fail-closed（provenance 未知
     不放行）；数据存在但无 manifest（096 前存量产物）→ 放行 + 打
     [CAP-PROVENANCE-MISSING] 警告（076 对非 tli 方法的既有覆盖语义保持，
     由本运行重建 provenance）。
  3. finalize_run_cap_manifest / write_run_cap_manifest —— 生成完成后回填
     每样本实际输入 token 数 / 截断标记 / 结果文件 sha256+行数（status
     generating→complete 两阶段：数据写之前先落 generating manifest，
     崩溃窗口留 manifest 可证 cap，不留无主数据）。
  4. validate_run_cap_manifest —— 消费/补制侧统一 schema 门（缺字段
     fail-closed，[CAP-SCHEMA-FAIL]）。
  5. resolve_producer_identity —— producer 代码身份（pred.py sha256 +
     git commit SHA；git 不可用时如实记 null）。

【与 081 treatment snapshot 的两层关系（设计约束，不得合并）】
sparse_attn/info.py 的 treatment manifest 刻意不消费 model2maxlen——
这是 081 snapshot 修复的既定语义（treatment 身份 = 算法配置，与实验 cap
解耦；T21 门禁固定该解耦）。本模块是**新增独立层**：run-cap manifest
回答「这次 run 用什么 cap、输入实际多长、结果文件是谁」，treatment
manifest 回答「这次 run 用什么稀疏算法配置」。两层同路径旁挂、互不
污染：cap 变更不触碰 treatment hash（tli 同 treatment 不同 cap 的两次
run：treatment sidecar 逐位相同，runcap sidecar 身份不同）。

【provenance 三态】
  inline                     —— 生产时同步落盘（pred.py 本路径）；
  backfilled-from-stdout     —— 存量产物事后由启动 stdout 日志补制
                               （exp/trace/backfill_run_cap_manifest.py，
                               096 审计建议第 4 条）；
  provenance_incomplete      —— 连 stdout 日志都缺/解析不出 cap 的产物，
                               是否入论文表由主 AI 裁决（如实降格，不冒称）。

只依赖标准库（json/os/sys/hashlib/subprocess/datetime/fcntl/contextlib），
测试与 backfill 工具可在无 torch/GPU 环境使用。
"""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys

# sidecar 后缀：{out_path}.runcap.json。刻意不以 .jsonl 结尾（不进任何
# *.jsonl glob，与 076 的 .tli_manifest.json 同纪律）；长度 13 字节——
# info.py 的 230 截断上限按 ".jsonl"(6)+".tli_manifest.json"(18) 预算，
# 本后缀更短不超 ext4 255 单文件名上限。
RUN_CAP_SIDECAR_SUFFIX = ".runcap.json"
RUN_CAP_SCHEMA = "run-cap-manifest-v1"
INPUT_IDENTITY_SCHEMA = "longbench-input-identity-v2"
INPUT_IDENTITY_CANONICALIZATION = "ordered-json-utf8-length-prefix-v1"
INPUT_IDS_ENCODING = "signed-int64-little-endian-v1"
# 103（TL-E096-CAP-GATE-TOCTOU-103）gate 写门锁后缀（**结果文件级**，
# 2026-10-12 粒度修正：首版为目录级，其「生产调度本就是串行 bash 链」
# 前提被生产打破——同目录多任务并发是常态派单形态，目录级锁令后到
# 者持整卡已加载显存空等数小时。GPT 103 威胁模型是「同一输出路径」
# 交错，锁挂结果文件粒度即闭合 TOCTOU，同目录不同结果文件不互斥）。
RUN_CAP_LOCK_SUFFIX = ".runcap_gate.lock"
RUN_CAP_PROVENANCES = ("inline", "backfilled-from-stdout",
                       "provenance_incomplete")

# validate_run_cap_manifest 的必填键（缺一即 [CAP-SCHEMA-FAIL]）。
# protocol_cap/effective_cap 允许 null 值（unregistered 模型 /
# provenance_incomplete 补制），但**键必须存在**——「没查」与「查了没有」
# 是两种不同的诚实。
_REQUIRED_KEYS = (
    "schema", "provenance", "status", "created_utc",
    "model_name", "model_path", "producer",
    "method", "dataset", "run_timestamp", "dataset_fingerprint",
    "effective_cap", "protocol_cap", "cap_treatment", "max_new_tokens",
    "cap_identity", "n_samples", "input_tokens", "truncated",
    "result_file", "result_sha256", "result_lines",
)


def _utcnow_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      default=str)


def _sha256_hex(value):
    return hashlib.sha256(value).hexdigest()


def build_lbv1_input_identity(data, prompt_template):
    """Build the v2 LongBench-v1 source and prompt-template identity.

    The old E116a fingerprint hashed only ``answers`` and ``length``.  This
    version hashes every ordered source row using unambiguous length framing,
    so a question/context/metadata edit or row reorder changes identity.  The
    prompt template is a separate field because source equality and rendering
    equality are different claims.
    """
    digest = hashlib.sha256()
    count = 0
    for row in data:
        encoded = json.dumps(
            row, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
        count += 1
    source_sha = digest.hexdigest()
    return {
        "schema": INPUT_IDENTITY_SCHEMA,
        "canonicalization": INPUT_IDENTITY_CANONICALIZATION,
        "source_sha256": source_sha,
        "source_fingerprint": "v2-" + source_sha,
        "sample_count": count,
        "prompt_template_sha256": _sha256_hex(
            str(prompt_template).encode("utf-8")),
    }


def input_ids_sha256(token_ids):
    """Hash one realized model input using a portable int64 byte contract."""
    digest = hashlib.sha256()
    for raw in token_ids:
        value = int(raw)
        if value < -(1 << 63) or value >= (1 << 63):
            raise ValueError("token id is outside signed int64 range")
        digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _input_identity_contract(identity):
    if identity is None:
        return None
    return {
        "schema": identity.get("schema"),
        "canonicalization": identity.get("canonicalization"),
        "source_sha256": identity.get("source_sha256"),
        "source_fingerprint": identity.get("source_fingerprint"),
        "sample_count": identity.get("sample_count"),
        "prompt_template_sha256": identity.get("prompt_template_sha256"),
    }


def _ordered_digest_hash(values):
    digest = hashlib.sha256()
    for value in values:
        raw = bytes.fromhex(value)
        digest.update(len(raw).to_bytes(8, "little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def validate_input_identity_uniformity(runcap_metas):
    """Reject unfair same-dataset method aggregates before publication."""
    modes = {meta.get('prompt_mode') for meta in runcap_metas.values() if meta is not None}
    if len(modes) > 1:
        raise SystemExit('[RUNCAP-FAIL] mixed prompt_mode protocols; refuse aggregation')
    grouped = {}
    for filename, meta in runcap_metas.items():
        if meta is not None:
            grouped.setdefault(meta["dataset"], []).append((filename, meta))
    for dataset, entries in grouped.items():
        if len(entries) < 2:
            continue
        schemas = {meta["input_identity_schema"] for _, meta in entries}
        if len(schemas) != 1:
            raise SystemExit(
                f"[RUNCAP-FAIL] LBV1-INPUT-UNIFORMITY: dataset={dataset!r} "
                "混合 v2 与 legacy strict 输入身份——同一比较表无法证明各 "
                "method 使用相同输入，拒绝写 result.json")
        if schemas == {None}:
            continue
        identities = {
            (meta["source_fingerprint"], meta["prompt_template_sha256"],
             meta["actual_input_ids_ordered_sha256"])
            for _, meta in entries
        }
        if any(value is None for identity in identities for value in identity):
            raise SystemExit(
                f"[RUNCAP-FAIL] LBV1-INPUT-UNIFORMITY: dataset={dataset!r} "
                "v2 身份缺 source/template/actual-token 闭包，拒绝比较")
        if len(identities) != 1:
            files = [filename for filename, _ in entries]
            raise SystemExit(
                f"[RUNCAP-FAIL] LBV1-INPUT-UNIFORMITY: dataset={dataset!r} 的 "
                f"{len(entries)} 个 method 产物使用了 {len(identities)} 组不同 "
                f"source/template/actual-token 身份（文件 {files}），拒绝写 "
                "result.json")
    return True


def resolve_producer_identity(script_path):
    """producer 代码身份：{script_sha256, git_commit, script_path}。

    script_sha256 = 生产脚本（pred.py）字节 sha256；git_commit = 该脚本
    所在仓库 checkout 的 HEAD（git 不可用/非 git 目录时如实记 null，
    不静默假装）。两层都记：GPU 机平行 clone 的 commit 与关键文件 md5
    可能不同步（历史教训），文件字节身份是最后防线。
    """
    script_path = os.path.realpath(script_path)
    ident = {"script_path": script_path, "script_sha256": None,
             "git_commit": None}
    if os.path.isfile(script_path):
        ident["script_sha256"] = _sha256_file(script_path)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(script_path))))  # benchmark/LongBench/x.py → 仓根
    try:
        out = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=30)
        ident["git_commit"] = out.strip() or None
    except (subprocess.CalledProcessError, OSError,
            subprocess.TimeoutExpired):
        ident["git_commit"] = None
    return ident


def summarize_cap_gate(cap_gate_info, effective_cap):
    """把 gate 返回值（096 前 pred.py 丢弃的那个 dict）折叠成 manifest 的
    protocol/cap_treatment 字段。

    cap_gate_info=None（未登记模型/注册表缺失，gate 跳过）→ protocol_cap
    =null、type="unregistered"——如实记录「无冻结能力记录」，不冒称
    protocol。effective 与 protocol 不同 → type=
    "restricted_context_treatment"、divergence=True（093 §4 语义）；
    相同 → type="protocol"。
    """
    if cap_gate_info is None:
        return {
            "protocol_cap": None,
            "cap_treatment": {
                "type": "unregistered",
                "divergence": None,
                "note": "模型未登记 model_context_caps.json（或注册表缺失），"
                        "无冻结 protocol cap——effective cap 即运行口径，"
                        "provenance 由本 manifest 承载",
            },
        }
    protocol = cap_gate_info.get("registry_protocol_cap")
    restricted = cap_gate_info.get(
        "registry_restricted_context_treatment")
    treatment = {
        "type": None,
        "divergence": None,
        "registry_protocol_cap": protocol,
        "registry_restricted_context_treatment": restricted,
        "capability": cap_gate_info.get("capability"),
        "max_position_embeddings": cap_gate_info.get(
            "max_position_embeddings"),
    }
    if protocol is None:
        treatment["type"] = "unregistered"
    elif effective_cap == protocol:
        treatment["type"] = "protocol"
        treatment["divergence"] = False
    else:
        treatment["type"] = "restricted_context_treatment"
        treatment["divergence"] = True
        treatment["note"] = (
            "effective cap 与注册表 protocol 值不同——按受限上下文 "
            "treatment 如实记录（启动门禁打印 [EXPERIMENT-CAP-"
            "DIVERGENCE]），不得表述为模型上下文上限")
        if restricted is not None and restricted == effective_cap:
            treatment["note"] += ("；与注册表 restricted_context_"
                                  "treatment 登记值一致")
    return {"protocol_cap": protocol, "cap_treatment": treatment}


def cap_identity_payload(model_name, dataset, dataset_fingerprint, method,
                         run_timestamp, effective_cap, protocol_cap,
                         cap_treatment_type, max_new_tokens,
                         input_identity=None, prompt_mode=None):
    """cap_identity hash 的输入子集（cap 进输出身份的核心）。

    刻意只含「run 口径」字段（模型/数据/method/时间戳/cap 三方/生成预算）
    ——不含 producer sha、每样本 stats、结果 sha：identity 用于写门比对
    「同一路径是否同一口径」，代码热修（producer sha 变）不应误伤同口径
    幂等重跑；cap/数据/时间戳任一不同则 identity 必不同（排序键 JSON→
    sha256 前 10 hex，与 076 treatment hash 同构造纪律）。
    """
    payload = {
        "schema": RUN_CAP_SCHEMA,
        "model_name": model_name,
        "dataset": dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "method": method,
        "run_timestamp": str(run_timestamp),
        "effective_cap": effective_cap,
        "protocol_cap": protocol_cap,
        "cap_treatment_type": cap_treatment_type,
        "max_new_tokens": max_new_tokens,
    }
    identity_contract = _input_identity_contract(input_identity)
    if prompt_mode is not None:
        if prompt_mode not in ('full', 'split'):
            raise ValueError('prompt_mode must be full or split')
        payload['prompt_mode'] = prompt_mode
    if identity_contract is not None:
        payload["input_identity"] = identity_contract
    return payload


def cap_identity_hash(**kwargs):
    payload = cap_identity_payload(**kwargs)
    return hashlib.sha256(_canonical(payload).encode("utf-8")
                          ).hexdigest()[:10]


# ---------------------------------------------------------------------------
# 【112 修复（GPT 审计 2026-10-12_0933 TL-E096-CAP-UNIFORMITY-112，P1
# 可用性阻断）】独立的 cap-comparability key——跨任务 cap 等价判断的正确
# 口径。
#
# 缺陷背景：cap_identity_payload 刻意包含 dataset/时间戳/max_new_tokens 等
# run 字段（写门「同一路径是否同一口径」需要它们），但 eval.py 的多任务
# 聚合单一性检查却错用了这个完整 run identity——不同数据集、同
# effective_cap=16384 的正常多任务聚合必然 hash 不同、被误判「混 cap」
# 拒绝（审计实测：strict 正常目录无法生成 result.json，可用性阻断）。
#
# 语义分工（两层身份，不得混用）：
#   - cap-comparability key（本函数）：只含可跨任务统一的 cap 语义字段
#     （effective/protocol/treatment 级）——多任务聚合「同口径」判断用它；
#   - cap_identity hash（完整 run identity）：含 dataset/时间戳等 run 字段
#     ——仅用于同一路径的防覆盖写门（gate_run_cap_identity），不参与跨
#     任务 cap 等价判断。
# ---------------------------------------------------------------------------
def cap_comparability_payload(effective_cap, protocol_cap,
                              cap_treatment_type, cap_treatment_divergence):
    """cap-comparability key 的输入子集（112）——刻意排除 dataset/
    fingerprint/method/时间戳/max_new_tokens 等因任务而异的 run 字段。"""
    return {
        "schema": RUN_CAP_SCHEMA,
        "effective_cap": effective_cap,
        "protocol_cap": protocol_cap,
        "cap_treatment_type": cap_treatment_type,
        "cap_treatment_divergence": cap_treatment_divergence,
    }


def cap_comparability_key(effective_cap, protocol_cap,
                          cap_treatment_type, cap_treatment_divergence):
    """跨任务 cap 等价 key（112）——排序键 JSON→sha256 前 10 hex，与
    cap_identity hash 同构造纪律。聚合侧比对用它；异 key = 异 cap 语义，
    不得进同一张 result.json 被当同一口径均值。"""
    payload = cap_comparability_payload(
        effective_cap=effective_cap, protocol_cap=protocol_cap,
        cap_treatment_type=cap_treatment_type,
        cap_treatment_divergence=cap_treatment_divergence)
    return hashlib.sha256(_canonical(payload).encode("utf-8")
                          ).hexdigest()[:10]


def build_run_cap_core(model_name, model_path, method, dataset,
                       dataset_fingerprint, run_timestamp, effective_cap,
                       cap_gate_info, max_new_tokens, producer,
                       treatment_type_hint=None, input_identity=None, prompt_mode=None):
    """run 级 manifest 核心字段（生成前可算——写门早期 CAP-CLASH 检测
    不必等整轮生成烧完 GPU 才发现口径冲突）。

    effective_cap 即本次 max_length（model2maxlen 活值，截断口径的
    单一事实源）；cap_gate_info = enforce_context_capability_gate 的
    返回值（096 修复的核心消费点：此前被 pred.py 丢弃）。
    treatment_type_hint：仅供 backfill 工具声明 "unknown"（连 stdout
    证据都没有的 provenance_incomplete 产物——不得冒称 unregistered/
    protocol/restricted，如实记未知）；生产路径（inline）不传。
    """
    summ = summarize_cap_gate(cap_gate_info, effective_cap)
    if treatment_type_hint is not None:
        summ = dict(summ)
        summ["cap_treatment"] = dict(summ["cap_treatment"])
        summ["cap_treatment"]["type"] = treatment_type_hint
    payload = cap_identity_payload(
        model_name=model_name, dataset=dataset,
        dataset_fingerprint=dataset_fingerprint, method=method,
        run_timestamp=run_timestamp, effective_cap=effective_cap,
        protocol_cap=summ["protocol_cap"],
        cap_treatment_type=summ["cap_treatment"]["type"],
        max_new_tokens=max_new_tokens, input_identity=input_identity, prompt_mode=prompt_mode)
    core = {
        "schema": RUN_CAP_SCHEMA,
        "provenance": "inline",
        "status": None,   # generating / complete（finalize 阶段定）
        "created_utc": _utcnow_iso(),
        "model_name": model_name,
        "model_path": (os.path.realpath(model_path)
                       if model_path else None),
        "producer": producer,
        "method": method,
        "dataset": dataset,
        "run_timestamp": str(run_timestamp),
        "dataset_fingerprint": dataset_fingerprint,
        # effective cap = 截断口径。source 显式声明取值来源（093 审计：
        # 「运行实现的显存上限」与「模型上下文上限」不得混同）。
        "effective_cap": effective_cap,
        "effective_cap_source": f"model2maxlen.json:{model_name}",
        "protocol_cap": summ["protocol_cap"],
        "cap_treatment": summ["cap_treatment"],
        "max_new_tokens": max_new_tokens,
        "cap_identity": {
            "hash": hashlib.sha256(_canonical(payload).encode("utf-8")
                                   ).hexdigest()[:10],
            "payload": payload,
        },
    }
    if input_identity is not None:
        core["input_identity"] = _input_identity_contract(input_identity)
    if prompt_mode is not None:
        core['prompt_mode'] = prompt_mode
    return core


def finalize_run_cap_manifest(core, status, n_samples, input_tokens,
                              truncated_flags, result_file, result_sha256,
                              result_lines, provenance="inline",
                              created_utc=None, input_ids_sha256=None):
    """组装完整 manifest（validate_run_cap_manifest 的消费口径）。

    input_tokens/truncated_flags：每样本列表（inline 生产路径）或 None
    （backfill：JSONL 里只有字符数 length，token 数不可复原——如实记
    null + note，不伪造）。status="generating" 表示数据写之前的崩溃窗
    口残留（result_sha256 必为 null）；"complete" = 结果文件 sha 已回填。
    """
    manifest = dict(core)
    manifest["status"] = status
    manifest["provenance"] = provenance
    manifest["created_utc"] = created_utc or _utcnow_iso()
    manifest["n_samples"] = n_samples
    if input_tokens is None:
        manifest["input_tokens"] = None
    else:
        toks = [int(t) for t in input_tokens]
        manifest["input_tokens"] = {
            "count": len(toks),
            "min": min(toks) if toks else None,
            "max": max(toks) if toks else None,
            "per_sample": toks,
        }
    if truncated_flags is None:
        manifest["truncated"] = None
    else:
        flags = [bool(b) for b in truncated_flags]
        manifest["truncated"] = {
            "count": sum(flags),
            "flags": flags,
        }
    manifest["result_file"] = result_file
    manifest["result_sha256"] = result_sha256
    manifest["result_lines"] = result_lines
    if "input_identity" in manifest:
        identity = dict(manifest["input_identity"])
        if input_ids_sha256 is None:
            identity["actual_input_ids"] = None
        else:
            hashes = [str(value) for value in input_ids_sha256]
            identity["actual_input_ids"] = {
                "encoding": INPUT_IDS_ENCODING,
                "count": len(hashes),
                "per_sample_sha256": hashes,
                "ordered_sha256": _ordered_digest_hash(hashes),
            }
        manifest["input_identity"] = identity
    validate_run_cap_manifest(manifest)
    return manifest


def validate_run_cap_manifest(manifest):
    """消费/补制侧统一 schema 门（缺字段 fail-closed）。

    必填键缺一 / schema 不符 / provenance 非法 / 状态与绑定语义矛盾 →
    [CAP-SCHEMA-FAIL] SystemExit（消息走 e.code）。允许的 null 值口径：
      - protocol_cap=null        ：unregistered 模型（键必须在）；
      - effective_cap=null       ：仅 provenance_incomplete（查无证据）；
      - input_tokens/truncated=null：仅非 inline（backfill 不可复原）；
      - result_sha256=null       ：仅 status=generating 或
                                   provenance_incomplete（缺日志补制时
                                   结果存在但 run 口径无证据——sha 仍会
                                   回填，见 backfill 工具；此处允许 null
                                   是为了补制半残产物的如实登记）。
    inline+complete 最强约束：per_sample 长度 == n_samples == 行数。
    """
    def _fail(msg):
        raise SystemExit(f"[CAP-SCHEMA-FAIL] 096: {msg}——run-cap manifest "
                         f"不完整/不自洽，拒绝消费，fail loudly")
    if not isinstance(manifest, dict):
        _fail(f"manifest 非对象（{type(manifest).__name__}）")
    for k in _REQUIRED_KEYS:
        if k not in manifest:
            _fail(f"缺必填键 {k!r}")
    if manifest["schema"] != RUN_CAP_SCHEMA:
        _fail(f"schema={manifest['schema']!r} != {RUN_CAP_SCHEMA!r}")
    if manifest["provenance"] not in RUN_CAP_PROVENANCES:
        _fail(f"provenance={manifest['provenance']!r} 非法"
              f"（合法：{RUN_CAP_PROVENANCES}）")
    if manifest["status"] not in ("generating", "complete"):
        _fail(f"status={manifest['status']!r} 非法")
    ct = manifest["cap_treatment"]
    if not isinstance(ct, dict) or "type" not in ct:
        _fail("cap_treatment 须为含 type 键的对象")
    ci = manifest["cap_identity"]
    if not isinstance(ci, dict) or not ci.get("hash"):
        _fail("cap_identity 须为含 hash 的对象（cap 进输出身份）")
    # 【115 修复（GPT 审计 2026-10-12_0933 TL-E096-CAP-SCHEMA-INTEGRITY-
    # 115，P1）】schema 门此前只查 identity hash 非空——不重算 payload
    # hash、不核对 payload↔外层字段闭包。审计红证：合法 manifest 仅改外
    # 层 effective_cap 16384→31500 / result_file→other.jsonl（保留原
    # payload/hash），validate 仍返回成功——误拷、手改或错误 backfill 的
    # sidecar 可带错 cap 身份静默进入结果元数据。现做两层核对：
    #   ① 重算 canonical payload hash == cap_identity.hash（payload 单侧
    #      篡改拦截）；
    #   ② payload 覆盖的每个字段与外层逐字段相等（外层单侧篡改拦截——
    #      payload↔外层闭包）。result_file↔结果 basename 的核对在消费
    #      侧（eval.validate_runcap_sidecar——validate 层不知 jsonl 路径）。
    payload = ci.get("payload")
    if not isinstance(payload, dict):
        _fail("cap_identity.payload 缺失/非对象——identity hash 不可复核")
    recomputed = hashlib.sha256(
        _canonical(payload).encode("utf-8")).hexdigest()[:10]
    if recomputed != ci["hash"]:
        _fail(f"cap_identity.hash={ci['hash']!r} != payload 重算 "
              f"{recomputed!r}——payload 与 hash 不自洽（payload 被篡改/"
              f"损坏，115 fail-closed）")
    _closure = {
        "schema": manifest["schema"],
        "model_name": manifest["model_name"],
        "dataset": manifest["dataset"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "method": manifest["method"],
        "run_timestamp": manifest["run_timestamp"],
        "effective_cap": manifest["effective_cap"],
        "protocol_cap": manifest["protocol_cap"],
        "max_new_tokens": manifest["max_new_tokens"],
    }
    for _pk, _outer in _closure.items():
        if payload.get(_pk) != _outer:
            _fail(f"cap_identity.payload.{_pk}={payload.get(_pk)!r} != "
                  f"外层 {_pk}={_outer!r}——payload 与外层字段不自洽"
                  f"（外层被篡改/错配，115 fail-closed）")
    if payload.get("cap_treatment_type") != \
            manifest["cap_treatment"].get("type"):
        _fail(f"payload.cap_treatment_type="
              f"{payload.get('cap_treatment_type')!r} != 外层 "
              f"cap_treatment.type="
              f"{manifest['cap_treatment'].get('type')!r}（115 fail-closed）")
    payload_identity = payload.get("input_identity")
    if (payload.get('prompt_mode') != manifest.get('prompt_mode')
            or manifest.get('prompt_mode') not in (None, 'full', 'split')):
        _fail('prompt_mode invalid or not bound to cap identity')
    manifest_identity = manifest.get("input_identity")
    if (payload_identity is None) != (manifest_identity is None):
        _fail("input_identity 在 cap payload 与外层 manifest 间仅单侧存在")
    if manifest_identity is not None:
        if not isinstance(manifest_identity, dict):
            _fail("input_identity 须为对象")
        contract = _input_identity_contract(manifest_identity)
        if payload_identity != contract:
            _fail("cap payload.input_identity 与外层身份合同不一致")
        if contract["schema"] != INPUT_IDENTITY_SCHEMA:
            _fail(f"input_identity.schema={contract['schema']!r} 非法")
        if contract["canonicalization"] != INPUT_IDENTITY_CANONICALIZATION:
            _fail("input_identity canonicalization 非法")
        source_sha = contract["source_sha256"]
        prompt_sha = contract["prompt_template_sha256"]
        if (not isinstance(source_sha, str) or len(source_sha) != 64
                or any(c not in "0123456789abcdef" for c in source_sha)):
            _fail("input_identity.source_sha256 不是小写 SHA-256")
        if contract["source_fingerprint"] != "v2-" + source_sha:
            _fail("input_identity source_fingerprint 未绑定 source_sha256")
        if manifest["dataset_fingerprint"] != contract["source_fingerprint"]:
            _fail("dataset_fingerprint 与 v2 source_fingerprint 不一致")
        if (not isinstance(prompt_sha, str) or len(prompt_sha) != 64
                or any(c not in "0123456789abcdef" for c in prompt_sha)):
            _fail("input_identity.prompt_template_sha256 不是小写 SHA-256")
        sample_count = contract["sample_count"]
        if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
                or sample_count < 0 or sample_count != manifest["n_samples"]):
            _fail("input_identity.sample_count 必须等于 n_samples")
        actual = manifest_identity.get("actual_input_ids")
        if (manifest.get("provenance") == "inline"
                and not isinstance(actual, dict)):
            _fail("v2 inline manifest 必须记录 actual_input_ids")
        if actual is not None:
            if not isinstance(actual, dict):
                _fail("actual_input_ids 须为对象或 null")
            if actual.get("encoding") != INPUT_IDS_ENCODING:
                _fail("actual_input_ids encoding 非法")
            hashes = actual.get("per_sample_sha256")
            if not isinstance(hashes, list):
                _fail("actual_input_ids.per_sample_sha256 须为列表")
            if actual.get("count") != len(hashes) \
                    or len(hashes) != manifest["n_samples"]:
                _fail("actual_input_ids 数量必须等于 n_samples")
            if any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in hashes):
                _fail("actual_input_ids 含非法 SHA-256")
            if actual.get("ordered_sha256") != _ordered_digest_hash(hashes):
                _fail("actual_input_ids ordered_sha256 与逐样本列表不一致")
    prov = manifest["provenance"]
    if manifest["effective_cap"] is None and prov != "provenance_incomplete":
        _fail("effective_cap=null 仅允许 provenance_incomplete")
    if manifest["input_tokens"] is None and prov == "inline":
        _fail("input_tokens=null 仅允许非 inline provenance（生产路径必须"
              "记录每样本实际输入 token 数）")
    if manifest["truncated"] is None and prov == "inline":
        _fail("truncated=null 仅允许非 inline provenance")
    if manifest["status"] == "generating":
        if manifest["result_sha256"] is not None:
            _fail("status=generating 但 result_sha256 非 null（生成完成"
                  "回填前数据不应有终态绑定）")
    if manifest["status"] == "complete":
        if (prov == "inline" and manifest["result_sha256"] is None):
            _fail("inline+complete 但 result_sha256 缺失（生成完成后必须"
                  "回填结果文件 sha256）")
        if prov == "inline":
            it = manifest["input_tokens"]
            if not isinstance(it, dict) or \
                    len(it.get("per_sample", [])) != manifest["n_samples"]:
                _fail("inline+complete 要求 input_tokens.per_sample 长度 =="
                      " n_samples")
            if manifest["n_samples"] != manifest["result_lines"]:
                _fail(f"inline+complete 要求 n_samples({manifest['n_samples']})"
                      f" == result_lines({manifest['result_lines']})")
    return True


def write_run_cap_manifest(sidecar_path, manifest):
    """原子落盘（tmp + os.replace）——读侧永不见半文件。"""
    d = os.path.dirname(os.path.abspath(sidecar_path))
    tmp = os.path.join(d, f".{os.path.basename(sidecar_path)}.tmp-"
                         f"{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_canonical(manifest))
    os.replace(tmp, sidecar_path)


def read_run_cap_manifest(sidecar_path):
    """读 + schema 门：损坏/不合法 → [CAP-SCHEMA-FAIL] fail-closed。"""
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"[CAP-SCHEMA-FAIL] 096: run-cap manifest 读取/解析失败"
            f"（{sidecar_path} → {type(e).__name__}: {e}）——provenance "
            f"未知，拒绝覆盖，fail loudly")
    validate_run_cap_manifest(manifest)
    return manifest


def gate_run_cap_identity(out_path, core_or_identity):
    """096 fail-closed 写门（076 gate_output_treatment_identity 的 cap 层
    对偶）。

    比对口径 = cap_identity.hash + effective_cap 双保险（hash 碰撞/手改
    manifest 时裸值直比仍拦得住）。三类拒绝：
      ① 已存在 manifest 且 cap/identity 不同 → [CAP-CLASH]（审计建议第
         2 条：同 task/method/时间戳不同 cap 不得覆盖——16K 与 31.5K 产物
         从此路径级可分）；
      ② 已存在 manifest 但损坏/schema 不合法 → [CAP-SCHEMA-FAIL]（经
         read_run_cap_manifest）；
      ③ 数据文件存在但 manifest 缺失（096 前存量产物/手工放置）→ 不拒
         绝（076 对非 tli 的既有覆盖语义保持），但打
         [CAP-PROVENANCE-MISSING] 警告并由本次运行重建 provenance——
         存量 5 个非 chat 格 + .24 在飞批走 backfill 工具补制，不靠本门。
    无 manifest + 无数据 → 放行（新产物）。
    返回 sidecar 路径。
    """
    sidecar = out_path + RUN_CAP_SIDECAR_SUFFIX
    if isinstance(core_or_identity, dict) and \
            "cap_identity" in core_or_identity:
        cur_hash = core_or_identity["cap_identity"]["hash"]
        cur_cap = core_or_identity.get("effective_cap")
    else:
        cur_hash = core_or_identity["hash"]
        cur_cap = core_or_identity.get("effective_cap")
    if os.path.lexists(sidecar):
        existing = read_run_cap_manifest(sidecar)
        ex_hash = existing["cap_identity"]["hash"]
        ex_cap = existing["effective_cap"]
        if ex_hash != cur_hash or ex_cap != cur_cap:
            raise SystemExit(
                f"[CAP-CLASH] 096: 输出目标 {out_path} 已存在 run-cap "
                f"manifest 且 cap 口径不同——existing cap={ex_cap} "
                f"(identity {ex_hash}) vs current cap={cur_cap} "
                f"(identity {cur_hash})。同 task/method/时间戳不同 cap "
                f"不得覆盖（16K 受限 treatment 与 31.5K 协议口径产物必须"
                f"路径级可分），拒绝运行，fail loudly")
    elif os.path.lexists(out_path):
        print(f"[CAP-PROVENANCE-MISSING] 096: 输出目标 {out_path} 已存在"
              f"但无 run-cap manifest（096 前存量产物/手工放置）——本运行"
              f"将重建 provenance；存量产物请用 "
              f"exp/trace/backfill_run_cap_manifest.py 补制",
              file=sys.stderr)
    return sidecar


@contextlib.contextmanager
def runcap_gate_lock(out_path):
    """【103 修复（GPT 审计 2026-10-12_0739 TL-E096-CAP-GATE-TOCTOU-103，
    P1）】gate 与发布之间的跨进程独占锁——消灭 TOCTOU 交错窗口。

    缺陷背景：两个进程可在 sidecar 尚不存在时同时通过
    gate_run_cap_identity()（gate 读与 manifest 写之间无锁），随后各自
    长时间生成、交错覆盖数据与 manifest——受控交错可得到「数据来自 B、
    complete manifest 来自 A」的错配产物，而 read_run_cap_manifest 只做
    schema 检查仍会放行（单文件 os.replace 不提供跨产物事务）。

    语义：锁文件 = out_path + ".runcap_gate.lock"（**结果文件级**——
    同一 out_path 串行化；同目录不同结果文件不互斥。2026-10-12 粒度
    修正：首版目录级锁的「生产调度本就是串行 bash 链」前提被生产打
    破——本机 fullkv 双任务同目录并发时后到者持 62GB 已加载显存空等
    锁数小时；GPT 103 威胁模型是同一输出路径交错，不同结果文件无共
    享写目标，目录级属过度串行）。持有窗口 = 调用方 with 块全程
    （gate → generating manifest → 数据写 → complete manifest）。锁内
    由调用方重新执行 gate_run_cap_identity 即 double-check（锁外先过
    的门不算数——后到者在锁内重读身份，先到者已写出的 manifest 会被
    CAP-CLASH 拒绝，或同 cap 幂等放行）。

    死锁纪律：try/finally 保证异常路径必释放（095 guard 在生成段 raise
    传播时锁不滞留）；锁文件只创建**不删除**（POSIX 锁绑定 inode，删了
    再建会让并发方锁到不同 inode 上、互斥失效）。
    """
    out_path = os.path.abspath(out_path)
    lock_dir = os.path.dirname(out_path)
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = out_path + RUN_CAP_LOCK_SUFFIX
    lock_file = open(lock_path, "a")   # 只创建不删除（见 docstring）
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def compute_result_binding(out_path):
    """生成完成后回填：结果文件 sha256 + 行数。"""
    return _sha256_file(out_path), sum(
        1 for _ in open(out_path, "rb"))
