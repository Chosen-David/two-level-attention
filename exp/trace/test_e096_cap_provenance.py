# -*- coding: utf-8 -*-
"""TL-DSV2-CAP-PROVENANCE-096 修复门禁测试（GPT 审计
2026-10-12_0527_twolevel_bug_audit_08819a7 §096，P1）。

红绿复现矩阵：
  红证（PRE_FIX 固定 SHA 锚点 6e918c5，不引用可变 main: ——锚点漂移教训）：
    R1  gate 返回值被丢弃：PRE_FIX pred.py 的 enforce_context_capability_gate
        调用是裸表达式（ast.Expr，非 Assign）——096 审计 P1 原话
    R2  fullkv 无任何 provenance 旁挂：PRE_FIX 版 sidecar 写仅在
        args.method == "tli" 分支内；无 run-cap manifest 概念
    R3  每行 JSONL 只记数据集原字段：PRE_FIX get_pred 无 run_stats 参数、
        无截断标记、无实际输入 token 数采集
    R4  产物不可分（缺陷行为本质）：PRE_FIX 版 out_fn 表达式不含任何 cap
        变量 + record 键集合不含 cap 字段 → 同 task/method/时间戳不同 cap
        的两次 run 产物路径与内容 schema 完全一致（16K/31.5K 互覆不可检）
  绿证（修复后）：
    G1  cap 进输出身份：同 task/method/时间戳仅改 cap 16384↔31500 →
        cap_identity hash 必不同（method=none 与 tli 双覆盖）
    G2  写门 fail-closed：同路径已存在不同 cap manifest → [CAP-CLASH]
        SystemExit；同 cap 幂等放行；sidecar 损坏 → [CAP-SCHEMA-FAIL]
    G3  schema 完整性：缺必填键 / inline+complete 缺 result_sha256 /
        per_sample 长度 ≠ n_samples / n_samples ≠ result_lines 一律
        [CAP-SCHEMA-FAIL] fail-closed
    G4  T21 反向门禁（096 建议第 3 条）：fullkv + tli 双路径，同
        task/method/ts 仅改 cap → 身份不同 + 写门拒绝；且 081 层不受污染
        （treatment manifest 对两 cap 逐字节相同——两层关系契约）
    G5  gate 返回值消费 + 注册表三键（registry_protocol_cap /
        registry_restricted_context_treatment / registered）；cap_treatment
        分派：16384→restricted_context_treatment（divergence=True）、
        31500→protocol；unregistered（cap_gate_info=None）→protocol_cap
        null 如实记录
    G6  pred.py 集成静态：gate 调用被接收（Assign cap_gate_info）；runcap
        写门/落盘不在 method=="tli" 分支内（全 method 覆盖）；写门在数据
        写之前、complete 回填在数据写之后
    G7  backfill 双 provenance 口径：有日志 → backfilled-from-stdout
        （cap 从 [EXPERIMENT-CAP-DIVERGENCE]/gate 打印解析）；缺日志 →
        provenance_incomplete；已存在 inline manifest → 拒绝覆盖；
        已存在 backfilled → 须显式 --overwrite-backfill
    G8  RULER 入口判断：pred_ruler.py 已有等价 cap 绑定（context_length
        进输出路径 L{N}/ 与 yarn receipt）——登记不接入，静态证明

环境：纯标准库 + info.py（treatment 层）需 torch（CPU 即可）。真实
DeepSeek-V2-Lite-Chat 快照只读 config/tokenizer 元数据（无模型加载）。
python 与 python -O 双跑（无 assert 依赖）。
"""
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PRED_FP = os.path.join(REPO, "benchmark", "LongBench", "pred.py")
GATE_FP = os.path.join(REPO, "benchmark", "LongBench",
                       "context_capability_gate.py")
RCM_FP = os.path.join(REPO, "benchmark", "LongBench", "run_cap_manifest.py")
BACKFILL_FP = os.path.join(REPO, "exp", "trace",
                           "backfill_run_cap_manifest.py")
RULER_FP = os.path.join(REPO, "benchmark", "RULER", "pred_ruler.py")
DSV2_SNAPSHOT = ("/home/wangyuanshuo02/models/huggingface.co/"
                 "deepseek-ai/DeepSeek-V2-Lite-Chat")
CONFIG_DIR = os.path.join(REPO, "benchmark", "LongBench", "config")

# 红证锚点固定为修复前最后 commit（6e918c5 = 096 修复合入前 HEAD）——
# 修复合入后 main: 引用会漂移到新版使红证失效（历史教训），必须钉 SHA。
PRE_FIX = "6e918c53d6353b5d7c289c5a712b2bb296cb6f45"

PASS_N = 0


def _check(cond, msg):
    global PASS_N
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    PASS_N += 1
    print(f"[PASS] {msg}")


def _sh(cmd):
    return subprocess.check_output(cmd, cwd=REPO, text=True)


def _get_main_body(src):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If):
            test = ast.unparse(node.test).replace('"', "'")
            if test == "__name__ == '__main__'":
                return node
    return None


def _call_name(node):
    try:
        return ast.unparse(node.func)
    except Exception:
        return ""


def _sys_exit_code(fn, *a, **kw):
    """捕获 SystemExit，返回 (code, stdout)。"""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            fn(*a, **kw)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


# ------------------------------------------------------------------ 红证
def test_red():
    print("== TL-DSV2-CAP-PROVENANCE-096 红证（PRE_FIX="
          f"{PRE_FIX[:7]}）==")
    old_src = _sh(["git", "show", f"{PRE_FIX}:benchmark/LongBench/pred.py"])

    # R1：gate 返回值被丢弃（裸表达式调用）
    main_body = _get_main_body(old_src)
    _check(main_body is not None, "R0 PRE_FIX __main__ 块可定位")
    bare_calls = [n for n in ast.walk(main_body)
                  if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)
                  and _call_name(n.value) == "enforce_context_capability_gate"]
    _check(len(bare_calls) == 1,
           f"R1-RED PRE_FIX gate 调用是裸表达式（返回值被丢弃，恰 1 处："
           f"{len(bare_calls)}）——096 审计 P1 原话复现")

    # R2：fullkv（method=none）无 provenance 旁挂——sidecar 写门仅在 tli 分支
    # （__main__ 里有两处 args.method=='tli' 判断：081 snapshot 解析 + 076
    # sidecar 写门——此处只锚 sidecar 写门分支）
    tli_branches = [n for n in ast.walk(main_body)
                    if isinstance(n, ast.If)
                    and "args.method == 'tli'" in ast.unparse(n.test)
                    and any(isinstance(c, ast.Call)
                            and _call_name(c) in (
                                "gate_output_treatment_identity",
                                "get_treatment_manifest_json")
                            for c in ast.walk(n))]
    _check(len(tli_branches) == 1,
           "R2a-RED PRE_FIX sidecar 写门仅存在于 args.method=='tli' 分支"
           f"（{len(tli_branches)} 处）——method=none/fullkv 无任何 "
           f"provenance 落盘")
    has_runcap_in_tli = any(
        "manifest" in ast.unparse(n) for n in ast.walk(tli_branches[0]))
    sidecar_outside = [n for n in ast.walk(main_body)
                       if isinstance(n, (ast.With, ast.Call))
                       and "tli_manifest" in ast.unparse(n)]
    _check(has_runcap_in_tli and all(
               _in_node(tli_branches[0], n) for n in sidecar_outside),
           "R2b-RED PRE_FIX 全部 manifest/sidecar 写都在 tli 分支内"
           "（none 臂零 provenance——096 审计第 4 条）")

    # R3：get_pred 不采集 cap 相关信息
    _check("run_stats" not in old_src and "truncated_flag" not in old_src
           and "input_tokens" not in old_src,
           "R3-RED PRE_FIX get_pred 无 run_stats/截断标记/实际输入 token "
           "采集——每行 JSONL 只记数据集原字段 length（096 审计第 2 条）")

    # R4：产物不可分——out_fn 不含 cap + record 不含 cap 字段
    fn_assigns = [n for n in ast.walk(main_body)
                  if isinstance(n, ast.Assign)
                  and ast.unparse(n).startswith("out_fn = ")]
    joined = " ".join(ast.unparse(a) for a in fn_assigns)
    _check("cap" not in joined.replace("capability", "").lower()
           and "max_length" not in joined,
           f"R4a-RED PRE_FIX out_fn 表达式不含任何 cap/max_length 变量"
           f"（{joined[:70]}…）——cap 不在输出文件身份中")
    tree = ast.parse(old_src)
    record_keys = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Dict)
                and any(getattr(k, "value", None) == "pred"
                        for k in node.value.keys)):
            record_keys = {getattr(k, "value", None)
                           for k in node.value.keys}
    _check(record_keys is not None
           and not ({"max_length", "input_tokens", "truncated"}
                    & record_keys),
           f"R4b-RED PRE_FIX record 键集合 {sorted(record_keys)} 不含"
           f" max_length/input_tokens/truncated——同 task/method/时间戳"
           f"不同 cap 产物路径与内容 schema 完全一致（16K/31.5K 互覆"
           f"不可检，096 审计缺陷本质）")


def _in_node(outer, inner):
    """inner 节点是否位于 outer 的行号范围内。"""
    return (outer.lineno <= getattr(inner, "lineno", -1)
            <= getattr(outer, "end_lineno", -1))


# ------------------------------------------------------------------ 绿证
def _mk_core(method="none", cap=16384, task="hotpotqa", ts="1012024000",
             gate_info=None, max_new=32):
    from benchmark.LongBench.run_cap_manifest import build_run_cap_core
    return build_run_cap_core(
        model_name="DeepSeek-V2-Lite-Chat", model_path=None, method=method,
        dataset=task, dataset_fingerprint="abc123def456",
        run_timestamp=ts, effective_cap=cap, cap_gate_info=gate_info,
        max_new_tokens=max_new, producer={"script_sha256": "deadbeef"})


def _mk_complete(core, n=2):
    from benchmark.LongBench.run_cap_manifest import (
        finalize_run_cap_manifest)
    return finalize_run_cap_manifest(
        core, status="complete", n_samples=n,
        input_tokens=[100 + i for i in range(n)],
        truncated_flags=[i % 2 == 0 for i in range(n)],
        result_file="hotpotqa-none-1012024000.jsonl",
        result_sha256="0" * 64, result_lines=n)


def test_green_identity_and_gate():
    print("== 096 绿证：cap 进输出身份 + 写门 ==")
    from benchmark.LongBench.run_cap_manifest import (
        RUN_CAP_SIDECAR_SUFFIX, finalize_run_cap_manifest,
        gate_run_cap_identity, read_run_cap_manifest,
        write_run_cap_manifest)

    # G1：同 task/method/ts 仅改 cap → identity 必不同（none + tli 双覆盖）
    for method in ("none", "tli"):
        c16, c315 = _mk_core(method=method, cap=16384), \
            _mk_core(method=method, cap=31500)
        _check(c16["cap_identity"]["hash"] != c315["cap_identity"]["hash"]
               and c16["cap_identity"]["hash"] != ""
               and c315["cap_identity"]["hash"] != "",
               f"G1 {method=} cap 16384↔31500 → cap_identity hash 不同"
               f"（{c16['cap_identity']['hash']} vs "
               f"{c315['cap_identity']['hash']}）——cap 进输出身份")
    # 同 cap 幂等：重复构造逐位稳定
    _check(_mk_core()["cap_identity"] == _mk_core()["cap_identity"],
           "G1b 同 cap 重复构造 identity 逐位稳定（幂等重跑可判同）")

    # G2：写门——不同 cap 同路径 CAP-CLASH；同 cap 幂等放行
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        core16 = _mk_core(cap=16384)
        sidecar = gate_run_cap_identity(out_path, core16)
        _check(sidecar.endswith(RUN_CAP_SIDECAR_SUFFIX),
               "G2a 无 manifest 无数据 → 放行并返回 sidecar 路径")
        write_run_cap_manifest(sidecar, _mk_complete(core16))
        code, out = _sys_exit_code(
            gate_run_cap_identity, out_path, _mk_core(cap=31500))
        _check(code is not None and "[CAP-CLASH]" in str(code)
               and "16384" in str(code) and "31500" in str(code),
               f"G2b-RED→GREEN 同路径不同 cap → [CAP-CLASH] SystemExit"
               f"（{str(code)[:70]}…）——16K 与 31.5K 产物路径级可分")
        side2 = gate_run_cap_identity(out_path, core16)
        _check(side2 == sidecar,
               "G2c 同 cap 幂等放行（SKIP 语义：同口径重跑不拒）")
        # sidecar 损坏 → fail-closed
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("{not-json")
        code, out = _sys_exit_code(
            gate_run_cap_identity, out_path, _mk_core(cap=16384))
        _check(code is not None and "[CAP-SCHEMA-FAIL]" in str(code),
               "G2d sidecar 损坏 → [CAP-SCHEMA-FAIL] 拒绝（provenance "
               "未知不放行，即便同 cap）")
        # 数据存在但 manifest 缺失（096 前存量产物）→ 放行 + 警告
        os.remove(sidecar)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("legacy\n")
        io_out = io.StringIO()
        with contextlib.redirect_stderr(io_out):
            gate_run_cap_identity(out_path, _mk_core(cap=16384))
        _check("[CAP-PROVENANCE-MISSING]" in io_out.getvalue(),
               "G2e 存量产物无 manifest → 放行 + [CAP-PROVENANCE-"
               "MISSING] 警告（076 非 tli 覆盖语义保持，backfill 补制）")

    # G3：schema 完整性 fail-closed
    from benchmark.LongBench.run_cap_manifest import (
        validate_run_cap_manifest)
    good = _mk_complete(_mk_core())
    code, _ = _sys_exit_code(validate_run_cap_manifest, good)
    _check(code is None, "G3a 完整 inline manifest 通过 schema 门")
    bad_cases = []
    for key in ("cap_identity", "effective_cap", "result_sha256",
                "cap_treatment", "provenance", "input_tokens"):
        m = dict(good)
        m.pop(key)
        bad_cases.append((f"缺键 {key}", m))
    m = dict(good); m["schema"] = "run-cap-manifest-v0"; bad_cases.append(
        ("schema 版本不符", m))
    m = dict(good); m["result_sha256"] = None; bad_cases.append(
        ("inline+complete 缺 result_sha256", m))
    m = _mk_complete(_mk_core(), n=3)
    m["input_tokens"]["per_sample"] = m["input_tokens"]["per_sample"][:2]
    bad_cases.append(("per_sample 长度≠n_samples", m))
    m = _mk_complete(_mk_core(), n=3)
    m["result_lines"] = 2; bad_cases.append(("n_samples≠result_lines", m))
    m = dict(good); m["effective_cap"] = None; bad_cases.append(
        ("effective_cap=null 但非 incomplete", m))
    n_fail = 0
    for name, m in bad_cases:
        code, _ = _sys_exit_code(validate_run_cap_manifest, m)
        if code is not None and "[CAP-SCHEMA-FAIL]" in str(code):
            n_fail += 1
    _check(n_fail == len(bad_cases),
           f"G3b {len(bad_cases)} 类残缺/不自洽 manifest 全部 "
           f"[CAP-SCHEMA-FAIL]（{n_fail}/{len(bad_cases)}）——缺字段 "
           f"fail-closed")

    # 读门同源：read_run_cap_manifest 对合法文件过、对缺键文件拒
    with tempfile.TemporaryDirectory() as td:
        sc = os.path.join(td, "x.runcap.json")
        write_run_cap_manifest(sc, good)
        code, _ = _sys_exit_code(read_run_cap_manifest, sc)
        _check(code is None, "G3c 合法落盘 manifest 读回通过")
        m = dict(good); m.pop("protocol_cap")
        write_run_cap_manifest(sc, m)
        code, _ = _sys_exit_code(read_run_cap_manifest, sc)
        _check(code is not None and "[CAP-SCHEMA-FAIL]" in str(code),
               "G3d protocol_cap 键缺失（值可为 null 但键必须在）→ 拒")


def test_green_t21_reverse():
    print("== 096 绿证：T21 反向门禁（fullkv + tli 双路径）==")
    from benchmark.LongBench.run_cap_manifest import (
        gate_run_cap_identity, write_run_cap_manifest)

    for method in ("none", "tli"):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, f"hotpotqa-{method}-1012024000.jsonl")
            core16 = _mk_core(method=method, cap=16384)
            core315 = _mk_core(method=method, cap=31500)
            write_run_cap_manifest(
                gate_run_cap_identity(out_path, core16), _mk_complete(core16))
            code, _ = _sys_exit_code(
                gate_run_cap_identity, out_path, core315)
            _check(code is not None and "[CAP-CLASH]" in str(code),
                   f"G4 {method=} 同 task/method/时间戳仅改 cap "
                   f"16384→31500 → 写门拒绝（[CAP-CLASH]，身份 "
                   f"{core16['cap_identity']['hash']} vs "
                   f"{core315['cap_identity']['hash']}）")
            # 换新路径（新时间戳）则两 cap 可各自落盘——身份可分而非全局互斥
            p2 = os.path.join(td, f"hotpotqa-{method}-1012024001.jsonl")
            gate_run_cap_identity(p2, core315)
            _check(os.path.lexists(p2 + ".runcap.json") is False,
                   f"G4b {method=} 新路径放行（不同 cap 产物各自成档，"
                   f"不互斥）")

    # 081 层不受污染：treatment manifest 对两 cap 逐字节相同（两层关系契约）
    types = SimpleNamespaceLike(method="tli")
    from sparse_attn.info import (get_treatment_manifest_json,
                                  resolve_treatment_snapshot)
    snap = resolve_treatment_snapshot(types)
    m16 = get_treatment_manifest_json(types, snap)
    m315 = get_treatment_manifest_json(types, snap)
    _check(m16 == m315 and "model2maxlen" not in m16,
           "G4c 081 treatment manifest 与 cap 解耦保持（两 cap 同 treatment"
           " 逐字节相同且不含 model2maxlen）——096 是新增独立层，未把 cap "
           "塞进 info.py（原 T21 门禁语义不动）")


class SimpleNamespaceLike:
    """tli 臂 args 桩：treatment 字段全部走 getattr 缺省（与
    TLIIndexer.__init__ 运行时缺省同源）。"""

    def __init__(self, method="tli"):
        self.method = method


def test_green_gate_info():
    print("== 096 绿证：gate 返回值消费 + 注册表三键 ==")
    from benchmark.LongBench.context_capability_gate import (
        enforce_context_capability_gate)
    from benchmark.LongBench.run_cap_manifest import (
        summarize_cap_gate)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        info16 = enforce_context_capability_gate(
            "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 16384, CONFIG_DIR)
    _check(info16 is not None
           and info16["registry_protocol_cap"] == 31500
           and info16["registry_restricted_context_treatment"] == 16384
           and info16["registered"] is True,
           "G5a gate 返回值新增注册表三键（protocol 31500 / restricted "
           "16384 / registered）")
    summ16 = summarize_cap_gate(info16, 16384)
    _check(summ16["protocol_cap"] == 31500
           and summ16["cap_treatment"]["type"] == "restricted_context_"
           "treatment"
           and summ16["cap_treatment"]["divergence"] is True,
           "G5b cap=16384 → cap_treatment=restricted_context_treatment"
           "（divergence=True，与注册表 restricted 登记值一致）")
    with contextlib.redirect_stdout(io.StringIO()):
        info315 = enforce_context_capability_gate(
            "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 31500, CONFIG_DIR)
    summ315 = summarize_cap_gate(info315, 31500)
    _check(summ315["cap_treatment"]["type"] == "protocol"
           and summ315["cap_treatment"]["divergence"] is False,
           "G5c cap=31500（协议值）→ cap_treatment=protocol")
    summ_un = summarize_cap_gate(None, 31500)
    _check(summ_un["protocol_cap"] is None
           and summ_un["cap_treatment"]["type"] == "unregistered",
           "G5d 未登记模型（gate 返回 None）→ protocol_cap=null + "
           "unregistered 如实记录（不冒称）")
    # 端到端：build_run_cap_core 消费真实 gate 返回值
    core = _mk_core(gate_info=info16)
    _check(core["protocol_cap"] == 31500
           and core["effective_cap"] == 16384
           and core["cap_treatment"]["type"] == "restricted_context_"
           "treatment"
           and core["effective_cap_source"] ==
           "model2maxlen.json:DeepSeek-V2-Lite-Chat",
           "G5e build_run_cap_core 端到端消费 gate 返回值（protocol 31500 +"
           " effective 16384 + restricted treatment + source 注记）")


def test_green_pred_integration():
    print("== 096 绿证：pred.py 集成静态 ==")
    src = open(PRED_FP, encoding="utf-8").read()
    main_body = _get_main_body(src)
    _check(main_body is not None, "G0 pred.py __main__ 块可定位")

    # gate 调用被接收（Assign）且消费（cap_gate_info 进 build_run_cap_core）
    assigns = [n for n in ast.walk(main_body)
               if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and _call_name(n.value) == "enforce_context_capability_gate"]
    _check(len(assigns) == 1
           and [t.id for t in assigns[0].targets] == ["cap_gate_info"],
           "G6a gate 返回值被接收（cap_gate_info = enforce…，修复前丢弃）")
    core_calls = [n for n in ast.walk(main_body)
                  if isinstance(n, ast.Call)
                  and _call_name(n) == "build_run_cap_core"]
    _check(len(core_calls) == 1
           and any(ast.unparse(k) == "cap_gate_info=cap_gate_info"
                   for k in core_calls[0].keywords),
           "G6b cap_gate_info 被消费进 run-cap manifest（build_run_cap_"
           "core kwargs 实证）")

    # runcap 写门/落盘不在 method=='tli' 分支内（全 method 覆盖 fullkv）。
    # __main__ 有两处 args.method=='tli' 判断（081 snapshot 解析 + 076
    # sidecar 写门），锚定含 gate_output_treatment_identity 的 sidecar 分支
    tli_sidecar = [n for n in ast.walk(main_body)
                   if isinstance(n, ast.If)
                   and "args.method == 'tli'" in ast.unparse(n.test)
                   and any(isinstance(c, ast.Call)
                           and _call_name(c) == "gate_output_treatment_"
                           "identity" for c in ast.walk(n))]
    runcap_nodes = [n for n in ast.walk(main_body)
                    if isinstance(n, ast.Call)
                    and _call_name(n) in (
                        "build_run_cap_core", "gate_run_cap_identity",
                        "write_run_cap_manifest",
                        "finalize_run_cap_manifest", "compute_result_binding")]
    _check(len(tli_sidecar) == 1
           and len(runcap_nodes) >= 4
           and all(not _in_node(tli_sidecar[0], n)
                   for n in runcap_nodes),
           f"G6c run-cap 写门/落盘（{len(runcap_nodes)} 处调用）全部在 "
           f"method=='tli' 分支之外——none/fullkv 全 method 覆盖（修复 "
           f"前 fullkv 零 provenance）")

    # 时序：写门（gate_run_cap_identity）→ 数据写（open(out_path,"w")）
    # → complete 回填（compute_result_binding）
    gate_ln = [n.lineno for n in runcap_nodes
               if _call_name(n) == "gate_run_cap_identity"][0]
    data_writes = [n.lineno for n in ast.walk(main_body)
                   if isinstance(n, ast.Call) and _call_name(n) == "open"
                   and any(ast.unparse(a) == "'w'" for a in n.args)]
    bind_ln = [n.lineno for n in runcap_nodes
               if _call_name(n) == "compute_result_binding"][0]
    _check(gate_ln < min(data_writes) < bind_ln,
           f"G6d 时序：CAP-CLASH 写门(L{gate_ln}) < 数据写"
           f"(L{min(data_writes)}) < 结果 sha 回填(L{bind_ln})"
           f"——写门前拒绝不动数据，complete 回填在数据之后")

    # get_pred 采集 run_stats
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_pred":
            gp = node
            break
    stats_nodes = [n for n in ast.walk(gp)
                   if isinstance(n, ast.Call)
                   and _call_name(n) == "run_stats.setdefault"]
    stat_keys = {getattr(n.args[0], "value", None) for n in stats_nodes
                 if n.args}
    _check(stat_keys == {"input_tokens", "truncated"},
           f"G6e get_pred 内 run_stats.setdefault 采集键集合 == "
           f"{stat_keys}——每样本实际输入 token 数 + 截断标记进 run-cap "
           f"manifest（修复前只记数据集原字段 length）")


def test_green_backfill():
    print("== 096 绿证：backfill 双 provenance 口径 ==")
    with tempfile.TemporaryDirectory() as td:
        result = os.path.join(td, "hotpotqa-none-1012024000.jsonl")
        rows = [{"pred": "a", "answers": ["x"], "all_classes": None,
                 "length": 1000, "budget": None, "score_sum": None,
                 "_id": "hotpotqa:abc123def456:0"},
                {"pred": "b", "answers": ["y"], "all_classes": None,
                 "length": 2000, "budget": None, "score_sum": None,
                 "_id": "hotpotqa:abc123def456:1"}]
        with open(result, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log = os.path.join(td, "run.log")
        # 真实 gate 打印格式（context_capability_gate.py 的输出原文）
        with open(log, "w", encoding="utf-8") as f:
            f.write("[CONTEXT-CAP-GATE] model=DeepSeek-V2-Lite-Chat\n"
                    "  experiment_input_cap (live)  = 16384   "
                    "<- model2maxlen.json:DeepSeek-V2-Lite-Chat\n"
                    "  experiment_input_cap (registry protocol) = 31500   "
                    "<- registry\n"
                    "[EXPERIMENT-CAP-DIVERGENCE] 本次实验 cap=16384 与"
                    "注册表 protocol 值 31500 不同——按受限上下文 treatment "
                    "如实记录\n")

        def run_backfill(extra):
            return subprocess.run(
                [sys.executable, BACKFILL_FP, "--result", result,
                 "--model", "DeepSeek-V2-Lite-Chat", "--method", "none",
                 "--task", "hotpotqa", "--t", "1012024000",
                 "--max-new-tokens", "32"] + extra,
                cwd=REPO, capture_output=True, text=True)

        # 有日志 → backfilled-from-stdout
        r = run_backfill(["--log", log])
        _check(r.returncode == 0, f"G7a backfill（有日志）成功"
               f"{'' if r.returncode == 0 else '：' + r.stderr[-200:]}")
        sc = result + ".runcap.json"
        m = json.load(open(sc, encoding="utf-8"))
        import hashlib as _h
        exp_sha = _h.sha256(open(result, "rb").read()).hexdigest()
        _check(m["provenance"] == "backfilled-from-stdout"
               and m["effective_cap"] == 16384
               and m["protocol_cap"] == 31500
               and m["cap_treatment"]["type"] == "restricted_context_"
               "treatment"
               and m["result_sha256"] == exp_sha
               and m["result_lines"] == 2
               and m["input_tokens"] is None,
               "G7b provenance=backfilled-from-stdout（cap 16384/31500 从"
               " gate 打印解析 + 结果 sha/行数绑定 + token 数如实 null）")
        # 数据指纹与生产口径同源（answers+length canonical sha 前 12）
        fp_src = json.dumps([[r["answers"], r["length"]] for r in rows],
                            ensure_ascii=False, sort_keys=True)
        exp_fp = _h.sha256(fp_src.encode("utf-8")).hexdigest()[:12]
        _check(m["dataset_fingerprint"] == exp_fp,
               "G7c backfill 数据指纹与 pred.py E116a 生产口径逐位同源")

        # 幂等重跑：未带 --overwrite-backfill → 拒；带 → 过
        r = run_backfill(["--log", log])
        _check(r.returncode != 0 and "[GATE-FAIL]" in (r.stdout + r.stderr),
               "G7d 已存在 backfilled manifest → 默认拒绝（防误覆盖）")
        r = run_backfill(["--log", log, "--overwrite-backfill"])
        _check(r.returncode == 0,
               "G7e --overwrite-backfill 显式幂等重跑通过")

        # inline manifest 不可被 backfill 覆盖
        m["provenance"] = "inline"
        m["input_tokens"] = {"count": 2, "min": 1, "max": 2,
                             "per_sample": [1, 2]}
        with open(sc, "w", encoding="utf-8") as f:
            json.dump(m, f)
        r = run_backfill(["--log", log, "--overwrite-backfill"])
        _check(r.returncode != 0 and "inline" in (r.stdout + r.stderr),
               "G7f 已存在 inline manifest → backfill 拒绝覆盖（真实 "
               "provenance 优先）")

        # 缺日志 → provenance_incomplete
        os.remove(sc)
        r = run_backfill([])
        _check(r.returncode == 0, f"G7g backfill（缺日志）成功降格落盘"
               f"{'' if r.returncode == 0 else '：' + r.stderr[-200:]}")
        m = json.load(open(sc, encoding="utf-8"))
        _check(m["provenance"] == "provenance_incomplete"
               and m["effective_cap"] is None
               and m["cap_treatment"]["type"] == "unknown",
               "G7h 缺日志 → provenance=provenance_incomplete + "
               "effective_cap=null + 类型 unknown（不冒称 unregistered/无"
               "证据口径，主 AI 决定是否入论文表）")


def test_green_ruler():
    print("== 096 绿证：RULER 入口判断（登记不接入）==")
    src = open(RULER_FP, encoding="utf-8").read()
    _check('f"L{args.context_length}"' in src
           and "context_length=args.context_length" in src,
           "G8 RULER 入口已有等价 cap 绑定：context_length 进输出目录路径 "
           "L{N}/ 且写入 yarn receipt（producer-yarn-config-v2 同代绑定"
           " SHA/行数）——096 无需重复接入，登记说明（fullkv/none 同样有"
           " receipt，与 LongBench fullkv 此前零 provenance 不同）")


def main():
    test_red()
    test_green_identity_and_gate()
    test_green_t21_reverse()
    test_green_gate_info()
    test_green_pred_integration()
    test_green_backfill()
    test_green_ruler()
    print(f"\nALL PASS ({PASS_N} checks) —— python"
          f"{' -O' if __debug__ is False else ''} 口径")


if __name__ == "__main__":
    main()
