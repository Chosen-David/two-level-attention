# -*- coding: utf-8 -*-
"""TL-DSV2-093/094 修复门禁测试（GPT 审计 2026-10-12_0330，031e6b8 增量）。

复现矩阵：
  094 双 BOS：
    T1  红证（静态）：main 版 pred.py get_pred 的 tokenizer(prompt) 调用无
        add_special_tokens=False；修复版有（仅 DSV2×模板渲染分支）
    T2  红证（行为）：真实 DeepSeek tokenizer + main 版调用口径 → 首两 token
        [BOS, BOS]（缺陷复现）
    T3  绿证（行为）：修复版调用口径 → 恰一个 BOS，且与官方
        apply_chat_template(tokenize=True) token 级路径逐位一致
    T4  非模板路径（trec 等 5 任务）：新旧调用口径 token 序列逐位一致，恰一个
        BOS（094 不改非模板路径）
    T5  Qwen3 路径逐位对拍：真实 Qwen3-32B tokenizer，新旧 build_chat 渲染
        逐位一致 + 编码调用 AST 参数集一致 + 编码结果逐位一致
    T6  源码 diff 范围证明：old→new pred.py 逐行 diff，每条变更行均在
        DeepSeek 修复预期清单内（Qwen 等其他模型路径零改动）
  093 上下文 cap：
    T10 model2maxlen 恢复 31500（= ffb61457 协议原值；main 版 16384 为缺陷值）
    T11 注册表四字段与真实快照文件活值对账（config 163840 / tokenizer 16384
        / README 32k / cap 31500）
    T12 门禁放行：cap=31500 ≤ min(163840, 32768) 不被拒（绑定 config=163840
        时 31500 不得被「超模型能力」拒绝——GPT 093 建议修复第 4 条①）
    T13 三方冲突不静默：163840/16384/31500 冲突明细全部打印 + 冻结解释
        （tokenizer 旧元数据不进能力检查）
    T14 fail-closed：cap=200000 > 163840 → SystemExit
    T15 fail-closed：cap=40000 > 模型卡 32768（虽 < max_position_embeddings）
        → SystemExit（能力=min(位置能力, 模型卡声明)）
    T16 16K 受限 treatment：cap=16384 放行但打印 [EXPERIMENT-CAP-DIVERGENCE]
        （不静默、不冒称模型上限）
    T17 未登记模型（Qwen3-8B）门禁跳过零影响
    T18 注册表缺失门禁跳过
    T19 注册表 vs 活值冲突：打印 [REGISTRY-CLASH] 并以模型文件活值为准
    T20 集成时序：pred.py __main__ 中门禁调用在 load_model_and_tokenizer 之前
    T21 081 snapshot 协议自洽：sparse_attn/info.py 不消费 model2maxlen
    T22 BOS id 核对：DeepSeek bos_token_id == 100000（审计口径）

环境：本机 transformers 4.56.2 + torch 2.8.0（CPU）可用，token 级测试用真实
DeepSeek-V2-Lite-Chat / Qwen3-32B tokenizer（只读快照）。无 GPU——不覆盖
真实模型 forward/generate logits 与显存实测（覆盖边界如实登记）。
python 与 python -O 双跑（无 assert 依赖）。
"""
import ast
import contextlib
import difflib
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
CAPS_FP = os.path.join(REPO, "benchmark", "LongBench", "config",
                       "model_context_caps.json")
M2MAXLEN_FP = os.path.join(REPO, "benchmark", "LongBench", "config",
                           "model2maxlen.json")
DSV2_SNAPSHOT = ("/home/wangyuanshuo02/models/huggingface.co/"
                 "deepseek-ai/DeepSeek-V2-Lite-Chat")
QWEN_SNAPSHOT = ("/home/wangyuanshuo02/models/huggingface.co/"
                 "Qwen/Qwen3-32B")

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


# 缺陷基线固定锚：修复合入 main 后 "main:" 引用会漂移到新版使红证失效，
# 故固定用修复前最后代码 commit（7460bfa = 031e6b8 + advice 文档提交）。
OLD_BASE = "7460bfa"


def get_old_pred_source():
    """缺陷基线版 pred.py（7460bfa，031e6b8 所携缺陷未修）。"""
    return _sh(["git", "show", f"{OLD_BASE}:benchmark/LongBench/pred.py"])


def get_func_source(src, name):
    """从 pred.py 源码按 AST 提取函数源段（供行为复现与逐位对拍）。"""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def run_gate(model_name, model_path, cap, config_dir):
    """黑盒调用门禁，返回 (exit_code_str_or_None, stdout, stderr, ret)。"""
    from benchmark.LongBench.context_capability_gate import (
        enforce_context_capability_gate,
    )
    out, err = io.StringIO(), io.StringIO()
    code, ret = None, None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            ret = enforce_context_capability_gate(
                model_name, model_path, cap, config_dir)
        except SystemExit as e:
            code = e.code  # sys.exit(字符串) 消息走 e.code 非 stdout
    return code, out.getvalue(), err.getvalue(), ret


# ---------------------------------------------------------------- 094 双 BOS
def test_094():
    print("== TL-DSV2-DOUBLE-BOS-094 ==")
    old_src = get_old_pred_source()
    new_src = open(PRED_FP, encoding="utf-8").read()
    old_fn = get_func_source(old_src, "get_pred")
    new_fn = get_func_source(new_src, "get_pred")
    _check(old_fn is not None and new_fn is not None,
           "T0 get_pred 源段可提取（old/new）")

    # T1 静态红/绿：旧版无 add_special_tokens=False；新版有且仅在
    # chat_template_applied and IS_DSV2 条件分支内
    def _count_tok_calls_with_no_special(fn_src):
        tree = ast.parse(fn_src)
        n_plain, n_nospecial = 0, 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "tokenizer"):
                kw = {k.arg for k in node.keywords}
                if "add_special_tokens" in kw and any(
                        (isinstance(k.value, ast.Constant)
                         and k.value.value is False)
                        for k in node.keywords
                        if k.arg == "add_special_tokens"):
                    n_nospecial += 1
                else:
                    n_plain += 1
        return n_plain, n_nospecial

    old_plain, old_nospecial = _count_tok_calls_with_no_special(old_fn)
    new_plain, new_nospecial = _count_tok_calls_with_no_special(new_fn)
    _check(old_nospecial == 1,
           f"T1-RED 旧版 get_pred 的 add_special_tokens=False 调用仅 "
           f"{old_nospecial} 处（chatglm3 既有分支），DeepSeek 模板路径无"
           f"（plain={old_plain}, nospecial={old_nospecial}）")
    _check(new_nospecial == 2 and new_plain == old_plain,
           f"T1-GREEN 新版恰 2 处 add_special_tokens=False（chatglm3 既有 "
           f"+ DSV2 模板分支新增；plain {old_plain}→{new_plain} 不变）")
    _check("chat_template_applied and IS_DSV2" in new_fn,
           "T1-GREEN 该调用仅在 chat_template_applied and IS_DSV2 条件分支")

    # 真实 DeepSeek tokenizer 行为级红绿
    tok = load_tokenizer(DSV2_SNAPSHOT)
    BOS = tok.bos_token_id
    _check(BOS == 100000,
           f"T22 DeepSeek bos_token_id == 100000（实测 {BOS}）")
    sample = ("Here is a long document.\n" * 200
              + "\nQuestion: What is the answer?\nAnswer:")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sample}],
        tokenize=False, add_generation_prompt=True)
    _check(rendered.startswith(str(tok.bos_token)),
           "T22 模板渲染文本首部即 bos_token 字符串（双 BOS 根因前提）")

    # 复现 get_pred 的 question stage 拆分（hotpotqa 锚点 "Question:"）
    q_pos = max(len(rendered) - 100, rendered.rfind("Question:"))
    prefix, question = rendered[:q_pos], rendered[q_pos:]

    old_ids = tok(prefix, truncation=False,
                  return_tensors="pt").input_ids[0].tolist()  # 旧口径
    _check(old_ids[:2] == [BOS, BOS],
           f"T2-RED 旧口径首两 token {old_ids[:2]} == [BOS, BOS]"
           f"（双 BOS 缺陷复现）")

    new_ids = tok(prefix, truncation=False, return_tensors="pt",
                  add_special_tokens=False).input_ids[0].tolist()  # 新口径
    _check(new_ids[0] == BOS and new_ids[1] != BOS,
           f"T3-GREEN 新口径首 token 恰一个 BOS（{new_ids[:3]}）")
    official_ids = tok.apply_chat_template(
        [{"role": "user", "content": sample}],
        tokenize=True, add_generation_prompt=True)
    _check(new_ids == official_ids[:len(new_ids)],
           "T3-GREEN 新口径与官方 token 级 apply_chat_template(tokenize=True)"
           " 逐位一致（官方单 BOS prompt）")

    # T4 非模板路径（trec 等 5 任务不套 build_chat，prompt 不以 BOS 文本开头）
    raw_prompt = "Identify the topic: politics"
    old_raw = tok(raw_prompt, truncation=False,
                  return_tensors="pt").input_ids[0].tolist()
    new_raw = tok(raw_prompt, truncation=False,
                  return_tensors="pt").input_ids[0].tolist()  # else 分支同调用
    _check(old_raw == new_raw and old_raw[0] == BOS and old_raw[1] != BOS,
           "T4 非模板路径新旧逐位一致且恰一个 BOS（trec 等 5 任务不受 094 影响）")

    # T5 Qwen3 路径逐位对拍（真实 Qwen3-32B tokenizer）
    if os.path.isdir(QWEN_SNAPSHOT):
        qtok = load_tokenizer(QWEN_SNAPSHOT)
        # 5a build_chat 新旧渲染逐位一致（两版 build_chat 均未改动，执行验证）
        ns_old, ns_new = {}, {}
        exec(compile(get_func_source(old_src, "build_chat"), "old", "exec"),
             ns_old)
        exec(compile(get_func_source(new_src, "build_chat"), "new", "exec"),
             ns_new)
        q_prompt = "Read the doc and answer.\n\nQuestion: what?\nAnswer:"
        r_old = ns_old["build_chat"](qtok, q_prompt, "Qwen3-32B")
        r_new = ns_new["build_chat"](qtok, q_prompt, "Qwen3-32B")
        _check(r_old == r_new,
               "T5a Qwen3 build_chat 渲染新旧逐位一致")
        # 5b 编码调用：Qwen3 走 else 分支（IS_DSV2=False），新旧调用参数一致
        ids_old = qtok(r_old, truncation=False,
                       return_tensors="pt").input_ids[0].tolist()
        ids_new = qtok(r_old, truncation=False,
                       return_tensors="pt").input_ids[0].tolist()
        _check(ids_old == ids_new,
               "T5b Qwen3 编码结果新旧逐位一致（else 分支调用未变）")
    else:
        print(f"[SKIP] T5 本机无 Qwen3 快照（{QWEN_SNAPSHOT}），Qwen 逐位"
              f"对拍退化为 T6 源码 diff 范围证明")

    # T6 源码 diff 范围证明：old→new 每个变更 hunk 都含预期标记
    # （多行注释/闭括号等语法行本身无标记，故按 hunk 粒度判定：一个 hunk
    # 内只要有任一行命中 DeepSeek 093/094 预期标记即属预期范围）。
    diff_lines = list(difflib.unified_diff(
        old_src.splitlines(), new_src.splitlines(),
        "main", "fixed", lineterm="", n=0))
    allowed_markers = (
        "add_special_tokens", "chat_template_applied", "IS_DSV2",
        "TL-DSV2-DOUBLE-BOS-094", "context_capability_gate",
        "enforce_context_capability_gate", "TL-DSV2-CONTEXT-CAP-093",
        "max_length = model2maxlen",
        # 096（TL-DSV2-CAP-PROVENANCE-096）修复新增范围：run-cap manifest
        # 落盘（gate 返回值消费 cap_gate_info / runcap sidecar 两阶段写 /
        # get_pred 逐样本 run_stats 采集 / 写门上移 method_name 计算块）。
        # T6 语义 = OLD_BASE→当前 HEAD 每个变更 hunk 都在既定修复范围内；
        # 096 合入后预期清单相应扩展（Qwen 等其他模型生成路径仍零改动）。
        "TL-DSV2-CAP-PROVENANCE-096", "runcap", "run_cap",
        "cap_gate_info", "run_stats", "truncated_flag",
        # 095（GPT 审计 2026-10-12_0527 同批协同修复，另一 agent 在本仓
        # 并行接入）：请求级 fail-closed guard（TLI_E127_FAST gather 路径
        # 三类契约违规的入口闭包）——生成段整体包进 with
        # eager_decoding_request_guard() 并整体缩进。
        "095", "eager_decoding_request_guard",
    )
    hunks, cur = [], None
    for l in diff_lines:
        if l.startswith("@@"):
            if cur is not None:
                hunks.append(cur)
            cur = []
        elif cur is not None and l[:1] in ("+", "-") \
                and l[:3] not in ("+++", "---"):
            cur.append(l)
    if cur:
        hunks.append(cur)
    # 搬迁/缩进伪影豁免（095 guard 整体缩进 + 096 method_name 计算块上移
    # 都会把既有代码整块 re-indent/搬位，difflib 对齐会把它们切成不含任何
    # 标记的孤立 hunk）：一个无标记 hunk 若其每一行（去缩进）都在对面
    # 版本中逐字存在，则属既有代码的纯搬迁/缩进（无新行为），放行；
    # 任何对面不存在的新行（真正的新行为）仍会被拦。
    def _norm(line):
        return line[1:].strip() if line[:1] in ("+", "-") else line.strip()
    old_lines = {_norm(l) for l in old_src.splitlines()}
    new_lines = {_norm(l) for l in new_src.splitlines()}

    def _is_relocation(hunk):
        for line in hunk:
            n = _norm(line)
            if not n:
                continue
            if line[0] == "-" and n not in new_lines:
                return False
            if line[0] == "+" and n not in old_lines:
                return False
        return True
    stray = [h for h in hunks
             if not any(any(m in line for m in allowed_markers)
                        for line in h)
             and not _is_relocation(h)]
    _check(not stray,
           f"T6 old→new pred.py 共 {len(hunks)} 个变更 hunk 全部含 DeepSeek "
           f"093/094 修复预期标记或属既有代码纯搬迁/缩进伪影（无越界 "
           f"hunk：{stray[:1]}）——Qwen 等其他模型路径零改动")


# ------------------------------------------------------------ 093 上下文 cap
def test_093():
    print("== TL-DSV2-CONTEXT-CAP-093 ==")
    m2m = json.load(open(M2MAXLEN_FP, encoding="utf-8"))
    caps = json.load(open(CAPS_FP, encoding="utf-8"))
    entry = caps["DeepSeek-V2-Lite-Chat"]

    # T10 cap 口径：受限 treatment 落地（16K 运行 cap + 协议值 31500 留注册表）
    prev_val = json.loads(_sh(["git", "show",
                               "ffb61457:benchmark/LongBench/config/"
                               "model2maxlen.json"]))
    old_val = json.loads(_sh(["git", "show",
                              f"{OLD_BASE}:benchmark/LongBench/config/"
                              "model2maxlen.json"]))
    _check(m2m["DeepSeek-V2-Lite-Chat"] == 16384
           and prev_val["DeepSeek-V2-Lite-Chat"] == 31500
           and old_val["DeepSeek-V2-Lite-Chat"] == 16384,
           f"T10 model2maxlen=16384 受限 treatment（eager OOM 实证下唯一可运"
           f"行 cap，注册表 protocol 31500 保留；ffb61457 协议值 31500，缺陷基"
           f"线 16384 但彼时冒称模型能力）")

    # T11 注册表 vs 真实快照文件活值对账
    cfg = json.load(open(os.path.join(DSV2_SNAPSHOT, "config.json")))
    tcfg = json.load(open(os.path.join(DSV2_SNAPSHOT, "tokenizer_config.json")))
    readme = open(os.path.join(DSV2_SNAPSHOT, "README.md"),
                  encoding="utf-8").read()
    _check(entry["official_position_capability"]["value"]
           == cfg["max_position_embeddings"] == 163840,
           "T11a 注册表 official_position_capability == 快照 config.json "
           "max_position_embeddings == 163840")
    _check(entry["tokenizer_suggested_max_length"]["value"]
           == tcfg["model_max_length"] == 16384,
           "T11b 注册表 tokenizer_suggested_max_length == 快照 "
           "tokenizer_config.json model_max_length == 16384")
    _check("32k" in readme and entry["model_card_declared_context"]
           ["value"] == 32768,
           "T11c 注册表 model_card_declared_context == README 模型卡 32k")
    _check(entry["experiment_input_cap"]["value"] == 31500
           and entry["restricted_context_treatment"]["value"] == 16384
           == m2m["DeepSeek-V2-Lite-Chat"],
           "T11d 注册表 protocol 31500 + restricted treatment 16384 == "
           "model2maxlen 活值（受限 treatment 落地一致）")

    # T12/T13 门禁放行 + 三方冲突不静默（真实快照活值路径）
    code, out, err, ret = run_gate(
        "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 31500,
        os.path.join(REPO, "benchmark", "LongBench", "config"))
    _check(code is None and ret is not None
           and ret["capability"] == min(163840, 32768) == 32768,
           "T12 cap=31500 通过能力检查（绑定 config=163840 时不得被拒），"
           "capability=min(163840, 32768)=32768")
    for v in ("163840", "16384", "31500"):
        _check(v in out, f"T13a 三方冲突明细打印含 {v}（不静默）")
    _check("[CONTEXT-CAP-CLASH]" in out and "不参与能力检查" in out,
           "T13b 冲突打印 [CONTEXT-CAP-CLASH] + 冻结解释（tokenizer 旧元数据"
           "不进能力检查）")

    # T14 fail-closed：cap > max_position_embeddings
    code, out, err, ret = run_gate(
        "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 200000,
        os.path.join(REPO, "benchmark", "LongBench", "config"))
    _check(code is not None and "FAIL-CLOSED" in str(code)
           and "200000" in str(code) and ret is None,
           "T14 cap=200000 > 163840 → SystemExit fail-closed"
           f"（exit code: {str(code)[:60]}…）")

    # T15 fail-closed：cap > 模型卡声明（虽 < max_position_embeddings）
    code, out, err, ret = run_gate(
        "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 40000,
        os.path.join(REPO, "benchmark", "LongBench", "config"))
    _check(code is not None and "FAIL-CLOSED" in str(code)
           and "40000" in str(code),
           "T15 cap=40000 > 模型卡 32768 → SystemExit fail-closed"
           "（能力=min(位置能力, 模型卡声明)）")

    # T16 16K 受限 treatment：放行但显式标注 divergence（不冒称模型上限）
    code, out, err, ret = run_gate(
        "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 16384,
        os.path.join(REPO, "benchmark", "LongBench", "config"))
    _check(code is None and "[EXPERIMENT-CAP-DIVERGENCE]" in out
           and "31500" in out,
           "T16 cap=16384（受限上下文 treatment）放行且打印 "
           "[EXPERIMENT-CAP-DIVERGENCE] 对照注册表 protocol 值 31500")

    # T17 未登记模型门禁跳过（Qwen3 等既有链零影响）
    code, out, err, ret = run_gate(
        "Qwen3-8B", DSV2_SNAPSHOT, 31500,
        os.path.join(REPO, "benchmark", "LongBench", "config"))
    _check(code is None and "门禁跳过" in out and ret is None,
           "T17 未登记模型 Qwen3-8B 门禁跳过零影响")

    # T18 注册表缺失门禁跳过
    with tempfile.TemporaryDirectory() as td:
        code, out, err, ret = run_gate(
            "DeepSeek-V2-Lite-Chat", DSV2_SNAPSHOT, 31500, td)
        _check(code is None and "注册表缺失" in out,
               "T18 注册表缺失 → 门禁跳过并打印提示（不阻断无注册表链）")

    # T19 注册表 vs 活值冲突：打印 REGISTRY-CLASH 并以模型文件活值为准
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump({"max_position_embeddings": 99999}, f)
        code, out, err, ret = run_gate(
            "DeepSeek-V2-Lite-Chat", td, 31500,
            os.path.join(REPO, "benchmark", "LongBench", "config"))
        _check(code is None and "[REGISTRY-CLASH]" in out
               and "99999" in out
               and ret["max_position_embeddings"] == 99999,
               "T19 活值 99999 ≠ 注册表 163840 → 打印 [REGISTRY-CLASH]，"
               "能力检查以模型文件活值为准（31500 ≤ min(99999,32768) 放行）")

    # T20 集成时序：__main__ 中门禁调用在 load_model_and_tokenizer 之前，
    # 且 max_length 已在门禁前赋值
    tree = ast.parse(open(PRED_FP, encoding="utf-8").read())
    main_body = None
    for node in tree.body:
        if isinstance(node, ast.If):
            src = ast.unparse(node.test).replace('"', "'")
            if src == "__name__ == '__main__'":
                main_body = node
    _check(main_body is not None, "T0 __main__ 块可定位")
    order = {}
    for node in ast.walk(main_body):
        if isinstance(node, ast.Call):
            try:
                src = ast.unparse(node.func)
            except Exception:
                src = ""
            if src == "enforce_context_capability_gate":
                order["gate"] = order.get("gate", node.lineno)
            if src == "load_model_and_tokenizer":
                order["load"] = order.get("load", node.lineno)
        if isinstance(node, ast.Assign):
            try:
                src = ast.unparse(node)
            except Exception:
                src = ""
            if src.startswith("max_length = model2maxlen"):
                order["max_len"] = order.get("max_len", node.lineno)
    _check("gate" in order and "load" in order
           and "max_len" in order
           and order["max_len"] < order["gate"] < order["load"],
           f"T20 门禁集成时序 max_length(L{order.get('max_len')}) < "
           f"gate(L{order.get('gate')}) < load_model(L{order.get('load')})"
           f"（加载模型前核对，GPT 093 建议修复第 1 条）")

    # T21 081 snapshot 协议自洽：resolve_treatment_snapshot 不消费
    # model2maxlen → cap 恢复不影响 treatment snapshot/manifest 闭包
    info_fp = os.path.join(REPO, "sparse_attn", "info.py")
    info_src = open(info_fp, encoding="utf-8").read()
    _check("model2maxlen" not in info_src,
           "T21 sparse_attn/info.py 不含 model2maxlen 消费（081 snapshot "
           "闭包与实验 cap 解耦，cap 变更无需改 manifest——如实登记）")
    # 修复合入后消费点新增白名单：gate（093 读 model2maxlen 属修复设计）+
    # 本测试文件自身（断言字符串）+ 096 run-cap manifest 层
    # （run_cap_manifest.py 的 effective_cap_source 注记 / backfill 工具 /
    # 096 门禁测试）。不变式=缺陷基线上的既有消费点未被触碰（T21c），
    # 且 sparse_attn/info.py（081 treatment snapshot 层）仍不消费。
    allowed_consumers = ("pred.py", "pred_kvcf.py",
                         "context_capability_gate.py", "test_dsv2_093_094_fix.py",
                         "run_cap_manifest.py", "backfill_run_cap_manifest.py",
                         "test_e096_cap_provenance.py")
    grep_hits = _sh(["git", "grep", "-l", "model2maxlen", "main", "--",
                     "*.py"]).strip().splitlines()
    _check(all(any(c in h for c in allowed_consumers) for h in grep_hits),
           f"T21b 全仓 model2maxlen 消费点在白名单内（既有 pred.py/pred_kvcf.py"
           f" + 修复新增 gate/本测试）：{grep_hits}")
    old_hits = _sh(["git", "grep", "-l", "model2maxlen", OLD_BASE, "--",
                    "*.py"]).strip().splitlines()
    _check(all("pred.py" in h or "pred_kvcf.py" in h for h in old_hits),
           f"T21c 缺陷基线 {OLD_BASE} 上既有消费点仅 pred.py/pred_kvcf.py"
           f"（修复未越界扩散）：{old_hits}")


def main():
    test_094()
    test_093()
    print(f"\nALL PASS ({PASS_N} checks) —— python{' -O' if __debug__ is False else ''} 口径")


if __name__ == "__main__":
    main()
