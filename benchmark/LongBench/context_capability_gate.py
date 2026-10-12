# -*- coding: utf-8 -*-
"""TL-DSV2-CONTEXT-CAP-093 修复（GPT 审计 2026-10-12_0330，031e6b8）：实验输入
cap 与模型上下文能力的启动门禁。

背景：031e6b8 把 DeepSeek-V2-Lite-Chat 的 model2maxlen 从 31500 静默降到
16384，理由「超过 max_position_embeddings=16384」错误——官方 config.json 实际
163840（yarn factor=40），模型卡声明 32K，仅 tokenizer_config.json 旧元数据是
16384。59GB OOM 根因是 eager attention 的二次显存（16 heads × 31500² × fp32），
不是位置能力。把「运行实现的显存上限」当「模型上下文上限」写进截断协议，会使
DeepSeek 与 Qwen3 的 31500 口径不再同臂可比。

设计（按 GPT 建议精简落地，不过度设计）：
  1. 注册表 config/model_context_caps.json 登记四字段 + 各自证据来源（文件名
     +字段名）：official_position_capability / tokenizer_suggested_max_length
     / model_card_declared_context / experiment_input_cap（实验 cap 独立命名，
     不与模型能力字段混同）。
  2. 启动时（加载 tokenizer/model 之前）打印并核对三方值：config.json 的
     max_position_embeddings、tokenizer model_max_length、实验 cap
     （model2maxlen 活值）。三者冲突时不静默选择——打印 [CONTEXT-CAP-CLASH]
     冲突明细 + 冻结解释（tokenizer 旧元数据不进能力检查，证据见注册表）。
  3. 能力检查：cap > min(max_position_embeddings, 模型卡声明) → sys.exit
     fail-closed（能力用本地快照活值优先，活值不可读回退注册表冻结值）。
  4. cap 与注册表 protocol 值不同（如再选 16K treatment）时打印
     [EXPERIMENT-CAP-DIVERGENCE]——16K 可作为受限上下文补充实验跑，但不得
     静默、不得冒称模型上限（GPT 093 建议修复第 2 条）。
  5. 未登记模型（Qwen3 等既有链）→ 门禁跳过，行为零变化；注册表文件缺失 →
     跳过并打印提示（门禁保护范围内 fail-open 只影响登记模型自身，不阻断
     既有派单链）。

只依赖标准库（json/os/sys），测试可在无 torch/GPU 环境 import。
"""
import json
import os
import sys

GATE_CONFIG_FILENAME = "model_context_caps.json"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_live_meta(model_path):
    """若 model_path 是本地目录，读其 config.json 的 max_position_embeddings
    与 tokenizer_config.json 的 model_max_length（官方快照活值）。任一文件
    缺失/损坏/非本地目录时对应项返回 None（回退注册表冻结值，不静默失败——
    回退会在输出中如实打印）。"""
    live_max_pos = None
    live_tok_meta = None
    if model_path and os.path.isdir(model_path):
        cfg_fp = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_fp):
            try:
                cfg = _read_json(cfg_fp)
                live_max_pos = cfg.get("max_position_embeddings")
            except (json.JSONDecodeError, OSError):
                live_max_pos = None
        tok_fp = os.path.join(model_path, "tokenizer_config.json")
        if os.path.isfile(tok_fp):
            try:
                tok = _read_json(tok_fp)
                live_tok_meta = tok.get("model_max_length")
            except (json.JSONDecodeError, OSError):
                live_tok_meta = None
    return live_max_pos, live_tok_meta


def enforce_context_capability_gate(model_name, model_path, experiment_cap,
                                    config_dir, stream=None):
    """启动门禁（见模块 docstring）。通过返回打印明细后的 None；超能力
    sys.exit（fail-closed，消息走 SystemExit.code）。未登记模型/注册表缺失
    时跳过（打印一行提示）。

    返回 dict（测试可消费）：{capability, max_position_embeddings,
    tokenizer_suggested_max_length, model_card_declared_context,
    experiment_cap}——fail-closed 路径不返回（SystemExit）。
    096（TL-DSV2-CAP-PROVENANCE-096）追加注册表侧三键
    {registry_protocol_cap, registry_restricted_context_treatment,
    registered}：run-cap manifest（benchmark/LongBench/run_cap_manifest.py）
    消费 protocol cap / restricted treatment 值把 cap 绑进结果产物身份——
    修复前该返回值被 pred.py 丢弃（096 审计 P1）。未登记/注册表缺失的
    跳过路径仍返回 None，消费侧按 unregistered 折叠。
    """
    if stream is None:
        stream = sys.stdout
    caps_path = os.path.join(str(config_dir), GATE_CONFIG_FILENAME)
    if not os.path.isfile(caps_path):
        print(f"[CONTEXT-CAP-GATE] 注册表缺失 {caps_path}，门禁跳过"
              f"（模型 {model_name} 无冻结能力记录）", file=stream)
        return None
    registry = _read_json(caps_path)
    entry = registry.get(model_name)
    if entry is None:
        print(f"[CONTEXT-CAP-GATE] 模型 {model_name} 未登记能力记录"
              f"（{GATE_CONFIG_FILENAME}），门禁跳过", file=stream)
        return None

    reg_pos = entry["official_position_capability"]["value"]
    reg_tok = entry["tokenizer_suggested_max_length"]["value"]
    reg_card = entry["model_card_declared_context"]["value"]
    reg_cap = entry["experiment_input_cap"]["value"]

    # 活值优先（模型文件是位置能力的第一事实源），活值与注册表冲突时打印
    # 明细并用活值——注册表需要人工更新，不静默。
    live_pos, live_tok = _read_live_meta(model_path)
    max_pos = live_pos if live_pos is not None else reg_pos
    tok_meta = live_tok if live_tok is not None else reg_tok
    pos_src = ("config.json:max_position_embeddings (live)"
               if live_pos is not None
               else f"registry frozen (model path 不可读活值: {model_path!r})")
    if live_pos is not None and live_pos != reg_pos:
        print(f"[REGISTRY-CLASH] 注册表 official_position_capability={reg_pos} "
              f"与本地快照活值 {live_pos} 不一致——以模型文件活值为准，"
              f"请更新 {GATE_CONFIG_FILENAME}", file=stream)
    if live_tok is not None and live_tok != reg_tok:
        print(f"[REGISTRY-CLASH] 注册表 tokenizer_suggested_max_length={reg_tok} "
              f"与本地快照活值 {live_tok} 不一致——以 tokenizer 文件活值为准，"
              f"请更新 {GATE_CONFIG_FILENAME}", file=stream)

    print(f"[CONTEXT-CAP-GATE] model={model_name}", file=stream)
    print(f"  max_position_embeddings      = {max_pos}   <- {pos_src}",
          file=stream)
    print(f"  tokenizer model_max_length    = {tok_meta}   "
          f"<- tokenizer_config.json:model_max_length", file=stream)
    print(f"  model_card_declared_context   = {reg_card}   "
          f"<- {entry['model_card_declared_context']['evidence']}", file=stream)
    print(f"  experiment_input_cap (live)  = {experiment_cap}   "
          f"<- model2maxlen.json:{model_name}", file=stream)
    print(f"  experiment_input_cap (registry protocol) = {reg_cap}   "
          f"<- {entry['experiment_input_cap']['evidence']}", file=stream)

    # 三方冲突（config vs tokenizer 元数据 vs 实验 cap）：不静默挑选——打印
    # 冲突明细 + 冻结解释。tokenizer 旧元数据是建议字段，不进能力检查
    # （GPT 093：不能选冲突元数据中最短的一项再称之为模型能力）。
    vals = {max_pos, tok_meta, experiment_cap}
    if len(vals) > 1:
        print(f"[CONTEXT-CAP-CLASH] 三方值不一致："
              f"max_position_embeddings={max_pos} / "
              f"tokenizer model_max_length={tok_meta} / "
              f"experiment_input_cap={experiment_cap}。冻结解释：tokenizer "
              f"model_max_length 是官方仓旧元数据（证据："
              f"{entry['tokenizer_suggested_max_length']['evidence']}），仅作"
              f"建议字段打印，不参与能力检查；能力检查 = "
              f"min(max_position_embeddings, model_card_declared_context)。",
              file=stream)

    if experiment_cap != reg_cap:
        print(f"[EXPERIMENT-CAP-DIVERGENCE] 本次实验 cap={experiment_cap} 与"
              f"注册表 protocol 值 {reg_cap} 不同——按受限上下文 treatment "
              f"如实记录，不得表述为模型上下文上限（模型能力 {max_pos}）",
              file=stream)

    capability = min(max_pos, reg_card)
    if experiment_cap > capability:
        sys.exit(
            f"[CONTEXT-CAP-GATE][FAIL-CLOSED] experiment_input_cap="
            f"{experiment_cap} 超过模型上下文能力 capability="
            f"min(max_position_embeddings={max_pos}, "
            f"model_card_declared_context={reg_card})={capability}"
            f"（model={model_name}）。拒绝启动：实验 cap 不得越过模型位置"
            f"能力；如需短 cap 请走受限上下文补充实验口径并如实标注。"
        )
    print(f"[CONTEXT-CAP-GATE] capability=min({max_pos}, {reg_card})="
          f"{capability} ≥ experiment_input_cap={experiment_cap} → 放行"
          f"（显存能否跑下留给运行时：eager attention 的 L² 显存是实现约束"
          f"，不是位置能力）", file=stream)
    return {
        "capability": capability,
        "max_position_embeddings": max_pos,
        "tokenizer_suggested_max_length": tok_meta,
        "model_card_declared_context": reg_card,
        "experiment_input_cap": experiment_cap,
        # 096：注册表侧字段（run-cap manifest 消费——cap 绑进结果产物
        # 身份；restricted_context_treatment 登记缺省时如实记 null）
        "registry_protocol_cap": reg_cap,
        "registry_restricted_context_treatment": (
            (entry.get("restricted_context_treatment") or {}).get("value")
        ),
        "registered": True,
    }
