"""Static per-layer near/far skip decisions for frozen-region NFA.

The frozen-region runner can consume a precomputed JSON artifact instead of
profiling attention during prefill.  This module owns the on-disk format,
validation, and atomic save/load helpers.
"""
from dataclasses import dataclass
import json
import os


STATIC_GATE_SCHEMA = "nfa_static_gate_v1"


@dataclass(frozen=True)
class StaticGateDecision:
    model_type: str
    num_layers: int
    skip_far: tuple
    skip_near: tuple
    gate: dict
    source: str = ""
    skip_far_frequency: tuple = ()
    skip_near_frequency: tuple = ()
    tau: float = 0.5

    def __post_init__(self):
        if self.num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if len(self.skip_far) != self.num_layers:
            raise ValueError(
                f"skip_far length {len(self.skip_far)} != num_layers "
                f"{self.num_layers}")
        if len(self.skip_near) != self.num_layers:
            raise ValueError(
                f"skip_near length {len(self.skip_near)} != num_layers "
                f"{self.num_layers}")
        for name, values in (("skip_far", self.skip_far),
                             ("skip_near", self.skip_near)):
            if not all(type(value) is bool for value in values):
                raise ValueError(f"{name} must contain only booleans")
        for name, values in (("skip_far_frequency", self.skip_far_frequency),
                             ("skip_near_frequency", self.skip_near_frequency)):
            if not values:
                continue
            if len(values) != self.num_layers:
                raise ValueError(
                    f"{name} length {len(values)} != num_layers "
                    f"{self.num_layers}")
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       for value in values):
                raise ValueError(f"{name} must contain numeric frequencies")
        if not isinstance(self.tau, (int, float)):
            raise ValueError("tau must be numeric")
        if not isinstance(self.gate, dict):
            raise ValueError("gate must be a dict")


def default_static_gate_path(model_type, repo_root=None):
    """Default tracked location for a model-specific static gate artifact."""
    if repo_root is None:
        repo_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    return os.path.join(repo_root, "exp", "trace", "results",
                        f"nfa_static_gate_{model_type}.json")


def _parse_static_gate(path, data):
    if data.get("schema") != STATIC_GATE_SCHEMA:
        raise ValueError(
            f"{path}: schema 应为 {STATIC_GATE_SCHEMA}，实际 "
            f"{data.get('schema')!r}")
    model_type = data.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"{path}: model_type 非法")
    num_layers = data.get("num_layers")
    if type(num_layers) is not int:
        raise ValueError(f"{path}: num_layers 非法")
    gate = data.get("gate", {})
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: gate 非法")
    skip_far = data.get("skip_far")
    skip_near = data.get("skip_near")
    if not isinstance(skip_far, list) or not isinstance(skip_near, list):
        raise ValueError(f"{path}: skip_far/skip_near 非法")

    skip_far_frequency = data.get("skip_far_frequency")
    skip_near_frequency = data.get("skip_near_frequency")
    tau_value = data.get("tau", gate.get("tau"))
    tau = float(0.5 if tau_value is None else tau_value)
    # 旧 artifact 同时保存布尔决策和诊断频率；没有显式 tau 时必须以
    # 布尔决策为准，不能在加载时静默改变 treatment 身份。
    if tau_value is not None:
        if isinstance(skip_far_frequency, list) and len(skip_far_frequency) == num_layers:
            skip_far = [float(x) >= tau for x in skip_far_frequency]
        if isinstance(skip_near_frequency, list) and len(skip_near_frequency) == num_layers:
            skip_near = [float(x) >= tau for x in skip_near_frequency]

    try:
        decision = StaticGateDecision(
            model_type=model_type,
            num_layers=num_layers,
            skip_far=tuple(bool(x) for x in skip_far),
            skip_near=tuple(bool(x) for x in skip_near),
            gate=gate,
            source=str(data.get("source", "")),
            skip_far_frequency=(
                tuple(float(x) for x in skip_far_frequency)
                if isinstance(skip_far_frequency, list)
                else ()),
            skip_near_frequency=(
                tuple(float(x) for x in skip_near_frequency)
                if isinstance(skip_near_frequency, list)
                else ()),
            tau=tau,
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return decision


def load_static_gate(model_type, path=None, repo_root=None):
    """Load a static gate artifact, or return None when absent."""
    path = path or default_static_gate_path(model_type, repo_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"[GATE-FAIL] static-gate 文件不可读/解析失败（{path} → "
            f"{type(exc).__name__}: {exc}）——文件型配置必须绑定实际内容，"
            f"拒绝运行") from exc
    try:
        decision = _parse_static_gate(path, data)
    except ValueError as exc:
        raise SystemExit(f"[GATE-FAIL] {exc}") from exc
    if decision.model_type != model_type:
        raise SystemExit(
            f"[GATE-FAIL] static-gate model_type={decision.model_type} 与当前 "
            f"模型 {model_type} 不一致（{path}）")
    return decision


def save_static_gate(decision, path=None, repo_root=None):
    """Atomically write a static gate artifact."""
    if not isinstance(decision, StaticGateDecision):
        raise TypeError("decision must be StaticGateDecision")
    path = path or default_static_gate_path(decision.model_type, repo_root)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema": STATIC_GATE_SCHEMA,
        "model_type": decision.model_type,
        "num_layers": decision.num_layers,
        "skip_far": list(decision.skip_far),
        "skip_near": list(decision.skip_near),
        "gate": decision.gate,
        "source": decision.source,
        "tau": float(decision.tau),
    }
    if decision.skip_far_frequency:
        payload["skip_far_frequency"] = list(decision.skip_far_frequency)
    if decision.skip_near_frequency:
        payload["skip_near_frequency"] = list(decision.skip_near_frequency)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def static_gate_from_audit(model_type, num_layers, audit, gate, source=""):
    """Build a decision from a frozen-runner prefill audit."""
    decisions = audit.get("decisions", {})
    if len(decisions) != num_layers:
        raise ValueError(
            f"audit has {len(decisions)} decisions, expected {num_layers}")
    return StaticGateDecision(
        model_type=model_type,
        num_layers=num_layers,
        skip_far=tuple(bool(decisions[i]["skip_far"]) for i in range(num_layers)),
        skip_near=tuple(bool(decisions[i]["skip_near"]) for i in range(num_layers)),
        gate=dict(gate),
        source=source,
    )


def static_gate_from_frequency(
        model_type, num_layers, skip_far_frequency, skip_near_frequency,
        gate, source="", tau=0.5):
    """Build frequency-threshold decisions, preserving frequencies for reuse."""
    skip_far = tuple(bool(float(x) >= tau) for x in skip_far_frequency)
    skip_near = tuple(bool(float(x) >= tau) for x in skip_near_frequency)
    gate = dict(gate)
    gate["statistic"] = "frequency"
    gate["tau"] = float(tau)
    return StaticGateDecision(
        model_type=model_type,
        num_layers=num_layers,
        skip_far=skip_far,
        skip_near=skip_near,
        gate=gate,
        source=source,
        skip_far_frequency=tuple(float(x) for x in skip_far_frequency),
        skip_near_frequency=tuple(float(x) for x in skip_near_frequency),
        tau=float(tau),
    )
