"""NFA/FlashInfer LongBench 真实质量入口的主机侧契约测试。"""
from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from benchmark.efficiency.benchmark_longbench_quality import (
    _validate_quality_receipt,
    resolve_nfa_quality_config,
)
from sparse_attn.scheduler.scheduler import StopStringMatcher


@pytest.mark.parametrize(
    "subspace,gate",
    [("full", "off"), ("tail", "off"), ("full", "on"), ("tail", "on")],
)
def test_nfa_quality_2x2_uses_one_config_for_patch_and_prefill(subspace, gate):
    patch_args, gate_config = resolve_nfa_quality_config(subspace, gate)
    assert patch_args.nfa_subspace == subspace
    assert patch_args.nfa_enable_layer_skip is (gate == "on")
    assert patch_args.nfa_dyn_dgate_eps == 0.10
    assert patch_args.nfa_dyn_dgate_rows == 256
    assert patch_args.nfa_near_tokens == "off"
    assert patch_args.tia_level2_cmp_ratio == 4
    assert patch_args.tia_level1_topk == 128
    assert patch_args.tia_level2_topk == 1024
    assert patch_args.tia_block_size == 64
    assert patch_args.nfa_near_pages == 32
    assert patch_args.nfa_near_len == 4096
    assert patch_args.nfa_sink_tokens == 4
    assert patch_args.nfa_swa_size == 128
    assert patch_args.nfa_near_method == "avg"
    assert patch_args.nfa_far_method == "minmax"
    assert patch_args.nfa_boundary_bypass is True
    assert gate_config.epsilon == (0.10 if gate == "on" else 0.0)
    assert gate_config.rows == 256


def test_nfa_quality_2x2_rejects_unknown_arm():
    with pytest.raises(ValueError, match="full/tail"):
        resolve_nfa_quality_config("random", "off")
    with pytest.raises(ValueError, match="full/tail"):
        resolve_nfa_quality_config("tail", "tau0.05")


def test_nfa_dynamic_gate_does_not_read_removed_default_static_mask(monkeypatch):
    from sparse_attn import info

    patch_args, _ = resolve_nfa_quality_config("tail", "on")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("dynamic gate must not read legacy static D' mask")

    monkeypatch.setattr(info, "_resolve_skip_content", unexpected_read)
    snapshot = info.resolve_treatment_snapshot(patch_args)
    assert snapshot.manifest["nfa_layer_skip_path"] is None
    assert snapshot.skip_ids is None

    monkeypatch.undo()
    patch_args.nfa_layer_skip_path = "/definitely/missing/nfa-static-mask.json"
    with pytest.raises(SystemExit, match="GATE-FAIL"):
        info.resolve_treatment_snapshot(patch_args)


def test_nfa_gate_off_receipt_requires_all_layers_to_keep():
    receipt = _nfa_receipt("eager")
    receipt.update(
        gate_epsilon=0.0,
        gate_counts={"keep": 36, "near": 0, "far": 0, "both": 0},
        prefill_decisions={
            str(layer): {"skip_near": False, "skip_far": False}
            for layer in range(36)
        },
    )
    _validate_quality_receipt(
        "nfa", "eager", 36, 64, receipt, nfa_gate_epsilon=0.0)
    receipt["gate_counts"]["far"] = 1
    receipt["gate_counts"]["keep"] = 35
    with pytest.raises(RuntimeError, match="gate-off"):
        _validate_quality_receipt(
            "nfa", "eager", 36, 64, receipt, nfa_gate_epsilon=0.0)


def _nfa_receipt(execution):
    physical_steps = 5
    return {
        "generated_tokens": 4,
        "stop_reason": "eos",
        "logical_decode_steps": 3,
        "submitted_decode_forwards": physical_steps,
        "nfa_decode_steps": 3,
        "graph_capture_count": 5 if execution == "graph" else 0,
        "graph_replays": physical_steps if execution == "graph" else 0,
        "graph_eager_fallbacks": 0 if execution == "graph" else physical_steps,
    }


@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_nfa_quality_receipt_accepts_physical_steps_beyond_logical_steps(execution):
    # 慢回执下第三个逻辑 token 已 EOS，但物理上允许多提交两个 forward。
    _validate_quality_receipt("nfa", execution, 36, 64, _nfa_receipt(execution))


def test_nfa_quality_receipt_rejects_logical_or_physical_count_mismatch():
    receipt = _nfa_receipt("graph")
    logical = deepcopy(receipt)
    logical["nfa_decode_steps"] = 4
    with pytest.raises(RuntimeError, match="逻辑 decode 步数"):
        _validate_quality_receipt("nfa", "graph", 36, 64, logical)

    short = deepcopy(receipt)
    short["submitted_decode_forwards"] = 2
    with pytest.raises(RuntimeError, match="实际提交数小于"):
        _validate_quality_receipt("nfa", "graph", 36, 64, short)

    wrong_graph = deepcopy(receipt)
    wrong_graph["graph_replays"] = 3
    with pytest.raises(RuntimeError, match="Graph 计数 mismatch"):
        _validate_quality_receipt("nfa", "graph", 36, 64, wrong_graph)

    wrong_capture = deepcopy(receipt)
    wrong_capture["graph_capture_count"] = 1
    with pytest.raises(RuntimeError, match="Graph 计数 mismatch"):
        _validate_quality_receipt("nfa", "graph", 36, 64, wrong_capture)

    visible_4096 = deepcopy(receipt)
    visible_4096["graph_capture_count"] = 4
    _validate_quality_receipt(
        "nfa", "graph", 36, 64, visible_4096, nfa_visible_len=4096
    )


def test_flashinfer_quality_receipt_accepts_first_token_stop_without_decode():
    receipt = {
        "generated_tokens": 1,
        "stop_reason": "stop_string",
        "logical_decode_steps": 0,
        "prefill_append_calls": 36,
        "python_decode_append_calls": 0,
        "python_decode_wrapper_calls": 0,
        "graph_capture_count": 0,
        "graph_replays": 0,
        "graph_eager_fallbacks": 0,
    }
    _validate_quality_receipt("flashinfer", "graph", 36, 64, receipt)


def test_qwen_stop_string_matches_merged_bpe_and_separate_newline_tokens():
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/wangyuanshuo02/Qwen3-8B", trust_remote_code=True
    )
    merged = tokenizer.encode("。\n", add_special_tokens=False)
    assert len(merged) == 1
    assert tokenizer.decode(merged) == "。\n"

    matcher = StopStringMatcher(tokenizer, ["\n"])
    assert matcher.append_and_match(merged[0])

    separate = tokenizer.encode("x\n", add_special_tokens=False)
    assert len(separate) == 2
    matcher = StopStringMatcher(tokenizer, ["\n"])
    assert not matcher.append_and_match(separate[0])
    assert matcher.append_and_match(separate[1])

    matcher = StopStringMatcher(tokenizer, ["\n"])
    assert not matcher.append_and_match(tokenizer.encode("x", add_special_tokens=False)[0])
