"""NFA 端到端性能入口的长上下文身份与长度门禁。"""
import json
from types import SimpleNamespace

import pytest
import torch

import benchmark.efficiency.benchmark_nfa_end_to_end as benchmark
from benchmark.efficiency.benchmark_nfa_end_to_end import (
    _fa2_unpacked_position_ids, _load_input)


class _StubTokenizer:
    def __init__(self, length):
        self.length = length

    def __call__(self, text, return_tensors, truncation):
        assert return_tensors == 'pt'
        assert truncation is False
        return SimpleNamespace(
            input_ids=torch.zeros((1, self.length), dtype=torch.long))


def _args(root, provenance='native-ruler'):
    return SimpleNamespace(
        data_root=str(root), context_length=32768, task='niah_single_1',
        sample_index=0, input_provenance=provenance)


def _write_input(root):
    directory = root / '32768'
    directory.mkdir(parents=True)
    (directory / 'niah_single_1.jsonl').write_text(
        json.dumps({'index': 7, 'input': 'stub'}) + '\n', encoding='utf-8')


def test_input_receipt_binds_explicit_native_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    _write_input(tmp_path)
    ids, receipt = _load_input(_args(tmp_path), _StubTokenizer(32612))
    assert ids.shape == (32612,)
    assert receipt['actual_prompt_tokens'] == 32612
    assert receipt['sample_id'] == 7
    assert receipt['provenance'] == 'native-ruler'


def test_short_placeholder_cannot_impersonate_long_context(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    _write_input(tmp_path)
    with pytest.raises(ValueError, match='prompt length mismatch'):
        _load_input(_args(tmp_path), _StubTokenizer(12))


def test_fa2_unpacked_mode_preserves_native_attention_and_restores_globals():
    from transformers import modeling_flash_attention_utils as fa_utils
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original_check = fa_utils._is_packed_sequence
    original_attention = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
    with _fa2_unpacked_position_ids():
        assert fa_utils._is_packed_sequence(None, 1) is False
        assert ALL_ATTENTION_FUNCTIONS['flash_attention_2'] is original_attention
    assert fa_utils._is_packed_sequence is original_check


@pytest.mark.parametrize('execution', ['eager', 'graph'])
def test_nfa_decode_elementwise_patches_are_execution_independent(
        monkeypatch, execution):
    """防止仅 Graph 启用融合链，重新引入跨执行数值分叉。"""
    config = SimpleNamespace(model_type='qwen3', num_hidden_layers=36)
    model = SimpleNamespace(
        cuda=lambda: model,
        eval=lambda: model,
    )
    args = SimpleNamespace(
        model_path='stub', context_length=32768,
        method='nfa', execution=execution)
    captured = {}

    monkeypatch.setattr(
        benchmark.AutoConfig, 'from_pretrained', lambda path: config)
    monkeypatch.setattr(
        benchmark.AutoModelForCausalLM, 'from_pretrained',
        lambda *args, **kwargs: model)
    monkeypatch.setattr(benchmark, 'configure_rope', lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, 'verify_runtime_rope', lambda model: None)
    monkeypatch.setattr(benchmark, 'nfa_args', lambda: object())
    monkeypatch.setattr(
        benchmark, 'resolve_treatment_snapshot', lambda args: object())

    def register_patch(model, args, snapshot, **kwargs):
        captured.update(kwargs)
        return config.num_hidden_layers

    monkeypatch.setattr(benchmark, 'register_patch', register_patch)
    assert benchmark._load_model(args, 512) is model
    assert captured == {'rmsnorm_decode': True, 'mlp_silu_decode': True}
