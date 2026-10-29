"""Request-boundary failure checks use CPU flags; no GPU experiments."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from benchmark.LongBench.prefill_frozen_skip import generate, generate_native
from sparse_attn.ops.frozen_region_gate import GateConfig
from sparse_attn.ops import eager_decoding


@pytest.fixture
def guarded_request(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    flag = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(eager_decoding, '_ERR_FLAG', {torch.device('cpu'): flag})
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=17, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: ' '.join(map(str, ids)))
    def run(kind):
        if kind == 'segmented':
            return generate(model, tokenizer, [3, 4, 5, 6], 3, GateConfig(), 'cpu')
        return generate_native(model, [3, 4, 5, 6], 3, 'cpu')
    yield model, flag, run
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('kind', ['segmented', 'native'])
@pytest.mark.parametrize('phase', [1, 2])
@pytest.mark.parametrize('error_code', [1, 2, 3])
def test_deferred_error_prevents_return_and_next_request_recovers(guarded_request, kind, phase, error_code):
    model, flag, run = guarded_request
    calls = 0
    original_sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']
    def inject(module, args, output):
        nonlocal calls
        calls += 1
        if calls == phase:
            flag.fill_(error_code)
    hook = model.register_forward_hook(inject)
    completed = []
    try:
        with pytest.raises(RuntimeError, match='eager_decoding_attn'):
            completed.append(run(kind))
    finally:
        hook.remove()
    assert completed == []  # Caller cannot checkpoint an invalid result.
    assert flag.item() == 0
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original_sdpa
    result = run(kind)
    assert len(result['generated_ids'] if kind == 'segmented' else result) == 3


@pytest.mark.parametrize('kind', ['segmented', 'native'])
def test_prefill_exception_restores_profile_hook_and_clears_stale_flag(guarded_request, kind):
    model, flag, run = guarded_request
    flag.fill_(2)  # Request entry must clear a previous request's stale flag.
    original_sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']
    def fail(module, args):
        assert flag.item() == 0
        raise ValueError('synthetic prefill failure')
    hook = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(ValueError, match='synthetic prefill failure'):
            run(kind)
    finally:
        hook.remove()
    assert flag.item() == 0
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original_sdpa
    run(kind)
