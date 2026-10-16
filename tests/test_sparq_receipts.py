"""B-track receipt regression: real tiny HF models, no checkpoints or GPU."""
import ast
import copy
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from sparse_attn.baseline_runtime import baseline_request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp/vendor_baselines/sparq_vendor'))
import qwen3_sparq_patch as adapter


def tiny():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).eval()


def test_benchmark_entrypoints_still_use_b_track():
    for path in ('benchmark/LongBench/pred.py', 'benchmark/RULER/pred_ruler.py'):
        tree = ast.parse((ROOT / path).read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and any(a.name == 'enable_sparq' for a in n.names)]
        assert len(imports) == 1
        assert imports[0].module == 'qwen3_sparq_patch'


def test_running_mean_outputs_unchanged_across_requests():
    original = tiny(); model = copy.deepcopy(original)
    adapter.enable_sparq(original, k=2, context_length=16)
    adapter.enable_sparq(model, k=2, context_length=16)
    # Restore the unchanged pre-fix numerical wrapper on the comparison model.
    for layer in original.model.layers:
        layer.self_attn.forward = types.MethodType(adapter._qwen3_sparq_wrapper, layer.self_attn)
    for prompt in ([1, 2, 3, 4, 5], [4, 3, 2, 1]):
        with torch.no_grad(), baseline_request(model) as receipt:
            old = original(input_ids=torch.tensor([prompt]), use_cache=True)
            new = model(input_ids=torch.tensor([prompt]), use_cache=True)
            torch.testing.assert_close(old.logits, new.logits, rtol=0, atol=0)
            for token in (6, 7, 8):
                old = original(input_ids=torch.tensor([[token]]), past_key_values=old.past_key_values)
                new = model(input_ids=torch.tensor([[token]]), past_key_values=new.past_key_values)
                torch.testing.assert_close(old.logits, new.logits, rtol=0, atol=0)
        assert receipt['routes'] == {'native_prefill_calls': 2, 'sparq_decode_calls': 6}
        assert 'running mean' in receipt['identity']['selection']


def test_rejected_decode_and_failed_prefill_do_not_count_or_update_cache():
    model = tiny(); adapter.enable_sparq(model, k=2, context_length=16)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([[1, 2, 3]])).past_key_values
    layer = model.model.layers[0].self_attn
    stats = dict(model._baseline_stats); length = cache.get_seq_length()
    for kwargs in ({'cache_position': None}, {'cache_position': torch.tensor([3]), 'use_cache': False},
                   {'cache_position': torch.tensor([3]), 'attention_mask': torch.full((1, 1, 1, 4), float('-inf'))}):
        with pytest.raises(ValueError):
            layer(hidden_states=torch.zeros(1, 1, 32), position_embeddings=(None, None), past_key_values=cache, **kwargs)
        assert model._baseline_stats == stats
        assert cache.get_seq_length() == length
    with patch.object(layer, 'flash_forward', side_effect=RuntimeError('injected')):
        with pytest.raises(RuntimeError, match='injected'):
            layer(hidden_states=torch.zeros(1, 2, 32), position_embeddings=(None, None))
    assert model._baseline_stats == stats


def test_import_collision_and_invalid_install_are_atomic():
    model = tiny(); original = [l.self_attn.forward for l in model.model.layers]
    core = sys.modules[adapter.sparq_attn.__module__]
    with patch.object(core, '__file__', '/tmp/shadowed_sparq.py'):
        with pytest.raises(RuntimeError, match='collision'):
            adapter.enable_sparq(model)
    for k, context in ((False, 16), (2, -1), (1.5, 16)):
        with pytest.raises(ValueError):
            adapter.enable_sparq(model, k=k, context_length=context)
    assert [l.self_attn.forward for l in model.model.layers] == original
    assert not hasattr(model, '_baseline_stats')
    adapter.enable_sparq(model)
    installed = [l.self_attn.forward for l in model.model.layers]
    with pytest.raises(ValueError, match='already installed'):
        adapter.enable_sparq(model)
    assert [l.self_attn.forward for l in model.model.layers] == installed
