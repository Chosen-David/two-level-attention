import ast
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def test_prefill_last_logits_and_kv_match_full_output():
    torch.manual_seed(27)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=37, hidden_size=32,
        intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16)).eval()
    x = torch.arange(63)[None] % 37
    with torch.no_grad():
        full = model(x, use_cache=True)
        last = model(x, use_cache=True, logits_to_keep=1)
    assert last.logits.shape == (1, 1, 37)
    torch.testing.assert_close(last.logits, full.logits[:, -1:], rtol=1e-5, atol=1e-6)
    for a, b in zip(full.past_key_values.layers, last.past_key_values.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_shipped_hybrid_prefill_limits_lm_head():
    path = Path(__file__).resolve().parents[1] / 'benchmark/RULER/pred_ruler_batch.py'
    module = ast.parse(path.read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
              and n.name == '_decode_hybrid_batch_chunk')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'model']
    assert calls
    for call in calls:
        assert any(k.arg == 'logits_to_keep' and isinstance(k.value, ast.Constant)
                   and k.value.value == 1 for k in call.keywords)
