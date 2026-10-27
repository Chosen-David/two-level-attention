"""Real tiny HF models: avoid full prompt logits without changing greedy IDs."""
import ast
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM


source = Path(__file__).resolve().parents[1] / 'benchmark/LongBench/pred.py'
tree = ast.parse(source.read_text())
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name == '_manual_generate'], type_ignores=[]),
             str(source), 'exec'), namespace)


@pytest.mark.parametrize('family', ['qwen3', 'llama'])
@torch.no_grad()
def test_prefill_last_logits_preserves_greedy_generation(family):
    torch.manual_seed(27)
    config_cls, model_cls = ((Qwen3Config, Qwen3ForCausalLM) if family == 'qwen3'
                             else (LlamaConfig, LlamaForCausalLM))
    cfg = config_cls(vocab_size=32, hidden_size=64, intermediate_size=96,
                     num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32, max_position_embeddings=128)
    cfg._attn_implementation = 'sdpa'
    model = model_cls(cfg).eval()
    ids = torch.arange(23)[None]
    output = model(ids, use_cache=True)
    expected = [output.logits[:, -1].argmax(-1).item()]
    for _ in range(2):
        output = model(torch.tensor([[expected[-1]]]),
                       past_key_values=output.past_key_values, use_cache=True)
        expected.append(output.logits[:, -1].argmax(-1).item())
    seen = []
    handle = model.register_forward_hook(lambda module, args, output: seen.append(output.logits.shape[1]))
    try:
        result = namespace['_manual_generate'](model, ids, ids[:, :0], 3, None)
    finally:
        handle.remove()
    assert result == expected
    assert seen == [1, 1, 1]
