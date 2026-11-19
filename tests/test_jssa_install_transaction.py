"""Real CPU models: reject invalid calibration before touching any layer."""
import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp/vendor_baselines/jssa_vendor'))
from qwen3_jssa_patch import enable_jssa as enable_qwen
from llama3_jssa_patch import enable_jssa as enable_llama


def make_model(family):
    config_cls, model_cls = ((Qwen3Config, Qwen3ForCausalLM) if family == 'qwen'
                             else (LlamaConfig, LlamaForCausalLM))
    cfg = config_cls(vocab_size=32, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(11)
    return model_cls(cfg).eval()


def decode(model):
    with torch.no_grad():
        prefill = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        return model(torch.tensor([[4]]), past_key_values=prefill.past_key_values,
                     use_cache=True).logits


@pytest.mark.parametrize('family', ['qwen', 'llama'])
@pytest.mark.parametrize('bad', ['missing', 'shape', 'duplicate_layer', 'count', 'malformed'])
def test_invalid_install_leaves_model_untouched(tmp_path, family, bad):
    model = make_model(family)
    expected = decode(model)
    modules = [layer.self_attn for layer in model.model.layers]
    before = [dict(vars(module)) for module in modules]
    basis = {i: torch.eye(8)[:4].repeat(2, 1, 1) for i in range(3)}
    if bad == 'missing':
        del basis[2]
    elif bad == 'shape':
        basis[2] = torch.zeros(2, 3, 8)
    elif bad == 'duplicate_layer':
        modules[2].layer_idx = 0
    elif bad == 'count':
        model.config.num_hidden_layers = 4
    elif bad == 'malformed':
        basis = torch.zeros(2, 4, 8)
    path = tmp_path / 'basis.pt'
    torch.save({'M': basis}, path)
    enable = enable_qwen if family == 'qwen' else enable_llama
    with pytest.raises(ValueError):
        enable(model, str(path), rank=4)
    for module, original in zip(modules, before):
        assert not hasattr(module, 'flash_forward')
        assert not hasattr(module, 'jssa_M')
        assert module.forward.__func__ == type(module).forward
        # No partial adapter attributes were installed before failure.
        assert set(vars(module)) == set(original)
    modules[2].layer_idx = 2
    model.config.num_hidden_layers = 3
    torch.testing.assert_close(decode(model), expected, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['qwen', 'llama'])
def test_valid_all_layer_install_preserves_dense_full_budget(tmp_path, family):
    model = make_model(family)
    reference = copy.deepcopy(model)
    path = tmp_path / 'basis.pt'
    torch.save({'M': {i: torch.eye(8)[:4].repeat(2, 1, 1) for i in range(3)}}, path)
    enable = enable_qwen if family == 'qwen' else enable_llama
    assert enable(model, str(path), rank=4, budget=1024) == 3
    torch.testing.assert_close(decode(model), decode(reference), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('family', ['qwen', 'llama'])
def test_reinstall_rejected_without_losing_original_forward(tmp_path, family):
    model = make_model(family)
    path = tmp_path / 'basis.pt'
    torch.save({'M': {i: torch.eye(8)[:4].repeat(2, 1, 1) for i in range(3)}}, path)
    enable = enable_qwen if family == 'qwen' else enable_llama
    enable(model, str(path), rank=4, budget=1024)
    expected = decode(model)
    originals = [layer.self_attn.flash_forward for layer in model.model.layers]
    with pytest.raises(ValueError, match='already installed'):
        enable(model, str(path), rank=4, budget=1024)
    assert [layer.self_attn.flash_forward for layer in model.model.layers] == originals
    torch.testing.assert_close(decode(model), expected, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['qwen', 'llama'])
def test_basis_binds_to_native_layer_id_not_registration_order(tmp_path, family):
    model = make_model(family)
    modules = [layer.self_attn for layer in model.model.layers]
    assert [module.layer_idx for module in modules] == [0, 1, 2]
    basis = {i: torch.full((2, 4, 8), float(i)) for i in range(3)}
    # Register the same blocks out of execution order; calibration IDs still
    # address native attention layers. No forward is run on this container.
    container = torch.nn.Module()
    container.config = model.config
    for i in (2, 0, 1):
        container.add_module(f'block{i}', modules[i])
    path = tmp_path / 'basis.pt'
    torch.save({'M': basis}, path)
    enable = enable_qwen if family == 'qwen' else enable_llama
    assert enable(container, str(path), rank=4) == 3
    for i, module in enumerate(modules):
        torch.testing.assert_close(module.jssa_M, basis[i], rtol=0, atol=0)
