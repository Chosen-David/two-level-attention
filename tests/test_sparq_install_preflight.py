"""Installation validation must reject before mutating any actual HF layer."""
import sys
from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'exp/vendor_baselines/sparq_vendor'))
from qwen3_sparq_patch import enable_sparq, _qwen3_sparq_wrapper
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope


def tiny():
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
        intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16)).eval()


@pytest.mark.parametrize('cause', ['missing', 'oversized_r'])
def test_invalid_install_is_atomic(cause):
    model = tiny()
    if cause == 'missing':
        model.model.layers[1].self_attn = torch.nn.Identity()
    attention = model.model.layers[0].self_attn
    before = attention.forward
    with pytest.raises(ValueError, match='complete|derived r'):
        enable_sparq(model, k=1024 if cause == 'oversized_r' else 2,
                     context_length=16)
    assert attention.forward == before
    assert not hasattr(attention, 'flash_forward')
    assert not hasattr(attention, 'sparq_config')
    assert not hasattr(model, '_baseline_identity')


def test_success_records_actual_r_and_reinstall_does_not_rewrite():
    model = tiny()
    assert enable_sparq(model,k=2,context_length=16)==2
    assert model._baseline_identity['r_by_head_dim']=={'16':4}
    before = model.model.layers[0].self_attn.forward
    with pytest.raises(ValueError,match='already installed'):
        enable_sparq(model,k=2,context_length=16)
    assert model.model.layers[0].self_attn.forward == before


def test_batch_context_is_rejected_before_decode_or_cache_mutation():
    model=tiny()
    enable_sparq(model,k=2,context_length=16)
    attn=model.model.layers[0].self_attn
    with batch_decode_scope(BatchCtx(1,[1],[0],[True],0)):
        with pytest.raises(ValueError,match='batched decode contexts'):
            _qwen3_sparq_wrapper(attn,torch.zeros(1,1,32),None,past_key_values=None)
    assert model._baseline_stats=={}
