from types import SimpleNamespace

import pytest
import torch

from sparse_attn.patches.glm4_moe_lite_flashinfer_patch import (
    FlashInferGLM4MoeLiteRuntime,
)
from sparse_attn.patches.qwen3_flashinfer_patch import (
    FlashInferQwen3Runtime,
    _FlashInferMaskMetadata,
)


def test_mask_metadata_supports_old_and_new_transformers_signatures():
    metadata = _FlashInferMaskMetadata(num_layers=2, max_seq_len=128)
    assert metadata.get_seq_length() == 0
    assert metadata.get_mask_sizes(torch.arange(24), 0) == (24, 0)
    assert metadata.get_mask_sizes(24, 0) == (24, 0)

    metadata.seq_length = 24
    assert metadata.get_seq_length() == 24
    assert metadata.get_mask_sizes(torch.tensor([24]), 0) == (128, 0)
    assert metadata.get_mask_sizes(1, 0) == (128, 0)


def test_glm_runtime_passes_expanded_head_dim(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256))
    captured = {}

    def fake_base_init(self, model, batch_size, prefill_length, max_new_tokens,
                       device, page_size=64, head_dim=None):
        captured.update(
            batch_size=batch_size, prefill_length=prefill_length,
            max_new_tokens=max_new_tokens, device=device,
            page_size=page_size, head_dim=head_dim)

    monkeypatch.setattr(FlashInferQwen3Runtime, '__init__', fake_base_init)
    FlashInferGLM4MoeLiteRuntime(model, 1, 32, 8, 'cpu')
    assert captured == dict(
        batch_size=1, prefill_length=32, max_new_tokens=8,
        device='cpu', page_size=64, head_dim=256)


def test_glm_runtime_rejects_mismatched_value_head_dim():
    model = SimpleNamespace(config=SimpleNamespace(
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=128))
    with pytest.raises(ValueError, match='head_dim equality'):
        FlashInferGLM4MoeLiteRuntime(model, 1, 32, 8, 'cpu')
