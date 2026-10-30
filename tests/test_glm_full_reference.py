"""Explicit full-width diagnostic: independent score oracle and state guards."""
import pytest
import torch
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.frozen_region_tli import tli_args, region_mask

@pytest.mark.parametrize("length", [127, 4352, 4353])
@torch.inference_mode()
def test_full_width_scores_and_region_mask(monkeypatch, length):
    monkeypatch.setenv("TLI_FINE_GATHER", "0")
    monkeypatch.setenv("TLI_OPERATOR_MODE", "off")
    monkeypatch.setenv("TLI_CUDA_GRAPH", "0")
    torch.set_num_threads(2)
    torch.manual_seed(123)
    idx = TLIIndexer(tli_args("full256-reference"))
    idx.head_dim = 256
    idx.layer_idx = 0
    # Signal lives outside tail32: accidental fallback cannot pass the oracle.
    k = torch.randn(1, length, 2, 256)
    q = torch.randn(1, 1, 2, 256)
    q[..., 192:] = 0
    cu = torch.tensor([0, length], dtype=torch.int32)
    qi = torch.tensor([length - 1], dtype=torch.int32)
    d = idx.prepare_index(k, cu)
    assert idx._inc is None
    assert d["k_min"].shape[-1] == 256
    scores = idx.compute_score(q, qi, d, 0.0625)
    expected = torch.einsum("bqhd,bkhd->bqhk", q, k) * 0.0625
    torch.testing.assert_close(scores["score_fine"], expected)
    mask = idx.compute_mask(qi, scores)
    actual, receipt = region_mask(idx, q, k, list(range(length)), length, 0.0625)
    assert receipt["dimensions"] == list(range(256))
    assert receipt["l2_subspace"] == "full" and not receipt["l2_quant"]
    assert torch.equal(mask, actual)

@pytest.mark.parametrize("key,value", [("TLI_CUDA_GRAPH", "1"), ("TLI_OPERATOR_MODE", "required")])
def test_full_reference_rejects_unverified_backend(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    idx = TLIIndexer(tli_args("full256-reference"))
    idx.head_dim = 256
    with pytest.raises(ValueError, match="eager operator-off"):
        idx.prepare_index(torch.zeros(1, 8, 2, 256), torch.tensor([0, 8]))

@torch.inference_mode()
def test_full_reference_valid_length(monkeypatch):
    monkeypatch.setenv("TLI_OPERATOR_MODE", "off")
    monkeypatch.setenv("TLI_CUDA_GRAPH", "0")
    idx = TLIIndexer(tli_args("full256-reference"))
    idx.head_dim = 256
    idx.layer_idx = 0
    d = idx.prepare_index(torch.randn(1, 256, 2, 256), torch.tensor([0, 256]), valid_length=127)
    assert d["valid_length"] == 127
    assert idx._inc is None
