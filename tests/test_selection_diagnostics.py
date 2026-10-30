import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "selection_diagnostics", Path(__file__).parents[1] / "exp/bench/selection_diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_uniform_gqa_valid_length_excludes_padding():
    q = torch.zeros(1, 1, 4, 2)
    k = torch.zeros(1, 5, 2, 2)
    k[:, 3:] = float("nan")
    mask = torch.tensor([[[[True, False, False, True, True],
                           [True, True, False, True, True]]]])
    row = diag.selection_mass(q, k, mask, 1.0, 3)
    assert row["selected_counts"] == [1, 2]
    assert row["retained_mass_by_query_head"] == pytest.approx([1/3, 1/3, 2/3, 2/3])
    assert row["oracle_mass_by_kv_head"] == pytest.approx([1/3, 2/3])


def test_known_distribution_and_empty_selection():
    q = torch.ones(1, 1, 1, 1)
    k = torch.tensor([1., 2., 4.]).log().reshape(1, 3, 1, 1)
    mask = torch.tensor([[[[True, False, False]]]])
    row = diag.selection_mass(q, k, mask, 1.0)
    assert row["retained_mass_by_query_head"] == pytest.approx([1/7])
    assert row["oracle_mass_by_kv_head"] == pytest.approx([4/7])
    assert row["peak_token_by_query_head"] == [2]
    row = diag.selection_mass(q, k, mask & False, 1.0)
    assert row["oracle_mass_by_kv_head"] == [0.0]
    assert row["retained_mass_by_query_head"] == [0.0]


def test_rejects_nonfinite_valid_keys():
    with pytest.raises(ValueError, match="nonfinite"):
        diag.selection_mass(torch.ones(1, 1, 1, 1),
                            torch.full((1, 1, 1, 1), float("nan")),
                            torch.ones(1, 1, 1, 1, dtype=torch.bool), 1.0)


def test_probe_restores_inherited_method_on_error(tmp_path, monkeypatch):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    monkeypatch.setenv("TLI_CUDA_GRAPH", "0")
    prepare, clear = TLIIndexer.prepare_mask, TLIIndexer.clear
    owned = "prepare_mask" in TLIIndexer.__dict__
    with pytest.raises(RuntimeError):
        with diag.record_selection_mass(tmp_path / "rows.jsonl"):
            raise RuntimeError("injected")
    assert TLIIndexer.prepare_mask is prepare
    assert TLIIndexer.clear is clear
    assert ("prepare_mask" in TLIIndexer.__dict__) == owned
