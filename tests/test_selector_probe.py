import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

spec = importlib.util.spec_from_file_location('selector_probe', Path(__file__).parents[1]/'exp/bench/selector_probe.py')
diag = importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


def test_exact_target_and_restore(tmp_path, monkeypatch):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    sentinel = object()
    fn = lambda *a, **kw: sentinel
    monkeypatch.setattr(TLIIndexer, 'compute_score', fn)
    old_mask = TLIIndexer.compute_mask_graph
    idx = SimpleNamespace(layer_idx=14, skip_far=False)
    q=torch.zeros(1,1,2,4);ids=torch.tensor([4]);index={'k_qat':q,'valid_length':5}
    with pytest.raises(RuntimeError,match='injected'):
        with diag.record_selector_probe(tmp_path/'probe',14,1):
            for _ in range(3):
                assert TLIIndexer.compute_score(idx,q,ids,index,0.5) is sentinel
            raise RuntimeError('injected')
    row=torch.load(tmp_path/'probe/eager.pt',weights_only=True)
    assert row['tensors']['probe_length']==5
    assert torch.equal(row['tensors']['q'],q)
    assert TLIIndexer.compute_score is fn
    assert TLIIndexer.compute_mask_graph is old_mask
