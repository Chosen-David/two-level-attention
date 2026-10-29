import pytest
import torch
from sparse_attn.ops.cuda_graph_state import graph_state, enable_graph, disable_graph

@pytest.mark.parametrize('mode', ['off', 'serial', 'overlap'])
def test_operator_preserves_explicit_graph_state(monkeypatch, mode):
    monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
    disable_graph()
    try:
        enable_graph(1)
        state = graph_state()
        state.pos_buf = torch.tensor([129])
        state.valid_len_buf = torch.tensor([130])
        assert graph_state() is state
        assert graph_state().enabled
        assert graph_state().pos_buf.item() == 129
    finally:
        disable_graph()
    assert not graph_state().enabled

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_serial_incremental_graph_crosses_block_boundary(monkeypatch, dtype):
    import importlib.util
    from pathlib import Path
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    spec = importlib.util.spec_from_file_location('graph_args', Path(__file__).parents[1] / 'exp/bench/test_tli_graph_gates.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'serial')
    idx = TLIIndexer(mod._make_prod_args()); idx.layer_idx = 2; idx.clear()
    ref = TLIIndexer(mod._make_prod_args()); ref.layer_idx = 2; ref.clear()
    k = torch.zeros(1, 193, 8, 128, device='cuda', dtype=dtype)
    k[:, :63].normal_()
    cu = torch.tensor([0, 193], device='cuda', dtype=torch.int32)
    enable_graph(1)
    st = graph_state()
    st.pos_buf = torch.tensor([63], device='cuda')
    st.valid_len_buf = torch.tensor([64], device='cuda')
    st.alive_buf = torch.tensor([True], device='cuda')
    try:
        idx.prepare_index(k, cu)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            actual = idx.prepare_index(k, cu)
        for pos in (63, 64, 65, 127, 128, 192):
            k[:, pos].normal_()
            st.pos_buf.fill_(pos); st.valid_len_buf.fill_(pos + 1)
            g.replay()
            expected = ref._prepare_index_full(k, cu)
            for key in ('k_min', 'k_max', 'k_avg', 'k_qat'):
                if key in expected and expected[key] is not None:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    finally:
        disable_graph()
