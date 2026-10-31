"""Query projection sharing: exact values and per-invocation ownership.

CPU tests mock only GPU score/selection consumers. CUDA variants additionally
exercise input updates and recapture; full native consumers have separate tests.
"""
from types import SimpleNamespace
import pytest
import torch
from sparse_attn.ops import batch_candidate_producer as producer


@pytest.mark.parametrize('batch', [2, 8, 32])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('mode', ['same', 'full', 'different', 'equal_distinct', 'qat'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_projection_values_alias_and_new_invocation(monkeypatch, batch, dtype, mode, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    coarse = torch.arange(96, 128, device=device)
    qat = torch.arange(0, 32, device=device)
    idx = SimpleNamespace(
        args=SimpleNamespace(tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024),
        nt_near_pages=32, nt_near_len=4096, sink_tokens=4, sliding_window_size=128,
        far_method='minmax', near_method='avg', enable_subspace=mode != 'qat',
        l2_subspace='full' if mode == 'full' else 'same',
        _subspace_indices=lambda device: coarse, _qat_indices=lambda device: qat)
    k = torch.empty(batch, 192, 2, 128, device=device, dtype=dtype)
    front = producer.BatchedCandidateProducer(idx, k)
    if mode == 'different':
        front.fine_dims = qat
    if mode == 'equal_distinct':
        front.fine_dims = coarse.clone()
    geometry = SimpleNamespace(lower=None, upper=None, quota=None)
    monkeypatch.setattr(front, 'geometry', lambda lengths: geometry)
    seen = []
    def score(*args, **kwargs):
        query = args[-1]
        seen.append(query)
        return query.sum(-1, keepdim=True).expand(batch, 8, 3).to(dtype)
    monkeypatch.setattr(producer, 'score_coarse_batched', score)
    monkeypatch.setattr(producer, 'score_average_batched', score)
    monkeypatch.setattr(producer, 'select_candidates', lambda *args, **kw: (None, None, None))
    summaries = dict(k_min=None, k_max=None, k_avg=True)
    # Noncontiguous source queries; scale rounding precedes FP32 projection.
    q = torch.randn(batch, 1, 8, 256, device=device, dtype=dtype)[..., ::2]
    original = q.clone()
    old_result = None
    for step in range(2):
        seen.clear()
        if step:
            q.add_(0.25)
        full = (q[:, 0] * (128**-.5)).float()
        expected_c = full if front.coarse_dims is None else full[..., front.coarse_dims]
        expected_f = full if front.fine_dims is None else full[..., front.fine_dims]
        result = front(q, k, summaries, None, 128**-.5)
        torch.testing.assert_close(seen[0], expected_c, rtol=0, atol=0)
        torch.testing.assert_close(result.q_fine, expected_f, rtol=0, atol=0)
        assert seen[0] is seen[1]
        assert (result.q_fine is seen[0]) == (front.fine_dims is front.coarse_dims)
        if not step:
            torch.testing.assert_close(q, original, rtol=0, atol=0)
        if old_result is not None:
            assert result.q_fine.data_ptr() != old_result.q_fine.data_ptr()
        old_result = result
    if device == 'cuda':
        # The producer performs no cross-call query caching; each capture owns
        # its output and reads the current input on subsequent replay.
        graphs = []
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = front(q, k, summaries, None, 128**-.5)
            graphs.append((graph, captured))
        for graph, captured in graphs:
            q.add_(0.5)
            graph.replay()
            full = (q[:, 0] * (128**-.5)).float()
            expected = full if front.fine_dims is None else full[..., front.fine_dims]
            torch.testing.assert_close(captured.q_fine, expected, rtol=0, atol=0)
