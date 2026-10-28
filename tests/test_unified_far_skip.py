"""Cross-PR workspace ownership: lazy B=1 and preseeded hybrid rows."""
import torch

from test_far_skip_hybrid_cache import (  # noqa: F401
    isolated_runtime, make_model, make_runner, prefill,
)
from sparse_attn.ops.hybrid_batch_cache import HybridBatchCacheBuilder


def test_graph_workspace_is_lazy_for_single_request_and_seeded_once_for_hybrid():
    model = make_model()
    runner = make_runner(model)
    runner.fill(prefill(model, 23), 23)
    ring = runner.cache.layers[0]
    assert ring.workspace is None and ring.eager_workspace is None
    workspace = ring.prepare_graph_workspace()
    assert workspace is ring.prepare_graph_workspace()
    before = [x.clone() for x in workspace.summaries]
    ring.prepare_graph_workspace()
    for actual, expected in zip(workspace.summaries, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    builder = HybridBatchCacheBuilder(model, 128, 2)
    for length in (23, 31):
        builder.add(prefill(model, length), length)
    rows = builder.finish().layers[0].rows
    assert all(row.workspace is not None for row in rows)
    assert rows[0].workspace is not rows[1].workspace
    assert all(row.eager_workspace is None for row in rows)
    assert rows[0].workspace.summaries[0].data_ptr() != rows[1].workspace.summaries[0].data_ptr()
