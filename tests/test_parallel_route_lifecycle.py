"""Production-function route contracts; no CUDA correctness/performance claim."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.ops.cuda_graph_state import graph_state, disable_graph


@pytest.fixture
def route(monkeypatch):
    import sparse_attn.patches.qwen3_attn_patch as module
    source = Path(os.environ.get('TLI_ROUTE_SOURCE', module.__file__))
    names = {'_batch_device_lengths', '_parallel_batch_enabled', '_tl_decode_batched'}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(module.__dict__)
    scope['__name__'] = module.__name__
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    monkeypatch.setenv('TLI_BATCH_INDEXER', 'parallel')
    disable_graph()
    yield scope
    disable_graph()


def test_lengths_refresh_and_share_within_step(route):
    ctx = SimpleNamespace(seq_lens=[7, 13])
    get = route['_batch_device_lengths']
    old = get(ctx, torch.device('cpu'))
    assert get(ctx, torch.device('cpu')) is old
    ctx.seq_lens[:] = [8, 13]  # second row finished
    new = get(ctx, torch.device('cpu'))
    assert new.tolist() == [8, 13]
    assert old.tolist() == [7, 13]
    assert get(ctx, torch.device('cpu')) is new


def test_parallel_graph_uses_live_graph_route(route, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('graph entered host-length eager selector')
    monkeypatch.setattr('sparse_attn.indexer.tli_batch.prepare_mask_batched', forbidden)
    gs = graph_state()
    gs.enabled = True
    gs.valid_len_buf = torch.tensor([8, 13])
    sentinel = object()
    route['_tl_decode_batched_graph'] = lambda *args: sentinel
    q = torch.zeros(2, 1, 1, 128)
    k = torch.zeros(2, 1, 16, 128)
    result = route['_tl_decode_batched'](
        SimpleNamespace(scaling=1., layer_idx=4), SimpleNamespace(dyn_dgate=False),
        q, k, k, SimpleNamespace(seq_lens=[7, 12]))
    assert result is sentinel


@pytest.mark.parametrize('batch', [1, 2])
def test_parallel_receipt_records_layer_number(route, monkeypatch, batch):
    indexer = SimpleNamespace(dyn_dgate=False,
                             _batch_tli=SimpleNamespace(cu=torch.arange(batch+1)*16))
    monkeypatch.setattr('sparse_attn.indexer.tli_batch.prepare_mask_batched',
                        lambda *args: (torch.ones(batch, 1, 1, 16, dtype=torch.bool), 1))
    recorded = []
    route['_paper_route'] = lambda layer, name: recorded.append((layer, name))
    route['eager_decoding_attn'] = lambda *args, **kwargs: args[0]
    q, k = torch.zeros(batch, 1, 1, 128), torch.zeros(batch, 1, 16, 128)
    route['_tl_decode_batched'](SimpleNamespace(scaling=1., layer_idx=4), indexer,
                               q, k, k, SimpleNamespace(seq_lens=[8]*batch))
    assert recorded == [(4, 'tli_batch_indexer_parallel_eager')]
