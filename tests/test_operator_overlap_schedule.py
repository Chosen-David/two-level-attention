"""Verify fork/join dependencies without requiring a free GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from sparse_attn.ops import operator_pipeline as pipeline


def test_protected_starts_before_index_and_join_follows_selected(monkeypatch):
    events = []
    class Stream:
        def __init__(self, name):
            self.name = name
        def wait_stream(self, other):
            events.append((self.name, 'wait', other.name))
    main, side = Stream('main'), Stream('side')
    class Tensor:
        is_cuda = True
        device = torch.device('cuda:0')
        dtype = torch.float32
        shape = (1, 1, 4, 32)
        def record_stream(self, stream):
            events.append(('record', stream.name))
    tensor = Tensor()
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'overlap')
    monkeypatch.setattr(pipeline, '_protected_stream', lambda device: side)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: main)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch, 'empty', lambda *args, **kw: tensor)
    def partial(*args, protected=False, **kw):
        events.append('protected' if protected else 'selected')
        return (tensor, tensor, tensor)
    def prepare(*args):
        events.append('prepare')
        return {}
    def select(*args):
        events.append('select')
        return SimpleNamespace(indices=tensor, counts=tensor)
    def merge(*args, **kw):
        events.append('merge')
        return tensor
    from sparse_attn.ops import partial_attention as ops
    monkeypatch.setattr(ops, 'partial_attention', partial)
    monkeypatch.setattr(ops, 'merge_partials', merge)
    indexer = SimpleNamespace(sink_tokens=4, sliding_window_size=16,
                              prepare_index=prepare, compute_selection_graph=select)
    assert pipeline.decode_indexed(indexer, None, tensor, tensor, tensor,
                                   tensor, tensor, 0.5) is tensor
    assert events.index(('side', 'wait', 'main')) < events.index('protected')
    assert events.index('protected') < events.index('prepare') < events.index('select')
    assert events.index('select') < events.index('selected')
    assert events.index('selected') < events.index(('main', 'wait', 'side'))
    assert events.index(('main', 'wait', 'side')) < events.index('merge')


def test_invalid_operator_backend_fails_before_dispatch(monkeypatch):
    import pytest
    monkeypatch.setenv('TLI_OPERATOR_BACKEND', 'unknown')
    with pytest.raises(ValueError, match='BACKEND'):
        pipeline.decode_indexed(None, None, None, None, None, None, None, 1.)
