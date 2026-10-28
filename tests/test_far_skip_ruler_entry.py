"""Execute the production staged entry on CPU; this is not CUDA emulation."""
import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.hybrid_batch_cache import HybridBatchLayer
from test_far_skip_compact import make_model


def entry():
    path = Path(__file__).resolve().parents[1] / 'benchmark/RULER/pred_ruler_batch.py'
    tree = ast.parse(path.read_text())
    names = {'_decode_4d_mask', '_decode_batch_chunk', '_decode_batch_graph',
             '_decode_hybrid_batch_chunk'}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    scope = dict(torch=torch, os=os, time=time, MAX_GEN=8, BatchCtx=BatchCtx,
                 batch_decode_scope=batch_decode_scope)
    exec(compile(module, str(path), 'exec'), scope)
    return scope['_decode_batch_chunk']


@torch.inference_mode()
@pytest.mark.parametrize('batch', [1, 2, 8])
def test_ruler_staged_compact_matches_dynamic_and_bounds_cache(monkeypatch, batch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv('TLI_CUDA_GRAPH', '0')
    original_to = torch.Tensor.to

    def cpu_input(self, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', cpu_input)
    try:
        reference, candidate = [make_model('evict', dynamic=False) for _ in range(2)]
        ids = torch.arange(79)[None] % 32
        out = reference(input_ids=ids)
        expected = [out.logits[0, -1].argmax().item()]
        for _ in range(7):
            out = reference(input_ids=torch.tensor([[expected[-1]]]),
                            past_key_values=out.past_key_values)
            expected.append(out.logits[0, -1].argmax().item())
        seen = []

        def check(module, args, kwargs):
            cache = kwargs.get('past_key_values')
            if cache is not None:
                assert cache._far_skip_attention_mask is kwargs['attention_mask']
                assert kwargs['cache_position'].shape == (1,)
                assert all(isinstance(layer, HybridBatchLayer) for layer in cache.layers)
                rings = [row for layer in cache.layers for row in layer.rows]
                assert all(isinstance(row, FarSkipStaticLayer) for row in rings)
                assert all(row.workspace is not None for row in rings)
                assert all(row.keys.shape[2] < 79 for row in rings)
                seen.append(tuple(row.keys.data_ptr() for row in rings))

        handle = candidate.register_forward_pre_hook(check, with_kwargs=True)
        actual = entry()(candidate, SimpleNamespace(pad_token_id=0, eos_token_id=2),
                         [{}]*batch, ids.expand(batch, -1), torch.arange(79)[None].expand(batch, -1),
                         torch.ones_like(ids).expand(batch, -1),
                         [79]*batch, set(), False, fixed_cache=True)
        handle.remove()
        assert actual == [expected]*batch
        assert len(seen) == 7 and len(set(seen)) == 1
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('batch', [1, 2])
def test_ruler_rejects_dynamic_batch_cache_before_prefill(batch):
    model = make_model('evict', dynamic=False)
    with pytest.raises(ValueError, match='fixed_cache=True'):
        entry()(model, None, [{}] * batch, None, None, None, None, None,
                False, fixed_cache=False)
