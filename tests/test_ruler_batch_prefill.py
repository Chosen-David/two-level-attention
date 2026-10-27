"""Tiny HF model coverage of batch producer prefill, without model downloads."""
from types import SimpleNamespace

import pytest
import torch

from benchmark.RULER import pred_ruler_batch as runner
from test_far_skip_hybrid_cache import make_model


@pytest.mark.parametrize('mode', ['compact', 'fixed', 'dynamic'])
@torch.no_grad()
def test_batch_prefill_honors_last_logit_request(mode, monkeypatch):
    model = make_model()
    if mode != 'compact':
        idx = model.model.layers[0].self_attn.indexer
        idx.far_skip_backend = 'legacy'
        idx._skip_ids = None
    monkeypatch.setattr(runner, 'MAX_GEN', 2)
    monkeypatch.setenv('TLI_CUDA_GRAPH', '0')
    # Existing producer stages on CUDA; map only this transfer in CPU coverage.
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    seen = []
    def observe(module, args, kwargs, output):
        if kwargs['input_ids'].shape[-1] > 1:
            seen.append(output.logits.shape[1])
    handle = model.register_forward_hook(observe, with_kwargs=True)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    try:
        runner._decode_batch_chunk(
            model, SimpleNamespace(pad_token_id=0, eos_token_id=31), [{}],
            ids, torch.arange(7)[None], torch.ones_like(ids), [7], [31], True,
            fixed_cache=mode != 'dynamic')
    finally:
        handle.remove()
    assert seen == [1]
