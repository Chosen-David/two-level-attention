from types import SimpleNamespace
import weakref

import pytest
import torch

from sparse_attn.ops.static_batch_prefill import consume_prefill_rows
from sparse_attn.ops.tli_static_cache import TLIStaticCache, TLIStaticLayer


def inputs(batch, device='cpu', dtype=torch.bfloat16):
    lengths = [9 + b for b in range(batch)]
    rows = []
    for b, n in enumerate(lengths):
        layers = []
        for li in range(3):
            # Deliberately strided source rows, with distinguishable K/V/layers.
            k = (torch.arange(2*n*8, device=device).reshape(1, 2, n, 8)
                 .to(dtype) + 100*b + 10*li)[..., ::2]
            layers.append(SimpleNamespace(keys=k, values=-k))
        rows.append(SimpleNamespace(layers=layers))
    cache = TLIStaticCache(SimpleNamespace(num_hidden_layers=3), max(lengths)+5, batch)
    return cache, rows, lengths


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('batch', [1, 2, 8])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_layerwise_conversion_bytes_padding_and_release(monkeypatch, device, batch, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cache, rows, lengths = inputs(batch, device, dtype)
    expected = [[(s.keys.clone(), s.values.clone()) for s in row.layers] for row in rows]
    refs = [[(weakref.ref(s.keys), weakref.ref(s.values)) for s in row.layers] for row in rows]
    original = TLIStaticLayer.lazy_initialization
    allocated = []
    def allocate(self, keys):
        li = len(allocated)
        # Before the next destination is allocated, previous source tensors
        # must already be released; merely clearing the rows list is too late.
        assert all(ref() is None for row in refs for pair in row[:li] for ref in pair)
        allocated.append(li)
        original(self, keys)
    monkeypatch.setattr(TLIStaticLayer, 'lazy_initialization', allocate)
    consume_prefill_rows(cache, rows, lengths)
    if device == 'cuda':
        torch.cuda.synchronize()
    assert allocated == [0, 1, 2]
    assert all(ref() is None for row in refs for pair in row for ref in pair)
    for li, dest in enumerate(cache.layers):
        assert dest.cumulative_length == max(lengths)
        for b, n in enumerate(lengths):
            for actual, want in zip((dest.keys, dest.values), expected[b][li]):
                assert torch.equal(actual[b:b+1, :, :n], want)
                assert torch.count_nonzero(actual[b, :, n:]) == 0


@pytest.mark.parametrize('bad', ['length', 'shape', 'dtype', 'layers', 'destination', 'gradient'])
def test_rejects_before_consuming_any_source(bad):
    cache, rows, lengths = inputs(2)
    if bad == 'length': lengths[-1] -= 1
    if bad == 'shape': rows[-1].layers[-1].values = torch.zeros(1)
    if bad == 'dtype': rows[-1].layers[-1].keys = rows[-1].layers[-1].keys.float()
    if bad == 'gradient': rows[-1].layers[-1].keys.requires_grad_()
    if bad == 'layers': rows[-1].layers.pop()
    if bad == 'destination': cache.layers[0].lazy_initialization(rows[0].layers[0].keys.expand(2,-1,-1,-1))
    sources = [(layer, layer.keys, layer.values) for row in rows for layer in row.layers]
    with pytest.raises(ValueError):
        consume_prefill_rows(cache, rows, lengths)
    assert all(layer.keys is k and layer.values is v for layer, k, v in sources)


def test_copy_failure_propagates_instead_of_returning_partial_cache(monkeypatch):
    cache, rows, lengths = inputs(2)
    original = TLIStaticLayer.lazy_initialization
    def fail_second(self, keys):
        if self is cache.layers[1]:
            raise RuntimeError('injected allocation failure')
        original(self, keys)
    monkeypatch.setattr(TLIStaticLayer, 'lazy_initialization', fail_second)
    with pytest.raises(RuntimeError, match='injected allocation'):
        consume_prefill_rows(cache, rows, lengths)
    assert all(row.layers[0].keys is None for row in rows)
    assert all(row.layers[1].keys is not None for row in rows)
