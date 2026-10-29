"""Quality entry must capture the ring consumer, preserving request ownership."""
import copy
import pytest
import torch
from test_far_skip_hybrid_cache import make_model, isolated_runtime
from sparse_attn.ops.quality_graph import from_dynamic
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA required')
@pytest.mark.parametrize('policy', ['1', 'auto'])
@pytest.mark.parametrize('mode', ['off', 'serial', 'overlap'])
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@torch.inference_mode()
def test_quality_ring_capture_and_fresh_request(monkeypatch, mode, backend, policy):
    monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
    monkeypatch.setenv('TLI_OPERATOR_BACKEND', backend)
    model = make_model(device='cuda').to(torch.bfloat16)
    reference = copy.deepcopy(model)
    # Explicit compact+evict already chosen by the caller, same selector/budget
    # for both references. The optimization must not silently enable it.
    for offset in (0, 7):
        ids = ((torch.arange(16384, device='cuda') + offset) % 32)[None]
        with eager_decoding_request_guard():
            monkeypatch.setenv('TLI_CUDA_GRAPH', '0')
            out = reference(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            expected = [out.logits[:, -1].argmax(-1).item()]
            del out
            for _ in range(3):
                out = reference(input_ids=torch.tensor([[expected[-1]]], device='cuda'),
                                past_key_values=past, use_cache=True, logits_to_keep=1)
                expected.append(out.logits[:, -1].argmax(-1).item())
            del out, past
            monkeypatch.setenv('TLI_CUDA_GRAPH', policy)
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            first = out.logits[:, -1].argmax(-1).item()
            del out
            # Every token is EOS, but the minimum must still produce four.
            actual = from_dynamic(model, past, first, 7, set(range(32)), min_new_tokens=4)
            assert actual == expected
            assert not past.layers
            receipt = model._quality_graph_execution
            assert receipt['actual'] == 'graph' and receipt['captures'] == 1
            assert receipt['replay_steps'] == 3 and receipt['fallback_reason'] is None
            rows = receipt['capture_layers']
            assert len(rows) == 1 and rows[0]['skip_far'] is True
            assert rows[0]['attention_route'] == ('ring_sdpa' if mode == 'off'
                                                  else f'ring_{mode}_{backend}')
            assert rows[0]['ring_capacity'] < 16384


@pytest.mark.parametrize('position', [None, torch.arange(3), torch.arange(4)+1])
@torch.inference_mode()
def test_direct_prefill_position_contract(position):
    from sparse_attn.ops.tli_static_cache import TLIStaticLayer, _DirectPrefillLayer, DirectPrefillStaging
    layer = TLIStaticLayer(8, 1)
    state = _DirectPrefillLayer(layer, needs_ring=False)
    staging = DirectPrefillStaging(None, 4, [state])
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn_like(k)
    if position is not None:
        with pytest.raises(ValueError, match='one-shot'):
            staging.update(0, k, v, {'cache_position': position})
        assert not state.written and layer.keys is None
    else:
        with pytest.raises(ValueError, match='one-shot'):
            staging.update(0, k[:, :, :3], v[:, :, :3], {})
        assert not state.written and layer.keys is None
        staging.update(0, k, v, {})
        assert layer.is_initialized and layer.get_seq_length() == 4
        torch.testing.assert_close(layer.keys[:, :, :4], k, atol=0, rtol=0)
        with pytest.raises(ValueError, match='one-shot'):
            staging.update(0, k, v, {})


def test_static_layer_hf_attention_contract():
    from transformers.cache_utils import CacheLayerMixin
    from sparse_attn.ops.tli_static_cache import TLIStaticLayer
    layer = TLIStaticLayer(8, 1)
    assert isinstance(layer, CacheLayerMixin) and not layer.is_initialized
    k = torch.zeros(1, 2, 3, 8)
    with pytest.raises(ValueError, match='matching'):
        layer.lazy_initialization(k, k.double())
    assert layer.keys is None and not layer.is_initialized
    layer.lazy_initialization(k, k)
    assert layer.is_initialized
    layer._direct_prefill_mask = True
    assert layer.get_mask_sizes(3) == layer.get_mask_sizes(torch.arange(3)) == (3, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA required')
@pytest.mark.parametrize('mode', ['serial', 'overlap'])
@pytest.mark.parametrize('failure_stage', ['allocation', 'receipt'])
@torch.inference_mode()
def test_ring_capture_failure_recovery(monkeypatch, mode, failure_stage):
    from sparse_attn.ops.cuda_graph_state import graph_state
    import sparse_attn.ops.tli_static_cache as storage
    monkeypatch.setenv('TLI_OPERATOR_MODE', mode)
    model = make_model(device='cuda').to(torch.bfloat16)
    ids = (torch.arange(16384, device='cuda') % 32)[None]
    caches = []
    original = storage.fill_static_from_dynamic
    def observed(cache, *args, **kwargs):
        original(cache, *args, **kwargs)
        caches.append(cache)
    monkeypatch.setattr(storage, 'fill_static_from_dynamic', observed)
    def fail(*args):
        raise torch.cuda.OutOfMemoryError('injected ring capture failure')
    outputs = []
    for policy, inject in [('auto', True), ('1', True), ('1', False)]:
        monkeypatch.setenv('TLI_CUDA_GRAPH', policy)
        with eager_decoding_request_guard():
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            first = out.logits[:, -1].argmax(-1).item()
            del out
            with monkeypatch.context() as patch:
                if inject:
                    if failure_stage == 'allocation':
                        patch.setattr(torch.cuda, 'CUDAGraph', fail)
                    else:
                        patch.setattr('sparse_attn.ops.quality_graph._capture_routes', fail)
                if inject and policy == '1':
                    with pytest.raises(torch.cuda.OutOfMemoryError):
                        from_dynamic(model, past, first, 5, set())
                else:
                    outputs.append(from_dynamic(model, past, first, 5, set()))
                    receipt = model._quality_graph_execution
                    if inject:
                        assert receipt['actual'] == 'staged_static'
                        assert receipt['captures'] == receipt['replay_steps'] == 0
                        assert 'capture_layers' not in receipt
                        assert 'capture_failed:OutOfMemoryError' in receipt['fallback_reason']
                    else:
                        assert receipt['actual'] == 'graph' and receipt['replay_steps'] == 4
            assert not graph_state().enabled
            assert caches[-1]._far_skip_attention_mask is None
            assert caches[-1]._quality_decode_mask is None
            assert not past.layers
    assert outputs[0] == outputs[1]
