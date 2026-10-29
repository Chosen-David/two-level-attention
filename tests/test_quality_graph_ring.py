"""Quality entry must capture the ring consumer, preserving request ownership."""
import copy
import pytest
import torch
from test_far_skip_hybrid_cache import make_model, isolated_runtime
from sparse_attn.ops.quality_graph import from_dynamic
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA required')
@pytest.mark.parametrize('mode', ['off', 'serial', 'overlap'])
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@torch.inference_mode()
def test_quality_ring_capture_and_fresh_request(monkeypatch, mode, backend):
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
            monkeypatch.setenv('TLI_CUDA_GRAPH', '1')
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
