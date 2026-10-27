"""Device-side generation versus the former host-driven real-model loop."""
import copy

import pytest
import torch

from test_hybrid_batch_cache import (build, buffers, cpu_reference_kernels,
                                    isolated_runtime, make_model)  # noqa: F401


def host_generate(runner, first, limit, eos):
    result = [[token] for token in first]
    lengths = list(runner.ctx.seq_lens)
    alive = [True]*runner.batch
    for _ in range(limit-1):
        if not any(alive):
            break
        lengths = [length+int(active) for length, active in zip(lengths, alive)]
        runner.inject([row[-1] if active else 0 for row, active in zip(result, alive)],
                      lengths, alive)
        tokens = runner.run().logits[:, -1].argmax(-1).tolist()
        for b, token in enumerate(tokens):
            if alive[b]:
                result[b].append(token)
                if token in eos:
                    alive[b] = False
    return result


@pytest.mark.parametrize('limit', [1, 2, 35])
@pytest.mark.parametrize('eos', [(), (0, 3, 11, 20), tuple(range(32))])
def test_generation_matches_host_tokens_eos_and_kv(limit, eos):
    model = make_model()
    lengths, skips, first = [1, 7, 23, 31], [True, False, True, False], [3, 7, 11, 1]
    capacity = max(lengths)+max(1, limit-1)
    reference = build(copy.deepcopy(model), lengths, skips, capacity)
    actual = build(model, lengths, skips, capacity)
    expected = host_generate(reference, first, limit, eos)
    observed = actual.generate(torch.tensor(first), limit, eos, poll_interval=8)
    assert observed == expected
    # Extra iterations until the next EOS poll must preserve *all* dead KV,
    # including native dense layers and both compact and linear row layouts.
    for a, e in zip(buffers(actual), buffers(reference)):
        torch.testing.assert_close(a, e, rtol=3e-5, atol=3e-6, equal_nan=True)


def test_generation_preflight_rejects_before_writes():
    runner = build(make_model(), [23, 31], [True, False], 33)
    before = [t.clone() for t in buffers(runner)]
    with pytest.raises(ValueError, match='capacity'):
        runner.generate([1, 2], 4)
    with pytest.raises(ValueError, match='one first token'):
        runner.generate([1], 2)
    with pytest.raises(ValueError, match='positive'):
        runner.generate([1, 2], 2, poll_interval=0)
    assert not runner.injected
    for a, e in zip(buffers(runner), before):
        torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA graph replay')
@pytest.mark.parametrize('backend', ['sdpa', 'triton'])
def test_cuda_generation_capture_matches_host_and_eos(backend, monkeypatch):
    monkeypatch.setenv('TLI_HYBRID_ATTN', backend)
    model = make_model(device='cuda')
    reference = build(copy.deepcopy(model), [23, 31], [True, False], 65)
    actual = build(model, [23, 31], [True, False], 65)
    expected = host_generate(reference, [3, 7], 35, [0, 3, 11, 20])
    observed = actual.generate(torch.tensor([3, 7], device='cuda'), 35,
                               [0, 3, 11, 20], use_graph=True)
    torch.cuda.synchronize()
    assert observed == expected
    assert actual.generation_graph is not None
    for a, e in zip(buffers(actual), buffers(reference)):
        torch.testing.assert_close(a, e, rtol=3e-4, atol=3e-5, equal_nan=True)
