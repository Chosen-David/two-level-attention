"""Actual decode output assembly, with the previous cat expression as oracle."""
import inspect
import pytest
import torch

import sparse_attn.ops.hybrid_batch_cache as hybrid
from test_hybrid_batch_cache import build
from test_far_skip_hybrid_cache import make_model, isolated_runtime  # noqa: F401


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('skip', [False, True])
def test_output_assembly_matches_previous_cat(monkeypatch, batch, skip):
    monkeypatch.setattr('sparse_attn.indexer.tli_indexer._HAS_TRITON', False)
    current = hybrid.try_hybrid_batch_decode
    source = inspect.getsource(current)
    new = 'return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)'
    assert new in source
    namespace = dict(current.__globals__)
    exec(compile(source.replace(new, 'return torch.cat(outputs, dim=0)'),
                 'frozen-before-single-output', 'exec'), namespace)
    baseline = namespace['try_hybrid_batch_decode']
    runners = [build(make_model(), [23+b*4 for b in range(batch)], [skip]*batch)
               for _ in range(2)]
    calls = []
    original_cat = torch.cat
    def cat(inputs, *args, **kwargs):
        if inspect.currentframe().f_back.f_code in (current.__code__, baseline.__code__):
            calls.append(len(inputs))
        return original_cat(inputs, *args, **kwargs)
    monkeypatch.setattr(torch, 'cat', cat)
    saved = None
    lengths = [23+b*4 for b in range(batch)]
    for step in range(36):
        alive = [not (step > 5 and b == 0) for b in range(batch)]
        lengths = [n+int(a) for n,a in zip(lengths,alive)]
        values = []
        for runner, fn in zip(runners, (baseline, current)):
            monkeypatch.setattr(hybrid, 'try_hybrid_batch_decode', fn)
            runner.inject([(step+b)%32 for b in range(batch)], lengths, alive)
            calls.clear()
            result = runner.run().logits
            values.append(result.clone())
            assert calls == ([] if fn is current and batch == 1 else [batch])
        torch.testing.assert_close(*values, rtol=0, atol=0)
        if saved is None:
            saved = result, result.clone()
        else:
            torch.testing.assert_close(*saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('skip', [False, True])
def test_single_output_cuda_recapture(monkeypatch, skip):
    current = hybrid.try_hybrid_batch_decode
    namespace = dict(current.__globals__)
    source = inspect.getsource(current).replace(
        'return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)',
        'return torch.cat(outputs, dim=0)')
    exec(compile(source, 'frozen-cat', 'exec'), namespace)
    baseline = namespace['try_hybrid_batch_decode']
    for length in (23, 31):
        runners = [build(make_model(device='cuda'), [length], [skip]) for _ in range(2)]
        for runner, fn in zip(runners, (baseline, current)):
            monkeypatch.setattr(hybrid, 'try_hybrid_batch_decode', fn)
            runner.inject([1], [length+1])
            runner.capture()
        for step in range(36):
            values = []
            for runner in runners:
                if step:
                    runner.inject([step%32], [length+1+min(step,30)], [step<=30])
                values.append(runner.run().logits.clone())
            torch.testing.assert_close(*values, rtol=0, atol=0)
