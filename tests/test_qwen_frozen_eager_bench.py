"""真实 Qwen frozen eager harness 的性能状态路由契约。"""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from exp.bench import check_qwen_frozen_eager as bench


@pytest.mark.parametrize('option,expected', [([], 'dual'), (['--performance-state', 'both'], 'both')])
def test_performance_state_reaches_abba_and_profiler(
        monkeypatch, tmp_path, option, expected):
    data = tmp_path/'data'
    (data/'8192').mkdir(parents=True)
    (data/'8192'/'niah_single_1.jsonl').write_text('{"input": "needle"}\n')
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', [
        'check_qwen_frozen_eager.py', '--model', '/unused/model',
        '--data-root', str(data), '--context-length', '8192',
        '--mode', 'performance', '--reps', '1', '--profile',
        '--out', str(out), *option])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    tokenizer = Tokenizer()
    config = SimpleNamespace(max_position_embeddings=32768, num_hidden_layers=1)
    model = SimpleNamespace(config=config)
    model.eval = lambda: model
    monkeypatch.setattr(bench.AutoTokenizer, 'from_pretrained', lambda *args: tokenizer)
    monkeypatch.setattr(bench.AutoConfig, 'from_pretrained', lambda *args: config)
    monkeypatch.setattr(
        bench.AutoModelForCausalLM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(bench, 'tli_args', lambda: SimpleNamespace(method='tli'))
    monkeypatch.setattr(bench, 'resolve_treatment_snapshot', lambda args: object())
    monkeypatch.setattr(bench, 'register_patch', lambda *args: 1)

    calls = []

    def fake_run(model, ids, max_new, arm, state, device, capture=False,
                 forced_tokens=None, profile_path=None):
        calls.append((arm, state, profile_path is not None))
        return dict(
            arm=arm, state=state, generated_ids=[1],
            audit=dict(decisions={}, gate_counts={}, prefill_full_kv=0,
                       frozen_kv=0, final_kv=0, tli_receipts={}),
            timings=dict(tpot_ms=1., prefill_seconds=1.),
            memory=dict(final=dict(allocated_bytes=0, peak_allocated_bytes=0)),
            profiler={})

    monkeypatch.setattr(bench, 'run_request', fake_run)
    bench.main()

    assert calls[:4] == [
        ('reference', expected, False), ('physical', expected, False),
        ('physical', expected, False), ('reference', expected, False)]
    assert calls[4] == ('physical', expected, True)
    assert calls[5] == ('physical', 'keep', False)
    report = json.loads((out/'report.json').read_text())
    assert report['protocol']['performance_state'] == expected
    assert [run['state'] for run in report['performance'][:4]] == [expected]*4
