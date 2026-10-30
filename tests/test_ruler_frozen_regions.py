"""RULER frozen-region opt-in, identity, EOS, and commit-boundary tests."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.RULER import pred_ruler
from sparse_attn.info import get_treatment_manifest_json, resolve_treatment_snapshot
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig


def _args(**overrides):
    values = dict(
        model='Qwen3-8B', model_path='/unused/model', task='niah_single_1',
        context_length=8192, data_root='/unused/data', output_dir='/unused/out',
        max_num=1, yarn=False, yarn_factor=None, method='tli', t='contract',
        pred_postfix='', tli_prefill_frozen_regions=True,
        tli_frozen_region_mode='dual', tli_frozen_region_epsilon=.05,
        tli_frozen_region_statistic='mean', tli_frozen_region_rows=256,
        tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
        tia_level2_cmp_ratio=4, tli_subspace='tail', tli_l2_subspace='same',
        tli_far_method='minmax', tli_near_method='avg', tli_far_select='4bit',
        tli_near_select='4bit', tli_near_pages=32, tli_near_tokens='off',
        tli_near_len=4096, tli_sink_tokens=4, tli_swa_size=128,
        tli_enable_subspace=True, tli_enable_layer_skip=False,
        tli_dyn_dgate=False, tli_far_skip_backend='legacy',
        tli_far_skip_cache='retain', tli_sparse_prefill=False,
        tli_decode_dense=False, tli_per_q_head=False,
        tia_enable_async_topk=False, tli_l2_quant=False,
        tli_sinkswa_only=False, tli_boundary_bypass=True,
        tli_static_pair=False, tli_proj_basis=None,
        tli_alpha=None, tli_beta=None, tli_gamma=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def _config():
    return SimpleNamespace(model_type='qwen3', head_dim=128,
                           num_hidden_layers=1, _attn_implementation='eager')


def test_default_off_does_not_touch_environment_or_config(monkeypatch):
    monkeypatch.setenv('TLI_OPERATOR_MODE', 'operator')
    config = _config()
    assert pred_ruler._configure_frozen_regions(
        _args(tli_prefill_frozen_regions=False), config) is None
    assert os.environ['TLI_OPERATOR_MODE'] == 'operator'
    assert config._attn_implementation == 'eager'


def test_valid_configuration_is_resolved_before_patch_registration(monkeypatch):
    for name in ('TLI_OPERATOR_MODE', 'TLI_FINE_GATHER', 'TLI_SPARSE_PREFILL',
                 'TLI_DECODE_DENSE', 'TLI_FROZEN_EAGER_REFERENCE', 'TLI_CUDA_GRAPH'):
        monkeypatch.delenv(name, raising=False)
    args = _args()
    config = _config()
    gate, geometry = pred_ruler._configure_frozen_regions(args, config)
    assert gate == GateConfig()
    assert geometry == RegionGeometry(max_new_tokens=pred_ruler.MAX_GEN)
    assert config._attn_implementation == 'sdpa'
    assert os.environ['TLI_OPERATOR_MODE'] == 'off'
    assert os.environ['TLI_FROZEN_EAGER_REFERENCE'] == '0'
    assert os.environ['TLI_CUDA_GRAPH'] == 'off'
    assert args._frozen_region_gate is gate
    assert args._frozen_region_geometry is geometry


def test_real_cli_accepts_environment_only_sparse_flags(monkeypatch):
    for name in ('TLI_OPERATOR_MODE', 'TLI_FINE_GATHER', 'TLI_SPARSE_PREFILL',
                 'TLI_DECODE_DENSE', 'TLI_FROZEN_EAGER_REFERENCE', 'TLI_CUDA_GRAPH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, 'argv', [
        'pred_ruler', '--model', 'Qwen3-8B', '--model_path', '/unused/model',
        '--task', 'niah_single_1', '--context_length', '65536',
        '--data-root', '/unused/data', '--output-dir', '/unused/out',
        '--method', 'tli', '--yarn', '--tli_prefill_frozen_regions',
        '--tli_far_method', 'minmax', '--tli_near_method', 'avg',
        '--tli_near_pages', '32', '--tli_near_tokens', 'off',
        '--tli_near_len', '4096', '--tli_subspace', 'tail',
        '--tia_level2_cmp_ratio', '4', '--tli_sink_tokens', '4',
        '--tli_swa_size', '128', '--tia_level1_topk', '128',
        '--tia_level2_topk', '1024', '--tli_enable_layer_skip', 'false',
        '--tli_far_skip_backend', 'legacy', '--tli_far_skip_cache', 'retain',
    ])
    args = pred_ruler.parse_args()
    assert not hasattr(args, 'tli_sparse_prefill')
    assert not hasattr(args, 'tli_decode_dense')
    pred_ruler._configure_frozen_regions(args, _config())
    assert os.environ['TLI_SPARSE_PREFILL'] == '0'
    assert os.environ['TLI_DECODE_DENSE'] == '0'


@pytest.mark.parametrize('field,value', [
    ('method', 'none'), ('tia_level1_topk', 64), ('tli_near_pages', 16),
    ('tli_enable_layer_skip', True), ('tli_dyn_dgate', True),
    ('tli_sparse_prefill', True), ('tli_decode_dense', True),
])
def test_unvalidated_geometry_or_method_fails_closed(field, value):
    with pytest.raises(ValueError):
        pred_ruler._configure_frozen_regions(_args(**{field: value}), _config())


def test_non_qwen_and_graph_execution_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match='Qwen3'):
        pred_ruler._configure_frozen_regions(
            _args(), SimpleNamespace(model_type='llama', head_dim=128))
    monkeypatch.setenv('TLI_CUDA_GRAPH', 'on')
    with pytest.raises(ValueError, match='eager execution only'):
        pred_ruler._configure_frozen_regions(_args(), _config())


def test_manifest_is_byte_identical_when_off_and_tracks_gate_and_geometry():
    args = SimpleNamespace(
        method='tli', tli_enable_layer_skip=False, tli_proj_basis=None,
        tli_far_skip_backend='legacy', tli_far_skip_cache='retain',
        tli_prefill_frozen_regions=False)
    baseline = get_treatment_manifest_json(args, resolve_treatment_snapshot(args))
    args.tli_frozen_region_mode = 'dual'
    args.tli_frozen_region_epsilon = .05
    assert get_treatment_manifest_json(args, resolve_treatment_snapshot(args)) == baseline

    args.tli_prefill_frozen_regions = True
    with pytest.raises(ValueError, match='resolved gate and geometry'):
        resolve_treatment_snapshot(args)
    args._frozen_region_gate = GateConfig()
    args._frozen_region_geometry = RegionGeometry(max_new_tokens=64)
    frozen = resolve_treatment_snapshot(args)
    payload = json.loads(frozen.manifest_json)
    assert payload['tli_prefill_frozen_regions']['gate']['epsilon'] == .05
    assert payload['tli_prefill_frozen_regions']['geometry']['max_new_tokens'] == 64
    assert frozen.manifest_json != baseline
    args._frozen_region_gate = GateConfig(epsilon=.1)
    assert resolve_treatment_snapshot(args).manifest_json != frozen.manifest_json


class _Tokenizer:
    eos_token_id = 1

    def __call__(self, text, truncation, return_tensors):
        return SimpleNamespace(input_ids=torch.tensor([[2, 3, 4]]))

    def encode(self, text, add_special_tokens):
        return []

    def decode(self, ids, skip_special_tokens):
        return ' '.join(map(str, ids))


class _Model:
    def __init__(self, config):
        self.config = config
        self.hf_device_map = {'': 'cpu'}
        self.model = SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(sliding_window=None))])
        self.decode_tokens = iter([5, 1])
        self.decode_calls = []

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values, use_cache, **kwargs):
        self.decode_calls.append(kwargs)
        token = next(self.decode_tokens)
        logits = torch.full((1, 1, 16), -100.)
        logits[0, 0, token] = 0.
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _run_ruler_main(monkeypatch, tmp_path, complete):
    args = _args(model_path=str(tmp_path/'model'), data_root=str(tmp_path/'data'),
                 output_dir=str(tmp_path/'out'))
    data = tmp_path/'data'/'8192'
    data.mkdir(parents=True)
    (data/'niah_single_1.jsonl').write_text(
        json.dumps({'input': 'request', 'outputs': ['answer'], 'length': 3, 'index': 0})+'\n')
    config = _config()
    model = _Model(config)
    monkeypatch.setattr(pred_ruler, 'parse_args', lambda: args)
    monkeypatch.setattr(pred_ruler, 'baseline_preflight', lambda args: None)
    monkeypatch.setattr(pred_ruler, 'seed_everything', lambda: None)
    monkeypatch.setattr(pred_ruler.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(pred_ruler.AutoTokenizer, 'from_pretrained', lambda *a, **k: _Tokenizer())
    monkeypatch.setattr(pred_ruler.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: model)
    monkeypatch.setattr(pred_ruler, 'resolve_far_skip_modes', lambda *a, **k: None)
    monkeypatch.setattr(pred_ruler, 'resolve_treatment_snapshot', lambda args: None)
    monkeypatch.setattr(pred_ruler, 'get_method_name_with_info', lambda *a: 'frozen_contract')
    monkeypatch.setattr(pred_ruler, 'get_treatment_manifest_json', lambda *a: '{}')
    monkeypatch.setattr(pred_ruler, 'build_yarn_receipt', lambda **kwargs: {'status': 'complete'})
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, 'to',
                        lambda tensor, *a, **k: tensor if a == ('cuda',) else original_to(tensor, *a, **k))

    def register(model, args, snapshot):
        # Environment and geometry must already be fixed before indexer construction.
        assert os.environ['TLI_OPERATOR_MODE'] == 'off'
        assert args._frozen_region_geometry.max_new_tokens == pred_ruler.MAX_GEN
        return 1

    def prefill(model, input_ids, gate, geometry):
        cache = object()
        logits = torch.full((1, 1, 16), -100.)
        logits[0, 0, 1] = 0.  # First-token EOS must not stop RULER generation.
        return SimpleNamespace(logits=logits, past_key_values=cache), {'profiles': {}}

    monkeypatch.setattr(pred_ruler, 'register_patch', register)
    monkeypatch.setattr(pred_ruler, 'prefill_frozen_request', prefill)
    monkeypatch.setattr(pred_ruler, 'complete_frozen_request', complete)
    with pytest.raises(ValueError, match='receipt failure') if complete is _fail_complete else _null_context():
        pred_ruler.main()
    out_path = (tmp_path/'out'/'L8192'/'pred'/
                'niah_single_1-frozen_contract-contract.jsonl')
    return model, out_path


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _fail_complete(*args):
    raise ValueError('receipt failure')


def test_ruler_preserves_first_token_eos_and_commits_frozen_audit(monkeypatch, tmp_path):
    def complete(model, cache, generated_count, audit):
        assert generated_count == 3
        return {**audit, 'gate_counts': {'keep': 1}}

    model, out_path = _run_ruler_main(monkeypatch, tmp_path, complete)
    pointer = Path(pred_ruler.generation_pointer_path(str(out_path)))
    generation = pointer.read_text().strip()
    record = json.loads((pointer.parent/generation/out_path.name).read_text())
    assert record['generated_token_ids'] == [1, 5, 1]
    assert record['_frozen_region']['gate_counts'] == {'keep': 1}
    assert all(call['attention_mask'] == {'full_attention': None}
               for call in model.decode_calls)


def test_receipt_failure_writes_no_prediction_and_does_not_commit(monkeypatch, tmp_path):
    _, out_path = _run_ruler_main(monkeypatch, tmp_path, _fail_complete)
    pointer = Path(pred_ruler.generation_pointer_path(str(out_path)))
    assert not pointer.exists()
    generations = list(out_path.parent.glob(out_path.name+'.gen-*'))
    assert len(generations) == 1
    assert (generations[0]/out_path.name).read_text() == ''
