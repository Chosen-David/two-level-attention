"""Shared frozen prefill/freeze/receipt lifecycle contract tests."""
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attn.cache.paged_region_cache import RequestPhase
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.policies.frozen_gate import GateConfig
from sparse_attn.engine.frozen_runner import (
    complete_frozen_request, decode_frozen_request, prefill_frozen_request)
from sparse_attn.scheduler.scheduler import (
    ContinuousBatchScheduler, greedy_next_tokens)
from sparse_attn.ops.paged_region_kernels_nfa import decode_graph_ready, nfa_args
from sparse_attn.patches import register_patch


def _model():
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=19, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).eval()
    args = nfa_args()
    assert register_patch(model, args) == config.num_hidden_layers
    return model


@torch.inference_mode()
def test_each_prefill_owns_a_complete_fresh_cache_and_json_receipt():
    model = _model()
    ids = torch.tensor([[2, 3, 4, 5]])
    caches = []
    for _ in range(2):
        output, audit = prefill_frozen_request(
            model, ids, GateConfig(mode='full'), RegionGeometry(
                sink=1, near=4, window=2, page=2, max_new_tokens=3))
        cache = output.past_key_values
        caches.append(cache)
        assert set(audit['profiles']) == {0, 1}
        assert set(audit['decisions']) == {0, 1}
        assert not any(d['skip_near'] or d['skip_far']
                       for d in audit['decisions'].values())
        assert cache.states[audit['request_id']].phase == RequestPhase.FREEZE
        receipt = complete_frozen_request(model, cache, 1, audit)
        assert receipt['request_phase'] == 'FREEZE'
        assert receipt['gate_counts'] == {'keep': 2, 'near': 0, 'far': 0, 'both': 0}
        json.dumps(receipt)
    assert caches[0] is not caches[1]
    assert caches[0].layers[0] is not caches[1].layers[0]


@torch.inference_mode()
def test_missing_real_nfa_receipt_is_rejected_before_commit():
    model = _model()
    output, audit = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'),
        RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3))
    model._frozen_nfa = True
    with pytest.raises(ValueError, match='missing real NFA decode receipts'):
        complete_frozen_request(model, output.past_key_values, 2, audit)


@torch.inference_mode()
def test_device_receipt_values_are_converted_once_at_request_boundary():
    model = _model()
    output, audit = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'),
        RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3))
    model._frozen_nfa = True
    cache = output.past_key_values
    cache.decode_trace_enabled = True
    cache.nfa_decode_steps = 1
    cache.visible_lens_buf.copy_(torch.tensor([[2], [1]]))
    for layer in cache.layers:
        layer.nfa_decode_trace = [dict(
            decode_step=0, selected_logical_ids=torch.tensor([[1, 2]]))]
    receipt = complete_frozen_request(model, cache, 2, audit)
    assert receipt['nfa_receipts'][0] == {
        'route': 'graph_ready_sparse', 'visible_lens': [2]}
    assert receipt['nfa_decode_trace'][0][0]['selected_logical_ids'] == [[1, 2]]
    json.dumps(receipt)


def test_decode_graph_ready_has_no_layer_receipt_or_hidden_d2h(monkeypatch):
    layer = SimpleNamespace()
    cache = SimpleNamespace(
        layers=[layer], max_requests=1,
        gather_indices_buf=torch.zeros((1, 2), dtype=torch.int32),
        visible_lens_buf=torch.tensor([2], dtype=torch.int32),
        query_to_req_buf=torch.tensor([0], dtype=torch.int32))
    module = SimpleNamespace(layer_idx=0, scaling=1.)
    query = torch.zeros((1, 1, 1))
    monkeypatch.setattr(
        'sparse_attn.ops.paged_region_kernels.sparse_paged_attention_gather',
        lambda *args, **kwargs: query)

    def forbidden_d2h(*args, **kwargs):
        raise AssertionError('decode forward must not synchronize device data')

    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden_d2h)
    monkeypatch.setattr(torch.Tensor, 'tolist', forbidden_d2h)
    assert decode_graph_ready(module, query, cache) is query
    assert not hasattr(layer, 'nfa_receipt')


@torch.inference_mode()
def test_incomplete_decode_trace_is_rejected_before_commit():
    model = _model()
    output, audit = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'),
        RegionGeometry(sink=1, near=4, window=2, page=2, max_new_tokens=3))
    model._frozen_nfa = True
    cache = output.past_key_values
    cache.decode_trace_enabled = True
    cache.nfa_decode_steps = 1
    for layer in cache.layers:
        layer.nfa_decode_trace = []
    with pytest.raises(ValueError, match='incomplete NFA decode trace at layer 0'):
        complete_frozen_request(model, cache, 3, audit)


def test_decode_trace_comparison_locates_head_and_validates_physical_mapping():
    from exp.bench.check_qwen_frozen_eager import compare_decode_traces
    reference = {'nfa_decode_trace': {0: [
        dict(decode_step=0, selected_logical_ids=[[1, 4, -1], [2, 5, -1]]),
        dict(decode_step=1, selected_logical_ids=[[1, 4, 7], [2, 5, 8]])]}}
    physical = {'nfa_decode_trace': {0: [
        dict(decode_step=0, selected_logical_ids=[[1, 4, -1], [2, 6, -1]],
             physical_mapping_matches=True),
        dict(decode_step=1, selected_logical_ids=[[1, 4, 7], [2, 5, 8]],
             physical_mapping_matches=False)]}}
    comparison = compare_decode_traces(reference, physical)
    assert comparison['compared_steps'] == 2
    assert comparison['compared_kv_head_steps'] == 4
    assert not comparison['all_selected_logical_ids_equal']
    assert not comparison['all_physical_mappings_valid']
    assert comparison['mismatch_count'] == 1
    assert comparison['mismatches'][0] == dict(
        layer_idx=0, decode_step=0, kv_head=1,
        reference_count=2, physical_count=2,
        reference_selected_logical_ids=[2, 5, -1],
        physical_selected_logical_ids=[2, 6, -1])


@torch.inference_mode()
def test_prefill_failure_restores_the_native_sdpa_hook():
    model = _model()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    hook = model.register_forward_pre_hook(
        lambda module, args: (_ for _ in ()).throw(ValueError('synthetic prefill failure')))
    try:
        with pytest.raises(ValueError, match='synthetic prefill failure'):
            prefill_frozen_request(
                model, torch.tensor([[2, 3, 4, 5]]), GateConfig(), RegionGeometry())
    finally:
        hook.remove()
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    output, _ = prefill_frozen_request(
        model, torch.tensor([[2, 3, 4, 5]]), GateConfig(mode='full'), RegionGeometry())
    assert output.past_key_values is not None


def test_greedy_next_tokens_masks_every_eos_before_min_new_tokens():
    # 4 和 1 都是 EOS；未达最小长度时必须选择次优 token 3。
    logits = torch.tensor([[[0., 8., 0., 5., 9., 0.]]])
    assert greedy_next_tokens(
        logits, {1, 4}, generated_counts=0, min_new_tokens=2).tolist() == [3]
    assert greedy_next_tokens(
        logits, {1, 4}, generated_counts=1, min_new_tokens=2).tolist() == [3]
    assert greedy_next_tokens(
        logits, {1, 4}, generated_counts=2, min_new_tokens=2).tolist() == [4]

    batched = logits.expand(2, -1, -1)
    assert greedy_next_tokens(
        batched, {1, 4}, generated_counts=[1, 2],
        min_new_tokens=2).tolist() == [3, 4]


class _LifecycleCache:
    def __init__(self):
        self.active_request_id = 0
        self.request_phases = {0: RequestPhase.FREEZE}

    def transition_request(self, request_id, phase):
        self.request_phases[request_id] = RequestPhase(phase)


def _decode_output(best_token):
    logits = torch.full((1, 1, 6), -10.)
    logits[0, 0, best_token] = 10.
    logits[0, 0, 3] = 5.
    cache = _LifecycleCache()
    return SimpleNamespace(logits=logits, past_key_values=cache)


def test_decode_stops_on_first_eos_and_honors_single_token_budget():
    output = _decode_output(4)
    assert decode_frozen_request(
        None, output, max_new_tokens=8, eos_ids={1, 4})[0] == [4]
    assert output.past_key_values.request_phases[0] == RequestPhase.FINISHED
    assert decode_frozen_request(
        None, _decode_output(2), max_new_tokens=1,
        eos_ids={1, 4})[0] == [2]
    output = _decode_output(2)
    assert decode_frozen_request(
        None, output, max_new_tokens=0, eos_ids={1, 4})[0] == []
    assert output.past_key_values.request_phases[0] == RequestPhase.FINISHED
    # 空 suffix 与缺省路径完全一致，不启动额外的 Runner 步骤。
    assert decode_frozen_request(
        None, _decode_output(2), max_new_tokens=1, eos_ids={1, 4},
        suffix_ids=[])[0] == [2]


def test_decode_does_not_treat_padding_as_an_implicit_eos():
    generated, _ = decode_frozen_request(
        None, _decode_output(1), max_new_tokens=1,
        eos_ids={4}, pad_token_id=1)
    assert generated == [1]


def test_decode_uses_the_same_min_length_policy_for_first_and_graph_tokens(
        monkeypatch):
    class FakeScheduler:
        def __init__(self, **kwargs):
            self.next_tokens = {}

        def add_request(self, request_id):
            self.request_id = request_id

    class FakeRunner:
        def __init__(self, **kwargs):
            self.calls = []

        def step(self, model, cache, *, generated_counts, min_new_tokens,
                 return_next_tokens):
            assert return_next_tokens
            self.calls.append((generated_counts, min_new_tokens))
            tokens = greedy_next_tokens(
                torch.tensor([[[0., 8., 0., 5., 9., 0.]]]), {1, 4},
                generated_counts=generated_counts,
                min_new_tokens=min_new_tokens)
            return None, tokens

    monkeypatch.setattr(
        'sparse_attn.scheduler.scheduler.ContinuousBatchScheduler',
        FakeScheduler)
    monkeypatch.setattr(
        'sparse_attn.engine.graph_runner.GraphRunner', FakeRunner)

    class FakeModel:
        def parameters(self):
            return iter([torch.empty(0)])

    generated, _ = decode_frozen_request(
        FakeModel(), _decode_output(4), max_new_tokens=8, eos_ids={1, 4},
        capture_graph=False, min_new_tokens=2)
    # 首 token 与第二 token 均屏蔽 EOS；达到最小长度后第三个 token 才停止。
    assert generated == [3, 3, 4]


def test_suffix_is_fully_consumed_and_only_answer_tokens_are_returned(monkeypatch):
    class FakeScheduler:
        def __init__(self, **kwargs):
            self.next_tokens = {}

        def add_request(self, request_id):
            self.request_id = request_id

    class FakeRunner:
        instances = []

        def __init__(self, scheduler, **kwargs):
            self.scheduler = scheduler
            self.consumed = []
            self.policies = []
            self.outputs = iter([11, 12, 3, 4])
            self.instances.append(self)

        def step(self, model, cache, *, generated_counts, min_new_tokens,
                 return_next_tokens, forced_tokens=None, stop_on_eos=True):
            self.consumed.append(self.scheduler.next_tokens[0])
            self.policies.append((forced_tokens, stop_on_eos))
            token = next(self.outputs)
            self.scheduler.next_tokens[0] = (
                forced_tokens[0] if forced_tokens is not None else token)
            return None, torch.tensor([token])

    monkeypatch.setattr(
        'sparse_attn.scheduler.scheduler.ContinuousBatchScheduler', FakeScheduler)
    monkeypatch.setattr('sparse_attn.engine.graph_runner.GraphRunner', FakeRunner)

    class FakeModel:
        def parameters(self):
            return iter([torch.empty(0)])

    output = _decode_output(2)
    generated, cache = decode_frozen_request(
        FakeModel(), output, max_new_tokens=8, eos_ids={1, 4},
        capture_graph=False, suffix_ids=[4, 2, 4])
    runner = FakeRunner.instances[0]
    # 三个 suffix（包括两个 EOS）全部作为输入消费；第四个输入才是首答案 token。
    # 返回值只含最后一个 suffix forward 产生的答案及后续答案。
    assert generated == [3, 4]
    assert runner.consumed == [4, 2, 4, 3]
    assert runner.policies == [
        ([2], False), ([4], False), (None, True), (None, True)]
    assert cache.request_phases[0] == RequestPhase.FINISHED


def test_decode_failure_marks_request_failed(monkeypatch):
    class FakeScheduler:
        def __init__(self, **kwargs):
            self.next_tokens = {}

        def add_request(self, request_id):
            pass

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def step(self, *args, **kwargs):
            raise RuntimeError('synthetic decode failure')

    monkeypatch.setattr(
        'sparse_attn.scheduler.scheduler.ContinuousBatchScheduler', FakeScheduler)
    monkeypatch.setattr('sparse_attn.engine.graph_runner.GraphRunner', FakeRunner)

    class FakeModel:
        def parameters(self):
            return iter([torch.empty(0)])

    output = _decode_output(2)
    with pytest.raises(RuntimeError, match='synthetic decode failure'):
        decode_frozen_request(
            FakeModel(), output, max_new_tokens=2, eos_ids={4},
            capture_graph=False)
    assert output.past_key_values.request_phases[0] == RequestPhase.FAILED


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_forced_suffix_eos_survives_async_scheduler_drain():
    class FakeCache:
        def __init__(self):
            self.layers = [SimpleNamespace(
                kv_pool_k=SimpleNamespace(device=torch.device('cuda')))]
            self.states = {0: SimpleNamespace(logical_length=0)}
            self.request_phases = {0: RequestPhase.GENERATE}
            self.removed = []

        def new_step(self, req_ids):
            pass

        def transition_request(self, request_id, phase):
            self.request_phases[request_id] = RequestPhase(phase)

        def remove_request(self, request_id):
            self.removed.append(request_id)

    cache = FakeCache()
    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=1, eos_ids={4}, pad_token_id=0)
    scheduler.add_request(0)
    scheduler.next_tokens[0] = 2
    req_ids, _, _ = scheduler.prepare_step()
    logits = torch.full((1, 1, 6), -10., device='cuda')
    scheduler.record_next_tokens(
        req_ids, logits, forced_tokens=[4], stop_on_eos=False)

    # 上一步的 EOS 是 forced suffix 输入，异步 drain 不得结束请求。
    req_ids, _, _ = scheduler.prepare_step()
    assert scheduler.active_req_ids == [0]
    assert scheduler.next_tokens[0] == 4
    assert cache.removed == []

    # 普通生成路径仍按 EOS 正常结束，验证修复没有全局关闭停止逻辑。
    logits[0, 0, 4] = 10.
    scheduler.record_next_tokens(req_ids, logits)
    scheduler.prepare_step()
    assert scheduler.active_req_ids == []
    assert cache.removed == [0]
    assert cache.request_phases[0] == RequestPhase.FINISHED
