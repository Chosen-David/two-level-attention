"""区域 kernel 审计按 batch 层发射计数，而非按请求重复计数。"""
import pytest

from benchmark.efficiency.measure_nfa_selector_e2e import _region_kernel_call_audit


def _request(near=(), far=(), n_layers=4):
    near, far = set(near), set(far)
    return {
        'n_layers': n_layers,
        'near_layers': sorted(near),
        'far_layers': sorted(far),
        'both_layers': sorted(near & far),
    }


def _kernels(expected, repeat=2):
    names = {
        'generic_l1': '_nfa_l1_scores_kernel',
        'near_only_l1': '_nfa_l1_near_scores_kernel',
        'far_only_l1': '_nfa_l1_far_scores_kernel',
        'active_page_stats': '_nfa_page_stats_prepare_q_kernel',
        'active_candidate': '_topk_candidate_prepare_kernel',
        'active_fine_score': '_nfa_fine_scores_paged_kernel',
    }
    return {
        name: {'calls': expected[key] * repeat, 'device_ms': 1.0}
        for key, name in names.items() if expected[key]
    }


def test_batched_identical_signatures_are_not_counted_per_request():
    requests = [_request(near=(1, 2), far=(2, 3))] * 2
    expected = {
        'generic_l1': 1, 'near_only_l1': 1, 'far_only_l1': 1,
        'active_page_stats': 3, 'active_candidate': 3, 'active_fine_score': 3,
    }
    result = {'skip': dict(enumerate(requests)), 'profile_repeat': 2}
    audit = _region_kernel_call_audit(result, _kernels(expected))
    assert audit['passed'] is True
    assert audit['expected_calls_per_step'] == expected


def test_mixed_batch_layer_uses_one_generic_launch():
    requests = [
        _request(near=(1, 2), far=(2, 3)),
        _request(near=(1,), far=(3,)),
    ]
    expected = {
        'generic_l1': 2, 'near_only_l1': 1, 'far_only_l1': 1,
        'active_page_stats': 4, 'active_candidate': 4, 'active_fine_score': 4,
    }
    result = {'skip': dict(enumerate(requests)), 'profile_repeat': 2}
    audit = _region_kernel_call_audit(result, _kernels(expected))
    assert audit['expected_calls_per_step'] == expected


def test_uniform_both_skip_has_no_active_selector_launch():
    result = {
        'skip': dict(enumerate([_request(near=range(4), far=range(4))] * 2)),
        'profile_repeat': 2,
    }
    audit = _region_kernel_call_audit(result, {})
    assert audit['expected_calls_per_step'] == {
        'generic_l1': 0, 'near_only_l1': 0, 'far_only_l1': 0,
        'active_page_stats': 0, 'active_candidate': 0, 'active_fine_score': 0,
    }


def test_per_request_overcount_is_rejected():
    requests = [_request(near=(1,))] * 2
    expected = {
        'generic_l1': 3, 'near_only_l1': 0, 'far_only_l1': 1,
        'active_page_stats': 4, 'active_candidate': 4, 'active_fine_score': 4,
    }
    wrong = dict(expected, far_only_l1=2)
    result = {'skip': dict(enumerate(requests)), 'profile_repeat': 2}
    with pytest.raises(ValueError, match='区域 kernel 实例数'):
        _region_kernel_call_audit(result, _kernels(wrong))
