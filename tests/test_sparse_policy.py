"""CPU checks for capture-time eligibility, including explicit fallback reasons."""
import pytest
import torch
from sparse_attn.ops.sparse_policy import geometry_fallback, selection_fallback


@pytest.fixture(autouse=True)
def policies(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.8.0+cu128')
    monkeypatch.setenv('TLI_SPARSE_SELECTION', '1')
    monkeypatch.setenv('TLI_SPARSE_STATS', 'owner')


def route(**updates):
    args = dict(dtype=torch.bfloat16, device=torch.device('cuda'), capacity=16401,
                group=4, heads=8, l2_bypass=False, protected_end=128, topk=1024)
    args.update(updates)
    return selection_fallback(**args)


@pytest.mark.parametrize('updates,reason', [
    ({}, None),
    ({'topk': 2048}, 'topk_requires_1024'),
    ({'topk': 0}, 'topk_requires_1024'),
    ({'l2_bypass': True}, 'l2_bypass'),
    ({'protected_end': 16402}, 'protected_domain_exceeds_capacity'),
    ({'dtype': torch.float32}, 'dtype'),
    ({'device': torch.device('cpu')}, 'device'),
    ({'capacity': 9216}, 'capacity'),
    ({'capacity': 2**31-1}, 'capacity'),
    ({'capacity': 2**30, 'heads': 2}, 'owner_index_overflow'),
    ({'group': 3}, 'gqa_group'),
    ({'group': 0}, 'gqa_group'),
    ({'heads': 0}, 'head_grid'),
    ({'heads': 65536, 'group': 1, 'capacity': 10000}, 'head_grid'),
])
def test_selection_reasons(updates, reason):
    assert route(**updates) == reason


def test_explicit_policies(monkeypatch):
    monkeypatch.delenv('TLI_SPARSE_SELECTION')
    assert route() == 'disabled'
    monkeypatch.setenv('TLI_SPARSE_SELECTION', 'auto')
    assert route() == 'selection_policy'
    monkeypatch.setenv('TLI_SPARSE_SELECTION', '1')
    monkeypatch.setenv('TLI_SPARSE_STATS', 'owenr')
    assert route() == 'stats_policy'
    monkeypatch.setenv('TLI_SPARSE_STATS', 'sorted')
    assert route() == 'unaligned_requires_owner'
    assert route(capacity=16400) is None
    monkeypatch.setattr(torch, '__version__', '2.9.0')
    assert route(capacity=16400) == 'torch_version'


def test_supported_geometry_uses_same_decision():
    from sparse_attn.ops.sparse_normalization import supported_geometry
    for capacity in (9216, 9217, 16400, 16401, 2**31-1):
        for group in (0, 1, 3, 4):
            args = (torch.bfloat16, torch.device('cuda'), capacity, group, 8)
            assert supported_geometry(*args) == (geometry_fallback(*args) is None)
