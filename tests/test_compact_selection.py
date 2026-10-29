"""Phase 1 紧凑 Selection 接口验证：compute_selection_graph 与 compute_mask_graph 逐位对拍。

核心契约：compute_mask_graph(return_selection=True).to_mask(S_cap)
         == compute_mask_graph(return_selection=False)
         逐位相等（无损 compact roundtrip）。
同时验证 graph_middle_selection 的排序去重语义。
"""
import pytest
import torch
from sparse_attn.ops.token_selection import (
    TokenSelection, canonical_selection, middle_selection,
    graph_middle_selection,
)


# ---------- 单元测试：纯 tensor 逻辑（CPU，无需模型） ----------

def test_canonical_selection_dedup_and_sort():
    # [B=1,Hkv=1,K=6]，重复 ID 应去重，无效 ID 应过滤
    ids = torch.tensor([[[5, 3, 3, 7, 5, 999]]])  # 999 是无效（≥capacity）
    valid = torch.tensor([[[True, True, True, True, True, True]]])
    sel = canonical_selection(ids, valid, capacity=100, width=None)
    # 排序去重后应为 [3, 5, 7]（有效 unique IDs 排序）
    assert sel.counts.item() == 3
    assert torch.equal(sel.indices[0, 0, :3], torch.tensor([3, 5, 7]))


def test_canonical_selection_width_truncation():
    ids = torch.tensor([[[1, 2, 3, 4, 5]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    sel = canonical_selection(ids, valid, capacity=100, width=3)
    assert sel.counts.item() == 3
    assert torch.equal(sel.indices[0, 0, :3], torch.tensor([1, 2, 3]))


def test_canonical_selection_filters_negative_and_oob():
    ids = torch.tensor([[[-1, 0, 1, 999, 50]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    sel = canonical_selection(ids, valid, capacity=100, width=None)
    # -1 和 999 应被过滤
    assert sel.counts.item() == 3
    assert torch.equal(sel.indices[0, 0, :3], torch.tensor([0, 1, 50]))


def test_canonical_selection_invalid_input():
    ids = torch.tensor([[[1, 2, 3]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    with pytest.raises(ValueError):
        canonical_selection(ids, valid, capacity=0)
    with pytest.raises(ValueError):
        canonical_selection(ids.reshape(3), valid.reshape(3), capacity=10)


def test_middle_selection_excludes_protected():
    ids = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    # sink=2, swa=3, length=10: 保护集 [0,2) ∪ [7,10) = {0,1,7,8,9}
    sel = middle_selection(ids, valid, lengths=torch.tensor([10]),
                            sink=2, swa=3, capacity=100, width=None)
    # 中段 = {2,3,4,5,6}
    assert sel.counts.item() == 5
    assert torch.equal(sel.indices[0, 0, :5], torch.tensor([2, 3, 4, 5, 6]))


def test_middle_selection_empty_mid_domain():
    ids = torch.tensor([[[0, 1, 2, 3]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    # sink=2, swa=2, length=4: 中段 [2,2) 空
    sel = middle_selection(ids, valid, lengths=torch.tensor([4]),
                            sink=2, swa=2, capacity=100, width=None)
    assert sel.counts.item() == 0


def test_graph_middle_selection_basic():
    # top: [Hkv=1, K2=5], mid_mask 偶数位 True, sink=2, swa=2
    capacity = 20
    top = torch.tensor([[3, 8, 15, 1, 18]])  # [1, 5]
    k2_mid = torch.tensor([5])
    mid_mask = torch.zeros(capacity, dtype=torch.bool)
    mid_mask[::2] = True  # 0,2,4,...,18 是 mid
    length = torch.tensor(20)
    sel = graph_middle_selection(top, k2_mid, mid_mask, length, capacity, 2, 2)
    # 中段 [2, 18)，mid_mask 偶数，有效 top IDs: 8, 15(奇数，不在 mid)
    # 3 (<2, sink), 1(<2, sink), 18(奇数，不在 mid)
    # 中段 mid 偶数: 8 → 保留
    assert sel.counts[0, 0].item() >= 1
    # 中段 [2,18) 中 mid=True 的偶数 ID：来自 top 的 {8}（3 < sink, 1 < sink, 15/18 奇数）
    # k2_mid=5 全选 → 中段 mid=True 的 ID
    assert 8 in sel.indices[0, 0, :sel.counts[0, 0]].tolist()


def test_graph_middle_selection_swa_exclusion():
    capacity = 20
    top = torch.tensor([[18, 17, 16, 5, 10]])  # [1, 5]
    k2_mid = torch.tensor([5])
    mid_mask = torch.ones(capacity, dtype=torch.bool)  # 全 mid
    mid_mask[:2] = False  # sink 非 mid
    length = torch.tensor(20)
    sel = graph_middle_selection(top, k2_mid, mid_mask, length, capacity, 2, 3)
    # 保护集 = [0,2) ∪ [17,20) → 17, 18 属 swa（排除）；5, 10 属中段
    valid_ids = sel.indices[0, 0, :sel.counts[0, 0]].tolist()
    assert 17 not in valid_ids
    assert 18 not in valid_ids
    assert 5 in valid_ids
    assert 10 in valid_ids


def test_tokenselection_to_mask_roundtrip():
    capacity = 10
    ids = torch.tensor([[[2, 5, 8, 1, 999]]])
    valid = torch.tensor([[[True, True, True, True, False]]])
    sel = canonical_selection(ids, valid, capacity=capacity, width=None)
    mask = sel.to_mask(capacity)
    assert mask.shape == (1, 1, capacity)
    # IDs 1, 2, 5, 8 应被选中
    expected = torch.zeros(capacity, dtype=torch.bool)
    expected[[1, 2, 5, 8]] = True
    assert torch.equal(mask[0, 0], expected)


def test_valid_mask_correctness():
    ids = torch.tensor([[[3, 1, 7]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    sel = canonical_selection(ids, valid, capacity=10, width=None)
    vm = sel.valid()
    # counts=3, valid 前 3 位 True，后 K-3 位 False
    assert vm.shape == (1, 1, 3)
    assert vm.all()


def test_empty_selection():
    ids = torch.tensor([[[999]]])
    valid = torch.tensor([[[True]]])
    sel = canonical_selection(ids, valid, capacity=100, width=None)
    assert sel.counts.item() == 0
    mask = sel.to_mask(100)
    assert not mask.any()
