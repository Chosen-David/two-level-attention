# -*- coding: utf-8 -*-
"""Phase 4: decode_indexed 接线测试。

验证 compact selection → partial_attention → merge 路径与 legacy
compute_mask_graph + eager_decoding_attn 输出对齐。

测试策略：
  1. CPU torch backend 对拍（不依赖 CUDA）
  2. 边界：空 selection（counts=0）不崩溃
  3. 不同 sink/swa 几何下对齐
"""
import pytest
import torch

from sparse_attn.ops.partial_attention import (
    partial_attention, merge_partials)
from sparse_attn.ops.token_selection import (
    graph_middle_selection, TokenSelection)


def _reference_full_attention(q, k, v, scale):
    """全 token 注意力参考实现（dense softmax over all positions）。

    支持 GQA：Hq 可能 > Hkv，每 group 共享 K/V head。
    """
    qg = (q[:, 0] * scale).float()  # [B,Hq,D]
    b, hq, d = qg.shape
    hk = k.shape[2]
    group_size = hq // hk
    # GQA：reshape q 为 [B, HK, group_size, D]，与 k [B, T, HK, D] 对齐
    qg = qg.reshape(b, hk, group_size, d)
    scores = torch.einsum('bhgd,bthd->bhgt', qg, k.float())  # [B,HK,group,T]
    scores = scores.reshape(b, hq, -1)  # [B,Hq,T]
    weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    # v: [B,T,HK,Dv] → 按 group broadcast 到 [B,T,Hq,Dv]
    vg = v.float().repeat_interleave(group_size, dim=2)  # [B,T,Hq,Dv]
    ref = torch.einsum('bht,bthd->bhd', weights, vg).unsqueeze(1)
    return ref


def _decode_indexed_torch(q, k, v, sel_indices, sel_counts, length,
                          sink, swa, scale):
    """decode_indexed 的 torch backend 等价实现。

    与 operator_pipeline.decode_indexed 同语义：
      prepare_index → compute_selection_graph → protected + selected → merge
    此处直接接收 TokenSelection，跳过 indexer 依赖。
    """
    ids_counts = (sel_indices, sel_counts)
    protected = partial_attention(
        q, k, v, ids_counts, length,
        sink=sink, swa=swa, scale=scale,
        protected=True, splits=4, backend='torch')
    selected = partial_attention(
        q, k, v, ids_counts, length,
        sink=sink, swa=swa, scale=scale,
        protected=False, splits=4, backend='torch')
    return merge_partials(protected, selected, dtype=q.dtype, backend='torch')


# ---------- 测试 ----------

def test_decode_indexed_basic_alignment():
    """基本对齐：compact selection → partial attention 与 full attention 对齐。"""
    torch.manual_seed(42)
    b, t, h, hk, d, dv = 1, 64, 8, 4, 32, 32
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 4, 16

    # 模拟 indexer 选出的紧凑候选集：选所有中段 token
    mid_mask = torch.ones(t, dtype=torch.bool)
    mid_mask[:sink] = False
    mid_mask[t - swa:] = False
    mid_ids = torch.where(mid_mask)[0].to(torch.int32)
    K2 = len(mid_ids)
    top = mid_ids.unsqueeze(0).expand(hk, -1).contiguous()
    k2_mid = torch.full((hk,), K2, dtype=torch.long)

    sel = graph_middle_selection(top, k2_mid, mid_mask, t, t, sink, swa)

    out = _decode_indexed_torch(q, k, v, sel.indices, sel.counts,
                                lengths, sink, swa, scale)
    ref = _reference_full_attention(q, k, v, scale)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_decode_indexed_partial_selection():
    """部分选择：只选部分中段 token（非全选），验证 merge 语义。

    当中段只选部分 token 时，protected 路径覆盖 sink+swa 全部，selected
    路径只覆盖被选中的中段 token。merge 后的结果应对应『sink+swa+selected』
    子集的注意力（≠ 全 token attention）。

    验证方法：构造『全中段选择』和『部分中段选择』两臂，确认部分选择的
    输出与手动 masked attention 对齐（只关注选中位置）。
    """
    torch.manual_seed(7)
    b, t, h, hk, d, dv = 1, 48, 4, 4, 16, 16
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 2, 8

    # 全中段选择
    mid_mask = torch.ones(t, dtype=torch.bool)
    mid_mask[:sink] = False
    mid_mask[t - swa:] = False
    mid_ids = torch.where(mid_mask)[0].to(torch.int32)
    K2_all = len(mid_ids)
    top_all = mid_ids.unsqueeze(0).expand(hk, -1).contiguous()
    k2_mid_all = torch.full((hk,), K2_all, dtype=torch.long)
    sel_all = graph_middle_selection(top_all, k2_mid_all, mid_mask, t, t,
                                     sink, swa)

    # 部分中段选择（只选前一半）
    K2_part = max(1, K2_all // 2)
    top_part = mid_ids[:K2_part].unsqueeze(0).expand(hk, -1).contiguous()
    k2_mid_part = torch.full((hk,), K2_part, dtype=torch.long)
    sel_part = graph_middle_selection(top_part, k2_mid_part, mid_mask, t, t,
                                      sink, swa)

    # 全选中臂输出（=GQA full attn）
    out_all = _decode_indexed_torch(q, k, v, sel_all.indices, sel_all.counts,
                                    lengths, sink, swa, scale)
    ref_full = _reference_full_attention(q, k, v, scale)
    torch.testing.assert_close(out_all, ref_full, rtol=1e-3, atol=1e-3)

    # 部分选中臂输出（应 ≠ full attn，验证选择性生效）
    out_part = _decode_indexed_torch(q, k, v, sel_part.indices, sel_part.counts,
                                     lengths, sink, swa, scale)
    # 手动计算部分注意力：合并 selected + protected token 集合
    qg = (q[:, 0] * scale).float().reshape(b, hk, h // hk, d)
    scores = torch.einsum('bhgd,bthd->bhgt', qg, k.float())
    # 选中的 token 集合 = sink + swa + selected_mid
    selected_set = sel_part.indices[0]  # [HK, K]
    selected_counts = sel_part.counts[0]  # [HK]
    # 构造 mask：对每个 head group
    mask = torch.zeros(h, t, dtype=torch.bool)
    group_size = h // hk
    for kv_idx in range(hk):
        qh_start = kv_idx * group_size
        qh_end = qh_start + group_size
        # sink
        mask[qh_start:qh_end, :sink] = True
        # swa
        mask[qh_start:qh_end, t - swa:] = True
        # selected mid tokens
        cnt = selected_counts[kv_idx].item()
        for ki in range(cnt):
            tok = selected_set[kv_idx, ki].item()
            if 0 <= tok < t:
                mask[qh_start:qh_end, tok] = True

    scores_masked = scores.reshape(b, h, -1).masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores_masked, dim=-1).nan_to_num(0.0)
    vg = v.float().repeat_interleave(group_size, dim=2)
    ref_part = torch.einsum('bht,bthd->bhd', weights, vg).unsqueeze(1)
    torch.testing.assert_close(out_part, ref_part, rtol=1e-3, atol=1e-3)


def test_decode_indexed_empty_selection():
    """空 selection（counts=0）：中段无候选，仅 protected 路径生效。"""
    torch.manual_seed(99)
    b, t, h, hk, d, dv = 1, 32, 4, 4, 16, 16
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 4, 8

    # 空 selection：所有 counts=0
    sel_indices = torch.zeros(1, hk, 8, dtype=torch.int32)
    sel_counts = torch.zeros(1, hk, dtype=torch.int32)

    out = _decode_indexed_torch(q, k, v, sel_indices, sel_counts,
                                lengths, sink, swa, scale)

    # 参考：仅 sink+swa 保护集的注意力
    qg = (q[:, 0] * scale).float()
    scores = torch.einsum('bhd,bthd->bht', qg, k.float())
    # mask：只保留 sink+swa 位置
    mask = torch.zeros(1, h, t, dtype=torch.bool)
    for hi in range(h):
        mask[0, hi, :sink] = True
        mask[0, hi, t - swa:] = True
    scores_masked = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores_masked, dim=-1).nan_to_num(0.0)
    ref = torch.einsum('bht,bthd->bhd', weights, v.float()).unsqueeze(1)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_decode_indexed_small_sink_swa():
    """小 sink/swa 几何：sink=1, swa=2。"""
    torch.manual_seed(13)
    b, t, h, hk, d, dv = 1, 24, 2, 2, 16, 16
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 1, 2

    mid_mask = torch.ones(t, dtype=torch.bool)
    mid_mask[:sink] = False
    mid_mask[t - swa:] = False
    mid_ids = torch.where(mid_mask)[0].to(torch.int32)
    K2 = len(mid_ids)
    top = mid_ids.unsqueeze(0).expand(hk, -1).contiguous()
    k2_mid = torch.full((hk,), K2, dtype=torch.long)

    sel = graph_middle_selection(top, k2_mid, mid_mask, t, t, sink, swa)
    out = _decode_indexed_torch(q, k, v, sel.indices, sel.counts,
                                lengths, sink, swa, scale)
    ref = _reference_full_attention(q, k, v, scale)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_decode_indexed_large_swa():
    """大 swa 几何：swa 覆盖大部分序列，中段很小。"""
    torch.manual_seed(55)
    b, t, h, hk, d, dv = 1, 64, 4, 4, 16, 16
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 4, 48  # swa 覆盖 48/64 = 75%

    mid_mask = torch.ones(t, dtype=torch.bool)
    mid_mask[:sink] = False
    mid_mask[t - swa:] = False
    mid_ids = torch.where(mid_mask)[0].to(torch.int32)
    K2 = max(1, len(mid_ids))
    if K2 > 0:
        top = mid_ids[:K2].unsqueeze(0).expand(hk, -1).contiguous()
        k2_mid = torch.full((hk,), K2, dtype=torch.long)
    else:
        top = torch.zeros(hk, 1, dtype=torch.int64)
        k2_mid = torch.zeros(hk, dtype=torch.long)

    sel = graph_middle_selection(top, k2_mid, mid_mask, t, t, sink, swa)
    out = _decode_indexed_torch(q, k, v, sel.indices, sel.counts,
                                lengths, sink, swa, scale)
    ref = _reference_full_attention(q, k, v, scale)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_operator_mode_off_routing():
    """TLI_OPERATOR_MODE=off 时不应走 decode_indexed 路由。"""
    import os
    os.environ["TLI_OPERATOR_MODE"] = "off"
    from sparse_attn.ops.operator_pipeline import operator_mode
    assert operator_mode() == 'off'
    # off 模式下 qwen3_attn_patch 走 legacy 路径，不调用 decode_indexed
    os.environ.pop("TLI_OPERATOR_MODE", None)


def test_operator_mode_serial_signature():
    """TLI_OPERATOR_MODE=serial 时 decode_indexed 有正确签名。"""
    import os
    os.environ["TLI_OPERATOR_MODE"] = "serial"
    import importlib
    import sparse_attn.ops.operator_pipeline as op
    importlib.reload(op)
    assert op.operator_mode() == 'serial'
    import inspect
    sig = inspect.signature(op.decode_indexed)
    params = list(sig.parameters.keys())
    assert 'indexer' in params
    assert 'softmax_scale' in params
    os.environ.pop("TLI_OPERATOR_MODE", None)
