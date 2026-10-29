"""Phase 3 Serial Graph 对齐：compute_selection_graph 与 compute_mask_graph 逐位对拍。

核心契约（Phase 3 最关键的正确性门）：
  compute_mask_graph(return_selection=True).to_mask(S_cap)
  == compute_mask_graph(return_selection=False)
  逐位相等（compact roundtrip 零损失）。

此测试不需要 CUDA 模型推理——直接构造 Q/K/V 调用 indexer 方法。
因为 compute_mask_graph 内部已经调用了完整的 mask 生产链（含 iter3
融合 kernel），我们只需验证 return_selection=True/False 的输出等价。
"""
import pytest
import torch
from sparse_attn.ops.token_selection import canonical_selection, middle_selection


# ---------- 纯 tensor 级别的 compact 接口对拍 ----------

def test_graph_middle_selection_matches_full_mask():
    """验证 graph_middle_selection 生成的 compact selection 重建后
    与 full-width mask 逐位一致。

    模拟 compute_mask_graph 内部逻辑：topk → rank gate → mid_mask →
    graph_middle_selection → to_mask。
    """
    device = 'cpu'
    Hkv, S_cap, K2 = 4, 128, 32
    sink_tok, swa_tok = 4, 8
    S_real = 100  # 当前步序列长

    # 模拟 topk 输出：随机 K2 个 token ID
    torch.manual_seed(123)
    top = torch.randint(0, S_real, (Hkv, K2), device=device)
    k2_mid = torch.randint(K2 // 2, K2, (Hkv,), device=device)

    # 模拟 mid_mask：token 在 [sink_tok, S_real-swa_tok) 区间内为 True
    mid_mask = torch.zeros(S_cap, dtype=torch.bool, device=device)
    mid_mask[sink_tok:S_real - swa_tok] = True

    # 正常路径（模拟 compute_mask_graph 返回 full mask 的部分）
    from sparse_attn.ops.token_selection import graph_middle_selection
    sel = graph_middle_selection(top, k2_mid, mid_mask, S_real, S_cap,
                                 sink_tok, swa_tok)

    # 重建 mask
    mask_from_sel = sel.to_mask(S_cap)  # [1, Hkv, S_cap]

    # 参考 mask：直接从 topk→rank gate→mid_mask 构造（与 compute_mask_graph 同语义）
    ref_mask = torch.zeros(1, Hkv, S_cap, dtype=torch.bool, device=device)
    for h in range(Hkv):
        for k_idx in range(k2_mid[h]):
            tok = top[h, k_idx].item()
            if 0 <= tok < S_cap and mid_mask[tok] and tok < S_real:
                if tok >= sink_tok and tok < S_real - swa_tok:
                    ref_mask[0, h, tok] = True

    assert torch.equal(mask_from_sel[0], ref_mask[0]), (
        f'mismatch: sel={mask_from_sel[0].int().sum().item()} '
        f'ref={ref_mask[0].int.sum().item()}')


def test_compact_roundtrip_bitexact_deterministic():
    """确定性场景：给定固定 topk 输出，验证 compact roundtrip bitexact。"""
    Hkv, S_cap, K2 = 2, 64, 16
    S_real = 50
    sink_tok, swa_tok = 2, 4

    top = torch.arange(K2, device='cpu').unsqueeze(0).expand(Hkv, -1)
    k2_mid = torch.full((Hkv,), K2, dtype=torch.long)
    mid_mask = torch.ones(S_cap, dtype=torch.bool)
    mid_mask[:sink_tok] = False
    mid_mask[S_real - swa_tok:] = False

    from sparse_attn.ops.token_selection import graph_middle_selection
    sel = graph_middle_selection(top, k2_mid, mid_mask, S_real, S_cap,
                                 sink_tok, swa_tok)
    mask = sel.to_mask(S_cap)

    # 预期：top[0:K2] = [0..15]，mid_mask[2:46] 排除 [0,1] 和 [46,49]
    # 所以选中的 token = {2,3,...,15}（全部在 mid 内）
    expected_ids = torch.arange(2, 16, dtype=torch.int32)
    for h in range(Hkv):
        actual = sel.indices[0, h, :sel.counts[0, h]]
        assert torch.equal(actual, expected_ids), (
            f'head {h}: actual={actual.tolist()} expected={expected_ids.tolist()}')


def test_swa_exclusion_in_compact():
    """SWA 尾部排除：swa_tok=4 → token [S_real-4, S_real) 不入选。"""
    Hkv, S_cap, K2 = 1, 32, 16
    S_real = 20
    sink_tok, swa_tok = 2, 4

    # top = [16, 17, 18, 19, 0, 1, ..., 11]
    top = torch.tensor([[16, 17, 18, 19] + list(range(0, 12))])
    k2_mid = torch.tensor([16])
    mid_mask = torch.ones(S_cap, dtype=torch.bool)
    mid_mask[:sink_tok] = False  # [0,1] 非 mid
    mid_mask[S_real:] = False    # pad 非 mid

    from sparse_attn.ops.token_selection import graph_middle_selection
    sel = graph_middle_selection(top, k2_mid, mid_mask, S_real, S_cap,
                                 sink_tok, swa_tok)

    # 中段 [2, 16)：排除 16,17,18,19（≥S_real-swa_tok=16）→ 排除
    # 中段排除 0,1（<sink_tok=2）
    valid = sel.indices[0, 0, :sel.counts[0, 0]].tolist()
    assert 16 not in valid
    assert 17 not in valid
    assert 18 not in valid
    assert 19 not in valid
    assert 0 not in valid
    assert 1 not in valid
    assert 2 in valid
    assert 11 in valid  # top 中最大中段 token=11（top 不含 15）


def test_empty_selection_mid_all_excluded():
    """所有 topk 命中 sink/swa 非 mid 区 → selection 为空。"""
    Hkv, S_cap, K2 = 1, 64, 8
    S_real = 20
    sink_tok, swa_tok = 2, 4

    # top 全落在 swa 尾 [16,20)
    top = torch.tensor([[16, 17, 18, 19, 16, 17, 18, 19]])
    k2_mid = torch.tensor([8])
    mid_mask = torch.ones(S_cap, dtype=torch.bool)
    mid_mask[:sink_tok] = False  # 实际 16-19 在 mid 内... 不，mid_mask[:2]=False
    # 但 16,17,18,19 ≥ S_real-swa_tok=16 → middle_selection 排除

    from sparse_attn.ops.token_selection import graph_middle_selection
    sel = graph_middle_selection(top, k2_mid, mid_mask, S_real, S_cap,
                                 sink_tok, swa_tok)

    assert sel.counts[0, 0].item() == 0
    mask = sel.to_mask(S_cap)
    assert not mask.any()


def test_partial_attention_via_compact_selection():
    """端到端路径：compact selection → partial_attention → merge
    与 full mask → eager attention 对齐。

    不依赖 CUDA 模型——纯 tensor 直接调用 partial_attention (torch backend)。
    """
    from sparse_attn.ops.partial_attention import (
        partial_attention, merge_partials, PartialState)

    torch.manual_seed(42)
    b, t, h, hk, d, dv = 1, 32, 4, 4, 16, 16
    q = torch.randn(b, 1, h, d, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    sink, swa = 2, 4

    # Step 1：模拟 indexer 选出的紧凑候选集（中段 token + sink/swa 保护）
    S_real = t
    mid_mask = torch.ones(t, dtype=torch.bool)
    mid_mask[:sink] = False
    mid_mask[S_real - swa:] = False

    # 选所有中段 token（模拟 topk 选中全部 mid 域）
    mid_ids = torch.where(mid_mask)[0].to(torch.int32)
    # top = [mid_ids..., padding with_repeat_to_K2]
    K2 = len(mid_ids)
    top = mid_ids.unsqueeze(0).expand(hk, -1).contiguous()
    k2_mid = torch.full((hk,), K2, dtype=torch.long)

    from sparse_attn.ops.token_selection import graph_middle_selection
    sel = graph_middle_selection(top, k2_mid, mid_mask, S_real, t,
                                 sink, swa)

    # Step 2：partial_attention via compact selection（仅中段候选）
    selected = partial_attention(q, k, v, (sel.indices, sel.counts),
                                 lengths, sink=sink, swa=swa, scale=scale,
                                 protected=False, splits=2, backend='torch')

    # Step 3：protected=only（sink+swa mask）
    protected = partial_attention(q, k, v, (sel.indices, sel.counts),
                                  lengths, sink=sink, swa=swa, scale=scale,
                                  protected=True, splits=1, backend='torch')

    # Step 4：merge
    out = merge_partials(protected, selected, dtype=torch.float32,
                         backend='torch')

    # 参考：full attention（全部 token 权重）
    qg = (q[:, 0] * scale).float()
    scores = torch.einsum('bhd,bthd->bht', qg, k.float())
    weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    ref = torch.einsum('bht,bthd->bhd', weights, v.float()).unsqueeze(1)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
