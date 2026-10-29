"""Phase 2 Partial Attention 正确性验证。

核心契约：
  1. torch backend 自洽：partial_attention → merge_partials 与标准
     softmax attention 数值对齐（rtol=1e-3, atol=1e-3）。
  2. protected=True 路径：仅在 sink+swa 保护集上执行，忽略 selection。
  3. protected=False 路径：仅在 selection 紧凑候选集上执行。
  4. merge_partials 两路合并等价于全量 attention。
  5. 输入验证：维度不匹配/dtype 不一致/设备不一致 → ValueError。
"""
import pytest
import torch
from sparse_attn.ops.partial_attention import (
    partial_attention, merge_partials, PartialState,
)


def _reference_attention(q, k, v, mask, scale):
    """标准 softmax attention 参考实现（per-head 精确计算）。

    q [B,1,H,D], k/v [B,T,Hkv,D], mask [B,T] bool。
    GQA: 每个 kv_head 对应 group = H // Hkv 个 q_head。
    返回 [B,1,H,Dv]。
    """
    b, _, h, d = q.shape
    t = k.shape[1]
    hk = k.shape[2]
    group = h // hk
    out = torch.zeros(b, 1, h, v.shape[-1])
    for bi in range(b):
        for ki in range(hk):
            q_h = q[bi, 0, ki * group:(ki + 1) * group].float() * scale  # [group, d]
            k_h = k[bi, :, ki, :].float()  # [T, d]
            v_h = v[bi, :, ki, :].float()  # [T, dv]
            scores = q_h @ k_h.T  # [group, T]
            if mask is not None:
                scores = scores.masked_fill(~mask[bi:bi + 1, :], float('-inf'))
            weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            out[bi, 0, ki * group:(ki + 1) * group] = weights @ v_h
    return out


def _make_inputs(b=1, t=64, h=8, hk=8, d=32, dv=32, device='cpu',
                 seed=42):
    """生成随机 Q/K/V + 全有效 mask。"""
    torch.manual_seed(seed)
    q = torch.randn(b, 1, h, d, device=device, dtype=torch.float32)
    k = torch.randn(b, t, hk, d, device=device, dtype=torch.float32)
    v = torch.randn(b, t, hk, dv, device=device, dtype=torch.float32)
    lengths = torch.tensor([t] * b, device=device, dtype=torch.int64)
    return q, k, v, lengths


# ---------- 基础正确性 ----------

def test_partial_torch_basic():
    """torch backend 基础路径：protected=False + 全选 selection。"""
    q, k, v, lengths = _make_inputs(b=1, t=32, h=4, hk=4, d=16, dv=16)
    t = k.shape[1]
    # 全选 selection：所有 token 都进候选集
    ids = torch.arange(t, device=q.device).unsqueeze(0).unsqueeze(0).expand(
        1, 4, -1).to(torch.int32)
    counts = torch.full((1, 4), t, device=q.device, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=0, swa=0, scale=1.0 / (16 ** 0.5),
                              protected=False, splits=2, backend='torch')
    out = merge_partials(
        PartialState(torch.full((1, 4, 2), -torch.inf),
                     torch.zeros(1, 4, 2),
                     torch.zeros(1, 4, 2, 16)),
        state, dtype=torch.float32, backend='torch')

    # 参考：全量 attention
    mask = torch.ones(1, t, dtype=torch.bool)
    ref = _reference_attention(q, k, v, mask, 1.0 / (16 ** 0.5))
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_partial_protected_only():
    """protected=True 路径：仅在 sink+swa 保护集上执行。"""
    q, k, v, lengths = _make_inputs(b=1, t=32, h=4, hk=4, d=16, dv=16)
    t = k.shape[1]
    sink, swa = 4, 4
    # selection 无关（protected=True 不用）
    ids = torch.zeros(1, 4, 1, device=q.device, dtype=torch.int32)
    counts = torch.zeros(1, 4, device=q.device, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=sink, swa=swa, scale=1.0 / (16 ** 0.5),
                              protected=True, splits=1, backend='torch')
    # protected 路径直接 merge（selected 为空）
    empty = PartialState(
        torch.full((1, 4, 1), -torch.inf),
        torch.zeros(1, 4, 1),
        torch.zeros(1, 4, 1, 16))
    out = merge_partials(state, empty, dtype=torch.float32, backend='torch')

    # 参考：仅 sink+swa 保护集
    mask = torch.zeros(1, t, dtype=torch.bool)
    mask[:, :sink] = True
    mask[:, t - swa:] = True
    ref = _reference_attention(q, k, v, mask, 1.0 / (16 ** 0.5))
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_partial_split_parity():
    """不同 splits 数结果一致（split 是性能优化，不改变数学结果）。"""
    q, k, v, lengths = _make_inputs(b=1, t=32, h=4, hk=4, d=16, dv=16)
    t = k.shape[1]
    ids = torch.arange(t, device=q.device).unsqueeze(0).unsqueeze(0).expand(
        1, 4, -1).to(torch.int32)
    counts = torch.full((1, 4), t, device=q.device, dtype=torch.int32)
    selection = (ids, counts)
    scale = 1.0 / (16 ** 0.5)

    results = []
    for splits in [1, 2, 4]:
        state = partial_attention(q, k, v, selection, lengths,
                                  sink=0, swa=0, scale=scale,
                                  protected=False, splits=splits,
                                  backend='torch')
        empty = PartialState(
            torch.full((1, 4, splits), -torch.inf),
            torch.zeros(1, 4, splits),
            torch.zeros(1, 4, splits, 16))
        out = merge_partials(state, empty, dtype=torch.float32,
                             backend='torch')
        results.append(out)

    for i, r in enumerate(results[1:], 1):
        torch.testing.assert_close(results[0], r, rtol=1e-4, atol=1e-4,
                                   msg=f'splits={i+1} != splits=1')


def test_partial_subset_selection():
    """protected=False + 部分 selection：仅候选集 token 参与 attention。"""
    q, k, v, lengths = _make_inputs(b=1, t=32, h=4, hk=4, d=16, dv=16)
    t = k.shape[1]
    scale = 1.0 / (16 ** 0.5)
    # 仅选偶数 token
    selected = list(range(0, t, 2))
    ids = torch.tensor([selected], device=q.device,
                       dtype=torch.int32).unsqueeze(0).expand(1, 4, -1).contiguous()
    counts = torch.full((1, 4), len(selected), device=q.device, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=0, swa=0, scale=scale,
                              protected=False, splits=1, backend='torch')
    empty = PartialState(
        torch.full((1, 4, 1), -torch.inf),
        torch.zeros(1, 4, 1),
        torch.zeros(1, 4, 1, 16))
    out = merge_partials(state, empty, dtype=torch.float32, backend='torch')

    # 参考：仅偶数 token mask
    mask = torch.zeros(1, t, dtype=torch.bool)
    mask[:, selected] = True
    ref = _reference_attention(q, k, v, mask, scale)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ---------- 输入验证 ----------

def test_validate_rejects_bad_dimensions():
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn(1, 32, 4, 16)
    v = torch.randn(1, 32, 4, 16)
    ids = torch.zeros(1, 4, 8, dtype=torch.int32)
    counts = torch.zeros(1, 4, dtype=torch.int32)
    lengths = torch.tensor([32], dtype=torch.int64)

    # q 不是 4D
    with pytest.raises(ValueError):
        partial_attention(q[:, 0], k, v, (ids, counts), lengths,
                          sink=0, swa=0, scale=1.0)

    # Hq 不是 Hkv 的整数倍
    q2 = torch.randn(1, 1, 7, 16)
    with pytest.raises(ValueError):
        partial_attention(q2, k, v, (ids, counts), lengths,
                          sink=0, swa=0, scale=1.0)


def test_validate_rejects_dtype_mismatch():
    q = torch.randn(1, 1, 4, 16, dtype=torch.float32)
    k = torch.randn(1, 32, 4, 16, dtype=torch.float16)
    v = torch.randn(1, 32, 4, 16, dtype=torch.float32)
    ids = torch.zeros(1, 4, 8, dtype=torch.int32)
    counts = torch.zeros(1, 4, dtype=torch.int32)
    lengths = torch.tensor([32], dtype=torch.int64)
    with pytest.raises(ValueError):
        partial_attention(q, k, v, (ids, counts), lengths,
                          sink=0, swa=0, scale=1.0)


def test_validate_rejects_invalid_splits():
    q, k, v, lengths = _make_inputs(h=4, hk=4, d=16, dv=16)
    ids = torch.zeros(1, 4, 8, dtype=torch.int32)
    counts = torch.zeros(1, 4, dtype=torch.int32)
    with pytest.raises(ValueError):
        partial_attention(q, k, v, (ids, counts), lengths,
                          sink=0, swa=0, scale=1.0, splits=0)
    with pytest.raises(ValueError):
        partial_attention(q, k, v, (ids, counts), lengths,
                          sink=0, swa=0, scale=1.0, splits=65)


# ---------- 边界情况 ----------

def test_empty_selection():
    """空 selection（counts=0）→ 输出全零。"""
    q, k, v, lengths = _make_inputs(h=4, hk=4, d=16, dv=16)
    ids = torch.zeros(1, 4, 8, dtype=torch.int32)
    counts = torch.zeros(1, 4, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=0, swa=0, scale=1.0,
                              protected=False, splits=1, backend='torch')
    # acc 应全零（无有效 token → 无贡献）
    assert torch.all(state.acc == 0)
    # m 应全 -inf
    assert torch.all(state.m == float('-inf'))


def test_single_token_selection():
    """单 token selection → 输出 = v[token]。"""
    q, k, v, lengths = _make_inputs(h=4, hk=4, d=16, dv=16)
    t = k.shape[1]
    scale = 1.0 / (16 ** 0.5)
    # 选最后一个 token
    ids = torch.full((1, 4, 8), t - 1, dtype=torch.int32)
    counts = torch.full((1, 4), 1, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=0, swa=0, scale=scale,
                              protected=False, splits=1, backend='torch')
    empty = PartialState(
        torch.full((1, 4, 1), -torch.inf),
        torch.zeros(1, 4, 1),
        torch.zeros(1, 4, 1, 16))
    out = merge_partials(state, empty, dtype=torch.float32, backend='torch')

    # 单 token softmax → 1.0，输出 = v[token]
    expected = v[:, t - 1].unsqueeze(0)  # [1,1,4,16]
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_gqa_grouping():
    """GQA：Hq=8, Hkv=2 → group_size=4。"""
    q, k, v, lengths = _make_inputs(h=8, hk=2, d=16, dv=16)
    t = k.shape[1]
    scale = 1.0 / (16 ** 0.5)
    ids = torch.arange(t, device=q.device).unsqueeze(0).unsqueeze(0).expand(
        1, 2, -1).to(torch.int32)
    counts = torch.full((1, 2), t, device=q.device, dtype=torch.int32)
    selection = (ids, counts)

    state = partial_attention(q, k, v, selection, lengths,
                              sink=0, swa=0, scale=scale,
                              protected=False, splits=1, backend='torch')
    empty = PartialState(
        torch.full((1, 8, 1), -torch.inf),
        torch.zeros(1, 8, 1),
        torch.zeros(1, 8, 1, 16))
    out = merge_partials(state, empty, dtype=torch.float32, backend='torch')

    # 参考：手动 GQA attention
    mask = torch.ones(1, t, dtype=torch.bool)
    ref = _reference_attention(q, k, v, mask, scale)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires native CUDA')
@pytest.mark.parametrize('capacity', [16384, 24576, 32768])
@pytest.mark.parametrize('count', [0, 37])
@pytest.mark.parametrize('pipeline_stages', [1, 2, 3])
def test_cuda_padded_selection_active_bound(capacity, count, pipeline_stages):
    """Empty splits and partial tiles must retain the protected softmax state."""
    torch.manual_seed(42)
    q = torch.randn(1, 1, 4, 32, device='cuda')
    k = torch.randn(1, capacity, 1, 32, device='cuda')
    v = torch.randn_like(k)
    ids = torch.full((1, 1, capacity), -1, device='cuda', dtype=torch.int32)
    ids[0, 0, :count] = torch.arange(4, 4 + count, device='cuda')
    counts = torch.tensor([[count]], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([capacity], device='cuda', dtype=torch.int64)
    kwargs = dict(sink=4, swa=64, scale=32 ** -.5, splits=4,
                  pipeline_stages=pipeline_stages)
    p = partial_attention(q, k, v, (ids, counts), lengths, protected=True, **kwargs)
    s = partial_attention(q, k, v, (ids, counts), lengths, **kwargs)
    out = merge_partials(p, s, q.dtype)
    mask = torch.zeros(capacity, device='cuda', dtype=torch.bool)
    mask[:4 + count] = True
    mask[-64:] = True
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.cpu().transpose(1, 2), k.cpu().transpose(1, 2), v.cpu().transpose(1, 2),
        attn_mask=mask.cpu(), enable_gqa=True).transpose(1, 2).to(q.device)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires native CUDA')
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@pytest.mark.parametrize('pipeline_stages', [2, 3])
def test_cuda_count_changes_during_graph_replay(backend, pipeline_stages):
    q = torch.randn(1, 1, 4, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 256, 1, 32, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    ids = torch.arange(256, device='cuda', dtype=torch.int32)[None, None]
    counts = torch.tensor([[0]], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([256], device='cuda', dtype=torch.int64)
    def run():
        kw = dict(sink=4, swa=16, scale=32 ** -.5, splits=4,
                  backend=backend, pipeline_stages=pipeline_stages)
        p = partial_attention(q, k, v, (ids, counts), lengths, protected=True, **kw)
        s = partial_attention(q, k, v, (ids, counts), lengths, **kw)
        return merge_partials(p, s, q.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for count in (0, 37, 256, 0):
        counts.fill_(count)
        graph.replay()
        expected = run()
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        ref_p = partial_attention(q.cpu(), k.cpu(), v.cpu(), (ids.cpu(), counts.cpu()),
                                 lengths.cpu(), sink=4, swa=16, scale=32 ** -.5,
                                 protected=True, backend='torch')
        ref_s = partial_attention(q.cpu(), k.cpu(), v.cpu(), (ids.cpu(), counts.cpu()),
                                 lengths.cpu(), sink=4, swa=16, scale=32 ** -.5,
                                 backend='torch')
        ref = merge_partials(ref_p, ref_s, q.dtype, backend='torch')
        torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('protected', [False, True])
@pytest.mark.parametrize('default_dtype', [torch.float32, torch.float64])
def test_empty_state_dtype_and_merge(protected, default_dtype):
    """The neutral-state contract is FP32 regardless of global defaults."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        q, k, v, lengths = _make_inputs(h=4, hk=1)
        ids = torch.empty((1, 1, 0), dtype=torch.int32)
        counts = torch.zeros((1, 1), dtype=torch.int32)
        empty = partial_attention(q, k, v, (ids, counts), lengths,
                                  sink=0, swa=0, scale=.25, protected=protected,
                                  splits=4, backend='torch')
        assert all(t.dtype == torch.float32 for t in empty)
        assert torch.isneginf(empty.m).all()
        assert torch.count_nonzero(empty.l) == torch.count_nonzero(empty.acc) == 0
        selected = torch.arange(64, dtype=torch.int32)[None, None]
        full = partial_attention(q, k, v, (selected, counts + 64), lengths,
                                 sink=0, swa=0, scale=.25, splits=2, backend='torch')
        out = merge_partials(empty, full, q.dtype, backend='torch')
        expected = merge_partials(full, full, q.dtype, backend='torch')
        torch.testing.assert_close(out, expected)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires native CUDA')
@pytest.mark.parametrize('backend', ['simt', 'tc'])
def test_cuda_ring_length_changes_during_graph_replay(backend):
    """Replay must read lengths, reject evicted IDs and handle sink/SWA overlap."""
    from sparse_attn.ops.far_skip import FarSkipLayout
    torch.manual_seed(71)
    layout = FarSkipLayout(sink=4, tail_capacity=64)
    q = torch.randn(1, 1, 4, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(1, layout.capacity, 1, 32, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    ids = torch.arange(256, device='cuda', dtype=torch.int32)[None, None]
    counts = torch.full((1, 1), 256, device='cuda', dtype=torch.int32)
    lengths = torch.tensor([256], device='cuda', dtype=torch.int64)
    def run(q, k, v, ids, counts, lengths, impl):
        kw = dict(sink=4, swa=16, scale=32 ** -.5, splits=4,
                  backend=impl, layout=layout)
        p = partial_attention(q, k, v, (ids, counts), lengths, protected=True, **kw)
        s = partial_attention(q, k, v, (ids, counts), lengths, **kw)
        return merge_partials(p, s, q.dtype, backend=impl)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(q, k, v, ids, counts, lengths, backend)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(q, k, v, ids, counts, lengths, backend)
    for length in (0, 3, 19, 68, 129, 256, 0):
        lengths.fill_(length)
        graph.replay()
        expected = run(q.cpu(), k.cpu(), v.cpu(), ids.cpu(), counts.cpu(),
                       lengths.cpu(), 'torch')
        torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires native CUDA')
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@pytest.mark.parametrize('protected', [False, True])
def test_cuda_static_empty_nonfinite_and_default_dtype(backend, protected):
    """A static empty task must not consume nonfinite Q/K/V or default dtype."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        q = torch.full((1, 1, 4, 32), torch.nan, device='cuda', dtype=torch.float16)
        k = torch.full((1, 64, 1, 32), torch.inf, device='cuda', dtype=q.dtype)
        ids = torch.empty((1, 1, 0), device='cuda', dtype=torch.int32)
        counts = torch.zeros((1, 1), device='cuda', dtype=torch.int32)
        lengths = torch.tensor([64], device='cuda', dtype=torch.int64)
        state = partial_attention(q, k, k, (ids, counts), lengths,
                                  sink=0, swa=0, scale=.25, backend=backend,
                                  protected=protected)
        assert all(t.dtype == torch.float32 for t in state)
        assert torch.isneginf(state.m).all()
        assert torch.count_nonzero(state.l) == torch.count_nonzero(state.acc) == 0
        out = merge_partials(state, state, q.dtype)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires native CUDA')
@pytest.mark.parametrize('backend', ['simt', 'tc'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('scale', [0.1767766952966369, 0.123456789])
@pytest.mark.parametrize('pipeline_stages', [1, 3])
def test_cuda_fused_scale_preserves_model_rounding(backend, dtype, scale, pipeline_stages):
    """Fused scaling must match a materialized model-dtype Q before attention."""
    torch.manual_seed(719)
    q = torch.randn(2, 1, 8, 32, device='cuda', dtype=dtype)
    k = torch.randn(2, 129, 2, 32, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    ids = torch.arange(129, device='cuda', dtype=torch.int32)[None, None].expand(2, 2, -1).contiguous()
    counts = torch.full((2, 2), 129, device='cuda', dtype=torch.int32)
    lengths = torch.full((2,), 129, device='cuda', dtype=torch.int64)
    kwargs = dict(sink=4, swa=16, splits=4, backend=backend,
                  pipeline_stages=pipeline_stages)
    fused = partial_attention(q, k, v, (ids, counts), lengths, scale=scale, **kwargs)
    materialized = partial_attention((q * scale).contiguous(), k, v,
                                    (ids, counts), lengths, scale=1., **kwargs)
    for actual, expected in zip(fused, materialized):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
