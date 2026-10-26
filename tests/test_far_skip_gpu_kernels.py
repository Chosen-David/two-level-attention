"""TRITON_INTERPRET=1 checks logic on CPU; it is not a CUDA execution test."""
import os

import pytest
import torch

pytest.importorskip('triton')
from sparse_attn.ops import far_skip_accel as accel
from sparse_attn.ops.far_skip import FarSkipConfig, FarSkipLayout, select_far_skip, attend_far_skip
from sparse_attn.ops.far_skip_cache import evict_far
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace
from transformers.cache_utils import DynamicCache


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and (not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1'):
        pytest.skip('requires real CUDA, without interpreter')
    if request.param == 'cpu' and os.environ.get('TRITON_INTERPRET') != '1':
        pytest.skip('CPU logic checks require TRITON_INTERPRET=1')
    return request.param


@pytest.mark.parametrize('n', [17, 513, 2334])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_order_keys_total_order_strides_and_padding(device, n, dtype):
    if device == 'cpu' and dtype == torch.bfloat16:
        pytest.skip('Triton interpreter does not validate BF16 arithmetic')
    torch.manual_seed(917)
    scores = torch.randn(n, 4, device=device, dtype=dtype).T
    scores[0] = 0
    scores[1, :6] = torch.tensor([0., -0., torch.inf, -torch.inf, torch.nan, -torch.nan], device=device)
    one = torch.tensor(1., device=device, dtype=dtype)
    scores[2, :3] = torch.stack([one, torch.nextafter(one, one+1), torch.nextafter(one, one-1)])
    valid = (torch.arange(n, device=device) % 3 != 0)[None].expand(4, -1)
    ws = FarSkipWorkspace()
    for mask in (None, valid):
        keys = accel.order_keys(scores, mask, ws)
        order = scores.argsort(dim=-1, descending=True, stable=True)
        if mask is not None:
            order = order.gather(-1, mask.gather(-1, order).to(torch.int8).argsort(dim=-1, descending=True, stable=True))
        ranked = keys.argsort(dim=-1, descending=True)
        if mask is None:
            assert torch.equal(ranked, order)
        else:
            # Padding keys intentionally tie at INT64_MIN; their internal order
            # is unobservable. Every valid lane must precede all padding.
            for h in range(4):
                count = int(mask[h].sum())
                assert torch.equal(ranked[h, :count], order[h, :count])
            assert (keys[~mask] == torch.iinfo(torch.int64).min).all()
        pointer = keys.data_ptr()
        assert accel.order_keys(scores, mask, ws).data_ptr() == pointer


@pytest.mark.parametrize('ring', [False, True])
@pytest.mark.parametrize('project', [False, True])
def test_gather_pair_and_projection_poison_padding_strides(device, ring, project):
    torch.manual_seed(19)
    layout = FarSkipLayout(3, 23) if ring else None
    cap, h, d = 26, 2, 12
    # Both sequence-major and HF head-major inputs must use their actual stride.
    k = torch.randn(1, h, cap, d, device=device).transpose(1, 2)
    v = torch.randn(1, cap, h, d*2, device=device)[..., ::2]
    ids = torch.tensor([[0, 1, 2, 25, 24, 22, 18], [2, 1, 0, 23, 21, 20, 19]], device=device)
    if ring:
        ids[:, 3:] += 23*100000000
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, -1] = False
    ids[:, -1] = 2**50  # invalid padding must never turn into a cache read
    k[:, 18:20] = torch.nan
    v[:, 18:20] = torch.nan
    dims = torch.tensor([11, 0, 2, 0, 5, 0, 1, 0], device=device)[::2] if project else None
    safe = ids.masked_fill(~valid, 0)
    physical = layout.map(safe) if layout else safe
    heads = torch.arange(h, device=device)[:, None]
    expected = []
    for source in (k, v):
        result = source[0][physical, heads]
        if dims is not None:
            result = result[..., dims]
        expected.append(result.masked_fill(~valid[..., None], 0))
    ws = FarSkipWorkspace()
    actual = accel.gather(k, ids, layout, ws, other=v, dims=dims, valid=valid)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('tensor_position', [False, True])
def test_write_pair_only_changes_one_ring_slot(device, tensor_position):
    layout = FarSkipLayout(3, 23)
    k, v = torch.randn(2, 1, 2, 26, 12, device=device)
    rk, rv = k.clone(), v.clone()
    for absolute in (0, 2, 3, 25, 26, 27, 70, 2**33):
        nk, nv = torch.randn(2, 1, 2, 1, 24, device=device)[..., ::2]
        pos = torch.tensor([absolute], device=device) if tensor_position else absolute
        accel.write_pair(k, v, nk, nv, pos, layout)
        slot = layout.slot(absolute)
        rk[:, :, slot:slot+1].copy_(nk)
        rv[:, :, slot:slot+1].copy_(nv)
        torch.testing.assert_close(k, rk, rtol=0, atol=0)
        torch.testing.assert_close(v, rv, rtol=0, atol=0)


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('quantize', [False, True])
@pytest.mark.parametrize('method', ['avg', 'minmax'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_complete_selector_ids_and_outputs_across_wrap(device, fixed, quantize, method, dtype, monkeypatch):
    if device == 'cpu' and dtype == torch.bfloat16:
        pytest.skip('BF16 end-to-end kernel validation requires CUDA')
    torch.manual_seed(442)
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4, method, quantize)
    dims = torch.tensor([11, 3, 5, 1], device=device)
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 59, 12, device=device, dtype=dtype),
                 torch.randn(1, 2, 59, 12, device=device, dtype=dtype), 0)
    ring = evict_far(cache, 0, cfg)
    k, v = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
    ws = FarSkipGraphWorkspace(cfg, ring.layout, 2, dims, dims) if fixed else FarSkipWorkspace()
    monkeypatch.setattr(accel, 'enabled', lambda _: True)
    if fixed:
        ws.initialize(k, 59)
    for length in range(60, 92):
        ring.update(torch.randn(1, 2, 1, 12, device=device, dtype=dtype),
                    torch.randn(1, 2, 1, 12, device=device, dtype=dtype))
        q = torch.randn(1, 1, 6, 12, device=device, dtype=dtype)
        with monkeypatch.context() as ref:
            ref.setattr(accel, 'enabled', lambda _: False)
            selected = select_far_skip(q, k, length, cfg, dims, dims, layout=ring.layout)
            expected = attend_far_skip(q, k, v, selected, cfg, layout=ring.layout)
        if fixed:
            ids, valid = ws.select(q, k, torch.tensor([length], device=device), 12**-.5)
            actual = ws.attend(q, k, v, ids, valid, 12**-.5)
            for h in range(2):
                assert torch.equal(ids[h, valid[h]], selected.positions[h])
        else:
            s = select_far_skip(q, k, length, cfg, dims, dims, layout=ring.layout, workspace=ws)
            assert torch.equal(s.positions, selected.positions)
            actual = attend_far_skip(q, k, v, s, cfg, layout=ring.layout, workspace=ws)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1',
                    reason='requires real CUDA capture/replay')
@torch.inference_mode()
def test_fused_order_and_kv_capture_changes_inputs(monkeypatch):
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS', 'triton')
    layout = FarSkipLayout(3, 23)
    k, v = [torch.randn(1, 2, 26, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
            for _ in range(2)]
    ids = torch.arange(8, device='cuda')[None].expand(2, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    scores = torch.randn(2, 17, device='cuda')
    ws = FarSkipWorkspace()
    pos = torch.tensor([32], device='cuda')
    nk, nv = [torch.randn(1, 2, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    def run():
        accel.write_pair(k.transpose(1, 2), v.transpose(1, 2), nk, nv, pos, layout)
        keys = accel.order_keys(scores, None, ws)
        kk, vv = accel.gather(k, ids, layout, ws, other=v, valid=valid)
        return keys, kk, vv
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(); run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for step in range(10):
        pos.add_(1)
        nk.normal_(); nv.normal_()
        ids.add_(1)
        valid[:, step % 8] = False
        scores.normal_()
        graph.replay()
        slot = layout.slot(int(pos.item()))
        torch.testing.assert_close(k[:, slot:slot+1], nk.transpose(1, 2), rtol=0, atol=0)
        for h in range(2):
            for source, actual in zip((k, v), out[1:]):
                reference = source[0, layout.map(ids[h]), h].masked_fill(~valid[h, :, None], 0)
                torch.testing.assert_close(actual[h], reference, rtol=0, atol=0)
        assert torch.equal(out[0].argsort(dim=-1, descending=True), scores.argsort(dim=-1, descending=True, stable=True))
