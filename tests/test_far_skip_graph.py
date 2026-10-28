"""Fixed geometry oracle tests; real CUDA capture tests are never CPU-emulated."""
from dataclasses import replace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from sparse_attn.ops.far_skip import FarSkipConfig, select_far_skip, attend_far_skip
from sparse_attn.ops.far_skip_cache import evict_far
from sparse_attn.ops.far_skip_graph import FarSkipGraphWorkspace
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace, stable_topk_positions
from sparse_attn.ops.tli_static_cache import TLIStaticCache, fill_static_from_dynamic
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.cuda_graph_state import enable_graph, disable_graph, graph_state
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize('method', ['avg', 'minmax'])
@pytest.mark.parametrize('quant', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('sink,window,pages', [(0, 0, 2), (3, 5, 0), (3, 5, 2), (8, 8, 10)])
@torch.inference_mode()
def test_fixed_selector_against_dynamic(method, quant, dtype, sink, window, pages):
    torch.manual_seed(153)
    cfg = FarSkipConfig(4, 20, pages, sink+window+7, sink, window, method, quant)
    coarse, fine = torch.arange(8), torch.arange(4, 12)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 1, 12, dtype=dtype)
    cache.update(k, v, 0)
    ring = evict_far(cache, 0, cfg)
    fixed = FarSkipGraphWorkspace(cfg, ring.layout, 2, coarse, fine)
    fixed.initialize(ring.keys.transpose(1, 2), 1)
    ws = FarSkipWorkspace()
    for length in range(2, 130):
        k, v = torch.randn(2, 1, 2, 1, 12, dtype=dtype)
        ring.update(k, v)
        q = torch.randn(1, 1, 6, 12, dtype=dtype)
        kk, vv = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
        ref = select_far_skip(q, kk, length, cfg, coarse, fine, layout=ring.layout, workspace=ws)
        ids, mask = fixed.select(q, kk, torch.tensor([length]), 12**-.5)
        for h in range(2):
            assert torch.equal(ids[h, mask[h]], ref.positions[h]), (length, h)
        expected = attend_far_skip(q, kk, vv, ref, cfg, layout=ring.layout)
        actual = fixed.attend(q, kk, vv, ids, mask, 12**-.5)
        torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_grouped_query_axis_avoids_kv_repeat():
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4)
    q = torch.randn(1, 1, 6, 12)
    k, v = torch.randn(2, 1, 39, 2, 12)
    sel = select_far_skip(q, k, 39, cfg, torch.arange(8), torch.arange(8))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        attend_far_skip(q, k, v, sel, cfg)
    assert not any('repeat_interleave' in e.key for e in p.key_averages())


@torch.inference_mode()
def test_fixed_padding_never_reads_poisoned_slots_and_ties():
    cfg = FarSkipConfig(4, 20, 2, 12, 3, 4)
    for length in [1, 3, 7, 11, 36, 99]:
        cache = DynamicCache()
        cache.update(torch.zeros(1, 2, length, 12), torch.ones(1, 2, length, 12), 0)
        ring = evict_far(cache, 0, cfg)
        safe = torch.cat([torch.arange(min(cfg.sink_tokens, length)),
                          torch.arange(cfg.tail_start(length), length)])
        poison = torch.ones(ring.layout.capacity, dtype=torch.bool)
        poison[ring.layout.map(safe)] = False
        ring.keys[:, :, poison] = float('nan')
        ring.values[:, :, poison] = float('nan')
        fixed = FarSkipGraphWorkspace(cfg, ring.layout, 2, torch.arange(8), torch.arange(8))
        fixed.initialize(ring.keys.transpose(1, 2), length)
        q = torch.zeros(1, 1, 6, 12)
        k, v = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
        ref = select_far_skip(q, k, length, cfg, torch.arange(8), torch.arange(8), layout=ring.layout)
        ids, mask = fixed.select(q, k, torch.tensor([length]), 12**-.5)
        for h in range(2):
            assert torch.equal(ids[h, mask[h]], ref.positions[h])
        torch.testing.assert_close(fixed.attend(q, k, v, ids, mask, 12**-.5), torch.ones_like(q))


def _qwen_pair(device, *, capture=False, tensor_position=False):
    from test_far_skip_compact import make_model
    a, b = [make_model('evict', dynamic=False).to(device) for _ in range(2)]
    if device == 'cuda':
        a, b = a.bfloat16(), b.bfloat16()
    static = TLIStaticCache(b.config, 256, 1)
    token = torch.zeros(1, 1, dtype=torch.long, device=device)
    pos = torch.zeros(1, dtype=torch.long, device=device)
    mask = torch.full((1, 1, 1, 256), -torch.inf, device=device, dtype=next(b.parameters()).dtype)
    if capture or tensor_position:
        enable_graph(1)
        graph_state().pos_buf = pos
        graph_state().valid_len_buf = pos + 1
    try:
        for prompt_length in (39, 7):
            caches = [DynamicCache(), DynamicCache()]
            ids = torch.arange(prompt_length, device=device)[None] % 32
            a(input_ids=ids, past_key_values=caches[0])
            ctx = BatchCtx(1, [prompt_length], [0], [True], 0)
            with batch_decode_scope(ctx):
                b(input_ids=ids, past_key_values=caches[1])
                fill_static_from_dynamic(static, caches[1], [prompt_length], model=b)
                static._far_skip_attention_mask = mask
                assert all(isinstance(l, FarSkipStaticLayer) for l in static.layers)
                pointers = [l.keys.data_ptr() for l in static.layers]
                def forward():
                    return b(input_ids=token, position_ids=pos[None], cache_position=pos,
                             attention_mask=mask, past_key_values=static).logits
                graph = None
                for step in range(73):
                    length = prompt_length+step+1
                    ctx.seq_lens[0] = length
                    token.fill_(step % 32)
                    pos.fill_(length-1)
                    mask.fill_(-torch.inf)
                    mask[..., :length] = 0
                    if capture and graph is None:
                        # Warm up on a side stream, capture once, then change
                        # token, RoPE position and ring write position on replay.
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2):
                                forward()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = forward()
                    if graph is not None:
                        graph.replay()
                        actual = graph_output
                    else:
                        actual = forward()
                    # Dynamic model must run without the batch context.
                    with batch_decode_scope(None):
                        expected = a(input_ids=token, past_key_values=caches[0]).logits
                    torch.testing.assert_close(actual, expected,
                                               atol=1e-2 if device == 'cuda' else 3e-6,
                                               rtol=1e-2 if device == 'cuda' else 3e-5)
                    assert pointers == [l.keys.data_ptr() for l in static.layers]
    finally:
        disable_graph()


@pytest.mark.parametrize('tensor_position', [False, True])
@torch.inference_mode()
def test_real_qwen_staged_static_and_tensor_position(tensor_position):
    _qwen_pair('cpu', tensor_position=tensor_position)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA capture/replay')
@torch.inference_mode()
def test_real_qwen_cuda_graph_replay_and_request_recapture():
    _qwen_pair('cuda', capture=True)


@torch.inference_mode()
def test_graph_runner_injection_preserves_mask_on_growth_repeat_and_rewind():
    from exp.bench.e152v4_graph_latency import GraphRunner
    model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    runner = GraphRunner(model, None, torch.finfo(torch.bfloat16).min, 256, 0, 'cpu')
    addresses = [x.data_ptr() for x in [runner.nxt_buf, runner.pid_buf, runner.pos_buf,
                                       runner.vl_buf, runner.m4_buf]]
    for length in [39, 40, 40, 41, 7, 8, 256, 1]:
        runner.inject(17, length)
        assert runner.nxt_buf.item() == 17
        assert runner.pid_buf.item() == runner.pos_buf.item() == length-1
        assert runner.vl_buf.item() == length
        assert (runner.m4_buf[..., :length] == 0).all()
        assert (runner.m4_buf[..., length:] == runner.neg_inf).all()
        assert addresses == [x.data_ptr() for x in [runner.nxt_buf, runner.pid_buf, runner.pos_buf,
                                                   runner.vl_buf, runner.m4_buf]]
    with pytest.raises(ValueError, match='capacity'):
        runner.inject(17, 257)


@torch.inference_mode()
def test_static_setup_mixed_gates_masks_and_mutation_guards():
    from test_far_skip_compact import make_model
    from sparse_attn.ops.far_skip_runtime import try_far_skip_decode
    model = make_model('evict', dynamic=False)
    dyn = DynamicCache()
    model(input_ids=torch.arange(79)[None] % 32, past_key_values=dyn)
    static = TLIStaticCache(model.config, 256, 1)
    modules = [layer.self_attn for layer in model.model.layers]
    for m in modules:
        m.indexer._skip_ids = {0}
    fill_static_from_dynamic(static, dyn, [79], model=model)
    assert isinstance(static.layers[0], FarSkipStaticLayer)
    assert not isinstance(static.layers[1], FarSkipStaticLayer)
    assert static.layers[0].keys.shape[-2] < 79
    ctx = BatchCtx(1, [80], [1], [True], 0)
    q, k, v = torch.randn(1, 2, 1, 128), *torch.randn(2, 1, 1, 1, 128)
    before = static.layers[0].keys.clone()
    with pytest.raises(ValueError, match='registered'):
        try_far_skip_decode(modules[0], q, k, v, static, torch.ones(1, 256), None, ctx)
    torch.testing.assert_close(before, static.layers[0].keys)
    with pytest.raises(ValueError, match='match detached'):
        try_far_skip_decode(modules[0], q, k, v[..., :1], static, None, None, ctx)
    torch.testing.assert_close(before, static.layers[0].keys)
    # Far-on routing leaves the ordinary update to its caller.
    old = static.layers[1].keys.clone()
    assert try_far_skip_decode(modules[1], q, k, v, static, None, None, ctx) is None
    torch.testing.assert_close(old, static.layers[1].keys)
    # New request gates can change in either direction; fill restores far-on
    # full storage and allocates a new ring for the newly skipped layer.
    for m in modules:
        m.indexer._skip_ids = {1}
    fill_static_from_dynamic(static, dyn, [79], model=model)
    assert not isinstance(static.layers[0], FarSkipStaticLayer)
    assert isinstance(static.layers[1], FarSkipStaticLayer)
    torch.testing.assert_close(static.layers[0].keys[..., :79, :], dyn.layers[0].keys)


@torch.inference_mode()
def test_staged_position_reuse_rejected_before_mutation():
    from test_far_skip_compact import make_model
    from sparse_attn.ops.far_skip_runtime import try_far_skip_decode
    model = make_model('evict', dynamic=False)
    dyn = DynamicCache()
    model(input_ids=torch.arange(79)[None] % 32, past_key_values=dyn)
    static = TLIStaticCache(model.config, 256, 1)
    fill_static_from_dynamic(static, dyn, [79], model=model)
    module = model.model.layers[0].self_attn
    q, k, v = torch.randn(1, 2, 1, 128), *torch.randn(2, 1, 1, 1, 128)
    ctx = BatchCtx(1, [80], [1], [True], 0)
    try_far_skip_decode(module, q, k, v, static, None, None, ctx)
    before = static.layers[0].keys.clone()
    for length in (80, 79, 82):
        ctx.seq_lens[0] = length
        with pytest.raises(ValueError, match='next position'):
            try_far_skip_decode(module, q, k+1, v, static, None, None, ctx)
        torch.testing.assert_close(before, static.layers[0].keys)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_invalid_topk_lanes_rank_after_valid_negative_infinity(dtype):
    scores = torch.tensor([[float('nan'), -torch.inf, -torch.inf, torch.inf]], dtype=dtype)
    valid = torch.tensor([[False, True, True, False]])
    assert stable_topk_positions(scores, 2, FarSkipWorkspace(), valid=valid).tolist() == [[1, 2]]


@pytest.mark.parametrize('bs,near', [(1, 5), (4, 21), (8, 8), (8, 25)])
@torch.inference_mode()
def test_graph_non_aligned_near_and_page_capacity(bs, near):
    torch.manual_seed(71)
    cfg = FarSkipConfig(bs, near, 1, 11, 2, 3)
    cache = DynamicCache()
    cache.update(torch.randn(1, 1, 49, 12), torch.randn(1, 1, 49, 12), 0)
    ring = evict_far(cache, 0, cfg)
    dims = torch.arange(8)
    fixed = FarSkipGraphWorkspace(cfg, ring.layout, 1, dims, dims)
    fixed.initialize(ring.keys.transpose(1, 2), 49)
    for length in range(50, 109):
        ring.update(torch.randn(1, 1, 1, 12), torch.randn(1, 1, 1, 12))
        q = torch.randn(1, 1, 2, 12)
        k = ring.keys.transpose(1, 2)
        ref = select_far_skip(q, k, length, cfg, dims, dims, layout=ring.layout)
        pos, valid = fixed.select(q, k, torch.tensor([length]), 12**-.5)
        assert torch.equal(pos[0, valid[0]], ref.positions[0])


@torch.inference_mode()
def test_fixed_summary_update_is_one_page_without_scalar_readback():
    cfg = FarSkipConfig(4, 32, 2, 12, 3, 4)
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 99, 12), torch.randn(1, 2, 99, 12), 0)
    ring = evict_far(cache, 0, cfg)
    ws = FarSkipGraphWorkspace(cfg, ring.layout, 2, torch.arange(8), torch.arange(8))
    ws.initialize(ring.keys.transpose(1, 2), 99)
    ring.update(torch.randn(1, 2, 1, 12), torch.randn(1, 2, 1, 12))
    q, length = torch.randn(1, 1, 6, 12), torch.tensor([100])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
        ws.select(q, ring.keys.transpose(1, 2), length, 12**-.5)
    assert not any(e.name in ('aten::item', 'aten::_local_scalar_dense') for e in p.events())
    reduced = [e.input_shapes[0] for e in p.events() if e.name == 'aten::mean']
    assert [1, 4, 2, 8] in reduced
    assert not any(len(shape) == 4 and shape[0] > 1 for shape in reduced)


@pytest.mark.parametrize('method,quant', [('avg', False), ('minmax', True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA capture/replay')
@torch.inference_mode()
def test_cuda_fixed_selection_ids_and_ring_replay(method, quant):
    cfg = FarSkipConfig(4, 21, 2, 12, 3, 4, method, quant)
    torch.manual_seed(291)
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 2, 79, 12, device='cuda', dtype=torch.bfloat16)
    cache.update(k, v, 0)
    ring = evict_far(cache, 0, cfg)
    coarse, fine = torch.arange(8, device='cuda'), torch.arange(4, 12, device='cuda')
    ws = FarSkipGraphWorkspace(cfg, ring.layout, 2, coarse, fine)
    kk, vv = ring.keys.transpose(1, 2), ring.values.transpose(1, 2)
    ws.initialize(kk, 79)
    q = torch.zeros(1, 1, 6, 12, device='cuda', dtype=torch.bfloat16)
    nk, nv = torch.zeros(2, 1, 2, 1, 12, device='cuda', dtype=torch.bfloat16)
    pos = torch.tensor([79], device='cuda')
    def step():
        slot = ring.layout.map(pos)
        ring.keys.index_copy_(2, slot, nk)
        ring.values.index_copy_(2, slot, nv)
        ids, valid = ws.select(q, kk, pos+1, 12**-.5)
        return ids, valid, ws.attend(q, kk, vv, ids, valid, 12**-.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(); step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, valid, actual = step()
    for length in range(80, 153):
        q.normal_(); nk.normal_(); nv.normal_(); pos.fill_(length-1)
        graph.replay()
        ref = select_far_skip(q, kk, length, cfg, coarse, fine, layout=ring.layout)
        for h in range(2):
            assert torch.equal(ids[h, valid[h]], ref.positions[h])
        expected = attend_far_skip(q, kk, vv, ref, cfg, layout=ring.layout)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
