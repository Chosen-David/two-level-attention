"""真实 Qwen 多请求 GraphRunner 的 eager/Graph token identity 门禁。"""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from sparse_attn.cache import GraphReadyPagedRegionCache
from sparse_attn.cache.paged_region_cache import RequestPhase
from sparse_attn.engine.graph_runner import GraphRunner
from sparse_attn.info import resolve_treatment_snapshot
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.patches import register_patch
from sparse_attn.policies.region_geometry import RegionGeometry
from sparse_attn.scheduler.scheduler import ContinuousBatchScheduler


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
GEOMETRY = RegionGeometry(sink=4, near=16, window=8, page=8, max_new_tokens=8)
LENGTHS = {0: 40, 1: 55, 2: 73, 3: 96}
REQUEST_ORDER = [1, 3, 0, 2]


def _model():
    torch.manual_seed(20261102)
    config = Qwen3Config(
        vocab_size=128, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, attention_dropout=0., max_position_embeddings=256)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
    args = nfa_args()
    args.nfa_near_len = GEOMETRY.near
    args.nfa_swa_size = GEOMETRY.window
    args.nfa_sink_tokens = GEOMETRY.sink
    args.nfa_near_pages = 2
    args.tia_block_size = GEOMETRY.page
    args.tia_level1_topk = 4
    args.tia_level2_topk = 16
    assert register_patch(model, args, resolve_treatment_snapshot(args)) == 2
    return model


def _input_ids(request_id):
    return torch.arange(
        2, 2 + LENGTHS[request_id], dtype=torch.long, device='cuda').remainder(120).unsqueeze(0)


def _prefill_cache(model, req_ids):
    cache = GraphReadyPagedRegionCache(
        2, GEOMETRY, max_requests=4, max_physical_pages=128, max_blocks=32,
        max_visible_len=192, max_query_tokens=4, num_kv_heads=2,
        num_query_heads=4, head_dim_k=64, head_dim_v=64,
        dtype=torch.bfloat16, device='cuda')
    first_tokens = {}
    for request_id in req_ids:
        cache.add_request(request_id)
        cache.set_active_request(request_id)
        output = model(
            _input_ids(request_id), past_key_values=cache, use_cache=True,
            logits_to_keep=1)
        first_tokens[request_id] = int(output.logits[0, -1].argmax())
        # 每个请求使用不同的四态组合，且与 batch row 顺序无隐含对应。
        cache.freeze(
            request_id,
            [request_id % 2 == 0] * 2,
            [request_id % 3 == 0] * 2,
            {0: [], 1: []})
    return cache, first_tokens


@torch.inference_mode()
def _decode(model, req_ids, *, capture_graph):
    cache, first_tokens = _prefill_cache(model, req_ids)
    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=cache.max_requests,
        eos_ids={127}, pad_token_id=0)
    for request_id in req_ids:
        scheduler.add_request(request_id)
        scheduler.next_tokens[request_id] = first_tokens[request_id]
    runner = GraphRunner(
        scheduler=scheduler, buckets=(1, 2, 4), device='cuda')
    if capture_graph:
        runner.capture_all(model, cache)

    generated = {request_id: [] for request_id in req_ids}
    for _ in range(6):
        active_count = len(scheduler.active_req_ids)
        bucket = runner.bucket_for(active_count, cache.max_requests)
        padded = scheduler.active_req_ids + [-1] * (bucket - active_count)
        counts = [len(generated[rid]) if rid >= 0 else 0 for rid in padded]
        _, tokens = runner.step(
            model, cache, generated_counts=counts, min_new_tokens=6,
            return_next_tokens=True)
        for row, request_id in enumerate(scheduler.active_req_ids):
            generated[request_id].append(int(tokens[row]))
    return generated


@pytest.mark.parametrize('capture_graph', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 4])
def test_multi_request_matches_isolated_b1_eager_and_graph(batch, capture_graph):
    model = _model()
    req_ids = REQUEST_ORDER[:batch]
    batched = _decode(model, req_ids, capture_graph=capture_graph)
    for request_id in req_ids:
        isolated = _decode(model, [request_id], capture_graph=capture_graph)
        assert batched[request_id] == isolated[request_id]


@pytest.mark.parametrize('capture_graph', [False, True])
@torch.inference_mode()
def test_dynamic_eos_completion_and_request_admission(capture_graph):
    model = _model()
    cache, first_tokens = _prefill_cache(model, [1, 3])
    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=cache.max_requests,
        eos_ids={127}, pad_token_id=0)
    for request_id in [1, 3]:
        scheduler.add_request(request_id)
        scheduler.next_tokens[request_id] = first_tokens[request_id]
        cache.transition_request(request_id, RequestPhase.GENERATE)
    runner = GraphRunner(
        scheduler=scheduler, buckets=(1, 2, 4), device='cuda')
    if capture_graph:
        runner.capture_all(model, cache)

    # 强制词元只用于确定性触发逐行 EOS；模型 forward、页写入、选择与
    # Graph/eager 路径仍真实执行。request 3 先退出，随后 request 0 动态加入。
    script = {1: [11, 12, 13, 127], 3: [21, 127], 0: [31, 32, 33]}
    generated = {1: [], 3: [], 0: []}
    admitted = False
    for _ in range(7):
        # 请求边界显式 drain 上一步异步 D2H，EOS 在这里转化为终态和槽位释放。
        scheduler.prepare_step()
        for request_id in list(scheduler.active_req_ids):
            if len(generated[request_id]) == len(script[request_id]):
                scheduler.finish_request(request_id)
        if not scheduler.active_req_ids:
            break
        if 3 not in cache.states and not admitted:
            request_id = 0
            cache.add_request(request_id)
            cache.set_active_request(request_id)
            output = model(
                _input_ids(request_id), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
            cache.freeze(request_id, [True] * 2, [True] * 2, {0: [], 1: []})
            scheduler.add_request(request_id)
            scheduler.next_tokens[request_id] = int(output.logits[0, -1].argmax())
            cache.transition_request(request_id, RequestPhase.GENERATE)
            admitted = True

        active = list(scheduler.active_req_ids)
        bucket = runner.bucket_for(len(active), cache.max_requests)
        padded = active + [-1] * (bucket - len(active))
        forced = [script[rid][len(generated[rid])] if rid >= 0 else 0
                  for rid in padded]
        counts = [len(generated[rid]) if rid >= 0 else 0 for rid in padded]
        _, tokens = runner.step(
            model, cache, generated_counts=counts, min_new_tokens=0,
            return_next_tokens=True, forced_tokens=forced, stop_on_eos=True)
        for row, request_id in enumerate(active):
            generated[request_id].append(int(tokens[row]))

    scheduler.prepare_step()
    assert generated == {1: [11, 12, 13, 127], 3: [21, 127], 0: [31, 32, 33]}
    assert admitted
    assert scheduler.active_req_ids == []
    assert cache.request_phases[1] == RequestPhase.FINISHED
    assert cache.request_phases[3] == RequestPhase.FINISHED
    assert cache.request_phases[0] == RequestPhase.FINISHED
    assert not cache.states


@torch.inference_mode()
def test_eager_request_admission_invalidates_protected_only_plan():
    model = _model()
    cache, first_tokens = _prefill_cache(model, [0])
    scheduler = ContinuousBatchScheduler(
        cache=cache, max_batch=cache.max_requests,
        eos_ids={127}, pad_token_id=0)
    scheduler.add_request(0)
    scheduler.next_tokens[0] = first_tokens[0]
    runner = GraphRunner(scheduler=scheduler, buckets=(1,), device='cuda')

    # 请求 0 两层均 both-skip，建立 protected_only 的 eager 快速重放计划。
    runner.step(
        model, cache, generated_counts=[0], min_new_tokens=1,
        return_next_tokens=True, stop_on_eos=False)
    scheduler.prepare_step()
    scheduler.finish_request(0)

    # 保持 B1/bucket/Q 几何不变，但换成完全不 skip 的请求 1。
    cache.add_request(1)
    cache.set_active_request(1)
    output = model(
        _input_ids(1), past_key_values=cache, use_cache=True,
        logits_to_keep=1)
    cache.freeze(1, [False] * 2, [False] * 2, {0: [], 1: []})
    scheduler.add_request(1)
    scheduler.next_tokens[1] = int(output.logits[0, -1].argmax())
    cache.transition_request(1, RequestPhase.GENERATE)
    runner.step(
        model, cache, generated_counts=[0], min_new_tokens=1,
        return_next_tokens=True, stop_on_eos=False)

    protected_width = GEOMETRY.sink + GEOMETRY.window
    # selector 输出按本步 batch 行写入；request 1 是当前唯一请求，位于第 0 行。
    assert torch.all(
        cache.visible_lens_buf[:, 0] > protected_width
    ), '请求切换后不得重放上一请求的 both-skip EMPTY 计划'


def test_capture_reuses_fixed_bucket_graph():
    cache = GraphReadyPagedRegionCache(
        1, GEOMETRY, max_requests=1, max_physical_pages=32, max_blocks=16,
        max_visible_len=64, max_query_tokens=1, num_kv_heads=2,
        num_query_heads=2, head_dim_k=64, head_dim_v=64,
        dtype=torch.bfloat16, device='cuda')
    runner = GraphRunner(
        scheduler=None, buckets=(1,), visible_buckets=(64,), device='cuda')
    calls = []

    def model(input_ids, position_ids, **kwargs):
        calls.append(1)
        return input_ids * 2 + position_ids

    first = runner.capture(model, cache, 1, 64)
    second = runner.capture(model, cache, 1, 64)
    assert first is second
    assert len(calls) == 2
