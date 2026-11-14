"""NFA/FlashInfer 长上下文两点法入口的协议与真实调用计数门禁。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import benchmark.efficiency.benchmark_long_context_two_point as harness
from benchmark.efficiency.benchmark_long_context_two_point import (
    RequestTwoPointEngine,
    validate_request_receipt,
)


def _nfa_receipt(tokens=33, *, initial_capture=False):
    steps = tokens - 1
    return {
        "generated_tokens": tokens,
        "nfa_decode_steps": steps,
        "graph_capture_count": int(steps > 0 and initial_capture),
        "graph_replays": steps,
        "graph_eager_fallbacks": 0,
    }


def _flashinfer_receipt(tokens=33, execution="graph"):
    steps = tokens - 1
    decode_calls = 0 if steps == 0 else (
        72 if execution == "graph" else 36 * steps)
    return {
        "generated_tokens": tokens,
        "prefill_append_calls": 36,
        "python_decode_append_calls": decode_calls,
        "python_decode_wrapper_calls": decode_calls,
        "graph_capture_count": int(execution == "graph" and steps > 0),
        "graph_replays": steps if execution == "graph" else 0,
        "graph_eager_fallbacks": 0,
    }


@pytest.mark.parametrize("tokens", [1, 33])
def test_nfa_graph_receipt_accepts_zero_decode_and_fixed_decode(tokens):
    validate_request_receipt("nfa", "graph", 36, tokens, _nfa_receipt(tokens))


def test_nfa_graph_receipt_allows_only_initial_cold_capture():
    validate_request_receipt(
        "nfa", "graph", 36, 33, _nfa_receipt(initial_capture=True),
        allow_initial_nfa_capture=True)
    with pytest.raises(RuntimeError, match="Graph 计数"):
        validate_request_receipt(
            "nfa", "graph", 36, 33, _nfa_receipt(initial_capture=True))


def test_nfa_eager_receipt_requires_real_steps_without_graph():
    receipt = _nfa_receipt()
    receipt.update(graph_capture_count=0, graph_replays=0,
                   graph_eager_fallbacks=32)
    validate_request_receipt("nfa", "eager", 36, 33, receipt)


def test_nfa_graph_fallback_is_fail_closed():
    receipt = _nfa_receipt()
    receipt["graph_eager_fallbacks"] = 1
    with pytest.raises(RuntimeError, match="Graph 计数"):
        validate_request_receipt("nfa", "graph", 36, 33, receipt)


def test_nfa_two_point_pairs_reuse_fixed_cache_and_captured_graph(monkeypatch):
    """T(1)→T(33)→T(1)→T(33) 只允许首个 decode 请求冷 capture。"""

    class FakeCache:
        def __init__(self):
            self.states = {}
            self.nfa_decode_steps = 0
            self.removed = []

        def transition_request(self, request_id, phase):
            if phase != "GENERATE" or request_id not in self.states:
                raise RuntimeError("请求状态迁移无效")

        def remove_request(self, request_id):
            self.states.pop(request_id, None)
            self.removed.append(request_id)

    class FakeScheduler:
        def __init__(self, cache, **kwargs):
            self.cache = cache
            self.next_tokens = {}

        def add_request(self, request_id):
            self.request_id = request_id

        def prepare_step(self):
            self.prepared = True

    class FakeRunner:
        def __init__(self, scheduler, capture):
            self.scheduler = scheduler
            self.capture = capture
            self.capture_count = int(capture)
            self.replay_count = 0
            self.eager_fallback_count = 0

        def step(self, model, cache, **kwargs):
            if self.scheduler is None:
                raise RuntimeError("请求结束后 scheduler 未释放")
            cache.nfa_decode_steps += 1
            if self.capture:
                self.replay_count += 1
            else:
                self.eager_fallback_count += 1
            token = 8 + self.replay_count + self.eager_fallback_count
            return None, torch.tensor([token], dtype=torch.long)

    class FakePool:
        def __init__(self):
            self.runner = None
            self.hit_count = 0
            self.miss_count = 0
            self.warmup_seconds = 0.0
            self.releases = 0

        def acquire(self, scheduler, capture):
            if self.runner is None:
                self.miss_count += 1
                self.warmup_seconds += 0.25
                self.runner = FakeRunner(scheduler, capture)
            else:
                self.hit_count += 1
                self.runner.scheduler = scheduler
            return self.runner

        def release(self, runner):
            runner.scheduler = None
            self.releases += 1

    cache = FakeCache()
    pool = FakePool()
    build_calls = []
    selector = SimpleNamespace(
        nfa_sink_tokens=4, nfa_near_len=4096, nfa_swa_size=128,
        tia_block_size=64, nfa_subspace="tail", tia_level2_cmp_ratio=4,
        tia_level1_topk=128, tia_level2_topk=1024, nfa_near_pages=32,
        nfa_near_tokens="off", nfa_enable_layer_skip=False)

    def fake_build(model, geometry, prompt_length, **kwargs):
        build_calls.append((geometry, prompt_length, kwargs))
        return cache

    def fake_prefill(model, ids, gate, geometry, *, cache, request_id):
        cache.states[request_id] = object()
        output = SimpleNamespace(logits=torch.zeros((1, 1, 16)))
        return output, {"request_id": request_id}

    monkeypatch.setattr(harness, "nfa_args", lambda: selector)
    monkeypatch.setattr(harness, "build_frozen_cache", fake_build)
    monkeypatch.setattr(harness, "prefill_frozen_request", fake_prefill)
    monkeypatch.setattr(harness, "ContinuousBatchScheduler", FakeScheduler)
    monkeypatch.setattr(
        harness, "greedy_next_tokens",
        lambda *args, **kwargs: torch.tensor([7], dtype=torch.long))
    monkeypatch.setattr(
        harness, "session_graph_runner",
        lambda model, cache, scheduler, **kwargs: pool.acquire(
            scheduler, kwargs["capture"]))
    monkeypatch.setattr(harness.torch.cuda, "synchronize", lambda: None)

    args = SimpleNamespace(
        method="nfa", execution="graph", device="cpu", visible_len=8192,
        max_new_tokens=33, warmup_requests=1)
    model = SimpleNamespace(
        dtype=torch.bfloat16,
        config=SimpleNamespace(num_hidden_layers=36),
        _nfa_graph_session_pool=pool)
    engine = RequestTwoPointEngine(args, model, None, ())
    for max_new_tokens in (1, 33, 1, 33):
        generated, _ = engine.generate(list(range(16)), max_new_tokens, True)
        if len(generated) != max_new_tokens:
            raise AssertionError("固定长度生成在主机回归中不完整")

    fields = (
        "generated_tokens", "graph_capture_count", "graph_replays",
        "graph_eager_fallbacks", "graph_pool_hits", "graph_pool_misses",
        "nfa_decode_steps")
    actual = [tuple(receipt[field] for field in fields)
              for receipt in engine.request_receipts]
    expected = [
        (1, 0, 0, 0, 0, 0, 0),
        (33, 1, 32, 0, 0, 1, 32),
        (1, 0, 0, 0, 0, 0, 0),
        (33, 0, 32, 0, 1, 0, 32),
    ]
    if actual != expected:
        raise AssertionError(f"请求级 Graph 增量不正确: {actual} != {expected}")
    prepare_ms = [receipt["graph_prepare_ms"] for receipt in engine.request_receipts]
    if prepare_ms != [0.0, 250.0, 0.0, 0.0]:
        raise AssertionError(f"Graph prepare 增量不正确: {prepare_ms}")
    if len(build_calls) != 1 or cache.states or cache.removed != [0, 0, 0, 0]:
        raise AssertionError("固定 cache arena 构建或请求清理不正确")
    if pool.releases != 2 or pool.runner.scheduler is not None:
        raise AssertionError("decode 请求边界未释放 scheduler")
    if engine._nfa_decode_requests != 2:
        raise AssertionError("冷 capture 资格没有按真实 decode 请求推进")


@pytest.mark.parametrize("execution", ["eager", "graph"])
@pytest.mark.parametrize("tokens", [1, 33])
def test_flashinfer_receipt_covers_t1_and_t33(execution, tokens):
    validate_request_receipt(
        "flashinfer", execution, 36, tokens,
        _flashinfer_receipt(tokens, execution))


def test_flashinfer_missing_real_backend_calls_is_fail_closed():
    receipt = _flashinfer_receipt()
    receipt["python_decode_wrapper_calls"] -= 1
    with pytest.raises(RuntimeError, match="decode API 计数"):
        validate_request_receipt("flashinfer", "graph", 36, 33, receipt)


def test_fixed_length_mismatch_is_fail_closed():
    receipt = _nfa_receipt()
    receipt["generated_tokens"] = 32
    with pytest.raises(RuntimeError, match="固定长度"):
        validate_request_receipt("nfa", "graph", 36, 33, receipt)


def test_clean_harness_has_no_per_step_events_or_memory_profiler_calls():
    path = Path(__file__).parents[1] / (
        "benchmark/efficiency/benchmark_long_context_two_point.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            name = ast.unparse(node.func)
        except Exception:
            continue
        called.add(name)
    forbidden = {
        "torch.cuda.Event",
        "torch.cuda.memory_allocated",
        "torch.cuda.memory_reserved",
        "torch.cuda.reset_peak_memory_stats",
    }
    unexpected = forbidden & called
    if unexpected:
        raise AssertionError(f"干净计时入口包含禁用调用: {sorted(unexpected)}")
