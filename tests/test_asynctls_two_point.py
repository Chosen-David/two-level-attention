"""AsyncTLS 干净两点入口的预算、Graph 复用与真实调用计数门禁。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import benchmark.efficiency.benchmark_asynctls_two_point as harness
from benchmark.efficiency.benchmark_asynctls_two_point import (
    AsyncTLSTwoPointEngine,
    validate_aligned_budget,
    validate_request_receipt,
)


def _args(batch_size=1):
    return SimpleNamespace(
        device="cpu", batch_size=batch_size, max_new_tokens=33,
        warmup_requests=1, execution="graph",
        asynctls_channel_file="unused",
        asynctls_budget=1024, asynctls_block_size=64, asynctls_top_kb=128,
    )


def _receipt(tokens=33, *, batch_size=1, layers=36, capture=False):
    steps = tokens - 1
    return {
        "batch_size": batch_size,
        "generated_tokens": tokens * batch_size,
        "generated_tokens_per_request": [tokens] * batch_size,
        "prefill_calls": batch_size * layers,
        "python_decode_calls_warmup_and_capture": 2 * layers if capture else 0,
        "graph_capture_count": int(capture and steps > 0),
        "graph_replays": steps,
        "graph_eager_fallbacks": 0,
    }


@pytest.mark.parametrize("tokens", [1, 33])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_receipt_accepts_fixed_rows_and_real_prefill(tokens, batch_size):
    validate_request_receipt(
        36, tokens, _receipt(
            tokens, batch_size=batch_size, capture=tokens > 1),
        allow_initial_capture=True,
    )


def test_receipt_rejects_replay_masquerading_as_python_backend_calls():
    receipt = _receipt()
    receipt["python_decode_calls_warmup_and_capture"] = 32 * 36
    with pytest.raises(RuntimeError, match="Graph replay 不得冒充"):
        validate_request_receipt(36, 33, receipt)


def test_receipt_rejects_missing_serial_prefill_rows():
    receipt = _receipt(batch_size=4)
    receipt["prefill_calls"] = 36
    with pytest.raises(RuntimeError, match="逐请求覆盖全部层"):
        validate_request_receipt(36, 33, receipt)


def test_only_k2_1024_block64_k1_128_is_accepted():
    validate_aligned_budget(_args())
    args = _args()
    args.asynctls_budget = 512
    with pytest.raises(ValueError, match="1024, 64, 128"):
        validate_aligned_budget(args)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fixed_arena_captures_once_and_reuses_graph_across_requests(
        monkeypatch, batch_size):
    layers = 2
    runtime_instances = []

    class FakeRuntime:
        def __init__(self, model, batch, length, max_new_tokens, **kwargs):
            self.num_layers = layers
            self.mask_metadata = object()
            self.prefill_calls = 0
            self.decode_calls = 0
            self.initial_blocks = None
            self.decode_step_resets = []
            runtime_instances.append(self)

        def receipt(self):
            return {
                "implementation": "fake semantic runtime",
                "token_budget": 1024,
                "block_size": 64,
                "top_kb": 128,
                "prefill_calls": self.prefill_calls,
                "decode_calls_warmup_and_capture": self.decode_calls,
            }

        def prepare_decode(self, execution, decode_steps):
            assert execution == "graph"
            assert self.prefill_calls == batch_size * layers
            self.initial_blocks = object()

        def prepare_decode_step(self, step, reset_pipeline=False):
            self.decode_step_resets.append((step, reset_pipeline))

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(num_hidden_layers=layers)
            self.runtime = None

        def __call__(self, input_ids, **kwargs):
            batch = input_ids.shape[0]
            logits = torch.zeros((batch, 1, 128), dtype=torch.float32)
            if input_ids.shape[1] != 1:
                self.runtime.prefill_calls += layers
                logits[0, 0, 7] = 1.0
            else:
                self.runtime.decode_calls += layers
                for row in range(batch):
                    logits[row, 0, 8 + row] = 1.0
            return SimpleNamespace(logits=logits)

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def replay(self):
            # 真实 Graph replay 不调用 Python model/runtime wrapper。
            return None

    model = FakeModel()
    monkeypatch.setattr(harness, "AsyncTLSQwen3GraphRuntime", FakeRuntime)
    monkeypatch.setattr(
        harness, "bind_asynctls_graph_runtime",
        lambda model, runtime: setattr(model, "runtime", runtime))
    monkeypatch.setattr(harness.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(harness.torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(harness.torch.cuda, "graph", lambda graph: graph)

    engine = AsyncTLSTwoPointEngine(_args(batch_size), model, None, 16)
    for max_new_tokens in (1, 33, 1, 33):
        generated, _ = engine.generate(list(range(16)), max_new_tokens, True)
        rows = generated if batch_size > 1 else [generated]
        assert len(rows) == batch_size
        assert all(len(row) == max_new_tokens for row in rows)

    fields = (
        "generated_tokens", "prefill_calls",
        "python_decode_calls_warmup_and_capture",
        "graph_capture_count", "graph_replays", "graph_eager_fallbacks",
    )
    actual = [tuple(receipt[field] for field in fields)
              for receipt in engine.request_receipts]
    expected = [
        (batch_size, 2 * batch_size, 0, 0, 0, 0),
        (33 * batch_size, 2 * batch_size, 2 * layers, 1, 32, 0),
        (batch_size, 2 * batch_size, 0, 0, 0, 0),
        (33 * batch_size, 2 * batch_size, 0, 0, 32, 0),
    ]
    assert actual == expected
    assert len(runtime_instances) == 1
    assert engine.graph_capture_count == 1
    assert engine.graph_replay_count == 64
    assert engine.runtime["graph_capture_count"] == 1
    assert engine.runtime["graph_replay_count"] == 64
    decode_resets = runtime_instances[0].decode_step_resets
    assert sum(step == 0 and reset for step, reset in decode_resets) >= 2


def test_clean_harness_has_no_per_step_events_or_memory_profiler_calls():
    path = Path(__file__).parents[1] / (
        "benchmark/efficiency/benchmark_asynctls_two_point.py")
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
        "torch.cuda.profiler.start",
    }
    unexpected = forbidden & called
    assert not unexpected, f"干净计时入口包含禁用调用: {sorted(unexpected)}"
