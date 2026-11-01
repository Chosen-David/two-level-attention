"""GraphRunner ownership regressions; host doubles are not CUDA validation."""
import gc
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

SOURCE = Path(os.environ.get(
    "GRAPH_RUNNER_SOURCE",
    Path(__file__).resolve().parents[1] / "sparse_attn/engine/graph_runner.py"))


def load_runner(torch_module):
    spec = importlib.util.spec_from_file_location("_ownership_graph_runner", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch_module}):
        spec.loader.exec_module(module)
    return module.GraphRunner


class Buffer:
    def __init__(self, shape):
        self.shape = shape
        self.copies = 0

    def zero_(self):
        return self

    def to(self, device):
        return self

    def copy_(self, source, **kwargs):
        if self.shape != source.shape:
            raise ValueError("shape mismatch")
        self.copies += 1
        return self


class Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class Capture:
    def __init__(self, cuda, graph):
        self.cuda, self.graph = cuda, graph

    def __enter__(self):
        self.cuda.capturing = True

    def __exit__(self, *args):
        self.cuda.capturing = False


class CUDA:
    capturing = False
    CUDAGraph = Graph

    def synchronize(self):
        pass

    def graph(self, graph):
        return Capture(self, graph)


class Cache:
    n_layers = 1

    def __init__(self):
        self.events = []

    def new_step(self, ids):
        self.events.append(("new", tuple(ids)))

    def prepare_metadata(self, ids, layer):
        self.events.append(("metadata", tuple(ids), layer))

    def commit_step(self):
        self.events.append(("commit",))


class Scheduler:
    def __init__(self, cache, batch=1, factory=Buffer):
        self.cache = cache
        self.active_req_ids = list(range(batch))
        self.prepares = 0
        self.records = []
        self.factory = factory

    def prepare_step(self):
        self.prepares += 1
        n = len(self.active_req_ids)
        bucket = 1
        while bucket < n:
            bucket *= 2
        ids = self.active_req_ids + [-1] * (bucket - n)
        self.cache.new_step(ids)
        return ids, self.factory((bucket, 1)), self.factory((bucket, 1))

    def record_next_tokens(self, ids, logits):
        self.records.append((ids, logits))


class Model:
    def __init__(self, cuda):
        self.cuda = cuda
        self.calls = 0
        self.fail = None

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail == "warmup" or (self.fail == "capture" and self.cuda.capturing):
            raise RuntimeError("injected capture failure")
        # Distinct output lets the test detect replay returning another owner's result.
        return SimpleNamespace(owner=id(self), cache=id(kwargs["past_key_values"]))


class HostOwnership(unittest.TestCase):
    def setUp(self):
        self.cuda = CUDA()
        fake = SimpleNamespace(device=lambda x: x, long="long",
                               empty=lambda shape, **kw: Buffer(shape), cuda=self.cuda)
        self.Runner = load_runner(fake)
        self.cache = Cache()
        self.scheduler = Scheduler(self.cache)
        self.runner = self.Runner(scheduler=self.scheduler)
        self.model = Model(self.cuda)

    def assert_rejected_without_work(self, model, cache):
        events = list(cache.events)
        prepares = self.scheduler.prepares
        copies = self.runner.input_ids_buf[1].copies
        with self.assertRaises(ValueError):
            self.runner.step(model, cache)
        self.assertEqual(cache.events, events)
        self.assertEqual(self.scheduler.prepares, prepares)
        self.assertEqual(self.runner.input_ids_buf[1].copies, copies)

    def test_model_replacement_rejected(self):
        graph = self.runner.capture(self.model, self.cache, 1)
        self.assert_rejected_without_work(Model(self.cuda), self.cache)
        self.assertEqual(graph.replays, 0)

    def test_cache_and_scheduler_replacement_rejected(self):
        graph = self.runner.capture(self.model, self.cache, 1)
        replacement = Cache()
        self.scheduler.cache = replacement
        self.assert_rejected_without_work(self.model, replacement)
        self.assertEqual(graph.replays, 0)

    def test_scheduler_cache_mismatch_eager_and_capture(self):
        replacement = Cache()
        self.assert_rejected_without_work(self.model, replacement)
        with self.assertRaises(ValueError):
            self.runner.capture(self.model, replacement, 1)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(replacement.events, [])

    def test_same_owner_changed_requests_and_buckets(self):
        self.runner = self.Runner(scheduler=self.scheduler, buckets=(1, 2, 4, 8, 16, 32))
        for bucket in (1, 2, 4, 8, 16, 32):
            self.runner.capture(self.model, self.cache, bucket)
        for n in (1, 2, 3, 8, 17, 31, 32, 1):
            # Emulate EOS removal / new admission without replacing storage.
            self.scheduler.active_req_ids = list(range(20, 20 + n))
            output = self.runner.step(self.model, self.cache)
            self.assertEqual(output.owner, id(self.model))
            self.assertEqual(output.cache, id(self.cache))
        self.assertEqual(self.scheduler.prepares, 8)
        self.assertEqual(sum(e == ("commit",) for e in self.cache.events), 8)

    def test_recapture_rebinds_only_that_bucket(self):
        self.runner.capture(self.model, self.cache, 1)
        old_two = self.runner.capture(self.model, self.cache, 2)
        replacement, model = Cache(), Model(self.cuda)
        self.scheduler.cache = replacement
        self.runner.capture(model, replacement, 1)
        self.assertEqual(self.runner.step(model, replacement).cache, id(replacement))
        self.scheduler.active_req_ids = [0, 1]
        with self.assertRaises(ValueError):
            self.runner.step(model, replacement)
        self.assertEqual(old_two.replays, 0)
        self.runner.capture(model, replacement, 2)
        self.assertEqual(self.runner.step(model, replacement).owner, id(model))

    def test_failed_recapture_drops_old_graph_and_uses_eager(self):
        for failure in ("warmup", "capture"):
            with self.subTest(failure=failure):
                old = self.runner.capture(self.model, self.cache, 1)
                self.model.fail = failure
                with self.assertRaises(RuntimeError):
                    self.runner.capture(self.model, self.cache, 1)
                self.model.fail = None
                calls = self.model.calls
                self.runner.step(self.model, self.cache)
                self.assertEqual(self.model.calls, calls + 1)
                self.assertEqual(old.replays, 0)
                self.assertNotIn(1, self.runner.static_logits)

    def test_owners_live_until_bucket_replaced(self):
        self.runner.capture(self.model, self.cache, 1)
        model_ref, cache_ref = weakref.ref(self.model), weakref.ref(self.cache)
        self.model = None
        self.cache = None
        self.scheduler.cache = Cache()
        gc.collect()
        self.assertIsNotNone(model_ref())
        self.assertIsNotNone(cache_ref())
        self.runner.capture(Model(self.cuda), self.scheduler.cache, 1)
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(cache_ref())

    def test_eager_without_capture_uses_current_model(self):
        model = Model(self.cuda)
        self.assertEqual(self.runner.step(model, self.cache).owner, id(model))
        self.assertEqual(model.calls, 1)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class NativeOwnership(unittest.TestCase):
    def test_replay_and_recapture_distinct_storage(self):
        Runner = load_runner(torch)

        class DeviceCache(Cache):
            def __init__(self, value):
                super().__init__()
                self.value = torch.tensor(value, device="cuda", dtype=torch.float32)

        class DeviceModel:
            def __init__(self, scale):
                self.scale = torch.tensor(scale, device="cuda", dtype=torch.float32)

            def __call__(self, input_ids, past_key_values, **kwargs):
                return input_ids.float() * self.scale + past_key_values.value

        cache = DeviceCache(3)
        scheduler = Scheduler(cache, factory=lambda shape: torch.ones(shape, dtype=torch.long))
        runner = Runner(scheduler=scheduler, buckets=(1, 2))
        model = DeviceModel(2)
        runner.capture(model, cache, 1)
        torch.testing.assert_close(runner.step(model, cache), torch.tensor([[5.]], device="cuda"),
                                   rtol=0, atol=0)
        replacement, other = DeviceCache(11), DeviceModel(7)
        scheduler.cache = replacement
        with self.assertRaises(ValueError):
            runner.step(other, replacement)
        runner.capture(other, replacement, 1)
        torch.testing.assert_close(runner.step(other, replacement),
                                   torch.tensor([[18.]], device="cuda"), rtol=0, atol=0)
        # In-place content changes keep addresses valid across replay.
        replacement.value.fill_(13)
        torch.testing.assert_close(runner.step(other, replacement),
                                   torch.tensor([[20.]], device="cuda"), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

