"""Exercise the actual E152 functions with deterministic clock/device doubles.

Only heavyweight imports are excluded via AST, as in the LongBench capture tests.
These tests validate host measurement/accounting, not CUDA or model performance.
"""
import argparse
import ast
import builtins
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exp/bench/e152_e2e_latency.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = ast.Module(
    body=[n for n in TREE.body if isinstance(n, ast.FunctionDef)],
    type_ignores=[],
)
ARGUMENTS = runpy.run_path(str(ROOT / "sparse_attn/arguments.py"))


class Clock:
    def __init__(self):
        self.ms = 0

    def tick(self, ms):
        self.ms += ms

    def __call__(self):
        return self.ms / 1000


class Tensor:
    def __init__(self, clock, ids=None):
        self.clock = clock
        self.ids = ids
        self.shape = (1, len(ids[0])) if ids is not None else (1, 1)

    def __getitem__(self, key):
        return self

    def argmax(self, dim):
        self.clock.tick(7)
        return self

    def unsqueeze(self, dim):
        self.clock.tick(2)
        return self

    def numpy(self):
        return SimpleNamespace(tobytes=lambda: struct.pack(
            "<" + "q" * len(self.ids[0]), *self.ids[0]))


class Device:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    def reset_peak_memory_stats(self):
        self.calls.append("reset")

    def memory_allocated(self):
        return 100

    def max_memory_allocated(self):
        return 100 + 2**30

    def synchronize(self):
        self.calls.append("sync")
        self.clock.tick(1)


class Model:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.clock.tick(10)
        return SimpleNamespace(logits=Tensor(self.clock), past_key_values=object())


def namespace(clock):
    device = Device(clock)
    torch = SimpleNamespace(
        cuda=device, long="long", no_grad=contextlib.nullcontext,
        tensor=lambda ids, **kwargs: Tensor(clock, ids),
    )
    ns = {"torch": torch, "time": SimpleNamespace(perf_counter=clock),
          "argparse": argparse, "os": os, "json": json, "hashlib": hashlib,
          "gc": gc, "REPO": str(ROOT), "QWEN3_NATIVE_MPE": 40960,
          "YARN_AUTO": {65536: 2.0, 131072: 4.0},
          "add_sparse_attn_args": ARGUMENTS["add_sparse_attn_args"]}
    exec(compile(FUNCTIONS, str(SOURCE), "exec"), ns)
    return ns, device


class RequestTiming(unittest.TestCase):
    def test_continuous_wall_includes_first_output_and_step_gaps(self):
        clock = Clock()
        ns, device = namespace(clock)
        model = Model(clock)
        rec = ns["timed_request"](model, [11, 22, 33], 2, "fake")
        # Forward+sync=11; first argmax+shape+sync=10; decode includes its
        # pre-step sync (1), forward (10), token handling (9), final sync (1).
        self.assertAlmostEqual(rec["request_wall_ms"], 42)
        self.assertAlmostEqual(rec["prefill_ms"], 11)
        self.assertAlmostEqual(rec["ttft_ms"], 21)
        self.assertAlmostEqual(rec["decode_after_first_wall_ms"], 21)
        self.assertAlmostEqual(rec["tpot_ms"], 21)
        self.assertAlmostEqual(rec["decode_ms"], 20)  # diagnostic step sum
        self.assertAlmostEqual(rec["per_token_wall_ms"][0], 21)
        self.assertGreater(rec["request_wall_ms"], rec["prefill_ms"] + rec["decode_ms"])
        self.assertEqual(rec["n_generated"], 2)
        self.assertEqual(rec["n_input_tokens"], 3)
        self.assertEqual(rec["peak_mem_delta_gib"], 1)
        self.assertNotIn("past_key_values", model.calls[0])
        self.assertIsNotNone(model.calls[1]["past_key_values"])

    def test_single_token_has_ttft_but_no_tpot(self):
        clock = Clock()
        ns, _ = namespace(clock)
        model = Model(clock)
        rec = ns["timed_request"](model, [11], 1, "fake")
        self.assertAlmostEqual(rec["request_wall_ms"], 21)
        self.assertAlmostEqual(rec["ttft_ms"], 21)
        self.assertEqual(rec["decode_after_first_wall_ms"], 0)
        self.assertEqual(rec["per_token_wall_ms"], [])
        self.assertIsNone(rec["tpot_ms"])
        self.assertEqual(len(model.calls), 1)

    def test_fixed_generation_count_and_timestamp_identity(self):
        for count in (2, 4, 128):
            with self.subTest(count=count):
                clock = Clock()
                ns, _ = namespace(clock)
                model = Model(clock)
                rec = ns["timed_request"](model, [11, 22], count, "fake")
                self.assertEqual(len(model.calls), count)
                self.assertEqual(len(rec["per_token_wall_ms"]), count - 1)
                self.assertAlmostEqual(rec["request_wall_ms"], 21 * count)
                self.assertAlmostEqual(rec["request_wall_ms"],
                                       rec["ttft_ms"] + rec["decode_after_first_wall_ms"])
                self.assertAlmostEqual(rec["request_wall_ms"], 1000 * (
                    rec["last_output_timestamp"] - rec["request_start_timestamp"]))
                self.assertAlmostEqual(rec["decode_after_first_wall_ms"],
                                       sum(rec["per_token_wall_ms"]))

    def test_bad_generation_count_rejected_before_device_work(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count):
                ns, device = namespace(Clock())
                with self.assertRaisesRegex(ValueError, "max_gen"):
                    ns["timed_request"](None, [11], count, "fake")
                self.assertEqual(device.calls, [])

    def test_model_failure_does_not_return_a_record(self):
        ns, _ = namespace(Clock())
        def fail(**kwargs):
            raise RuntimeError("forward failed")
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            ns["timed_request"](fail, [11], 2, "fake")

    def test_cli_rejects_wrong_protocol_before_loader_even_under_O(self):
        base = ["e152", "--method", "tli", "-L", "64", "--out", "unused.json"]
        cases = [([], {"TLI_SPARSE_PREFILL": "1"}),
                 ([], {"TLI_SPARSE_PREFILL": "invalid"}),
                 (["--max-gen", "0"], {}), (["--repeats", "0"], {}),
                 (["--warmup", "-1"], {}), (["-L", "0"], {}),
                 (["--method", "tia"], {}), (["--method", "sparq_vendor"], {})]
        for extra, env in cases:
            with self.subTest(extra=extra, env=env):
                ns, _ = namespace(Clock())
                with patch.object(sys, "argv", base + extra), \
                     patch.dict(os.environ, {"TLI_SPARSE_PREFILL": "0", **env}), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        ns["main"]()
                self.assertEqual(raised.exception.code, 2)

    def test_missing_method_is_not_silently_dense(self):
        ns, _ = namespace(Clock())
        with patch.object(sys, "argv", ["e152", "-L", "64", "--out", "unused.json"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                ns["main"]()
        self.assertEqual(raised.exception.code, 2)

    def test_main_serializes_v2_and_preserves_old_records_single_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.json"
            old = {"schema": "e152-e2e-latency-v1", "runs": []}
            path.write_text(json.dumps([old]))
            for method in ("none", "tli"):
                clock = Clock()
                ns, device = namespace(clock)
                model = Model(clock)
                model.parameters = lambda: iter([SimpleNamespace(device="fake")])
                model.eval = lambda: model
                config = SimpleNamespace(num_hidden_layers=4, num_key_value_heads=1,
                                         head_dim=8)
                model.config = config
                loaders = SimpleNamespace(
                    AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
                )
                yarn = SimpleNamespace(resolve_yarn_config=lambda *a: (None, None))
                real_import = builtins.__import__
                def local_import(name, *args, **kwargs):
                    if name == "transformers":
                        return loaders
                    if name == "benchmark.RULER.yarn_receipt":
                        return yarn
                    return real_import(name, *args, **kwargs)
                ns["__builtins__"] = {**vars(builtins), "__import__": local_import}
                # Recompile so these functions bind the local builtins, without
                # replacing sys.modules or mutating imports for other test suites.
                exec(compile(FUNCTIONS, str(SOURCE), "exec"), ns)
                ns.update(
                    build_prompt_tokens=lambda *a: [11, 22, 33],
                    register_patch=lambda *a: 1 if method == "tli" else 0,
                    resolve_treatment_snapshot=lambda args: SimpleNamespace(
                        manifest_json='{"test_budget": 7}'),
                    subprocess=SimpleNamespace(check_output=lambda *a, **kw: "baseline-sha\n"),
                    gc=SimpleNamespace(collect=lambda: None),
                )
                device.empty_cache = lambda: None
                device.get_device_name = lambda index: "fake-device-not-hardware"
                ns["torch"].bfloat16 = "bf16"
                ns["torch"].__version__ = "test-double"
                args = ["e152", "--method", method, "-L", "64", "--out", str(path),
                        "--max-gen", "1", "--warmup", "1", "--repeats", "1"]
                stream = io.StringIO()
                with patch.object(sys, "argv", args), \
                     patch.dict(os.environ, {"TLI_SPARSE_PREFILL": "0"}), \
                     contextlib.redirect_stdout(stream):
                    ns["main"]()
                self.assertIn("tpot=N/A", stream.getvalue())
                records = json.loads(path.read_text())
                self.assertEqual(records[0], old)
                record = records[-1]
                self.assertEqual(record["schema"], "e152-e2e-latency-v2")
                self.assertFalse(record["caliber"]["backend_verified"])
                self.assertEqual(record["caliber"]["instrumentation_mode"],
                                 "per-step-device-synchronized")
                self.assertEqual(record["caliber"]["treatment_manifest"],
                                 {"test_budget": 7} if method == "tli" else None)
                self.assertEqual([r["phase"] for r in record["runs"]], ["warmup", "measured"])
                self.assertTrue(all(r["tpot_ms"] is None for r in record["runs"]))
                self.assertTrue(all(r["request_wall_ms"] == r["ttft_ms"]
                                    for r in record["runs"]))
            self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
