"""CPU process-control tests, NOT selector/kernel correctness or GPU benchmarks.

Run the unmodified top-level validation scripts with dependency stubs. No CUDA
allocation is made; faults are injected at specific selector calls. Both normal
Python and -O must abort before *any* benchmark synchronization on bad output.
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
import unittest


HERE = Path(__file__).resolve().parent
SCRIPTS = ("e139_bs_graph_bitdiff.py", "e139_step2_cand_bitdiff.py")


def probe(script, fault, fault_call):
    class Tensor:
        def __init__(self, shape, dtype="bool", value=0):
            self.shape, self.dtype, self.value = shape, dtype, value
            self.device = "cuda:0"

        def __ne__(self, other):
            return Tensor((), value=int(self.value != other.value))

        def sum(self):
            return self.value

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.bool = "bool"
    torch.bfloat16 = "bfloat16"
    torch.Generator = lambda **kw: types.SimpleNamespace(manual_seed=lambda seed: seed)
    torch.randn = lambda *shape, **kw: Tensor(shape, kw["dtype"])
    torch.equal = lambda a, b: a.shape == b.shape and a.value == b.value
    timing_started = False
    def synchronize():
        nonlocal timing_started
        timing_started = True
        print("BENCH_ENTERED")
    torch.cuda = types.SimpleNamespace(synchronize=synchronize)
    sys.modules["torch"] = torch

    def module(name):
        result = types.ModuleType(name)
        sys.modules[name] = result
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, result)
        return result

    module("e131_arm_bench").make_args = lambda _: types.SimpleNamespace()
    module("sparse_attn")
    module("sparse_attn.indexer")
    indexer = module("sparse_attn.indexer.tli_indexer")
    indexer.TLIIndexer = lambda args: types.SimpleNamespace(args=args, clear=lambda: None)
    module("sparse_attn.ops")
    bs = module("sparse_attn.ops.batch_select")
    bs._GRAPH_CACHE = {}
    bs._GRAPH_STATS = dict(capture=0,replay=0,fallback=0)
    bs._gate_ok = lambda *args: fault != "ineligible"
    calls = 0

    def select(idx, q, k, scale, ends, dense, *, require_graph=False):
        nonlocal calls
        calls += 1
        # CPU process-control stub, not a GPU capture/replay proof.
        if os.environ.get('TLI_PREFILL_SELECT_GRAPH')=='1':
            event='replay' if bs._GRAPH_CACHE else 'capture'
            bs._GRAPH_STATS[event]+=1
            bs._GRAPH_CACHE['stub']=True
        rows = [Tensor((8, end)) for end in ends]
        graph_failed = ((fault == "graph_failure" and calls == fault_call)
                        or (fault == "graph_timing" and timing_started
                            and os.environ.get("TLI_PREFILL_SELECT_GRAPH") == "1"))
        if graph_failed:
            if require_graph:
                raise RuntimeError("validation failed: graph execution failed")
            return rows  # production's silent eager fallback: same mask bits
        if calls == fault_call:
            if fault == "none":
                return None
            if fault == "short":
                return rows[:-1]
            if fault == "empty":
                return []
            if fault == "extra":
                return rows + rows[-1:]
            if fault == "mismatch":
                rows[-1].value = 1
            if fault == "dtype":
                rows[-1].dtype = "float32"
            if fault == "shape":
                rows[-1].shape = (1,)
            if fault == "device":
                rows[-1].device = "cpu"
            if fault == "not_tensor":
                rows[-1] = object()
        return rows

    bs.batched_group_sels = select
    os.environ["E139_TREE"] = str(HERE.parents[1])
    runpy.run_path(str(HERE / script), run_name="__main__")


class ValidationGateTests(unittest.TestCase):
    def run_probe(self, script, fault, call, optimized):
        return subprocess.run(
            [sys.executable, *(["-O"] if optimized else []), __file__,
             "--probe", script, fault, str(call)],
            text=True, capture_output=True, timeout=20,
        )

    def test_faults_abort_before_timing_even_under_optimized_python(self):
        for optimized in (False, True):
            for script in SCRIPTS:
                # Every comparison stage, including K2=8192 (after the old
                # script had already printed K2=1024 performance numbers).
                calls = (2, 3, 5, 7, 9) if script == SCRIPTS[0] else (
                    2, 3, 4, 6, 8, 10, 11, 12, 14, 16)
                cases = [("mismatch", call) for call in calls]
                cases += [(fault, 2) for fault in (
                    "none", "short", "empty", "extra", "dtype", "shape",
                    "device", "not_tensor", "ineligible")]
                for fault, call in cases:
                    with self.subTest(script=script, optimized=optimized,
                                      fault=fault, call=call):
                        result = self.run_probe(script, fault, call, optimized)
                        self.assertNotEqual(result.returncode, 0, result.stdout)
                        self.assertNotIn("BENCH_ENTERED", result.stdout)
                        self.assertNotIn("选择链 wall:", result.stdout)
                        self.assertNotIn("=== DONE ===", result.stdout)
                        self.assertIn("validation failed", result.stderr)

    def test_equal_results_reach_timing_and_exit_zero(self):
        for optimized in (False, True):
            for script in SCRIPTS:
                with self.subTest(script=script, optimized=optimized):
                    result = self.run_probe(script, "ok", 0, optimized)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("BENCH_ENTERED", result.stdout)
                    self.assertIn("选择链 wall:", result.stdout)
                    self.assertIn("=== DONE ===", result.stdout)

    def test_graph_fallback_never_produces_performance_summary(self):
        for optimized in (False, True):
            for script in SCRIPTS:
                calls = (3, 5) if script == SCRIPTS[0] else (4, 6)
                for fault, call in [("graph_failure", c) for c in calls] + [("graph_timing", 0)]:
                    with self.subTest(script=script, fault=fault, call=call, optimized=optimized):
                        result = self.run_probe(script, fault, call, optimized)
                        self.assertNotEqual(result.returncode, 0, result.stdout)
                        self.assertIn("graph execution failed", result.stderr)
                        if fault == "graph_failure":
                            self.assertNotIn("BENCH_ENTERED", result.stdout)
                        self.assertNotIn("选择链 wall:", result.stdout)
                        self.assertNotIn("=== DONE ===", result.stdout)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        probe(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        unittest.main()
