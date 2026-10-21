import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.LongBench.run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX,
    build_run_cap_core,
    finalize_run_cap_manifest,
)


def load_consumer():
    """Execute unchanged consumer functions without optional scorer imports."""
    path = Path(__file__).resolve().parents[1] / "benchmark/LongBench/eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_runcap_file_sha256", "_runcap_file_lines",
             "validate_runcap_sidecar"}
    tree.body = [node for node in tree.body if (
        isinstance(node, ast.Import) and all(
            alias.name in {"os", "hashlib", "sys"} for alias in node.names)
    ) or (
        isinstance(node, ast.ImportFrom) and node.module == "run_cap_manifest"
    ) or (isinstance(node, ast.FunctionDef) and node.name in names)]
    namespace = {"__package__": "benchmark.LongBench"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["validate_runcap_sidecar"]


class DatasetBindingTests(unittest.TestCase):
    def setUp(self):
        self.consume = load_consumer()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_result(self, dataset, prefix):
        path = Path(self.temp.name) / f"{prefix}-tli-test.jsonl"
        raw = b'{"pred":"a","answers":["a"],"all_classes":null}\n'
        path.write_bytes(raw)
        core = build_run_cap_core(
            model_name="Qwen3-8B", model_path="/model", method="tli",
            dataset=dataset, dataset_fingerprint="a" * 12,
            run_timestamp="dataset-binding-test", effective_cap=31500,
            cap_gate_info=None, max_new_tokens=128, producer=None,
        )
        manifest = finalize_run_cap_manifest(
            core, status="complete", n_samples=1, input_tokens=[4],
            truncated_flags=[False], result_file=path.name,
            result_sha256=hashlib.sha256(raw).hexdigest(), result_lines=1,
        )
        Path(str(path) + RUN_CAP_SIDECAR_SUFFIX).write_text(
            json.dumps(manifest), encoding="utf-8")
        return str(path)

    def test_mismatched_dataset_rejected_even_when_legacy_allowed(self):
        for allowed in (False, True):
            with self.subTest(legacy_allowed=allowed):
                path = self.make_result("qasper", "hotpotqa")
                with self.assertRaisesRegex(SystemExit, "DATASET-BINDING"):
                    self.consume(path, allowed)

    def test_same_dataset_is_accepted(self):
        path = self.make_result("qasper", "qasper")
        self.assertEqual(self.consume(path, False)["dataset"], "qasper")

    def test_producer_repobench_prefix_is_accepted(self):
        path = self.make_result("repobench-p", "repobench")
        self.assertEqual(self.consume(path, False)["dataset"], "repobench-p")

    def test_precomputed_snapshot_cannot_bypass_dataset_check(self):
        path = self.make_result("gov_report", "qasper")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with self.assertRaisesRegex(SystemExit, "DATASET-BINDING"):
            self.consume(path, False, actual_sha=digest, actual_lines=1)


if __name__ == "__main__":
    unittest.main()
