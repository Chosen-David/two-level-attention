import ast
import copy
from pathlib import Path
import unittest

from benchmark.LongBench.run_cap_manifest import (
    INPUT_IDENTITY_SCHEMA,
    validate_input_identity_uniformity,
)


def meta(dataset="qasper", source="v2-" + "a" * 64,
         prompt="b" * 64, tokens="c" * 64, schema=INPUT_IDENTITY_SCHEMA):
    return {
        "dataset": dataset,
        "input_identity_schema": schema,
        "source_fingerprint": source if schema else None,
        "prompt_template_sha256": prompt if schema else None,
        "actual_input_ids_ordered_sha256": tokens if schema else None,
    }


class Lbv1InputUniformityTests(unittest.TestCase):
    def test_eval_wires_gate_before_result_publication(self):
        source = (Path(__file__).resolve().parents[1]
                  / "benchmark" / "LongBench" / "eval.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        gates = [node for node in calls if isinstance(node.func, ast.Name)
                 and node.func.id == "validate_input_identity_uniformity"]
        writes = [node for node in calls if isinstance(node.func, ast.Name)
                  and node.func.id == "open" and len(node.args) > 1
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value == "w"]
        self.assertEqual(len(gates), 1)
        self.assertTrue(writes)
        self.assertLess(gates[0].lineno, min(node.lineno for node in writes))

    def test_same_dataset_same_realized_input_is_comparable(self):
        self.assertTrue(validate_input_identity_uniformity({
            "qasper-aavg.jsonl": meta(),
            "qasper-mavg.jsonl": meta(),
        }))

    def test_different_datasets_keep_independent_identity_groups(self):
        self.assertTrue(validate_input_identity_uniformity({
            "qasper-aavg.jsonl": meta(),
            "hotpotqa-aavg.jsonl": meta(dataset="hotpotqa", tokens="d" * 64),
        }))

    def test_same_dataset_rejects_each_identity_axis_drift(self):
        for field, value in (
            ("source_fingerprint", "v2-" + "d" * 64),
            ("prompt_template_sha256", "d" * 64),
            ("actual_input_ids_ordered_sha256", "d" * 64),
        ):
            changed = meta()
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                    SystemExit, "LBV1-INPUT-UNIFORMITY"):
                validate_input_identity_uniformity({"a.jsonl": meta(),
                                                    "b.jsonl": changed})

    def test_same_dataset_rejects_v2_legacy_strict_mix(self):
        with self.assertRaisesRegex(SystemExit, "v2 与 legacy strict"):
            validate_input_identity_uniformity({"new.jsonl": meta(),
                                                "old.jsonl": meta(schema=None)})

    def test_legacy_strict_files_remain_readable(self):
        self.assertTrue(validate_input_identity_uniformity(
            {"old-a.jsonl": meta(schema=None),
             "old-b.jsonl": meta(schema=None)}))

    def test_missing_v2_actual_tokens_rejected_for_comparison(self):
        incomplete = copy.deepcopy(meta())
        incomplete["actual_input_ids_ordered_sha256"] = None
        with self.assertRaisesRegex(SystemExit, "缺 source/template/actual-token"):
            validate_input_identity_uniformity({"a.jsonl": incomplete,
                                                "b.jsonl": incomplete})


if __name__ == "__main__":
    unittest.main()
