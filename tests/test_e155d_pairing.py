"""Fixed-quota CI identity gates; no model/scorer dependencies required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "exp/trace/analyze_e155d_grid.py"
spec = importlib.util.spec_from_file_location("e155d_pairing", SCRIPT)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def cells(values):
    return {task: dict(values) for task in grid.AVG5_TASKS}


class PairingTests(unittest.TestCase):
    def test_partial_overlap_is_rejected_not_silently_intersected(self):
        # Full means: A=0.25, B=0.5; intersection falsely says A wins by 0.5.
        with self.assertRaisesRegex(ValueError, "hotpotqa.*ID"):
            grid.paired_bootstrap(cells({"shared": .5, "a": 0}),
                                  cells({"shared": 0, "b": 1}), B=40)

    def test_disjoint_sets_rejected_with_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "hotpotqa.*ID"):
            grid.paired_bootstrap(cells({"a": 0}), cells({"b": 1}), B=40)

    def test_empty_sets_rejected_with_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "hotpotqa.*ID"):
            grid.paired_bootstrap(cells({}), cells({}), B=40)

    def test_single_task_mismatch_rejected(self):
        a, b = cells({"a": .5}), cells({"a": 0})
        b["qasper"]["b"] = 1
        with self.assertRaisesRegex(ValueError, "qasper.*ID"):
            grid.paired_bootstrap(a, b, B=40)

    def test_matching_ids_preserve_seeded_result_and_order_invariance(self):
        a = cells({"c": .1, "a": .9, "b": .4})
        b = cells({"a": .2, "b": .3, "c": .8})
        result = grid.paired_bootstrap(a, b, B=40)
        self.assertEqual(result, grid.paired_bootstrap(
            {t: dict(reversed(list(v.items()))) for t, v in a.items()}, b, B=40))
        self.assertEqual(result["aligned_n"], {t: 3 for t in grid.AVG5_TASKS})
        self.assertEqual(result["point_diff"], 3.3333)

    def read_rows(self, rows):
        fake_eval = types.ModuleType("benchmark.LongBench.eval")
        fake_eval.sample_score = lambda task, pred, answers, classes: pred
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotpotqa-run.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with patch.dict(sys.modules, {"benchmark.LongBench.eval": fake_eval}):
                return grid.per_example_scores({"files": {"hotpotqa": str(path)}})

    def test_missing_id_rejected_instead_of_row_number(self):
        with self.assertRaisesRegex(ValueError, "hotpotqa.*_id"):
            self.read_rows([{"pred": .5, "answers": ["yes"]}])

    def test_duplicate_id_rejected_instead_of_overwrite(self):
        with self.assertRaisesRegex(ValueError, "hotpotqa.*duplicate"):
            self.read_rows([{"_id": "a", "pred": p, "answers": ["yes"]}
                            for p in (.1, .9)])

    def test_null_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "hotpotqa.*_id"):
            self.read_rows([{"_id": None, "pred": .5, "answers": ["yes"]}])

    def test_malformed_ids_rejected(self):
        for value in (True, False, .5, [], {}, ""):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "_id"):
                self.read_rows([{"_id": value, "pred": .5, "answers": ["yes"]}])

    def test_integer_and_string_ids_remain_distinct(self):
        rows = [{"_id": key, "pred": value, "answers": []}
                for key, value in ((0, .2), ("0", .8))]
        self.assertEqual(self.read_rows(rows), {"hotpotqa": {0: .2, "0": .8}})
        result = grid.paired_bootstrap(cells({0: .2, "0": .8}),
                                      cells({"0": .8, 0: .2}), B=40)
        self.assertEqual(result["ci95"], [0, 0])

    def test_valid_row_order_does_not_change_identity(self):
        rows = [{"_id": key, "pred": value, "answers": ["yes"]}
                for key, value in (("b", .2), ("a", .8))]
        self.assertEqual(self.read_rows(rows), self.read_rows(rows[::-1]))

    def test_main_mismatch_stops_before_verdict_publication(self):
        tags = ["pred_nb8nt320nl8192", "pred_nb4nt128nl1024"]
        complete = {tag: {} for tag in tags}
        scores = {tag: {"avg5": score} for tag, score in zip(tags, (50, 25))}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "verdict.json"
            with patch.object(grid, "collect_cells", return_value=(complete, {}, [], [])), \
                 patch.object(grid, "stage_and_score", return_value=scores), \
                 patch.object(grid, "per_example_scores", side_effect=[
                     cells({"shared": .5, "a": 0}), cells({"shared": 0, "b": 1})]), \
                 patch.object(grid, "OUT", str(out)), \
                 patch.object(grid.sys, "argv", ["grid", "--bootstrap", "40"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "identical ID sets"):
                    grid.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
