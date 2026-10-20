"""E155d CI must use the same score-point scale as the AVG5 ranking."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "exp/trace/analyze_e155d_grid.py"
spec = importlib.util.spec_from_file_location("e155d_units", SCRIPT)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def cells(value):
    return {task: {"one": value} for task in grid.AVG5_TASKS}


class ScoreUnitTests(unittest.TestCase):
    def test_constant_positive_difference_in_score_points(self):
        result = grid.paired_bootstrap(cells(.75), cells(.5), B=40)
        self.assertEqual(result["point_diff"], 25)
        self.assertEqual(result["ci95"], [25, 25])
        self.assertEqual(result["unit"], "score_points")
        self.assertTrue(result["ci_excludes_zero"])

    def test_constant_negative_difference_in_score_points(self):
        result = grid.paired_bootstrap(cells(.25), cells(.75), B=40)
        self.assertEqual(result["point_diff"], -50)
        self.assertEqual(result["ci95"], [-50, -50])
        self.assertTrue(result["ci_excludes_zero"])

    def test_scale_before_rounding_preserves_small_difference(self):
        result = grid.paired_bootstrap(cells(.00004), cells(0), B=40)
        self.assertEqual(result["point_diff"], .004)
        self.assertEqual(result["ci95"], [.004, .004])

    def test_unequal_task_sizes_keep_macro_not_micro_average(self):
        a, b = cells(0), cells(0)
        a["repobench"] = {i: 1 for i in range(500)}
        b["repobench"] = {i: 0 for i in range(500)}
        result = grid.paired_bootstrap(a, b, B=40)
        self.assertEqual(result["point_diff"], 20)
        self.assertEqual(result["ci95"], [20, 20])
        self.assertEqual(result["aligned_n"]["repobench"], 500)

    def test_zero_interval_not_significant(self):
        result = grid.paired_bootstrap(cells(.75), cells(.75), B=40)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertFalse(result["ci_excludes_zero"])

    def test_mixed_differences_interval_still_includes_zero(self):
        a = {t: {"a": 0, "b": 1} for t in grid.AVG5_TASKS}
        b = {t: {"a": 1, "b": 0} for t in grid.AVG5_TASKS}
        result = grid.paired_bootstrap(a, b, B=200)
        self.assertEqual(result["point_diff"], 0)
        self.assertLess(result["ci95"][0], 0)
        self.assertGreater(result["ci95"][1], 0)
        self.assertFalse(result["ci_excludes_zero"])

    def test_main_verdict_ranking_and_ci_share_scale(self):
        tags = ["pred_nb8nt320nl8192", "pred_nb4nt128nl1024"]
        complete = {tag: {} for tag in tags}
        scores = {tag: {"avg5": value} for tag, value in zip(tags, (75, 50))}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "verdict.json"
            with patch.object(grid, "collect_cells", return_value=(complete, {}, [], [])), \
                 patch.object(grid, "stage_and_score", return_value=scores), \
                 patch.object(grid, "per_example_scores", side_effect=[cells(.75), cells(.5)]), \
                 patch.object(grid, "OUT", str(out)), \
                 patch.object(grid.sys, "argv", ["grid", "--bootstrap", "40"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(grid.main(), 0)
            result = json.loads(out.read_text())
        ranking = result["ranking"]
        ci = next(iter(result["paired_ci"].values()))
        self.assertEqual(ci["point_diff"], ranking[0]["avg5"] - ranking[1]["avg5"])
        self.assertEqual(result["meta"]["score_unit"], "points_0_100")
        self.assertEqual(ci["unit"], "score_points")


if __name__ == "__main__":
    unittest.main()
