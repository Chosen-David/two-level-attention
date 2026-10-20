"""Collection gates use real files, without model or scoring dependencies."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "exp/trace/analyze_e155d_grid.py"
spec = importlib.util.spec_from_file_location("e155d_grid", SCRIPT)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


class GridCollectionTests(unittest.TestCase):
    tag = "pred_nb8nt320nl8192"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for obj, name, value in ((grid, "COLLECT_ROOT", tmp.name),
                                  (grid, "HOSTS", ["a", "b"])):
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def cell(self, host, tasks=None, marker="same"):
        directory = self.root / host / self.tag
        directory.mkdir(parents=True)
        for task in grid.AVG5_TASKS if tasks is None else tasks:
            rows = grid.MIN_ROWS.get(task, 200)
            (directory / (task + "-run.jsonl")).write_text(
                ('{"pred":"' + marker + '"}\n') * rows)
        return directory

    def test_ambiguous_task_is_not_complete(self):
        directory = self.cell("a")
        (directory / "hotpotqa-other.jsonl").write_text('{}\n' * 200)
        complete, incomplete, pending, excluded = grid.collect_cells()
        self.assertNotIn(self.tag, complete)
        self.assertTrue(any("ambiguous_tasks" in item for item in excluded))
        self.assertEqual(incomplete[self.tag]["a"], ["hotpotqa"])
        self.assertEqual(pending[0]["per_host_missing"], incomplete[self.tag])

    def test_valid_host_can_replace_ambiguous_host(self):
        directory = self.cell("a")
        (directory / "hotpotqa-other.jsonl").write_text('{}\n' * 200)
        self.cell("b")
        complete, _, _, _ = grid.collect_cells()
        self.assertEqual(complete[self.tag]["source_host"], "b")

    def test_conflicting_complete_replicas_are_not_scored(self):
        self.cell("a")
        self.cell("b", marker="different")
        complete, incomplete, pending, _ = grid.collect_cells()
        self.assertNotIn(self.tag, complete)
        self.assertIn(self.tag, incomplete)
        self.assertEqual(pending[0]["reason"], "sha_conflict")

    def test_identical_complete_replicas_are_deduplicated(self):
        self.cell("a")
        self.cell("b")
        complete, incomplete, pending, _ = grid.collect_cells()
        self.assertEqual(complete[self.tag]["dup_hosts"], ["b"])
        self.assertFalse(incomplete or pending)

    def test_complementary_partial_hosts_are_not_merged(self):
        self.cell("a", tasks=grid.AVG5_TASKS[:2])
        self.cell("b", tasks=grid.AVG5_TASKS[2:])
        complete, incomplete, pending, _ = grid.collect_cells()
        self.assertFalse(complete)
        self.assertIn(self.tag, incomplete)
        self.assertEqual(len(pending), 1)

    def test_empty_collection(self):
        self.assertEqual(grid.collect_cells(), ({}, {}, [], []))

    def test_one_row_short_is_incomplete(self):
        directory = self.cell("a")
        (directory / "repobench-run.jsonl").write_text('{}\n' * 499)
        complete, incomplete, pending, _ = grid.collect_cells()
        self.assertFalse(complete)
        self.assertEqual(incomplete[self.tag]["a"], ["repobench"])
        self.assertEqual(len(pending), 1)

    def test_three_identical_task_files_remain_ambiguous(self):
        directory = self.cell("a")
        content = (directory / "qasper-run.jsonl").read_bytes()
        for suffix in ("first", "last"):
            (directory / f"qasper-{suffix}.jsonl").write_bytes(content)
        complete, incomplete, _, _ = grid.collect_cells()
        self.assertFalse(complete)
        self.assertEqual(incomplete[self.tag]["a"], ["qasper"])

    def test_single_task_conflict_keeps_host_and_full_hashes(self):
        a = self.cell("a")
        b = self.cell("b")
        filename = "qasper-run.jsonl"
        (b / filename).write_text('{}\n' * 200)
        complete, _, pending, _ = grid.collect_cells()
        self.assertFalse(complete)
        conflict = pending[0]
        self.assertEqual(conflict["source_host"], "a")
        self.assertEqual(conflict["sha_conflicts"], [{"host": "b", "task_sha256": {
            "qasper": [grid.file_sha(a / filename), grid.file_sha(b / filename)]}}])
        self.assertEqual(conflict["per_host_missing"], {"a": [], "b": []})

    def test_majority_identical_cannot_overrule_conflict(self):
        self.cell("a")
        self.cell("b")
        self.cell("c", marker="different")
        with patch.object(grid, "HOSTS", ["a", "b", "c"]):
            complete, _, pending, _ = grid.collect_cells()
        self.assertFalse(complete)
        self.assertEqual(pending[0]["sha_conflicts"][0]["host"], "c")

    def test_complete_host_not_replaced_by_partial(self):
        self.cell("a", tasks=grid.AVG5_TASKS[:1], marker="different")
        self.cell("b")
        complete, incomplete, pending, _ = grid.collect_cells()
        self.assertEqual(complete[self.tag]["source_host"], "b")
        self.assertFalse(incomplete or pending)

    def test_legacy_axis_excluded_and_control_admitted(self):
        self.tag = "pred_nb8nt160nl8192"
        self.cell("a")
        self.tag = "pred_ctrl_singlepool"
        self.cell("a")
        complete, _, _, excluded = grid.collect_cells()
        self.assertEqual(list(complete), [self.tag])
        self.assertEqual(excluded, ["a/pred_nb8nt160nl8192"])

    def test_main_does_not_send_conflict_to_scoring_or_ci(self):
        conflict_tag = self.tag
        self.cell("a")
        self.cell("b", marker="different")
        self.tag = "pred_nb4nt128nl1024"
        self.cell("a")
        out = self.root / "output" / "verdict.json"
        def fake_score(complete):
            self.assertEqual(list(complete), [self.tag])
            return {name: {"avg5": 10.0} for name in complete}
        with patch.object(grid, "stage_and_score", side_effect=fake_score) as score, \
             patch.object(grid, "per_example_scores") as per_example, \
             patch.object(grid, "OUT", str(out)), \
             patch.object(grid.sys, "argv", ["analyze_e155d_grid.py"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main(), 0)
        score.assert_called_once()
        per_example.assert_not_called()
        verdict = json.loads(out.read_text())
        self.assertEqual([c["tag"] for c in verdict["ranking"]], [self.tag])
        self.assertEqual(verdict["counts"]["complete"], 1)
        self.assertEqual(verdict["counts"]["incomplete"], 1)
        self.assertIn(conflict_tag, verdict["incomplete_cells"])
        self.assertEqual(verdict["incomplete_before_closure"][0]["reason"], "sha_conflict")


if __name__ == "__main__":
    unittest.main()
