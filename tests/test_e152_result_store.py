"""CPU-only process/fault tests of the actual E152 output publication code."""
import ast
import json
import multiprocessing
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "exp/bench/e152_e2e_latency.py"


def load_append():
    tree = ast.parse(SOURCE.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    helper = next((n for n in functions if n.name == "_append_record"), None)
    if helper is None:
        # Execute the baseline's actual main tail, without importing a model.
        main = next(n for n in functions if n.name == "main")
        start = next(i for i, n in enumerate(main.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "existing")
        helper = ast.parse("def _append_record(path, record):\n    args = SimpleNamespace(out=path)\n").body[0]
        helper.body.extend(main.body[start:-1])  # exclude only the final print
        ast.fix_missing_locations(helper)
    ns = {"os": os, "json": json, "SimpleNamespace": SimpleNamespace}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def race_worker(path, role, a_read, b_started, b_read, a_done):
    ns = load_append()
    def read(stream):
        rows = json.load(stream)
        if role == "A":
            a_read.set()
            if not b_started.wait(5):
                raise RuntimeError("second writer did not start")
            # Baseline B can read before A publishes; with a lock it cannot.
            b_read.wait(.2)
        else:
            b_read.set()
            if not a_done.wait(5):
                raise RuntimeError("first writer did not finish")
        return rows
    ns["json"] = SimpleNamespace(load=read, dump=json.dump)
    if role == "B":
        if not a_read.wait(5):
            raise RuntimeError("first writer did not read")
        b_started.set()
    ns["_append_record"](path, {"writer": role})
    if role == "A":
        a_done.set()


def crash_worker(path):
    ns = load_append()
    def crash(rows, stream, **kwargs):
        stream.write('[{"incomplete":')
        stream.flush()
        os._exit(23)
    ns["json"] = SimpleNamespace(load=json.load, dump=crash)
    ns["_append_record"](path, {"writer": "crash"})


@unittest.skipUnless(os.name == "posix", "E152 writer uses POSIX file locking")
class ResultStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "runs.json"
        self.before = b'[{"schema":"e152-e2e-latency-v1","runs":[]}]\n'
        self.path.write_bytes(self.before)
        self.append = load_append()["_append_record"]

    def test_serial_append_preserves_history_and_unicode(self):
        self.append(self.path, {"label": "新测量", "runs": []})
        rows = json.loads(self.path.read_text())
        self.assertEqual(rows[0], json.loads(self.before)[0])
        self.assertEqual(rows[1]["label"], "新测量")

    def test_overlapping_processes_cannot_lose_a_record(self):
        ctx = multiprocessing.get_context("fork")
        events = [ctx.Event() for _ in range(4)]
        workers = [ctx.Process(target=race_worker, args=(str(self.path), role, *events))
                   for role in ("A", "B")]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(8)
                self.assertEqual(worker.exitcode, 0)
            rows = json.loads(self.path.read_text())
            self.assertEqual([r.get("writer") for r in rows], [None, "A", "B"])
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(3)

    def test_interrupted_writer_preserves_previous_complete_file(self):
        worker = multiprocessing.get_context("fork").Process(
            target=crash_worker, args=(str(self.path),))
        worker.start()
        worker.join(8)
        try:
            self.assertEqual(worker.exitcode, 23)
            self.assertEqual(self.path.read_bytes(), self.before)
            # Kernel releases the advisory lock on process exit; retry can append.
            self.append(self.path, {"writer": "retry"})
            self.assertEqual(len(json.loads(self.path.read_text())), 2)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(3)

    def test_serialization_failure_preserves_bytes_and_removes_temp(self):
        def fail(rows, stream, **kwargs):
            stream.write("partial")
            raise TypeError("serialization failed")
        with patch.object(json, "dump", side_effect=fail):
            with self.assertRaisesRegex(TypeError, "serialization failed"):
                self.append(self.path, {"writer": "bad"})
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_replace_failure_preserves_previous_complete_file(self):
        with patch.object(os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.append(self.path, {"writer": "bad"})
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_fsync_failure_preserves_previous_complete_file(self):
        with patch.object(os, "fsync", side_effect=OSError("sync failed")):
            with self.assertRaisesRegex(OSError, "sync failed"):
                self.append(self.path, {"writer": "bad"})
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_corrupt_or_nonarray_input_not_overwritten(self):
        for content in (b"broken", b'{"wrong":"schema"}'):
            self.path.write_bytes(content)
            with self.assertRaises(ValueError):
                self.append(self.path, {"writer": "new"})
            self.assertEqual(self.path.read_bytes(), content)

    def test_new_file_and_existing_mode(self):
        fresh = self.path.parent / "fresh.json"
        self.append(str(fresh), {"writer": "first"})
        self.assertEqual(json.loads(fresh.read_text()), [{"writer": "first"}])
        self.path.chmod(0o640)
        self.append(self.path, {"writer": "second"})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
