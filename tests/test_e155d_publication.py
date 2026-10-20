"""Exercise the real T2 main publication path without model dependencies."""
import contextlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'exp/trace/analyze_e155d_grid.py'
spec = importlib.util.spec_from_file_location('grid_publication', SOURCE)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def crash_publication():
    def crash(value, stream, **kwargs):
        stream.write('{"ranking":')
        stream.flush()
        os._exit(23)
    with patch.object(grid, 'collect_cells', return_value=({}, {}, [], [])), \
         patch.object(sys, 'argv', ['grid']), patch.object(grid.json, 'dump', side_effect=crash):
        grid.main()


def paused_publication(halfway, resume):
    def slow_dump(value, stream, **kwargs):
        text = json.dumps(value, **kwargs)
        middle = len(text) // 2
        stream.write(text[:middle])
        stream.flush()
        halfway.set()
        if not resume.wait(5):
            raise RuntimeError('reader did not release writer')
        stream.write(text[middle:])
    with patch.object(grid, 'collect_cells', return_value=({}, {}, [], [])), \
         patch.object(sys, 'argv', ['grid']), patch.object(grid.json, 'dump', side_effect=slow_dump):
        grid.main()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / 'verdict.json'
        self.before = b'{"previous":"verified"}\n'
        self.out.write_bytes(self.before)
        for name, value in [('OUT', str(self.out)), ('STAGING', str(self.root / 'stage'))]:
            p = patch.object(grid, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_main(self):
        with patch.object(grid, 'collect_cells', return_value=({}, {}, [], [])), \
             patch.object(sys, 'argv', ['grid']), \
             contextlib.redirect_stdout(io.StringIO()):
            return grid.main()

    def assert_preserved(self):
        self.assertEqual(self.out.read_bytes(), self.before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['verdict.json'])

    @staticmethod
    def interrupt(value, stream, **kwargs):
        stream.write('{"ranking":')
        stream.flush()
        raise OSError('injected output interruption')

    def test_interrupted_serialization_preserves_previous_verdict(self):
        with patch.object(grid.json, 'dump', side_effect=self.interrupt):
            with self.assertRaisesRegex(OSError, 'injected'):
                self.run_main()
        self.assert_preserved()

    def test_interrupted_first_publication_leaves_no_partial_verdict(self):
        self.out.unlink()
        with patch.object(grid.json, 'dump', side_effect=self.interrupt):
            with self.assertRaises(OSError):
                self.run_main()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_fsync_failure_preserves_previous_verdict(self):
        with patch.object(grid.os, 'fsync', side_effect=OSError('injected fsync')):
            with self.assertRaisesRegex(OSError, 'fsync'):
                self.run_main()
        self.assert_preserved()

    def test_replace_failure_preserves_previous_verdict(self):
        with patch.object(grid.os, 'replace', side_effect=OSError('injected replace')):
            with self.assertRaisesRegex(OSError, 'replace'):
                self.run_main()
        self.assert_preserved()

    @unittest.skipUnless(os.name == 'posix', 'fork fault test requires POSIX')
    def test_process_exit_during_write_preserves_public_verdict(self):
        worker = multiprocessing.get_context('fork').Process(target=crash_publication)
        worker.start()
        try:
            worker.join(5)
            self.assertEqual(worker.exitcode, 23)
            self.assertEqual(self.out.read_bytes(), self.before)
        finally:
            if worker.is_alive():
                worker.kill()
            worker.join()

    @unittest.skipUnless(os.name == 'posix', 'reader/writer test requires POSIX')
    def test_reader_sees_old_complete_verdict_until_writer_commits(self):
        ctx = multiprocessing.get_context('fork')
        halfway, resume = ctx.Event(), ctx.Event()
        worker = ctx.Process(target=paused_publication, args=(halfway, resume))
        worker.start()
        try:
            self.assertTrue(halfway.wait(5))
            self.assertEqual(self.out.read_bytes(), self.before)
        finally:
            resume.set()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
            worker.join()
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(json.loads(self.out.read_bytes())['ranking'], [])

    def test_success_keeps_existing_mode_and_complete_payload(self):
        self.out.chmod(0o640)
        self.assertEqual(self.run_main(), 0)
        value = json.loads(self.out.read_bytes())
        self.assertEqual(value['counts'], dict(complete=0, incomplete=0, eval_errors=0, excluded=[]))
        self.assertEqual(value['ranking'], [])
        self.assertEqual(value['paired_ci'], {})
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o640)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['verdict.json'])

    def test_mode_setup_failure_closes_and_removes_owned_temporary(self):
        with patch.object(grid.os, 'fchmod', side_effect=OSError('injected mode')):
            with self.assertRaisesRegex(OSError, 'mode'):
                self.run_main()
        self.assert_preserved()

    def test_temporary_collision_does_not_remove_another_file(self):
        collision = self.root / '.verdict.json.tmp-collision'
        collision.write_bytes(b'not owned by this publication')
        fake_uuid = SimpleNamespace(uuid4=lambda: SimpleNamespace(hex='collision'))
        with patch.object(grid, 'uuid', fake_uuid, create=True):
            with self.assertRaises(FileExistsError):
                self.run_main()
        self.assertEqual(self.out.read_bytes(), self.before)
        self.assertEqual(collision.read_bytes(), b'not owned by this publication')

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlink unsupported')
    def test_symlink_destination_does_not_modify_external_target(self):
        target = self.root / 'other.json'
        target.write_bytes(self.before)
        self.out.unlink()
        self.out.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_main()
        self.assertEqual(target.read_bytes(), self.before)
        self.assertTrue(self.out.is_symlink())


if __name__ == '__main__':
    unittest.main()
