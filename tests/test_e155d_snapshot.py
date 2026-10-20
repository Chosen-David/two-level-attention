"""T2 ranking and CI must consume the same admitted prediction bytes."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'exp/trace/analyze_e155d_grid.py'
spec = importlib.util.spec_from_file_location('e155d_snapshot', SCRIPT)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tags = ['pred_nb8nt320nl8192', 'pred_nb4nt128nl1024']
        self.paths = {}
        for tag, score in zip(self.tags, (.75, .5)):
            directory = self.root / 'collect/local' / tag
            directory.mkdir(parents=True)
            _, nb, nt, nl = grid.parse_tag(tag)
            manifest = json.dumps(dict(tli_near_pages=nb, tli_near_tokens=nt,
                                       tli_near_len=nl, tia_block_size=64),
                                  sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(manifest.encode()).hexdigest()[:10]
            for task in grid.AVG5_TASKS:
                path = directory / (task + '-tli_h' + digest + '-run.jsonl')
                self.write_rows(path, score)
                Path(str(path) + '.tli_manifest.json').write_text(manifest)
                self.paths[tag, task] = path
        for name, value in [('COLLECT_ROOT', str(self.root / 'collect')),
                            ('STAGING', str(self.root / 'staging')),
                            ('OUT', str(self.root / 'verdict.json')),
                            ('HOSTS', ['local']),
                            ('MIN_ROWS', dict.fromkeys(grid.AVG5_TASKS, 2))]:
            mock = patch.object(grid, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        fake = types.ModuleType('benchmark.LongBench.eval')
        fake.sample_score = lambda task, pred, answers, classes: pred
        mock = patch.dict(sys.modules, {'benchmark.LongBench.eval': fake})
        mock.start()
        self.addCleanup(mock.stop)
        self.after_eval = lambda tag, pred: None

    @staticmethod
    def write_rows(path, score):
        path.write_text(''.join(json.dumps({'_id': i, 'pred': score,
                                           'answers': ['yes']}) + '\n' for i in range(2)))

    def eval_stub(self, command, **kwargs):
        root = Path(command[command.index('--output-dir') + 1])
        pred = root / 'pred'
        scores = {}
        for path in pred.glob('*.jsonl'):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            scores[path.stem] = {'score': 100 * sum(r['pred'] for r in rows) / len(rows)}
        (pred / 'result.json').write_text(json.dumps(scores))
        self.after_eval(root.name, pred)
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')

    def run_main(self):
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub), \
             patch.object(sys, 'argv', ['grid', '--bootstrap', '40']), \
             contextlib.redirect_stdout(io.StringIO()):
            grid.main()
        return json.loads(Path(grid.OUT).read_text())

    def test_source_replacement_after_score_cannot_reverse_ci(self):
        def replace(tag, pred):
            if tag == self.tags[0]:
                for task in grid.AVG5_TASKS:
                    self.write_rows(self.paths[tag, task], 0)
        self.after_eval = replace
        verdict = self.run_main()
        self.assertEqual([v['avg5'] for v in verdict['ranking']], [75, 50])
        ci = next(iter(verdict['paired_ci'].values()))
        self.assertEqual(ci['point_diff'], 25)
        self.assertEqual(ci['ci95'], [25, 25])

    def test_repeated_scoring_preserves_prior_prediction_snapshot(self):
        complete, *_ = grid.collect_cells()
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            first = grid.stage_and_score(complete)
            for path in self.paths.values():
                self.write_rows(path, 0)
            second = grid.stage_and_score(grid.collect_cells()[0])
        for tag in self.tags:
            self.assertNotEqual(first[tag]['files'], second[tag]['files'])
            self.assertEqual(grid.per_example_scores(first[tag]),
                             {t: {0: .75 if tag == self.tags[0] else .5,
                                  1: .75 if tag == self.tags[0] else .5}
                              for t in grid.AVG5_TASKS})

    def test_scoring_preserves_unowned_staging_contents(self):
        sentinel = Path(grid.STAGING) / 'other-active-run' / 'evidence.txt'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b'owned by another invocation')
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            grid.stage_and_score(grid.collect_cells()[0])
        self.assertEqual(sentinel.read_bytes(), b'owned by another invocation')

    def test_interleaved_scoring_keeps_both_snapshots_usable_for_ci(self):
        nested = {}
        def interleave(tag, pred):
            if not nested:
                nested['started'] = True
                nested['scored'] = grid.stage_and_score(grid.collect_cells()[0])
        self.after_eval = interleave
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            first = grid.stage_and_score(grid.collect_cells()[0])
        for scored in (first, nested['scored']):
            self.assertEqual([scored[t]['avg5'] for t in self.tags], [75, 50])
            a, b = [grid.per_example_scores(scored[t]) for t in self.tags]
            self.assertEqual(grid.paired_bootstrap(a, b, B=40)['ci95'], [25, 25])
        self.assertNotEqual(first[self.tags[0]]['files'],
                            nested['scored'][self.tags[0]]['files'])

    def test_failed_new_evaluation_preserves_prior_result_file(self):
        complete, *_ = grid.collect_cells()
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            first = grid.stage_and_score(complete)
        prior = Path(first[self.tags[0]]['files']['qasper']).parent / 'result.json'
        raw = prior.read_bytes()
        failed = types.SimpleNamespace(returncode=1, stdout='', stderr='injected failure')
        with patch.object(grid.subprocess, 'run', return_value=failed):
            second = grid.stage_and_score(complete)
        self.assertTrue(all('eval_error' in row for row in second.values()))
        self.assertEqual(prior.read_bytes(), raw)

    def test_empty_scoring_does_not_delete_existing_snapshots(self):
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            first = grid.stage_and_score(grid.collect_cells()[0])
        self.assertEqual(grid.stage_and_score({}), {})
        self.assertEqual(grid.per_example_scores(first[self.tags[0]]),
                         {t: {0: .75, 1: .75} for t in grid.AVG5_TASKS})

    def test_source_deletion_after_score_does_not_break_ci(self):
        self.after_eval = lambda tag, pred: [self.paths[tag, t].unlink() for t in grid.AVG5_TASKS]
        self.assertEqual(next(iter(self.run_main()['paired_ci'].values()))['point_diff'], 25)

    def test_changed_staging_during_eval_is_not_ranked(self):
        self.after_eval = lambda tag, pred: self.write_rows(next(pred.glob('qasper-*.jsonl')), 0)
        verdict = self.run_main()
        self.assertEqual(verdict['ranking'], [])
        self.assertEqual(verdict['counts']['eval_errors'], 2)

    def test_changed_source_after_admission_is_not_ranked(self):
        complete, *_ = grid.collect_cells()
        self.write_rows(self.paths[self.tags[0], 'qasper'], 0)
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            scored = grid.stage_and_score(complete)
        self.assertIn('eval_error', scored[self.tags[0]])
        self.assertEqual(scored[self.tags[1]]['avg5'], 50)

    def test_staged_prediction_change_before_ci_is_rejected(self):
        complete, *_ = grid.collect_cells()
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            scored = grid.stage_and_score(complete)
        info = scored[self.tags[0]]
        self.assertIn('files', info)
        self.write_rows(Path(info['files']['qasper']), 0)
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            grid.per_example_scores(info)

    def test_unchanged_snapshot_preserves_scores_and_exposes_hashes(self):
        verdict = self.run_main()
        for row in verdict['ranking']:
            self.assertEqual(set(row['file_sha256']), set(grid.AVG5_TASKS))
            for task, path in row['files'].items():
                self.assertEqual(grid.file_sha(path), row['file_sha256'][task])
        self.assertEqual(next(iter(verdict['paired_ci'].values()))['point_diff'], 25)

    def test_source_deleted_after_admission_is_not_ranked(self):
        complete, *_ = grid.collect_cells()
        self.paths[self.tags[0], 'qasper'].unlink()
        with patch.object(grid.subprocess, 'run', side_effect=self.eval_stub):
            scored = grid.stage_and_score(complete)
        self.assertIn('eval_error', scored[self.tags[0]])

    def test_ci_change_stops_main_before_new_verdict(self):
        real_stage = grid.stage_and_score
        def stage_then_change(complete):
            scored = real_stage(complete)
            self.write_rows(Path(scored[self.tags[0]]['files']['qasper']), 0)
            return scored
        with patch.object(grid, 'stage_and_score', side_effect=stage_then_change):
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                self.run_main()
        self.assertFalse(Path(grid.OUT).exists())

    def test_line_endings_and_unicode_bytes_remain_bound(self):
        # Collection counts physical LF lines; standalone CI also historically
        # accepts lone CR separators, so preserve that helper compatibility.
        for ending in ('\n', '\r\n', '\r'):
            with self.subTest(ending=repr(ending)):
                for path in self.paths.values():
                    rows = [dict(_id=i, pred=.5, answers=['中文']) for i in range(2)]
                    path.write_bytes((ending.join(json.dumps(row, ensure_ascii=False)
                                                  for row in rows) + ending).encode('utf-8'))
                info = {'files': {t: str(self.paths[self.tags[0], t]) for t in grid.AVG5_TASKS}}
                info['file_sha256'] = {t: grid.file_sha(p) for t, p in info['files'].items()}
                scores = grid.per_example_scores(info)
                self.assertEqual(scores, {t: {0: .5, 1: .5} for t in grid.AVG5_TASKS})
                if ending != '\r':
                    verdict = self.run_main()
                    self.assertEqual(next(iter(verdict['paired_ci'].values()))['point_diff'], 0)
                    self.assertEqual([v['avg5'] for v in verdict['ranking']], [50, 50])


if __name__ == '__main__':
    unittest.main()
