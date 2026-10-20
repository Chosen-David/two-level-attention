"""Grid labels must agree with frozen producer treatment, before scoring."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('grid_geometry', ROOT / 'exp/trace/analyze_e155d_grid.py')
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)
tree = ast.parse((ROOT / 'sparse_attn/info.py').read_text())
DEFAULTS = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_TREATMENT_FIELD_DEFAULTS'
                        for t in n.targets))


def manifest_for(tag):
    m = copy.deepcopy(DEFAULTS)
    kind, nb, nt, nl = grid.parse_tag(tag)
    if kind == 'geo':
        m.update(tli_near_pages=nb, tli_near_tokens=nt, tli_near_len=nl)
    return m


class GeometryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name, value in [('COLLECT_ROOT', str(self.root / 'collect')),
                            ('STAGING', str(self.root / 'stage')),
                            ('HOSTS', ['local']),
                            ('MIN_ROWS', dict.fromkeys(grid.AVG5_TASKS, 2))]:
            p = patch.object(grid, name, value)
            p.start()
            self.addCleanup(p.stop)

    def cell(self, tag, manifest=None):
        m = manifest_for(tag) if manifest is None else manifest
        text = json.dumps(m, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(text.encode()).hexdigest()[:10]
        folder = self.root / 'collect/local' / tag
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for task in grid.AVG5_TASKS:
            p = folder / f'{task}-tli_h{digest}-run.jsonl'
            p.write_text('{"_id":"one","pred":"yes","answers":["yes"]}\n' * 2)
            Path(str(p) + '.tli_manifest.json').write_text(text)
            paths[task] = p
        return paths

    @staticmethod
    def score(command, **kwargs):
        pred = Path(command[command.index('--output-dir') + 1]) / 'pred'
        (pred / 'result.json').write_text(json.dumps({
            p.stem: {'score': 50, 'budget': 1024} for p in pred.glob('*.jsonl')}))
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')

    def rejected(self, tag):
        complete, *_ = grid.collect_cells()
        with patch.object(grid.subprocess, 'run', side_effect=self.score) as score:
            result = grid.stage_and_score(complete)
        self.assertIn('eval_error', result[tag])
        score.assert_not_called()

    def test_wrong_declared_geometry_is_not_scored(self):
        tag = 'pred_nb8nt320nl8192'
        m = manifest_for(tag)
        m['tli_near_pages'] = 4
        self.cell(tag, m)
        self.rejected(tag)

    def mismatch(self, key, value):
        tag = 'pred_nb8nt320nl8192'
        m = manifest_for(tag)
        m[key] = value
        self.cell(tag, m)
        self.rejected(tag)

    def test_wrong_token_is_not_scored(self):
        self.mismatch('tli_near_tokens', 'off')

    def test_wrong_width_is_not_scored(self):
        self.mismatch('tli_near_len', 4096)

    def test_wrong_page_size_is_not_scored(self):
        self.mismatch('tia_block_size', 32)

    def test_numeric_string_is_not_integer_quota_evidence(self):
        self.mismatch('tli_near_pages', '8')

    def test_float_is_not_integer_quota_evidence(self):
        self.mismatch('tli_near_tokens', 320.0)

    def test_missing_manifest_is_not_scored(self):
        tag = 'pred_nb8nt320nl8192'
        paths = self.cell(tag)
        Path(str(paths['qasper']) + '.tli_manifest.json').unlink()
        self.rejected(tag)

    def test_manifest_filename_hash_mismatch_is_not_scored(self):
        tag = 'pred_nb8nt320nl8192'
        paths = self.cell(tag)
        sidecar = Path(str(paths['qasper']) + '.tli_manifest.json')
        m = json.loads(sidecar.read_text())
        m['tli_near_method'] = 'minmax'
        sidecar.write_text(json.dumps(m, sort_keys=True, ensure_ascii=False))
        self.rejected(tag)

    def test_control_cannot_contain_explicit_geometry(self):
        tag = 'pred_ctrl_singlepool'
        self.cell(tag, manifest_for('pred_nb8nt320nl8192'))
        self.rejected(tag)

    def test_control_cannot_declare_ratio_partition(self):
        tag = 'pred_ctrl_singlepool'
        m = manifest_for(tag)
        m.update(tli_alpha=.25, tli_beta=.5)
        self.cell(tag, m)
        self.rejected(tag)

    def test_invalid_metadata_is_not_scored(self):
        tag = 'pred_nb8nt320nl8192'
        paths = self.cell(tag)
        Path(str(paths['qasper']) + '.tli_manifest.json').write_text('[]')
        self.rejected(tag)

    def test_non_tli_method_is_not_a_grid_cell(self):
        self.mismatch('method', 'fasa_vendor')

    def test_missing_geometric_field_is_unknown(self):
        tag = 'pred_nb8nt320nl8192'
        m = manifest_for(tag)
        del m['tli_near_len']
        self.cell(tag, m)
        self.rejected(tag)

    def test_changed_treatment_during_evaluation_is_not_ranked(self):
        tag = 'pred_nb8nt320nl8192'
        self.cell(tag)
        complete, *_ = grid.collect_cells()
        def mutate(command, **kwargs):
            result = self.score(command, **kwargs)
            pred = Path(command[command.index('--output-dir') + 1]) / 'pred'
            sidecar = next(pred.glob('qasper-*.tli_manifest.json'))
            # Same object and filename identity, different snapshot bytes.
            sidecar.write_text(json.dumps(json.loads(sidecar.read_text()), indent=2))
            return result
        with patch.object(grid.subprocess, 'run', side_effect=mutate):
            self.assertIn('eval_error', grid.stage_and_score(complete)[tag])

    def test_wrong_geometry_never_reaches_main_ranking(self):
        good, bad = 'pred_nb4nt128nl1024', 'pred_nb8nt320nl8192'
        self.cell(good)
        self.cell(bad, manifest_for(good))
        out = self.root / 'verdict.json'
        with patch.object(grid.subprocess, 'run', side_effect=self.score), \
             patch.object(grid, 'OUT', str(out)), \
             patch.object(grid.sys, 'argv', ['grid', '--bootstrap', '40']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main(), 0)
        result = json.loads(out.read_text())
        self.assertEqual([r['tag'] for r in result['ranking']], [good])
        self.assertEqual(result['counts']['eval_errors'], 1)
        self.assertIn('treatment identity', result['evaluation_errors'][bad])
        self.assertFalse(result['paired_ci'])

    def test_all_87_matching_cells_keep_scores(self):
        tags = ['pred_ctrl_singlepool']
        for nb in sorted(grid.NB_AXIS):
            for nt in sorted(grid.NT_AXIS, key=str):
                for nl in sorted(grid.NL_AXIS):
                    tag = f'pred_nb{nb}nt{nt}nl{nl}'
                    if grid.parse_tag(tag):
                        tags.append(tag)
        self.assertEqual(len(tags), 87)
        for tag in tags:
            self.cell(tag)
        complete, *_ = grid.collect_cells()
        with patch.object(grid.subprocess, 'run', side_effect=self.score):
            results = grid.stage_and_score(complete)
        self.assertEqual(len(results), 87)
        self.assertTrue(all(v.get('avg5') == 50 for v in results.values()))
        self.assertTrue(all(set(v['treatment_sha256']) == set(grid.AVG5_TASKS)
                            for v in results.values()))


if __name__ == '__main__':
    unittest.main()
