import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from exp.paper import qwen_fig1 as f
from exp.paper.observations import o1_mass, protected_ids


class QwenFig1Test(unittest.TestCase):
    def test_per_head_softmax_then_group_mean_and_causal_visibility(self):
        q = np.zeros((2, 128), np.float32)
        k = np.zeros((4, 1, 128), np.float32)
        q[0, 0] = 4
        k[:, 0, 0] = [0, 1, 2, 1000]
        heads, groups = f.reference(q, k, 2)
        logits = torch.tensor(q) @ torch.tensor(k[:3, 0]).T / 128 ** .5
        expected = logits.softmax(-1).numpy()
        np.testing.assert_allclose(heads, expected, atol=1e-7)
        np.testing.assert_allclose(groups[0], expected.mean(0), atol=1e-7)
        self.assertGreater(float(np.max(np.abs(groups[0] - logits.sum(0).softmax(0).numpy()))), .05)
        k[3] = -999
        np.testing.assert_array_equal(f.reference(q, k, 2)[1], groups)

    def test_bounds_match_repaired_collector(self):
        path = f.ROOT / 'agent_doc/guide/paper_figs/trace/e140_o_trace.py'
        spec = importlib.util.spec_from_file_location('existing_e140', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for pos in (0, 126, 127, 128, 254, 255, 256, 258, 1023, 1024, 32767):
            self.assertEqual(f.bounds(pos), module.regions(pos))

    def test_uniform_concentration_and_partial_absolute_pages(self):
        p = np.full(321, 1/321, np.float32)
        row = f.concentration(p, 129, 257)
        self.assertEqual(row['token_count_90'], 116)
        self.assertEqual(row['page_ids'], [2, 3, 4])
        self.assertEqual(row['page_token_counts'], [63, 64, 1])
        self.assertEqual(row['page_count_90'], 2)
        self.assertAlmostEqual(row['token_fraction_90'], 116/128)
        self.assertAlmostEqual(row['page_fraction_90'], 2/3)
        self.assertAlmostEqual(row['mass'], 128/321, places=6)

    def test_empty_zero_mass_and_spike(self):
        p = np.zeros(321, np.float32); p[256] = 1
        self.assertEqual(f.concentration(p, 128, 128)['undefined_reason'], 'empty_region')
        self.assertEqual(f.concentration(p, 128, 256)['undefined_reason'], 'zero_region_mass')
        self.assertIsNone(f.concentration(p, 128, 256)['token_fraction_90'])
        r = f.concentration(p, 129, 257)
        self.assertEqual(r['token_count_90'], 1)
        self.assertEqual(r['page_count_90'], 1)

    def test_distance_bins_density_conservation_and_head_levels(self):
        for pos in (0, 127, 128, 255, 256, 1024):
            rows = f.query_rows(np.zeros((4, 128), np.float32), np.zeros((pos+1, 2, 128), np.float32), pos)
            self.assertEqual(len(rows), 6)
            self.assertEqual([r['kv_group'] for r in rows], [0, 0, 1, 1, 0, 1])
            p = np.full(pos+1, 1/(pos+1), np.float32)
            edges = list(range(0, (pos//64+1)*64+1, 64))
            oracle = o1_mass(p, pos, f.bounds(pos)['near_lo'], protected_ids(pos, 128, 128), edges)
            oracle = {(r['region'], r['distance_lo']): r for r in oracle if r['token_count']}
            for row in rows:
                self.assertAlmostEqual(row['mass_protected_union']+row['near']['mass']+row['far']['mass'], 1, places=6)
                for b in row['distance_bins']:
                    expected = oracle[b['region'], b['distance_lo']]
                    self.assertEqual(b['token_count'], expected['token_count'])
                    self.assertAlmostEqual(b['mass'], expected['mass'], places=6)
                    self.assertAlmostEqual(b['density'], 1/(pos+1), places=7)
                self.assertEqual(len(row['distance_bins']), len(oracle))

    def test_invalid_inputs_fail_under_optimized_python_too(self):
        q = np.zeros((2, 128), np.float32); k = np.zeros((300, 1, 128), np.float32)
        for badq, badk, pos in ((q, k, -1), (q, k, 300), (q, k, True),
                                (q.astype(int), k, 2), (q[:, :64], k, 2),
                                (q[:0], k, 2), (q+np.nan, k, 2),
                                (q, k+np.inf, 2), (q, np.zeros((3, 3, 128), np.float32), 2),
                                (q+1e30, k+1e30, 2)):
            with self.subTest(pos=pos), self.assertRaises(ValueError):
                f.reference(badq, badk, pos)
        for p, lo, hi in ((np.array([np.nan]), 0, 1), (np.array([-.1, 1.1]), 0, 2),
                           (np.array([.5]), 0, 1), (np.array([1.]), -1, 1),
                           (np.array([1.]), 0, 2), (np.array([1]), 0, 1)):
            with self.assertRaises(ValueError):
                f.concentration(p, lo, hi)

    def test_distance_contributor_counts_do_not_impute_missing_queries(self):
        rows = []
        for pos in (300, 1024):
            for row in f.query_rows(np.zeros((2, 128), np.float32), np.zeros((pos+1, 1, 128), np.float32), pos):
                row.update(source_sequence='s', layer=8)
                rows.append(row)
        summary = f.summarize_distance(rows)
        near = next(r for r in summary if r['level']=='q_head' and r['head_index']==0 and r['region']=='near' and r['distance_lo']==128)
        self.assertEqual(near['contributing_query_count'], 2)
        self.assertAlmostEqual(near['mean_per_token_density'], (1/301+1/1025)/2, places=7)
        self.assertTrue(all(r['contributing_query_count']==1 for r in summary if r['region']=='far'))
        with self.assertRaisesRegex(ValueError, 'duplicate query'):
            f.summarize_distance(rows+rows)

    def test_missing_archive_cli_blocks_without_rows_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'report.json'
            flags = ['-O'] if sys.flags.optimize else []
            cmd = [sys.executable, *flags, str(Path(f.__file__)), '--root', temp, '--analyze', '--out', str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            result = json.loads(out.read_text())
            self.assertFalse(result['paper_ready'])
            self.assertEqual(result['rows'], [])
            self.assertEqual(result['status'], 'blocked')
            original = out.read_bytes()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), original)

    def test_adapter_selects_only_anchors_and_keeps_identity_unresolved(self):
        # Exercise audited-file consumption without substituting the real index.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'lb_hotpotqa_0' / 'layer08.pt'
            path.parent.mkdir()
            positions = sorted(set(range(64, 320)) | set(range(10, 320, 20)))
            torch.save(dict(q=torch.zeros(len(positions), 2, 128), k=torch.zeros(320, 1, 128),
                            qpos=torch.tensor(positions), S=320), path)
            entry = dict(relative_path='lb_hotpotqa_0/layer08.pt', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         schema=dict(anchor_positions=list(range(10, 320, 20)),
                                     selected_q_rows=[positions.index(p) for p in range(10, 320, 20)]))
            receipt = dict(exit_code=0, files=[entry])
            with patch.object(f, 'audit', return_value=receipt):
                result = f.analyze_archive(root, analyze=True)
                self.assertFalse(result['paper_ready'])
                self.assertEqual(result['status'], 'diagnostic-unverified-provenance')
                self.assertEqual(len(result['rows']), 16*3)
                self.assertEqual(sorted({r['query_position'] for r in result['rows']}), list(range(10, 320, 20)))
                self.assertTrue(all(r['layer'] == 8 and r['source_sequence']=='lb_hotpotqa_0' for r in result['rows']))
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed after audit'):
                    f.analyze_archive(root, analyze=True)


if __name__ == '__main__':
    unittest.main()
