"""Backfill must inspect every row before selecting the legacy ID fallback."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'exp/trace/backfill_run_cap_manifest.py'
spec = importlib.util.spec_from_file_location('backfill_identity_format', SCRIPT)
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def rows_with(*ids):
    rows = [{'pred': 'yes', 'answers': ['yes'], 'length': 10} for _ in ids]
    for row, sid in zip(rows, ids):
        if sid is not None:
            row['_id'] = sid
    return rows


class IdentityFormatTests(unittest.TestCase):
    def check_rows(self, rows, task='hotpotqa'):
        fp = backfill.recompute_dataset_fingerprint_rows(rows, task)
        return backfill.check_id_fingerprint(rows, task, fp)

    def test_all_missing_legacy_ids_remain_supported(self):
        self.assertEqual(self.check_rows(rows_with(None, None)), (None, False))

    def test_official_ids_remain_supported(self):
        self.assertEqual(self.check_rows(rows_with('official-a', 'official-b'), 'lbv2'),
                         (None, False))

    def test_empty_rows_remain_unproven(self):
        self.assertEqual(self.check_rows([]), (None, False))

    def test_valid_composite_ids_remain_proven(self):
        rows = rows_with(None, None)
        fp = backfill.recompute_dataset_fingerprint_rows(rows, 'hotpotqa')
        for i, row in enumerate(rows):
            row['_id'] = f'hotpotqa:{fp}:{i}'
        self.assertEqual(self.check_rows(rows), (fp, True))

    def test_legacy_first_cannot_hide_later_identity(self):
        for sid in ('hotpotqa:aaaaaaaaaaaa:1', 'musique:aaaaaaaaaaaa:1',
                    'hotpotqa:malformed:1', 'official-a'):
            with self.subTest(sid=sid), self.assertRaisesRegex(SystemExit, 'GATE-FAIL'):
                self.check_rows(rows_with(None, sid))

    def test_official_first_cannot_hide_composite_or_missing_id(self):
        for sid in (None, 'lbv2:aaaaaaaaaaaa:1', 'musique:aaaaaaaaaaaa:1'):
            with self.subTest(sid=sid), self.assertRaisesRegex(SystemExit, 'GATE-FAIL'):
                self.check_rows(rows_with('official-a', sid), 'lbv2')

    def test_composite_first_still_rejects_missing_id(self):
        with self.assertRaisesRegex(SystemExit, 'GATE-FAIL'):
            self.check_rows(rows_with('hotpotqa:aaaaaaaaaaaa:0', None))

    def test_v2_ids_cannot_be_backfilled_without_source_and_token_evidence(self):
        rows = rows_with(
            'hotpotqa:v2-' + 'a' * 64 + ':0',
            'hotpotqa:v2-' + 'a' * 64 + ':1',
        )
        with self.assertRaisesRegex(SystemExit, 'GATE-FAIL'):
            self.check_rows(rows)

    def test_cli_rejects_mixed_file_without_publishing_sidecar(self):
        rows = rows_with(None, 'hotpotqa:aaaaaaaaaaaa:1')
        with tempfile.TemporaryDirectory() as td:
            result = Path(td) / 'hotpotqa-none-123.jsonl'
            result.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            log = Path(td) / 'run.log'
            log.write_text('[CONTEXT-CAP-GATE] model=DeepSeek-V2-Lite-Chat\n'
                           'experiment_input_cap (live) = 16384\n'
                           'experiment_input_cap (registry protocol) = 31500\n')
            command = [sys.executable]
            if sys.flags.optimize:
                command.append('-O')
            command += [str(SCRIPT), '--result', str(result), '--log', str(log),
                        '--model', 'DeepSeek-V2-Lite-Chat', '--method', 'none',
                        '--task', 'hotpotqa', '--t', '123', '--max-new-tokens', '32',
                        '--expect-count', '2']
            observed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(observed.returncode, 0, observed.stdout + observed.stderr)
            self.assertIn('GATE-FAIL', observed.stdout + observed.stderr)
            self.assertFalse(Path(str(result) + '.runcap.json').exists())


if __name__ == '__main__':
    unittest.main()
