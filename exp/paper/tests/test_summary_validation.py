"""CPU regression for per-invocation protocol validation reuse (not row reuse)."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from exp.paper import contract
from exp.paper.tests.test_paper_contract import protocol, row


class SummaryValidation(unittest.TestCase):
    def test_protocol_is_validated_once_per_summary_not_once_per_row(self):
        p = protocol()
        rows = [row(p, arm) for arm in p['arms']]
        with patch.object(contract, 'protocol_errors', wraps=contract.protocol_errors) as check:
            contract.summarize(rows, p, 'e'*64)
        self.assertEqual(check.call_count, 1)

    def test_public_row_validation_still_checks_protocol_every_call(self):
        p = protocol()
        r = row(p, 'dense')
        contract.validate_timing(r, p, 'e'*64)
        p['status'] = 'unfrozen'
        with self.assertRaisesRegex(ValueError, 'unfrozen protocol'):
            contract.validate_timing(r, p, 'e'*64)

    def test_summary_does_not_cache_protocol_across_invocations(self):
        p = protocol()
        rows = [row(p, arm) for arm in p['arms']]
        contract.summarize(rows, p, 'e'*64)
        p['samples'][0]['input_ids_sha256'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'unfrozen protocol'):
            contract.summarize(rows, p, 'e'*64)

    def test_later_rows_keep_all_validation_gates(self):
        p = protocol()
        a, b = [row(p, arm) for arm in p['arms']]
        for key, value in [('protocol_sha256', 'wrong'), ('sample_id', 'unplanned'),
                           ('document_id', 'wrong'), ('input_ids_sha256', 'wrong'),
                           ('identity', {}), ('repetition', True), ('status', 'unknown'),
                           ('measured_this_run', False), ('aggregation', 'sum'),
                           ('request_wall_ms', float('nan')), ('fallback_count', 1),
                           ('actual_routes', {}), ('order', 0), ('support', {})]:
            with self.subTest(key=key):
                bad = dict(b, **{key: value})
                for validate in (lambda: contract.validate_timing(bad, p, 'e'*64),
                                 lambda: contract.summarize([a, bad], p, 'e'*64)):
                    with self.assertRaises(ValueError):
                        validate()

    def test_duplicate_missing_failed_and_empty_inputs_remain_distinct(self):
        p = protocol()
        a, b = [row(p, arm) for arm in p['arms']]
        with self.assertRaisesRegex(ValueError, 'duplicate timing row'):
            contract.summarize([a, b, copy.deepcopy(b)], p, 'e'*64)
        for rows, missing, failed in [([], 2, 0), ([a], 1, 0),
                                      ([a, dict(b, status='oom')], 0, 1)]:
            result = contract.summarize(rows, p, 'e'*64)
            self.assertEqual((result['status'], result['n_missing'], result['n_failed']),
                             ('blocked', missing, failed))
            self.assertEqual(result['pairs'], [])

    def test_actual_timing_cli_keeps_summary_and_rejects_bad_later_row(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            p = protocol()
            protocol_path = folder / 'protocol.json'
            protocol_path.write_text(json.dumps(p))
            sha = contract.digest(protocol_path)
            rows = [dict(row(p, arm), protocol_sha256=sha) for arm in p['arms']]
            rows_path = folder / 'rows.jsonl'
            command = [sys.executable, str(root / 'exp/paper/workflow.py'), 'timing',
                       '--protocol', str(protocol_path), '--rows', str(rows_path)]
            for bad in (False, True):
                out = folder / ('bad' if bad else 'ok')
                if bad:
                    rows[-1]['fallback_count'] = 1
                rows_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                result = subprocess.run(command+['--out', str(out)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2 if bad else 0, result.stderr)
                if bad:
                    self.assertFalse(out.exists())
                else:
                    self.assertEqual(json.loads((out/'summary.json').read_text()),
                                     contract.summarize(rows, p, sha))


if __name__ == '__main__':
    unittest.main()
