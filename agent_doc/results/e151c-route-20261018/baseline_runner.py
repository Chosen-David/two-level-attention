"""Run final regression tests against original modules, without changing files."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sparse_attn.ops import batch_select, reference_prefill

BASE = '44d09d7781d235db94d7ec5ffe5a90b529221af3'
for module, path in ((batch_select, 'sparse_attn/ops/batch_select.py'),
                     (reference_prefill, 'sparse_attn/ops/reference_prefill.py')):
    source = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
    exec(compile(source, f'{BASE}:{path}', 'exec'), module.__dict__)
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), 'test_e151c_prefill_routing.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
