"""Run the quota input regression against the pinned original constructor."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import sparse_attn.indexer as package
from sparse_attn.indexer import tli_indexer

BASE = 'cb0c8f430b9e2595caef48590b01690a99d416b3'
path = 'sparse_attn/indexer/tli_indexer.py'
source = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
exec(compile(source, f'{BASE}:{path}', 'exec'), tli_indexer.__dict__)
package.TLIIndexer = tli_indexer.TLIIndexer
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), 'test_e151c_quota_input.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
