"""Reproduce baseline ABI failures and fixed CPU regressions from repo root."""
import hashlib
import io
import platform
import subprocess
import types
import unittest

import torch
from sparse_attn.ops.tests import test_eager_cu_layout as probe

BASE = 'bb14bc8a0347f1e440c78d397a3e2019e4a16c82'
PATH = 'sparse_attn/ops/eager_decoding.py'
print('baseline', BASE)
print('Python', platform.python_version(), 'torch', torch.__version__,
      'CUDA available', torch.cuda.is_available(), flush=True)
cu = torch.tensor([99, 0, 777, 16384, 888], dtype=torch.int32)[1:4:2]
print('cu logical', cu.tolist(), 'stride', cu.stride(), 'offset', cu.storage_offset())
print('cu logical bytes sha256', hashlib.sha256(cu.numpy().tobytes()).hexdigest())
print('cu physical adjacent', cu.as_strided((2,), (1,)).tolist())
original = probe.ed
source = subprocess.check_output(['git', 'show', f'{BASE}:{PATH}'], text=True)
baseline = types.ModuleType('eager_decoding_baseline')
exec(compile(source, f'{BASE}:{PATH}', 'exec'), baseline.__dict__)
for name, module in [('baseline', baseline), ('patched', original)]:
    probe.ed = module
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(probe))
    print(name, 'tests', result.testsRun, 'errors', len(result.errors),
          'failures', len(result.failures), flush=True)
    print(log.getvalue(), flush=True)
    if name == 'baseline':
        assert result.testsRun == 3 and len(result.errors) == 2 and len(result.failures) == 2
    else:
        assert result.wasSuccessful()
probe.ed = original
