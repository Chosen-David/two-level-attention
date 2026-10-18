"""Replay the pre-fix failures or compare actual selector masks to frozen main."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import torch
import test_e151c_prefill_routing as routing
import test_e151c_quota_layer as tests

BASE = '58358135a887031b294f55d67482d1aa43644226'
source = subprocess.check_output(['git', 'show', BASE + ':sparse_attn/indexer/tli_indexer.py'], cwd=ROOT)
old = types.ModuleType('sparse_attn.indexer._layer_baseline')
old.__package__ = 'sparse_attn.indexer'
exec(compile(source, BASE, 'exec'), old.__dict__)

if '--baseline-fail' in sys.argv:
    with patch.object(routing, 'TLIIndexer', old.TLIIndexer):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(tests.QuotaLayerTests))
    sys.exit(0 if result.wasSuccessful() else 1)

torch.set_num_threads(1)
hashes, cases = [], []
with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=''):
    args = routing.make_index().args
    for length in (512, 577):
        gen = torch.Generator().manual_seed(15120 + length)
        q = torch.randn(1, 1, 2, 128, generator=gen)
        k = torch.randn(1, length, 1, 128, generator=gen)
        hashes.append(hashlib.sha256(q.numpy().tobytes() + k.numpy().tobytes()).hexdigest())
        cu = torch.tensor([0, length], dtype=torch.int32)
        for layer in range(3):
            for active in (False, True):
                for skip in (False, True):
                    outputs = []
                    for cls in (old.TLIIndexer, routing.TLIIndexer):
                        idx = cls(args)
                        idx.layer_idx = layer
                        idx._e151c_quota = [0, 32, 128]
                        idx._e151c_on = active
                        idx.skip_far = skip
                        outputs.append(idx.prepare_mask(q, cu[1:] - 1, k, cu, 128 ** -.5)[0])
                    if not torch.equal(*outputs):
                        raise AssertionError((length, layer, active, skip))
                    cases.append([length, layer, active, skip])
print(json.dumps(dict(baseline=BASE, baseline_sha256=hashlib.sha256(source).hexdigest(),
                      input_sha256=hashes, cases=cases, exact_mask_parity=True,
                      torch=torch.__version__, cuda=torch.cuda.is_available()), indent=2))
