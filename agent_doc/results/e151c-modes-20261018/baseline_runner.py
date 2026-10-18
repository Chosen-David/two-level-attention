"""Reproduce ignored gamma-off quotas and run regressions on pinned main."""
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import torch
import sparse_attn.indexer as package
from sparse_attn.indexer import tli_indexer
from sparse_attn.ops import reference_prefill

BASE = 'b8719fa7023837dc724470eeb23795c5557232c1'
path = 'sparse_attn/indexer/tli_indexer.py'
source = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
exec(compile(source, f'{BASE}:{path}', 'exec'), tli_indexer.__dict__)
package.TLIIndexer = tli_indexer.TLIIndexer
from test_e151c_quota_modes import make_index

torch.set_num_threads(1)
gen = torch.Generator().manual_seed(15119)
q = torch.randn(1, 512, 2, 128, generator=gen)
k = torch.randn(1, 512, 1, 128, generator=gen)
v = torch.randn(1, 512, 1, 128, generator=gen)
digest = hashlib.sha256(b''.join(t.numpy().tobytes() for t in (q, k, v)))
print('input_sha256=' + digest.hexdigest(), flush=True)
masks = []
for quota in (0, 128):
    idx = make_index(dict(tli_gamma=None), [quota])
    idx.layer_idx = 0
    original = idx.prepare_mask

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        masks.append(result[0].clone())
        return result

    idx.prepare_mask = record
    reference_prefill.sparse_prefill_attn(idx, q, k, v, dense_below=0)
    print(f'quota={quota} far={masks[-1][..., 64:256].sum().item()} '
          f'total={masks[-1].sum().item()}', flush=True)
print('quota_masks_equal=' + str(torch.equal(masks[0], masks[1])), flush=True)
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), 'test_e151c_quota_modes.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
