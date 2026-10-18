"""Independent CPU host/glue check; no CUDA execution or tie-equivalence claim."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from sparse_attn.ops import batch_select as current

torch.set_num_threads(1)
baseline = ModuleType('baseline_batch_select')
exec(subprocess.check_output(['git', 'show',
    '4f577242bcd88aa30e2eb25107961030fd3c4aa9:sparse_attn/ops/batch_select.py'],
    cwd=ROOT, text=True), baseline.__dict__)


def fixture(bs, budget):
    length = 1024
    k = torch.zeros(1, length, 2, 128)
    q = torch.zeros(1, length, 4, 128)
    coarse = k[..., :32].reshape(1, length // bs, bs, 2, 32).mean(2)
    idx = SimpleNamespace(
        args=SimpleNamespace(tia_block_size=bs, tia_level1_topk=2,
                             tia_level2_topk=budget),
        sink_tokens=bs, sliding_window_size=bs, _sink_pool_lo_blk=1,
        alpha=.25, beta=.5, gamma=.5, far_method='minmax', near_method='avg',
        _TILE_FINE={}, _TILE_COARSE={}, _triton_ok=True,
        _subspace_indices=lambda dev: torch.arange(32, device=dev),
        prepare_index=lambda *a: dict(k_min=coarse, k_max=coarse,
                                     k_avg=coarse, k_qat=k[..., :32]))
    def score(keys, query, dtype):
        return torch.arange(keys.shape[0], dtype=dtype).expand(query.shape[0], -1) / 100
    idx._fast_score_fine = score
    idx._fast_score_coarse = lambda lo, hi, query, dtype: score(lo, query, dtype)
    return idx, q, k, 128 ** -.5, [512, 1024], 256


def run(module, args, flag, shared='0', graph=False, keys=None):
    def host_executor(key, body, inputs):
        keys.append(key)
        return body(**inputs)
    with patch.object(module, '_gate_ok', return_value=True), patch.dict(os.environ, {
            'TLI_PREFILL_SELECT_GRAPH': '1' if graph else '0',
            'TLI_PREFILL_SHARED_GQA_INDICES': shared,
            'TLI_PREFILL_L2_TOPK_CAND': flag}), patch.object(
            torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=7)), patch.object(
            torch.cuda, 'device', return_value=nullcontext()), patch.object(
            module, '_graph_cached_run', side_effect=host_executor):
        return module.batched_group_sels(*args, require_graph=graph)


class Independent(unittest.TestCase):
    def masks_equal(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertIsNotNone(expected)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))

    def test_budget_boundaries_and_baseline_unchanged_paths(self):
        count = 0
        for bs in (16, 32, 64, 128, 256):
            for budget in (0, 2 * bs, 2 * bs + 1, 512, 2048):
                for shared in ('0', '1'):
                    with self.subTest(bs=bs, budget=budget, shared=shared):
                        args = fixture(bs, budget)
                        old = run(baseline, args, '0', shared)
                        self.masks_equal(run(current, args, '0', shared), old)
                        if bs != 64:
                            self.masks_equal(run(current, args, '1', shared), old)
                        else:
                            self.masks_equal(run(current, args, '1', shared),
                                             run(baseline, args, '1', shared))
                        for mask in old:
                            self.assertTrue(mask[:, :bs].all())
                            self.assertTrue(mask[:, -bs:].all())
                            # Forced regions survive even when K2 is smaller.
                            self.assertTrue((mask.sum(-1) <= max(budget, 2 * bs)).all())
                        count += 1
        print(f'PASS {count} budget/block/shared-index scenarios; baseline differential')

    def test_graph_effective_mode_host_only(self):
        for bs in (16, 32, 64, 128, 256):
            with self.subTest(bs=bs):
                args, keys = fixture(bs, 512), []
                off = run(current, args, '0', graph=True, keys=keys)
                on = run(current, args, '1', graph=True, keys=keys)
                self.assertEqual(keys[0][2][2], False)
                self.assertEqual(keys[1][2][2], bs == 64)
                self.assertEqual(keys[0] == keys[1], bs != 64)
                if bs != 64:
                    self.masks_equal(off, on)
        print('PASS 5 host graph-signature cases; no CUDA capture/replay performed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
