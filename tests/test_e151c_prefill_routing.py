"""E151c routing contracts. CPU selector tests are not CUDA kernel acceptance."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sparse_attn.indexer import TLIIndexer
from sparse_attn.ops import batch_select, eager_prefill, reference_prefill


class CudaMetadata:
    """Only for the pure metadata gate; never passed to tensor computations."""
    shape = (1, 512, 2, 128)
    is_cuda = True

    def dim(self):
        return 4


def gate_index():
    return SimpleNamespace(
        args=SimpleNamespace(tia_block_size=64), _triton_ok=True,
        far_select='4bit', near_select='4bit', alpha=.5, beta=.5, gamma=.5,
        per_q_head=False, sigma_select='none', moba_gate=False,
        sinkswa_only=False, enable_async=False, _basis=None, _basis_all=None,
        l2_subspace='same', enable_subspace=True, subspace='tail',
        static_pair=False, sink_tokens=64, sliding_window_size=64,
    )


def make_index():
    return TLIIndexer(SimpleNamespace(
        tia_block_size=64, tia_level1_topk=8, tia_level2_topk=256,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_layer_skip=False, tli_enable_kmeans=False,
        tli_subspace='tail', tli_sink_tokens=64, tli_swa_size=64,
        tli_alpha=.5, tli_beta=.5, tli_gamma=.5,
    ))


class QuotaRouting(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.env = patch.dict(os.environ, {
            'TLI_E151C_ORACLE_QUOTA': '', 'TLI_PREFILL_RUNTIME': 'candidate',
            'TLI_PREFILL_SELECT_GROUP': '0', 'TLI_PREFILL_PREFETCH': '0',
            'TLI_PREFILL_BATCH_SELECT': '0', 'TLI_PREFILL_FUSED_ATTN': '0',
            'TLI_PREFILL_SELECT_GRAPH': '0',
        })
        self.env.start()
        gen = torch.Generator().manual_seed(15118)
        self.q = torch.randn(1, 512, 2, 128, generator=gen)
        self.k = torch.randn(1, 512, 1, 128, generator=gen)
        self.v = torch.randn(1, 512, 1, 128, generator=gen)

    def tearDown(self):
        self.env.stop()
        torch.set_num_threads(self.threads)

    def test_gate_preserves_absent_and_none_quota(self):
        idx = gate_index()
        self.assertTrue(batch_select._gate_ok(idx, CudaMetadata(), CudaMetadata()))
        idx._e151c_quota = None
        self.assertTrue(batch_select._gate_ok(idx, CudaMetadata(), CudaMetadata()))

    def test_gate_rejects_configured_quota_even_before_flag_activation(self):
        for quota in ([0], [32], []):
            for active in (False, True):
                with self.subTest(quota=quota, active=active):
                    idx = gate_index()
                    idx._e151c_quota, idx._e151c_on = quota, active
                    self.assertFalse(batch_select._gate_ok(idx, CudaMetadata(), CudaMetadata()))

    def test_required_graph_rejects_quota_before_tensor_work(self):
        idx = gate_index()
        idx._e151c_quota = [32]
        with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='1'):
            with self.assertRaisesRegex(RuntimeError, 'unsupported selector'):
                batch_select.batched_group_sels(
                    idx, CudaMetadata(), CudaMetadata(), 1., [], 0,
                    require_graph=True)

    def run_real(self, runtime, quota, grouped=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quota.json'
            path.write_text(json.dumps([quota]))
            with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=str(path)):
                idx = make_index()
        idx.layer_idx = 0
        masks = []
        original = idx.prepare_mask

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            masks.append((args[2].shape[1], result[0].clone()))
            return result

        with patch.object(idx, 'prepare_mask', side_effect=record), patch.dict(
                os.environ, TLI_PREFILL_RUNTIME=runtime,
                TLI_PREFILL_SELECT_GROUP='64' if grouped else '0',
                TLI_PREFILL_BATCH_SELECT='1' if grouped else '0'):
            out = eager_prefill.sparse_prefill_attn(
                idx, self.q, self.k, self.v, dense_below=256 if grouped else 0)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(idx._e151c_on)
        self.assertEqual(masks[-1][0], 512)
        mask = masks[-1][1]
        # S=512, sink/swa=64, alpha=.5 -> mid=384, far=[64,256).
        self.assertEqual(mask[..., 64:256].sum().item(), quota)
        self.assertEqual(mask.sum().item(), 256)
        self.assertTrue(mask[..., :64].all())
        self.assertTrue(mask[..., 448:512].all())
        # Same selector after prefill: the decode flag must be off again.
        cu = torch.tensor([0, 512], dtype=torch.int32)
        decoded, _ = original(self.q[:, -1:], cu[1:]-1, self.k, cu, 128**-.5)
        self.assertEqual(decoded[..., 64:256].sum().item(), 0)
        return out, mask

    def test_reference_real_quota_and_decode_handoff(self):
        for quota in (0, 32, 128):
            with self.subTest(quota=quota):
                self.run_real('reference', quota)

    def test_candidate_real_quota_and_decode_handoff(self):
        for quota in (0, 32, 128):
            with self.subTest(quota=quota):
                self.run_real('candidate', quota)

    def test_grouped_candidate_cpu_fallback_keeps_quota(self):
        self.run_real('candidate', 32, grouped=True)

    def test_reference_and_candidate_same_selection_and_output(self):
        ref, ref_mask = self.run_real('reference', 32)
        cand, cand_mask = self.run_real('candidate', 32)
        self.assertTrue(torch.equal(ref_mask, cand_mask))
        torch.testing.assert_close(ref, cand, rtol=1e-5, atol=1e-6)

    def test_unconfigured_real_selector_keeps_existing_output(self):
        outputs = []
        for runtime in ('reference', 'candidate'):
            idx = make_index()
            self.assertIsNone(idx._e151c_quota)
            with patch.dict(os.environ, TLI_PREFILL_RUNTIME=runtime):
                outputs.append(eager_prefill.sparse_prefill_attn(
                    idx, self.q, self.k, self.v, dense_below=0))
            self.assertFalse(idx._e151c_on)
        torch.testing.assert_close(*outputs, rtol=1e-5, atol=1e-6)

    def test_empty_and_dense_inputs_do_not_activate_selector(self):
        for module in (eager_prefill, reference_prefill):
            for length in (0, 1, 17):
                idx = make_index()
                idx._e151c_quota = [32]
                with patch.object(idx, 'prepare_mask', side_effect=AssertionError('unexpected selector')):
                    out = module.sparse_prefill_attn(
                        idx, self.q[:, :length], self.k[:, :length], self.v[:, :length],
                        dense_below=1024)
                self.assertEqual(out.shape, self.q[:, :length].shape)
                self.assertTrue(torch.isfinite(out).all())
                self.assertFalse(idx._e151c_on)

    def test_exception_clears_prefill_flag(self):
        for module in (eager_prefill, reference_prefill):
            with self.subTest(module=module.__name__):
                idx = make_index()
                idx._e151c_quota = [32]
                active = []

                def fail(*args):
                    active.append(idx._e151c_on)
                    raise RuntimeError('selector failure')

                with patch.object(idx, 'prepare_mask', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'selector failure'):
                        module.sparse_prefill_attn(idx, self.q, self.k, self.v, dense_below=0)
                self.assertEqual(active, [True])
                self.assertFalse(idx._e151c_on)


if __name__ == '__main__':
    unittest.main()
