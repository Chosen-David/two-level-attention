"""Stdlib-only execution of production host guards; not GPU parity tests.

Extract the unchanged function ASTs so these dispatch regressions also run on
audit machines without Torch. Tensor access is a hard failure, not a mock
selector result. Full selector/output parity still requires CUDA validation.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load_host_dispatch():
    source = Path(__file__).resolve().parents[1] / 'batch_select.py'
    names = {'_gate_ok', '_unsupported_selection', 'batched_group_sels'}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('production dispatch functions missing')

    class NoTensorWork:
        def __getattr__(self, name):
            raise AssertionError('unexpected tensor work: ' + name)

    scope = {'os': os, 'torch': NoTensorWork()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class CudaMetadata:
    shape = (1, 256, 2, 128)
    is_cuda = True
    device = 'cuda:metadata-only'
    dtype = 'metadata-only'

    def dim(self):
        return 4


def legacy_index(**overrides):
    values = dict(
        args=SimpleNamespace(tia_block_size=64), _triton_ok=True,
        far_select='4bit', near_select='4bit', alpha=.25, beta=.25, gamma=.5,
        per_q_head=False, sigma_select='none', moba_gate=False,
        sinkswa_only=False, enable_async=False, _basis=None, _basis_all=None,
        l2_subspace='same', enable_subspace=True, subspace='tail',
        static_pair=False, sink_tokens=64, sliding_window_size=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# Zero is an explicit quota, not an unset/default value. Single-field cases
# also protect programmatic instances and future constructor changes.
QUOTAS = (
    dict(nt_near_pages=0), dict(nt_near_pages=2),
    dict(nt_near_tokens=0), dict(nt_near_tokens=32),
    dict(nt_near_len=0), dict(nt_near_len=128),
    dict(nt_near_pages=2, nt_near_tokens=32),
    dict(nt_near_pages=0, nt_near_tokens=0, nt_near_len=128),
)


class ExplicitQuotaGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = load_host_dispatch()

    def test_legacy_absent_and_none_fields_remain_eligible(self):
        for fields in ({}, dict(nt_near_pages=None, nt_near_tokens=None,
                                nt_near_len=None)):
            with self.subTest(fields=fields):
                self.assertTrue(self.batch._gate_ok(
                    legacy_index(**fields), CudaMetadata(), CudaMetadata()))

    def test_each_explicit_quota_is_ineligible(self):
        for fields in QUOTAS:
            with self.subTest(fields=fields):
                self.assertFalse(self.batch._gate_ok(
                    legacy_index(**fields), CudaMetadata(), CudaMetadata()))

    def test_e155d_normalized_geometry_is_ineligible(self):
        # E155d now forbids mixing CLI ratios with geometry, and normalizes
        # numeric geometry to alpha=beta=gamma=1. This is runtime metadata,
        # not a constructor call with mutually exclusive user arguments.
        idx = legacy_index(alpha=1., beta=1., gamma=1.,
                           nt_near_pages=2, nt_near_tokens=16, nt_near_len=128)
        self.assertFalse(self.batch._gate_ok(idx, CudaMetadata(), CudaMetadata()))
        self.assertIsNone(self.batch.batched_group_sels(
            idx, CudaMetadata(), CudaMetadata(), 1., [192, 256], 128))

    def test_public_fallback_precedes_tensor_and_cache_work(self):
        for fields in QUOTAS:
            with self.subTest(fields=fields):
                self.assertIsNone(self.batch.batched_group_sels(
                    legacy_index(**fields), CudaMetadata(), CudaMetadata(),
                    1., [192, 256], 128))

    def test_required_graph_rejects_explicit_quotas(self):
        with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='1'):
            for fields in QUOTAS:
                with self.subTest(fields=fields):
                    with self.assertRaisesRegex(RuntimeError, 'unsupported selector'):
                        self.batch.batched_group_sels(
                            legacy_index(**fields), CudaMetadata(), CudaMetadata(),
                            1., [192, 256], 128, require_graph=True)

    def test_aligned_boundary_and_quantization_flags_are_not_new_gate_rejections(self):
        # Boundary bypass has no partial page here. L2 precision is owned by
        # shared prepare_index, not recomputed by the batch budget formulas.
        # This checks dispatch only; it does not certify GPU parity for flags.
        for bypass in (False, True):
            for quant in (False, True):
                with self.subTest(bypass=bypass, quant=quant):
                    self.assertTrue(self.batch._gate_ok(legacy_index(
                        boundary_bypass=bypass, l2_quant=quant),
                        CudaMetadata(), CudaMetadata()))


if __name__ == '__main__':
    unittest.main()
