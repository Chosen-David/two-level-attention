"""CPU regressions for the active B-track adapter, not the parallel GPT adapter.

Execute the actual wrapper and upstream SparQ/RunningVMean on CPU. Qwen's
projection, rotary, native prefill and DynamicCache are small test doubles;
these tests do not validate pretrained HF, CUDA or benchmark speed.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import weakref

import torch


ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / 'exp/vendor_baselines/sparq_vendor'


@contextmanager
def module_overrides(replacements):
    # Restore only our aliases. patch.dict(sys.modules) restores the *whole*
    # mapping, unloading lazy imports while leaving their C++ registrations live
    # (the upstream @torch.compile decorator imports Dynamo even on CPU).
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_core():
    spec = importlib.util.spec_from_file_location('sparq_lifecycle_core', VENDOR / 'sparq_attention_official.py')
    core = importlib.util.module_from_spec(spec)
    cache_spec = importlib.util.spec_from_file_location('kv_cache', VENDOR / 'kv_cache.py')
    cache = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache)
    with module_overrides({'kv_cache': cache, spec.name: core}):
        spec.loader.exec_module(core)
    return core


CORE = load_core()


def repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


def load_wrapper():
    source = VENDOR / 'qwen3_sparq_patch.py'
    native = types.ModuleType('transformers.models.qwen3.modeling_qwen3')
    native.Qwen3Attention = type('Qwen3Attention', (), {})
    native.apply_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    native.repeat_kv = repeat_kv
    spec = importlib.util.spec_from_file_location('sparq_lifecycle_adapter', source)
    adapter = importlib.util.module_from_spec(spec)
    with module_overrides({native.__name__: native, 'sparq_attention_official': CORE}):
        spec.loader.exec_module(adapter)
    return adapter._qwen3_sparq_wrapper


class Cache:
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v, layer, kwargs):
        self.k = k if self.k is None else torch.cat((self.k, k), dim=-2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=-2)
        return self.k, self.v


class Attention:
    def __init__(self, groups=2):
        self.head_dim = 4
        self.layer_idx = 0
        self.num_key_value_groups = groups
        self.q_norm = self.k_norm = self.k_proj = self.v_proj = self.o_proj = lambda x: x
        self.q_proj = lambda x: x.repeat(1, 1, groups)
        self.sparq_vmean = CORE.RunningVMean(1, 1, 4)
        self.sparq_k = 2
        self.sparq_r = 2
        self.sparq_config = CORE.SparQArgs()

    def flash_forward(self, hidden, position_embeddings, attention_mask, cache, cache_position, **kwargs):
        kv = hidden.unsqueeze(1)
        cache.update(kv, kv, 0, {})
        return hidden, None


class SparQRequestLifecycle(unittest.TestCase):
    def setUp(self):
        self.forward = load_wrapper()

    def call(self, attention, values, cache):
        return self.forward(attention, values, (None, None), past_key_values=cache)[0]

    def values(self, length, offset):
        return torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4) / 16 + offset

    def assert_mean(self, attention, cache):
        torch.testing.assert_close(attention.sparq_vmean.V_mean, cache.v.mean(-2, keepdim=True))
        self.assertEqual(attention.sparq_vmean.n.item(), cache.v.shape[-2])

    def test_second_request_matches_fresh_adapter(self):
        reused, fresh = Attention(), Attention()
        first, second, oracle = Cache(), Cache(), Cache()
        self.call(reused, self.values(5, 10), first)
        self.call(reused, self.values(1, 10), first)
        for attn, cache in ((reused, second), (fresh, oracle)):
            self.call(attn, self.values(5, -2), cache)
        actual = self.call(reused, self.values(1, -1), second)
        expected = self.call(fresh, self.values(1, -1), oracle)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_mean(reused, second)

    def test_interleaved_caches_rebuild_mean(self):
        attn, a, b = Attention(), Cache(), Cache()
        self.call(attn, self.values(5, 8), a)
        self.call(attn, self.values(1, 9), a)
        self.call(attn, self.values(5, -3), b)
        self.call(attn, self.values(1, -2), b)
        self.call(attn, self.values(1, 10), a)
        self.assert_mean(attn, a)

    def test_same_cache_refill_invalidates_length_collision(self):
        attn, cache = Attention(), Cache()
        self.call(attn, self.values(5, 10), cache)
        self.call(attn, self.values(1, 10), cache)
        cache.k = cache.v = None
        # Previous mean length=6, new decode length=7: length alone aliases.
        self.call(attn, self.values(6, -5), cache)
        self.call(attn, self.values(1, -4), cache)
        self.assert_mean(attn, cache)

    def test_continuation_uses_running_update_without_tensor_item(self):
        attn, cache = Attention(), Cache()
        self.call(attn, self.values(5, 2), cache)
        # Only the wrapper's scalar decision is forbidden; upstream arithmetic
        # stays real. CPU proof of no Tensor.item(), not a CUDA timing result.
        # This host-decision test executes the original softmax arithmetic,
        # without its optional compiler wrapper. Dynamo itself may inspect the
        # mocked Tensor.item or call it while constructing fake scalar tensors;
        # those compiler actions are outside the adapter's host-decision scope.
        # Keep the item trap active across ALL adapter/upstream arithmetic.
        softmax = CORE._scaled_softmax
        eager_softmax = getattr(softmax, '_torchdynamo_orig_callable', softmax)
        with patch.object(CORE, '_scaled_softmax', eager_softmax), patch.object(
                torch.Tensor, 'item', side_effect=AssertionError('host scalar extraction')):
            self.call(attn, self.values(1, 3), cache)
            with patch.object(attn.sparq_vmean, 'init', side_effect=AssertionError('unexpected full rebuild')):
                self.call(attn, self.values(1, 4), cache)
        self.assert_mean(attn, cache)

    def test_no_cache_calls_rebuild_independent_values(self):
        attn = Attention()
        for offset in (10, -3, 2):
            values = self.values(1, offset)
            self.call(attn, values, None)
            torch.testing.assert_close(attn.sparq_vmean.V_mean, values.unsqueeze(1))
            self.assertEqual(attn.sparq_vmean.n.item(), 1)

    def test_owner_does_not_keep_cache_alive(self):
        attn, cache = Attention(), Cache()
        self.call(attn, self.values(3, 1), cache)
        self.call(attn, self.values(1, 2), cache)
        owner = weakref.ref(cache)
        del cache
        self.assertIsNone(owner())

    def test_cache_crop_rebuilds_mean(self):
        attn, cache = Attention(), Cache()
        self.call(attn, self.values(5, 1), cache)
        self.call(attn, self.values(1, 2), cache)
        cache.k, cache.v = cache.k[..., :2, :], cache.v[..., :2, :]
        self.call(attn, self.values(1, 3), cache)
        self.assert_mean(attn, cache)

    def test_nonweakref_cache_uses_safe_rebuild(self):
        class SlottedCache:
            __slots__ = ('k', 'v')
            update = Cache.update
            def __init__(self):
                self.k = self.v = None
        attn, cache = Attention(), SlottedCache()
        self.call(attn, self.values(5, 2), cache)
        for offset in (-1, 3):
            with patch.object(attn.sparq_vmean, 'update', side_effect=AssertionError('untracked owner')):
                self.call(attn, self.values(1, offset), cache)
            self.assert_mean(attn, cache)

    def test_failed_prefill_invalidates_previous_mean(self):
        attn, cache = Attention(), Cache()
        self.call(attn, self.values(5, 1), cache)
        self.call(attn, self.values(1, 2), cache)
        with patch.object(attn, 'flash_forward', side_effect=RuntimeError('prefill failed')):
            with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
                self.call(attn, self.values(2, 3), cache)
        self.assertIsNone(attn._sparq_mean_cache)
        with patch.object(attn.sparq_vmean, 'update', side_effect=AssertionError('invalid state reused')):
            self.call(attn, self.values(1, 4), cache)
        self.assert_mean(attn, cache)

    def test_normal_request_matches_official_incremental_reference(self):
        # Independent reference uses the official arithmetic and selector,
        # deliberately not the adapter's ownership/length decision.
        for groups in (1, 2, 4):
            with self.subTest(groups=groups):
                attn, cache = Attention(groups), Cache()
                reference = CORE.RunningVMean(1, 1, 4)
                self.call(attn, self.values(5, 2), cache)
                for step in range(4):
                    values = self.values(1, step + 3)
                    actual = self.call(attn, values, cache)
                    if step == 0:
                        reference.init(cache.v)
                    else:
                        reference.update(values.unsqueeze(1))
                    query = repeat_kv(values.unsqueeze(1), groups)
                    key, value = repeat_kv(cache.k, groups), repeat_kv(cache.v, groups)
                    expected = CORE.sparq_attn(
                        query, key, key, value, repeat_kv(reference.V_mean, groups),
                        torch.ones(1, 1, 1, key.shape[-2], dtype=torch.bool),
                        attn.sparq_r, attn.sparq_k, attn.sparq_config,
                    ).transpose(1, 2).contiguous().reshape(1, 1, -1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(attn.sparq_vmean.V_mean, reference.V_mean, rtol=0, atol=0)

    def test_second_request_dtypes_and_strided_input(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for groups in (1, 2, 4):
                with self.subTest(dtype=dtype, groups=groups):
                    reused, fresh = Attention(groups), Attention(groups)
                    old, new, oracle = Cache(), Cache(), Cache()
                    def data(length, offset):
                        wide = torch.arange(length * 8, dtype=torch.float32).reshape(1, length, 8)
                        return (wide / 16 + offset).to(dtype)[..., ::2]
                    self.call(reused, data(5, 10), old)
                    self.call(reused, data(1, 11), old)
                    for attn, cache in ((reused, new), (fresh, oracle)):
                        self.call(attn, data(3, -2), cache)
                    actual = self.call(reused, data(1, -1), new)
                    expected = self.call(fresh, data(1, -1), oracle)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(reused.sparq_vmean.V_mean, fresh.sparq_vmean.V_mean, rtol=0, atol=0)

    def test_single_token_new_request_reinitializes(self):
        attn, old, new = Attention(), Cache(), Cache()
        self.call(attn, self.values(5, 10), old)
        self.call(attn, self.values(1, 11), old)
        self.call(attn, self.values(1, -3), new)
        self.assert_mean(attn, new)

    def test_module_overrides_restore_aliases_but_keep_lazy_imports(self):
        existing = '_sparq_fixture_existing'
        added = '_sparq_fixture_added'
        lazy = '_sparq_fixture_lazy'
        original = object()
        sys.modules[existing] = original
        try:
            with self.assertRaisesRegex(RuntimeError, 'fixture failed'):
                with module_overrides({existing: object(), added: object()}):
                    sys.modules[lazy] = object()  # Simulates a new framework import.
                    raise RuntimeError('fixture failed')
            self.assertIs(sys.modules[existing], original)
            self.assertNotIn(added, sys.modules)
            self.assertIn(lazy, sys.modules)
            # The real lazy-import consumer must also remain usable after
            # loading the official core and adapter, without runner preloading.
            self.assertEqual(torch.manual_seed(142).initial_seed(), 142)
        finally:
            for name in (existing, added, lazy):
                sys.modules.pop(name, None)


if __name__ == '__main__':
    unittest.main()
