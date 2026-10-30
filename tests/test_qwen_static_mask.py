"""CPU mask contracts using the production functions and actual HF cache/SDPA.

AST loading omits unrelated indexer imports (which require CUDA dependencies).
No model weights are downloaded; this is not GPU, model-quality or speed proof.
"""
import ast
import gc
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3Config, apply_rotary_pos_emb, eager_attention_forward,
)
from sparse_attn.baseline_runtime import batch_decode_ctx

SOURCE = Path(__file__).resolve().parents[1] / 'sparse_attn/patches/qwen3_attn_patch.py'

def load_functions():
    tree = ast.parse(SOURCE.read_text())
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(__name__='sparse_attn.patches.qwen3_attn_patch', __package__='sparse_attn.patches', torch=torch, os=os, sys=sys, weakref=weakref, batch_decode_ctx=batch_decode_ctx,
              apply_rotary_pos_emb=apply_rotary_pos_emb,
              eager_attention_forward=eager_attention_forward,
              ALL_ATTENTION_FUNCTIONS={'sdpa': sdpa_attention_forward})
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'), ns)
    return ns


def causal(length=3, capacity=5):
    m = torch.zeros(1, 1, length, capacity, dtype=torch.bool)
    m[..., :length] = torch.ones(length, length, dtype=torch.bool).tril()
    return m


class StaticMask(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261018)
        self.ns = load_functions()
        self.cache = SimpleNamespace()

    def check(self, mask, length=3, prefill=True):
        return self.ns['_can_drop_static_mask'](self.cache, mask, length, prefill=prefill)

    def forward(self, mask, *, dynamic=False, layer=0, cache=None, module=None, length=3, start=0):
        cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, num_hidden_layers=2)
        cfg._attn_implementation = 'sdpa'
        attn = module or Qwen3Attention(cfg, layer).eval()
        attn.indexer = SimpleNamespace(clear=lambda: None)
        cache = cache if cache is not None else (DynamicCache(config=cfg) if dynamic else StaticCache(cfg, max_cache_len=5))
        x = torch.arange(length * 16, dtype=torch.float32).reshape(1, length, 16) / 48
        observed = []
        def observe(mod, q, k, v, m, **kw):
            observed.append(m)
            return sdpa_attention_forward(mod, q, k, v, m, **kw)
        self.ns['ALL_ATTENTION_FUNCTIONS'] = {'sdpa': observe}
        with torch.no_grad(), patch.dict(os.environ, {'TLI_SPARSE_PREFILL': '0', 'TLI_E134_DEBUG': ''}):
            out, _ = self.ns['qwen3_attn_forward'](
                attn, x, (torch.ones(1, length, 4), torch.zeros(1, length, 4)), mask,
                past_key_values=cache, cache_position=torch.arange(start, start + length))
        return out, observed, cache, attn

    def test_forward_preserves_fully_blocked_rows(self):
        mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
        out, seen, _, _ = self.forward(mask)
        self.assertIs(seen[0], mask)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_forward_dynamic_cache_is_not_static(self):
        _, _, cache, _ = self.forward(None, dynamic=True)
        self.assertEqual(cache.get_max_cache_shape(), -1)
        self.assertFalse(hasattr(cache, '_tli_seq_len'))

    def test_forward_canonical_static_matches_dynamic(self):
        static, seen, _, attn = self.forward(causal())
        dynamic, _, _, _ = self.forward(None, dynamic=True, module=attn)
        self.assertIsNone(seen[0])
        torch.testing.assert_close(static, dynamic, rtol=0, atol=0)

    def test_forward_changed_layer_mask_not_reused(self):
        _, _, cache, _ = self.forward(causal())
        blocked = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
        out, seen, _, _ = self.forward(blocked, layer=1, cache=cache)
        self.assertIs(seen[0], blocked)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_forward_decode_does_not_discard_float_bias(self):
        _, _, cache, attn = self.forward(causal())
        mask = torch.tensor([1., -1., -2., 1., 0.]).reshape(1, 1, 1, 5)
        with self.assertRaisesRegex(RuntimeError, 'attention_mask'):
            self.forward(mask, cache=cache, module=attn, length=1, start=3)

    def test_exhaustive_three_by_three_boolean_masks(self):
        target = causal(3, 3)
        accepted = 0
        for bits in itertools.product((False, True), repeat=9):
            mask = torch.tensor(bits).reshape(1, 1, 3, 3)
            result = self.check(mask)
            self.assertEqual(result, torch.equal(mask, target))
            accepted += result
        self.assertEqual(accepted, 1)

    def test_prefill_validation_uses_bounded_chunks(self):
        m = causal(5, 7)
        original = torch.ones_like
        sizes = []
        def observe(x, *args, **kwargs):
            sizes.append(x.numel())
            return original(x, *args, **kwargs)
        with patch.object(torch, 'ones_like', side_effect=observe):
            self.assertTrue(self.ns['_can_drop_static_mask'](
                self.cache, m, 5, prefill=True, _chunk_elements=12))
        self.assertEqual(len(sizes), 3)
        self.assertLessEqual(max(sizes), 12)
        m[0, 0, 4, 0] = False
        self.assertFalse(self.ns['_can_drop_static_mask'](
            self.cache, m, 5, prefill=True, _chunk_elements=12))

    def test_decode_only_identity_bool(self):
        for dtype in (torch.bool, torch.float32, torch.float16, torch.bfloat16):
            for prefix in ([1, 1, 1], [1, 0, 1], [1, -1, -2]):
                mask = torch.tensor(prefix + [0, 0], dtype=dtype).reshape(1, 1, 1, 5)
                expected = dtype == torch.bool and bool(mask[..., :3].all())
                self.assertEqual(self.check(mask, prefill=False), expected)
        for value in (float('nan'), float('inf')):
            self.assertFalse(self.check(torch.tensor([1, 1, value, 0, 0]).reshape(1, 1, 1, 5), prefill=False))

    def test_tail_shape_and_empty_rejected(self):
        m = causal(); m[..., -1] = True
        self.assertFalse(self.check(m))
        for m, length in ((causal()[0], 3), (causal().expand(2, -1, -1, -1), 3),
                          (causal().expand(-1, 2, -1, -1), 3), (causal()[..., :2], 3),
                          (causal(), 0), (causal(), 4)):
            self.assertFalse(self.check(m, length))

    def test_same_mask_version_reuses_and_mutation_rechecks(self):
        m = causal()
        self.assertTrue(self.check(m))
        with patch.object(torch, 'equal', side_effect=AssertionError('unexpected validation')):
            self.assertTrue(self.check(m))
        m[..., 2, 0] = False
        self.assertFalse(self.check(m))
        m[..., 2, 0] = True
        self.assertTrue(self.check(m))
        self.assertFalse(self.check(torch.zeros_like(m)))

    def test_phase_length_stride_and_alias_mutation(self):
        m = causal()
        self.assertTrue(self.check(m))
        self.assertFalse(self.check(m, prefill=False))
        self.assertFalse(self.check(m, 2))
        view = m.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.assertFalse(view.is_contiguous())
        self.assertTrue(self.check(view))
        view[..., 2, 0] = False
        self.assertFalse(self.check(view))
        m.view(-1)[10] = False
        self.assertFalse(self.check(m))

    def test_inference_tensor_revalidated(self):
        with torch.inference_mode():
            m = causal()
            self.assertTrue(self.check(m))
            m[..., 2, 0] = False
            self.assertFalse(self.check(m))
        self.assertIsNone(self.cache._tli_mask_validation)

    def test_cache_does_not_retain_mask(self):
        m = causal(); ref = weakref.ref(m)
        self.assertTrue(self.check(m))
        del m; gc.collect()
        self.assertIsNone(ref())


if __name__ == '__main__':
    unittest.main()
