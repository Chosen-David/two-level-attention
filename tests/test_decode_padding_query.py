"""Real HF attention/cache plus production CPU decode; selector is all-selected.

AST loading avoids unrelated CUDA indexer imports, not the forward implementation.
This verifies wrapper packing, not GPU kernels or sparse-selector quality.
"""
import ast
import hashlib
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers.cache_utils import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3 import modeling_qwen3 as qwen
from transformers.models.llama import modeling_llama as llama
from sparse_attn.ops.eager_decoding import eager_decoding_attn

ROOT = Path(__file__).resolve().parents[1]


def functions(path, ns):
    tree = ast.parse(path.read_text())
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)


def load_forward(family):
    model = qwen if family == 'qwen3' else llama
    ns = dict(torch=torch, os=os, sys=sys, weakref=weakref,
              apply_rotary_pos_emb=model.apply_rotary_pos_emb,
              eager_attention_forward=model.eager_attention_forward,
              ALL_ATTENTION_FUNCTIONS={'sdpa': sdpa_attention_forward},
              eager_decoding_attn=eager_decoding_attn)
    functions(ROOT / 'sparse_attn/patches/utils.py', ns)
    functions(ROOT / f'sparse_attn/patches/{family}_attn_patch.py', ns)
    return ns[f'{family}_attn_forward']


class PaddingQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = hashlib.sha256()

    @classmethod
    def tearDownClass(cls):
        print('decode-padding input SHA256:', cls.inputs.hexdigest())

    def scenario(self, family, values, fast='0', strided=False,
                 reject=False, mask_override=None, omit_mask=False,
                 dtype=torch.float32, kv_heads=2):
        torch.manual_seed(20261019)
        config_cls, attn_cls = ((qwen.Qwen3Config, qwen.Qwen3Attention) if family == 'qwen3'
                                else (llama.LlamaConfig, llama.LlamaAttention))
        cfg = config_cls(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                         num_key_value_heads=kv_heads, head_dim=4, num_hidden_layers=2)
        cfg._attn_implementation = 'sdpa'
        attn = attn_cls(cfg, 0).eval().to(dtype)
        selected = []

        def prepare(q, positions, k, cu):
            selected.append((q.shape, positions.tolist(), k.shape, cu.tolist()))
            return torch.ones(1, 1, kv_heads, k.shape[1], dtype=torch.bool), 1

        attn.indexer = SimpleNamespace(clear=lambda: None, prepare_mask=prepare)
        forward = load_forward(family)
        cache, reference_cache = DynamicCache(config=cfg), DynamicCache(config=cfg)
        length = len(values)
        prefix, token = torch.randn(1, length - 1, 16, dtype=dtype), torch.randn(1, 1, 16, dtype=dtype)
        rope_prefix = (torch.ones(1, length - 1, 4, dtype=dtype), torch.zeros(1, length - 1, 4, dtype=dtype))
        rope_token = (torch.ones(1, 1, 4, dtype=dtype), torch.zeros(1, 1, 4, dtype=dtype))
        mask = torch.tensor([values], dtype=torch.bool)
        if strided:
            mask = mask.repeat_interleave(2, dim=-1)[:, ::2]
            self.assertFalse(mask.is_contiguous())
            prefix = prefix.repeat_interleave(2, dim=-1)[..., ::2]
            token = token.repeat_interleave(2, dim=-1)[..., ::2]
            self.assertFalse(token.is_contiguous())
        if mask_override is not None:
            mask = mask_override
        if omit_mask:
            mask = None
        self.inputs.update(repr((family, values, fast, strided, reject, omit_mask, dtype, kv_heads)).encode())
        for tensor in [*attn.state_dict().values(), prefix, token, *rope_prefix, *rope_token,
                       *([] if mask is None else [mask])]:
            self.inputs.update(repr((tuple(tensor.shape), tensor.dtype, tensor.stride())).encode())
            self.inputs.update(tensor.detach().contiguous().numpy().tobytes())
        env = {'TLI_SPARSE_PREFILL': '0', 'TLI_DECODE_DENSE': '0', 'TLI_E127_FAST': fast}
        with torch.no_grad(), patch.dict(os.environ, env):
            forward(attn, prefix, rope_prefix, None, past_key_values=cache,
                    cache_position=torch.arange(length - 1))
            attn(prefix, rope_prefix, None, past_key_values=reference_cache,
                 cache_position=torch.arange(length - 1))
            if reject:
                with self.assertRaisesRegex(RuntimeError, 'attention_mask'):
                    forward(attn, token, rope_token, mask, past_key_values=cache,
                            cache_position=torch.tensor([length - 1]))
                self.assertEqual(selected, [])
                return
            expected, _ = attn(token, rope_token, None if mask is None else mask[:, None, None, :],
                               past_key_values=reference_cache,
                               cache_position=torch.tensor([length - 1]))
            actual, _ = forward(attn, token, rope_token, mask, past_key_values=cache,
                                cache_position=torch.tensor([length - 1]))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(actual.shape), (1, 1, 16))
        kept = sum(values)
        self.assertEqual(selected, [(torch.Size([1, 1, 4, 4]), [kept - 1],
                                     torch.Size([1, kept, kv_heads, 4]), [0, kept])])

    def test_all_valid_mask_does_not_index_single_query_with_kv_length(self):
        for family in ('qwen3', 'llama3'):
            with self.subTest(family=family):
                self.scenario(family, [True, True, True, True])

    def test_padding_patterns_match_dense_masked_attention(self):
        for family, fast, bits in itertools.product(('qwen3', 'llama3'), ('0', '1'),
                                                    itertools.product((False, True), repeat=3)):
            with self.subTest(family=family, fast=fast, bits=bits):
                self.scenario(family, list(bits) + [True], fast, strided=True)

    def test_masked_current_query_rejected_before_selector(self):
        for family, bits in itertools.product(('qwen3', 'llama3'),
                                               itertools.product((False, True), repeat=3)):
            with self.subTest(family=family, bits=bits):
                self.scenario(family, list(bits) + [False], reject=True)

    def test_mask_shape_rejected_before_selector(self):
        for family, shape in itertools.product(('qwen3', 'llama3'),
                                               ((1, 0), (1, 3), (1, 5), (2, 4), (4,))):
            with self.subTest(family=family, shape=shape):
                self.scenario(family, [True] * 4, reject=True,
                              mask_override=torch.ones(shape, dtype=torch.bool))

    def test_none_mask_path_matches_hf(self):
        for family, fast in itertools.product(('qwen3', 'llama3'), ('0', '1')):
            with self.subTest(family=family, fast=fast):
                self.scenario(family, [True] * 4, fast, omit_mask=True)

    def test_short_tail_and_head_layouts(self):
        for family, length, dtype, kv_heads, fast in itertools.product(
                ('qwen3', 'llama3'), (2, 3, 7, 65), (torch.float32, torch.float64), (1, 4), ('0', '1')):
            with self.subTest(family=family, length=length, dtype=dtype, kv_heads=kv_heads, fast=fast):
                values = [i % 3 != 0 for i in range(length - 1)] + [True]
                self.scenario(family, values, fast, strided=True, dtype=dtype, kv_heads=kv_heads)


if __name__ == '__main__':
    unittest.main()
