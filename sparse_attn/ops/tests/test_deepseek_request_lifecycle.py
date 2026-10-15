"""CPU execution of the real MLA forward; model/cache/indexer are test doubles.

No transformers weights or CUDA required. Identity RoPE isolates host routing
and request lifetime; this does not validate production RoPE or GPU kernels.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest

import torch
import torch.nn.functional as F

from sparse_attn.ops.eager_decoding import eager_decoding_attn


def load_forward():
    root = Path(__file__).resolve().parents[2]
    source = root / "patches/deepseek_v2_attn_patch.py"
    namespace = dict(torch=torch, F=F, Optional=Optional, Cache=object,
                     Indexer=object, eager_decoding_attn=eager_decoding_attn,
                     _module_apply_rope=lambda _: lambda q, k, *args: (q, k))
    for path, names in [(root / "patches/utils.py", {"prepare_seqlens", "prepare_cu_seqlens"}),
                        (source, {"deepseek_v2_attn_forward"})]:
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if getattr(n, "name", None) in names]
        if {n.name for n in nodes} != names:
            raise ValueError(f"missing production definitions: {path}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["deepseek_v2_attn_forward"]


class CacheDouble:
    def __init__(self):
        self.k = self.v = None

    def get_seq_length(self, layer):
        return 0 if self.k is None else self.k.shape[-2]

    def get_usable_length(self, length, layer):
        return self.get_seq_length(layer)

    def update(self, k, v, layer, kwargs):
        if self.k is not None:
            k, v = torch.cat((self.k, k), dim=-2), torch.cat((self.v, v), dim=-2)
        self.k, self.v = k, v
        return k, v


class IndexerDouble:
    def __init__(self):
        self.clears = self.selects = 0
        self.request_state = "previous request"

    def clear(self):
        self.clears += 1
        self.request_state = None

    def prepare_mask(self, q, pos, k, cu, **kwargs):
        self.selects += 1
        self.request_state = "current decode"
        return torch.ones(1, 1, q.shape[2], k.shape[1], dtype=torch.bool), 1


class DeepseekRequestLifecycleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        self.forward = load_forward()
        self.attn = SimpleNamespace(
            q_lora_rank=None, num_heads=2, q_head_dim=192,
            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
            kv_lora_rank=16, layer_idx=0, softmax_scale=192 ** -.5,
            q_proj=torch.nn.Linear(8, 384, bias=False),
            kv_a_proj_with_mqa=torch.nn.Linear(8, 80, bias=False),
            kv_a_layernorm=torch.nn.Identity(),
            kv_b_proj=torch.nn.Linear(16, 512, bias=False),
            rotary_emb=lambda value, seq_len: (None, None),
            o_proj=torch.nn.Linear(256, 8, bias=False), indexer=IndexerDouble())

    def call(self, hidden, cache, output_attentions=True):
        n, q_len = cache.get_seq_length(0), hidden.shape[1]
        # Absolute causal additive mask, including a potentially cached prefix.
        forbidden = torch.arange(n + q_len)[None, :] > (n + torch.arange(q_len))[:, None]
        mask = torch.zeros(q_len, n + q_len).masked_fill(forbidden, float("-inf"))[None, None]
        return self.forward(self.attn, hidden, attention_mask=mask,
                            position_ids=torch.arange(n, n + q_len)[None],
                            past_key_value=cache, use_cache=True,
                            output_attentions=output_attentions)

    def test_new_prompt_clears_before_selection_and_returns_dense_weights(self):
        for length in (1, 3):
            with self.subTest(length=length):
                self.setUp()
                cache = CacheDouble()
                hidden = torch.randn(1, length, 8)
                out, weights, returned_cache = self.call(hidden, cache)
                self.assertEqual(self.attn.indexer.clears, 1)
                self.assertEqual(self.attn.indexer.selects, 0)
                self.assertIsNone(self.attn.indexer.request_state)
                self.assertIs(returned_cache, cache)
                self.assertEqual(weights.shape, (1, 2, length, length))
                q = self.attn.q_proj(hidden).view(1, length, 2, 192).transpose(1, 2)
                logits = (q @ cache.k.transpose(-2, -1)) * self.attn.softmax_scale
                logits.masked_fill_(torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf"))
                expected = (logits.softmax(-1) @ cache.v).transpose(1, 2).reshape(1, length, 256)
                torch.testing.assert_close(out, self.attn.o_proj(expected))

    def test_decode_retains_current_request_state(self):
        for length in (1, 3):
            with self.subTest(length=length):
                self.setUp()
                cache = CacheDouble()
                self.call(torch.randn(1, length, 8), cache)
                self.attn.indexer.request_state = "current prefix"
                self.call(torch.randn(1, 1, 8), cache, output_attentions=False)
                self.assertEqual(cache.get_seq_length(0), length + 1)
                self.assertEqual(self.attn.indexer.clears, 1)
                self.assertEqual(self.attn.indexer.selects, 1)

    def test_empty_prompt_remains_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "q_len=0"):
            self.call(torch.empty(1, 0, 8), CacheDouble())

    def test_rejected_continuation_does_not_append_to_cache(self):
        for batch, length in ((1, 2), (2, 1)):
            with self.subTest(batch=batch, length=length):
                self.setUp()
                cache = CacheDouble()
                prefix = torch.randn(batch, 3, 8)
                self.forward(self.attn, prefix, past_key_value=cache, use_cache=True)
                old_k, old_v = cache.k, cache.v
                counts = self.attn.indexer.clears, self.attn.indexer.selects
                with self.assertRaisesRegex(RuntimeError, "仅支持 batch=1 且 q_len=1"):
                    self.forward(self.attn, torch.randn(batch, length, 8),
                                 past_key_value=cache, use_cache=True)
                self.assertIs(cache.k, old_k)
                self.assertIs(cache.v, old_v)
                self.assertEqual(cache.get_seq_length(0), 3)
                self.assertEqual((self.attn.indexer.clears, self.attn.indexer.selects), counts)

    def test_empty_prompt_rejection_does_not_initialize_cache(self):
        cache = CacheDouble()
        with self.assertRaisesRegex(RuntimeError, "q_len=0"):
            self.call(torch.empty(1, 0, 8), cache)
        self.assertIsNone(cache.k)
        self.assertIsNone(cache.v)
        self.assertEqual(self.attn.indexer.clears, 0)

    def test_retry_after_rejected_chunk_matches_untouched_cache(self):
        prefix, rejected, token = torch.randn(1, 3, 8), torch.randn(1, 2, 8), torch.randn(1, 1, 8)
        actual_cache, expected_cache = CacheDouble(), CacheDouble()
        self.call(prefix, actual_cache)
        self.call(prefix, expected_cache)
        with self.assertRaises(RuntimeError):
            self.call(rejected, actual_cache)
        actual, _, _ = self.call(token, actual_cache, output_attentions=False)
        expected, _, _ = self.call(token, expected_cache, output_attentions=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reused_attention_resets_for_single_token_new_request(self):
        first = CacheDouble()
        self.call(torch.randn(1, 3, 8), first)
        self.call(torch.randn(1, 1, 8), first, output_attentions=False)
        self.assertEqual(self.attn.indexer.request_state, "current decode")
        self.call(torch.randn(1, 1, 8), CacheDouble())
        self.assertEqual(self.attn.indexer.clears, 2)
        self.assertEqual(self.attn.indexer.selects, 1)
        self.assertIsNone(self.attn.indexer.request_state)

    def test_maskless_prefill_matches_explicit_causal_mask(self):
        for length in (1, 3, 5):
            with self.subTest(length=length):
                hidden = torch.randn(1, length, 8)
                expected, weights, _ = self.call(hidden, CacheDouble())
                actual, actual_weights, _ = self.forward(
                    self.attn, hidden, attention_mask=None,
                    position_ids=torch.arange(length)[None],
                    past_key_value=CacheDouble(), use_cache=True,
                    output_attentions=True)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_weights, weights)

    def test_maskless_prefill_cannot_read_replaced_suffix(self):
        hidden = torch.randn(1, 5, 8)
        changed = hidden.clone()
        changed[:, 2:] = 20 * torch.randn_like(changed[:, 2:])
        outputs = []
        for x in (hidden, changed):
            out, _, _ = self.forward(
                self.attn, x, attention_mask=None,
                position_ids=torch.arange(5)[None],
                past_key_value=CacheDouble(), use_cache=True)
            outputs.append(out[:, :2])
        torch.testing.assert_close(*outputs)

    def test_invalid_additive_mask_rejected_before_cache_update(self):
        bad_masks = [torch.ones(1, 3), torch.ones(1, 1, 3, 3, dtype=torch.bool),
                     torch.zeros(1, 1, 3, 2), torch.zeros(1, 2, 3, 3),
                     torch.empty(1, 1, 3, 3, device='meta')]
        for mask in bad_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                cache = CacheDouble()
                with self.assertRaisesRegex(ValueError, 'attention_mask'):
                    self.forward(self.attn, torch.randn(1, 3, 8),
                                 attention_mask=mask,
                                 position_ids=torch.arange(3)[None],
                                 past_key_value=cache, use_cache=True)
                self.assertEqual(cache.get_seq_length(0), 0)

    def test_invalid_decode_mask_leaves_existing_prefix_unchanged(self):
        for mask in (torch.zeros(1, 4), torch.zeros(1, 1, 1, 3),
                     torch.ones(1, 1, 1, 4, dtype=torch.bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                cache = CacheDouble()
                self.call(torch.randn(1, 3, 8), cache)
                prefix = cache.k
                selects = self.attn.indexer.selects
                with self.assertRaisesRegex(ValueError, 'attention_mask'):
                    self.forward(self.attn, torch.randn(1, 1, 8),
                                 attention_mask=mask, position_ids=torch.tensor([[3]]),
                                 past_key_value=cache, use_cache=True)
                self.assertIs(cache.k, prefix)
                self.assertEqual(cache.get_seq_length(0), 3)
                self.assertEqual(self.attn.indexer.selects, selects)

    def test_maskless_prefill_batched_strided_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.setUp()
                for value in vars(self.attn).values():
                    if isinstance(value, torch.nn.Module):
                        value.to(dtype=dtype)
                hidden = torch.randn(2, 5, 16, dtype=dtype)[..., ::2]
                self.assertFalse(hidden.is_contiguous())
                mask = torch.zeros(2, 1, 5, 5, dtype=dtype).masked_fill(
                    torch.ones(5, 5, dtype=torch.bool).triu(1), float('-inf'))
                outputs = []
                for m in (mask, None):
                    out, _, _ = self.forward(
                        self.attn, hidden, attention_mask=m,
                        position_ids=torch.arange(5)[None].expand(2, -1),
                        past_key_value=CacheDouble(), use_cache=True)
                    outputs.append(out)
                torch.testing.assert_close(*outputs)


if __name__ == "__main__":
    unittest.main()
