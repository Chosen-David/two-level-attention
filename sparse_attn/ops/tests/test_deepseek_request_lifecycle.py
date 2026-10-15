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

    def test_reused_attention_resets_for_single_token_new_request(self):
        first = CacheDouble()
        self.call(torch.randn(1, 3, 8), first)
        self.call(torch.randn(1, 1, 8), first, output_attentions=False)
        self.assertEqual(self.attn.indexer.request_state, "current decode")
        self.call(torch.randn(1, 1, 8), CacheDouble())
        self.assertEqual(self.attn.indexer.clears, 2)
        self.assertEqual(self.attn.indexer.selects, 1)
        self.assertIsNone(self.attn.indexer.request_state)


if __name__ == "__main__":
    unittest.main()
