"""CPU regression for native HF 4/5 calls through Quest and SparQ adapters."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
for vendor in ('quest_vendor', 'sparq_vendor'):
    sys.path.insert(0, str(ROOT / 'exp/vendor_baselines' / vendor))
from qwen3_quest_patch import enable_quest
from qwen3_sparq_patch import enable_sparq


class QwenVendorAPITests(unittest.TestCase):
    def model(self, method, head_dim=8):
        torch.manual_seed(19)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=head_dim)
        cfg._attn_implementation = 'eager'
        native = Qwen3ForCausalLM(cfg).eval()
        candidate = copy.deepcopy(native)
        if method == 'quest':
            enable_quest(candidate, token_budget=128, chunk_size=1)
        else:
            enable_sparq(candidate, k=128, context_length=256)
        return native, candidate

    def test_real_prefill_decode_dense_parity(self):
        # head_dim=16 also covers Qwen32B-like hidden != heads*head_dim.
        for method in ('quest', 'sparq'):
            for head_dim in (8, 16):
                with self.subTest(method=method, head_dim=head_dim):
                    native, candidate = self.model(method, head_dim)
                    with torch.no_grad():
                        ids = torch.tensor([[1, 2, 3, 4, 5]])
                        a, b = native(ids, use_cache=True), candidate(ids, use_cache=True)
                        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
                        for step in range(3):
                            ids = torch.tensor([[6 + step]])
                            a = native(ids, past_key_values=a.past_key_values, use_cache=True)
                            b = candidate(ids, past_key_values=b.past_key_values, use_cache=True)
                            torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
                        for layer in range(3):
                            self.assertEqual(b.past_key_values.get_seq_length(layer), 8)

    def test_direct_decode_positions_and_masks_still_required(self):
        for method in ('quest', 'sparq'):
            _, model = self.model(method)
            with torch.no_grad():
                out = model(torch.tensor([[1, 2, 3]]), use_cache=True)
            attn = model.model.layers[-1].self_attn
            hidden = torch.zeros(1, 1, 32)
            args = dict(hidden_states=hidden, position_embeddings=(None, None),
                        past_key_values=out.past_key_values)
            for pos in (None, torch.tensor([3]), torch.tensor([[3], [3]]),
                        torch.tensor([[9]]), torch.tensor([[3.0]])):
                with self.subTest(method=method, pos=pos):
                    extra = {} if pos is None else {'position_ids': pos}
                    with self.assertRaises(ValueError):
                        attn(**args, **extra)
            with self.assertRaises(ValueError):
                attn(**args, position_ids=torch.tensor([[3]]),
                     attention_mask=torch.ones(1, 1, 1, 4))
            self.assertEqual(out.past_key_values.get_seq_length(2), 3)

    def test_explicit_cache_position_forwarded_as_keyword(self):
        for method in ('quest', 'sparq'):
            _, model = self.model(method)
            attn = model.model.layers[0].self_attn
            position = torch.tensor([0, 1])
            with patch.object(attn, 'flash_forward', return_value=('ok', None)) as dense:
                self.assertEqual(attn(torch.zeros(1, 2, 32), (None, None),
                                      cache_position=position), ('ok', None))
                self.assertEqual(len(dense.call_args.args), 4)
                self.assertIs(dense.call_args.kwargs['cache_position'], position)


if __name__ == '__main__':
    unittest.main()
