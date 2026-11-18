import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from exp.vendor_baselines.jssa_vendor.qwen3_jssa_nfa_semantic import (
    JSSANFASemanticRuntime,
)
from sparse_attn.indexer.nfa_layer_batch_selector import _nfa_filtered_topk


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class JSSANFASemanticTest(unittest.TestCase):
    def _runtime(self, root, length=400, budget=64):
        device = torch.device("cuda")
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=1024,
            head_dim=128,
        )
        model = SimpleNamespace(
            config=config,
            dtype=torch.bfloat16,
            model=SimpleNamespace(layers=[SimpleNamespace(
                self_attn=SimpleNamespace(sliding_window=None))]),
        )
        basis = torch.randn(2, 32, 128, dtype=torch.bfloat16) * 0.05
        path = Path(root) / "basis.pt"
        torch.save({"M": {0: basis}}, path)
        runtime = JSSANFASemanticRuntime(
            model, batch_size=1, prefill_length=length, max_new_tokens=2,
            device=device, basis_path=str(path), budget=budget, rank=32,
            page_size=64)
        runtime.seq_lens.fill_(length)
        torch.manual_seed(20261118)
        runtime.projk_cache.copy_(
            (torch.randn_like(runtime.projk_cache, dtype=torch.bfloat16) * 0.1)
            .to(torch.float8_e4m3fn))
        runtime.projq_hrc.copy_(
            torch.randn_like(runtime.projq_hrc) * 0.1)
        return runtime

    def _reference_scores(self, runtime, length):
        proj_k = runtime.projk_cache[0].permute(2, 0, 1, 3).reshape(
            runtime.num_kv_heads, runtime.max_kv_len, runtime.rank)
        proj_q = runtime.projq_hrc[:, 0]
        scores = torch.einsum(
            "hlc,hc->hl", proj_k.float(), proj_q.float()).bfloat16()
        scores[:, length:] = -torch.inf
        return scores

    def _reference_topk(self, runtime, length):
        scores = self._reference_scores(runtime, length)
        return torch.topk(scores, runtime.budget, dim=-1).indices

    def test_full_candidate_score_topk_and_graph_replay(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self._runtime(root)
            with torch.inference_mode():
                actual = runtime._score_topk(0)
                expected_scores = self._reference_scores(runtime, 400)
                actual_scores = runtime._nfa_logits[0]
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(
                    actual_scores.cpu(), expected_scores.cpu()))
                self.assertTrue(torch.isinf(
                    actual_scores[:, 400:]).all().item())
                self.assertTrue((actual_scores[:, 400:] < 0).all().item())
                expected = self._reference_topk(runtime, 400)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(
                    actual[0].sort(dim=-1).values.cpu(),
                    expected.sort(dim=-1).values.cpu()))

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    runtime._score_topk(0)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(
                    runtime._nfa_top_idx[0].sort(dim=-1).values.cpu(),
                    expected.sort(dim=-1).values.cpu()))
            self.assertEqual(runtime.receipt()["score_candidate_scope"],
                             "every logical token in [0, seq_len), no L1 pruning")
    def test_dynamic_merge_full_512k_candidates_bit_exact(self):
        device = torch.device("cuda")
        capacity, budget = 524321, 1024
        torch.manual_seed(20261118)
        scores = torch.randn(
            (1, 2, capacity), dtype=torch.bfloat16, device=device)
        candidates = torch.arange(
            capacity, dtype=torch.int32, device=device).view(
                1, 1, capacity).repeat(1, 2, 1)
        lower = torch.zeros(1, dtype=torch.int32, device=device)
        upper = torch.full((1,), capacity, dtype=torch.int32, device=device)
        quota = torch.full((1,), budget, dtype=torch.int32, device=device)
        selection = _nfa_filtered_topk(
            scores, candidates, lower, upper, quota, capacity, budget,
            sink=0, workspace={}, merge_dynamic=True)

        bits = scores.view(torch.int16).to(torch.int32) & 0xFFFF
        negative = (bits & 0x8000) != 0
        ordered = torch.where(
            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
        keys = ((ordered.to(torch.int64) << 32)
                + capacity - candidates.to(torch.int64))
        reference_keys = torch.topk(keys[0], budget, dim=-1).values
        reference = capacity - (reference_keys & 0xFFFFFFFF)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(
            selection.indices[0].sort(dim=-1).values.cpu(),
            reference.sort(dim=-1).values.cpu()))
        self.assertTrue(torch.equal(
            selection.counts.cpu(),
            torch.full((1, 2), budget, dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
