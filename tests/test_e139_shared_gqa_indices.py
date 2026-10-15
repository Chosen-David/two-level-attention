"""CPU semantic/allocation proxies for shared GQA indices, not CUDA validation."""
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch_select', ROOT / 'sparse_attn/ops/batch_select.py')
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def check_candidates(tokens, valid, width, sink):
    # Independent test oracle; no GPU synchronization introduced in production.
    for row, keep in zip(tokens, valid):
        chosen = row[keep]
        if bool(((chosen < sink) | (chosen >= width)).any()):
            raise ValueError('candidate outside eligible range')
        if chosen.unique().numel() != chosen.numel():
            raise ValueError('duplicate valid candidate')


def oracle(scores, tokens, valid, width):
    """Per-head scatter using independent repetition/indexing, no GQA reshape."""
    heads = scores.shape[0]
    group = heads // tokens.shape[0]
    out = scores.new_full((1, 1, heads, width), float('-inf'))
    for head in range(heads):
        kv = head // group
        idx = torch.where(valid[kv], tokens[kv], 0)
        src = torch.where(valid[kv], scores[head], float('-inf'))
        out[0, 0, head].scatter_(0, idx, src)
    return out


class Index:
    _TILE_FINE = {32: (64, 4)}
    _TILE_COARSE = {32: (64, 4)}
    _triton_ok = True
    def __init__(self, *, alpha=.25, sink=64, swa=64, k1=4, k2=256):
        from types import SimpleNamespace
        self.args = SimpleNamespace(tia_block_size=64, tia_level1_topk=k1,
                                    tia_level2_topk=k2)
        self.alpha, self.beta, self.gamma = alpha, .5, .5
        self.sink_tokens, self.sliding_window_size = sink, swa
        self._sink_pool_lo_blk = sink // 64
        self.far_method, self.near_method = 'minmax', 'avg'
    def prepare_index(self, k, cu):
        x = k[..., :32].contiguous()
        p = x.reshape(1, k.shape[1] // 64, 64, k.shape[2], 32)
        return dict(k_min=p.amin(2), k_max=p.amax(2), k_avg=p.mean(2), k_qat=x)
    def _subspace_indices(self, dev): return torch.arange(32, device=dev)
    def _fast_score_fine(self, k, q, dtype):
        hkv, dim = k.shape[1:]
        return torch.einsum('thd,hgd->hgt', k.float(), q.reshape(hkv, -1, dim)).reshape(q.shape[0], -1).to(dtype)
    def _fast_score_coarse(self, lo, hi, q, dtype):
        return (self._fast_score_fine(hi, q.clamp(min=0), dtype)
                + self._fast_score_fine(lo, q.clamp(max=0), dtype))


class SharedIndices(unittest.TestCase):
    def assertBits(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def test_scatter_softmax_gqa_and_ties(self):
        torch.manual_seed(214)
        for dtype in (torch.float32, torch.bfloat16):
            for width, count in ((1, 0), (17, 8), (64, 16)):
                for empty in (False, True):
                    # Each KV pool has unique eligible candidates; padding can
                    # have duplicate/out-of-range IDs because it is remapped.
                    tokens = torch.arange(1, count + 1).repeat(2, 1)
                    valid = torch.ones_like(tokens, dtype=torch.bool)
                    if count:
                        valid[:, -2:] = False
                        tokens[:, -2:] = width + 10
                    if empty: valid.zero_()
                    scores = torch.randn(8, count).to(dtype)
                    if count:
                        scores[:, 0] = -120  # candidate probability underflow
                        scores[:, 1] = 0
                    check_candidates(tokens, valid, width, 1)
                    expected = oracle(scores, tokens, valid, width)
                    outputs = []
                    for enabled in (False, True):
                        out = torch.full_like(expected, float('-inf'))
                        bs._scatter_candidate_scores(out, scores, tokens, valid, enabled)
                        self.assertBits(out, expected)
                        probs = out.softmax(-1).nan_to_num(nan=0)
                        probs = probs.view(1, 1, 2, 4, width).mean(-2)
                        outputs.append(probs)
                    self.assertBits(*outputs)
                    # Exact probability tensor/order is preserved, including
                    # candidate-zero vs padding-zero ties on this CPU backend.
                    if count and not empty:
                        self.assertEqual(float(outputs[0][0, 0, 0, 1]), 0.)
                    for k in (0, 1, width):
                        self.assertTrue(torch.equal(outputs[0].topk(k).indices,
                                                    outputs[1].topk(k).indices))

    def test_noncontiguous_layout_and_schema_rejections(self):
        scores = torch.randn(6, 8).t()  # [8,6], noncontiguous
        tokens = torch.arange(1, 13).reshape(6, 2).t()
        valid = torch.ones(6, 2, dtype=torch.bool).t()
        # Destination is deliberately strided along history; head dimensions
        # can still be split into [Hkv,G] without changing the storage.
        buf = torch.full((1, 1, 8, 32), float('-inf'))[..., ::2]
        self.assertFalse(buf.is_contiguous())
        bs._scatter_candidate_scores(buf, scores, tokens, valid, True)
        self.assertBits(buf, oracle(scores, tokens, valid, 16))
        bad = [(scores, tokens.float(), valid),
               (scores, tokens, valid.long()),
               (scores[:7], tokens, valid),
               (scores[:, :5], tokens, valid),
               (scores.double(), tokens, valid),
               (scores.to('meta'), tokens, valid)]
        for args in bad:
            with self.assertRaises(ValueError):
                bs._scatter_candidate_scores(buf, *args, True)

    def test_candidate_invariant_oracle_negative_controls(self):
        valid = torch.ones(1, 2, dtype=torch.bool)
        for tokens in (torch.tensor([[1, 1]]), torch.tensor([[0, 2]]),
                       torch.tensor([[-1, 2]]), torch.tensor([[1, 16]])):
            with self.assertRaises(ValueError): check_candidates(tokens, valid, 16, 1)
        check_candidates(torch.tensor([[1, 900]]), torch.tensor([[True, False]]), 16, 1)

    def test_actual_selector_boundaries_and_switch(self):
        torch.manual_seed(512)
        cases = [
            ('normal', 512, {}), ('empty_far', 512, {'alpha': 1.}),
            ('empty_near', 512, {'alpha': .125, 'swa': 0}),
            ('short_overlap', 64, {}), ('protected_only', 128, {}),
            ('k_clipped', 192, {'k1': 64, 'k2': 4096}),
            ('candidate_shortage', 512, {'k1': 2, 'k2': 512}),
            ('below_protected_budget', 256, {'k2': 32}),
            ('no_sink_fallback', 512, {'sink': 0}),
            ('single_kv_fallback', 512, {'test_hkv': 1}),
            ('no_gqa_fallback', 512, {'test_group': 1})]
        for name, length, config in cases:
            config = dict(config)
            hkv = config.pop('test_hkv', 2)
            group = config.pop('test_group', 4)
            for cand in ('0', '1'):
                q = torch.randn(1, length, hkv*group, 128)
                k = torch.randn(1, length, hkv, 128)
                seen = []
                original = bs._scatter_candidate_scores
                def recording(buf, scores, tokens, valid, shared_indices=False):
                    seen.append(shared_indices)
                    if shared_indices:
                        check_candidates(tokens, valid, buf.shape[-1], config.get('sink', 64))
                    return original(buf, scores, tokens, valid, shared_indices)
                with self.subTest(case=name, candidate_topk=cand), patch.object(bs, '_gate_ok', return_value=True), patch.object(bs, '_scatter_candidate_scores', side_effect=recording):
                    outputs = []
                    for shared in ('0', '1'):
                        with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='0', TLI_PREFILL_L2_TOPK_CAND=cand, TLI_PREFILL_SHARED_GQA_INDICES=shared):
                            outputs.append(bs.batched_group_sels(Index(**config), q, k, 128**-.5, [length], 0))
                    self.assertIsNotNone(outputs[0]); self.assertIsNotNone(outputs[1])
                    self.assertBits(outputs[0][0], outputs[1][0])
                    self.assertEqual(seen, [False, config.get('sink', 64) >= 64 and hkv > 1 and group > 1])
                    mask = outputs[1][0]
                    self.assertEqual(tuple(mask.shape), (hkv, length))
                    sink, swa = config.get('sink', 64), config.get('swa', 64)
                    self.assertTrue(bool(mask[:, :sink].all()))
                    if swa: self.assertTrue(bool(mask[:, max(0, length-swa):].all()))
                    protected = min(length, sink + swa)
                    self.assertLessEqual(int(mask.sum(-1).max()), max(protected, min(length, config.get('k2', 256))))

    def test_graph_key_binds_switch(self):
        from types import SimpleNamespace
        from contextlib import nullcontext
        keys = []
        def run(key, body, inputs):
            keys.append(key)
            return body(**inputs)
        q, k = torch.randn(1, 256, 8, 128), torch.randn(1, 256, 2, 128)
        with patch.object(bs, '_gate_ok', return_value=True), patch.object(bs, '_graph_cached_run', side_effect=run), patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=1)), patch.object(torch.cuda, 'device', return_value=nullcontext()):
            for mode in ('0', '1', '0'):
                with patch.dict(os.environ, TLI_PREFILL_SELECT_GRAPH='1', TLI_PREFILL_L2_TOPK_CAND='0', TLI_PREFILL_SHARED_GQA_INDICES=mode):
                    self.assertIsNotNone(bs.batched_group_sels(Index(), q, k, 128**-.5, [256], 0))
        self.assertNotEqual(keys[0], keys[1]); self.assertEqual(keys[0], keys[2])

    def test_gqa_normalization_negative_control(self):
        weights = torch.tensor([[9., 1., 90.], [1., 4., 5.]], dtype=torch.float64)
        joint = weights.log().softmax(-1).mean(0)
        separate_far = weights[:, :2].log().softmax(-1).mean(0)
        torch.testing.assert_close(joint[:2], torch.tensor([.095, .205], dtype=torch.float64))
        torch.testing.assert_close(separate_far, torch.tensor([.55, .45], dtype=torch.float64))
        self.assertEqual(int(joint[:2].argmax()), 1)
        self.assertEqual(int(separate_far.argmax()), 0)

    def test_cpu_allocation_proxy(self):
        hkv, group, count, width = 2, 4, 128, 512
        tokens = torch.arange(1, count+1).repeat(hkv, 1)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        scores = torch.randn(hkv*group, count)
        allocation = []
        for enabled in (False, True):
            buf = scores.new_full((1, 1, hkv*group, width), float('-inf'))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
                bs._scatter_candidate_scores(buf, scores, tokens, valid, enabled)
            allocation.append(sum(max(0, e.self_cpu_memory_usage) for e in prof.events()))
        # Allocation traffic proxy, NOT peak CUDA memory or a timing speedup.
        self.assertLess(allocation[1], allocation[0])
        print('CPU_ALLOCATION_PROXY ' + json.dumps(dict(Hkv=hkv,G=group,C=count,width=width,
            baseline_positive_self_cpu_bytes=allocation[0], shared_positive_self_cpu_bytes=allocation[1],
            full_history_softmax_unchanged=True, cuda_measured=False)), flush=True)

if __name__ == '__main__': unittest.main()
