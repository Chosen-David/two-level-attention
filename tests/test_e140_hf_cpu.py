"""Offline random-weight HF Qwen3 CPU parity; never downloads weights or uses CUDA."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as hf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'agent_doc/guide/paper_figs/trace/e140_o_trace.py'
spec = importlib.util.spec_from_file_location('e140_trace_hf_test', SCRIPT)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def digest_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class HFObservation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(140)
        config = Qwen3Config(vocab_size=32, hidden_size=128, intermediate_size=192,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=128,
                            rope_theta=12345.0, attention_dropout=0.0)
        config._attn_implementation = 'eager'
        self.model = Qwen3ForCausalLM(config).eval().cpu()
        # Nonuniform LN weights expose omission of input_layernorm; a scalar
        # normalization can otherwise be approximately cancelled by q/k RMSNorm.
        with torch.no_grad():
            for layer in self.model.model.layers:
                layer.input_layernorm.weight.copy_(torch.linspace(.3, 1.7, 128))
        self.ids = torch.arange(641).remainder(32).unsqueeze(0)

    def collect(self, ids=None, layers=None):
        with patch.object(trace, 'LAYERS', [0, 1] if layers is None else layers), patch.object(trace, 'DEV', 'cpu'):
            # Deliberately incorrect theta: the collector must use actual supplied RoPE.
            return trace.get_qk(self.model, 1.0, self.ids if ids is None else ids)

    def test_native_post_rope_and_causal_group_probabilities(self):
        native, attentions, old_projection = [], [], []
        original_rope = hf.apply_rotary_pos_emb
        original_eager = hf.eager_attention_forward
        def rope(*args, **kwargs):
            result = original_rope(*args, **kwargs)
            native.append(tuple(v[0].transpose(0, 1).detach().clone() for v in result))
            return result
        def eager(*args, **kwargs):
            result = original_eager(*args, **kwargs)
            attentions.append(result[1].detach().clone())
            return result
        def pre_layer(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            a = module.self_attn
            # Recreate the historical defect only for a sensitivity check.
            raw = a.q_proj(hidden).view(1, 641, 4, 128).float()
            old_projection.append((raw * torch.rsqrt(raw.square().mean(-1, keepdim=True)
                                                     + a.q_norm.variance_epsilon) * a.q_norm.weight).detach())
        handle = self.model.model.layers[0].register_forward_pre_hook(pre_layer, with_kwargs=True)
        try:
            with patch.object(hf, 'apply_rotary_pos_emb', rope), patch.object(hf, 'eager_attention_forward', eager):
                q, k = self.collect()
        finally:
            handle.remove()
        self.assertEqual(len(native), 2)
        self.assertEqual(len(attentions), 2)
        max_prob_error = 0.0
        for layer in (0, 1):
            torch.testing.assert_close(q[layer], native[layer][0], atol=0, rtol=0)
            torch.testing.assert_close(k[layer], native[layer][1], atol=0, rtol=0)
            for pos in (0, 1, 127, 128, 255, 256, 512, 640):
                p = trace.group_mean_probs(q[layer][pos], k[layer][:pos+1], 128**-.5)
                expected = attentions[layer][0, :, pos, :pos+1].reshape(2, 2, pos+1).mean(1)
                torch.testing.assert_close(p, expected, atol=2e-7, rtol=2e-6)
                max_prob_error = max(max_prob_error, float((p-expected).abs().max()))
                self.assertEqual(int(torch.count_nonzero(attentions[layer][0, :, pos, pos+1:])), 0)
                o1 = trace.o1_trace(p, pos)
                self.assertAlmostEqual(o1['mass_protected_union']+o1['mass_near']+o1['mass_far'], 1, places=5)
            for subspace in trace.SUBSPACES.values():
                rows = trace.o3_trace(q[layer][-1], k[layer], 128**-.5, subspace)
                for row in rows:
                    self.assertEqual(row['mid_count'], 385)
                    self.assertEqual(row['selected_count'], 385)
                    self.assertAlmostEqual(row['reference_token_overlap_recall'], 1)
        # Position zero has cos=1/sin=0 and isolates the missing pre-attention LN.
        old_error = float((old_projection[0][0, 0]-native[0][0][0]).abs().max())
        self.assertGreater(old_error, .01)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(old_projection[0][0, 0], native[0][0][0], atol=2e-7, rtol=2e-6)
        state = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            state.update(name.encode()); state.update(value.detach().numpy().tobytes())
        receipt = dict(schema='e140-hf-cpu-parity-v1', scope='random-initialized CPU fp32; not pretrained/GPU parity',
                       torch=torch.__version__, transformers=transformers.__version__, seed=140,
                       config=self.model.config.to_dict(), state_sha256=state.hexdigest(),
                       input_ids_sha256=digest_tensor(self.ids), collector_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                       hf_source_sha256=hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest(),
                       qk_max_abs_error=0.0, group_probability_max_abs_error=max_prob_error,
                       omitted_input_layernorm_max_abs_error=old_error,
                       query_positions=[0,1,127,128,255,256,512,640], layers=[0,1],
                       gpu_used=False, downloaded_weights=False)
        destination = os.environ.get('E140_CPU_RECEIPT')
        if destination:
            Path(destination).write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')

    def test_input_layer_identity_rejects_before_forward(self):
        with patch.object(self.model, 'forward', side_effect=AssertionError('forward must not run')):
            for ids in (torch.zeros(2, 3, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 3), torch.zeros(3, dtype=torch.long)):
                with self.assertRaises(ValueError): self.collect(ids=ids)
            for layers in ([], [0, 0], [-1], [2], [True]):
                with self.assertRaises(ValueError): self.collect(layers=layers)
            self.model.train()
            with self.assertRaises(ValueError): self.collect()

    def test_failure_removes_only_owned_hooks(self):
        attn = self.model.model.layers[0].self_attn
        caller = attn.q_norm.register_forward_hook(lambda *args: None)
        before = {id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in self.model.modules()}
        try:
            with patch.object(hf, 'apply_rotary_pos_emb', side_effect=RuntimeError('injected forward failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected'): self.collect()
            after = {id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in self.model.modules()}
            self.assertEqual(before, after)
        finally:
            caller.remove()
        # Partial hook registration must also unwind cleanly.
        with patch.object(attn.k_norm, 'register_forward_hook', side_effect=RuntimeError('registration failed')):
            with self.assertRaisesRegex(RuntimeError, 'registration'): self.collect()
        self.assertFalse(attn.q_norm._forward_hooks)

    def test_qk_and_subspace_boundaries(self):
        q, k = torch.zeros(4, 128), torch.zeros(259, 2, 128)
        for scale in (True, 0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): trace.group_mean_probs(q, k, scale)
        for bad_q, bad_k in ((q.long(), k.long()), (q.double(), k), (q[:3], k), (q[:, :0], k[:, :, :0]), (q+float('nan'), k), (q, k[:0])):
            with self.assertRaises(ValueError): trace.group_mean_probs(bad_q, bad_k, 128**-.5)
        for indices in ([], [0, 0], [True], [-1], [128], [1.5]):
            with self.assertRaises(ValueError): trace.o3_trace(q, k, 128**-.5, indices)
        with self.assertRaises(ValueError): trace.o3_trace(q, k, 32**-.5, [0, 1])
        with self.assertRaises(ValueError): trace.o3_trace(q[:, :64], k[:, :, :64], 128**-.5, [0, 1])

    def test_nonfinite_projection_and_missing_rope_fail_closed(self):
        attn = self.model.model.layers[0].self_attn
        caller = attn.q_norm.register_forward_hook(lambda module, args, output: output*float('nan'))
        try:
            with self.assertRaisesRegex(ValueError, 'values'): self.collect(ids=self.ids[:, :3])
            self.assertEqual(tuple(attn.q_norm._forward_hooks), (caller.id,))
            self.assertFalse(attn.k_norm._forward_hooks)
            self.assertFalse(attn._forward_pre_hooks)
        finally:
            caller.remove()
        def remove_rope(module, args, kwargs):
            kwargs.pop('position_embeddings', None)
            return args, kwargs
        caller = attn.register_forward_pre_hook(remove_rope, with_kwargs=True)
        try:
            with self.assertRaisesRegex(ValueError, 'position_embeddings'): self.collect(ids=self.ids[:, :3])
            self.assertFalse(attn.q_norm._forward_hooks)
            self.assertFalse(attn.k_norm._forward_hooks)
            self.assertEqual(tuple(attn._forward_pre_hooks), (caller.id,))
        finally:
            caller.remove()

if __name__ == '__main__':
    unittest.main()
