"""Qwen3 adapter of pinned gpt-fast SparQ, per query-head selection + mean-V.

Full KV remains resident; mean-V is recomputed (cost included), not an optimized
running cache. This is a model adapter candidate, not upstream Qwen3 support.
"""
import types
import torch
from sparse_attn.baseline_runtime import verified_vendor


def enable_sparq(model, k=1024, context_length=32768):
    if type(k) is not int or k <= 0 or type(context_length) is not int or context_length <= 0:
        raise ValueError('SparQ k/context_length must be positive integers')
    root = verified_vendor('sparq_vendor')
    import sys
    from pathlib import Path
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sparq import sparq_attn, SparQArgs
    for name in ('sparq', 'kv_cache'):
        if Path(sys.modules[name].__file__).resolve().parent != root.resolve():
            raise RuntimeError(f'SparQ import collision: {name}')
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb, repeat_kv
    layers = [m for m in model.modules() if isinstance(m, Qwen3Attention)]
    if not layers or len(layers) != model.config.num_hidden_layers:
        raise ValueError('SparQ adapter supports complete Qwen3 attention models only')
    if any(hasattr(m, '_sparq_native_forward') for m in layers):
        raise ValueError('SparQ already installed')
    stats = {}
    config = SparQArgs(running_V_mean=False)
    def bump(key):
        stats[key] = stats.get(key, 0) + 1
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        b, s, _ = hidden_states.shape
        if b != 1 or s < 1:
            raise ValueError('SparQ candidate supports batch=1 only')
        if s > 1:
            out = self._sparq_native_forward(hidden_states, position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values,
                cache_position=cache_position, **kwargs)
            bump('native_prefill_calls')
            return out
        from sparse_attn.baseline_runtime import require_unmasked_decode
        require_unmasked_decode(hidden_states, attention_mask, past_key_values, cache_position)
        if kwargs.get('use_cache', True) is not True:
            raise ValueError('SparQ decode requires use_cache=True')
        q = self.q_norm(self.q_proj(hidden_states).view(b, s, -1, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(b, s, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, key = apply_rotary_pos_emb(q, key, cos, sin)
        key, value = past_key_values.update(key, value, self.layer_idx,
                      {'cos': cos, 'sin': sin, 'cache_position': cache_position})
        length = past_key_values.get_seq_length(self.layer_idx)
        if key.shape[-2] != length:
            raise ValueError('SparQ candidate requires non-padded dynamic cache')
        key, value = repeat_kv(key, self.num_key_value_groups), repeat_kv(value, self.num_key_value_groups)
        r = max(1, min(self.head_dim, round(self.head_dim / (context_length / (2 * k)))))
        mask = torch.ones((1, 1, 1, length), dtype=torch.bool, device=q.device)
        mean = value.mean(-2, dtype=torch.float32, keepdim=True).to(value.dtype)
        out = sparq_attn(q, key, key, value, mean, mask, r, min(k, length), config)
        out = self.o_proj(out.transpose(1, 2).reshape(b, s, -1).contiguous())
        bump('sparq_decode_calls')
        return out, None
    for layer in layers:
        layer._sparq_native_forward = layer.forward
        layer.forward = types.MethodType(forward, layer)
    model._baseline_stats = stats
    import hashlib
    model._baseline_identity = dict(method='sparq_vendor',
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream='f2eeb9d10e79d3ffecc48015e55157bab0f6e269', k=k,
        context_length=context_length, r_by_head_dim={str(m.head_dim):max(1,min(m.head_dim,round(m.head_dim/(context_length/(2*k))))) for m in layers},
        selection='per-query-head; mean-V compensation enabled; full resident KV',
        comparison='Qwen3 adapter, not group-shared native quality library; union K unobserved',
        gpu_quality_validated=False)
    return len(layers)
