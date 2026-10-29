"""Benchmark-only FA2 KV-cache baseline for a fixed unpadded Qwen3 decode graph.

HF's generic FA2 unpadding path cannot consume the runner's static 4D mask in
capture. This adapter uses FA2's public cache_seqlens tensor API, after the
unchanged Qwen3 projections, RoPE and TLIStaticCache update. It never updates KV
inside FA2 a second time, and does not alter native prefill. Single-threaded
benchmark use only; the Qwen3 attention registry is restored after each forward.
"""
from contextlib import contextmanager
import torch


class GraphFA2:
    def __init__(self, model, mask, lengths, *, kernel=None):
        from transformers.models.qwen3 import modeling_qwen3
        self.module = modeling_qwen3
        targets = [m for m in model.modules() if isinstance(m, modeling_qwen3.Qwen3Attention)]
        if (not targets or any(m.training or m.head_dim != 128 or m.sliding_window is not None
                               or m.config._attn_implementation != 'flash_attention_2' for m in targets)
                or mask.ndim != 4 or mask.shape[:3] != (1,1,1)
                or lengths.shape != (1,) or lengths.dtype != torch.int64
                or lengths.device != mask.device):
            raise ValueError('GraphFA2 requires eval Qwen3 D128, B1 unpadded static cache')
        self.targets = {id(m) for m in targets}
        self.mask = mask
        self.lengths = torch.empty_like(lengths, dtype=torch.int32)
        self.calls = {'capture': 0, 'eager': 0}
        if kernel is None:
            from flash_attn import flash_attn_with_kvcache
            kernel = flash_attn_with_kvcache
        self.kernel = kernel

    def attention(self, module, query, key, value, attention_mask,
                  dropout=0., scaling=None, sliding_window=None, **kwargs):
        if (id(module) not in self.targets or query.ndim != 4
                or query.shape[0] != 1 or query.shape[2] != 1
                or key.ndim != 4 or key.shape != value.shape
                or key.shape[0] != 1 or key.shape[2] != self.mask.shape[-1]
                or query.shape[-1] != 128 or key.shape[-1] != 128
                or query.shape[1] % key.shape[1]
                or any(t.dtype != torch.bfloat16 or t.device != self.mask.device
                       or t.stride(-1) != 1 for t in (query,key,value))
                or attention_mask is None or attention_mask.shape != self.mask.shape
                or attention_mask.data_ptr() != self.mask.data_ptr()
                or dropout != 0. or sliding_window is not None
                or kwargs.get('output_attentions', False)
                or kwargs.get('is_causal', True) is False):
            raise ValueError('unsupported GraphFA2 decode or untrusted mask')
        name = 'capture' if query.is_cuda and torch.cuda.is_current_stream_capturing() else 'eager'
        self.calls[name] += 1
        out = self.kernel(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2),
                          cache_seqlens=self.lengths, softmax_scale=scaling,
                          causal=True, num_splits=0)
        return out, None

    @contextmanager
    def activate(self, lengths):
        from transformers.modeling_utils import AttentionInterface
        self.lengths.copy_(lengths)
        old = self.module.ALL_ATTENTION_FUNCTIONS
        interface = AttentionInterface()
        interface['flash_attention_2'] = self.attention
        self.module.ALL_ATTENTION_FUNCTIONS = interface
        try:
            yield
        finally:
            self.module.ALL_ATTENTION_FUNCTIONS = old
