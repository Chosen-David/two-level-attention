import ast
import hashlib
import itertools
import json
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3 import modeling_qwen3 as qwen
from transformers.models.llama import modeling_llama as llama

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import sparse_attn.ops.eager_decoding as consumer

torch.set_num_threads(1)
torch.manual_seed(8532026)
results = []
input_hash = hashlib.sha256()


def load(family):
    model = qwen if family == 'qwen3' else llama
    ns = dict(torch=torch, os=os, sys=sys, weakref=weakref,
              apply_rotary_pos_emb=model.apply_rotary_pos_emb,
              eager_attention_forward=model.eager_attention_forward,
              ALL_ATTENTION_FUNCTIONS={'sdpa': sdpa_attention_forward},
              eager_decoding_attn=consumer.eager_decoding_attn)
    for path in (ROOT / 'sparse_attn/patches/utils.py',
                 ROOT / f'sparse_attn/patches/{family}_attn_patch.py'):
        tree = ast.parse(path.read_text())
        tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns[f'{family}_attn_forward']


def check(family, dtype, kvheads, prefix_len, pattern, fast, long=False):
    cfg_cls, attn_cls = ((qwen.Qwen3Config, qwen.Qwen3Attention) if family == 'qwen3'
                         else (llama.LlamaConfig, llama.LlamaAttention))
    cfg = cfg_cls(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                  num_key_value_heads=kvheads, head_dim=8, num_hidden_layers=1)
    cfg._attn_implementation = 'sdpa'
    attn = attn_cls(cfg, 0).to(dtype).eval()
    forward = load(family)
    cache, refcache = DynamicCache(config=cfg), DynamicCache(config=cfg)
    calls = []
    valid = torch.ones(prefix_len, dtype=torch.bool)
    if pattern == 'left':
        valid[:max(1, prefix_len // 2)] = False
    elif pattern == 'holes':
        valid[::8 if long else 2] = False
    elif pattern == 'last':
        valid[:] = False

    def selector(q, pos, k, cu):
        expected_k = refcache.layers[0].keys.transpose(1, 2)[:, valid]
        torch.testing.assert_close(k, expected_k, atol=0, rtol=0)
        assert q.shape == (1, 1, 4, 8)
        assert pos.tolist() == [int(valid.sum()) - 1]
        assert cu.tolist() == [0, int(valid.sum())]
        mask = torch.ones(1, 1, kvheads, k.shape[1], dtype=torch.bool)
        if long:
            mask.zero_()
            for h in range(kvheads):
                mask[0, 0, h, [h, k.shape[1] // 2 + h, k.shape[1] - 1]] = True
        calls.append(mask)
        return mask, 1

    attn.indexer = SimpleNamespace(clear=lambda: None, prepare_mask=selector)
    with torch.no_grad(), patch.dict(os.environ, {'TLI_SPARSE_PREFILL': '0',
                    'TLI_DECODE_DENSE': '0', 'TLI_E127_FAST': fast, 'TLI_E127_W_UPPER': '2048'}):
        if long:
            # Avoid quadratic prefill: populate the actual HF cache with fixed K/V.
            keys = torch.randn(1, kvheads, prefix_len, 8, dtype=dtype)
            values = torch.randn_like(keys)
            input_hash.update(keys.float().numpy().tobytes())
            input_hash.update(values.float().numpy().tobytes())
            cache.update(keys.clone(), values.clone(), 0)
            refcache.update(keys.clone(), values.clone(), 0)
        else:
            prefix = torch.randn(1, prefix_len, 32, dtype=dtype)
            input_hash.update(prefix.float().numpy().tobytes())
            angles = torch.arange(prefix_len).view(1, -1, 1).float() * torch.arange(8).view(1, 1, -1).float() / 100
            rope = (angles.cos().to(dtype), angles.sin().to(dtype))
            forward(attn, prefix, rope, None, past_key_values=cache, cache_position=torch.arange(prefix_len))
            attn(prefix, rope, None, past_key_values=refcache, cache_position=torch.arange(prefix_len))
        maxerr = 0.
        dense_original = consumer._eager_decoding_attn_dense
        dense_calls = []
        def dense_spy(*args, **kwargs):
            dense_calls.append(1)
            return dense_original(*args, **kwargs)
        with patch.object(consumer, '_eager_decoding_attn_dense', side_effect=dense_spy):
            for step in range(3):
                pos = prefix_len + step
                valid = torch.cat((valid, torch.ones(1, dtype=torch.bool)))
                token = torch.randn(1, 1, 64, dtype=dtype)[..., ::2]
                assert not token.is_contiguous()
                angles = torch.arange(8).view(1, 1, -1).float() * pos / 100
                rope = (angles.cos().to(dtype), angles.sin().to(dtype))
                input_hash.update(token.float().numpy().tobytes())
                padded_mask = torch.zeros(1, 2 * len(valid), dtype=torch.bool)
                padded_mask[:, ::2] = valid
                mask = padded_mask[:, ::2]
                assert not mask.is_contiguous()
                dense_mask = valid.view(1, 1, 1, -1).expand(1, 4, 1, -1).clone()
                if long:
                    dense_mask.zero_()
                    kept = valid.nonzero().flatten()
                    for h in range(kvheads):
                        selected = kept[[h, len(kept) // 2 + h, len(kept) - 1]]
                        dense_mask[:, h * (4 // kvheads):(h + 1) * (4 // kvheads), :, selected] = True
                expected, _ = attn(token, rope, dense_mask, past_key_values=refcache, cache_position=torch.tensor([pos]))
                actual, _ = forward(attn, token, rope, mask, past_key_values=cache, cache_position=torch.tensor([pos]))
                tol = {torch.float32: (1e-6, 1e-5), torch.bfloat16: (.01, .02), torch.float16: (.002, .005)}[dtype]
                torch.testing.assert_close(actual, expected, atol=tol[0], rtol=tol[1])
                torch.testing.assert_close(cache.layers[0].keys, refcache.layers[0].keys, atol=0, rtol=0)
                torch.testing.assert_close(cache.layers[0].values, refcache.layers[0].values, atol=0, rtol=0)
                maxerr = max(maxerr, (actual - expected).abs().max().item())
            consumer.eager_decoding_check_errors()
        if long:
            assert dense_calls == [], 'Long scenario did not actually execute gather'
        assert len(calls) == 3
        results.append(dict(family=family, dtype=str(dtype), kvheads=kvheads, prefix_len=prefix_len,
                            pattern=pattern, fast=fast, long=long, steps=3,
                            dense_calls=len(dense_calls), max_abs_error=maxerr))


for args in itertools.product(('qwen3', 'llama3'), (torch.float32, torch.bfloat16, torch.float16),
                              (1, 2, 4), (1, 7, 129), ('left', 'holes', 'last'), ('0', '1')):
    check(*args)
for args in itertools.product(('qwen3', 'llama3'), (torch.float32, torch.bfloat16, torch.float16), (1, 2, 4)):
    check(*args, 20000, 'holes', '1', long=True)
print(json.dumps(dict(torch=torch.__version__, cuda=torch.cuda.is_available(), cases=len(results),
                      steps=sum(r['steps'] for r in results), input_sha256=input_hash.hexdigest(),
                      short_max_abs_error=max(r['max_abs_error'] for r in results if not r['long']),
                      long_max_abs_error=max(r['max_abs_error'] for r in results if r['long']),
                      results=results), indent=2))

