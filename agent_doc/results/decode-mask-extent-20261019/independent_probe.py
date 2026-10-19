"""Independent CPU mask-extent probe; run from repository root with PYTHONPATH=."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import torch
from sparse_attn.ops import eager_decoding as ed


def inputs(t, bs, extra, perqh, hetero):
    h, hkv, dk, dv = 6, 2, 8, (5 if hetero else 8)
    gen = torch.Generator().manual_seed(781 + t + bs)
    q = torch.randn(1, 1, h, dk, generator=gen)
    k = torch.randn(1, t, hkv, dk, generator=gen)
    v = torch.randn(1, t, hkv, dv, generator=gen)
    width = (t + bs - 1) // bs + extra
    rows = h if perqh else hkv
    # Last-axis non-unit stride AND a nonzero storage offset.
    storage = torch.zeros(1, 1, rows, 2 * width + 1, dtype=torch.bool)
    mask = storage[..., 1::2]
    for row in range(rows):
        mask[0, 0, row, row % min(width - extra, 5)] = True
        mask[0, 0, row, (t - 1) // bs] = True
    if extra:
        mask[..., -extra:] = True
    return q, k, v, mask, bs, torch.tensor([0, t])


def reference(args, scale):
    q, k, v, mask, bs, _ = args
    h, hkv, t = q.shape[-2], k.shape[-2], k.shape[1]
    result = []
    for qh in range(h):
        kvh = qh // (h // hkv)
        row = qh if mask.shape[-2] == h else kvh
        selected = torch.tensor([i for i in range(t) if mask[0, 0, row, i // bs]])
        # Float64 oracle, explicit logical token indexing, no production
        # repeat/einsum/topk or dense implementation reuse.
        logits = k[0, selected, kvh].double() @ (q[0, 0, qh] * scale).double()
        result.append(logits.softmax(0) @ v[0, selected, kvh].double())
    return torch.stack(result)[None, None].float()


def main():
    torch.set_num_threads(2)
    results = []
    for (t, bs, extra), perqh, hetero, fast in itertools.product(
        [(5, 1, 0), (5, 2, 0), (65, 64, 3), (16384, 1, 0),
         (16384, 2, 0), (16384, 1, 3)], [False, True], [False, True], ['0', '1']
    ):
        args = inputs(t, bs, extra, perqh, hetero)
        expected = reference(args, .37)
        with patch.dict(os.environ, {'TLI_E127_FAST': fast, 'TLI_E127_W_UPPER': '32'}), \
             patch.object(ed, '_HAS_TRITON', False), patch.object(ed, '_ERR_FLAG', {}), \
             patch.object(torch, 'topk', wraps=torch.topk) as topk:
            actual = ed.eager_decoding_attn(*args, softmax_scale=.37, allow_kv_hetero=hetero)
            ed.eager_decoding_check_errors()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            expected_gather = fast == '1' and t >= 16384 and extra == 0
            assert bool(topk.call_count) == expected_gather
        results.append({'t': t, 'block_size': bs, 'extra_blocks': extra,
                        'per_query_head': perqh, 'heterogeneous_v': hetero, 'fast': fast,
                        'torch_gather_executed': expected_gather,
                        'max_abs_error': (actual - expected).abs().max().item()})

    baseline = types.ModuleType('independent_baseline')
    source = subprocess.check_output(['git', 'show',
        '5549339870004cb9ffb5dba6f20a88c89ac48547:sparse_attn/ops/eager_decoding.py'], text=True)
    exec(compile(source, '<baseline-eager-decoding>', 'exec'), baseline.__dict__)
    singleton = (torch.zeros(1, 1, 2, 4), torch.zeros(1, 3, 1, 4),
                 torch.arange(3.).view(1, 3, 1, 1).expand(1, 3, 1, 4).contiguous(),
                 torch.ones(1, 1, 1, 1, dtype=torch.bool), 1, torch.tensor([0, 3]))
    with patch.dict(os.environ, {'TLI_E127_FAST': '1'}):
        old = baseline.eager_decoding_attn(*singleton)
        assert old.eq(1).all()
        try:
            ed.eager_decoding_attn(*singleton)
        except ValueError as exc:
            assert 'mask must cover' in str(exc)
        else:
            raise AssertionError('singleton malformed mask accepted')

    rejected = 0
    for fn, (t, bs, width) in itertools.product(
        [ed.eager_decoding_attn, ed._eager_decoding_attn_dense, ed._eager_decoding_attn_gather],
        [(5, 1, 1), (65, 64, 1), (16385, 2, 8192), (5, 1, 0)]
    ):
        args = list(inputs(t, bs, 0, True, True))
        args[3] = args[3][..., :width]
        with patch.object(torch, 'topk', side_effect=AssertionError('unexpected computation')), \
             patch.object(torch, 'where', side_effect=AssertionError('unexpected computation')):
            try:
                fn(*args, allow_kv_hetero=True)
            except ValueError as exc:
                assert 'mask must cover' in str(exc)
                rejected += 1
            else:
                raise AssertionError('short mask accepted')
    summary = {'device': 'cpu', 'torch_version': torch.__version__,
               'cuda_available': torch.cuda.is_available(), 'valid_cases': len(results),
               'invalid_cases_rejected': rejected, 'baseline_singleton_output': old.flatten().tolist(),
               'fixed_singleton_rejected': True, 'cases': results}
    out = Path(__file__).with_name('independent-probe.json')
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
