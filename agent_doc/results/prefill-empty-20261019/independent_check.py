"""Independent CPU oracle checks; deliberately enumerates support per row/head.

When archived below agent_doc/results, run from the repository root with:
PYTHONPATH=. python agent_doc/results/prefill-empty-20261019/independent_check.py
"""
import itertools
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sparse_attn.ops import reference_prefill as ref

torch.set_num_threads(1)
torch.manual_seed(4241)


def oracle(q, k, v, sel, offset, sink=0, window=0):
    out = torch.empty_like(q)
    groups = q.shape[2] // k.shape[2]
    for row in range(q.shape[1]):
        pos = offset + row
        for head in range(q.shape[2]):
            kv = head // groups
            sh = kv if sel is not None and sel.shape[0] == k.shape[2] else head
            visible = [col for col in range(k.shape[1]) if col <= pos and (
                sel is None or bool(sel[sh, col]) or col < sink or col >= pos - window + 1
            )]
            if not visible:
                visible = [pos]
            score = ((q[0, row, head] * 0.37).float() * k[0, visible, kv].float()).sum(-1)
            prob = torch.softmax(score, -1).to(q.dtype)
            out[0, row, head] = (prob[:, None] * v[0, visible, kv]).sum(0)
    return out


checks = 0
for dtype, (h, hkv), per_query, pattern, tile, protection, strided in itertools.product(
    (torch.float32, torch.float16, torch.bfloat16), ((1, 1), (4, 1), (4, 2)),
    (False, True), ('empty', 'full', 'mixed', 'future'), (1, 2, 100),
    ((0, 0), (1, 0), (0, 2)), (False, True),
):
    n, offset, d, t = 5, 2, 6, 7
    def tensor(length, heads):
        x = torch.randn(1, length, heads, d * (2 if strided else 1), dtype=dtype)
        return x[..., ::2] if strided else x
    q, k, v = tensor(n, h), tensor(t, hkv), tensor(t, hkv)
    sel = torch.zeros(h if per_query else hkv, t, dtype=torch.bool)
    if pattern == 'full':
        sel[:] = True
    elif pattern == 'mixed':
        for head in range(sel.shape[0]):
            sel[head, 5 if head % 2 == 0 else 0] = True
    elif pattern == 'future':
        sel[:, -1] = True
    sink, window = protection
    with patch.object(ref, '_ROW_BLOCK_MAX_ELEM', tile * h * t):
        actual = ref._chunk_rows_attn(q, k, v, sel, offset, h // hkv, 0.37, sink, window)
    expected = oracle(q, k, v, sel, offset, sink, window)
    # Oracle uses explicit multiply/sum rather than einsum; reduced-precision
    # multiplication can round before the sum while einsum accumulates internally.
    tol = 2e-6 if dtype == torch.float32 else (0.004 if dtype == torch.float16 else 0.03)
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    assert torch.isfinite(actual).all()
    checks += 1

# Dense path, tiny/empty sequences, one-row sparse path, and public wrapper.
for length, threshold, dtype in itertools.product((0, 1, 2, 7), (0, 100), (torch.float32, torch.float16, torch.bfloat16)):
    q = torch.zeros(1, length, 4, 6, dtype=dtype)
    k = torch.zeros(1, length, 2, 6, dtype=dtype)
    v = torch.arange(length, dtype=dtype).view(1, length, 1, 1).expand(1, length, 2, 6)
    indexer = SimpleNamespace(args=SimpleNamespace(tia_level2_topk=0), sink_tokens=0, sliding_window_size=0,
        prepare_mask=lambda query, ids, key, cu, scale: (torch.zeros(1, 1, 2, key.shape[1], dtype=torch.bool), 1))
    actual = ref.sparse_prefill_attn(indexer, q, k, v, softmax_scale=0.37, chunk_size=2, dense_below=threshold)
    expected = torch.empty_like(q)
    for row in range(length):
        expected[0, row] = row if threshold == 0 else row / 2
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.03)
    checks += 1

# With zero q/k, support means expose the original pollution exactly. Also
# demand invariance to execution tile size for mixed row AND per-query heads.
q = torch.zeros(1, 5, 4, 1)
k = torch.zeros(1, 7, 2, 1)
v = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1).expand(1, 7, 2, 1)
sel = torch.zeros(4, 7, dtype=torch.bool)
sel[0, 5] = True
sel[1, 0] = True
sel[2, 4] = True
sel[3, 1] = True
expected = oracle(q, k, v, sel, 2)
for tile in (1, 2, 3, 100):
    with patch.object(ref, '_ROW_BLOCK_MAX_ELEM', tile * 4 * 7):
        actual = ref._chunk_rows_attn(q, k, v, sel, 2, 2, 0.37)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checks += 1

print(f'PASS: {checks} independent CPU cases; torch={torch.__version__}; CUDA={torch.cuda.is_available()}')
