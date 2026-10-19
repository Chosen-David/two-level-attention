"""Independent CPU support oracle; no reference attention or compact helpers reused.

Run from repository root: PYTHONPATH=. python agent_doc/results/prefill-sigma-cap-20261019/verify.py
Uses actual TLI masks, but computes every expected row from absolute token positions.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sparse_attn.indexer import TLIIndexer
from sparse_attn.ops import eager_prefill


def tensor_hash(t):
    if t.numel() == 0:
        return hashlib.sha256(b'').hexdigest()
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def check(mode, length=1025, chunk=128, group=0, dtype=torch.float32,
          heads=2, kvheads=1, strided=False, tiny_rows=False):
    dim = 128
    q = torch.zeros(1, length, heads, dim, dtype=dtype)
    k = torch.zeros(1, length, kvheads, dim * (2 if strided else 1), dtype=dtype)
    v = torch.arange(length, dtype=torch.float32).to(dtype).view(1, length, 1, 1).expand_as(k).clone()
    if strided:
        k, v = k[..., ::2], v[..., ::2]
    index = TLIIndexer(SimpleNamespace(
        tia_block_size=64, tia_level1_topk=8, tia_level2_topk=256,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_layer_skip=False, tli_enable_kmeans=False,
        tli_subspace='tail', tli_sink_tokens=64, tli_swa_size=64,
        tli_alpha=.5, tli_beta=.5, tli_gamma=.5,
        tli_sigma_select=mode, tli_sigma=8.0,
    ))
    masks = {}
    caps = []
    original_prepare = index.prepare_mask
    original_rows = eager_prefill._chunk_rows_attn
    original_size = eager_prefill._row_block_size

    def record_prepare(*args, **kwargs):
        mask, bs = original_prepare(*args, **kwargs)
        if bs != 1:
            raise RuntimeError('unexpected non-token mask')
        masks[args[2].shape[1]] = mask[0, 0].clone()
        return mask, bs

    def record_rows(*args, **kwargs):
        caps.append(kwargs['w_sel_cap'])
        return original_rows(*args, **kwargs)

    def small_size(rhg, width, swa, sink=0):
        return original_size(rhg, width, swa, sink, budget=4096)

    env = {'TLI_PREFILL_RUNTIME': 'candidate', 'TLI_PREFILL_SELECT_GROUP': str(group),
           'TLI_PREFILL_PREFETCH': '0', 'TLI_PREFILL_BATCH_SELECT': '0',
           'TLI_PREFILL_FUSED_ATTN': '0', 'TLI_PREFILL_SELECT_GRAPH': '0',
           'TLI_E151C_ORACLE_QUOTA': ''}
    with patch.dict(os.environ, env), patch.object(index, 'prepare_mask', record_prepare), \
            patch.object(eager_prefill, '_chunk_rows_attn', record_rows), \
            patch.object(eager_prefill, '_row_block_size', small_size if tiny_rows else original_size):
        out = eager_prefill.sparse_prefill_attn(index, q, k, v, chunk_size=chunk, dense_below=256)

    # Build chunk/group assignment without using attention or selection adapters.
    endpoints = []
    if group:
        endpoints += [(0, min(256, length), None)]
        for start in range(256, length, chunk):
            stop = min(start + chunk, length)
            endpoints += [(g, min(g + group, stop), min(g + group, stop))
                          for g in range(start, stop, group)]
    else:
        endpoints = [(start, min(start + chunk, length),
                      None if start < 256 else min(start + chunk, length))
                     for start in range(0, length, chunk)]
    expected = torch.empty_like(out)
    # One conversion keeps the intentionally scalar oracle inexpensive on CPU.
    mask_lists = {endpoint: mask.tolist() for endpoint, mask in masks.items()}
    value_lists = v[0, :, :, 0].float().T.tolist()
    for start, stop, endpoint in endpoints:
        for row in range(start, stop):
            for head in range(heads):
                kh = head // (heads // kvheads)
                support = list(range(row + 1)) if endpoint is None else [
                    col for col in range(row + 1)
                    if col < 64 or col >= row - 63 or mask_lists[endpoint][kh][col]]
                # Input bf16 position values are quantized; average actual values.
                values = [value_lists[kh][col] for col in support]
                expected[0, row, head] = sum(values) / len(values)
    error = float((out.float() - expected.float()).abs().max()) if length else 0.0
    tolerance = 4.0 if dtype == torch.bfloat16 else 0.002
    expected_cap = 512 if mode == 'none' else 0
    cap_ok = (set(caps) == {0, expected_cap}) if mode == 'none' else all(cap == 0 for cap in caps)
    result = dict(mode=mode, length=length, chunk=chunk, group=group, dtype=str(dtype),
                  heads=heads, kvheads=kvheads, strided=strided, tiny_rows=tiny_rows,
                  input_sha256=[tensor_hash(t) for t in (q,k,v)],
                  last_count=int(masks[max(masks)].sum()) if masks else None,
                  caps=sorted(set(caps)), max_abs_error=error,
                  final=float(out[0,-1,0,0]) if length else None,
                  oracle_final=float(expected[0,-1,0,0]) if length else None,
                  pass_numeric=error <= tolerance, pass_cap=cap_ok)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result['pass_numeric'] and result['pass_cap']


if __name__ == '__main__':
    torch.set_num_threads(1)
    source = Path('sparse_attn/ops/eager_prefill.py')
    print(json.dumps({'torch':torch.__version__, 'cuda':torch.cuda.is_available(),
                      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    results = [check(mode) for mode in ('mid','far','near','none')]
    results += [check('mid', heads=4, kvheads=2, strided=True, tiny_rows=True),
                check('mid', dtype=torch.bfloat16),
                check('near', group=64),
                check('mid', length=2051, chunk=2048),
                check('mid', length=0), check('mid', length=17)]
    print('PASS' if all(results) else 'FAIL', flush=True)
    raise SystemExit(0 if all(results) else 1)
