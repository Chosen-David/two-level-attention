"""Reproduce the reviewer's CPU checks; run from the repository root.

  timeout 120s env PYTHONPATH=. python agent_doc/results/prefill-mask-contract-20261019/verify.py

Hash protocol: ordered length-framed JSON metadata and logical C-order tensor
bytes, SHA256. Metadata includes labels, dtype, shape, strides, storage offset
and device. A meta Tensor has metadata but no physical payload. Hashing is
verification instrumentation only, not production implementation.
"""
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sparse_attn.ops.eager_prefill import sparse_prefill_attn
from sparse_attn.ops.prefill_contract import token_selection


DIGEST = hashlib.sha256()
TENSOR_EVENTS = 0


def record(label, tensor):
    global TENSOR_EVENTS
    payload = b'' if tensor.device.type == 'meta' else tensor.detach().contiguous().numpy().tobytes()
    metadata = json.dumps({
        'label': label, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
        'device': str(tensor.device), 'stride': list(tensor.stride()),
        'storage_offset': tensor.storage_offset(), 'payload_bytes': len(payload),
        'payload_kind': 'unmaterialized-meta' if tensor.is_meta else 'logical-c-order',
    }, sort_keys=True, separators=(',', ':')).encode()
    for part in (metadata, payload):
        DIGEST.update(len(part).to_bytes(8, 'big'))
        DIGEST.update(part)
    TENSOR_EVENTS += 1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def env(runtime, group, batch='0'):
    return patch.dict(os.environ, {
        'TLI_PREFILL_RUNTIME': runtime, 'TLI_PREFILL_SELECT_GROUP': str(group),
        'TLI_PREFILL_PREFETCH': '0', 'TLI_PREFILL_BATCH_SELECT': batch,
        'TLI_PREFILL_FUSED_ATTN': '0',
    }, clear=True)


torch.set_num_threads(1)
torch.manual_seed(107)
helper_checks = 0
for kind in ('stride2_offset', 'zero_head_stride', 'zero_token_stride'):
    for heads in (2, 4):
        base = torch.rand(1, 1, heads, 30) > .5
        if kind == 'stride2_offset':
            mask = base[..., 1::2]
        elif kind == 'zero_head_stride':
            mask = base[:, :, :1, :].expand(1, 1, heads, 30)
        else:
            mask = base[..., :1].expand(1, 1, heads, 30)
        record(f'helper/{kind}/heads={heads}', mask)
        out = token_selection(mask, np.int64(1), length=7, heads=4,
                              kv_heads=2, device=mask.device)
        require(out.data_ptr() == mask.data_ptr(), 'storage pointer changed')
        require(out.stride() == mask.stride()[2:], 'stride changed')
        require(out.storage_offset() == mask.storage_offset(), 'offset changed')
        require(torch.equal(out, mask[0, 0, :, :7]), 'slice values changed')
        helper_checks += 1

blocks = (True, False, 0, -1, 2, 1.0, torch.tensor(1),
          torch.tensor(1, device='meta'), np.bool_(True))
for i, block in enumerate(blocks):
    mask = torch.ones(1, 1, 2, 7, dtype=torch.bool)
    record(f'reject/{i}/mask', mask)
    if isinstance(block, torch.Tensor):
        record(f'reject/{i}/block', block)
    try:
        token_selection(mask, block, length=7, heads=4,
                        kv_heads=2, device=torch.device('cpu'))
    except ValueError:
        helper_checks += 1
    else:
        raise AssertionError(f'accepted invalid block case {i}')
print('helper_metadata_storage_checks', helper_checks)

# Reset to the exact seed used by the successful independent oracle run.
torch.manual_seed(107)
oracle_cases = 0
for runtime, group in (('reference', 0), ('candidate', 0), ('candidate', 2)):
    for length in (5, 7, 9):
        for heads in (2, 4):
            for kind in ('strided', 'zero_head_stride', 'zero_token_stride'):
                tag = f'oracle/{runtime}/{group}/{length}/{heads}/{kind}'
                q = torch.randn(1, length, 4, 3)
                k = torch.randn(1, length, 2, 3)
                v = torch.randn_like(k)
                for name, tensor in (('q', q), ('k', k), ('v', v)):
                    record(f'{tag}/{name}', tensor)
                masks = {}

                def prepare(qc, ids, kc, cu, scale):
                    end = kc.shape[1]
                    m = (torch.rand(1, 1, heads, (end + 3) * 2) > .6)[..., 1::2]
                    if kind == 'zero_head_stride':
                        m = m[:, :, :1, :].expand(1, 1, heads, end + 3)
                    if kind == 'zero_token_stride':
                        m = m[..., :1].expand(1, 1, heads, end + 3)
                    for name, tensor in (('qc', qc), ('ids', ids), ('kc', kc), ('cu', cu), ('mask', m)):
                        record(f'{tag}/prefix={end}/{name}', tensor)
                    masks[end] = m
                    return m, 1

                idx = SimpleNamespace(
                    args=SimpleNamespace(tia_block_size=1, tia_level2_topk=length),
                    sink_tokens=0, sliding_window_size=0, prepare_mask=prepare)
                with env(runtime, group):
                    out = sparse_prefill_attn(idx, q, k, v, softmax_scale=.4,
                                              chunk_size=4, dense_below=4)
                want = torch.empty_like(out)
                for row in range(length):
                    end = min(4 + ((row - 4) // (group or 4) + 1) * (group or 4), length)
                    for h in range(4):
                        kh = h // 2
                        positions = list(range(row + 1)) if row < 4 else [
                            t for t in range(row + 1)
                            if masks[end][0, 0, h if heads == 4 else kh, t]]
                        if not positions:
                            positions = [row]
                        weights = torch.softmax(((q[0, row, h] * .4) * k[0, positions, kh]).sum(-1), dim=0)
                        want[0, row, h] = (weights[:, None] * v[0, positions, kh]).sum(0)
                torch.testing.assert_close(out, want, rtol=2e-5, atol=2e-6)
                oracle_cases += 1
print('independent_random_logit_oracle_cases', oracle_cases)

q, k = torch.zeros(1, 5, 4, 3), torch.zeros(1, 5, 2, 3)
record('batch/q', q)
record('batch/k', k)
record('batch/v', k)


def invalid_prepare(qc, ids, kc, cu, scale):
    m = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    for name, tensor in (('qc', qc), ('ids', ids), ('kc', kc), ('cu', cu), ('mask', m)):
        record(f'batch/{route}/{name}', tensor)
    return m, 1


idx = SimpleNamespace(args=SimpleNamespace(tia_block_size=2, tia_level2_topk=2),
                      sink_tokens=0, sliding_window_size=0, prepare_mask=invalid_prepare)
batch_checks = 0
for route, result in enumerate((None, [torch.ones(2, 2, dtype=torch.bool),
                                      torch.ones(2, 4, dtype=torch.bool)])):
    for i, mask in enumerate(result or []):
        record(f'batch/{route}/aligned={i}', mask)
    with env('candidate', 2, '1'), patch('sparse_attn.ops.batch_select.batched_group_sels', return_value=result):
        try:
            sparse_prefill_attn(idx, q, k, k, chunk_size=4, dense_below=0)
        except ValueError as exc:
            require('prefill selection' in str(exc), 'wrong rejection boundary')
            batch_checks += 1
        else:
            raise AssertionError('batch fallback or tail bypassed contract')
print('batch_fallback_unaligned_tail_rejections', batch_checks)
print('input_tensor_events', TENSOR_EVENTS)
print('input_sha256', DIGEST.hexdigest())
print('torch', torch.__version__, 'numpy', np.__version__, 'cuda_available', torch.cuda.is_available())
print('PASS')
