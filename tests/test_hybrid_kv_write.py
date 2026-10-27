"""Bitwise row-write checks; CPU interpreter is logic evidence, never GPU timing."""
import os
from types import SimpleNamespace

import pytest
import torch

from sparse_attn.ops import far_skip_accel as accel
from sparse_attn.ops.far_skip import FarSkipLayout
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer
from sparse_attn.ops.hybrid_batch_cache import write_hybrid_row


def make_row(ring, device, dtype, capacity=67):
    # Deliberately different K/V strides, including non-unit dimension stride.
    row = object.__new__(FarSkipStaticLayer) if ring else SimpleNamespace()
    row.keys = torch.randn(1, 3, capacity, 128, device=device, dtype=dtype)
    row.values = torch.randn(1, capacity, 3, 256, device=device, dtype=dtype).transpose(1, 2)[..., ::2]
    if ring:
        row.layout = FarSkipLayout(3, capacity-3)
    return row


def reference(row, k, v, position, alive):
    # Main b0a378c's write sequence, kept independent of the new dispatch.
    slot = row.layout.map(position) if isinstance(row, FarSkipStaticLayer) else position
    for storage, incoming in ((row.keys, k), (row.values, v)):
        old = storage.index_select(2, slot)
        storage.index_copy_(2, slot, torch.where(alive.view(1, 1, 1, 1), incoming, old))


def copy_row(row):
    result = object.__new__(FarSkipStaticLayer) if isinstance(row, FarSkipStaticLayer) else SimpleNamespace()
    result.keys, result.values = row.keys.clone(), row.values.clone()
    if isinstance(row, FarSkipStaticLayer):
        result.layout = row.layout
    return result


@pytest.fixture(params=['torch', 'interpreter', 'cuda'])
def backend(request, monkeypatch):
    mode = request.param
    interpreting = os.environ.get('TRITON_INTERPRET') == '1'
    if mode == 'cuda' and (not torch.cuda.is_available() or interpreting):
        pytest.skip('requires real CUDA')
    if mode == 'interpreter' and not interpreting:
        pytest.skip('requires TRITON_INTERPRET=1')
    monkeypatch.setattr(accel, 'enabled', lambda _: mode != 'torch')
    return mode, 'cuda' if mode == 'cuda' else 'cpu'


@pytest.mark.parametrize('ring', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_row_write_wrap_eos_strides_and_launch_contract(backend, ring, dtype, monkeypatch):
    mode, device = backend
    if mode == 'interpreter' and dtype == torch.bfloat16:
        pytest.skip('BF16 needs real GPU validation')
    row = make_row(ring, device, dtype)
    ref = copy_row(row)
    pointers = row.keys.data_ptr(), row.values.data_ptr()
    calls = []
    original = accel.write_pair
    def track(*args, **kwargs):
        calls.append(kwargs['alive'])
        return original(*args, **kwargs)
    monkeypatch.setattr(accel, 'write_pair', track)
    pos = torch.tensor([0], device=device)
    active = torch.tensor([True], device=device)
    # Three complete wraps, prefix slots, and large absolute ring positions.
    positions = list(range(201)) + [2**33] if ring else list(range(67))
    for step, absolute in enumerate(positions):
        k, v = torch.randn(2, 1, 3, 1, 256, device=device, dtype=dtype)[..., ::2]
        pos.fill_(absolute)
        active.fill_(step < len(positions)-3)
        if not bool(active.item()):
            k.fill_(torch.nan)
            v.fill_(torch.nan)
        reference(ref, k, v, pos, active)
        write_hybrid_row(row, k, v, pos, active)
        for actual, expected in ((row.keys, ref.keys), (row.values, ref.values)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (row.keys.data_ptr(), row.values.data_ptr()) == pointers
    assert len(calls) == (0 if mode == 'torch' else len(positions))


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1',
                    reason='requires real CUDA capture/replay')
@pytest.mark.parametrize('batch', [1, 2, 8, 32])
@torch.inference_mode()
def test_cuda_mixed_writes_replay_and_recapture(batch, monkeypatch):
    monkeypatch.setenv('TLI_FAR_SKIP_KERNELS', 'triton')
    for repeat in range(2):
        rows = [make_row((b+repeat) % 2 == 0, 'cuda', torch.bfloat16, 17+b)
                for b in range(batch)]
        refs = [copy_row(row) for row in rows]
        k, v = [torch.randn(batch, 3, 1, 128, device='cuda', dtype=torch.bfloat16)
                for _ in range(2)]
        pos = torch.zeros(batch, device='cuda', dtype=torch.long)
        alive = torch.ones(batch, device='cuda', dtype=torch.bool)
        def run():
            for b, row in enumerate(rows):
                write_hybrid_row(row, k[b:b+1], v[b:b+1], pos[b:b+1], alive[b:b+1])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(); run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Account for the idempotent warmup/capture write in the oracle.
        for b, row in enumerate(refs):
            reference(row, k[b:b+1], v[b:b+1], pos[b:b+1], alive[b:b+1])
        for step in range(1, 60):
            k.normal_(); v.normal_()
            for b, row in enumerate(rows):
                # Linear rows stop before capacity; rings wrap repeatedly.
                active = step < (13 if b % 3 == 0 else 60) and (
                    isinstance(row, FarSkipStaticLayer) or step < row.keys.shape[2])
                alive[b] = active
                if active:
                    pos[b] = step
            for b, row in enumerate(refs):
                reference(row, k[b:b+1], v[b:b+1], pos[b:b+1], alive[b:b+1])
            graph.replay()
            for row, ref in zip(rows, refs):
                torch.testing.assert_close(row.keys, ref.keys, rtol=0, atol=0)
                torch.testing.assert_close(row.values, ref.values, rtol=0, atol=0)
