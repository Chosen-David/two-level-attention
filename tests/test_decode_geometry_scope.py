"""Scope lifecycle tests; CUDA coverage executes real capture when available."""
import pytest
import torch
from sparse_attn.ops.decode_geometry import decode_geometry_scope, scoped_geometry


@pytest.mark.parametrize('inference', [False, True])
def test_reuse_36_layers_32_rows_and_invalidate_next_forward(inference):
    with torch.inference_mode(inference):
        lengths = torch.arange(32)
        calls = []
        def run():
            result = []
            for layer in range(36):
                for row in range(32):
                    def build():
                        calls.append((layer, row))
                        return lengths[row:row+1] * 2
                    result.append(scoped_geometry(lengths[row:row+1], ('geo',), build))
            return result
        with decode_geometry_scope():
            a = run()
            assert len(calls) == 32
            assert a[0] is a[32]
        lengths.add_(1)
        with decode_geometry_scope():
            b = run()
            assert len(calls) == 64
            assert b[0] is not a[0]
            assert b[0].item() == 2
        assert scoped_geometry(lengths[:1], ('geo',), lambda: 1) is None


def test_nested_exception_and_configuration_isolation():
    x = torch.ones(1)
    with decode_geometry_scope():
        a = scoped_geometry(x, (1,), lambda: object())
        with pytest.raises(RuntimeError), decode_geometry_scope():
            assert scoped_geometry(x, (1,), lambda: object()) is not a
            raise RuntimeError('capture failed')
        assert scoped_geometry(x, (1,), lambda: None) is a
        assert scoped_geometry(x, (2,), lambda: object()) is not a


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_warmup_capture_replay_and_second_capture():
    x = torch.ones(1, device='cuda')
    def forward():
        with decode_geometry_scope():
            values = [scoped_geometry(x, (1,), lambda: x*2) for _ in range(36)]
            assert all(v is values[0] for v in values)
            return sum(values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    for value in (3, 7):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = forward()
        x.fill_(value)
        g.replay()
        torch.testing.assert_close(out, x*72, rtol=0, atol=0)
