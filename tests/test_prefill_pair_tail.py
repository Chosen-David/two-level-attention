"""Compare real pair selection to the frozen main observer, including copy scope."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sparse_attn.indexer.tli_indexer import TLIIndexer


def make_indexer():
    return TLIIndexer(SimpleNamespace(
        tia_block_size=16, tia_level1_topk=8, tia_level2_topk=64,
        tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
        tli_enable_layer_skip=False, tli_sink_tokens=4, tli_swa_size=32,
        tli_near_pages=4, tli_near_tokens='off', tli_near_len=128,
        tli_subspace='tail', tli_static_pair=True))


def reference(q):
    # Frozen main17e56e8 expression: deliberately casts the entire prompt.
    qm = q.detach().float()[:, :, -256:].reshape(-1, q.shape[-1]).abs().mean(0)
    pairs = torch.topk(qm[:64] + qm[64:], 16).indices
    return torch.cat([pairs, pairs + 64]).sort().values.to(q.device)


class CastShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and kwargs.get('dtype') == torch.float32:
            self.shapes.append(tuple(args[0].shape))
        # Inference tensors dispatch the composite conversion instead.
        if (func == torch.ops.aten.to.dtype and args[1] == torch.float32
                and args[0].dtype != torch.float32):
            self.shapes.append(tuple(args[0].shape))
        return func(*args, **kwargs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape_layout', [(1, 1, 'head'), (2, 255, 'transposed'),
                                        (1, 256, 'strided'), (2, 257, 'head'),
                                        (2, 1025, 'transposed'), (1, 4097, 'strided')])
@torch.inference_mode()
def test_exact_selection_copy_scope_and_request_reset(device, dtype, shape_layout):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    b, s, layout = shape_layout
    torch.manual_seed(784)
    if layout == 'head':
        q = torch.randn(b, 4, s, 128, device=device, dtype=dtype)
    elif layout == 'transposed':
        q = torch.randn(b, s, 4, 128, device=device, dtype=dtype).transpose(1, 2)
    else:
        q = torch.randn(b, 4, s, 256, device=device, dtype=dtype)[..., ::2]
    idx = make_indexer()
    for generation in range(2):
        idx.clear()
        if generation:
            # Tied channel scores stress top-k equality, while the prefix is
            # deliberately irrelevant (including nonfinite values).
            q.fill_(1)
            if s > 256:
                q[:, :, :-256].fill_(float('nan'))
        expected = reference(q)
        with CastShapes() as cast:
            idx.observe_prefill_q(q)
        assert torch.equal(idx._pair_idx, expected)
        if dtype != torch.float32:
            assert cast.shapes == [(b, 4, min(s, 256), 128)]
        else:
            assert cast.shapes == []
        cached = idx._pair_idx
        with CastShapes() as second:
            idx.observe_prefill_q(q * 2)
        assert idx._pair_idx is cached and second.shapes == []
        idx.clear()
        idx.static_pair = False
        with CastShapes() as disabled:
            idx.observe_prefill_q(q)
        assert idx._pair_idx is None and disabled.shapes == []
        idx.static_pair = True
