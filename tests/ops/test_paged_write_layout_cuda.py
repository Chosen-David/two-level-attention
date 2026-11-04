"""Native paged-write exact-value checks, including changed-input graph replay.

Run: python tests/ops/test_paged_write_layout_cuda.py
Requires the repository's Torch/Triton environment and a CUDA device.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import torch
    import triton
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA/Triton required')
class NativeWriteTest(unittest.TestCase):
    def test_eager_and_graph(self):
        from sparse_attn.ops.paged_region_kernels import paged_write_decode
        for dtype in (torch.float16, torch.bfloat16):
            for batch in (1, 3, 32):
                for hk, d, dv in ((2, 24, 32), (8, 192, 256)):
                    for layout in ('packed', 'glm', 'feature'):
                        with self.subTest(dtype=dtype, batch=batch, hk=hk, layout=layout):
                            self.check_case(paged_write_decode, dtype, batch, hk, d, dv, layout)

    def check_case(self, write, dtype, b, hk, d, dv, layout):
        kw = dict(device='cuda', dtype=dtype)
        keys = torch.empty((b, hk, 1, d), **kw)
        if layout == 'glm':
            joint = torch.empty((b, 1, hk, d+dv), **kw)
            vals = joint.transpose(1, 2)[..., d:]
        elif layout == 'feature':
            keys = torch.empty((b, hk, 1, d*2), **kw)[..., ::2]
            vals = torch.empty((b, hk, 1, dv*2), **kw)[..., ::2]
        else:
            vals = torch.empty((b, hk, 1, dv), **kw)
        page, pages = 4, b+2
        kp = torch.empty((pages, 1, hk, page, d), **kw)
        vp = torch.empty((pages, 1, hk, page, dv), **kw)
        slots = torch.arange(b, dtype=torch.int32, device='cuda')*page+page
        layer = SimpleNamespace(num_kv_heads=hk, head_dim_k=d, head_dim_v=dv,
            geometry=SimpleNamespace(page=page), PAD_PAGE_ID=0,
            kv_pool_k=kp, kv_pool_v=vp)
        workspace = {}

        def run():
            write(layer, keys, vals, slots, workspace=workspace)

        keys.fill_(1); vals.fill_(2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for step in range(3):
            keys.copy_(torch.randn_like(keys))
            vals.copy_(torch.randn_like(vals))
            host_slots = torch.arange(b, dtype=torch.int32)*page+page+step
            if b > 1:
                host_slots[step % b] = -1
            slots.copy_(host_slots)
            ek, ev = torch.full_like(kp, -99), torch.full_like(vp, -99)
            for row, slot in enumerate(host_slots.tolist()):
                if slot >= 0:
                    pp, off = divmod(slot, page)
                    ek[pp, 0, :, off] = keys[row, :, 0]
                    ev[pp, 0, :, off] = vals[row, :, 0]
            for fn in (run, graph.replay):
                kp.fill_(-99); vp.fill_(-99)
                fn()
                torch.testing.assert_close(kp, ek, rtol=0, atol=0)
                torch.testing.assert_close(vp, ev, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
