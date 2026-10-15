"""CPU geometry proof checks plus direct GPU differential tests.

CPU tests do not compile or validate Triton. GPU tests never use the caller's
exception-catching fallback. Run: python -m unittest test_prefill_blocklist -v
"""
import random
import unittest


def legacy_blocks(bits, first, last, sink, swa):
    return [b for b, selected in enumerate(bits)
            if selected or (b * 64 < sink and b * 64 <= last)
            or (b * 64 <= last and b * 64 + 64 > first - swa + 1)]


class Geometry(unittest.TestCase):
    def test_sorted_group_superset_retains_exact_tile_sequence(self):
        rng = random.Random(20261015)
        for length in (1, 63, 64, 65, 127, 129, 2048):
            for pos in (0, 1031, 64000):
                blocks = (pos + length + 63) // 64
                for grp in (64, 128):
                    for G in (1, 2, 4, 8):
                        br = 64 // G
                        for sink, swa in ((0, 0), (1, 1), (128, 128)):
                            for pattern in ("empty", "full", "random"):
                                bits = [pattern == "full" or
                                        (pattern == "random" and rng.random() < .08)
                                        for _ in range(blocks)]
                                for group_start in range(0, length, grp):
                                    group_end = min(length, group_start + grp)
                                    ids = legacy_blocks(bits, pos + group_start,
                                                        pos + group_end - 1, sink, swa)
                                    self.assertEqual(ids, sorted(set(ids)))
                                    for start in range(group_start, group_end, br):
                                        wanted = legacy_blocks(bits, pos + start,
                                                               pos + min(start + br, group_end) - 1,
                                                               sink, swa)
                                        # The consumer applies the original need predicate.
                                        wanted_set = set(wanted)
                                        actual = [b for b in ids if b in wanted_set]
                                        self.assertEqual(actual, wanted)


try:
    import torch
    import triton
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False


@unittest.skipUnless(GPU, "requires CUDA and Triton; CPU is not kernel validation")
class GPUChecks(unittest.TestCase):
    def test_packing_and_attention(self):
        from sparse_attn.ops.fused_attn_triton import fused_group_attn, _pack_group_blocks
        torch.manual_seed(20261015)
        for G, D, length, pos, sink, swa, dtype in (
            (1, 64, 1, 0, 0, 0, torch.float16),
            (4, 128, 65, 1024, 128, 128, torch.bfloat16),
            (8, 64, 129, 4096, 0, 0, torch.float16),
            (4, 128, 2048, 8192, 128, 128, torch.bfloat16),
        ):
            T, hkv, grp = pos + length, 2, 64
            ng, tb = (length + grp - 1) // grp, (T + 63) // 64
            q = torch.randn(1, length, hkv * G, D, device="cuda", dtype=dtype)
            k = torch.randn(1, T, hkv, D, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            for probability in (0., .03, 1.):
                # Include future-selected tokens deliberately; causal stays per row.
                sels = [torch.rand(hkv, T, device="cuda") < probability for _ in range(ng)]
                padded = torch.nn.functional.pad(torch.stack(sels), (0, tb * 64 - T))
                bitmap = padded.view(ng, hkv, tb, 64).any(-1).contiguous()
                ids = torch.full((ng, hkv, tb), -7, dtype=torch.int32, device="cuda")
                counts = torch.empty((ng, hkv), dtype=torch.int32, device="cuda")
                _pack_group_blocks[(ng * hkv,)](
                    bitmap, ids, counts, tb, length, pos, grp, hkv, swa, sink,
                    BLK=64, WIDTH=triton.next_power_of_2(tb))
                actual_ids, actual_counts, bits = ids.cpu(), counts.cpu(), bitmap.cpu()
                for j in range(ng):
                    for h in range(hkv):
                        expected = legacy_blocks(bits[j, h].tolist(), pos + j * grp,
                                                 pos + min((j + 1) * grp, length) - 1,
                                                 sink, swa)
                        count = actual_counts[j, h].item()
                        self.assertEqual(count, len(expected))
                        self.assertEqual(actual_ids[j, h, :count].tolist(), expected)
                        self.assertTrue((actual_ids[j, h, count:] == -7).all())
                kwargs = dict(pos_lo=pos, G=G, softmax_scale=D ** -.5,
                              sink_tok=sink, swa_tok=swa, grp=grp)
                before = fused_group_attn(q, k, v, sels, **kwargs, use_block_list=False)
                after = fused_group_attn(q, k, v, sels, **kwargs, use_block_list=True)
                torch.testing.assert_close(after, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
