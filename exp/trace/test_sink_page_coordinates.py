"""CPU tests of the actual nested trace page selector, without private traces."""
import ast
import hashlib
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parent
ANALYZERS = ("analyze_e135_sinkswa_grid.py", "analyze_e148_sink4_grid.py")


def selector(filename, sink, mid_len, scores):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "select_pages")
    env = dict(torch=torch, BS=64, Hkv=scores.shape[0], mid_len=mid_len,
               sink=sink, pool_lo_blk=sink // 64, device="cpu",
               sc_blk=scores, sc_blk_av=scores)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), env)
    return env["select_pages"]


def reference(scores, sink, mid_len, lo, hi, count):
    """Enumerate absolute tokens; no copied vectorized overlap formula."""
    result = torch.zeros(scores.shape[0], mid_len, dtype=torch.bool)
    if count <= 0 or hi <= lo:
        return result
    pages = {}
    for page in range(scores.shape[1]):
        absolute = range((sink // 64 + page) * 64, (sink // 64 + page + 1) * 64)
        tokens = [p - sink for p in absolute if sink + lo <= p < sink + hi]
        if tokens:
            pages[page] = tokens
    for head in range(scores.shape[0]):
        chosen = sorted(pages, key=lambda page: float(scores[head, page]), reverse=True)
        for page in chosen[:count]:
            result[head, pages[page]] = True
    return result


class SinkPageCoordinatesTest(unittest.TestCase):
    def test_subblock_far_boundary_uses_absolute_page_origin(self):
        scores = torch.arange(80, dtype=torch.float32).repeat(2, 1)
        scores[:, 48] = 1000
        for filename in ANALYZERS:
            with self.subTest(filename=filename):
                actual = selector(filename, 4, 4096, scores)(0, 3072, False, 1)
                expected = reference(scores, 4, 4096, 0, 3072, 1)
                self.assertEqual(torch.where(expected[0])[0].tolist(), list(range(3068, 3072)))
                self.assertTrue(torch.equal(actual, expected))

    def test_page_overlap_matches_token_enumeration(self):
        generator = torch.Generator().manual_seed(148)
        # Distinct scores avoid imposing an unpromised topk tie order.
        scores = torch.stack([torch.randperm(80, generator=generator).float()
                              for _ in range(2)])
        for filename in ANALYZERS:
            for sink in (0, 4, 16, 63, 64, 65, 127, 128):
                select = selector(filename, sink, 4096, scores)
                for lo, hi in ((0, 3072), (3072, 4096), (0, 1), (59, 65),
                               (60, 64), (63, 129), (4095, 4096), (12, 12)):
                    for use_avg in (False, True):
                        with self.subTest(file=filename, sink=sink, lo=lo, hi=hi,
                                          use_avg=use_avg):
                            actual = select(lo, hi, use_avg, 1)
                            expected = reference(scores, sink, 4096, lo, hi, 1)
                            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    torch.set_num_threads(1)
    fixture = torch.arange(80, dtype=torch.float32).repeat(2, 1)
    fixture[:, 48] = 1000
    print("fixture_sha256=" + hashlib.sha256(fixture.numpy().tobytes()).hexdigest())
    unittest.main()
