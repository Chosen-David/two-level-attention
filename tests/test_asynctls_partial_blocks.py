"""Actual CPU Torch checks for the adapters' partial-block score prefix.

Extract only the existing Stage-1 statements so this targeted check does not
need model downloads, Transformers, CUDA, or import-time optional kernels.
No Torch operation is mocked. The oracle reduces each unpadded real block.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(os.environ.get("ASYNCTLS_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
VENDOR = ROOT / "exp/vendor_baselines/asynctls_vendor"


def load_stage1(adapter):
    filename = "llama3" if adapter == "llama" else adapter
    source = ast.parse((VENDOR / f"{filename}_asynctls_patch.py").read_text())
    shared = ast.parse((VENDOR / "qwen3_asynctls_patch.py").read_text())
    score = next(n for n in shared.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_qwen3_block_score")
    forward = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                   and n.name == f"_{adapter}_asynctls_forward")
    start_name = "L" if adapter == "qwen3" else "length"
    start = next(i for i, n in enumerate(forward.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == start_name)
    stop = next(i for i, n in enumerate(forward.body)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "top_blocks")
    wrapper = ast.parse("def stage1(self, query_states, key_states):\n pass\n").body[0]
    wrapper.body = forward.body[start:stop + 1] + ast.parse(
        "return block_score, top_blocks").body
    module = ast.fix_missing_locations(ast.Module(body=[score, wrapper], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(VENDOR / f"{adapter}_asynctls_patch.py"), "exec"), namespace)
    return namespace["stage1"]


def oracle(query, key, block_size):
    kv_heads, length, width = key.shape[1:]
    groups = query.shape[1] // kv_heads
    grouped = query[0, :, 0].reshape(kv_heads, groups, width)
    scores = []
    for start in range(0, length, block_size):
        real = key[0, :, start:start + block_size]
        hi, lo = real.amax(dim=1), real.amin(dim=1)
        # Same accumulation contract, but no fictitious token is ever present.
        per_head = (torch.einsum("hgd,hd->h", grouped.clamp(min=0), hi)
                    + torch.einsum("hgd,hd->h", grouped.clamp(max=0), lo))
        scores.append(per_head.sum())
    return torch.stack(scores)


class PartialBlockScores(unittest.TestCase):
    def test_padding_must_not_replace_better_real_block(self):
        for adapter in ("qwen3", "llama"):
            for sign in (-1, 1):
                with self.subTest(adapter=adapter, sign=sign):
                    query = torch.tensor([[[[float(sign)]]]])
                    key = torch.tensor([-1.0 * sign] * 8 + [-2.0 * sign]).reshape(1, 1, 9, 1)
                    config = SimpleNamespace(asynctls_block_size=8, asynctls_top_kb=1)
                    scores, selected = load_stage1(adapter)(config, query, key)
                    torch.testing.assert_close(scores, torch.tensor([-1., -2.]), rtol=0, atol=0)
                    self.assertEqual(selected.tolist(), [0])

    def test_unpadded_oracle_matrix(self):
        torch.manual_seed(23)
        for adapter in ("qwen3", "llama"):
            stage1 = load_stage1(adapter)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for block_size in (8, 64):
                    for length in (1, block_size - 1, block_size, block_size + 1, 2 * block_size - 1):
                        for kv_heads, groups in ((1, 1), (2, 4)):
                            with self.subTest(adapter=adapter, dtype=dtype, block=block_size,
                                              length=length, kv_heads=kv_heads, groups=groups):
                                # Dyadic values avoid noise in exact min/max comparisons.
                                key = torch.randint(-8, 9, (1, kv_heads, length, 8)).to(dtype) / 8
                                query = torch.randint(-8, 9, (1, kv_heads * groups, 1, 8)).to(dtype) / 8
                                config = SimpleNamespace(asynctls_block_size=block_size,
                                                         asynctls_top_kb=1)
                                actual, selected = stage1(config, query, key)
                                expected = oracle(query, key, block_size)
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                self.assertEqual(selected.tolist(), torch.topk(expected, 1).indices.tolist())

    def test_real_zero_and_best_partial_block_remain_valid(self):
        for adapter in ("qwen3", "llama"):
            for tail in (0., 2.):
                with self.subTest(adapter=adapter, tail=tail):
                    key = torch.tensor([-1.] * 8 + [tail]).reshape(1, 1, 9, 1)
                    query = torch.ones(1, 1, 1, 1)
                    config = SimpleNamespace(asynctls_block_size=8, asynctls_top_kb=1)
                    scores, selected = load_stage1(adapter)(config, query, key)
                    torch.testing.assert_close(scores, torch.tensor([-1., tail]), rtol=0, atol=0)
                    self.assertEqual(selected.tolist(), [1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
