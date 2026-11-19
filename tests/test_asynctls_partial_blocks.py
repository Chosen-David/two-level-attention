"""Actual CPU Torch checks for the shared per-group partial-block scorer.

Extract only the existing Stage-1 helper definitions so this targeted check does not
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
    shared = ast.parse((VENDOR / "qwen3_asynctls_patch.py").read_text())
    names = {"_padded_key", "_qwen3_block_score", "_select_blocks"}
    body = [n for n in shared.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(VENDOR / "qwen3_asynctls_patch.py"), "exec"), namespace)
    def stage1(config, query, key):
        padded = namespace["_padded_key"](key, config.asynctls_block_size)
        return (namespace["_qwen3_block_score"](query, padded, config.asynctls_block_size),
                namespace["_select_blocks"](query, padded, config.asynctls_block_size,
                                            config.asynctls_top_kb))
    return stage1


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
        scores.append(per_head)
    return torch.stack(scores, dim=-1)


class PartialBlockScores(unittest.TestCase):
    def test_empty_and_complete_helper_preserve_input_identity(self):
        shared = ast.parse((VENDOR / "qwen3_asynctls_patch.py").read_text())
        node = next(n for n in shared.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_padded_key")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "_padded_key", "exec"), namespace)
        for length in (0, 8, 16):
            key = torch.randn(1, 2, length, 4)
            self.assertIs(namespace["_padded_key"](key, 8), key)

    def test_padding_must_not_replace_better_real_block(self):
        for adapter in ("shared",):
            for sign in (-1, 1):
                with self.subTest(adapter=adapter, sign=sign):
                    query = torch.tensor([[[[float(sign)]]]])
                    key = torch.tensor([-1.0 * sign] * 8 + [-2.0 * sign]).reshape(1, 1, 9, 1)
                    config = SimpleNamespace(asynctls_block_size=8, asynctls_top_kb=1)
                    scores, selected = load_stage1(adapter)(config, query, key)
                    torch.testing.assert_close(scores, torch.tensor([[-1., -2.]]), rtol=0, atol=0)
                    self.assertEqual(selected.tolist(), [[0]])

    def test_unpadded_oracle_matrix(self):
        torch.manual_seed(23)
        for adapter in ("shared",):
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
        for adapter in ("shared",):
            for tail in (0., 2.):
                with self.subTest(adapter=adapter, tail=tail):
                    key = torch.tensor([-1.] * 8 + [tail]).reshape(1, 1, 9, 1)
                    query = torch.ones(1, 1, 1, 1)
                    config = SimpleNamespace(asynctls_block_size=8, asynctls_top_kb=1)
                    scores, selected = load_stage1(adapter)(config, query, key)
                    torch.testing.assert_close(scores, torch.tensor([[-1., tail]]), rtol=0, atol=0)
                    self.assertEqual(selected.tolist(), [[1]])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
