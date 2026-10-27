import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sparse_attn" / "indexer" / "tli_indexer.py"


def load_harness(method_name, base_source="class Base:\n    pass\n"):
    """Execute one production method in a dependency-light class harness."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TLIIndexer"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"_graph_state_fn": None}
    source = base_source + "\nclass Harness(Base):\n"
    source += "\n".join("    " + line for line in ast.unparse(method).splitlines())
    exec(compile(source, str(SOURCE), "exec"), namespace)
    return namespace["Harness"]


def key_metadata(shape=(1, 8, 4, 128), dtype="bf16", device="cpu"):
    return SimpleNamespace(shape=shape, dtype=dtype, device=device)


class TliRuntimeShapeContractTests(unittest.TestCase):
    def _prepare_harness(self):
        harness = load_harness("prepare_index")()
        harness._inc = {
            "S": 4,
            "sub_key": ("full", 1),
            "k_avg": None,
            "input_signature": ("cpu", "float32", 4, 128, 64),
        }
        harness.args = SimpleNamespace(tia_block_size=64)
        harness._basis = None
        harness._basis_all = None
        harness.moba_gate = False
        harness.subspace = "full"
        harness._need_avg_score = lambda: False
        harness._sub_key = lambda: ("full", 1)
        harness._prepare_index_inc = lambda *_args: "incremental"
        harness._prepare_index_full = lambda *_args, **_kwargs: "full"
        return harness

    def test_incremental_entry_rejects_extra_batch_instead_of_ignoring_it(self):
        harness = self._prepare_harness()
        key = SimpleNamespace(shape=(2, 8, 4, 128))
        with self.assertRaisesRegex(ValueError, "batch"):
            harness.prepare_index(key, object())

    def test_public_entry_rejects_wrong_key_width_before_full_fallback(self):
        harness = self._prepare_harness()
        key = SimpleNamespace(shape=(1, 8, 4, 64))
        with self.assertRaisesRegex(ValueError, "末维"):
            harness.prepare_index(key, object())

    def test_valid_key_contract_preserves_incremental_and_full_dispatch(self):
        harness = self._prepare_harness()
        def key(length, width=128):
            return SimpleNamespace(shape=(1, length, 4, width), device="cpu", dtype="float32")
        self.assertEqual(
            harness.prepare_index(key(8), object()),
            "incremental",
        )
        self.assertEqual(
            harness.prepare_index(key(4), object()),
            "full",
        )
        harness.subspace = "mla_rope"
        self.assertEqual(
            harness.prepare_index(key(8, 192), object()),
            "full",
        )

    def test_score_entry_rejects_multi_query_even_on_parent_shortcut(self):
        base = """class Base:
    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        assert q.shape[0] == 1 and q.shape[1] == 1
        return {'score_coarse': None, 'score_fine': None}
"""
        harness = load_harness("compute_score", base)()
        harness.enable_subspace = False
        harness.skip_far = False
        harness.alpha = 0.0
        harness.beta = 0.0
        harness.far_select = "4bit"
        harness.near_select = "4bit"
        harness._fine_gather = False
        harness._need_avg_score = lambda: False
        query = SimpleNamespace(shape=(1, 2, 4, 128))
        with self.assertRaisesRegex(ValueError, "query"):
            harness.compute_score(query, object(), {}, 1.0)

    def test_valid_score_contract_preserves_parent_shortcut(self):
        base = """class Base:
    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        return {'score_coarse': 'coarse', 'score_fine': 'fine'}
"""
        harness = load_harness("compute_score", base)()
        harness.enable_subspace = False
        harness.skip_far = False
        harness.alpha = 0.0
        harness.beta = 0.0
        harness.far_select = "4bit"
        harness.near_select = "4bit"
        harness._fine_gather = False
        harness._need_avg_score = lambda: False
        query = SimpleNamespace(shape=(1, 1, 4, 128))
        self.assertEqual(
            harness.compute_score(query, object(), {}, 1.0),
            {
                'score_coarse': 'coarse',
                'score_fine': 'fine',
                'valid_length': None,
            },
        )

    def test_moba_missing_gate_score_has_stable_error_under_optimized_python(self):
        harness = load_harness("_moba_mask")()
        with self.assertRaisesRegex(ValueError, "score_moba"):
            harness._moba_mask({}, 1, 64, 1)


if __name__ == "__main__":
    unittest.main()

