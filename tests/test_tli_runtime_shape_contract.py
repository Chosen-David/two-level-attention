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
    namespace = {}
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
        }
        harness.args = SimpleNamespace(tia_block_size=64)
        harness._inc["input_signature"] = ("cpu", "bf16", 4, 128, 64)
        harness._basis = None
        harness._basis_all = None
        harness.moba_gate = False
        harness.subspace = "full"
        harness._need_avg_score = lambda: False
        harness._sub_key = lambda: ("full", 1)
        harness._prepare_index_inc = lambda *_args: "incremental"
        harness._prepare_index_full = lambda *_args: "full"
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
        self.assertEqual(
            harness.prepare_index(key_metadata(), object()),
            "incremental",
        )
        self.assertEqual(
            harness.prepare_index(key_metadata(shape=(1, 4, 4, 128)), object()),
            "full",
        )
        self.assertEqual(
            harness.prepare_index(key_metadata(shape=(1, 2, 4, 128)), object()),
            "full",
        )
        harness.subspace = "mla_rope"
        self.assertEqual(
            harness.prepare_index(key_metadata(shape=(1, 8, 4, 192)), object()),
            "full",
        )

    def test_changed_metadata_rebuilds_before_incremental_buffer_writes(self):
        for key, block_size in (
            (key_metadata(shape=(1, 8, 1, 128)), 64),
            (key_metadata(dtype="float32"), 64),
            (key_metadata(device="cuda:1"), 64),
            (key_metadata(), 32),
        ):
            with self.subTest(key=key, block_size=block_size):
                harness = self._prepare_harness()
                harness.args.tia_block_size = block_size
                self.assertEqual(harness.prepare_index(key, object()), "full")

    def test_legacy_cache_without_signature_rebuilds(self):
        harness = self._prepare_harness()
        del harness._inc["input_signature"]
        self.assertEqual(harness.prepare_index(key_metadata(), object()), "full")

    def test_full_preparation_records_signature_for_next_growth(self):
        class Buffer:
            shape = (1, 2, 4, 32)

            def dim(self):
                return 4

            def squeeze(self, _axis):
                return self

            def clone(self):
                return self

        harness = load_harness("_prepare_index_full")()
        harness.args = SimpleNamespace(tia_block_size=64)
        harness._basis = None
        harness.moba_gate = False
        harness._sub_key = lambda: ("full", 1)
        result = {"k_min": Buffer(), "k_max": Buffer(), "k_qat": Buffer()}
        harness._prepare_index_base = lambda *_: result
        key = key_metadata()
        self.assertIs(harness._prepare_index_full(key, object()), result)
        growing = self._prepare_harness()
        growing._inc = harness._inc
        self.assertEqual(growing.prepare_index(
            key_metadata(shape=(1, 9, 4, 128)), object()), "incremental")
        growing._inc = None
        self.assertEqual(growing.prepare_index(key, object()), "full")

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
