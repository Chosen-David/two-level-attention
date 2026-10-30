import ast
from contextlib import nullcontext
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "benchmark" / "LongBench" / "pred.py"


def load_functions(*names):
    """Load dependency-light producer helpers without importing GPU deps."""
    tree = ast.parse(PRED.read_text(encoding="utf-8"), filename=str(PRED))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in functions}
    if found != set(names):
        raise AssertionError(
            f"pred.py helper mismatch: expected={set(names)!r}, found={found!r}"
        )
    namespace = {"torch": type("FakeTorch", (), {"no_grad": nullcontext})}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PRED), "exec"),
         namespace)
    return tuple(namespace[name] for name in names)


class FakeToken:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, _dim):
        return self

    def item(self):
        return self.value


class FakeIds:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, key):
        if key == 0:
            return [FakeToken(value) for value in self.values]
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if row == 0 and isinstance(col, int):
                return FakeToken(self.values[col])
            if isinstance(row, slice) and isinstance(col, slice):
                return FakeIds(self.values[col])
        raise AssertionError(f"unexpected FakeIds index: {key!r}")


class FakeLogits:
    def __getitem__(self, _key):
        return self

    def argmax(self, dim=-1):
        assert dim == -1
        return FakeToken(7)


class FakeOutput:
    def __init__(self):
        self.past_key_values = object()
        self.logits = FakeLogits()


class RejectEmptyModel:
    def __init__(self):
        self.input_widths = []

    def __call__(self, *, input_ids, past_key_values, use_cache):
        del past_key_values
        assert use_cache
        width = input_ids.shape[-1] if hasattr(input_ids, "shape") else 1
        self.input_widths.append(width)
        if width == 0:
            raise RuntimeError("model received an empty token sequence")
        return FakeOutput()


class FakeCapture:
    def __init__(self):
        self.phases = []
        self.ready = []
        self.completed = False

    def call(self, phase, model, **kwargs):
        self.phases.append(phase)
        return model(**kwargs)

    def token_ready(self, token):
        self.ready.append(token)

    def generation_complete(self):
        self.completed = True


class LongBenchEmptyHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate, = load_functions("_manual_generate")
        cls.generate = staticmethod(generate)

    def test_empty_material_seeds_cache_from_first_question_token(self):
        model = RejectEmptyModel()
        capture = FakeCapture()
        generated = self.generate(
            model, FakeIds([]), FakeIds([11, 12]), 1, None, capture
        )
        self.assertEqual(generated, [7])
        self.assertEqual(model.input_widths, [1, 1])
        self.assertEqual(capture.phases, ["question", "question"])
        self.assertEqual(capture.ready, [7])
        self.assertTrue(capture.completed)

    def test_both_segments_empty_fail_before_model_invocation(self):
        model = RejectEmptyModel()
        with self.assertRaisesRegex(ValueError, "no input tokens"):
            self.generate(model, FakeIds([]), FakeIds([]), 1, None)
        self.assertEqual(model.input_widths, [])

    def test_nonempty_material_path_is_unchanged(self):
        model = RejectEmptyModel()
        capture = FakeCapture()
        self.generate(model, FakeIds([3, 4]), FakeIds([11]), 1, None, capture)
        self.assertEqual(model.input_widths, [2, 1])
        self.assertEqual(capture.phases, ["material", "question"])

if __name__ == "__main__":
    unittest.main()
