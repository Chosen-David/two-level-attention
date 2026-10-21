import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "benchmark" / "LongBench" / "pred.py"


def load_splitter():
    """Load the pure prompt splitter without importing optional GPU deps."""
    tree = ast.parse(PRED.read_text(encoding="utf-8"), filename=str(PRED))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_split_prompt_question"
    ]
    if len(functions) != 1:
        raise AssertionError("pred.py must define one _split_prompt_question")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {}
    exec(compile(module, str(PRED), "exec"), namespace)
    return namespace["_split_prompt_question"]


class LongBenchPromptSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splitter = staticmethod(load_splitter())

    def test_short_unanchored_prompt_uses_whole_prompt_as_question(self):
        for prompt in ("", "x", "short prompt", "x" * 99):
            with self.subTest(length=len(prompt)):
                prefix, question = self.splitter(prompt, "qmsum")
                self.assertEqual(prefix, "")
                self.assertEqual(question, prompt)

    def test_unanchored_question_suffix_is_capped_at_100_characters(self):
        prompt = "p" * 17 + "q" * 100
        prefix, question = self.splitter(prompt, "qmsum")
        self.assertEqual(prefix, "p" * 17)
        self.assertEqual(question, "q" * 100)
        self.assertEqual(prefix + question, prompt)

    def test_task_anchor_wins_when_it_is_later_than_suffix_boundary(self):
        prompt = "context " * 30 + "Question: final?"
        prefix, question = self.splitter(prompt, "qasper")
        self.assertTrue(question.startswith("Question:"))
        self.assertEqual(prefix + question, prompt)

    def test_missing_task_anchor_falls_back_without_negative_index(self):
        prompt = "tiny"
        prefix, question = self.splitter(prompt, "qasper")
        self.assertEqual((prefix, question), ("", prompt))

    def test_production_entry_consumes_the_splitter_once(self):
        tree = ast.parse(PRED.read_text(encoding="utf-8"), filename=str(PRED))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_split_prompt_question"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 2)


if __name__ == "__main__":
    unittest.main()
