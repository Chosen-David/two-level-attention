import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.RULER.yarn_receipt import (
    generation_pointer_path,
    validate_committed_generation,
)
from benchmark.official_baselines.ruler_official import (
    generate_ruler,
    resolve_rope,
)


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[11, 12, 13]]))

    def decode(self, ids, skip_special_tokens=True):
        return "answer"


class _Engine:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.runtime = {"implementation": "official-test", "fallback": False}
        self.calls = []

    def generate(self, ids, max_new_tokens, **kwargs):
        self.calls.append((ids, max_new_tokens, kwargs))
        return [21, 22], 0.1


class OfficialRulerTest(unittest.TestCase):
    def test_independent_runner_has_no_nfa_or_legacy_vendor_import(self):
        path = ROOT / "benchmark/official_baselines/ruler_official.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = (
            "sparse_attn", "exp.vendor_baselines", "benchmark.RULER.pred_ruler",
            "benchmark.LongBench.pred", "evaluation.quest_attention",
        )
        self.assertFalse(any(
            name == item or name.startswith(item + ".")
            for name in imported for item in forbidden
        ))

    def test_qwen_yarn_policy_is_effective_before_engine_construction(self):
        from transformers import Qwen3Config

        config = Qwen3Config(
            hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            max_position_embeddings=40960,
        )
        args = SimpleNamespace(
            model="Qwen3-8B", task="niah_single_1", context_length=65536,
            yarn=True, yarn_factor=None,
        )
        native, factor, scaling, policy, preflight = resolve_rope(args, config)
        self.assertEqual(native, 40960)
        self.assertEqual(factor, 2.0)
        self.assertEqual(scaling["original_max_position_embeddings"], 32768)
        self.assertEqual(config.rope_scaling["factor"], 2.0)
        self.assertGreaterEqual(config.max_position_embeddings, 65536 + 128)
        self.assertEqual(policy["requested_factor"], 2.0)
        self.assertEqual(preflight["status"], "verified")
        self.assertEqual(preflight["requested_factor"], 2.0)

    def test_long_context_without_yarn_fails_closed(self):
        from transformers import Qwen3Config

        config = Qwen3Config(max_position_embeddings=40960)
        args = SimpleNamespace(
            model="Qwen3-8B", task="vt", context_length=131072,
            yarn=False, yarn_factor=None,
        )
        with self.assertRaisesRegex(ValueError, "必须显式启用 YaRN"):
            resolve_rope(args, config)

    def test_prediction_and_v2_receipt_pass_unchanged_consumer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                method="quest", model_path=str(model), task="niah_single_1",
                context_length=32768, output_dir=str(root / "out"),
                pred_postfix="", t="test", budget=1024, quest_page_size=16,
                jssa_rank=32, yarn=False, yarn_factor=None, skip=0, max_num=1,
            )
            rows = [{"input": "ruler", "outputs": ["answer"], "length": 3, "index": 0}]
            engine = _Engine()
            out = generate_ruler(
                args, engine, rows,
                native_mpe=40960, factor=None, rope_scaling=None,
                rope_policy={"policy": "test-policy"}, rope_preflight={},
                runtime_rope={"runtime_max_position_embeddings": 40960},
                data_sha256="0" * 64, data_path=root / "input.jsonl",
                eos_ids=(99,),
            )
            self.assertEqual(engine.calls[0][1], 128)
            self.assertEqual(engine.calls[0][2]["min_new_tokens"], 1)
            self.assertFalse(engine.calls[0][2]["ignore_eos"])
            pointer = Path(generation_pointer_path(str(out)))
            generation = out.parent / pointer.read_text(encoding="utf-8").strip()
            prediction = generation / out.name
            receipt = generation / out.name.replace(".jsonl", "-yarn_receipt.json")
            checked = validate_committed_generation(str(prediction), str(receipt))
            self.assertEqual(checked["binding_lines"], 1)
            record = json.loads(prediction.read_text(encoding="utf-8"))
            self.assertEqual(record["_id"], "niah_single_1:0")
            self.assertEqual(record["generated_token_ids"], [21, 22])
            self.assertEqual(record["implementation"], "quest_official")
            self.assertEqual(record["budget"], 1024)
            self.assertEqual(checked["receipt"]["generation_params"]["method"], "quest")


if __name__ == "__main__":
    unittest.main()
