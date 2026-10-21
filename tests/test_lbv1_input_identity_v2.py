import ast
import copy
import hashlib
from pathlib import Path
import unittest

from benchmark.LongBench.run_cap_manifest import (
    INPUT_IDENTITY_SCHEMA,
    build_lbv1_input_identity,
    build_run_cap_core,
    finalize_run_cap_manifest,
    input_ids_sha256,
    validate_run_cap_manifest,
)


ROWS = [
    {
        "input": "What changed?",
        "context": "alpha",
        "answers": ["same"],
        "all_classes": None,
        "length": 10,
    },
    {
        "input": "Why?",
        "context": "beta 中文",
        "answers": ["same"],
        "all_classes": None,
        "length": 10,
    },
]
PROMPT = "Context: {context}\nQuestion: {input}\nAnswer:"


def _core(identity):
    return build_run_cap_core(
        model_name="Qwen3-8B",
        model_path="/model",
        method="tli",
        dataset="qasper",
        dataset_fingerprint=identity["source_fingerprint"],
        run_timestamp="20261021-identity",
        effective_cap=31500,
        cap_gate_info=None,
        max_new_tokens=128,
        producer={"script_sha256": "a" * 64},
        input_identity=identity,
    )


class Lbv1InputIdentityV2Tests(unittest.TestCase):
    def test_producer_wires_v2_into_gate_and_both_sidecar_states(self):
        source = (Path(__file__).resolve().parents[1]
                  / "benchmark" / "LongBench" / "pred.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

        def named(name):
            return [node for node in calls
                    if isinstance(node.func, ast.Name) and node.func.id == name]

        self.assertEqual(len(named("build_lbv1_input_identity")), 1)
        self.assertEqual(len(named("input_ids_sha256")), 1)
        cores = named("build_run_cap_core")
        self.assertEqual(len(cores), 1)
        self.assertIn("input_identity", {kw.arg for kw in cores[0].keywords})
        finalizers = named("finalize_run_cap_manifest")
        self.assertEqual(len(finalizers), 2)
        self.assertTrue(all("input_ids_sha256" in
                            {kw.arg for kw in call.keywords}
                            for call in finalizers))

    def test_source_identity_binds_context_question_unicode_and_order(self):
        base = build_lbv1_input_identity(ROWS, PROMPT)
        self.assertEqual(base["schema"], INPUT_IDENTITY_SCHEMA)
        self.assertRegex(base["source_fingerprint"], r"^v2-[0-9a-f]{64}$")

        mutations = []
        changed = copy.deepcopy(ROWS)
        changed[0]["context"] = "changed"
        mutations.append(changed)
        changed = copy.deepcopy(ROWS)
        changed[0]["input"] = "Different question?"
        mutations.append(changed)
        changed = copy.deepcopy(ROWS)
        changed[1]["context"] = "beta 文本"
        mutations.append(changed)
        changed = copy.deepcopy(ROWS)
        changed[0]["source_revision"] = "corrected-v2"
        mutations.append(changed)
        mutations.append(list(reversed(copy.deepcopy(ROWS))))

        for rows in mutations:
            with self.subTest(rows=rows):
                self.assertNotEqual(
                    build_lbv1_input_identity(rows, PROMPT)["source_fingerprint"],
                    base["source_fingerprint"],
                )

    def test_template_is_separate_and_part_of_run_identity(self):
        first = build_lbv1_input_identity(ROWS, PROMPT)
        second = build_lbv1_input_identity(ROWS, PROMPT + "\nBe concise.")
        self.assertEqual(first["source_fingerprint"], second["source_fingerprint"])
        self.assertNotEqual(first["prompt_template_sha256"],
                            second["prompt_template_sha256"])
        self.assertNotEqual(_core(first)["cap_identity"]["hash"],
                            _core(second)["cap_identity"]["hash"])

    def test_actual_token_hash_matches_int64_little_endian_contract(self):
        ids = [1, 2, 6, 7]
        expected = hashlib.sha256(
            b"".join(value.to_bytes(8, "little", signed=True)
                     for value in ids)
        ).hexdigest()
        self.assertEqual(input_ids_sha256(ids), expected)
        self.assertNotEqual(input_ids_sha256(ids), input_ids_sha256([1, 2, 7, 6]))
        with self.assertRaisesRegex(ValueError, "signed int64"):
            input_ids_sha256([1 << 63])

    def test_v2_manifest_closes_source_template_and_actual_tokens(self):
        identity = build_lbv1_input_identity(ROWS, PROMPT)
        hashes = [input_ids_sha256([1, 2]), input_ids_sha256([3, 4])]
        manifest = finalize_run_cap_manifest(
            _core(identity), status="complete", n_samples=2,
            input_tokens=[2, 2], truncated_flags=[False, False],
            result_file="qasper-test.jsonl", result_sha256="b" * 64,
            result_lines=2, input_ids_sha256=hashes,
        )
        self.assertTrue(validate_run_cap_manifest(manifest))
        actual = manifest["input_identity"]["actual_input_ids"]
        self.assertEqual(actual["per_sample_sha256"], hashes)
        self.assertEqual(actual["count"], 2)

        for path, value in (
            (("source_fingerprint",), "v2-" + "0" * 64),
            (("prompt_template_sha256",), "0" * 64),
            (("actual_input_ids", "ordered_sha256"), "0" * 64),
        ):
            bad = copy.deepcopy(manifest)
            target = bad["input_identity"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(SystemExit):
                validate_run_cap_manifest(bad)

    def test_legacy_manifest_remains_legacy_not_reinterpreted_as_v2(self):
        core = build_run_cap_core(
            model_name="Qwen3-8B", model_path="/model", method="none",
            dataset="qasper", dataset_fingerprint="8a44386d7d5a",
            run_timestamp="legacy", effective_cap=31500,
            cap_gate_info=None, max_new_tokens=128, producer=None,
        )
        manifest = finalize_run_cap_manifest(
            core, status="complete", n_samples=1, input_tokens=[4],
            truncated_flags=[False], result_file="legacy.jsonl",
            result_sha256="c" * 64, result_lines=1,
        )
        self.assertNotIn("input_identity", manifest)
        self.assertNotIn("input_identity", manifest["cap_identity"]["payload"])
        self.assertTrue(validate_run_cap_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
