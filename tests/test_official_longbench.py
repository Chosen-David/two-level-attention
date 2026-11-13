import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.official_baselines.longbench_official import (
    perf_summary,
    quality_manifest_path,
    run_quality,
    sha256_ids,
    timed_pair,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "benchmark/official_baselines"


class OfficialLongBenchIndependenceTest(unittest.TestCase):
    def test_no_nfa_or_vendor_imports(self):
        forbidden = (
            "sparse_attn",
            "exp.vendor_baselines",
            "benchmark.LongBench.pred",
            "evaluation.quest_attention",
            "qwen3_jssa_hf",
        )
        offenders = []
        for path in OFFICIAL_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if any(name == item or name.startswith(item + ".") for item in forbidden):
                        offenders.append((str(path), name))
        self.assertEqual(offenders, [])

    def test_perf_summary_uses_task_median_then_macro_mean(self):
        cells = []
        for task, values in (("qasper", [1.0, 3.0]), ("trec", [10.0, 20.0])):
            for repeat, value in enumerate(values):
                cells.append({
                    "task": task,
                    "request_timing": {
                        "ttft_ms": value,
                        "tpot_ms": value * 2,
                        "request_wall_ms": value * 3,
                    },
                })
        summary = perf_summary(cells)
        self.assertEqual(summary["per_task"]["qasper"]["tpot_ms"], 4.0)
        self.assertEqual(summary["per_task"]["trec"]["tpot_ms"], 30.0)
        self.assertEqual(summary["macro_task_mean"]["tpot_ms"], 17.0)
        self.assertEqual(summary["all_request_median"]["request_wall_ms"], 19.5)

    def test_input_id_hash_is_stable_and_order_sensitive(self):
        self.assertEqual(sha256_ids([1, 2, 3]), sha256_ids([1, 2, 3]))
        self.assertNotEqual(sha256_ids([1, 2, 3]), sha256_ids([1, 3, 2]))

    def test_two_point_timing_excludes_prefill(self):
        class Tokenizer:
            eos_token_id = 99

        class Engine:
            tokenizer = Tokenizer()

            def generate(self, ids, max_new_tokens, ignore_eos):
                self_ids = list(ids)
                self.assert_ids = self_ids
                elapsed = 0.1 if max_new_tokens == 1 else 3.3
                return list(range(max_new_tokens)), elapsed

        generated, timing = timed_pair(Engine(), [5, 6], 33)
        self.assertEqual(len(generated), 33)
        self.assertAlmostEqual(timing["ttft_ms"], 100.0)
        self.assertAlmostEqual(timing["tpot_ms"], 100.0)
        self.assertAlmostEqual(timing["request_wall_ms"], 3300.0)
        self.assertFalse(timing["eos_early_stop"])

    def test_long_prefill_tokenwise_chunk_is_bit_exact(self):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        from benchmark.official_baselines.qwen3_quest_kernel import (
            _tokenwise_prefill_chunk,
        )

        torch.manual_seed(20261113)
        norm = Qwen3RMSNorm(16, eps=1e-6)
        hidden = torch.randn(1, 100, 16, dtype=torch.float32)
        expected = norm(hidden)
        actual = _tokenwise_prefill_chunk(norm.forward, hidden, 17)
        self.assertTrue(torch.equal(actual, expected))
        short = hidden[:, :17]
        self.assertTrue(torch.equal(
            _tokenwise_prefill_chunk(norm.forward, short, 17), norm(short)
        ))
        heads = torch.randn(1, 100, 4, 16, dtype=torch.float32)
        self.assertTrue(torch.equal(
            _tokenwise_prefill_chunk(norm.forward, heads, 17), norm(heads)
        ))

    @unittest.skipUnless(
        os.environ.get("RUN_QUEST_CUDA_TESTS") == "1",
        "显式设置 RUN_QUEST_CUDA_TESTS=1 后运行 Quest CUDA 对拍",
    )
    def test_quest_native_gqa_matches_repeated_reference(self):
        import torch
        from benchmark.official_baselines.qwen3_quest_kernel import quest_utils

        device = torch.device("cuda:0")
        dtype = torch.float16
        groups = 4
        num_heads = 32
        num_kv_heads = num_heads // groups
        head_dim = 128
        page_size = 16
        common = dict(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            page_budget=4,
            max_seq_len=160,
            dtype=dtype,
            device=device,
        )
        reference = quest_utils.InferenceController(**common)
        native = quest_utils.InferenceController(**common, num_kv_heads=num_kv_heads)

        def assert_equal(actual, expected, label):
            self.assertTrue(
                torch.equal(actual, expected),
                f"{label}: max diff={(actual.float() - expected.float()).abs().max().item()}",
            )

        def check_active_pages():
            ref_indices = torch.tensor(reference.kv_cache.indicies, device=device)
            native_indices = torch.tensor(native.kv_cache.indicies, device=device)
            assert_equal(native_indices, ref_indices, "KV page indices")
            ref_kv = reference.kv_cache.buf_layer(0)[ref_indices]
            native_kv = native.kv_cache.buf_layer(0)[native_indices]
            assert_equal(native_kv, ref_kv[:, :, :, 0::groups, :], "physical KV")
            ref_meta_indices = torch.tensor(
                reference.metadata_cache.indicies, device=device
            )
            native_meta_indices = torch.tensor(
                native.metadata_cache.indicies, device=device
            )
            assert_equal(native_meta_indices, ref_meta_indices, "metadata page indices")
            ref_meta = reference.metadata_cache.buf_layer(0)[ref_meta_indices]
            native_meta = native.metadata_cache.buf_layer(0)[native_meta_indices]
            assert_equal(native_meta, ref_meta, "Quest metadata")

        def run_step(q_len, generator):
            q = torch.randn(q_len, num_heads, head_dim, device=device, dtype=dtype, generator=generator)
            k = torch.randn(q_len, num_kv_heads, head_dim, device=device, dtype=dtype, generator=generator)
            v = torch.randn(q_len, num_kv_heads, head_dim, device=device, dtype=dtype, generator=generator)
            repeated_k = k.repeat_interleave(groups, dim=1).contiguous()
            repeated_v = v.repeat_interleave(groups, dim=1).contiguous()
            ref_q = q.clone()
            native_q = q.clone()
            ref_k = repeated_k.clone()
            native_k = k.clone()
            past_len = reference.kv_cache.seqlen
            for controller in (reference, native):
                controller.prepare_metadata(q_len)
                controller.begin_forward(q_len)
            quest_utils.apply_rope_in_place(ref_q, ref_k, past_len)
            quest_utils.apply_rope_in_place(native_q, native_k, past_len)
            assert_equal(native_q, ref_q, "RoPE Q")
            assert_equal(native_k, ref_k[:, 0::groups, :], "RoPE K")
            quest_utils.append_kv(ref_k, repeated_v, reference, 0)
            quest_utils.append_kv(native_k, v, native, 0)
            if q_len > 1:
                ref_output = quest_utils.prefill_forward(ref_q, reference, 0)
                native_output = quest_utils.prefill_forward(native_q, native, 0)
                assert_equal(native_output, ref_output, "prefill attention")
            else:
                self.assertTrue(reference.need_estimate())
                self.assertTrue(native.need_estimate())
                ref_score = quest_utils.decode_estimate(ref_q, reference, 0)
                native_score = quest_utils.decode_estimate(native_q, native, 0)
                assert_equal(native_score, ref_score, "decode estimate")
                quest_utils.decode_topk(ref_score, reference)
                quest_utils.decode_topk(native_score, native)
                assert_equal(
                    native.topk_dindices_buffer,
                    reference.topk_dindices_buffer,
                    "decode Top-K indices",
                )
                ref_output = quest_utils.decode_sparse_attn(
                    ref_q, reference, 0, reference.topk_dindices_buffer
                )
                native_output = quest_utils.decode_sparse_attn(
                    native_q, native, 0, native.topk_dindices_buffer
                )
                head_diff = (native_output.float() - ref_output.float()).abs().amax(dim=(0, 2))
                assert_equal(
                    native_output,
                    ref_output,
                    f"decode attention per-head={head_diff.tolist()}",
                )
            reference.end_forward()
            native.end_forward()
            check_active_pages()

        try:
            generator = torch.Generator(device=device).manual_seed(20261113)
            run_step(127, generator)
            for _ in range(5):
                run_step(1, generator)
        finally:
            reference.clean_states()
            native.clean_states()

    def test_quality_manifest_is_shard_scoped_and_empty_shard_is_valid(self):
        self.assertNotEqual(
            quality_manifest_path("out", 0, 2),
            quality_manifest_path("out", 1, 2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trec.jsonl").write_text(
                json.dumps({"answers": [], "all_classes": [], "length": 0}) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_dir=str(root / "out"),
                data_root=str(root),
                model_path="unused-test-model",
                method="jssa",
                budget=1024,
                jssa_rank=32,
                quest_page_size=16,
                shard_index=0,
                num_shards=1,
                tasks=["trec"],
                sample_indices=[999],
            )
            engine = SimpleNamespace(runtime={"implementation": "test"})
            run_quality(args, engine, {"trec": "{context}"}, {"trec": 1}, 31500)
            manifest_path = quality_manifest_path(args.output_dir, 0, 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tasks"]["trec"]["rows"], 0)
            prediction = root / "out" / manifest["tasks"]["trec"]["path"]
            self.assertEqual(prediction.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
