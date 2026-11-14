import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmark.LongBench.run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX,
    compute_result_binding,
    input_ids_sha256 as runcap_input_ids_sha256,
    write_run_cap_manifest,
)
from benchmark.official_baselines.merge_longbench_quality import (
    merge_round_robin_shards,
)
from benchmark.official_baselines.longbench_official import (
    OfficialEngine,
    perf_summary,
    quality_generation_options,
    quality_manifest_path,
    run_quality,
    sha256_file,
    sha256_ids,
    timed_pair,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "benchmark/official_baselines"


def load_runcap_consumer():
    """加载未修改的评分端校验函数，避免测试引入可选评分依赖。"""
    path = ROOT / "benchmark/LongBench/eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_runcap_file_sha256", "_runcap_file_lines", "validate_runcap_sidecar"}
    tree.body = [node for node in tree.body if (
        isinstance(node, ast.Import) and all(
            alias.name in {"os", "hashlib", "sys"} for alias in node.names)
    ) or (
        isinstance(node, ast.ImportFrom) and node.module == "run_cap_manifest"
    ) or (isinstance(node, ast.FunctionDef) and node.name in names)]
    namespace = {"__package__": "benchmark.LongBench"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["validate_runcap_sidecar"]


class _TokenList(list):
    def tolist(self):
        return list(self)


class _QualityTokenizer:
    eos_token_id = 99

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[_TokenList(range(len(text) % 7 + 3))])

    def encode(self, text, add_special_tokens=False):
        self.last_encode = (text, add_special_tokens)
        return [198] if text == "\n" else [7]

    def decode(self, ids, skip_special_tokens=True):
        return "generated"


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

    def test_warmup_pair_excludes_cold_negative_delta_but_formal_rejects(self):
        class Tokenizer:
            eos_token_id = 99

        class Engine:
            tokenizer = Tokenizer()

            def generate(self, ids, max_new_tokens, ignore_eos):
                elapsed = 3.3 if max_new_tokens == 1 else 0.1
                return list(range(max_new_tokens)), elapsed

        generated, timing = timed_pair(
            Engine(), [5, 6], 33, require_positive=False
        )
        self.assertEqual(len(generated), 33)
        self.assertLess(timing["tpot_ms"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "两点法 decode 时间非正"):
            timed_pair(Engine(), [5, 6], 33)

    def test_quest_quality_stops_and_perf_length_contract(self):
        import torch

        engine = OfficialEngine.__new__(OfficialEngine)
        engine.args = SimpleNamespace(method="quest", device="cpu")
        engine.torch = torch
        engine.tokenizer = _QualityTokenizer()
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            ids = kwargs["input_ids"][0].tolist()
            return torch.tensor([ids + [7, 8]])

        engine.model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[99, 100]),
            generate=generate,
            reset_quest_controller=lambda: None,
        )
        options = quality_generation_options("samsum", engine.tokenizer)
        self.assertEqual(options, {
            "stop_token_ids": [198], "stop_strings": ["\n"], "min_new_tokens": 1,
        })
        with mock.patch.object(torch.cuda, "synchronize", lambda: None):
            engine._generate_quest([5, 6, 7], 32, False, **options)
            engine._generate_quest([5, 6, 7], 33, True)
        self.assertEqual(calls[0]["min_length"], 4)
        self.assertEqual(calls[0]["eos_token_id"], [99, 100, 198])
        self.assertEqual(calls[0]["stop_strings"], ["\n"])
        self.assertIs(calls[0]["tokenizer"], engine.tokenizer)
        self.assertTrue(torch.equal(
            calls[0]["attention_mask"], torch.ones_like(calls[0]["input_ids"])
        ))
        self.assertFalse(calls[0]["do_sample"])
        self.assertEqual(calls[1]["min_length"], 3)
        self.assertIsNone(calls[1]["eos_token_id"])
        self.assertIsNone(calls[1]["stop_strings"])

    def test_jssa_quality_stops_and_perf_length_contract(self):
        engine = OfficialEngine.__new__(OfficialEngine)
        engine.args = SimpleNamespace(method="jssa")
        engine.torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None)
        )
        engine.tokenizer = _QualityTokenizer()
        engine.TokensPrompt = lambda prompt_token_ids: SimpleNamespace(
            prompt_token_ids=prompt_token_ids
        )
        calls = []

        def sampling_params(**kwargs):
            calls.append(kwargs)
            return kwargs

        engine.SamplingParams = sampling_params
        engine.model = SimpleNamespace(generate=lambda *args, **kwargs: [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7, 8])])
        ])
        engine._jssa_calls = {
            "fp8_projected_store": 1,
            "radix_score_topk": 1,
            "indexed_attend": 1,
        }
        options = quality_generation_options("samsum", engine.tokenizer)
        engine._generate_jssa([5, 6, 7], 32, False, **options)
        engine._generate_jssa([5, 6, 7], 33, True)
        self.assertEqual(calls[0]["min_tokens"], 1)
        self.assertEqual(calls[0]["stop_token_ids"], [198])
        self.assertEqual(calls[0]["stop"], ["\n"])
        self.assertTrue(calls[0]["include_stop_str_in_output"])
        self.assertFalse(calls[0]["ignore_eos"])
        self.assertEqual(calls[1]["min_tokens"], 0)
        self.assertIsNone(calls[1]["stop_token_ids"])
        self.assertIsNone(calls[1]["stop"])
        self.assertFalse(calls[1]["include_stop_str_in_output"])
        self.assertTrue(calls[1]["ignore_eos"])

    def test_non_samsum_only_suppresses_first_eos(self):
        self.assertEqual(
            quality_generation_options("qasper", _QualityTokenizer()),
            {"stop_token_ids": [], "min_new_tokens": 1},
        )

    def test_quest_hf_rope_matches_hf_four_dimensional_layout(self):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from benchmark.official_baselines.qwen3_quest_kernel import (
            _apply_hf_position_embeddings,
        )

        torch.manual_seed(20261113)
        query = torch.randn(7, 4, 16, dtype=torch.float32)
        key = torch.randn(7, 2, 16, dtype=torch.float32)
        cos = torch.randn(1, 7, 16, dtype=torch.float32)
        sin = torch.randn(1, 7, 16, dtype=torch.float32)
        actual_q, actual_k = _apply_hf_position_embeddings(query, key, (cos, sin))
        expected_q, expected_k = apply_rotary_pos_emb(
            query.unsqueeze(0), key.unsqueeze(0), cos, sin, unsqueeze_dim=2
        )
        self.assertTrue(torch.equal(actual_q, expected_q[0]))
        self.assertTrue(torch.equal(actual_k, expected_k[0]))

    def test_quest_rope_mode_rejects_native_yarn_and_hf_default(self):
        from benchmark.official_baselines.qwen3_quest_kernel import (
            _validate_quest_rope_mode,
        )

        yarn = SimpleNamespace(
            rope_scaling={"rope_type": "yarn", "factor": 2.0},
            rope_parameters=None,
        )
        default = SimpleNamespace(rope_scaling=None, rope_parameters=None)
        _validate_quest_rope_mode(default, "native")
        _validate_quest_rope_mode(yarn, "hf-position-embeddings")
        with self.assertRaisesRegex(ValueError, "native RoPE"):
            _validate_quest_rope_mode(yarn, "native")
        with self.assertRaisesRegex(ValueError, "仅验收 YaRN"):
            _validate_quest_rope_mode(default, "hf-position-embeddings")

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

    def test_long_context_yarn_is_configured_before_engine_construction(self):
        from transformers import Qwen3Config
        from benchmark.official_baselines.long_context_official import (
            configure_long_context_rope,
        )

        for context_length, expected_factor in ((262144, 8.0), (524288, 16.0)):
            for method in ("quest", "jssa"):
                with self.subTest(context_length=context_length, method=method):
                    config = Qwen3Config(
                        hidden_size=64,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=16,
                        max_position_embeddings=40960,
                    )
                    args = SimpleNamespace(
                        method=method,
                        context_length=context_length,
                        max_new_tokens=33,
                    )
                    policy, preflight = configure_long_context_rope(args, config)
                    self.assertIs(args.model_config, config)
                    self.assertEqual(args.quest_rope_mode, "hf-position-embeddings")
                    self.assertEqual(
                        args.hf_overrides["rope_scaling"]["factor"], expected_factor
                    )
                    self.assertEqual(
                        args.hf_overrides["max_position_embeddings"],
                        context_length + 33,
                    )
                    self.assertEqual(policy["requested_factor"], expected_factor)
                    self.assertEqual(preflight["status"], "verified")
                    self.assertEqual(preflight["requested_factor"], expected_factor)

    def test_long_context_runtime_verification_uses_constructed_engine(self):
        from benchmark.official_baselines import long_context_official

        quest_model = object()
        quest_engine = SimpleNamespace(model=quest_model)
        runtime_config = object()
        jssa_engine = SimpleNamespace(
            model=SimpleNamespace(
                llm_engine=SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        model_config=SimpleNamespace(hf_config=runtime_config)
                    )
                )
            )
        )
        with mock.patch.object(
            long_context_official, "verify_runtime_rope", return_value={"status": "verified"}
        ) as verify:
            self.assertEqual(
                long_context_official.verify_engine_rope(quest_engine, "quest"),
                {"status": "verified"},
            )
            verify.assert_called_once_with(quest_model)
            verify.reset_mock()
            long_context_official.verify_engine_rope(jssa_engine, "jssa")
            verify.assert_called_once_with(runtime_config)

    def test_long_context_input_identity_and_length_contract(self):
        from benchmark.official_baselines.long_context_official import (
            load_performance_input,
        )

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=[range(100)])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "100" / "toy.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps({"input": "performance text", "index": 7}) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data_root=str(root), context_length=100, task="toy",
                sample_index=7,
                input_provenance="repeated-needle-body-extension",
            )
            ids, receipt = load_performance_input(args, Tokenizer())
            self.assertEqual(ids, list(range(100)))
            self.assertEqual(receipt["actual_prompt_tokens"], 100)
            self.assertEqual(receipt["file_row_index"], 0)
            self.assertEqual(receipt["sample_index"], 7)
            self.assertEqual(receipt["sample_id"], 7)
            self.assertEqual(receipt["sha256"], sha256_file(source))
            self.assertTrue(receipt["performance_only"])

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

    def _execute_quality_with_runcap(
        self, root, sample_indices, shard_index=0, num_shards=1, fail=False,
        receipts=False,
    ):
        rows = [
            {
                "context": f"source row {index}", "answers": ["answer"],
                "all_classes": [], "length": 0,
            }
            for index in range(3)
        ]
        (root / "samsum.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        args = SimpleNamespace(
            output_dir=str(root / "out"), data_root=str(root),
            model_path=str(root / "model"), model_name="Qwen3-8B",
            method="nfa" if receipts else "jssa",
            budget=1024, jssa_rank=32, quest_page_size=16,
            shard_index=shard_index, num_shards=num_shards,
            tasks=["samsum"], sample_indices=sample_indices,
        )
        if receipts:
            args.quality_label = "nfa_graph"
            args.quality_implementation = "nfa_graph"
            args.quality_producer_file = str(
                ROOT / "benchmark/efficiency/benchmark_longbench_quality.py"
            )

        class Engine:
            runtime = {"implementation": "test-official-jssa"}

            def __init__(self):
                self.tokenizer = _QualityTokenizer()
                self.calls = []
                if receipts:
                    self.request_receipts = []

            def generate(self, ids, max_new_tokens, **kwargs):
                self.calls.append((list(ids), max_new_tokens, kwargs))
                if fail:
                    raise RuntimeError("injected generation interruption")
                if receipts:
                    self.request_receipts.append({
                        "generated_tokens": 1,
                        "stop_reason": "eos",
                        "logical_decode_steps": 0,
                        "submitted_decode_forwards": 0,
                    })
                return [7], 0.1

        engine = Engine()
        cap_info = {
            "registry_protocol_cap": 31500,
            "registry_restricted_context_treatment": None,
            "capability": 131072,
            "max_position_embeddings": 131072,
        }
        if fail:
            with self.assertRaisesRegex(RuntimeError, "injected generation"):
                run_quality(
                    args, engine, {"samsum": "{context}"}, {"samsum": 32},
                    31500, cap_info,
                )
            return args, engine, None
        run_quality(
            args, engine, {"samsum": "{context}"}, {"samsum": 32}, 31500,
            cap_info,
        )
        manifest_path = quality_manifest_path(
            args.output_dir, shard_index, num_shards
        )
        return args, engine, json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_runcap_full_shard_passes_unchanged_strict_consumer(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engine, manifest = self._execute_quality_with_runcap(
                Path(temporary), None, shard_index=1, num_shards=2
            )
            entry = manifest["tasks"]["samsum"]
            prediction = Path(args.output_dir) / entry["path"]
            sidecar = Path(str(prediction) + RUN_CAP_SIDECAR_SUFFIX)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["n_samples"], 1)
            self.assertEqual(payload["result_sha256"], sha256_file(prediction))
            self.assertEqual(payload["sample_sharding"]["selected_samples"], 1)
            self.assertEqual(payload["input_identity"]["sample_count"], 3)
            self.assertEqual(
                payload["input_identity"]["actual_input_ids"]["per_sample_sha256"],
                [runcap_input_ids_sha256(engine.calls[0][0])],
            )
            self.assertEqual(engine.calls[0][2]["min_new_tokens"], 1)
            self.assertEqual(engine.calls[0][2]["stop_token_ids"], [198])
            receipt = load_runcap_consumer()(str(prediction), False)
            self.assertEqual(receipt["dataset"], "samsum")
            with prediction.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(SystemExit, "sha256"):
                load_runcap_consumer()(str(prediction), False)

    def test_runcap_filtered_subset_preserves_source_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engine, manifest = self._execute_quality_with_runcap(
                Path(temporary), [0, 2]
            )
            prediction = Path(args.output_dir) / manifest["tasks"]["samsum"]["path"]
            sidecar = Path(str(prediction) + RUN_CAP_SIDECAR_SUFFIX)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertNotIn("sample_sharding", payload)
            self.assertEqual(payload["input_identity"]["sample_count"], 2)
            self.assertEqual(payload["source_input_identity"]["sample_count"], 3)
            self.assertEqual(payload["sample_selection"]["source_indices"], [0, 2])
            self.assertEqual(len(engine.calls), 2)
            load_runcap_consumer()(str(prediction), False)

    def test_runcap_empty_filtered_subset_is_valid_not_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engine, manifest = self._execute_quality_with_runcap(
                Path(temporary), []
            )
            prediction = Path(args.output_dir) / manifest["tasks"]["samsum"]["path"]
            sidecar = Path(str(prediction) + RUN_CAP_SIDECAR_SUFFIX)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(prediction.read_bytes(), b"")
            self.assertEqual(engine.calls, [])
            self.assertEqual(payload["n_samples"], 0)
            self.assertEqual(
                payload["input_identity"]["actual_input_ids"]["count"], 0
            )
            load_runcap_consumer()(str(prediction), False)

    def test_quality_receipts_survive_complete_skip_with_bound_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, engine, manifest = self._execute_quality_with_runcap(
                root, [0, 2], receipts=True
            )
            entry = manifest["tasks"]["samsum"]
            receipt_path = Path(args.output_dir) / entry["receipts"]
            prediction_path = Path(args.output_dir) / entry["path"]
            self.assertEqual(entry["receipt_rows"], 2)
            self.assertEqual(entry["receipts_sha256"], sha256_file(receipt_path))
            self.assertEqual(len(manifest["request_receipts"]), 2)
            self.assertEqual(
                [row["source_index"] for row in manifest["request_receipts"]],
                [0, 2],
            )
            sidecar = json.loads(Path(
                str(prediction_path) + RUN_CAP_SIDECAR_SUFFIX
            ).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["method"], "nfa_graph")
            self.assertTrue(all(
                json.loads(line)["implementation"] == "nfa_graph"
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
            ))

            args, skipped_engine, skipped = self._execute_quality_with_runcap(
                root, [0, 2], receipts=True
            )
            self.assertEqual(skipped_engine.calls, [])
            self.assertEqual(skipped["request_receipts"], manifest["request_receipts"])
            self.assertEqual(len(skipped_engine.request_receipts), 2)

    def test_quality_receipts_skip_fails_closed_when_missing_or_mismatched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, manifest = self._execute_quality_with_runcap(
                root, [0, 2], receipts=True
            )
            receipt_path = Path(args.output_dir) / manifest["tasks"]["samsum"]["receipts"]
            original = receipt_path.read_text(encoding="utf-8")
            receipt_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "缺少逐请求回执"):
                self._execute_quality_with_runcap(root, [0, 2], receipts=True)

            receipt_path.write_text(original, encoding="utf-8")
            receipts = json.loads(original)
            receipts[0]["source_index"] = 99
            receipt_path.write_text(json.dumps(receipts), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "回执与当前输入身份不一致"):
                self._execute_quality_with_runcap(root, [0, 2], receipts=True)

    def test_partial_prediction_rerun_discards_stale_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _, manifest = self._execute_quality_with_runcap(
                root, [0, 2], receipts=True
            )
            entry = manifest["tasks"]["samsum"]
            prediction_path = Path(args.output_dir) / entry["path"]
            receipt_path = Path(args.output_dir) / entry["receipts"]
            prediction_path.write_text(
                prediction_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            stale = json.loads(receipt_path.read_text(encoding="utf-8"))
            stale.append({"stale": True})
            receipt_path.write_text(json.dumps(stale), encoding="utf-8")

            _, rerun_engine, rerun = self._execute_quality_with_runcap(
                root, [0, 2], receipts=True
            )
            self.assertEqual(len(rerun_engine.calls), 2)
            self.assertEqual(len(rerun["request_receipts"]), 2)
            self.assertNotIn("stale", receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                rerun["tasks"]["samsum"]["receipts_sha256"],
                sha256_file(receipt_path),
            )

    def test_interrupted_generation_leaves_strictly_rejected_runcap(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = self._execute_quality_with_runcap(
                Path(temporary), None, fail=True
            )
            predictions = list((Path(args.output_dir) / "pred").glob("*.jsonl"))
            self.assertEqual(len(predictions), 1)
            prediction = predictions[0]
            sidecar = Path(str(prediction) + RUN_CAP_SIDECAR_SUFFIX)
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8"))["status"],
                "generating",
            )
            with self.assertRaisesRegex(SystemExit, "非 complete"):
                load_runcap_consumer()(str(prediction), False)

    def _make_round_robin_shards(self, root, world=4):
        config = root / "config"
        config.mkdir()
        (config / "dataset2prompt.json").write_text(
            json.dumps({"samsum": "{context}"}), encoding="utf-8"
        )
        shard_paths = []
        shard_manifests = []
        for rank in range(world):
            args, _, quality_manifest = self._execute_quality_with_runcap(
                root, None, shard_index=rank, num_shards=world
            )
            shard_paths.append(Path(args.output_dir) / quality_manifest["tasks"]["samsum"]["path"])
            shard_manifests.append(json.loads(
                Path(str(shard_paths[-1]) + RUN_CAP_SIDECAR_SUFFIX).read_text(encoding="utf-8")
            ))
        output = Path(args.output_dir) / "pred" / "samsum-jssa_official_vllm_b1024_r32.jsonl"
        return shard_paths, shard_manifests, output

    def test_round_robin_merge_restores_source_order_and_canonical_runcap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards, manifests, output = self._make_round_robin_shards(root)
            merged, manifest = merge_round_robin_shards(
                list(reversed(shards)), output, "samsum", root,
                root / "config",
            )
            self.assertEqual(merged, output.resolve())
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["_id"] for row in records], ["samsum:0", "samsum:1", "samsum:2"])
            self.assertEqual(manifest["n_samples"], 3)
            self.assertNotIn("sample_sharding", manifest)
            self.assertEqual(manifest["input_identity"]["sample_count"], 3)
            self.assertEqual(
                manifest["input_tokens"]["per_sample"],
                [manifests[0]["input_tokens"]["per_sample"][0],
                 manifests[1]["input_tokens"]["per_sample"][0],
                 manifests[2]["input_tokens"]["per_sample"][0]],
            )
            self.assertEqual(
                manifest["input_identity"]["actual_input_ids"]["per_sample_sha256"],
                [manifests[rank]["input_identity"]["actual_input_ids"]["per_sample_sha256"][0]
                 for rank in range(3)],
            )
            self.assertEqual(manifest["truncated"]["flags"], [False, False, False])
            self.assertEqual(manifest["run_timestamp"], "canonical")
            self.assertEqual(
                [entry["rank"] for entry in manifest["canonical_merge"]["shards"]],
                [0, 1, 2, 3],
            )
            self.assertEqual(load_runcap_consumer()(str(output), False)["dataset"], "samsum")

    def test_round_robin_merge_rejects_duplicate_rank_and_record_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards, _, output = self._make_round_robin_shards(root)
            with self.assertRaisesRegex(RuntimeError, "恰好覆盖"):
                merge_round_robin_shards(
                    shards[:-1] + [shards[0]], output, "samsum", root,
                    root / "config",
                )
            lines = shards[0].read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[0])
            record["_id"] = "samsum:2"
            shards[0].write_text(json.dumps(record) + "\n", encoding="utf-8")
            sidecar_path = str(shards[0]) + RUN_CAP_SIDECAR_SUFFIX
            sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
            result_sha, result_lines = compute_result_binding(str(shards[0]))
            sidecar["result_sha256"] = result_sha
            sidecar["result_lines"] = result_lines
            write_run_cap_manifest(sidecar_path, sidecar)
            with self.assertRaisesRegex(RuntimeError, "记录顺序/身份错误"):
                merge_round_robin_shards(
                    shards, output, "samsum", root,
                    root / "config",
                )

    def test_round_robin_merge_rejects_source_and_provenance_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards, _, output = self._make_round_robin_shards(root)
            sidecar_path = Path(str(shards[0]) + RUN_CAP_SIDECAR_SUFFIX)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["input_tokens"]["per_sample"][0] += 1
            write_run_cap_manifest(str(sidecar_path), sidecar)
            with self.assertRaisesRegex(RuntimeError, "行内 input 与 runcap"):
                merge_round_robin_shards(
                    shards, output, "samsum", root,
                    root / "config",
                )
            sidecar["input_tokens"]["per_sample"][0] -= 1
            write_run_cap_manifest(str(sidecar_path), sidecar)
            input_hashes = sidecar["input_identity"]["actual_input_ids"]["per_sample_sha256"]
            original_hash = input_hashes[0]
            input_hashes[0] = "0" * 64
            write_run_cap_manifest(str(sidecar_path), sidecar)
            with self.assertRaisesRegex(SystemExit, "ordered_sha256"):
                merge_round_robin_shards(
                    shards, output, "samsum", root,
                    root / "config",
                )
            input_hashes[0] = original_hash
            write_run_cap_manifest(str(sidecar_path), sidecar)
            rows = [json.loads(line) for line in (root / "samsum.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["answers"] = ["changed"]
            (root / "samsum.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "完整源身份"):
                merge_round_robin_shards(
                    shards, output, "samsum", root,
                    root / "config",
                )


if __name__ == "__main__":
    unittest.main()
