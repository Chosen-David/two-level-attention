import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "benchmark" / "RULER" / "pred_ruler_batch.py"


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [3]

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class _Model:
    def eval(self):
        return self


class RulerBatchTreatmentSnapshotTests(unittest.TestCase):
    def _load_entry(self, tmp, events, captured, snapshot):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.manual_seed = lambda seed: None
        fake_torch.cuda = types.SimpleNamespace(
            manual_seed=lambda seed: None, empty_cache=lambda: None)
        fake_torch.no_grad = contextlib.nullcontext

        transformers = types.ModuleType("transformers")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                events.append("tokenizer")
                return _Tokenizer()

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return types.SimpleNamespace()

        class AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                events.append("model")
                return _Model()

        transformers.AutoTokenizer = AutoTokenizer
        transformers.AutoConfig = AutoConfig
        transformers.AutoModelForCausalLM = AutoModel

        receipt = types.ModuleType("benchmark.RULER.yarn_receipt")
        receipt._file_sha256 = lambda path: "a" * 64
        receipt.acquire_output_lock = lambda path: object()

        def build_yarn_receipt(**kwargs):
            captured["receipt"] = kwargs
            return {}

        receipt.build_yarn_receipt = build_yarn_receipt
        receipt.commit_yarn_generation = lambda *args: None
        receipt.generation_pointer_path = lambda path: path + ".tli_gen"
        receipt.release_output_lock = lambda fd: None
        receipt.resolve_yarn_config = lambda *args: (None, None)
        receipt.stage_yarn_generation = lambda path, run: path + ".tmp"
        receipt.stage_yarn_receipt = lambda path, data, run: path + ".receipt.tmp"

        arguments = types.ModuleType("sparse_attn.arguments")
        arguments.add_sparse_attn_args = lambda parser: None

        runtime = types.ModuleType("sparse_attn.baseline_runtime")
        runtime.BatchCtx = object
        runtime.baseline_preflight = lambda args: None
        runtime.batch_decode_scope = contextlib.nullcontext

        @contextlib.contextmanager
        def baseline_request(model):
            yield {}

        runtime.baseline_request = baseline_request

        info = types.ModuleType("sparse_attn.info")

        def resolve_treatment_snapshot(args):
            events.append("snapshot")
            return snapshot

        def get_method_name_with_info(args, value):
            captured["name_snapshot"] = value
            return "tli_h0123456789"

        def get_treatment_manifest_json(args, value):
            captured["manifest_snapshot"] = value
            return '{"frozen":true}'

        info.resolve_treatment_snapshot = resolve_treatment_snapshot
        info.get_method_name_with_info = get_method_name_with_info
        info.get_treatment_manifest_json = get_treatment_manifest_json
        info.stage_fasa_agreement = lambda *args: None

        patches = types.ModuleType("sparse_attn.patches")

        def register_patch(model, args, value):
            captured["runtime_snapshot"] = value
            return 1

        patches.register_patch = register_patch

        metrics = types.ModuleType("sparse_attn.metrics")
        metrics.get_metrics = lambda: types.SimpleNamespace(
            clear=lambda: None, get_select_tokens=lambda: 17)

        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda values: values
        numpy = types.ModuleType("numpy")
        numpy.random = types.SimpleNamespace(seed=lambda seed: None)

        module_name = "_pred_ruler_batch_snapshot_test"
        spec = importlib.util.spec_from_file_location(module_name, ENTRY)
        module = importlib.util.module_from_spec(spec)
        injected = {
            "torch": fake_torch,
            "transformers": transformers,
            "benchmark.RULER.yarn_receipt": receipt,
            "sparse_attn.arguments": arguments,
            "sparse_attn.baseline_runtime": runtime,
            "sparse_attn.info": info,
            "sparse_attn.patches": patches,
            "sparse_attn.metrics": metrics,
            "tqdm": tqdm,
            "numpy": numpy,
        }
        guard_module = types.ModuleType('sparse_attn.ops.eager_decoding')
        @contextlib.contextmanager
        def guard():
            yield
            events.append('guard_exit')
            if captured.get('inject_kernel_error'):
                raise RuntimeError('deferred kernel error')
        guard_module.eager_decoding_request_guard = guard
        injected['sparse_attn.ops.eager_decoding'] = guard_module
        with patch.dict(sys.modules, injected):
            spec.loader.exec_module(module)
        return module

    def test_tli_uses_one_preload_snapshot_for_runtime_name_and_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data = tmp / "data" / "4096"
            data.mkdir(parents=True)
            row = {"input": "prompt", "outputs": ["answer"],
                   "length": 4096, "index": 0}
            (data / "niah_single_1.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8")

            events = []
            captured = {}
            snapshot = object()
            module = self._load_entry(tmp, events, captured, snapshot)
            args = types.SimpleNamespace(
                batch_size=1, method="tli", pred_postfix="", yarn=False,
                context_length=4096, yarn_factor=None,
                data_root=str(tmp / "data"), task="niah_single_1",
                max_num=1, model_path=str(tmp / "model"),
                model="Qwen3-8B", output_dir=str(tmp / "out"), t="test")
            module.parse_args = lambda: args
            module._pack_rows = lambda *args: (object(), object(), object(), [1])
            module._decode_batch_chunk = lambda *args, **kwargs: [[1]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                module.main()

            self.assertLess(events.index("snapshot"), events.index("model"))
            self.assertIs(captured["runtime_snapshot"], snapshot)
            self.assertIs(captured["name_snapshot"], snapshot)
            self.assertIs(captured["manifest_snapshot"], snapshot)
            self.assertEqual(
                captured["receipt"]["treatment_manifest_json"],
                '{"frozen":true}')
            self.assertIn('guard_exit', events)

    def test_deferred_kernel_error_prevents_publication(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data = tmp / 'data' / '4096'
            data.mkdir(parents=True)
            (data / 'niah_single_1.jsonl').write_text(json.dumps(
                dict(input='x', outputs=['a'], length=1, index=0))+'\n')
            captured = {'inject_kernel_error': True}
            module = self._load_entry(tmp, [], captured, object())
            module.parse_args = lambda: types.SimpleNamespace(
                batch_size=1, method='tli', pred_postfix='', yarn=False,
                context_length=4096, yarn_factor=None, data_root=str(tmp/'data'),
                task='niah_single_1', max_num=1, model_path=str(tmp/'model'),
                model='Qwen3-8B', output_dir=str(tmp/'out'), t='test')
            module._pack_rows = lambda *a: (None, None, None, [1])
            module._decode_batch_chunk = lambda *a, **kw: [[1]]
            with self.assertRaisesRegex(RuntimeError, 'deferred kernel'):
                module.main()
            self.assertNotIn('receipt', captured)
            self.assertTrue(all(p.stat().st_size == 0 for p in (tmp/'out').rglob('*.tmp')))


if __name__ == "__main__":
    unittest.main()
