from types import SimpleNamespace
import unittest
from unittest import mock

from benchmark.LongBench import perf_vendor
from benchmark.LongBench.pred import parse_args as parse_pred_args


class HostWallRequestTimerTests(unittest.TestCase):
    def test_receipt_uses_token_boundaries_and_records_eos(self):
        timer = perf_vendor.HostWallRequestTimer([7, 9])
        with mock.patch.object(perf_vendor.torch.cuda, "synchronize") as sync, \
                mock.patch.object(
                    perf_vendor.time, "perf_counter",
                    side_effect=[10.0, 10.1, 10.2, 10.3, 10.4]):
            timer.start()
            timer.token_ready(7)
            timer.token_ready(8)
            timer.token_ready(7)
            timer.finish()
        self.assertEqual(sync.call_count, 3)
        self.assertAlmostEqual(timer.receipt["ttft_ms"], 100.0)
        self.assertAlmostEqual(timer.receipt["decode_after_first_wall_ms"], 200.0)
        self.assertAlmostEqual(timer.receipt["tpot_ms"], 100.0)
        self.assertAlmostEqual(timer.receipt["request_wall_ms"], 400.0)
        self.assertEqual(timer.receipt["actual_generated_tokens"], 3)
        self.assertEqual(timer.receipt["generated_token_ids"], [7, 8, 7])
        self.assertEqual(timer.receipt["eos_positions"], [0, 2])
        self.assertFalse(timer.receipt["eos_early_stop"])

    def test_incomplete_request_cannot_finish(self):
        timer = perf_vendor.HostWallRequestTimer(7)
        with mock.patch.object(perf_vendor.torch.cuda, "synchronize"), \
                mock.patch.object(perf_vendor.time, "perf_counter", return_value=1.0):
            timer.start()
            with self.assertRaisesRegex(RuntimeError, "no generated token"):
                timer.finish()


class VendorPerformanceConfigurationTests(unittest.TestCase):
    @staticmethod
    def _args(method):
        return SimpleNamespace(
            tasks=["qasper"], method=method, model_path="/model",
            config_path="/config", data_root="/data")

    def _parsed(self, method):
        return parse_pred_args(perf_vendor._pred_argv(self._args(method)))

    def test_quest_uses_aligned_official_kernel_budget(self):
        args = self._parsed("quest_vendor_kernel")
        self.assertEqual(args.quest_token_budget, 1024)
        self.assertEqual(args.quest_chunk_size, 16)
        self.assertEqual(args.quest_page_size, 16)
        self.assertEqual(args.quest_max_seq_len, 40960)
        self.assertFalse(args.nfa_enable_layer_skip)

    def test_jssa_uses_official_runtime_and_aligned_budget(self):
        args = self._parsed("jssa_official")
        self.assertEqual(args.jssa_budget, 1024)
        self.assertEqual(args.jssa_rank, 32)
        self.assertTrue(args.jssa_basis_path.endswith("bases/qwen3_8b.pt"))
        self.assertFalse(args.nfa_enable_layer_skip)

    def test_asynctls_aligns_both_stages_and_channel_asset(self):
        args = self._parsed("asynctls_vendor")
        self.assertEqual(args.asynctls_token_budget, 1024)
        self.assertEqual(args.asynctls_block_size, 64)
        self.assertEqual(args.asynctls_top_kb, 128)
        self.assertTrue(args.asynctls_channel_file.endswith(
            "dimension_channels/qwen3-8b/qasper_channels_per_group.json"))
        self.assertFalse(args.nfa_enable_layer_skip)


if __name__ == "__main__":
    unittest.main()
