"""RULER 正式质量入口的主机侧人口、评分与完成回执测试。"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.LongBench.run_cap_manifest import (
    input_ids_sha256,
    read_run_cap_manifest,
)
from benchmark.RULER.pred_ruler import MIN_NEW_TOKENS, _ruler_max_new_tokens
from benchmark.efficiency.benchmark_ruler_quality import (
    load_ruler_rows,
    ruler_official_score,
    run_quality,
    select_ruler_rows,
)


def test_niah_formal_generation_budget():
    assert _ruler_max_new_tokens("niah_multikey_2") == 128
    assert MIN_NEW_TOKENS == 1
    with pytest.raises(ValueError, match="unknown RULER task"):
        _ruler_max_new_tokens("not-a-ruler-task")


def _rows(count=5):
    return [
        {"index": idx, "input": str(idx + 1), "outputs": [str(idx + 1)],
         "length": 100 + idx}
        for idx in range(count)
    ]


def test_source_order_shard_and_explicit_selection_are_deterministic():
    rows = _rows()
    assert [idx for idx, _ in select_ruler_rows(rows, 1, 2)] == [1, 3]
    assert [idx for idx, _ in select_ruler_rows(rows, 1, 2, [0, 1, 2, 3])] == [1, 3]
    with pytest.raises(ValueError, match="不得重复"):
        select_ruler_rows(rows, 0, 1, [0, 0])
    with pytest.raises(ValueError, match="超出"):
        select_ruler_rows(rows, 0, 1, [5])
    with pytest.raises(ValueError, match="为空"):
        select_ruler_rows(rows, 1, 2, [0])


def test_ruler_source_rejects_row_index_identity_mismatch(tmp_path):
    path = tmp_path / "32768" / "niah_multikey_2.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**_rows(1)[0], "index": 7}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index 必须等于有序行号"):
        load_ruler_rows(tmp_path, 32768, "niah_multikey_2")


def test_signed_int64_input_hash_contract():
    values = [0, -1, (1 << 63) - 1]
    expected = hashlib.sha256(b"".join(
        value.to_bytes(8, "little", signed=True) for value in values)).hexdigest()
    assert input_ids_sha256(values) == expected
    with pytest.raises(ValueError, match="outside signed int64"):
        input_ids_sha256([1 << 63])


def test_official_per_sample_score_is_not_a_custom_metric():
    assert ruler_official_score("prefix AbC suffix", ["abc", "missing"]) == 50.0
    assert ruler_official_score("6073160", ["6073160"]) == 100.0
    assert ruler_official_score("wrong", ["6073160"]) == 0.0


class _FakeTokenizer:
    def __call__(self, text, truncation=False):
        assert truncation is False
        return {"input_ids": [int(text)]}

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(value) for value in ids)


class _FakeEngine:
    def __init__(self, *, fail_on_second=False, max_new_stop=False):
        self.tokenizer = _FakeTokenizer()
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=32768 + 128))
        self.runtime = {"implementation": "host-contract-test"}
        self.request_receipts = []
        self.calls = 0
        self.fail_on_second = fail_on_second
        self.max_new_stop = max_new_stop

    def generate(self, ids, max_new_tokens, ignore_eos, min_new_tokens=0):
        assert max_new_tokens == 128
        assert ignore_eos is False
        assert min_new_tokens == 1
        self.calls += 1
        if self.fail_on_second and self.calls == 2:
            raise RuntimeError("injected generation failure")
        generated = ids * max_new_tokens if self.max_new_stop else list(ids)
        stop_reason = "max_new_tokens" if self.max_new_stop else "eos"
        self.request_receipts.append({
            "generated_tokens": len(generated),
            "stop_reason": stop_reason,
            "logical_decode_steps": len(generated) - 1,
        })
        return generated, 0.0


def _args(tmp_path):
    return SimpleNamespace(
        task="niah_multikey_2", context_length=32768,
        shard_index=0, num_shards=1, sample_indices=None,
        output_dir=str(tmp_path), quality_label="nfa_tail_gate_off_eager",
        quality_implementation="nfa_tail_gate_off_eager",
        quality_producer_file=str(Path(
            "benchmark/efficiency/benchmark_ruler_quality.py").resolve()),
        model_name="Qwen3-8B", model_path="/models/Qwen3-8B", method="nfa",
    )


def test_complete_outputs_scores_and_idempotent_skip(tmp_path):
    args = _args(tmp_path)
    engine = _FakeEngine()
    manifest = run_quality(args, engine, _rows(2), "source-sha")
    assert engine.calls == 2
    assert manifest["rows"] == 2
    assert manifest["official_score"] == 100.0
    prediction = tmp_path / manifest["prediction"]
    records = [json.loads(line) for line in prediction.read_text().splitlines()]
    assert [record["_id"] for record in records] == [
        "niah_multikey_2:0", "niah_multikey_2:1"]
    assert all(record["generated_tokens"] == 1 for record in records)
    assert all(record["output_truncated"] is False for record in records)
    runcap = read_run_cap_manifest(str(prediction) + ".runcap.json")
    assert runcap["status"] == "complete"
    assert runcap["result_lines"] == 2

    skipped_engine = _FakeEngine()
    skipped = run_quality(args, skipped_engine, _rows(2), "source-sha")
    assert skipped_engine.calls == 0
    assert len(skipped_engine.request_receipts) == 2
    assert skipped["prediction_sha256"] == manifest["prediction_sha256"]
    assert skipped["scores_sha256"] == manifest["scores_sha256"]


def test_partial_or_failed_generation_never_marks_complete(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        run_quality(args, _FakeEngine(fail_on_second=True), _rows(2), "source-sha")
    prediction = next((tmp_path / "L32768" / "pred").glob("*.jsonl"))
    assert len(prediction.read_text().splitlines()) == 1
    runcap = read_run_cap_manifest(str(prediction) + ".runcap.json")
    assert runcap["status"] == "generating"
    assert runcap["result_sha256"] is None
    assert not (tmp_path / "quality_manifest_L32768_s0of1.json").exists()


def test_max_new_stop_is_explicitly_marked_truncated(tmp_path):
    args = _args(tmp_path)
    manifest = run_quality(
        args, _FakeEngine(max_new_stop=True), _rows(1), "source-sha")
    record = json.loads((tmp_path / manifest["prediction"]).read_text())
    assert record["generated_tokens"] == 128
    assert record["stop_reason"] == "max_new_tokens"
    assert record["output_truncated"] is True
