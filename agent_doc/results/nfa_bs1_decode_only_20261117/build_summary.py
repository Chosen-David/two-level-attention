#!/usr/bin/env python3
"""对账 BS1 direct decode clean ABBA 与独立 nsys，并冻结原始证据哈希。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("nfa", "asynctls", "jssa", "quest")
LENGTHS = (131072, 262144, 524288)
EXPECTED_QUEST_EXTENSION = "de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f"


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing(method, result):
    if method == "nfa":
        return result["timing"]
    if method == "quest":
        return result["cells"][0]["request_timing"]
    return result


def mean_ms(summary, name):
    return summary[name]["mean_ms_per_step"]


def validate_common(method, length, result):
    assert result["context_length"] == length
    assert result["batch_size"] == 1
    assert result["task"] == "niah_multikey_2"
    assert result["sample_index"] == 2
    receipt = result["input"]
    assert receipt["file_row_index"] == 0
    assert receipt["provenance"] == "repeated-needle-body-extension"
    return receipt["sha256"]


def validate_clean(method, length, pass_name):
    stem = f"{method}_L{length}_{pass_name}"
    output = ROOT / "clean" / f"{stem}.json"
    receipt_path = ROOT / "clean" / f"{stem}.receipt.json"
    log_path = ROOT / "logs" / f"{stem}.log"
    result = load(output)
    receipt = load(receipt_path)
    input_hash = validate_common(method, length, result)
    assert receipt["abba_pass"] == pass_name
    assert receipt["preflight_compute_apps"] == []
    assert receipt["postflight_compute_apps"] == []
    assert receipt["output_sha256"] == sha256(output)
    assert receipt["log_sha256"] == sha256(log_path)
    assert receipt["input_file_sha256"] == input_hash
    current = timing(method, result)
    assert current["measured_decode_steps"] == 29
    assert current["decode_host_tpot_ms"] == receipt["decode_host_tpot_ms"]
    assert current["decode_host_tpot_ms"] > 0
    if method == "nfa":
        assert result["force_skip_mode"] == "natural"
        assert result["subspace"] == "tail" and result["l2_subspace"] == "same"
        assert result["gate_epsilon"] == 0.05 and result["operator_mode"] == "off"
        skip = result["skip"]["0"]
        assert skip["near_layers"] or skip["far_layers"] or skip["both_layers"]
        assert (result["graph_capture_count"], result["graph_replays"],
                result["graph_eager_fallbacks"]) == (1, 32, 0)
    elif method in ("asynctls", "jssa"):
        assert (result["graph_capture_count"], result["graph_replays"],
                result["graph_eager_fallbacks"]) == (1, 32, 0)
    else:
        counts = result["quest_call_counts"]
        assert counts == {
            "prefill_forward": 36,
            "decode_estimate": 34 * 32,
            "decode_topk": 34 * 32,
            "decode_sparse_attn": 36 * 32,
            "dense_or_torch_fallback": 0,
        }
        assert result["runtime"]["kernel_extension_sha256"] == EXPECTED_QUEST_EXTENSION
        assert result["runtime"]["dense_or_torch_fallback"] is False
    return {
        "method": method,
        "pass": pass_name,
        "decode_host_tpot_ms": current["decode_host_tpot_ms"],
        "input_sha256": input_hash,
        "output": str(output.relative_to(ROOT)),
        "receipt": str(receipt_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "nfa_skip": result.get("skip", {}).get("0"),
    }


def validate_operator(method, length, clean_input_hash):
    stem = f"{method}_L{length}"
    output = ROOT / "operator" / f"{stem}.json"
    report = ROOT / "operator" / f"{stem}.nsys-rep"
    sqlite = ROOT / "operator" / f"{stem}.sqlite"
    result = load(output)
    assert validate_common(method, length, result) == clean_input_hash
    assert result["nsys_profile_finalized"] is True
    assert result["profiled_decode_steps"] == 29
    assert report.stat().st_size > 0 and sqlite.stat().st_size > 0
    summary = result["operator_summary"]
    if method == "nfa":
        assert result["region_kernel_audit"]["passed"] is True
        assert (result["graph_capture_count"], result["graph_replays"],
                result["graph_eager_fallbacks"]) == (1, 32, 0)
        names = ("l1_ms", "l2_ms", "selected_attention_ms", "nfa_operator_total_ms")
    elif method == "asynctls":
        assert result["official_tilelang"] is False
        assert result["real_async_offload"] is False
        names = ("topk_sort_family_ms", "gather_family_ms", "index_family_ms")
    elif method == "jssa":
        assert result["jssa_runtime"]["radix_topk_available"] is False
        assert summary["store_ms"]["calls_total"] == 36 * 29
        assert summary["score_ms"]["calls_total"] == 36 * 29
        assert summary["topk_ms"]["calls_total"] > 0
        names = ("store_ms", "score_ms", "topk_ms", "selected_attention_ms",
                 "official_named_operator_total_ms")
    else:
        assert result["quest_call_counts"]["dense_or_torch_fallback"] == 0
        assert result["runtime"]["kernel_extension_sha256"] == EXPECTED_QUEST_EXTENSION
        expected = {
            "decode_estimate_ms": 34 * 29,
            "decode_topk_ms": 34 * 29,
            "sparse_partition_ms": 36 * 29,
            "sparse_merge_ms": 36 * 29,
            "topk_canonicalize_ms": 0,
        }
        for name, calls in expected.items():
            assert summary[name]["calls_total"] == calls, (name, summary[name])
        names = ("decode_estimate_ms", "decode_topk_ms", "decode_sparse_attn_ms",
                 "sparse_partition_ms", "sparse_merge_ms", "topk_canonicalize_ms")
    nsys_receipt = result.get("nsys_receipt")
    if nsys_receipt is not None:
        assert nsys_receipt["report_sha256"] == sha256(report)
        assert nsys_receipt["sqlite_sha256"] == sha256(sqlite)
    return {
        "method": method,
        "input_sha256": clean_input_hash,
        "mean_ms_per_step": {name: mean_ms(summary, name) for name in names},
        "scope": summary.get("scope"),
        "output": str(output.relative_to(ROOT)),
        "nsys_rep": str(report.relative_to(ROOT)),
        "sqlite": str(sqlite.relative_to(ROOT)),
        "nfa_region_kernel_audit": result.get("region_kernel_audit"),
    }


def main():
    rows = []
    for length in LENGTHS:
        clean = [validate_clean(method, length, pass_name)
                 for pass_name in ("forward", "reverse") for method in METHODS]
        input_hashes = {row["input_sha256"] for row in clean}
        assert len(input_hashes) == 1, input_hashes
        by_method = {}
        for method in METHODS:
            pair = [row for row in clean if row["method"] == method]
            assert [row["pass"] for row in pair] == ["forward", "reverse"]
            if method == "nfa":
                assert pair[0]["nfa_skip"] == pair[1]["nfa_skip"]
            values = [row["decode_host_tpot_ms"] for row in pair]
            by_method[method] = {
                "forward_ms": values[0], "reverse_ms": values[1],
                "abba_mean_ms": sum(values) / 2,
                "nfa_skip": pair[0]["nfa_skip"],
            }
        for method in METHODS[1:]:
            by_method[method]["speedup_over_method"] = (
                by_method[method]["abba_mean_ms"] / by_method["nfa"]["abba_mean_ms"])
        operators = [validate_operator(method, length, input_hashes.pop())
                     for method in METHODS]
        rows.append({
            "context_length": length,
            "clean": clean,
            "clean_abba": by_method,
            "operator": operators,
        })
    summary = {
        "schema": "qwen3-bs1-direct-decode-abba-operator-v1",
        "verification": "pending-independent-verification",
        "production_audit": "pass",
        "model": "Qwen3-8B",
        "hardware": "pc33 H20-3e; one measured process per GPU; clean and profiler processes separated",
        "task": "niah_multikey_2",
        "sample_index": 2,
        "input_file_row_index": 0,
        "input_provenance": "repeated-needle-body-extension",
        "batch_size": 1,
        "token_budget": 1024,
        "clean_protocol": "one prefill outside timing; 3 warmup decode steps; 29 measured steps; host wall ends after final synchronization; forward nfa/asynctls/jssa/quest then exact reverse",
        "operator_protocol": "independent nsys --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop; 29 measured steps; observed CUDA kernel time",
        "valid_natural_skip_lengths": list(LENGTHS),
        "excluded_natural_no_skip_lengths": [32768, 65536],
        "limitations": [
            "Production-side audit is not independent verification; no final scientific ranking is certified by this file.",
            "32K/64K NFA naturally triggered no region skip and remain diagnostics only; no forced-skip replacement was used.",
            "AsyncTLS is an independent PyTorch paper-semantic implementation, not official TileLang, and has no real asynchronous host offload.",
            "AsyncTLS index_family_ms is only the uniquely identifiable Top-K/sort/gather subset, not the full selector or selected attention.",
            "JSSA uses official Triton store/score/indexed-attend but official torch.topk fallback; vLLM radix binding is unavailable.",
            "Quest uses the fixed upstream BS1 official CUDA extension with RAFT Top-K and no canonicalize kernel; no Torch/dense fallback occurred.",
            "Profiler timings must not be subtracted from clean TPOT or interpreted as host overhead.",
        ],
        "rows": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema": "bs1-direct-decode-artifact-manifest-v1",
        "verification": "pending-independent-verification",
        "summary_sha256": sha256(ROOT / "summary.json"),
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PRODUCTION_AUDIT_PASS lengths={LENGTHS} clean={len(rows) * 8} operator={len(rows) * 4} artifacts={len(artifacts)}")


if __name__ == "__main__":
    main()
