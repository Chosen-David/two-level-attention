#!/usr/bin/env python3
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path("/home/wangyuanshuo02/two-level-attention/agent_doc/results/qwen3_official_long_context_20261113")
FIXED_REPO = Path("/tmp/qwen3-official-smoke-19edfebd-20261115")
MAIN_REPO = Path("/home/wangyuanshuo02/two-level-attention")
NEW_HEAD = "19edfebd404cfffadeadc30de7378f9fbe3e0069"
REUSED_HEAD = "41006cf8298167eed4b9bac386dfaa2a6c24bc85"
EMPTY_DIFF = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CHANNEL_SHA = "613a971cbc36d8e312a76af5ff2ee85bcd1bb0959c333ebd8e0605afd1524da4"
QUEST_EXTENSION_SHA = "de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f"
QUEST_DIFF_SHA = "edb9e2d579687781a9b293ed87dc2445664db25118811bc5a8227b5ea6792b8f"
RESOURCE_BLOCKED = {
    ("nfa", 262144, 4): {
        "path": ROOT / "logs" / "nfa_graph_gpu0_256k_b4_oom.log",
        "sha256": "213b8de52ad2aa8208a14d1f96e1b4addd60d21baac2cfa5cd2ec16f77c244e2",
        "reason": "CUDA OOM allocating paged KV V cache during warmup T(1); do not retry",
    },
    ("asynctls", 262144, 4): {
        "path": ROOT / "logs" / "asynctls_graph_gpu1_256k_b4_oom.log",
        "sha256": "a00b956497831cfbffb0133fb9a4fc11670f21939d34699ef342a31891e1c656",
        "reason": "CUDA OOM allocating AsyncTLS V cache; do not retry",
    },
}
INPUT_256K_SHA = "05325d14fa6868ab22b6964d160adaa1968323ca80e396261b97ac75b55c093d"
LENGTHS = (32768, 65536, 131072, 262144)
BATCHES = (1, 2, 4)
ROPE_FACTORS = {32768: None, 65536: 2.0, 131072: 4.0, 262144: 8.0}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), message)


def check_source(record, path, method, reused):
    source = record["source"]
    require(source["head"] == (REUSED_HEAD if reused else NEW_HEAD), f"{path}: HEAD")
    if method == "nfa":
        name = "benchmark/efficiency/benchmark_long_context_two_point.py"
        expected_sha = ("62308ee0d29e63c36ab61e985ede1e719b7e3d7c136a956829d4120453a21f4a"
                        if reused else
                        "30ecca792c9cb1d7e8913a4f895d8235ec889e32ebd46d3d3cb71ba09fd394ae")
    elif method == "asynctls":
        name = "benchmark/efficiency/benchmark_asynctls_two_point.py"
        expected_sha = "0fee70569ad72e4a76c5ff55ad71fd83bfb21b3a0c7371a2d08c6b7f3e26d3eb"
    else:
        name = "benchmark/official_baselines/long_context_official.py"
        expected_sha = ("0102961858e0a9455fcb0ad78d269fba5bc90c957bfe16f9ed9bad462ead63a9"
                        if reused else
                        "9154276939ebff8355136c921dd8d9e771ddd7a4dbf41285e4286c7185fd34ce")
    repo = MAIN_REPO if reused else FIXED_REPO
    require(source["harness"] == str(repo / name), f"{path}: harness path")
    require(source["harness_sha256"] == expected_sha, f"{path}: harness SHA")
    if not reused and method in ("nfa", "asynctls"):
        require(source["source_diff_sha256"] == EMPTY_DIFF, f"{path}: source diff")


def check_common(record, path, method, length, batch, reused):
    require(record["status"] == "complete", f"{path}: incomplete")
    require(record["verification"] == "pending-independent-verification", f"{path}: verification")
    require(record.get("batch_size", 1) == batch, f"{path}: batch")
    require(record["context_length"] == length, f"{path}: context")
    require(record["task"] == "niah_multikey_2" and record["sample_index"] == 2, f"{path}: task/sample")
    require(record["warmup_requests"] == 1 and record["repeats"] == 3, f"{path}: protocol")
    require(record["max_new_tokens"] == 33, f"{path}: max-new")
    require(record["runtime_rope_verification"]["status"] == "verified", f"{path}: RoPE")
    require(record["runtime_rope_verification"]["requested_factor"] == ROPE_FACTORS[length], f"{path}: RoPE factor")
    input_receipt = record["input"]
    require(0.95 * length <= input_receipt["actual_prompt_tokens"] <= length, f"{path}: prompt length")
    if length == 262144:
        require(input_receipt["input_ids_sha256"] == INPUT_256K_SHA, f"{path}: 256K input identity")
    require(len(record["cells"]) == 3, f"{path}: repeats")
    check_source(record, path, method, reused)
    tpots = []
    for repeat, cell in enumerate(record["cells"]):
        require(cell["repeat"] == repeat, f"{path}: repeat index")
        timing = cell["request_timing"]
        generated = timing["generated_token_ids"]
        if batch == 1:
            require(len(generated) == 33 and all(isinstance(token, int) for token in generated), f"{path}: flat B1 tokens")
        else:
            require(len(generated) == batch and all(len(row) == 33 for row in generated), f"{path}: batch-major tokens")
        require(cell["output_ids_sha256"] == ids_sha256(generated), f"{path}: output SHA")
        require(timing["actual_generated_tokens"] == 33 * batch, f"{path}: generated total")
        require(timing["eos_early_stop"] is False, f"{path}: early EOS")
        close(timing["tpot_ms"], timing["decode_after_first_wall_ms"] / 32, f"{path}: TPOT formula")
        close(timing["request_wall_ms"], timing["ttft_ms"] + timing["decode_after_first_wall_ms"], f"{path}: request wall")
        tpots.append(timing["tpot_ms"])
    close(record["summary"]["all_request_median"]["tpot_ms"], statistics.median(tpots), f"{path}: median")
    return tpots


def check_nfa(record, path, length, batch, reused):
    tpots = check_common(record, path, "nfa", length, batch, reused)
    require(record["method"] == "nfa" and record["execution"] == "graph", f"{path}: method/execution")
    require(record["host_readback_in_decode_loop"] is False, f"{path}: host readback")
    runtime = record["runtime"]
    require(runtime["dense_or_torch_decode_fallback"] is False, f"{path}: fallback")
    config = runtime["nfa_selector_config"]
    require(config["level1_topk"] == 128 and config["level2_topk"] == 1024, f"{path}: K1/K2")
    require(config["subspace"] == "tail" and config["subspace_cmp_ratio"] == 4, f"{path}: subspace")
    require(config["page_size"] == 64 and config["enable_layer_skip"] is False, f"{path}: geometry")
    receipts = record["request_receipts"]
    require(len(receipts) == 8, f"{path}: receipt count")
    for index, receipt in enumerate(receipts):
        tokens = 1 if index % 2 == 0 else 33
        steps = 0 if index % 2 == 0 else 32
        require(receipt["request_index"] == index, f"{path}: receipt index")
        require(receipt["request_role"] == ("warmup" if index < 2 else "measured"), f"{path}: receipt role")
        require(receipt["cache_reused"] is (index > 0), f"{path}: cache reuse")
        if "generated_tokens_per_request" in receipt:
            require(receipt["generated_tokens_per_request"] == [tokens] * batch, f"{path}: receipt tokens")
        else:
            require(reused and batch == 1 and receipt["generated_tokens"] == tokens, f"{path}: legacy receipt tokens")
        require(receipt["nfa_decode_steps"] == steps, f"{path}: decode steps")
        expected_capture = int(index == 1)
        require(receipt["graph_capture_count"] == expected_capture, f"{path}: capture")
        require(receipt["graph_replays"] == steps, f"{path}: replay")
        require(receipt["graph_eager_fallbacks"] == 0, f"{path}: graph fallback")
        if steps:
            require(receipt["graph_pool_misses"] == int(index == 1), f"{path}: pool miss")
            require(receipt["graph_pool_hits"] == int(index > 1), f"{path}: pool hit")
    return tpots


def check_asynctls(record, path, length, batch, reused):
    tpots = check_common(record, path, "asynctls", length, batch, reused)
    require(record["method"] == "asynctls-semantic-pytorch" and record["execution"] == "graph", f"{path}: method/execution")
    require(record["host_readback_in_decode_loop"] is False, f"{path}: host readback")
    require(record["channel_file_sha256"] == CHANNEL_SHA, f"{path}: channel SHA")
    runtime = record["runtime"]
    require(runtime["official_tilelang"] is False and runtime["real_async_offload"] is False, f"{path}: implementation labels")
    require(runtime["dense_or_torch_decode_fallback"] is False, f"{path}: fallback")
    require(runtime["token_budget"] == 1024 and runtime["block_size"] == 64 and runtime["top_kb"] == 128, f"{path}: budget")
    require(runtime["channels_per_kv_group"] == 32, f"{path}: channels")
    receipts = record["request_receipts"]
    require(len(receipts) == 8, f"{path}: receipt count")
    for index, receipt in enumerate(receipts):
        tokens = 1 if index % 2 == 0 else 33
        steps = 0 if index % 2 == 0 else 32
        require(receipt["request_index"] == index, f"{path}: receipt index")
        require(receipt["request_role"] == ("warmup" if index < 2 else "measured"), f"{path}: receipt role")
        require(receipt["cache_reused"] is (index > 0), f"{path}: cache reuse")
        require(receipt["generated_tokens_per_request"] == [tokens] * batch, f"{path}: receipt tokens")
        require(receipt["decode_steps"] == steps, f"{path}: decode steps")
        require(receipt["prefill_calls"] == 36 * batch, f"{path}: prefill calls")
        expected_capture = int(index == 1)
        require(receipt["graph_capture_count"] == expected_capture, f"{path}: capture")
        require(receipt["graph_replays"] == steps, f"{path}: replay")
        require(receipt["graph_eager_fallbacks"] == 0, f"{path}: graph fallback")
        require(receipt["python_decode_calls_warmup_and_capture"] == (72 if index == 1 else 0), f"{path}: Python decode calls")
    return tpots


def check_quest(record, path, length, reused):
    tpots = check_common(record, path, "quest", length, 1, reused)
    require(record["method"] == "quest", f"{path}: method")
    runtime = record["runtime"]
    require(runtime["dense_or_torch_fallback"] is False, f"{path}: fallback")
    require(runtime["budget"] == 1024 and runtime["page_size"] == 16 and runtime["page_budget"] == 64, f"{path}: budget")
    require(runtime["quest"]["commit"] == "01c1623bf9395009520874e989e29f683203b357", f"{path}: Quest commit")
    require(runtime["quest"]["uncommitted_diff_sha256"] == QUEST_DIFF_SHA, f"{path}: Quest diff SHA")
    require(runtime["kernel_extension_sha256"] == QUEST_EXTENSION_SHA, f"{path}: extension SHA")
    counts = record["quest_call_counts"]
    require(counts["dense_or_torch_fallback"] == 0, f"{path}: fallback count")
    require(counts["prefill_forward"] == 288, f"{path}: prefill calls")
    require(counts["decode_sparse_attn"] == 4608, f"{path}: sparse calls")
    require(counts["decode_estimate"] == 4352 and counts["decode_topk"] == 4352, f"{path}: estimate/topk calls")
    return tpots


def specs():
    result = []
    for length in LENGTHS:
        for batch in BATCHES:
            if length == 262144 and batch == 1:
                path = ARCHIVE / "nfa_256k_two_point_performance.json"
                result.append(("nfa", length, batch, path, True))
            else:
                path = ROOT / "nfa" / f"nfa_graph_b{batch}_{length // 1024}k.json"
                result.append(("nfa", length, batch, path, False))
            path = ROOT / "asynctls" / f"asynctls_graph_b{batch}_{length // 1024}k.json"
            result.append(("asynctls", length, batch, path, False))
    for length in LENGTHS:
        if length == 262144:
            path = ARCHIVE / "quest_256k_workspace_fix_performance.json"
            result.append(("quest", length, 1, path, True))
        else:
            path = ROOT / "quest" / f"quest_b1_{length // 1024}k.json"
            result.append(("quest", length, 1, path, False))
    return result


def main():
    rows = []
    pending = []
    completed = {}
    for method, length, batch, path, reused in specs():
        if (method, length, batch) in RESOURCE_BLOCKED:
            require(not path.is_file(), f"{path}: resource-blocked cell unexpectedly has output")
            continue
        if not path.is_file():
            pending.append({"method": method, "context_length": length, "batch_size": batch, "status": "missing", "path": str(path)})
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            pending.append({"method": method, "context_length": length, "batch_size": batch, "status": record.get("status"), "path": str(path)})
            continue
        if method == "nfa":
            tpots = check_nfa(record, path, length, batch, reused)
        elif method == "asynctls":
            tpots = check_asynctls(record, path, length, batch, reused)
        else:
            tpots = check_quest(record, path, length, reused)
        row = {
            "method": method,
            "context_length": length,
            "batch_size": batch,
            "status": "complete",
            "source_head": record["source"]["head"],
            "reused_closed_result": reused,
            "input_ids_sha256": record["input"]["input_ids_sha256"],
            "actual_prompt_tokens": record["input"]["actual_prompt_tokens"],
            "repeat_tpot_ms": tpots,
            "median_tpot_ms": statistics.median(tpots),
            "path": str(path),
            "sha256": sha256(path),
        }
        rows.append(row)
        completed[(method, length, batch)] = row
    for length in LENGTHS:
        same_length = [row for row in rows if row["context_length"] == length]
        if same_length:
            require(len({row["input_ids_sha256"] for row in same_length}) == 1, f"{length}: cross-method input SHA")
            require(len({row["actual_prompt_tokens"] for row in same_length}) == 1, f"{length}: cross-method prompt length")
    matrix = []
    for method in ("quest", "jssa", "asynctls", "nfa"):
        for length in LENGTHS:
            for batch in BATCHES:
                key = (method, length, batch)
                if method in ("quest", "jssa") and batch > 1:
                    cell = {"status": "unsupported", "reason": "no official high-performance batched path; fail-closed"}
                elif method == "jssa":
                    cell = {"status": "blocked", "reason": "fixed cu129 wheel remains unavailable; no version change or fallback"}
                elif key in RESOURCE_BLOCKED:
                    blocked = RESOURCE_BLOCKED[key]
                    require(sha256(blocked["path"]) == blocked["sha256"], f"{key}: OOM log SHA")
                    cell = {
                        "status": "resource-blocked",
                        "reason": blocked["reason"],
                        "log_path": str(blocked["path"]),
                        "log_sha256": blocked["sha256"],
                    }
                elif key in completed:
                    row = completed[key]
                    cell = {"status": "complete", "median_tpot_ms": row["median_tpot_ms"], "sha256": row["sha256"]}
                else:
                    cell = {"status": "pending"}
                matrix.append({"method": method, "context_length": length, "batch_size": batch, **cell})
    require(len(matrix) == 48, "matrix size")
    result = {
        "schema": "qwen3-official-ruler-multikey2-performance-audit-v2",
        "verification": "pending-independent-verification",
        "status": "PASS" if len(rows) == 26 else "PARTIAL",
        "scope": "26 measurable cells + 2 resource-blocked OOM cells + 4 blocked JSSA B1 cells + 16 unsupported Quest/JSSA B2/B4 cells = 48",
        "completed_cells": len(rows),
        "resource_blocked_cells": len(RESOURCE_BLOCKED),
        "pending_cells": len(pending),
        "rows": sorted(rows, key=lambda row: (row["method"], row["context_length"], row["batch_size"])),
        "pending": pending,
        "matrix": matrix,
    }
    output = ROOT / "performance_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "completed_cells": len(rows), "pending_cells": len(pending), "matrix_cells": len(matrix), "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
