#!/usr/bin/env python3
"""Qwen3 RULER 五长度 BS1 clean/operator 生产侧闭集审计与汇总。"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
LENGTHS = (32768, 65536, 131072, 262144, 524288)
ROPE_FACTORS = {32768: None, 65536: 2.0, 131072: 4.0, 262144: 8.0, 524288: 16.0}
CURRENT_HEAD = "b3ee770f8865549ef7271bc83f73968a328950d1"
QUEST_COMMIT = "01c1623bf9395009520874e989e29f683203b357"
PENDING = "pending-independent-verification"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), message)


def finite_nonnegative(value, message):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, message)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_row(record):
    source = record["source"]
    require(len(source.get("head", "")) in (40, 64), "source HEAD 缺失")
    require(len(source.get("harness_sha256", "")) == 64, "source harness SHA 缺失")
    return {
        "source_head": source["head"],
        "harness_sha256": source["harness_sha256"],
        "harness_path": source.get("harness"),
    }


def check_input(record, length, require_ids=True):
    receipt = record["input"]
    require(record["task"] == "niah_multikey_2" and record["sample_index"] == 2, "task/sample")
    require(receipt["file_row_index"] == 0, "必须使用扩展文件第 0 行")
    require(receipt["provenance"] == "repeated-needle-body-extension", "输入 provenance")
    require(0.95 * length <= receipt["actual_prompt_tokens"] <= length, "prompt 长度")
    require(len(receipt["sha256"]) == 64, "输入文件 SHA")
    require(sha256(receipt["path"]) == receipt["sha256"], "输入文件实际 SHA 不一致")
    if require_ids:
        require(len(receipt["input_ids_sha256"]) == 64, "输入 IDs SHA")
    return receipt


def check_clean_common(record, path, length):
    require(record["status"] == "complete", f"{path}: 未完成")
    require(record["verification"] == PENDING, f"{path}: 验证状态")
    require(record.get("batch_size", 1) == 1, f"{path}: 仅允许 BS1")
    require(record["context_length"] == length, f"{path}: 长度")
    require(record["warmup_requests"] == 1 and record["repeats"] == 3, f"{path}: warmup/repeats")
    require(record["max_new_tokens"] == 33, f"{path}: max-new")
    rope = record["runtime_rope_verification"]
    require(rope["status"] == "verified", f"{path}: RoPE 未验证")
    require(rope["requested_factor"] == ROPE_FACTORS[length], f"{path}: RoPE factor")
    receipt = check_input(record, length)
    require(len(record["cells"]) == 3, f"{path}: repeat 数")
    tpots = []
    for index, cell in enumerate(record["cells"]):
        require(cell["repeat"] == index, f"{path}: repeat index")
        timing = cell["request_timing"]
        generated = timing["generated_token_ids"]
        require(len(generated) == 33 and all(isinstance(token, int) for token in generated), f"{path}: B1 token 数")
        require(cell["output_ids_sha256"] == ids_sha256(generated), f"{path}: 输出 SHA")
        require(timing["actual_generated_tokens"] == 33, f"{path}: 生成 token 总数")
        require(timing.get("actual_generated_tokens_per_request", [timing["actual_generated_tokens"]]) == [33], f"{path}: 逐请求 token 数")
        require(timing["eos_early_stop"] is False, f"{path}: EOS 提前停止")
        require(timing["decode_after_first_wall_ms"] > 0, f"{path}: clean 两点差分非正")
        close(timing["tpot_ms"], timing["decode_after_first_wall_ms"] / 32, f"{path}: TPOT 公式")
        close(timing["request_wall_ms"], timing["ttft_ms"] + timing["decode_after_first_wall_ms"], f"{path}: request wall")
        tpots.append(timing["tpot_ms"])
    close(record["summary"]["all_request_median"]["tpot_ms"], statistics.median(tpots), f"{path}: 中位数")
    return tpots, receipt


def check_graph_receipts(record, path, method):
    receipts = record["request_receipts"]
    require(len(receipts) == 8, f"{path}: 请求回执数")
    for index, receipt in enumerate(receipts):
        tokens = 1 if index % 2 == 0 else 33
        steps = 0 if index % 2 == 0 else 32
        require(receipt["request_index"] == index, f"{path}: 请求序号")
        require(receipt["request_role"] == ("warmup" if index < 2 else "measured"), f"{path}: 请求角色")
        require(receipt["cache_reused"] is (index > 0), f"{path}: cache reuse")
        require(receipt.get("generated_tokens_per_request", [receipt["generated_tokens"]]) == [tokens], f"{path}: 回执 token 数")
        require(receipt["graph_capture_count"] == int(index == 1), f"{path}: capture")
        require(receipt["graph_replays"] == steps, f"{path}: replay")
        require(receipt["graph_eager_fallbacks"] == 0, f"{path}: eager fallback")
        if method == "nfa":
            require(receipt["nfa_decode_steps"] == steps, f"{path}: NFA decode steps")
            if steps:
                require(receipt["graph_pool_misses"] == int(index == 1), f"{path}: pool miss")
                require(receipt["graph_pool_hits"] == int(index > 1), f"{path}: pool hit")
        else:
            require(receipt["decode_steps"] == steps, f"{path}: AsyncTLS decode steps")
            require(receipt["prefill_calls"] == 36, f"{path}: prefill calls")
            require(receipt["python_decode_calls_warmup_and_capture"] == (72 if index == 1 else 0), f"{path}: capture decode calls")


def clean_spec(method, length):
    suffix = f"{length // 1024}k"
    if method == "quest":
        name = "quest_b1_32k_current.json" if length == 32768 else f"quest_b1_{suffix}.json"
        origin = "current-rerun" if length == 32768 else "reused-closed-result"
    elif method == "nfa":
        name = f"nfa_graph_b1_{suffix}.json"
        origin = "reused-closed-result"
    else:
        name = f"asynctls_graph_b1_{suffix}.json"
        origin = "current-measurement" if length == 524288 else "reused-closed-result"
    return ROOT / method / name, origin


def audit_clean(method, length):
    path, origin = clean_spec(method, length)
    if not path.is_file():
        return None, {"method": method, "context_length": length, "status": "missing", "path": str(path)}
    record = load(path)
    if record.get("status") != "complete":
        return None, {"method": method, "context_length": length, "status": record.get("status"), "path": str(path)}
    tpots, receipt = check_clean_common(record, path, length)
    runtime = record["runtime"]
    if method == "quest":
        require(record["method"] == "quest", f"{path}: method")
        require(runtime["implementation"].startswith("official Quest CUDA"), f"{path}: 非官方实现")
        require(runtime["dense_or_torch_fallback"] is False, f"{path}: fallback")
        require((runtime["budget"], runtime["page_size"], runtime["page_budget"]) == (1024, 16, 64), f"{path}: budget")
        require(runtime["quest"]["commit"] == QUEST_COMMIT, f"{path}: Quest commit")
        counts = record["quest_call_counts"]
        require(counts == {
            "prefill_forward": 288, "decode_estimate": 4352, "decode_topk": 4352,
            "decode_sparse_attn": 4608, "dense_or_torch_fallback": 0,
        }, f"{path}: 官方调用计数")
    elif method == "nfa":
        require(record["method"] == "nfa" and record["execution"] == "graph", f"{path}: method/execution")
        require(record["host_readback_in_decode_loop"] is False, f"{path}: decode host readback")
        require(runtime["dense_or_torch_decode_fallback"] is False, f"{path}: fallback")
        config = runtime["nfa_selector_config"]
        require(config["level1_topk"] == 128 and config["level2_topk"] == 1024, f"{path}: K1/K2")
        require(config["subspace"] == "tail" and config["subspace_cmp_ratio"] == 4, f"{path}: 子空间")
        require(config["page_size"] == 64 and config["enable_layer_skip"] is False, f"{path}: page/skip")
        check_graph_receipts(record, path, method)
    else:
        require(record["method"] == "asynctls-semantic-pytorch" and record["execution"] == "graph", f"{path}: method/execution")
        require(record["host_readback_in_decode_loop"] is False, f"{path}: decode host readback")
        require(runtime["official_tilelang"] is False and runtime["real_async_offload"] is False, f"{path}: 实现标签")
        require(runtime["dense_or_torch_decode_fallback"] is False, f"{path}: fallback")
        require((runtime["token_budget"], runtime["block_size"], runtime["top_kb"]) == (1024, 64, 128), f"{path}: budget")
        require(runtime["channels_per_kv_group"] == 32, f"{path}: channel dims")
        check_graph_receipts(record, path, method)
    row = {
        "method": method, "context_length": length, "batch_size": 1,
        "status": "complete", "measurement_origin": origin,
        "new_gpu_measurement": origin != "reused-closed-result",
        **source_row(record),
        "input_ids_sha256": receipt["input_ids_sha256"],
        "input_file_sha256": receipt["sha256"],
        "actual_prompt_tokens": receipt["actual_prompt_tokens"],
        "repeat_tpot_ms": tpots, "median_tpot_ms": statistics.median(tpots),
        "path": str(path), "sha256": sha256(path),
    }
    if method == "asynctls":
        row["channel_file_sha256"] = record["channel_file_sha256"]
    if method == "quest":
        row["kernel_extension_sha256"] = runtime["kernel_extension_sha256"]
        row["quest_diff_sha256"] = runtime["quest"]["uncommitted_diff_sha256"]
    return row, None


def evidence_pair(base):
    base = Path(base)
    nsys = base.with_suffix(".nsys-rep")
    sqlite = base.with_suffix(".sqlite")
    require(nsys.is_file() and sqlite.is_file(), f"缺 nsys 证据: {base}")
    return {
        "nsys_path": str(nsys), "nsys_sha256": sha256(nsys),
        "sqlite_path": str(sqlite), "sqlite_sha256": sha256(sqlite),
    }


def check_profile_common(record, length, clean_row):
    require(record.get("batch_size", 1) == 1 and record["context_length"] == length, "profile BS/长度")
    require(record["measurement_mode"] == "operator-profile" or record["measurement_mode"] == "nsys-cuda-full-process-operator-profile", "profile 模式")
    require(record.get("max_new_tokens", record.get("decode_steps", 0) + 1) == 33, "profile 33-token 请求")
    receipt = check_input(record, length, require_ids=False)
    require(receipt["sha256"] == clean_row["input_file_sha256"], "profile/clean 输入文件不一致")
    require(receipt["actual_prompt_tokens"] == clean_row["actual_prompt_tokens"], "profile/clean token 数不一致")
    return receipt


def nfa_profile_spec(length):
    suffix = f"{length // 1024}k"
    if length <= 131072:
        return ROOT / "profiles" / f"nfa_operator_b1_{suffix}_current.json", 29, True
    return ROOT / "profiles" / f"nfa_operator_b1_{suffix}_reused.json", 5, False


def audit_nfa_profile(length, clean_row):
    path, repeat, current = nfa_profile_spec(length)
    if not path.is_file():
        return None, {"method": "nfa", "context_length": length, "status": "missing", "path": str(path)}
    record = load(path)
    if not record.get("nsys_profile_finalized"):
        return None, {"method": "nfa", "context_length": length, "status": "not-finalized", "path": str(path)}
    check_profile_common(record, length, clean_row)
    require(record["execution"] == "graph" and record["subspace"] == "tail" and record["l2_subspace"] == "same", f"{path}: NFA 配置")
    require(record["score_dimensions"] == {"head_dim": 128, "l1_score_dims": 32, "l2_score_dims": 32}, f"{path}: 打分子空间")
    require(record["gate_epsilon"] == 0.0 and record["force_skip_mode"] == "natural" and record["operator_mode"] == "off", f"{path}: gate/operator")
    require(record["profile_repeat"] == repeat and record["profiled_decode_steps"] == repeat, f"{path}: profile steps")
    require(record["warmup_steps"] == 3 and record["decode_steps"] == 32, f"{path}: warmup/decode 合同")
    require(record["graph_eager_fallbacks"] == 0, f"{path}: Graph fallback")
    require(record["graph_replays"] == record["warmup_steps"] + repeat, f"{path}: Graph replay 数")
    require(record["timing"] is None, f"{path}: profiler 不得含 clean TPOT")
    expected_tokens = 1 + record["warmup_steps"] + repeat
    require(record["generated_token_ids"] == [record["generated_token_ids"][0]] and len(record["generated_token_ids"][0]) == expected_tokens, f"{path}: 总生成 token 数")
    for request in record["skip"].values():
        require(request["n_layers"] == 36, f"{path}: 层数")
        require(not request["near_layers"] and not request["far_layers"] and not request["both_layers"], f"{path}: 实际触发了 skip")
    for decisions in record["prefill_decisions"].values():
        require(len(decisions) == 36, f"{path}: prefill 决策层数")
        require(all(not flag["skip_near"] and not flag["skip_far"] for flag in decisions.values()), f"{path}: prefill skip")
    require(record["region_kernel_audit"]["passed"] is True, f"{path}: region kernel audit")
    observed = record["region_kernel_audit"]["observed_calls_per_step"]
    require(observed == {"generic_l1": 36.0, "near_only_l1": 0.0, "far_only_l1": 0.0, "active_page_stats": 36.0, "active_candidate": 36.0, "active_fine_score": 36.0}, f"{path}: no-skip kernel 入口")
    summary = record["operator_summary"]
    for key in ("pure_l1_score_ms", "pure_l2_score_ms", "l1_ms", "l2_ms", "selected_attention_ms"):
        finite_nonnegative(summary[key]["mean_ms_per_step"], f"{path}: {key}")
    close(summary["l1_ms"]["total_ms"], sum(summary[key]["total_ms"] for key in ("l1_score_ms", "l1_topk_ms", "candidate_ms")), f"{path}: 宽 L1 求和")
    close(summary["l2_ms"]["total_ms"], sum(summary[key]["total_ms"] for key in ("l2_score_ms", "l2_topk_ms", "selection_union_ms")), f"{path}: 宽 L2 求和")
    if current:
        evidence = evidence_pair(path.with_suffix(""))
        provenance = "current-measurement"
    else:
        reuse = record["reuse_provenance"]
        require(reuse["original_operator_fields_unchanged"] is True, f"{path}: 旧字段重算一致性")
        require(reuse["new_gpu_measurement"] is False and reuse["current_binary_measurement"] is False, f"{path}: 复用标签")
        require(sha256(reuse["original_json_path"]) == reuse["original_json_sha256"], f"{path}: 原 JSON SHA")
        require(sha256(reuse["original_sqlite_path"]) == reuse["original_sqlite_sha256"], f"{path}: 原 SQLite SHA")
        require(sha256(reuse["original_nsys_path"]) == reuse["original_nsys_sha256"], f"{path}: 原 nsys SHA")
        evidence = {
            "nsys_path": reuse["original_nsys_path"], "nsys_sha256": reuse["original_nsys_sha256"],
            "sqlite_path": reuse["original_sqlite_path"], "sqlite_sha256": reuse["original_sqlite_sha256"],
        }
        provenance = "reused-original-5-step-profile-refinalized-copy"
    row = {
        "method": "nfa", "context_length": length, "batch_size": 1,
        "measurement_origin": provenance, "profiled_decode_steps": repeat,
        **source_row(record),
        "pure_l1_score_ms_per_step": summary["pure_l1_score_ms"]["mean_ms_per_step"],
        "pure_l2_score_ms_per_step": summary["pure_l2_score_ms"]["mean_ms_per_step"],
        "wide_l1_ms_per_step": summary["l1_ms"]["mean_ms_per_step"],
        "wide_l2_ms_per_step": summary["l2_ms"]["mean_ms_per_step"],
        "selected_attention_ms_per_step": summary["selected_attention_ms"]["mean_ms_per_step"],
        "skip_near_layers": 0, "skip_far_layers": 0,
        "path": str(path), "sha256": sha256(path), **evidence,
    }
    return row, None


def quest_profile_spec(length):
    suffix = f"{length // 1024}k"
    if length <= 131072:
        return ROOT / "profiles" / f"quest_official_b1_{suffix}.json", False
    if length == 524288:
        return ROOT / "profiles" / "quest_official_b1_512k_operator_alloc.json", True
    return ROOT / "profiles" / f"quest_official_b1_{suffix}_operator.json", True


def audit_quest_profile(length, clean_row):
    path, current = quest_profile_spec(length)
    if not path.is_file():
        return None, {"method": "quest", "context_length": length, "status": "missing", "path": str(path)}
    record = load(path)
    if not record.get("nsys_profile_finalized"):
        return None, {"method": "quest", "context_length": length, "status": "not-finalized", "path": str(path)}
    receipt = check_profile_common(record, length, clean_row)
    require(record["status"] == "complete" and record["verification"] == PENDING, f"{path}: 状态")
    require(record["warmup_requests"] == 0 and record["repeats"] == 1, f"{path}: operator warmup/repeat")
    require(record["profiler_timing_quarantined"] is True and record["profiled_decode_steps"] == 32, f"{path}: profiler 隔离")
    require(receipt["input_ids_sha256"] == clean_row["input_ids_sha256"], f"{path}: 输入 IDs")
    runtime = record["runtime"]
    require(runtime["implementation"].startswith("official Quest CUDA") and runtime["dense_or_torch_fallback"] is False, f"{path}: 官方 runtime")
    require((runtime["budget"], runtime["page_size"], runtime["page_budget"]) == (1024, 16, 64), f"{path}: budget")
    require(runtime["quest"]["commit"] == QUEST_COMMIT, f"{path}: Quest commit")
    require(record["quest_call_counts"] == {
        "prefill_forward": 72, "decode_estimate": 1088, "decode_topk": 1088,
        "decode_sparse_attn": 1152, "dense_or_torch_fallback": 0,
    }, f"{path}: 调用计数")
    summary = record["operator_summary"]
    for key in ("decode_estimate_ms", "decode_topk_ms", "decode_sparse_attn_ms", "topk_raft_ms", "topk_canonicalize_ms", "sparse_partition_ms", "sparse_merge_ms"):
        finite_nonnegative(summary[key]["mean_ms_per_step"], f"{path}: {key}")
    close(summary["decode_topk_ms"]["total_ms"], summary["topk_raft_ms"]["total_ms"] + summary["topk_canonicalize_ms"]["total_ms"], f"{path}: Top-K 求和")
    close(summary["decode_sparse_attn_ms"]["total_ms"], summary["sparse_partition_ms"]["total_ms"] + summary["sparse_merge_ms"]["total_ms"], f"{path}: attention 求和")
    nsys = record["nsys_receipt"]
    require(sha256(nsys["report_path"]) == nsys["report_sha256"], f"{path}: nsys SHA")
    require(sha256(nsys["sqlite_path"]) == nsys["sqlite_sha256"], f"{path}: SQLite SHA")
    row = {
        "method": "quest", "context_length": length, "batch_size": 1,
        "measurement_origin": "current-measurement" if current else "reused-closed-profile",
        "profiled_decode_steps": 32, **source_row(record),
        "estimate_ms_per_step": summary["decode_estimate_ms"]["mean_ms_per_step"],
        "topk_ms_per_step": summary["decode_topk_ms"]["mean_ms_per_step"],
        "topk_raft_ms_per_step": summary["topk_raft_ms"]["mean_ms_per_step"],
        "topk_canonicalize_ms_per_step": summary["topk_canonicalize_ms"]["mean_ms_per_step"],
        "sparse_attention_ms_per_step": summary["decode_sparse_attn_ms"]["mean_ms_per_step"],
        "kernel_extension_sha256": runtime["kernel_extension_sha256"],
        "path": str(path), "sha256": sha256(path),
        "nsys_path": nsys["report_path"], "nsys_sha256": nsys["report_sha256"],
        "sqlite_path": nsys["sqlite_path"], "sqlite_sha256": nsys["sqlite_sha256"],
    }
    return row, None


def audit_asynctls_profile(length, clean_row):
    path = ROOT / "profiles" / f"asynctls_operator_b1_{length // 1024}k_current.json"
    if not path.is_file():
        return None, {"method": "asynctls", "context_length": length, "status": "missing", "path": str(path)}
    record = load(path)
    if not record.get("nsys_profile_finalized"):
        return None, {"method": "asynctls", "context_length": length, "status": "not-finalized", "path": str(path)}
    check_profile_common(record, length, clean_row)
    require(record["method"] == "asynctls-semantic-pytorch" and record["execution"] == "graph", f"{path}: method/execution")
    require(record["warmup_steps"] == 1 and record["profiled_decode_steps"] == 31, f"{path}: steady steps")
    require(record["graph_capture_count"] == 1 and record["graph_replays"] == 32 and record["graph_eager_fallbacks"] == 0, f"{path}: Graph")
    require(record["official_tilelang"] is False and record["real_async_offload"] is False, f"{path}: 实现标签")
    runtime = record["asynctls_runtime"]
    require((runtime["token_budget"], runtime["block_size"], runtime["top_kb"]) == (1024, 64, 128), f"{path}: budget")
    require(runtime["channels_per_kv_group"] == 32, f"{path}: channels")
    require(record["channel_file_sha256"] == clean_row["channel_file_sha256"], f"{path}: channel 身份")
    summary = record["operator_summary"]
    require("not official TileLang" in summary["implementation"], f"{path}: profile 实现说明")
    for key in ("topk_sort_family_ms", "gather_family_ms", "index_family_ms"):
        finite_nonnegative(summary[key]["mean_ms_per_step"], f"{path}: {key}")
        require(summary[key]["calls_total"] > 0, f"{path}: {key} 空调用")
    close(summary["index_family_ms"]["total_ms"], summary["topk_sort_family_ms"]["total_ms"] + summary["gather_family_ms"]["total_ms"], f"{path}: index family 求和")
    evidence = evidence_pair(path.with_suffix(""))
    row = {
        "method": "asynctls", "context_length": length, "batch_size": 1,
        "measurement_origin": "current-measurement", "profiled_decode_steps": 31,
        **source_row(record),
        "topk_sort_family_ms_per_step": summary["topk_sort_family_ms"]["mean_ms_per_step"],
        "gather_family_ms_per_step": summary["gather_family_ms"]["mean_ms_per_step"],
        "index_family_ms_per_step": summary["index_family_ms"]["mean_ms_per_step"],
        "attribution_scope": summary["scope"],
        "path": str(path), "sha256": sha256(path), **evidence,
    }
    return row, None


def main():
    clean_rows, operator_rows, pending = [], [], []
    for method in ("quest", "nfa", "asynctls"):
        for length in LENGTHS:
            row, missing = audit_clean(method, length)
            if row:
                clean_rows.append(row)
            if missing:
                pending.append({"kind": "clean-e2e", **missing})
    clean_by_key = {(row["method"], row["context_length"]): row for row in clean_rows}
    for length in LENGTHS:
        same_length = [row for row in clean_rows if row["context_length"] == length]
        if same_length:
            require(len({row["input_ids_sha256"] for row in same_length}) == 1, f"{length}: clean 跨方法输入 IDs")
            require(len({row["input_file_sha256"] for row in same_length}) == 1, f"{length}: clean 跨方法输入文件")
    for method, auditor in (("nfa", audit_nfa_profile), ("quest", audit_quest_profile), ("asynctls", audit_asynctls_profile)):
        for length in LENGTHS:
            clean = clean_by_key.get((method, length))
            if clean is None:
                continue
            row, missing = auditor(length, clean)
            if row:
                operator_rows.append(row)
            if missing:
                pending.append({"kind": "operator-profile", **missing})
    clean_matrix, speedups = [], []
    for length in LENGTHS:
        cells = {}
        for method in ("nfa", "quest", "asynctls"):
            row = clean_by_key.get((method, length))
            cells[method] = {"status": "complete", "median_tpot_ms": row["median_tpot_ms"]} if row else {"status": "pending"}
        cells["jssa"] = {"status": "blocked", "reason": "官方 radix Top-K binding 缺失；禁止 Torch fallback 或替换 wheel"}
        clean_matrix.append({"context_length": length, "batch_size": 1, "cells": cells})
        nfa = clean_by_key.get(("nfa", length))
        if nfa:
            comparison = {"context_length": length, "nfa_median_tpot_ms": nfa["median_tpot_ms"]}
            for baseline in ("quest", "asynctls"):
                other = clean_by_key.get((baseline, length))
                if other:
                    comparison[f"nfa_speedup_vs_{baseline}"] = other["median_tpot_ms"] / nfa["median_tpot_ms"]
                    comparison[f"nfa_faster_than_{baseline}"] = nfa["median_tpot_ms"] < other["median_tpot_ms"]
            speedups.append(comparison)
    jssa_path = ROOT / "jssa_binding_blocked.json"
    jssa = load(jssa_path)
    require(jssa["status"] == "blocked" and jssa["radix_topk_available"] is False, "JSSA blocked 回执")
    require(jssa["verification"] == PENDING, "JSSA 验证状态")
    complete = len(clean_rows) == 15 and len(operator_rows) == 15 and not pending
    negative = []
    for row in speedups:
        for baseline in ("quest", "asynctls"):
            if row.get(f"nfa_faster_than_{baseline}") is False:
                negative.append({
                    "context_length": row["context_length"], "baseline": baseline,
                    "nfa_speedup": row[f"nfa_speedup_vs_{baseline}"],
                    "handling": "保留负结果；未删除 repeat、修改计时或宣称 NFA 全格领先",
                })
    summary = {
        "schema": "qwen3-ruler-bs1-vendor-performance-v1",
        "verification": PENDING,
        "status": "complete" if complete else "in_progress",
        "producer_audit": "PASS" if complete else "PARTIAL",
        "scope": "Qwen3-8B RULER niah_multikey_2 sample-index2/扩展文件第0行，32K/64K/128K/256K/512K，BS1，K2/token budget=1024；clean 与 nsys 独立进程。",
        "clean_protocol": "1 warmup + 3 repeats；固定33 token、忽略EOS；完整request host wall TPOT=(T(33)-T(1))/32；不按batch归一化。",
        "clean_completed_cells": len(clean_rows),
        "operator_completed_cells": len(operator_rows),
        "jssa_blocked_cells": {"clean_e2e": 5, "operator_profile": 5},
        "pending": pending,
        "clean_matrix": clean_matrix,
        "nfa_relative_performance": speedups,
        "negative_results": negative,
        "nfa_skip_audit": "clean enable_layer_skip=false；五份 operator profile 的36层 near/far skip 均false，region kernel audit均generic/no-skip，未把收益归因于动态skip。",
        "operator_interpretation": "NFA 同时报告纯L1/L2打分与宽阶段；Quest 报告estimate/RAFT/canonicalize/sparse-attention；AsyncTLS仅报告可唯一归属的Top-K/sort/gather family，不伪造K1/K2/selected-attention总时延。",
        "timing_caveat": "所有原始repeat原样保留。512K两点法对独立prefill墙钟敏感；中位数不代表每次repeat稳定，profiler TPOT永不进入clean排名。",
        "implementation_caveat": "Quest为官方CUDA；NFA为本项目生产Graph；AsyncTLS为独立PyTorch语义复现，非官方TileLang且无真实异步offload；JSSA官方binding blocked。",
        "clean_rows": clean_rows,
        "operator_rows": operator_rows,
    }
    atomic_json(ROOT / "summary.json", summary)
    audit = {
        "schema": "qwen3-ruler-bs1-vendor-performance-audit-v1",
        "verification": PENDING,
        "status": "PASS" if complete else "PARTIAL",
        "checks": ["clean协议/输出SHA/输入身份/RoPE/调用计数/Graph fallback", "operator输入/步数/实现标签/分类求和/nsys哈希", "NFA真实no-skip与负排序"],
        "clean_checked": len(clean_rows), "operator_checked": len(operator_rows),
        "pending": pending, "summary_path": str(ROOT / "summary.json"),
        "summary_sha256": sha256(ROOT / "summary.json"),
    }
    atomic_json(ROOT / "audit.json", audit)
    artifacts = []
    for path in [Path(__file__), ROOT / "refinalize_reused_nfa.py", jssa_path, ROOT / "summary.json", ROOT / "audit.json"]:
        artifacts.append({"path": str(path), "sha256": sha256(path), "kind": "audit-or-summary"})
    for row in clean_rows + operator_rows:
        artifacts.append({"path": row["path"], "sha256": row["sha256"], "kind": "json-receipt"})
    binaries = []
    for row in operator_rows:
        binaries.append({
            "method": row["method"], "context_length": row["context_length"],
            "nsys_path": row["nsys_path"], "nsys_sha256": row["nsys_sha256"],
            "sqlite_path": row["sqlite_path"], "sqlite_sha256": row["sqlite_sha256"],
            "storage": "large-binary-reference; not required for clean TPOT consumption",
        })
    manifest = {
        "schema": "qwen3-ruler-bs1-vendor-performance-manifest-v1",
        "verification": PENDING, "status": "complete" if complete else "in_progress",
        "run_id": ROOT.name, "code_heads": sorted({row["source_head"] for row in clean_rows + operator_rows}),
        "current_source_head": CURRENT_HEAD,
        "independent_validation": "pending different-owner verify_experiment_result; this manifest is producer-side integrity evidence only",
        "artifacts": artifacts, "binary_evidence": binaries,
    }
    atomic_json(ROOT / "manifest.json", manifest)
    print(json.dumps({
        "status": audit["status"], "clean": len(clean_rows), "operator": len(operator_rows),
        "pending": pending, "negative_results": negative,
        "summary_sha256": sha256(ROOT / "summary.json"),
        "audit_sha256": sha256(ROOT / "audit.json"),
        "manifest_sha256": sha256(ROOT / "manifest.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
