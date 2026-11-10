"""RULER 三长度 dynamic-gate quick scan：每任务 sample0，dense prefill 后首个决策。"""
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import torch
import transformers
from transformers import AutoTokenizer

from benchmark.efficiency import benchmark_nfa_end_to_end as bench
from sparse_attn.engine.frozen_runner import (
    build_frozen_cache, prefill_frozen_request)
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
from sparse_attn.policies.frozen_gate import (
    GateConfig, decide_regions)
from sparse_attn.policies.region_geometry import RegionGeometry

MODEL = os.environ.get("MODEL", "/home/wangyuanshuo02/Qwen3-8B")
DATA = os.environ["DATA"]
CONTEXT_LENGTH = int(os.environ["CONTEXT_LENGTH"])
OUTPUT = Path(os.environ["OUTPUT"])
TASKS = list(filter(None, os.environ["TASKS"].split(",")))
SAMPLE_INDICES = sorted(set(
    int(value) for value in
    filter(None, os.environ.get("SAMPLE_INDICES", "0").split(","))))
REPO = os.environ["NFA_REPO"]
EXPECTED_HEAD = os.environ["EXPECTED_REPO_HEAD"]


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args, binary=False):
    result = subprocess.check_output(
        ["git", "-C", REPO, *args], text=not binary)
    return result if binary else result.strip()


def _atomic_write(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUTPUT)


def _source_identity():
    head = _git("rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"source HEAD {head} != {EXPECTED_HEAD}")
    diff = _git("diff", "--no-ext-diff", "--binary", "HEAD", binary=True)
    staged = _git("diff", "--no-ext-diff", "--binary", "--cached", binary=True)
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if untracked:
        raise RuntimeError(f"untracked source files are not allowed: {untracked}")
    return {
        "repo": REPO,
        "repo_head": head,
        "worktree_diff_sha256": _sha256_bytes(diff),
        "index_diff_sha256": _sha256_bytes(staged),
        "worktree_diff_bytes": len(diff),
        "index_diff_bytes": len(staged),
        "untracked_files": [],
    }


def _load_existing():
    if not OUTPUT.exists():
        return []
    with OUTPUT.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["source"]["repo_head"] != EXPECTED_HEAD:
        raise RuntimeError("refusing to resume output from another source HEAD")
    if payload["context_length"] != CONTEXT_LENGTH:
        raise RuntimeError("refusing to resume output from another context length")
    if payload.get("freeze_mode") != "decision-only-no-reclaim":
        raise RuntimeError("refusing to resume output from another freeze mode")
    return payload["results"]


def _install_decision_only_freeze(cache):
    """保留真实 prefill/profile/gate，跳过与扫描无关的物理页回收。

    固定提交的 freeze 页记账会在部分 far-skip 请求上抛错。该错误发生在
    decisions 计算之后；本扫描不 decode，因此只把同一组决策写入 cache state，
    用于 receipt/state 一致性核验，不宣称物理 freeze 或显存回收成功。
    """
    def _freeze(self, request_id, skip_far_list, skip_near_list,
                prefill_topk_by_layer=None):
        state = self.states[request_id]
        if len(skip_far_list) != self.n_layers or len(skip_near_list) != self.n_layers:
            raise ValueError("decision-only freeze expects per-layer decisions")
        if state.logical_length == 0:
            raise ValueError("prefill before decision-only freeze")
        state.skip_far_by_layer = [bool(value) for value in skip_far_list]
        state.skip_near_by_layer = [bool(value) for value in skip_near_list]
    cache.freeze = MethodType(_freeze, cache)


def main():
    if not TASKS or len(TASKS) != len(set(TASKS)):
        raise ValueError("TASKS must be a nonempty duplicate-free list")
    if not SAMPLE_INDICES or SAMPLE_INDICES != [0]:
        raise ValueError("this frozen quick-scan protocol requires SAMPLE_INDICES=0")
    torch.manual_seed(20261102)
    source = _source_identity()
    available = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in sorted(glob.glob(os.path.join(DATA, "*.jsonl")))
    }
    missing = sorted(set(TASKS) - set(available))
    if missing:
        raise RuntimeError(f"missing RULER tasks: {missing}")
    expected = {(task, sample_index) for task in TASKS
                for sample_index in SAMPLE_INDICES}
    results = _load_existing()
    by_key = {(row["task"], row["sample_index"]): row for row in results}
    if set(by_key) - expected:
        raise RuntimeError("existing output contains cells outside this shard")
    if len(by_key) != len(results):
        raise RuntimeError("existing output contains duplicate cells")

    base = {
        "schema": "ruler-dynamic-gate-quick-scan-v2",
        "scope": "one RULER sample (index 0) per task; first gate decision after dense prefill; decision-only freeze; no reclaim, decode or generation",
        "freeze_mode": "decision-only-no-reclaim",
        "source": source,
        "model_path": MODEL,
        "model_config_sha256": _sha256_file(
            os.path.join(MODEL, "config.json")),
        "data_path": DATA,
        "context_length": CONTEXT_LENGTH,
        "sample_indices": SAMPLE_INDICES,
        "assigned_tasks": TASKS,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }

    def payload(complete=False):
        ordered = sorted(by_key.values(), key=lambda row: (
            row["task"], row["sample_index"]))
        return {**base, "complete": complete and set(by_key) == expected,
                "expected_cells": len(expected), "realized_cells": len(by_key),
                "results": ordered}

    if set(by_key) == expected:
        _atomic_write(payload(complete=True))
        print(json.dumps({"status": "already-complete", "output": str(OUTPUT),
                          "cells": len(by_key)}), flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = bench._load_model(SimpleNamespace(
        model_path=MODEL, context_length=CONTEXT_LENGTH, method="nfa"), 128)
    patch_args = nfa_args()
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens,
        near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size,
        page=patch_args.tia_block_size,
        max_new_tokens=128)
    config = GateConfig()
    selector = {
        "nfa_far_method": patch_args.nfa_far_method,
        "nfa_near_method": patch_args.nfa_near_method,
        "nfa_near_pages": patch_args.nfa_near_pages,
        "nfa_near_tokens": patch_args.nfa_near_tokens,
        "nfa_near_len": patch_args.nfa_near_len,
        "nfa_subspace": patch_args.nfa_subspace,
        "tia_level2_cmp_ratio": patch_args.tia_level2_cmp_ratio,
        "tia_block_size": patch_args.tia_block_size,
        "tia_level1_topk": patch_args.tia_level1_topk,
        "tia_level2_topk": patch_args.tia_level2_topk,
        "nfa_enable_layer_skip": patch_args.nfa_enable_layer_skip,
        "nfa_far_skip_backend": patch_args.nfa_far_skip_backend,
        "nfa_far_skip_cache": patch_args.nfa_far_skip_cache,
    }
    base["selector"] = selector
    base["geometry"] = dict(geometry.__dict__)
    base["gate"] = {
        "epsilon": config.epsilon, "rows": config.rows,
        "statistic": config.statistic, "mode": config.mode,
    }
    expected_layers = int(model.config.num_hidden_layers)

    for task in TASKS:
        path = available[task]
        input_file_sha256 = _sha256_file(path)
        with open(path, encoding="utf-8") as handle:
            for sample_index, line in enumerate(handle):
                if sample_index > SAMPLE_INDICES[-1]:
                    break
                if sample_index not in SAMPLE_INDICES:
                    continue
                key = (task, sample_index)
                if key in by_key:
                    continue
                row = json.loads(line)
                ids = tokenizer(
                    row["input"], return_tensors="pt").input_ids.cuda()
                cache = build_frozen_cache(
                    model, geometry, ids.shape[-1], max_requests=1,
                    max_visible_len=8192)
                _install_decision_only_freeze(cache)
                with torch.inference_mode():
                    output, receipt = prefill_frozen_request(
                        model, ids, config, geometry, cache=cache,
                        request_id=0)
                profiles = receipt["profiles"]
                decisions = receipt["decisions"]
                state = cache.states[receipt["request_id"]]
                if len(profiles) != expected_layers or len(decisions) != expected_layers:
                    raise RuntimeError(
                        f"{task}: layers profiles={len(profiles)} "
                        f"decisions={len(decisions)} expected={expected_layers}")
                far_layers = sorted(
                    int(layer) for layer, decision in decisions.items()
                    if decision["skip_far"])
                near_layers = sorted(
                    int(layer) for layer, decision in decisions.items()
                    if decision["skip_near"])
                if far_layers != [
                        layer for layer, skip in enumerate(state.skip_far_by_layer)
                        if skip]:
                    raise RuntimeError(f"{task}: far receipt/cache mismatch")
                if near_layers != [
                        layer for layer, skip in enumerate(state.skip_near_by_layer)
                        if skip]:
                    raise RuntimeError(f"{task}: near receipt/cache mismatch")
                normalized_profiles = {}
                normalized_decisions = {}
                for layer in range(expected_layers):
                    profile = profiles[layer]
                    decision = decisions[layer]
                    expected_decision = decide_regions(profile, config)
                    if (bool(decision["skip_near"]) != expected_decision["skip_near"]
                            or bool(decision["skip_far"]) != expected_decision["skip_far"]):
                        raise RuntimeError(f"{task}: layer {layer} decision mismatch")
                    normalized_profiles[str(layer)] = profile
                    normalized_decisions[str(layer)] = {
                        "skip_near": bool(decision["skip_near"]),
                        "skip_far": bool(decision["skip_far"]),
                    }
                result = {
                    "repo_head": source["repo_head"],
                    "context_length": CONTEXT_LENGTH,
                    "freeze_mode": "decision-only-no-reclaim",
                    "gate": base["gate"],
                    "geometry": base["geometry"],
                    "selector": selector,
                    "input_path": path,
                    "input_file_sha256": input_file_sha256,
                    "raw_line_sha256": _sha256_bytes(line.encode("utf-8")),
                    "task": task,
                    "sample_index": sample_index,
                    "sample_id": row.get("id", sample_index),
                    "input_sha256": _sha256_bytes(row["input"].encode("utf-8")),
                    "prompt_tokens": int(ids.shape[-1]),
                    "skip_far_layers": far_layers,
                    "skip_near_layers": near_layers,
                    "skip_both_layers": sorted(set(far_layers) & set(near_layers)),
                    "near_mean_min": min(
                        p["near"]["mean"] for p in profiles.values()),
                    "near_mean_min_layer": min(
                        profiles, key=lambda layer: profiles[layer]["near"]["mean"]),
                    "near_mean_max": max(
                        p["near"]["mean"] for p in profiles.values()),
                    "far_mean_min": min(
                        p["far"]["mean"] for p in profiles.values()),
                    "far_mean_min_layer": min(
                        profiles, key=lambda layer: profiles[layer]["far"]["mean"]),
                    "far_mean_max": max(
                        p["far"]["mean"] for p in profiles.values()),
                    "profiles": normalized_profiles,
                    "decisions": normalized_decisions,
                }
                by_key[key] = result
                _atomic_write(payload())
                print(json.dumps({
                    key: value for key, value in result.items()
                    if key not in ("profiles", "decisions", "selector", "geometry")
                }), flush=True)
                del output, ids, cache
    _atomic_write(payload(complete=True))
    print(json.dumps({"status": "complete", "output": str(OUTPUT),
                      "cells": len(by_key)}), flush=True)


if __name__ == "__main__":
    main()
