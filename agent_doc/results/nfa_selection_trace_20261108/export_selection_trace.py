#!/usr/bin/env python3
"""导出真实 Qwen 请求的生产 NFA 选择轨迹；独立进程，不测量 TPOT。

默认只做计划检查。``--execute`` 使用原生 RULER 输入完成一次 dense prefill
和一次 eager decode，在指定层的生产 selector 返回后立即快照持久 workspace。
不修改生产源码，不在 Graph capture 内做 D2H，也不把本脚本用于性能结论。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark.efficiency.benchmark_nfa_end_to_end as benchmark_runtime  # noqa: E402
from benchmark.efficiency.benchmark_nfa_end_to_end import (  # noqa: E402
    _load_input,
    _load_model,
    _strict_region_sets,
)
from benchmark.RULER.pred_ruler import _ruler_eos_ids  # noqa: E402
from sparse_attn.engine.frozen_runner import (  # noqa: E402
    build_frozen_cache,
    prefill_frozen_request,
)
from sparse_attn.engine.graph_runner import GraphRunner  # noqa: E402
from sparse_attn.ops.paged_region_kernels_nfa import nfa_args  # noqa: E402
from sparse_attn.policies.frozen_gate import GateConfig  # noqa: E402
from sparse_attn.policies.region_geometry import RegionGeometry  # noqa: E402
from sparse_attn.scheduler.scheduler import (  # noqa: E402
    ContinuousBatchScheduler,
    greedy_next_tokens,
)

SCHEMA = "nfa-selection-trace-v1"
SOURCE_LABELS = (
    "invalid",
    "l1_far_page",
    "l1_near_page",
    "sink_boundary_bypass",
    "swa_boundary_bypass",
    "sink_protected",
    "swa_protected",
)
REGION_LABELS = ("invalid", "sink", "far", "near", "swa")
WORKSPACE_ARRAYS = {
    "q_full": "query_l1_scaled",
    "q_fine_grouped": "query_l2_grouped_scaled",
    "l1_far": "l1_far_scores",
    "l1_near": "l1_near_scores",
    "topk_far": "l1_far_pages",
    "topk_near": "l1_near_pages",
    "candidate_ids": "candidate_ids",
    "candidate_valid": "candidate_valid",
    "candidate_lower": "candidate_lower",
    "candidate_upper": "candidate_upper",
    "candidate_quota": "candidate_quota",
    "fine_logits": "l2_packed_keys",
    "middle_indices": "middle_ids",
    "middle_counts": "middle_counts",
    "fine_owner_generations": "fine_owner_generations",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True).strip()


def source_identity():
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"])
    staged = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--cached", "--binary"])
    return {
        "head": git_output("rev-parse", "HEAD"),
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(__file__),
        "note": "脚本自身 SHA 单独闭合；HEAD+diff 不表示未跟踪脚本已进入该提交。",
    }


def model_identity(model_path):
    root = Path(model_path).resolve()
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    records = {}
    for name in names:
        path = root / name
        if path.is_file():
            records[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    weights = []
    for path in sorted(root.glob("*.safetensors")):
        weights.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "declared_local_path": str(root),
        "hf_revision": None,
        "hf_revision_status": "本地路径不冒充 HF revision；权重与配置逐文件 SHA 闭合",
        "files": records,
        "weight_files": weights,
    }


def workspace_buffer(workspace, name):
    matches = [
        value for key, value in workspace.items()
        if isinstance(key, tuple) and key and key[0] == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"workspace {name!r} expected once, got {len(matches)}")
    return matches[0]


def cpu_float(tensor):
    return tensor.detach().float().cpu().numpy()


def cpu_array(tensor):
    return tensor.detach().cpu().numpy()


def bf16_bits(tensor):
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {tensor.dtype}")
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy()


def decode_l2_packed(packed, capacity):
    """独立解码 signed-BF16 ordered key；无效项返回 token=-1、score=NaN。"""
    token_bits = 32 if capacity >= 65536 else 16
    packed = np.asarray(packed)
    invalid = np.int64(-32768 << token_bits)
    valid = packed != invalid
    mask = np.int64((1 << token_bits) - 1)
    low = packed.astype(np.int64) & mask
    token = np.where(valid, np.int64(capacity) - low, -1).astype(np.int64)
    ordered = (packed.astype(np.int64) - low) >> token_bits
    bits = np.where(
        ordered >= 0,
        ordered,
        (~(ordered + 0x8000)) & 0xFFFF,
    ).astype(np.uint16)
    scores = (bits.astype(np.uint32) << 16).view(np.float32)
    scores = np.where(valid, scores, np.nan).astype(np.float32)
    return token, scores, bits


def candidate_sources(shape, k_f, k_n, block_size, sink, swa, capacity):
    """按 candidate virtual slot 布局记录来源，不从选中结果反推来源。"""
    if len(shape) != 3:
        raise ValueError("candidate shape must be [M,HKV,R]")
    width = shape[-1]
    r = (k_f + k_n + 2) * block_size
    head = min(sink, capacity)
    tail = min(swa, capacity)
    if width != r + head + tail:
        raise ValueError(f"candidate width {width} != expected {r + head + tail}")
    slots = np.arange(width)
    source = np.zeros(width, dtype=np.uint8)
    source[slots < k_f * block_size] = 1
    source[(slots >= k_f * block_size) & (slots < (k_f + k_n) * block_size)] = 2
    source[(slots >= (k_f + k_n) * block_size) & (slots < (k_f + k_n + 1) * block_size)] = 3
    source[(slots >= (k_f + k_n + 1) * block_size) & (slots < r)] = 4
    source[(slots >= r) & (slots < r + head)] = 5
    source[(slots >= r + head) & (slots < r + head + tail)] = 6
    return np.broadcast_to(source, shape).copy()


def token_regions(token_ids, planned_length, geometry):
    protected, far, near = _strict_region_sets(planned_length, geometry)
    sink = protected - set(range(max(0, planned_length - geometry.window), planned_length))
    swa = protected - sink
    result = np.zeros(np.shape(token_ids), dtype=np.uint8)
    for label, members in ((1, sink), (2, far), (3, near), (4, swa)):
        if members:
            result[np.isin(token_ids, list(members))] = label
    return result


def invert_gather(gather, visible, block_table, page, pad_page):
    """物理槽独立反演逻辑 token，并重算物理地址。"""
    gather = np.asarray(gather)
    visible = np.asarray(visible)
    result = np.full(gather.shape, -1, dtype=np.int64)
    for kv_head in range(gather.shape[0]):
        count = int(visible[kv_head])
        slots = gather[kv_head, :count].astype(np.int64)
        if np.any(slots < 0):
            raise RuntimeError("valid gather contains a negative physical slot")
        physical_pages = slots // page
        offsets = slots % page
        for index, (physical, offset) in enumerate(zip(physical_pages, offsets)):
            matches = np.flatnonzero(block_table == physical)
            if len(matches) != 1 or int(physical) == int(pad_page):
                raise RuntimeError(f"physical page {physical} does not invert uniquely")
            logical = int(matches[0]) * page + int(offset)
            result[kv_head, index] = logical
            if int(block_table[matches[0]]) * page + int(offset) != int(slots[index]):
                raise RuntimeError("logical/physical address round-trip failed")
    return result


def expected_native_kv(prefill, current, logical_ids, prompt_length):
    """从原生 prefill/decode K/V 按逻辑 ID 重建 Gather 期望值。"""
    prefill_k, prefill_v = prefill
    current_k, current_v = current
    hkv = prefill_k.shape[1]
    max_count = logical_ids.shape[-1]
    expected_k = torch.zeros(
        (hkv, max_count, prefill_k.shape[-1]), dtype=torch.bfloat16)
    expected_v = torch.zeros(
        (hkv, max_count, prefill_v.shape[-1]), dtype=torch.bfloat16)
    for kv_head in range(hkv):
        for index, token in enumerate(logical_ids[kv_head]):
            token = int(token)
            if token < 0:
                continue
            if token < prompt_length:
                expected_k[kv_head, index] = prefill_k[0, kv_head, token]
                expected_v[kv_head, index] = prefill_v[0, kv_head, token]
            elif token == prompt_length:
                expected_k[kv_head, index] = current_k[0, kv_head, 0]
                expected_v[kv_head, index] = current_v[0, kv_head, 0]
            else:
                raise RuntimeError(f"token {token} exceeds first decode position")
    return expected_k, expected_v


def capture_layer(cache, layer_idx, req_ids, query_states, attention, decode_write,
                  prefill_native, prompt_length, arrays, reports):
    workspace = cache._nfa_selector_workspace
    request_id = req_ids[0]
    state = cache.states[request_id]
    if state.skip_far_by_layer[layer_idx] and state.skip_near_by_layer[layer_idx]:
        raise RuntimeError(
            f"layer {layer_idx} is protected-only; choose a layer with real L1/L2 work")
    prefix = f"layer_{layer_idx:03d}/"
    for workspace_name, output_name in WORKSPACE_ARRAYS.items():
        tensor = workspace_buffer(workspace, workspace_name)
        if workspace_name in {"q_full", "q_fine_grouped", "l1_far", "l1_near"}:
            arrays[prefix + output_name] = cpu_float(tensor)
        else:
            arrays[prefix + output_name] = cpu_array(tensor)

    stats_matches = [
        buffers for key, buffers in cache._nfa_page_stats_persistent.items()
        if key[0] == layer_idx and key[1] == 1
    ]
    if len(stats_matches) != 1:
        raise RuntimeError("target layer page statistics are not unique")
    for name, tensor in zip(("page_min", "page_max", "page_sum"), stats_matches[0]):
        arrays[prefix + name] = cpu_float(tensor[:, request_id:request_id + 1])
    arrays[prefix + "page_stats_lengths"] = cpu_array(
        cache._nfa_page_stats_lengths[layer_idx, request_id:request_id + 1])
    arrays[prefix + "query_post_rope"] = cpu_float(query_states[:1])
    arrays[prefix + "subspace_coordinates"] = cpu_array(
        attention.indexer._subspace_indices(query_states.device).long())

    planned_length = int(cache.planned_lengths_buf[0])
    if planned_length != prompt_length + 1:
        raise RuntimeError("trace only supports the first eager decode step")
    geometry = cache.geometry
    visible_full = cache.visible_lens_for_layer(layer_idx)[request_id].detach().cpu()
    max_count = int(visible_full.max())
    gather_full = cache.gather_indices_for_layer(layer_idx)[request_id]
    gather = cpu_array(gather_full[:, :max_count])
    visible = cpu_array(visible_full)
    block_table = cpu_array(cache.block_table_buf[layer_idx, request_id])
    logical = invert_gather(
        gather, visible, block_table, geometry.page, cache.layers[layer_idx].PAD_PAGE_ID)

    candidate = arrays[prefix + "candidate_ids"]
    arrays[prefix + "candidate_source"] = candidate_sources(
        candidate.shape,
        arrays[prefix + "l1_far_pages"].shape[-1],
        arrays[prefix + "l1_near_pages"].shape[-1],
        int(attention.indexer.args.tia_block_size),
        int(attention.indexer.sink_tokens),
        int(attention.indexer.sliding_window_size),
        cache.max_blocks * int(attention.indexer.args.tia_block_size),
    )
    candidate_region = token_regions(candidate, planned_length, geometry)
    candidate_region[~arrays[prefix + "candidate_valid"]] = 0
    arrays[prefix + "candidate_region"] = candidate_region
    middle = arrays[prefix + "middle_ids"]
    middle_counts = arrays[prefix + "middle_counts"]
    middle_region = token_regions(middle, planned_length, geometry)
    middle_valid = np.arange(middle.shape[-1])[None, None, :] < middle_counts[..., None]
    middle_region[~middle_valid] = 0
    arrays[prefix + "middle_region"] = middle_region
    arrays[prefix + "final_logical_ids"] = logical.astype(np.int32)
    arrays[prefix + "final_region"] = token_regions(logical, planned_length, geometry)
    arrays[prefix + "gather_physical_slots"] = gather
    arrays[prefix + "visible_counts"] = visible
    arrays[prefix + "block_table"] = block_table

    capacity = cache.max_blocks * int(attention.indexer.args.tia_block_size)
    l2_tokens, l2_scores, l2_bits = decode_l2_packed(
        arrays[prefix + "l2_packed_keys"], capacity)
    arrays[prefix + "l2_token_ids"] = l2_tokens
    arrays[prefix + "l2_scores"] = l2_scores
    arrays[prefix + "l2_score_bf16_bits"] = l2_bits
    valid_l2 = l2_tokens >= 0
    if not np.array_equal(l2_tokens[valid_l2], candidate[valid_l2]):
        raise RuntimeError("decoded L2 token and candidate token differ")
    if not arrays[prefix + "candidate_valid"][valid_l2].all():
        raise RuntimeError("valid L2 score references an invalid candidate slot")

    protected, far, near = _strict_region_sets(planned_length, geometry)
    excluded = ((far if state.skip_far_by_layer[layer_idx] else set())
                | (near if state.skip_near_by_layer[layer_idx] else set()))
    for kv_head in range(gather.shape[0]):
        count = int(visible[kv_head])
        final = set(logical[kv_head, :count].tolist())
        middle_count = int(middle_counts[0, kv_head])
        expected = set(middle[0, kv_head, :middle_count].tolist()) | protected
        if final != expected:
            raise RuntimeError(f"final union mismatch at layer {layer_idx}, KV {kv_head}")
        if len(final) != count or final & excluded or not protected <= final:
            raise RuntimeError("strict region/duplicate contract failed")
        ordered = logical[kv_head, :count]
        if len(ordered) > 1 and np.any(ordered[1:] <= ordered[:-1]):
            raise RuntimeError("final logical IDs are not strictly increasing")

    current = (decode_write[0].detach().cpu(), decode_write[1].detach().cpu())
    expected_k, expected_v = expected_native_kv(
        prefill_native[layer_idx], current, logical, prompt_length)
    actual_k = torch.zeros_like(expected_k)
    actual_v = torch.zeros_like(expected_v)
    pool = cache.layers[layer_idx]
    for kv_head in range(gather.shape[0]):
        count = int(visible[kv_head])
        slots = gather_full[kv_head, :count]
        physical = slots // geometry.page
        offsets = slots % geometry.page
        actual_k[kv_head, :count] = pool.kv_pool_k[
            physical, 0, kv_head, offsets].detach().cpu()
        actual_v[kv_head, :count] = pool.kv_pool_v[
            physical, 0, kv_head, offsets].detach().cpu()
    if not torch.equal(actual_k, expected_k) or not torch.equal(actual_v, expected_v):
        raise RuntimeError("Gather physical KV differs from native logical KV")
    arrays[prefix + "gathered_k_bf16_bits"] = bf16_bits(actual_k)
    arrays[prefix + "gathered_v_bf16_bits"] = bf16_bits(actual_v)
    arrays[prefix + "native_source_k_bf16_bits"] = bf16_bits(expected_k)
    arrays[prefix + "native_source_v_bf16_bits"] = bf16_bits(expected_v)
    reports.append({
        "layer": layer_idx,
        "request_id": request_id,
        "decode_step": 0,
        "planned_length": planned_length,
        "skip_far": bool(state.skip_far_by_layer[layer_idx]),
        "skip_near": bool(state.skip_near_by_layer[layer_idx]),
        "visible_counts": visible.tolist(),
        "middle_counts": middle_counts.tolist(),
        "valid_l2_scores": int(valid_l2.sum()),
        "logical_physical_address_roundtrip": True,
        "final_union_matches_middle_plus_protected": True,
        "gather_kv_matches_native_prefill_and_current_decode_bit_exact": True,
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model-path", default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--data-root", default="/home/wangyuanshuo02/datasets/RULER")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--context-length", type=int, default=32768,
                        choices=(32768, 65536, 131072))
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--gate-epsilon", type=float, default=0.05)
    parser.add_argument("--visible-len", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    layers = sorted({int(value) for value in args.layers.split(",")})
    if not args.execute:
        print(json.dumps({
            "status": "plan-only",
            "gpu_started": False,
            "requires": ["--execute", "idle CUDA device", "model/data local identity"],
            "scope": "one native RULER request, first eager decode, selected layers",
            "layers": layers,
        }, ensure_ascii=False, indent=2))
        return
    if not layers or layers[0] < 0:
        raise ValueError("layers must be non-negative")
    output_dir = Path(args.output_dir).resolve()
    npz_path = output_dir / "selection_trace.npz"
    manifest_path = output_dir / "manifest.json"
    receipt_path = output_dir / "receipt.json"
    for path in (npz_path, manifest_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(20261108)

    benchmark_args = SimpleNamespace(
        model_path=args.model_path,
        data_root=args.data_root,
        context_length=args.context_length,
        task=args.task,
        sample_index=args.sample_index,
        input_provenance="native-ruler",
        method="nfa",
    )
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model_path)
    prompt, input_receipt = _load_input(benchmark_args, tokenizer)
    patch_args = nfa_args()
    # 生产 helper 的通用默认值不是本证据包工作点；模型 indexer 与 cache
    # 必须共享同一个显式 page=16 配置，不能只修改局部 geometry。
    patch_args.tia_block_size = 16
    original_nfa_args = benchmark_runtime.nfa_args
    benchmark_runtime.nfa_args = lambda: patch_args
    try:
        model = _load_model(benchmark_args, 2)
    finally:
        benchmark_runtime.nfa_args = original_nfa_args
    configured_pages = {
        int(layer.self_attn.indexer.args.tia_block_size)
        for layer in model.model.layers
    }
    if configured_pages != {16}:
        raise RuntimeError(f"model indexer page sizes differ from 16: {configured_pages}")
    if layers[-1] >= model.config.num_hidden_layers:
        raise ValueError("target layer exceeds model depth")
    eos_ids = _ruler_eos_ids(model, tokenizer)
    geometry = RegionGeometry(
        sink=patch_args.nfa_sink_tokens,
        near=patch_args.nfa_near_len,
        window=patch_args.nfa_swa_size,
        page=patch_args.tia_block_size,
        max_new_tokens=2,
    )
    cache = build_frozen_cache(
        model, geometry, prompt.numel(), max_requests=1,
        max_visible_len=args.visible_len)
    gate = GateConfig(epsilon=args.gate_epsilon)
    prefill_native = {}
    arrays = {}
    reports = []
    original_write_prefill = cache.write_prefill

    def write_prefill_with_native(layer_idx, request_id, key_states, value_states, *extra, **kwargs):
        if layer_idx in layers:
            prefill_native[layer_idx] = (
                key_states.detach().cpu(), value_states.detach().cpu())
        return original_write_prefill(
            layer_idx, request_id, key_states, value_states, *extra, **kwargs)

    import sparse_attn.indexer.nfa_layer_batch_selector as selector_module
    original_select = selector_module.select_topk_for_layer_batched

    def select_with_trace(cache_arg, layer_idx, req_ids, query_states,
                          attention=None, decode_write=None):
        result = original_select(
            cache_arg, layer_idx, req_ids, query_states,
            attention=attention, decode_write=decode_write)
        if layer_idx in layers and layer_idx not in prefill_native:
            raise RuntimeError(f"native prefill K/V missing for layer {layer_idx}")
        if layer_idx in layers and not any(row["layer"] == layer_idx for row in reports):
            capture_layer(
                cache_arg, layer_idx, req_ids, query_states, attention, decode_write,
                prefill_native, prompt.numel(), arrays, reports)
        return result

    cache.write_prefill = write_prefill_with_native
    selector_module.select_topk_for_layer_batched = select_with_trace
    try:
        with torch.inference_mode():
            output, _ = prefill_frozen_request(
                model, prompt.unsqueeze(0), gate, geometry,
                cache=cache, request_id=0)
            first = greedy_next_tokens(
                output.logits, eos_ids, generated_counts=0, min_new_tokens=1)
            generated = [int(first[0])]
            del output
            cache.transition_request(0, "GENERATE")
            scheduler = ContinuousBatchScheduler(
                cache=cache, max_batch=1, eos_ids=eos_ids, pad_token_id=0)
            scheduler.add_request(0)
            scheduler.next_tokens[0] = generated[0]
            runner = GraphRunner(
                scheduler=scheduler, buckets=(1,), visible_buckets=(args.visible_len,),
                device=prompt.device)
            _, tokens = runner.step(
                model, cache, generated_counts=[1], min_new_tokens=1,
                return_next_tokens=True, stop_on_eos=False)
            generated.append(int(tokens[0]))
            torch.cuda.synchronize()
    finally:
        selector_module.select_topk_for_layer_batched = original_select
        cache.write_prefill = original_write_prefill
    if [row["layer"] for row in reports] != layers:
        raise RuntimeError(f"incomplete traces: {[row['layer'] for row in reports]} != {layers}")

    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    input_ids = prompt.detach().cpu().numpy().astype("<i8", copy=False)
    input_receipt["input_ids_sha256"] = hashlib.sha256(input_ids.tobytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "status": "complete-real-model-eager-trace",
        "classification": "A-within-single-request-diagnostic-scope",
        "scope": {
            "requests": 1,
            "decode_steps": 1,
            "execution": "eager",
            "layers": layers,
            "not_a_performance_measurement": True,
            "not_a_dataset_wide_quality_result": True,
        },
        "coordinates": {
            "request_id": 0,
            "decode_step": 0,
            "planned_length": prompt.numel() + 1,
            "token_coordinate": "zero-based logical token ID",
            "physical_slot": "physical_page * page_size + page_offset",
            "layer_key_prefix": "layer_XXX/",
        },
        "model": model_identity(args.model_path),
        "input": input_receipt,
        "nfa_config": {
            key: getattr(patch_args, key) for key in (
                "nfa_far_method", "nfa_near_method", "nfa_near_pages",
                "nfa_near_tokens", "nfa_near_len", "nfa_subspace",
                "tia_level2_cmp_ratio", "nfa_sink_tokens", "nfa_swa_size",
                "tia_level1_topk", "tia_level2_topk", "tia_block_size",
                "nfa_enable_layer_skip", "nfa_far_skip_cache",
            )
        },
        "gate": {"mode": "dynamic-frozen", "epsilon": args.gate_epsilon},
        "source": source_identity(),
        "array_conventions": {
            "bf16_bits": "uint16 raw BF16 payload；按 little-endian uint16 还原，不是 FP16",
            "l2_scores": "从生产 packed key 解码的 BF16 分数，以 FP32 数组无损表示；无效项 NaN",
            "l2_score_bf16_bits": "与 l2_scores 对应的原始 BF16 位；无效项位模式不用于打分",
            "candidate_source_labels": dict(enumerate(SOURCE_LABELS)),
            "region_labels": dict(enumerate(REGION_LABELS)),
            "candidate_source": "由 virtual slot 布局直接生成，不从最终集合反推",
            "native_source": "原生 prefill K/V 加第一步 decode 当前 K/V，按逻辑 ID 重建",
        },
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(arrays.items())
        },
        "validation": reports,
        "generated_token_ids": generated,
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "cuda_device": torch.cuda.get_device_name(prompt.device),
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    write_json(manifest_path, manifest)
    write_json(receipt_path, {
        "schema": "nfa-selection-trace-receipt-v1",
        "status": "success",
        "selection_trace_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
        "layers": layers,
        "validation_passed": True,
    })
    print(json.dumps({
        "status": "NFA_SELECTION_TRACE_OK",
        "npz": str(npz_path),
        "arrays": len(arrays),
        "layers": layers,
        "visible_counts": {row["layer"]: row["visible_counts"] for row in reports},
        "sha256": sha256_file(npz_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
