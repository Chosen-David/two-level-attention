#!/usr/bin/env python3
"""独立校验 selection_trace.npz；不加载模型、不导入生产 selector 或导出器。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def region_members(length, sink, near, window, page):
    sink_hi = min(sink, length)
    near_hi = max(sink_hi, length - window)
    far_hi = min(near_hi, max(sink_hi, ((length - window - near) // page) * page))
    protected = set(range(sink_hi)) | set(range(near_hi, length))
    far = set(range(sink_hi, far_hi))
    near_set = set(range(far_hi, near_hi))
    return protected, far, near_set


def regions(tokens, length, sink, near, window, page):
    protected, far, near_set = region_members(length, sink, near, window, page)
    swa_start = max(0, length - window)
    sink_set = protected - set(range(swa_start, length))
    swa = protected - sink_set
    output = np.zeros(tokens.shape, dtype=np.uint8)
    for label, members in ((1, sink_set), (2, far), (3, near_set), (4, swa)):
        if members:
            output[np.isin(tokens, list(members))] = label
    return output


def expected_sources(shape, k_f, k_n, block, sink, swa, capacity):
    width = shape[-1]
    r = (k_f + k_n + 2) * block
    expected_width = r + min(sink, capacity) + min(swa, capacity)
    require(width == expected_width, "candidate width/layout mismatch")
    slots = np.arange(width)
    output = np.zeros(width, dtype=np.uint8)
    output[slots < k_f * block] = 1
    output[(slots >= k_f * block) & (slots < (k_f + k_n) * block)] = 2
    output[(slots >= (k_f + k_n) * block) & (slots < (k_f + k_n + 1) * block)] = 3
    output[(slots >= (k_f + k_n + 1) * block) & (slots < r)] = 4
    output[(slots >= r) & (slots < r + min(sink, capacity))] = 5
    output[slots >= r + min(sink, capacity)] = 6
    return np.broadcast_to(output, shape).copy()


def unpack_scores(packed, capacity):
    bits_per_token = 32 if capacity >= 65536 else 16
    packed = packed.astype(np.int64, copy=False)
    invalid = np.int64(-32768 << bits_per_token)
    valid = packed != invalid
    token_mask = np.int64((1 << bits_per_token) - 1)
    delta = packed & token_mask
    tokens = np.where(valid, np.int64(capacity) - delta, -1)
    ordered = (packed - delta) >> bits_per_token
    bf16 = np.where(
        ordered >= 0,
        ordered,
        (~(ordered + 32768)) & 65535,
    ).astype(np.uint16)
    scores = (bf16.astype(np.uint32) << 16).view(np.float32)
    scores = np.where(valid, scores, np.nan).astype(np.float32)
    return tokens, scores, bf16, valid


def validate_layer(data, manifest, layer):
    prefix = f"layer_{layer:03d}/"

    def arr(name):
        return data[prefix + name]

    length = manifest["coordinates"]["planned_length"]
    config = manifest["nfa_config"]
    sink = int(config["nfa_sink_tokens"])
    near = int(config["nfa_near_len"])
    window = int(config["nfa_swa_size"])
    page = int(config["tia_block_size"])
    block = int(config["tia_block_size"])
    capacity = arr("block_table").shape[0] * block
    visible = arr("visible_counts")
    gather = arr("gather_physical_slots")
    logical = arr("final_logical_ids")
    table = arr("block_table")
    require(logical.shape == gather.shape, "logical/gather shape mismatch")
    require(visible.shape == (gather.shape[0],), "visible shape mismatch")
    require(np.all(visible >= 0) and np.all(visible <= gather.shape[1]), "visible out of range")

    protected, far, near_set = region_members(length, sink, near, window, page)
    report = next(row for row in manifest["validation"] if row["layer"] == layer)
    excluded = ((far if report["skip_far"] else set())
                | (near_set if report["skip_near"] else set()))
    middle = arr("middle_ids")
    middle_counts = arr("middle_counts")
    for kv in range(gather.shape[0]):
        count = int(visible[kv])
        ids = logical[kv, :count].astype(np.int64)
        slots = gather[kv, :count].astype(np.int64)
        require(np.all(ids >= 0) and np.all(ids < length), "final logical ID out of range")
        require(np.all(slots >= 0), "negative valid physical slot")
        require(len(set(ids.tolist())) == count, "duplicate final logical IDs")
        require(len(set(slots.tolist())) == count, "duplicate final physical slots")
        if count > 1:
            require(np.all(ids[1:] > ids[:-1]), "final IDs not strictly increasing")
        physical = slots // page
        offsets = slots % page
        recovered = []
        for phys, off, slot in zip(physical, offsets, slots):
            pages = np.flatnonzero(table == phys)
            require(len(pages) == 1, "physical page not uniquely owned")
            token = int(pages[0]) * page + int(off)
            recovered.append(token)
            require(int(table[pages[0]]) * page + int(off) == int(slot), "address mismatch")
        require(np.array_equal(np.asarray(recovered), ids), "logical inversion mismatch")
        expected = set(middle[0, kv, :int(middle_counts[0, kv])].tolist()) | protected
        require(set(ids.tolist()) == expected, "final union mismatch")
        require(protected <= set(ids.tolist()), "protected tokens missing")
        require(not set(ids.tolist()) & excluded, "frozen token re-entered")
        require(np.all(logical[kv, count:] == -1), "invalid logical tail must be -1")

    candidate = arr("candidate_ids")
    candidate_valid = arr("candidate_valid")
    tokens, scores, score_bits, valid_l2 = unpack_scores(arr("l2_packed_keys"), capacity)
    require(np.array_equal(tokens, arr("l2_token_ids")), "L2 token decode mismatch")
    require(np.array_equal(score_bits, arr("l2_score_bf16_bits")), "L2 BF16 bits mismatch")
    require(np.array_equal(scores, arr("l2_scores"), equal_nan=True), "L2 score mismatch")
    require(np.array_equal(tokens[valid_l2], candidate[valid_l2]), "L2/candidate mismatch")
    require(candidate_valid[valid_l2].all(), "L2 score references invalid candidate")
    present = set(candidate[candidate_valid].ravel().tolist())
    require(set(tokens[valid_l2].ravel().tolist()) <= present, "L2 token outside candidates")

    expected_source = expected_sources(
        candidate.shape,
        arr("l1_far_pages").shape[-1],
        arr("l1_near_pages").shape[-1],
        block,
        sink,
        window,
        capacity,
    )
    require(np.array_equal(expected_source, arr("candidate_source")), "candidate source mismatch")
    candidate_region = regions(candidate, length, sink, near, window, page)
    candidate_region[~candidate_valid] = 0
    require(np.array_equal(candidate_region, arr("candidate_region")), "candidate region mismatch")
    middle_region = regions(middle, length, sink, near, window, page)
    middle_valid = np.arange(middle.shape[-1])[None, None, :] < middle_counts[..., None]
    middle_region[~middle_valid] = 0
    require(np.array_equal(middle_region, arr("middle_region")), "middle region mismatch")
    final_region = regions(logical, length, sink, near, window, page)
    final_region[logical < 0] = 0
    require(np.array_equal(final_region, arr("final_region")), "final region mismatch")

    for kind in ("k", "v"):
        gathered = arr(f"gathered_{kind}_bf16_bits")
        native = arr(f"native_source_{kind}_bf16_bits")
        require(gathered.dtype == np.uint16 and native.dtype == np.uint16, "KV not BF16 bits")
        require(np.array_equal(gathered, native), f"Gather {kind} differs from native source")

    return {
        "layer": layer,
        "visible_counts": visible.tolist(),
        "middle_counts": middle_counts.tolist(),
        "candidate_slots": int(candidate.size),
        "valid_l2_scores": int(valid_l2.sum()),
        "checks": [
            "logical-to-physical roundtrip",
            "strict region and final union",
            "packed L2 score/token decode",
            "candidate source and region",
            "Gather/native K/V bit equality",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-production-scope", action="store_true")
    args = parser.parse_args()
    package = Path(args.package_dir).resolve()
    npz_path = package / "selection_trace.npz"
    manifest_path = package / "manifest.json"
    receipt_path = package / "receipt.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with receipt_path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    require(receipt["status"] == "success", "export receipt not successful")
    require(receipt["selection_trace_sha256"] == sha256_file(npz_path), "NPZ SHA mismatch")
    require(receipt["manifest_sha256"] == sha256_file(manifest_path), "manifest SHA mismatch")
    if args.require_production_scope:
        expected_config = {
            "nfa_far_method": "minmax", "nfa_near_method": "avg",
            "nfa_near_pages": 32, "nfa_near_tokens": "off",
            "nfa_near_len": 4096, "nfa_subspace": "tail",
            "tia_level2_cmp_ratio": 4, "nfa_sink_tokens": 4,
            "nfa_swa_size": 128, "tia_level1_topk": 128,
            "tia_level2_topk": 1024, "tia_block_size": 16,
            "nfa_enable_layer_skip": False, "nfa_far_skip_cache": "retain",
        }
        for key, expected in expected_config.items():
            actual = manifest["nfa_config"].get(key)
            require(actual == expected,
                    f"production scope mismatch {key}: {actual!r} != {expected!r}")
        require(manifest["scope"] == {
            "requests": 1, "decode_steps": 1, "execution": "eager",
            "layers": [0, 17, 35], "not_a_performance_measurement": True,
            "not_a_dataset_wide_quality_result": True,
        }, "production request/layer/execution scope mismatch")
        require(manifest["coordinates"]["planned_length"] == 32613,
                "production planned length mismatch")
        require(manifest["input"]["sha256"] ==
                "e851c310be0cc7d1fa32469c9443f8fb2dd65ab6b3b28d8f2d700db86399c2db",
                "production input SHA mismatch")
        require(manifest["input"]["input_ids_sha256"] ==
                "db96ef119fc1815ce43912b85b17180f6b33a5ae2eca2dbae6056643c2008999",
                "production input IDs SHA mismatch")
        require(manifest["gate"] == {"mode": "dynamic-frozen", "epsilon": 0.05},
                "production gate mismatch")
    reports = []
    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            require(key in manifest["arrays"], f"undeclared array {key}")
            require(list(data[key].shape) == manifest["arrays"][key]["shape"], f"shape mismatch {key}")
            require(str(data[key].dtype) == manifest["arrays"][key]["dtype"], f"dtype mismatch {key}")
        require(set(data.files) == set(manifest["arrays"]), "array inventory mismatch")
        for layer in manifest["scope"]["layers"]:
            reports.append(validate_layer(data, manifest, layer))
    result = {
        "schema": "nfa-selection-trace-independent-validation-v1",
        "status": "PASS",
        "production_scope_checked": bool(args.require_production_scope),
        "selection_trace_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
        "layers": reports,
        "scope": manifest["scope"],
        "note": "独立读取 NPZ；未加载模型、未导入生产 selector 或导出器。",
    }
    output = Path(args.output) if args.output else package / "validation.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": "NFA_SELECTION_TRACE_VALIDATION_PASS",
        "layers": manifest["scope"]["layers"],
        "visible_counts": {row["layer"]: row["visible_counts"] for row in reports},
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
