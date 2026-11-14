"""按源索引合并官方 LongBench round-robin 分片并重建 canonical runcap。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from benchmark.LongBench.run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX,
    build_lbv1_input_identity,
    build_run_cap_core,
    cap_identity_payload,
    compute_result_binding,
    finalize_run_cap_manifest,
    read_run_cap_manifest,
    resolve_producer_identity,
    runcap_gate_lock,
    write_run_cap_manifest,
)

REPO = Path(__file__).resolve().parents[2]
SHARD_SUFFIX = re.compile(r"_s(?P<rank>\d+)of(?P<world>\d+)\.jsonl$")


def _load_rows(path):
    raw = Path(path).read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _validate_shard(path, dataset):
    path = Path(path).resolve()
    manifest = read_run_cap_manifest(str(path) + RUN_CAP_SIDECAR_SUFFIX)
    result_sha, result_lines = compute_result_binding(str(path))
    if manifest["status"] != "complete":
        raise RuntimeError(f"分片尚未完成: {path}")
    if manifest["dataset"] != dataset:
        raise RuntimeError(f"分片数据集错配: {path}: {manifest['dataset']} != {dataset}")
    if manifest["result_file"] != path.name:
        raise RuntimeError(f"分片 result_file 未绑定 basename: {path}")
    if manifest["result_sha256"] != result_sha or manifest["result_lines"] != result_lines:
        raise RuntimeError(f"分片结果 SHA/行数与 runcap 不一致: {path}")
    if manifest["n_samples"] != result_lines:
        raise RuntimeError(f"分片 n_samples 与实际行数不一致: {path}")
    sharding = manifest.get("sample_sharding")
    if not isinstance(sharding, dict) or sharding.get("algorithm") != "source_order[rank::world_size]":
        raise RuntimeError(f"分片缺少标准 round-robin sample_sharding: {path}")
    rank = sharding["rank"]
    world = sharding["world_size"]
    source_samples = sharding["source_samples"]
    source_fingerprint = manifest["input_identity"]["source_fingerprint"]
    source_indices = list(range(rank, source_samples, world))
    selected_ids = [f"{dataset}:{source_fingerprint}:{index}" for index in source_indices]
    selected_sha = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    if sharding["selected_ids_sha256"] != selected_sha:
        raise RuntimeError(f"分片 selected_ids_sha256 与预期源人口不一致: {path}")
    records = _load_rows(path)
    tokens = manifest["input_tokens"]["per_sample"]
    truncated = manifest["truncated"]["flags"]
    input_hashes = manifest["input_identity"]["actual_input_ids"]["per_sample_sha256"]
    if not (len(records) == len(tokens) == len(truncated) == len(input_hashes)):
        raise RuntimeError(f"分片记录与逐样本 provenance 数量不一致: {path}")
    for position, (record, source_index) in enumerate(zip(records, source_indices)):
        if record.get("_id") != f"{dataset}:{source_index}":
            raise RuntimeError(
                f"分片记录顺序/身份错误: {path}: position={position}, "
                f"_id={record.get('_id')!r}, expected={dataset}:{source_index}"
            )
        expected_input = {
            "input_tokens": tokens[position],
            "input_ids_sha256": record.get("input", {}).get("input_ids_sha256"),
            "truncated": truncated[position],
        }
        if record.get("input") != expected_input:
            raise RuntimeError(f"分片行内 input 与 runcap 不一致: {path}: {record.get('_id')}")
        if record.get("implementation") != manifest["method"]:
            raise RuntimeError(f"分片 implementation 与 runcap method 不一致: {path}")
    suffix = SHARD_SUFFIX.search(path.name)
    if suffix is None or (int(suffix.group("rank")), int(suffix.group("world"))) != (rank, world):
        raise RuntimeError(f"分片文件名与 sample_sharding rank/world 不一致: {path}")
    return {
        "path": path,
        "canonical_path": path.with_name(path.name[:suffix.start()] + ".jsonl"),
        "manifest": manifest,
        "records": records,
        "source_indices": source_indices,
        "tokens": tokens,
        "truncated": truncated,
        "input_hashes": input_hashes,
        "runcap_sha256": hashlib.sha256(
            (Path(str(path) + RUN_CAP_SIDECAR_SUFFIX)).read_bytes()
        ).hexdigest(),
    }


def merge_round_robin_shards(shard_paths, output_path, dataset, data_root, config_path):
    shards = [_validate_shard(path, dataset) for path in shard_paths]
    if not shards:
        raise RuntimeError("至少需要一个分片")
    worlds = {shard["manifest"]["sample_sharding"]["world_size"] for shard in shards}
    if len(worlds) != 1:
        raise RuntimeError(f"分片 world_size 不一致: {sorted(worlds)}")
    world = worlds.pop()
    by_rank = {shard["manifest"]["sample_sharding"]["rank"]: shard for shard in shards}
    if len(by_rank) != len(shards) or set(by_rank) != set(range(world)):
        raise RuntimeError(f"分片必须恰好覆盖 rank=0..{world - 1} 各一次")
    shards = [by_rank[rank] for rank in range(world)]
    output_path = Path(output_path).resolve()
    if any(shard["canonical_path"] != output_path for shard in shards):
        raise RuntimeError(f"canonical 输出名不能由所有分片共同推出: {output_path}")
    if output_path in {shard["path"] for shard in shards}:
        raise RuntimeError("canonical 输出不得覆盖输入分片")

    first = shards[0]["manifest"]
    identity = first["input_identity"]
    shared_fields = (
        "schema", "model_name", "model_path", "method", "dataset",
        "dataset_fingerprint", "effective_cap", "protocol_cap", "cap_treatment",
        "max_new_tokens", "input_identity", "producer",
    )
    for shard in shards[1:]:
        manifest = shard["manifest"]
        for field in shared_fields:
            if field == "input_identity":
                left = dict(identity)
                right = dict(manifest[field])
                left.pop("actual_input_ids", None)
                right.pop("actual_input_ids", None)
                if left != right:
                    raise RuntimeError(f"分片完整源 input_identity 不一致: {shard['path']}")
            elif field == "producer":
                left = {key: first[field].get(key) for key in ("script_sha256", "git_commit")}
                right = {key: manifest[field].get(key) for key in ("script_sha256", "git_commit")}
                if left != right:
                    raise RuntimeError(f"分片 producer 代码身份不一致: {shard['path']}")
            elif manifest[field] != first[field]:
                raise RuntimeError(f"分片 {field} 不一致: {shard['path']}")

    rows = _load_rows(Path(data_root) / f"{dataset}.jsonl")
    prompt_config = json.loads((Path(config_path) / "dataset2prompt.json").read_text(encoding="utf-8"))
    source_identity = build_lbv1_input_identity(rows, prompt_config[dataset])
    if identity["sample_count"] != len(rows) or any(
        identity[key] != source_identity[key]
        for key in ("schema", "canonicalization", "source_sha256", "source_fingerprint",
                    "sample_count", "prompt_template_sha256")
    ):
        raise RuntimeError("runcap 完整源身份与本地原始数据/模板不一致")
    if first["sample_sharding"]["source_samples"] != len(rows):
        raise RuntimeError("sample_sharding.source_samples 与原始数据行数不一致")

    ordered = {}
    for shard in shards:
        for position, (record, source_index) in enumerate(zip(shard["records"], shard["source_indices"])):
            source_row = rows[source_index]
            for field in ("answers", "all_classes", "length"):
                if record.get(field) != source_row.get(field):
                    raise RuntimeError(f"记录与原始源行不一致: {record.get('_id')}: {field}")
            if source_index in ordered:
                raise RuntimeError(f"重复源索引: {source_index}")
            ordered[source_index] = {
                "record": record,
                "token": shard["tokens"][position],
                "truncated": shard["truncated"][position],
                "input_hash": shard["input_hashes"][position],
            }
    if set(ordered) != set(range(len(rows))):
        missing = sorted(set(range(len(rows))) - set(ordered))
        raise RuntimeError(f"合并未恰好覆盖完整源人口，缺失示例: {missing[:10]}")
    samples = [ordered[index] for index in range(len(rows))]

    treatment = first["cap_treatment"]
    cap_gate_info = None
    if treatment.get("type") != "unregistered":
        cap_gate_info = {
            key: treatment.get(key)
            for key in ("registry_protocol_cap", "registry_restricted_context_treatment",
                        "capability", "max_position_embeddings")
        }
    core = build_run_cap_core(
        model_name=first["model_name"], model_path=first["model_path"],
        method=first["method"], dataset=dataset,
        dataset_fingerprint=first["dataset_fingerprint"], run_timestamp="canonical",
        effective_cap=first["effective_cap"], cap_gate_info=cap_gate_info,
        max_new_tokens=first["max_new_tokens"], producer=first["producer"],
        input_identity=source_identity,
    )
    core["canonical_merge"] = {
        "algorithm": "sort_by_source_index_after_round_robin_validation",
        "world_size": world,
        "tool": resolve_producer_identity(__file__),
        "shards": [
            {
                "rank": shard["manifest"]["sample_sharding"]["rank"],
                "result_file": shard["path"].name,
                "result_sha256": shard["manifest"]["result_sha256"],
                "runcap_sha256": shard["runcap_sha256"],
                "n_samples": shard["manifest"]["n_samples"],
            }
            for shard in shards
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with runcap_gate_lock(str(output_path)):
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample["record"], ensure_ascii=False) + "\n")
        result_sha, result_lines = compute_result_binding(str(temporary))
        manifest = finalize_run_cap_manifest(
            core, status="complete", n_samples=len(samples),
            input_tokens=[sample["token"] for sample in samples],
            truncated_flags=[sample["truncated"] for sample in samples],
            result_file=output_path.name, result_sha256=result_sha,
            result_lines=result_lines,
            input_ids_sha256=[sample["input_hash"] for sample in samples],
        )
        # 显式复核 canonical cap payload/hash，避免重建时引入字段漂移。
        expected_payload = cap_identity_payload(
            model_name=first["model_name"], dataset=dataset,
            dataset_fingerprint=first["dataset_fingerprint"], method=first["method"],
            run_timestamp="canonical", effective_cap=first["effective_cap"],
            protocol_cap=first["protocol_cap"],
            cap_treatment_type=first["cap_treatment"]["type"],
            max_new_tokens=first["max_new_tokens"], input_identity=source_identity,
        )
        if manifest["cap_identity"]["payload"] != expected_payload:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("canonical cap_identity payload 重建不一致")
        os.replace(temporary, output_path)
        write_run_cap_manifest(str(output_path) + RUN_CAP_SIDECAR_SUFFIX, manifest)
    return output_path, manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-path", default=str(REPO / "benchmark/LongBench/config"))
    return parser.parse_args()


def main():
    args = parse_args()
    output, manifest = merge_round_robin_shards(
        args.shard_files, args.output, args.dataset, args.data_root, args.config_path
    )
    print(f"merged {len(args.shard_files)} shards -> {output}")
    print(f"lines={manifest['result_lines']} sha256={manifest['result_sha256']}")


if __name__ == "__main__":
    main()
