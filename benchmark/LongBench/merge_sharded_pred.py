# -*- coding: utf-8 -*-
"""Merge sharded LongBench pred jsonl files back into the canonical output
and rebuild the inline run-cap sidecar with the full-dataset identity.

Why this exists: for long-tail tasks (e.g. gov_report) a single GPU is the
wall-clock bottleneck. We split the input jsonl by line range and run one
pred.py per shard (each with a distinct --t). pred.py buffers predictions and
writes the jsonl + runcap sidecar only at the end, so shard outputs are
complete, valid per-shard artifacts. This script concatenates them in source
order and rebuilds the sidecar for the merged file so eval.py's fail-closed
runcap validation still passes.

The merged manifest is byte-equivalent in policy to what an unsharded pred.py
run would have produced: full input identity, full per-sample token hashes,
full per-sample input_tokens/truncated flags, canonical --t.
"""
import argparse
import glob
import json
import os
import re

from benchmark.LongBench.run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX,
    build_lbv1_input_identity,
    build_run_cap_core,
    compute_result_binding,
    finalize_run_cap_manifest,
    resolve_producer_identity,
    write_run_cap_manifest,
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_lines(path):
    with open(path, "rb") as f:
        return f.read().splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-tags", required=True,
                    help="comma-separated --t values used by each shard, in source order")
    ap.add_argument("--canonical-tag", required=True,
                    help="canonical --t for the merged output")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--method", default="asynctls_vendor")
    args = ap.parse_args()

    config_path = args.config_path
    prompt_format = _read_json(os.path.join(config_path, "dataset2prompt.json"))[args.dataset]
    max_gen = _read_json(os.path.join(config_path, "dataset2maxlen.json"))[args.dataset]
    max_length = _read_json(os.path.join(config_path, "model2maxlen.json"))[args.model]

    # Full ordered source rows, identical to pred.py's load_dataset("json", ...).
    data = []
    with open(os.path.join(args.dataset_path, args.dataset + ".jsonl"),
              "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    input_identity = build_lbv1_input_identity(data, prompt_format)
    producer = resolve_producer_identity(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred.py"))
    core = build_run_cap_core(
        model_name=args.model,
        model_path=args.model_path,
        method=args.method,
        dataset=args.dataset,
        dataset_fingerprint=input_identity["source_fingerprint"],
        run_timestamp=args.canonical_tag,
        effective_cap=max_length,
        cap_gate_info=None,  # GLM-4.7-Flash is unregistered in model_context_caps.json
        max_new_tokens=max_gen,
        producer=producer,
        input_identity=input_identity,
    )

    pred_dir = os.path.join(args.output_dir, "pred")
    dataset_prefix = args.dataset.split("-")[0]
    shard_paths = []
    all_tokens = []
    all_truncated = []
    all_input_ids = []
    for tag in args.shard_tags.split(","):
        tag = tag.strip()
        matches = sorted(glob.glob(
            os.path.join(pred_dir, f"{dataset_prefix}-*{tag}.jsonl")))
        if len(matches) != 1:
            raise SystemExit(f"shard tag {tag!r} matched {len(matches)} files: {matches}")
        shard_paths.append(matches[0])
        sidecar = matches[0] + RUN_CAP_SIDECAR_SUFFIX
        manifest = _read_json(sidecar)
        toks = manifest.get("input_tokens") or {}
        trunc = manifest.get("truncated") or {}
        ids = (manifest.get("input_identity") or {}).get("actual_input_ids") or {}
        per_sample_tokens = toks.get("per_sample") or []
        per_sample_flags = trunc.get("flags") or []
        per_sample_ids = ids.get("per_sample_sha256") or []
        if not (len(per_sample_tokens) == len(per_sample_flags) == len(per_sample_ids)):
            raise SystemExit(
                f"shard {tag!r} sidecar lists inconsistent: "
                f"{len(per_sample_tokens)}/{len(per_sample_flags)}/{len(per_sample_ids)}")
        all_tokens.extend(per_sample_tokens)
        all_truncated.extend(per_sample_flags)
        all_input_ids.extend(per_sample_ids)

    if len(all_tokens) != len(data) or len(all_input_ids) != len(data):
        raise SystemExit(
            f"merged provenance count {len(all_tokens)}/{len(all_input_ids)} "
            f"!= dataset rows {len(data)}")

    canonical_path = None
    for tag in args.shard_tags.split(","):
        suffix = f"-{tag.strip()}.jsonl"
        if shard_paths[0].endswith(suffix):
            canonical_path = shard_paths[0][:-len(suffix)] + f"-{args.canonical_tag}.jsonl"
            break
    if canonical_path is None:
        canonical_path = re.sub(r"_govs\d+\.jsonl$", ".jsonl", shard_paths[0])
    if not canonical_path.endswith(".jsonl") or canonical_path == shard_paths[0]:
        raise SystemExit(f"failed to derive canonical path from {shard_paths[0]}")

    merged = []
    for path in shard_paths:
        merged.extend(_load_lines(path))
    with open(canonical_path, "wb") as f:
        for line in merged:
            f.write(line)
            f.write(b"\n")

    runcap_sha, runcap_lines = compute_result_binding(canonical_path)
    manifest = finalize_run_cap_manifest(
        core,
        status="complete",
        n_samples=len(data),
        input_tokens=all_tokens,
        truncated_flags=all_truncated,
        result_file=os.path.basename(canonical_path),
        result_sha256=runcap_sha,
        result_lines=runcap_lines,
        input_ids_sha256=all_input_ids,
    )
    write_run_cap_manifest(canonical_path + RUN_CAP_SIDECAR_SUFFIX, manifest)

    # Move shard jsonl + sidecars out of pred/ so eval.py only sees the
    # canonical file, while keeping shard provenance for the audit trail.
    shard_dir = os.path.join(pred_dir, f"{dataset_prefix}_shards")
    os.makedirs(shard_dir, exist_ok=True)
    for path in shard_paths:
        for src in [path] + [path + s for s in
                            (RUN_CAP_SIDECAR_SUFFIX, ".runcap_gate.lock")]:
            if os.path.lexists(src):
                dst = os.path.join(shard_dir, os.path.basename(src))
                os.replace(src, dst)

    print(f"merged {len(shard_paths)} shards -> {canonical_path}")
    print(f"lines={runcap_lines} sha256={runcap_sha}")


if __name__ == "__main__":
    main()
