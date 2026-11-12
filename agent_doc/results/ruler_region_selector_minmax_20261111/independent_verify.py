#!/usr/bin/env python3
"""Single-pool MinMax RULER 独立只读验收；不导入 producer 汇总/指针逻辑。"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
LENGTHS = (32768, 65536, 131072)
BUDGET = {task: (120 if task == "cwe" else 50 if task == "fwe" else 30 if task == "vt" else 128)
          for task in TASKS}
LEFTCAP_CELLS = {(131072, "fwe"), (65536, "niah_multivalue"), (32768, "niah_multivalue")}
SKIP_CELLS = {(131072, "niah_single_1"), (131072, "niah_single_2")}
EOS = [151645, 151643]
STATUS_NAMES = [
    "ruler_minmax_only_20261111_status_pc33_gpu0.json",
    "ruler_minmax_only_20261111_status_pc33_gpu1.json",
    *[f"ruler_minmax_only_20261111_status_pc02_gpu{i}.json" for i in range(4)],
    "ruler_minmax_only_20261111_status_pc40_gpu1.json",
    "ruler_minmax_only_20261111_status_pc32_gpu0.json",
    "ruler_minmax_leftcap_20261112_status_pc31_gpu0.json",
]


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_answers(answers):
    return hashlib.sha256(json.dumps(
        answers, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def atomic_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def independent_sample_score(pred, answers):
    if not answers:
        return 0.0
    lowered = pred.lower()
    return 100.0 * sum(ref.lower() in lowered for ref in answers) / len(answers)


def check_assets(config, source_root, model_root, data_root, check_model=True):
    for relative, expected in config["source"]["files"].items():
        path = source_root / relative
        require(path.is_file() and not path.is_symlink(), f"source missing/symlink: {path}")
        require(sha256(path) == expected, f"source SHA mismatch: {path}")
    if check_model:
        for relative, expected in config["model"]["weight_files_sha256"].items():
            path = model_root / relative
            require(path.is_file(), f"model file missing: {path}")
            require(sha256(path) == expected, f"model SHA mismatch: {path}")
        for relative, expected in config["inputs_sha256"].items():
            path = data_root / relative
            require(path.is_file(), f"input missing: {path}")
            require(sha256(path) == expected, f"input SHA mismatch: {path}")


def resolve_cell(root, length, task):
    directory = root / f"L{length}" / "pred_rsel"
    pointers = sorted(directory.glob(f"{task}-*.jsonl.nfa_gen"))
    require(len(pointers) == 1, f"pointer count {length}/{task}: {len(pointers)}")
    pointer = pointers[0]
    logical = Path(str(pointer)[:-len(".nfa_gen")])
    generation_name = pointer.read_text(encoding="utf-8").strip()
    require(generation_name.startswith(logical.name + ".gen-"), f"bad generation name: {pointer}")
    require(Path(generation_name).name == generation_name, f"generation escapes directory: {pointer}")
    generation = pointer.parent / generation_name
    prediction = generation / logical.name
    receipt_path = generation / (logical.name[:-len(".jsonl")] + "-yarn_receipt.json")
    require(prediction.is_file() and receipt_path.is_file(), f"generation files missing: {generation}")
    return pointer, logical, generation_name, prediction, receipt_path


def check_statuses(status_root, original_sha, leftcap_sha):
    cells = []
    artifacts = []
    for name in STATUS_NAMES:
        path = status_root / name
        require(path.is_file(), f"status missing: {path}")
        status = json.loads(path.read_text(encoding="utf-8"))
        expected_sha = leftcap_sha if "leftcap" in name else original_sha
        require(status["state"] == "complete", f"driver not complete: {path}")
        require(status["run_config_sha256"] == expected_sha, f"driver config mismatch: {path}")
        require(status["jobs_complete"] == status["jobs_total"] == len(status["cells"]),
                f"driver population mismatch: {path}")
        artifacts.append({"path": str(path), "sha256": sha256(path)})
        log_dir = status_root / name.replace("_status_", "_driver_logs_").removesuffix(".json")
        for cell in status["cells"]:
            key = (cell["length"], cell["task"])
            require(cell["arm"] == "single_minmax" and cell["state"] == "complete",
                    f"bad driver cell: {path} {cell}")
            require(cell["action"] == ("skip" if key in SKIP_CELLS else "produced"),
                    f"unexpected driver action: {path} {cell}")
            cells.append(key)
            if cell["action"] == "produced":
                log = log_dir / Path(cell["log"]).name
                require(log.is_file(), f"producer log missing: {log}")
                artifacts.append({"path": str(log), "sha256": sha256(log)})
    expected = {(length, task) for length in LENGTHS for task in TASKS}
    require(len(cells) == 33 and set(cells) == expected, "driver cells are not the unique 33-cell matrix")
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--original-source-root", type=Path, required=True)
    parser.add_argument("--leftcap-source-root", type=Path, required=True)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--leftcap-config", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_sha = sha256(args.original_config)
    leftcap_sha = sha256(args.leftcap_config)
    configs = {
        original_sha: (json.loads(args.original_config.read_text(encoding="utf-8")), args.original_source_root),
        leftcap_sha: (json.loads(args.leftcap_config.read_text(encoding="utf-8")), args.leftcap_source_root),
    }
    require(original_sha != leftcap_sha, "leftcap config did not change identity")
    original_config = configs[original_sha][0]
    leftcap_config = configs[leftcap_sha][0]
    require(original_config["model"] == leftcap_config["model"], "model identities differ")
    require(original_config["inputs_sha256"] == leftcap_config["inputs_sha256"], "input identities differ")
    original_sources = original_config["source"]["files"]
    leftcap_sources = leftcap_config["source"]["files"]
    require(original_sources.keys() == leftcap_sources.keys(), "source file sets differ")
    changed_sources = [key for key in original_sources if original_sources[key] != leftcap_sources[key]]
    require(changed_sources == ["benchmark/RULER/pred_ruler.py"], f"unexpected leftcap diff: {changed_sources}")

    # 独立复核全部冻结资产；模型/输入两 config 相同，只读一次。
    check_assets(*configs[original_sha], args.model_root, args.data_root, check_model=True)
    check_assets(*configs[leftcap_sha], args.model_root, args.data_root, check_model=False)

    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_root), trust_remote_code=True)
    sys.path.insert(0, str(args.original_source_root))
    official_module = importlib.import_module("benchmark.RULER.score_ruler")
    official_sample_score = official_module.string_match_all

    synthetic = [
        ("Alpha beta", ["alpha", "BETA"]),
        ("alpha", ["alpha", "missing"]),
        ("", ["missing"]),
        ("anything", []),
    ]
    for pred, answers in synthetic:
        expected = independent_sample_score(pred, answers)
        official = official_sample_score([pred], [answers])
        require(expected == official, f"official metric synthetic mismatch: {pred!r} {answers!r}")

    status_artifacts = check_statuses(args.status_root, original_sha, leftcap_sha)
    per_sample = []
    cell_summaries = []
    prediction_artifacts = []
    truncations = []
    seen_ids = set()
    for length in LENGTHS:
        for task in TASKS:
            key = (length, task)
            expected_config_sha = leftcap_sha if key in LEFTCAP_CELLS else original_sha
            config, _ = configs[expected_config_sha]
            pointer, logical, generation_name, prediction, receipt_path = resolve_cell(
                args.prediction_root, length, task)
            prediction_sha = sha256(prediction)
            receipt_sha = sha256(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            raw_lines = prediction.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in raw_lines]
            source_path = args.data_root / str(length) / f"{task}.jsonl"
            source_rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()][:100]
            require(len(rows) == len(source_rows) == 100, f"population mismatch: {key}")
            require(receipt["status"] == "complete", f"receipt incomplete: {key}")
            require(receipt["run_config_sha256"] == expected_config_sha, f"cell config mismatch: {key}")
            require(receipt["prediction_basename"] == logical.name, f"prediction basename mismatch: {key}")
            require(receipt["prediction_sha256"] == prediction_sha, f"prediction SHA mismatch: {key}")
            require(receipt["prediction_lines"] == len(raw_lines) == 100, f"receipt lines mismatch: {key}")
            require(receipt["run_id"] == generation_name.removeprefix(logical.name + ".gen-"),
                    f"run id mismatch: {key}")
            require(receipt["source_files_sha256"] == config["source"]["files"], f"source closure mismatch: {key}")
            require(receipt["model_weight_files_sha256"] == config["model"]["weight_files_sha256"],
                    f"model closure mismatch: {key}")
            require(receipt["input_data"] == {
                "relative_path": f"{length}/{task}.jsonl",
                "sha256": config["inputs_sha256"][f"{length}/{task}.jsonl"],
            }, f"input closure mismatch: {key}")
            require(receipt["producer_script"]["sha256"] == config["source"]["files"]["benchmark/RULER/pred_ruler.py"],
                    f"producer script mismatch: {key}")
            require(receipt["context_length"] == length and receipt["task"] == task, f"cell identity mismatch: {key}")
            params = receipt["generation_params"]
            require(params == {
                "max_gen": BUDGET[task], "min_new_tokens": 1, "max_num": 100, "seed": 42,
                "method": "nfa", "pred_postfix": "_rsel", "t": "rsel20261111",
                "graph_policy": "auto", "operator_mode": "off", "eos_token_ids": EOS,
            }, f"generation params mismatch: {key}: {params}")
            expected_factor = {32768: None, 65536: 2.0, 131072: 4.0}[length]
            require(receipt["effective_yarn_factor"] == expected_factor, f"YaRN factor mismatch: {key}")
            require(receipt["yarn_enabled"] == (length != 32768), f"YaRN switch mismatch: {key}")
            require(receipt["native_max_position_embeddings"] == 40960, f"native MPE mismatch: {key}")
            rope = receipt["rope_policy"]
            if length == 32768:
                require(rope["rope_scaling"] is None and "requested_factor" not in rope
                        and rope["snapshot_preserved"] is True,
                        f"32K YaRN-off RoPE policy mismatch: {key}")
            else:
                require(rope["requested_factor"] == expected_factor
                        and rope["snapshot_preserved"] is False,
                        f"RoPE requested factor mismatch: {key}")
            require(rope["runtime_max_position_embeddings"] == max(length + BUDGET[task], 40960),
                    f"runtime capacity mismatch: {key}")
            verification = receipt["runtime_rope_verification"]
            require(verification["status"] == "verified"
                    and all(math.isfinite(verification[field])
                            and 0.0 <= verification[field] <= tolerance
                            for field, tolerance in (("max_inv_freq_error", 1e-7),
                                                     ("max_cos_error", 5e-4),
                                                     ("max_sin_error", 5e-4))),
                    f"runtime RoPE not verified: {key}: {verification}")

            treatment_envelope = receipt["treatment_manifest"]
            treatment = json.loads(treatment_envelope["json"])
            require(json.dumps(treatment, sort_keys=True) == treatment_envelope["json"],
                    f"treatment JSON not canonical: {key}")
            require(hashlib.sha256(treatment_envelope["json"].encode("utf-8")).hexdigest()
                    == treatment_envelope["sha256"], f"treatment SHA mismatch: {key}")
            expected_treatment = {
                "tia_block_size": 64, "tia_level1_topk": 128, "tia_level2_topk": 1024,
                "tia_level2_cmp_ratio": 4, "nfa_subspace": "tail", "nfa_l2_subspace": "same",
                "nfa_far_method": "minmax", "nfa_near_method": "avg", "nfa_near_pages": 0,
                "nfa_near_tokens": "off", "nfa_near_len": 0, "nfa_sink_tokens": 4,
                "nfa_swa_size": 128, "nfa_boundary_bypass": True, "nfa_l2_quant": False,
                "nfa_enable_layer_skip": False, "nfa_enable_static_gate": False, "nfa_dyn_dgate": False,
            }
            for field, expected in expected_treatment.items():
                require(treatment[field] == expected, f"treatment {field} mismatch: {key}")
            frozen = treatment["nfa_prefill_frozen_regions"]
            require(frozen["gate"] == {"epsilon": 0.05, "mode": "full", "rows": 256, "statistic": "mean"},
                    f"gate mismatch: {key}")
            require(frozen["geometry"] == {
                "max_new_tokens": BUDGET[task], "near": 64, "page": 64, "sink": 4, "window": 128,
            }, f"single-pool geometry mismatch: {key}")

            receipt_trunc = receipt.get("input_truncation")
            if key in LEFTCAP_CELLS:
                require(receipt_trunc is not None and receipt_trunc["policy"] == "left-to-nominal-context-v1",
                        f"leftcap receipt policy missing: {key}")
                require(receipt_trunc["nominal_context_length"] == length, f"leftcap nominal mismatch: {key}")
                require(receipt_trunc["count"] == len(receipt_trunc["samples"]), f"leftcap count mismatch: {key}")
            else:
                require(receipt_trunc is None, f"old cell unexpectedly has leftcap receipt: {key}")

            cell_scores = []
            cell_ids = []
            for index, (row, source) in enumerate(zip(rows, source_rows)):
                sample_id = f"{task}:{source['index']}"
                require(source["index"] == index, f"source sample index mismatch: {key}/{index}")
                require(row["_id"] == sample_id, f"prediction ID mismatch: {key}/{index}")
                require(row["answers"] == source["outputs"], f"answers mismatch: {sample_id}")
                require(row["_answers_sha"] == canonical_answers(source["outputs"]), f"answers SHA mismatch: {sample_id}")
                require(row["length"] == source["length"], f"length mismatch: {sample_id}")
                global_id = (length, sample_id)
                require(global_id not in seen_ids, f"duplicate global sample: {global_id}")
                seen_ids.add(global_id)
                cell_ids.append(sample_id)
                generated = row["generated_token_ids"]
                require(isinstance(generated, list) and 1 <= len(generated) <= BUDGET[task],
                        f"generation budget mismatch: {sample_id}")
                require(all(type(token) is int for token in generated), f"non-integer generated token: {sample_id}")
                if len(generated) < BUDGET[task]:
                    require(generated[-1] in EOS and all(token not in EOS for token in generated[:-1]),
                            f"early stop is not a unique final EOS: {sample_id}")
                require(tokenizer.decode(generated, skip_special_tokens=True) == row["pred"],
                        f"tokenizer decode mismatch: {sample_id}")
                require(row["execution"] == {"requested": "graph", "actual": "graph"},
                        f"not graph execution: {sample_id}")
                audit = row["_frozen_region"]
                require(audit["request_phase"] == "FINISHED", f"request not finished: {sample_id}")
                require(audit["gate_counts"] == {"keep": 36, "near": 0, "far": 0, "both": 0},
                        f"skip counts mismatch: {sample_id}")
                require(set(audit["decisions"]) == {str(i) for i in range(36)}, f"decision layers mismatch: {sample_id}")
                require(all(decision == {"skip_near": False, "skip_far": False}
                            for decision in audit["decisions"].values()), f"skip decision found: {sample_id}")
                require(set(audit["nfa_receipts"]) == {str(i) for i in range(36)}, f"NFA layers mismatch: {sample_id}")
                for layer_receipt in audit["nfa_receipts"].values():
                    require(layer_receipt["route"] == "graph_ready_sparse", f"route mismatch: {sample_id}")
                    visible = layer_receipt["visible_lens"]
                    require(visible and all(value == 1024 for step in visible for value in step),
                            f"visible length mismatch: {sample_id}")
                final_kv = audit["final_kv"]
                require(final_kv["n_layers"] == 36 and len(final_kv["layers"]) == 36,
                        f"final KV layers mismatch: {sample_id}")
                require(all(layer["num_kv_heads"] == 8 and layer["head_dim_k"] == 128 and layer["head_dim_v"] == 128
                            for layer in final_kv["layers"]), f"GQA KV shape mismatch: {sample_id}")

                record = row.get("_input_truncation")
                if record is not None:
                    truncations.append((key, record, receipt_trunc))
                score = independent_sample_score(row["pred"], row["answers"])
                official = official_sample_score([row["pred"]], [row["answers"]])
                require(math.isclose(score, official, rel_tol=0.0, abs_tol=1e-12),
                        f"official metric mismatch: {sample_id}: {score} != {official}")
                require(math.isfinite(score) and 0.0 <= score <= 100.0, f"invalid score: {sample_id}")
                cell_scores.append(score)
                per_sample.append({
                    "length": length, "task": task, "sample_id": sample_id,
                    "score": score, "prediction_sha256": prediction_sha,
                })
            require(len(set(cell_ids)) == 100, f"duplicate task IDs: {key}")
            cell_score = sum(cell_scores) / len(cell_scores)
            require(math.isfinite(cell_score) and 0.0 <= cell_score <= 100.0, f"invalid cell score: {key}")
            cell_summaries.append({
                "length": length, "task": task, "n": 100, "score": cell_score,
                "run_config_sha256": expected_config_sha, "prediction_sha256": prediction_sha,
                "receipt_sha256": receipt_sha,
            })
            prediction_artifacts.append({
                "length": length, "task": task, "pointer": str(pointer),
                "prediction": str(prediction), "prediction_sha256": prediction_sha,
                "receipt": str(receipt_path), "receipt_sha256": receipt_sha,
                "run_config_sha256": expected_config_sha,
            })

    require(len(per_sample) == 3300 and len(cell_summaries) == 33, "final population is not 3300/33")
    require(len(truncations) == 1 and truncations[0][0] == (131072, "fwe"),
            f"unexpected truncation population: {[(key, rec.get('sample_id')) for key, rec, _ in truncations]}")
    key, truncation, receipt_trunc = truncations[0]
    require(truncation["sample_id"] == "fwe:69", f"wrong truncated sample: {truncation}")
    require(receipt_trunc["count"] == 1 and receipt_trunc["samples"] == [truncation],
            "fwe row/receipt truncation mismatch")
    source_fwe69 = [json.loads(line) for line in
                    (args.data_root / "131072" / "fwe.jsonl").read_text(encoding="utf-8").splitlines()][69]
    token_ids = tokenizer(source_fwe69["input"], truncation=False, return_tensors="pt").input_ids
    retained = token_ids[0, -131072:].tolist()
    expected_truncation = {
        "policy": "left-to-nominal-context-v1", "sample_id": "fwe:69",
        "original_input_sha256": hashlib.sha256(source_fwe69["input"].encode("utf-8")).hexdigest(),
        "retained_token_ids_sha256": hashlib.sha256(json.dumps(
            retained, separators=(",", ":")).encode("ascii")).hexdigest(),
        "original_tokens": token_ids.shape[-1], "retained_tokens": 131072,
        "dropped_left_tokens": token_ids.shape[-1] - 131072,
    }
    require(truncation == expected_truncation, f"leftcap record does not reproduce: {truncation} != {expected_truncation}")
    for length in (65536, 32768):
        receipt = next(json.loads(Path(item["receipt"]).read_text(encoding="utf-8"))
                       for item in prediction_artifacts
                       if item["length"] == length and item["task"] == "niah_multivalue")
        require(receipt["input_truncation"]["count"] == 0 and receipt["input_truncation"]["samples"] == [],
                f"unexpected truncation in {length}/niah_multivalue")

    per_sample_path = args.output_dir / "per_sample_scores.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for record in per_sample:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    scores = {str(length): {} for length in LENGTHS}
    for cell in cell_summaries:
        scores[str(cell["length"])][cell["task"]] = cell["score"]
    averages = {length: sum(scores[str(length)].values()) / len(TASKS) for length in LENGTHS}
    for length in LENGTHS:
        assert len(scores[str(length)]) == 11

    plan_path = args.output_dir / "validation_plan.json"
    script_path = args.output_dir / "independent_verify.py"
    wrapper = args.output_dir / "pred_ruler_region_selector.py"
    original_producer = args.output_dir / "pred_ruler_original.py"
    leftcap_producer = args.output_dir / "pred_ruler_leftcap.py"
    require(sha256(wrapper) == sha256(args.leftcap_source_root / "benchmark/RULER/pred_ruler_region_selector.py"),
            "archived wrapper differs from production source")
    require(sha256(original_producer) == original_sources["benchmark/RULER/pred_ruler.py"],
            "archived original producer hash mismatch")
    require(sha256(leftcap_producer) == leftcap_sources["benchmark/RULER/pred_ruler.py"],
            "archived leftcap producer hash mismatch")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "transformers": transformers.__version__, "tokenizer_class": type(tokenizer).__name__,
        "official_metric_source": str(Path(official_module.__file__).resolve()),
        "official_metric_sha256": sha256(official_module.__file__),
    }
    raw_artifacts = list(status_artifacts)
    for item in prediction_artifacts:
        raw_artifacts.extend([
            {"path": item["prediction"], "sha256": item["prediction_sha256"]},
            {"path": item["receipt"], "sha256": item["receipt_sha256"]},
        ])
    manifest = {
        "schema_version": "experiment-result/v1",
        "result_id": "ruler-region-selector-minmax-20261111",
        "run_id": "rsel20261111-independent-readonly-v1",
        "code_revision": "84d6110c929a09cda0f3e3fd659c9cb8a6c62f1e + left-to-nominal-context-v1",
        "scope": json.loads(plan_path.read_text(encoding="utf-8"))["scope"],
        "execution": {"seeds": [42], "repeats": 1,
                      "environment_description": environment,
                      "command": ["Single-pool MinMax 33-cell generation drivers; independent readonly scoring"]},
        "run_configs": {
            "original": {"path": str(args.original_config), "sha256": original_sha, "cells": 30},
            "leftcap": {"path": str(args.leftcap_config), "sha256": leftcap_sha, "cells": 3},
        },
        "artifacts": {
            "code": [
                {"path": str(script_path), "sha256": sha256(script_path)},
                {"path": str(args.driver), "sha256": sha256(args.driver)},
                {"path": str(wrapper), "sha256": sha256(wrapper)},
                {"path": str(original_producer), "sha256": sha256(original_producer)},
                {"path": str(leftcap_producer), "sha256": sha256(leftcap_producer)},
            ],
            "inputs": [{"path": str(args.data_root / relative), "sha256": digest}
                       for relative, digest in sorted(original_config["inputs_sha256"].items())],
            "config": [
                {"path": str(args.original_config), "sha256": original_sha},
                {"path": str(args.leftcap_config), "sha256": leftcap_sha},
                {"path": str(plan_path), "sha256": sha256(plan_path)},
            ],
            "raw_data": raw_artifacts,
            "outputs": [{"path": str(per_sample_path), "sha256": sha256(per_sample_path)}],
            "environment": [{"path": str(Path(official_module.__file__).resolve()),
                             "sha256": sha256(official_module.__file__)}],
        },
        "metrics": [{"name": f"RULER_AVG11_L{length}", "value": averages[length], "unit": "percent"}
                    for length in LENGTHS],
        "leftcap_source_diff": changed_sources,
        "leftcap_truncation": truncation,
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    atomic_json(manifest_path, manifest)

    checks = {name: {"verdict": "pass", "reason": detail,
                     "evidence": [str(manifest_path), str(per_sample_path)]}
              for name, detail in {
                  "implementation": "33 格 GraphReady、36 层全 keep、MinMax 单池、K1/K2、预算/EOS/YaRN 和双 config 身份全部闭合。",
                  "reference_boundary": "3300 条及 4 组合成边界的独立逐样本分数与官方 string_match_all 在零相对容差、1e-12 绝对容差内一致。",
                  "data_integrity": "33 个源输入、3300 个唯一样本、答案/length/token/decode、prediction/receipt/status/log SHA 全部闭合。",
                  "numerical_sanity": "全部逐样本/逐格分数有限且在 [0,100]，三长度宏平均由 3300 条原始分数独立重建。",
                  "measurement_validity": "每格前 100 条和任务级预算闭合，AVG 为 11 任务等权；无性能主张。",
                  "reproducibility": "两套 1195 源文件及模型/输入 SHA 全闭合；真实 fwe:69 截断可重建，旧 30 格保持原 config。",
              }.items()}
    limitations = json.loads(plan_path.read_text(encoding="utf-8"))["limitations"]
    record = {
        "schema_version": "experiment-validation/v1",
        "result_id": "ruler-region-selector-minmax-20261111",
        "status": "usable-with-scope", "manifest_sha256": sha256(manifest_path),
        "validation_plan_sha256": sha256(plan_path),
        "artifact_hashes": {
            "artifact_manifest": sha256(manifest_path),
            "validation_plan": sha256(plan_path),
            "independent_verify": sha256(script_path),
            "per_sample_scores": sha256(per_sample_path),
        },
        "verifier": {"actor": "independent-readonly-process", "independent": True,
                     "source": str(script_path), "run_id": "rsel20261111-independent-readonly-v1",
                     "method": "different implementation; official metric cross-check only"},
        "checks": checks, "limitations": limitations, "environment": environment,
        "population": {"cells": 33, "samples": 3300, "per_cell": 100},
        "truncated_samples": [truncation],
    }
    record_path = args.output_dir / "validation_record.json"
    atomic_json(record_path, record)
    summary = {
        "schema_version": "ruler-minmax-independent-summary/v1",
        "result_id": "ruler-region-selector-minmax-20261111",
        "status": "usable-with-scope", "arm": "single_minmax", "model": "Qwen3-8B",
        "population": {"cells": 33, "samples": 3300, "tasks_per_length": 11, "samples_per_cell": 100},
        "scores": scores, "avg11_percent": {str(length): averages[length] for length in LENGTHS},
        "cells": cell_summaries, "run_configs": manifest["run_configs"],
        "input_truncation": truncation,
        "validation_record": {"path": str(record_path), "sha256": sha256(record_path)},
        "artifact_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "per_sample_scores": {"path": str(per_sample_path), "sha256": sha256(per_sample_path)},
        "limitations": limitations,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "cells": 33, "samples": 3300,
                      "avg11_percent": summary["avg11_percent"],
                      "summary": str(args.output_dir / "summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
