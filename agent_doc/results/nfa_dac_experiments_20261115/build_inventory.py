"""从真实归档生成 NFA DAC inventory；不生成或修改实验预测。"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
B_ROOT = REPO / "agent_doc/results/nfa_ablation_20261030/B_final_review"
PERF = REPO / "agent_doc/results/qwen3_official_ruler_multikey2_perf_20261115/summary.json"
QWEN_LB = REPO / "agent_doc/results/qwen3_nfa_fi_longbench16_20261115"
GLM_AUDITS = Path("/home/wangyuanshuo02/ak_verify_results/glm_subspace_ablation_20261106/full_audits")
FIELDS = [
    "record_id", "target_group", "model", "dataset_or_input", "context_length",
    "batch_size", "arm", "sample_scope", "configuration", "generation_protocol",
    "timing_protocol", "input_identity", "result", "result_location",
    "evidence_status", "reuse_decision", "missing_work",
]
rows = []


def add(record_id, group, model, dataset, arm, **kw):
    row = {key: "" for key in FIELDS}
    row.update(record_id=record_id, target_group=group, model=model,
               dataset_or_input=dataset, arm=arm)
    row.update(kw)
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown columns: {unknown}")
    rows.append(row)


b_manifest = json.loads((B_ROOT / "data_package/manifest.json").read_text())
b_stats = json.loads((B_ROOT / "paired_statistics.json").read_text())
b_common = "page64;K1=128;K2=1024;near4096;near_pages32;sink4;SWA128;dc32 tail;gate full/off;epsilon0.05;reclaim true"
for cell in b_stats["cells"]:
    spec = b_manifest["cell_specs"][cell["cell"]]
    dataset = cell["dataset"]
    config = (b_common + f";near={spec['near_score']};far={spec['far_score']}"
              f";partition={spec['partition']};boundary={spec['boundary']}")
    add(f"old_B_{dataset}_{cell['cell']}", "new_C_historical_two_task", "Qwen3-8B",
        dataset, cell["cell"], sample_scope="source rows 0..199; n=200",
        configuration=config,
        generation_protocol="full-prompt dense prefill; greedy EOS; HotpotQA max-new32/GovReport max-new512",
        input_identity="per-row input/raw-request SHA256 in data_package/plot_data.csv",
        result=json.dumps({"mean_percent": cell["mean_percent"],
                           "delta_pp": cell["delta_pp"], "ci95_pp": cell["ci95_pp"]}),
        result_location="agent_doc/results/nfa_ablation_20261030/B_final_review/",
        evidence_status="usable-with-scope; independent official rescore 2000/2000; old producer 4dab6e8c",
        reuse_decision="reuse only as explicitly labelled historical epsilon0.05/reclaim group",
        missing_work="do not rerun HotpotQA/GovReport; MK2 requires new matched-protocol evidence")

for arm in ["full128", "tail32", "highfreq32", "random32"]:
    for dataset in ["hotpotqa", "gov_report"]:
        add(f"old_A_{dataset}_{arm}", "new_B_historical_coordinate", "Qwen3-8B",
            dataset, arm, sample_scope="historical report: 200 rows/arm; raw IDs unavailable",
            configuration="old A coordinate ablation; exact per-arm runtime manifest unavailable",
            generation_protocol="historical report only",
            result_location="agent_doc/task/task_details/TL-NFA-ABLATION-20261030.md",
            evidence_status="historical scoped-acceptance reported; raw predictions/audits missing; 113/125 unreachable",
            reuse_decision="reference_only_not_consumable",
            missing_work="recover original predictions and independent audit; otherwise rerun only required missing configurations")
for dataset in ["hotpotqa", "gov_report"]:
    for near in [2048, 4096, 8192]:
        add(f"old_E_{dataset}_near{near}", "new_F_historical_two_task", "Qwen3-8B",
            dataset, f"near{near}", sample_scope="historical report: source rows 0..199; n=200",
            configuration="near-length ablation; gate off; exact complete token/environment identity unavailable",
            generation_protocol="historical report only",
            result_location="agent_doc/task/task_details/TL-NFA-ABLATION-20261030.md",
            evidence_status="historical independent rescore/CI reported; raw data package missing",
            reuse_decision="reference_only_not_consumable",
            missing_work="recover raw predictions, scores, manifests and hashes; do not claim current matched group")
for threshold in ["0.01", "0.025", "0.05", "0.10"]:
    add(f"old_C_tau{threshold}", "new_E_historical_gate", "Qwen3-8B",
        "hotpotqa+gov_report", f"tau{threshold}", sample_scope="historical mixed reuse/generation; exact rows unavailable",
        configuration="old gate/cache bridge; not observation256/tau0.10 new group",
        result_location="agent_doc/task/task_details/TL-NFA-ABLATION-20261030.md",
        evidence_status="raw final manifests/scores missing; low thresholds had input-identity STOP history",
        reuse_decision="reference_only_not_consumable",
        missing_work="recover final raw evidence before any reuse decision")
add("observation_oracle", "mechanism_only", "Qwen3-8B", "fixed real Q/K CPU oracle",
    "nearAvg_farMinMax", context_length="mixed", sample_scope="40 Q/K;1280 query cells;10240 KV-group cells",
    configuration="sink4;SWA128;page64;K1=128;K2=1024;near4096;tail32",
    result_location="agent_doc/results/nfa_obs_ablation_evidence_20261106/observation/current_main_complement/RESULT.md",
    evidence_status="usable-with-scope; CPU dense-oracle/L1 observation only",
    reuse_decision="mechanism evidence only; never substitute end-to-end quality",
    missing_work="none for its stated scope")

for arm, values in {
    "native_full": {"hotpotqa": 62.4985, "gov_report": 33.4201},
    "nope_random32_s0": {"hotpotqa": 61.6348, "gov_report": 32.8492},
    "nope_random32_s1": {"hotpotqa": 62.4117, "gov_report": 32.8805},
    "nope_uniform32": {"hotpotqa": 62.6530, "gov_report": 32.2180},
}.items():
    for dataset, score in values.items():
        add(f"glm_{dataset}_{arm}", "GLM_NoPE", "GLM-4.7-Flash", dataset, arm,
            sample_scope="n=200; complete audit", configuration="gate off/full; selector subspace only",
            generation_protocol="LongBench formal generation; see source audit",
            result=json.dumps({"score_percent": score}), result_location=str(GLM_AUDITS),
            evidence_status="complete 200-row PASS audit present",
            reuse_decision="reuse after verifying exact coordinates and run identity",
            missing_work="uniform32 must not be relabelled first32/last32")
for arm in ["nope_first32", "nope_last32", "nope_all192"]:
    for dataset in ["hotpotqa", "gov_report"]:
        add(f"glm_{dataset}_{arm}", "GLM_NoPE", "GLM-4.7-Flash", dataset, arm,
            sample_scope="target n=200", configuration="gate off/full; selector subspace only",
            result_location="historical README reference only" if arm == "nope_last32" else "",
            evidence_status="raw full result missing" if arm != "nope_first32" else "full 200-row result not found",
            reuse_decision="no reusable raw result",
            missing_work="low priority: recover or run full 200 rows after Qwen3-8B main matrix")

old_perf = json.loads(PERF.read_text())
for cell in old_perf["cells"]:
    method = cell["method"]
    length = cell["context_length"]
    batch = cell["batch_size"]
    identity = cell.get("input_ids_sha256", "")
    result = {key: cell[key] for key in ["status", "median_tpot_ms", "repeat_tpot_ms", "reason"] if key in cell}
    add(f"old_perf_{method}_{length}_b{batch}", "old_performance_host_wall", "Qwen3-8B",
        "RULER-derived niah_multikey_2 sample2", method, context_length=length, batch_size=batch,
        sample_scope="one fixed input; warmup1; repeats3",
        configuration=old_perf["implementation_labels"][method],
        generation_protocol="ignore EOS;33 decode tokens",
        timing_protocol=old_perf["protocol"]["timing"], input_identity=identity,
        result=json.dumps(result), result_location=str(PERF.relative_to(REPO)),
        evidence_status="pending-independent-verification; old measurement group closed",
        reuse_decision="retain as old host-wall group only; never mix with new CUDA-event curves",
        missing_work="new protocol required" if cell["status"] == "complete" else "none; preserve blocked/unsupported/OOM status")

quality_data = [("hotpotqa", "", 200), ("gov_report", "", 200),
                ("niah_multikey_2", 32768, 100), ("niah_multikey_2", 65536, 100),
                ("niah_multikey_2", 131072, 100)]
qwen_lb = json.loads((QWEN_LB / "quality_summary.json").read_text())
qwen_lb_audit = json.loads((QWEN_LB / "rank_audit.json").read_text())
if (qwen_lb["status"] != "PASS" or qwen_lb_audit["status"] != "PASS"
        or qwen_lb_audit["cross_arm_input_identity"] != "PASS"
        or qwen_lb["verification"] != "pending-independent-verification"):
    raise ValueError("Qwen3 LongBench reuse source is not a pending production-side closed set")
a_arms = {
    "A1_full_gate_off": "full-dim index;gate off;epsilon0",
    "A2_tail32_gate_off": "tail32;gate off;epsilon0",
    "A3_full_gate_on": "full-dim index;gate on;obs256;tau0.10",
    "A4_tail32_gate_on": "tail32;gate on;obs256;tau0.10",
    "FullKV": "dense FullKV reference",
}
for arm, config in a_arms.items():
    for dataset, length, count in quality_data:
        record_id = f"new_A_{arm}_{dataset}_{length or 'lb'}"
        common = dict(
            context_length=length, sample_scope=f"n={count}; canonical sample IDs required",
            configuration=config + ";page64;K1=128;near32/far96;near_tokens=off;K2=1024 incl protected;sink4;SWA128;near4096;boundary true",
            generation_protocol="dense prefill;greedy;formal EOS;task-specific max-new",
            evidence_status="missing", reuse_decision="no matching complete result",
            missing_work="run and independently audit")
        if arm == "FullKV":
            common["configuration"] = "dense FullKV reference;page64 for FlashInfer decode"
        reusable_method = {"FullKV": "flashinfer"}.get(arm)
        if dataset in {"hotpotqa", "gov_report"} and reusable_method:
            source = qwen_lb["arms"][reusable_method]
            pred = QWEN_LB / source["canonical_predictions"][dataset]["path"]
            runcap = json.loads(Path(str(pred) + ".runcap.json").read_text())
            if (runcap["status"] != "complete" or runcap["n_samples"] != 200
                    or runcap["result_sha256"] != source["canonical_predictions"][dataset]["sha256"]):
                raise ValueError(f"invalid reusable LongBench runcap: {runcap}")
            producer = source["source_head"]
            common.update(
                sample_scope="all 200 source rows; canonical source order; production-side 3750-row arm closed",
                generation_protocol="dense prefill;eager;greedy;real EOS/stop-string/max-new;HotpotQA32/GovReport512",
                input_identity=runcap["input_identity"]["actual_input_ids"]["ordered_sha256"],
                result=json.dumps({"score_percent": source["scores"][dataset],
                                   "prediction_sha256": runcap["result_sha256"],
                                   "producer": producer}),
                result_location=str(QWEN_LB.relative_to(REPO)),
                evidence_status="complete canonical 200/200; production-side rank/input audit PASS; pending-independent-verification",
                reuse_decision=f"reuse as {arm}; do not regenerate matched HotpotQA/GovReport rows",
                missing_work="independent owner verification; MK2 cells remain missing")
            common["configuration"] += ";SDPA prefill+FlashInfer page64 eager decode;producer 7fac274f"
        elif arm == "A2_tail32_gate_off" and dataset in {"hotpotqa", "gov_report"}:
            source = qwen_lb["arms"]["nfa"]
            common.update(
                result=json.dumps({"historical_score_percent": source["scores"][dataset],
                                   "producer": source["source_head"]}),
                result_location=str(QWEN_LB.relative_to(REPO)),
                evidence_status="complete historical 200/200 but protocol mismatch: prefill dynamic GateConfig epsilon0.05 could skip regions",
                reuse_decision="reference_only_not_consumable_as_gate_off; layer-skip=false only disabled removed static D-prime path",
                missing_work="rerun with explicit prefill epsilon0 and verify all 36 decisions keep")
        add(record_id, "new_A_2x2", "Qwen3-8B", dataset, arm, **common)
for arm, config, source in [
    ("B_low32", "paired low-frequency16 RoPE pairs=32 scalars;gate off", "A2_tail32_gate_off"),
    ("B_high32", "paired high-frequency16 RoPE pairs=32 scalars;gate off", ""),
    ("B_random32_s0", "paired random32;seed0;gate off", ""),
    ("B_random32_s1", "paired random32;seed1;gate off", ""),
    ("B_random32_s2", "paired random32;seed2;gate off", ""),
    ("B_full_reference", "full-dim index;gate off", "A1_full_gate_off"),
]:
    for dataset, length, count in quality_data:
        add(f"new_B_{arm}_{dataset}_{length or 'lb'}", "new_B_coordinates", "Qwen3-8B", dataset, arm,
            context_length=length, sample_scope=f"n={count}; same IDs as new A",
            configuration=config + ";near_tokens=off", generation_protocol="same as new A", evidence_status="missing",
            reuse_decision=f"cross-reference {source}; no duplicate generation" if source else "no matching complete result",
            missing_work="verify cross-reference after new A" if source else "run and independently audit")
for arm, config in [
    ("C2_nearMinMax_farMinMax", "near MinMax;far MinMax;partitioned;boundary true"),
    ("C3_nearAvg_farAvg", "near Avg;far Avg;partitioned;boundary true"),
    ("C4_globalMinMax", "global MinMax single candidate pool;partition false"),
    ("C5_no_boundary", "default regional scores;boundary admission false"),
]:
    for length in [32768, 65536, 131072]:
        add(f"new_C_{arm}_mk2_{length}", "new_C_regional", "Qwen3-8B", "niah_multikey_2", arm,
            context_length=length, sample_scope="n=100;same IDs as new A",
            configuration="tail32;gate off;near_tokens=off;" + config, generation_protocol="same as new A",
            evidence_status="missing", reuse_decision="old B covers only HotpotQA/GovReport under historical protocol",
            missing_work="run and independently audit")
for dc in [16, 32, 64, 128]:
    required = [("hotpotqa", "", 200), ("gov_report", "", 200), ("niah_multikey_2", 65536, 100)]
    if dc != 128:
        required += [("niah_multikey_2", 32768, 100), ("niah_multikey_2", 131072, 100)]
    else:
        required += [("niah_multikey_2", 32768, 100), ("niah_multikey_2", 131072, 100)]
    for dataset, length, count in required:
        source = "A2_tail32_gate_off" if dc == 32 else ("A1_full_gate_off" if dc == 128 else "")
        add(f"new_D_dc{dc}_{dataset}_{length or 'lb'}", "new_D_subspace_width", "Qwen3-8B", dataset, f"dc{dc}",
            context_length=length, sample_scope=f"n={count};same IDs as new A",
            configuration=f"gate off;low-frequency dc{dc};obs256;near_tokens=off", generation_protocol="same as new A",
            evidence_status="missing", reuse_decision=f"cross-reference {source} only after runtime equivalence check" if source else "no matching complete result",
            missing_work="verify equivalence; no duplicate run" if source else "run quality plus 64K/128K L1/L2/decode performance as applicable")
for tau in [0.01, 0.025, 0.05, 0.10, 0.20, "off"]:
    required = [("hotpotqa", "", 200), ("gov_report", "", 200), ("niah_multikey_2", 65536, 100)]
    if tau in [0.05, 0.10, 0.20, "off"]:
        required += [("niah_multikey_2", 32768, 100), ("niah_multikey_2", 131072, 100)]
    for dataset, length, count in required:
        source = "A4_tail32_gate_on" if tau == 0.10 else ("A2_tail32_gate_off" if tau == "off" else "")
        add(f"new_E_tau{tau}_{dataset}_{length or 'lb'}", "new_E_gate_threshold", "Qwen3-8B", dataset, f"tau{tau}",
            context_length=length, sample_scope=f"n={count};same IDs as new A",
            configuration=f"tail32;obs256;tau_near=tau_far={tau};near_tokens=off", generation_protocol="same as new A",
            evidence_status="missing", reuse_decision=f"cross-reference {source}" if source else "no matching complete result",
            missing_work="verify cross-reference" if source else "run and independently audit")
for near in [2048, 4096, 8192]:
    for length in [32768, 65536, 131072]:
        source = "A2_tail32_gate_off" if near == 4096 else ""
        add(f"new_F_near{near}_mk2_{length}", "new_F_near_length", "Qwen3-8B", "niah_multikey_2", f"near{near}",
            context_length=length, sample_scope="n=100;same IDs as new A",
            configuration=f"tail32;gate off;near_length{near};near_pages32;page64;near_tokens=off",
            generation_protocol="same as new A", evidence_status="missing",
            reuse_decision=f"cross-reference {source}" if source else "old E has no MK2 and lacks raw package",
            missing_work="verify cross-reference" if source else "run and independently audit")

for method in ["flashinfer", "nfa_gate_on", "nfa_gate_off", "quest", "jssa", "asynctls"]:
    for length in [32768, 65536, 131072, 262144, 524288]:
        for batch in [1, 2, 4]:
            unsupported = method in {"quest", "jssa"} and batch > 1
            blocked = method == "jssa" and batch == 1
            status = "unsupported-official-path" if unsupported else ("blocked-cu129" if blocked else "missing")
            work = "none; fail-closed" if unsupported or blocked else "measure if memory permits" if length == 524288 else "run new CUDA-event protocol"
            add(f"new_perf_{method}_{length}_b{batch}", "new_performance_cuda_event", "Qwen3-8B",
                "RULER-derived MultiKey-2 fixed performance inputs", method,
                context_length=length, batch_size=batch,
                sample_scope="3 fixed batch inputs;warmup2 full requests;5 measured requests;rotated method order",
                configuration="NFA K1=128,K2=1024,near32/far96,near_tokens=off;Quest/AsyncTLS comparable budgets;BF16;H20-3e",
                generation_protocol="dense prefill;3 untimed decode;32 timed decode;ignore EOS;reset request each repeat",
                timing_protocol="CUDA events around full 32-step interval;TPOT=interval/32;separate host-wall;B>1 not divided by B",
                input_identity="record actual tokens,construction,seed,input_ids SHA256,RoPE/YaRN per input",
                evidence_status=status, reuse_decision="old host-wall cells are not substitutes", missing_work=work)

out = ROOT / "inventory.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "schema": "nfa-dac-inventory-summary-v1",
    "verification": "pending-independent-verification",
    "rows": len(rows),
    "by_group": {},
    "notes": [
        "Historical summaries without raw artifacts are not consumable data.",
        "Old host-wall and new CUDA-event performance are separate measurement groups.",
        "Cross-references prohibit duplicate generation and require runtime equivalence checks.",
        "Qwen3 LongBench NFA canonical rows used dynamic prefill epsilon0.05 despite static layer-skip=false; they are historical references, not new A2 gate-off/epsilon0.",
        "nfa_near_tokens=off preserves L1 near/far page pools but allows free L2 token competition.",
        "FlashInfer HotpotQA/GovReport rows are same-input FullKV references with page64 eager decode.",
        "All reused Qwen3 LongBench rows remain pending independent verification by a different owner.",
        "Old 113/125 were unreachable with No route to host on 2026-11-15.",
    ],
}
for row in rows:
    summary["by_group"][row["target_group"]] = summary["by_group"].get(row["target_group"], 0) + 1
new_a_rows = [row for row in rows if row["target_group"] == "new_A_2x2"]
reused_a_rows = [row for row in new_a_rows if row["reuse_decision"].startswith("reuse as")]
request_counts = {(dataset, length): count for dataset, length, count in quality_data}
reused_requests = sum(
    request_counts[(row["dataset_or_input"], row["context_length"] or "")]
    for row in reused_a_rows)
summary.update(
    unique_ids=len({row["record_id"] for row in rows}),
    new_quality_rows=sum(row["target_group"] in {
        "new_A_2x2", "new_B_coordinates", "new_C_regional",
        "new_D_subspace_width", "new_E_gate_threshold", "new_F_near_length"}
        for row in rows),
    new_A_reuse_rows=len(reused_a_rows),
    new_A_missing_rows=len(new_a_rows) - len(reused_a_rows),
    reused_requests=reused_requests,
    remaining_new_A_requests=3500 - reused_requests,
)
assert summary["unique_ids"] == len(rows)
assert [row["record_id"] for row in reused_a_rows] == [
    "new_A_FullKV_hotpotqa_lb", "new_A_FullKV_gov_report_lb"]
(ROOT / "inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
