import json, os, sys
sys.path.insert(0, os.getcwd())
from benchmark.LongBench.eval import sample_score

OUT = "exp/results_longbench_vendor/Qwen3-32B/pred_asynctls"
SUFFIX = "asynctls_vendor_qwen3-32b_qasper_channels_k1024_b64_kb128_h639e15473c-11092222.jsonl"
TASKS = ["narrativeqa","qasper","multifieldqa_en","hotpotqa","2wikimqa","musique",
         "gov_report","qmsum","multi_news","trec","triviaqa","samsum",
         "passage_count","passage_retrieval_en","lcc","repobench"]

recomputed = {}
for fname in sorted(os.listdir(OUT)):
    if not fname.endswith(SUFFIX):
        continue
    task = fname.split("-")[0]
    preds = []
    with open(os.path.join(OUT, fname), "rb") as f:
        for line in f.read().decode("utf-8").splitlines():
            d = json.loads(line)
            preds.append(sample_score(task, d["pred"], d["answers"], d["all_classes"]))
    recomputed[task] = round(100.0 * sum(preds) / len(preds), 2)

official = json.load(open(os.path.join(OUT, "result.json")))
official_scores = {f.split("-")[0]: v["score"] for f, v in official.items()
                   if f.endswith(".jsonl")}
macro = round(sum(recomputed[t] for t in TASKS) / len(TASKS), 2)
mismatch = [t for t in TASKS if abs(recomputed[t] - official_scores.get(t, -999)) > 0.005]
verdict = {
    "tasks": len(TASKS),
    "macro_avg_recomputed": macro,
    "recomputed_matches_official": len(mismatch) == 0,
    "mismatches": mismatch,
    "per_task": recomputed,
}
print(json.dumps(verdict, indent=2, ensure_ascii=False))
