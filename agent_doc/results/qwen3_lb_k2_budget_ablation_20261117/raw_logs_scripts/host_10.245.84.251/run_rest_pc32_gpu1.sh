#!/usr/bin/env bash
set -u
REPO=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/two-level-attention
ROOT=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/budget_ablation_v1
PY=/home/fluentllmenv/bin/python
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
run_rest () {
  task="$1"; budget="$2"; start="$3"; out="$4"; log="$5"
  echo "===== $(date) start ${task} k${budget} REST gpu1 -> ${out} ===="
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$PY" -u -m benchmark.efficiency.benchmark_longbench_quality \
    --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off \
    --level2-topk "$budget" \
    --model-path /home/wangyuanshuo02/Qwen3-8B --model-name Qwen3-8B \
    --data-root "$DATA" --config-path "$REPO/benchmark/LongBench/config" \
    --output-dir "$out" --tasks "$task" --device cuda:0 --visible-len 8192 \
    --sample-indices $(seq "$start" 199) > "$log" 2>&1
  rc=$?
  echo "===== $(date) ${task} k${budget} REST exit=${rc} ===="
}
cd "$REPO"
run_rest hotpotqa 2048 45 "$ROOT/hotpotqa_k2048_rest" "$ROOT/logs/hotpotqa_k2048_rest.log"
echo "===== $(date) start gov_report k2048 FULL gpu1 ===="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$PY" -u -m benchmark.efficiency.benchmark_longbench_quality \
  --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off \
  --level2-topk 2048 --model-path /home/wangyuanshuo02/Qwen3-8B --model-name Qwen3-8B \
  --data-root "$DATA" --config-path "$REPO/benchmark/LongBench/config" \
  --output-dir "$ROOT/gov_report_k2048_full" --tasks gov_report --device cuda:0 --visible-len 8192 \
  > "$ROOT/logs/gov_report_k2048_full.log" 2>&1
rc=$?
echo "===== $(date) gov_report k2048 FULL exit=${rc} ===="
