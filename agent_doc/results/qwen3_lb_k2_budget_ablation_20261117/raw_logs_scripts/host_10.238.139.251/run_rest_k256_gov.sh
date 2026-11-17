#!/usr/bin/env bash
set -u
REPO=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/two-level-attention
ROOT=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/budget_ablation_v1
PY=/home/fluentllmenv/bin/python
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$REPO"
out="$ROOT/gov_report_k256_rest"
log="$ROOT/logs/gov_report_k256_rest.log"
mkdir -p "$out"
echo "===== $(date) start gov_report k256 REST gpu1 -> ${out} ===="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$PY" -u -m benchmark.efficiency.benchmark_longbench_quality \
  --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off \
  --level2-topk 256 --model-path /home/wangyuanshuo02/Qwen3-8B --model-name Qwen3-8B \
  --data-root "$DATA" --config-path "$REPO/benchmark/LongBench/config" \
  --output-dir "$out" --tasks gov_report --device cuda:0 --visible-len 8192 \
  --sample-indices $(seq 23 199) > "$log" 2>&1
rc=$?
echo "===== $(date) gov_report k256 REST exit=${rc} ===="
