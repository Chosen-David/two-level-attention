#!/usr/bin/env bash
set -u
REPO=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/two-level-attention
ROOT=/home/wangyuanshuo02/qwen3_lb_subspace_20261116/budget_ablation_v1
PY=/home/fluentllmenv/bin/python
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$REPO"
out="$ROOT/hotpotqa_k512"
log="$ROOT/logs/hotpotqa_k512.log"
mkdir -p "$out"
echo "===== $(date) start hotpotqa k512 gpu0 -> ${out} ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$PY" -u -m benchmark.efficiency.benchmark_longbench_quality \
  --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 --nfa-gate off \
  --level2-topk 512 --model-path /home/wangyuanshuo02/Qwen3-8B --model-name Qwen3-8B \
  --data-root "$DATA" --config-path "$REPO/benchmark/LongBench/config" \
  --output-dir "$out" --tasks hotpotqa --device cuda:0 --visible-len 8192 > "$log" 2>&1
rc=$?
echo "===== $(date) hotpotqa k512 exit=${rc} ====="
