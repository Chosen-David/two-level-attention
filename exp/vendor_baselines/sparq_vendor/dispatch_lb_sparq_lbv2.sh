#!/bin/bash
# SparQ vendor LongBench-v2 503 题全量派单（GPU 释放后执行）
# 用法：bash dispatch_lb_sparq_lbv2.sh <gpu>
# 与 quest_lbv2_1024_16 同构：--task lbv2 --dataset-path 指向 LongBench-v2 data.json
# 主表口径：k=1024 固定，r 按协议 cap 31500 反推（=8，与 LBv1 一致）
set -u

GPU=${1:-0}
REPO=/home/wangyuanshuo02/two-level-attention
PY=/home/fluentllmenv/bin/python3
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_PATH=/home/wangyuanshuo02/datasets/LongBench-v2/data.json
CONFIG=$REPO/benchmark/LongBench/config
OUT=$REPO/exp/results_longbench_sparq/Qwen3-8B/pred_sparq_1024_lbv2
TS=$(date +%Y%m%d-%H%M%S)
LOG=/tmp/sparq_lbv2_gpu${GPU}_$TS.log
echo "GPU$GPU LBv2 TS=$TS OUT=$OUT" | tee "$LOG"

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO" "$PY" -u "$REPO/benchmark/LongBench/pred.py" \
  --model Qwen3-8B --model_path "$MODEL_PATH" \
  --task lbv2 \
  --output-dir "$OUT" \
  --dataset-path "$DATA_PATH" \
  --config-path "$CONFIG" \
  --method sparq_vendor \
  --sparq_k 1024 \
  --t "$TS" \
  >> "$LOG" 2>&1
echo "=== LBv2 done rc=$? TS=$TS ===" | tee -a "$LOG"
