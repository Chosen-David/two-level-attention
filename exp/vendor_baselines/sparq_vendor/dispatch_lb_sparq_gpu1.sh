#!/bin/bash
# SparQ vendor LongBench v1 GPU1 8 任务派单（等待 Opus E138 smoke 释放 GPU1 后执行）
# 用法：bash dispatch_lb_sparq_gpu1.sh
set -u

REPO=/home/wangyuanshuo02/two-level-attention
PY=/home/fluentllmenv/bin/python3
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_DIR=/home/wangyuanshuo02/datasets/LongBench/data
CONFIG=$REPO/benchmark/LongBench/config
OUT=$REPO/exp/results_longbench_sparq/Qwen3-8B/pred_sparq_1024
TS=$(date +%Y%m%d-%H%M%S)
LOG=/tmp/sparq_lb_gpu1_$TS.log
echo "GPU1 TS=$TS OUT=$OUT" | tee "$LOG"

for task in multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p; do
  echo "=== GPU1 $task start $(date +%H:%M:%S) ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" "$PY" -u "$REPO/benchmark/LongBench/pred.py" \
    --model Qwen3-8B --model_path "$MODEL_PATH" \
    --task "$task" \
    --output-dir "$OUT" \
    --dataset-path "$DATA_DIR" \
    --config-path "$CONFIG" \
    --method sparq_vendor \
    --sparq_k 1024 \
    --t "$TS" \
    >> "$LOG" 2>&1
  echo "=== GPU1 $task done rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done
echo "GPU1 ALL DONE TS=$TS" | tee -a "$LOG"
