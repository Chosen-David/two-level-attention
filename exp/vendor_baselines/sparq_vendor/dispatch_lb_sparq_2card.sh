#!/bin/bash
# SparQ vendor LongBench v1 16 任务派单（GPU0 + GPU1 各 8 任务并行）
# 主表口径（用户 2026-10-14 拍板）：k 固定 1024；r 随序列长度反推
# （context_length 默认模型 cap 31500 → Qwen3-8B r=8）。
# 用法：bash dispatch_lb_sparq_2card.sh
set -u

REPO=/home/wangyuanshuo02/two-level-attention
PY=/home/fluentllmenv/bin/python3
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_DIR=/home/wangyuanshuo02/datasets/LongBench/data
CONFIG=$REPO/benchmark/LongBench/config
OUT=$REPO/exp/results_longbench_sparq/Qwen3-8B/pred_sparq_1024
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/sparq_lb_${TS}
mkdir -p "$LOG_DIR"
echo "TS=$TS OUT=$OUT" | tee "$LOG_DIR/summary.txt"

run_one() { # $1=gpu $2=task
  local gpu=$1 task=$2
  echo "[$(date +%H:%M:%S)] GPU$gpu $task start" >> "$LOG_DIR/card$gpu.log"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO" "$PY" -u "$REPO/benchmark/LongBench/pred.py" \
    --model Qwen3-8B --model_path "$MODEL_PATH" \
    --task "$task" \
    --output-dir "$OUT" \
    --dataset-path "$DATA_DIR" \
    --config-path "$CONFIG" \
    --method sparq_vendor \
    --sparq_k 1024 \
    --t "$TS" \
    >> "$LOG_DIR/card$gpu.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu $task done rc=$?" >> "$LOG_DIR/card$gpu.log"
}

# GPU0：8 任务
(
  for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum; do
    run_one 0 "$task"
  done
) &

# GPU1：8 任务
(
  for task in multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p; do
    run_one 1 "$task"
  done
) &

wait
echo "ALL DONE at $(date +%H:%M:%S)  TS=$TS" | tee -a "$LOG_DIR/summary.txt"
