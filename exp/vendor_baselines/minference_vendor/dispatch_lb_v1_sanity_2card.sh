#!/bin/bash
# MInference vendor（B 轨官方源码）LongBench v1 sanity 对拍派单（本地 2 卡）。
# 前置：正式 pattern 已搜出（qwen3_minference_search.py 输出
#   exp/vendor_baselines/minference_vendor/best_pattern_qwen3_8b.json）
# 用法：bash dispatch_lb_v1_sanity_2card.sh
set -u

REPO=/home/wangyuanshuo02/two-level-attention
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_DIR=/home/wangyuanshuo02/datasets/LongBench/data
CONFIG=$REPO/benchmark/LongBench/config
PATTERN=$REPO/exp/vendor_baselines/minference_vendor/best_pattern_qwen3_8b.json
OUT=$REPO/exp/results_longbench_vendor/Qwen3-8B/pred_minf_v1_sanity
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/minf_lb_v1_${TS}
mkdir -p "$OUT" "$LOG_DIR"

# 本地 GPU0/1 各 1 任务（sanity：qasper 短检索 + multi_news 长摘要，对拍 dense 近无损）
declare -A CARD_TASKS
CARD_TASKS[0]="qasper"
CARD_TASKS[1]="multi_news"

for card in 0 1; do
  (
    for task in ${CARD_TASKS[$card]}; do
      echo "[$(date +%H:%M:%S)] card=$card task=$task start" >> "$LOG_DIR/card$card.log"
      CUDA_VISIBLE_DEVICES=$card PYTHONPATH="$REPO" /home/fluentllmenv/bin/python3 -u "$REPO/benchmark/LongBench/pred.py" \
        --model Qwen3-8B \
        --model_path "$MODEL_PATH" \
        --task "$task" \
        --output-dir "$OUT" \
        --dataset-path "$DATA_DIR" \
        --config-path "$CONFIG" \
        --method minference_vendor \
        --minference_config_path "$PATTERN" \
        --minference_starting_layer -1 \
        --t "$TS" \
        >> "$LOG_DIR/card$card.log" 2>&1
      echo "[$(date +%H:%M:%S)] card=$card task=$task done rc=$?" >> "$LOG_DIR/card$card.log"
    done
  ) &
done
wait
echo "ALL DONE at $(date +%H:%M:%S)  TS=$TS" | tee -a "$LOG_DIR/../minf_lb_v1_summary.txt"
