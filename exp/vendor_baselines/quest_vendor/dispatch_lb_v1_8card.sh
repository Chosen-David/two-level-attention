#!/bin/bash
# Quest vendor（B 轨官方源码）LongBench v1 16 任务新口径派单（.24 8 卡，每卡 2 任务串行）
# 用法：bash dispatch_lb_v1_8card.sh
set -u

REPO=/home/wangyuanshuo02/two-level-attention
MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_DIR=/home/wangyuanshuo02/datasets/LongBench/data
CONFIG=$REPO/benchmark/LongBench/config
OUT=$REPO/exp/results_longbench_vendor/Qwen3-8B/pred_qv_1024_16
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/qv_lb_v1_${TS}
mkdir -p "$OUT" "$LOG_DIR"

# 8 卡 × 2 任务
declare -A CARD_TASKS
CARD_TASKS[0]="narrativeqa qasper"
CARD_TASKS[1]="multifieldqa_en hotpotqa"
CARD_TASKS[2]="2wikimqa musique"
CARD_TASKS[3]="gov_report qmsum"
CARD_TASKS[4]="multi_news trec"
CARD_TASKS[5]="triviaqa samsum"
CARD_TASKS[6]="passage_count passage_retrieval_en"
CARD_TASKS[7]="lcc repobench-p"

for card in 0 1 2 3 4 5 6 7; do
  (
    for task in ${CARD_TASKS[$card]}; do
      echo "[$(date +%H:%M:%S)] card=$card task=$task start" >> "$LOG_DIR/card$card.log"
      CUDA_VISIBLE_DEVICES=$card PYTHONPATH="$REPO" python3 -u "$REPO/benchmark/LongBench/pred.py" \
        --model Qwen3-8B \
        --model_path "$MODEL_PATH" \
        --task "$task" \
        --output-dir "$OUT" \
        --dataset-path "$DATA_DIR" \
        --config-path "$CONFIG" \
        --method quest_vendor \
        --quest_token_budget 1024 \
        --quest_chunk_size 16 \
        --t "$TS" \
        >> "$LOG_DIR/card$card.log" 2>&1
      echo "[$(date +%H:%M:%S)] card=$card task=$task done rc=$?" >> "$LOG_DIR/card$card.log"
    done
  ) &
done
wait
echo "ALL DONE at $(date +%H:%M:%S)  TS=$TS" | tee -a "$LOG_DIR/../qv_lb_v1_summary.txt"
