#!/bin/bash
# SparQ vendor RULER 批量派单（主表口径：k=1024 固定；r 随 L 档位反推）
# 用法: bash run_ruler_sparq.sh <ctx> <num> "<task list>" "<gpu list>"
#   ctx : 32768 / 65536 / 131072（>40960 自动 --yarn）
#   num : 每任务样本数（32K 档 100；64K/128K 按派单口径）
#   例  : bash run_ruler_sparq.sh 32768 100 \
#          "niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt" \
#          "0 1"
set -euo pipefail

CTX=$1
NUM=$2
TASKS=$3
GPUS=$4

MODEL_PATH=/home/wangyuanshuo02/Qwen3-8B
DATA_ROOT=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
REPO=/home/wangyuanshuo02/two-level-attention
OUT=$REPO/exp/trace/results/e119_ruler${CTX}_formal_sparq_vendor
TS=$(date +%Y%m%d-%H%M%S)

gpu_arr=(${GPUS//,/ })
yarn_flag=""
if [ "$CTX" -gt 40960 ]; then yarn_flag="--yarn"; fi
i=0
for task in $TASKS; do
  gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}
  echo "[sparq] GPU$gpu $task L=$CTX n=$NUM ts=$TS"
  CUDA_VISIBLE_DEVICES=$gpu nohup /home/fluentllmenv/bin/python3 -u \
    $REPO/benchmark/RULER/pred_ruler.py \
    --model Qwen3-8B --model_path $MODEL_PATH \
    --task $task --context_length $CTX $yarn_flag \
    --data-root $DATA_ROOT --output-dir $OUT --t $TS --max-num $NUM \
    --method sparq_vendor --sparq_k 1024 \
    > /tmp/sparq_${CTX}_${task}.log 2>&1 &
  i=$((i+1))
done
echo "[sparq] dispatched $i tasks -> $OUT ts=$TS"
