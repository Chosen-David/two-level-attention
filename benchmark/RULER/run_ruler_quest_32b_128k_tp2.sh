#!/bin/bash
# Quest vendor Qwen3-32B RULER 128K TP2 兜底派单（.187 8 卡 = 4 对并行）
# 背景 2026-10-21：32B 单卡 128K 实测峰值 ~134GB/139GB 临界（首派 2 进程并发 OOM
# 崩 33/33；重派 1 卡 1 进程 + expandable_segments 后 32K/64K 正常，128K 仍临界）。
# 本脚本把 128K 段改为 2 GPU/任务（CUDA_VISIBLE_DEVICES="g0,g1" 成对；device_map=
# auto 自动把 67GB 权重分摊两卡 → 每卡 ~34GB + 激活余量大）→ 4 并发 × 3 轮。
# 用法：bash benchmark/RULER/run_ruler_quest_32b_128k_tp2.sh [ts] [tasks...]
# 无 tasks 时跑默认 11 任务；已有完成产物且 --max-num 对齐时跳过（同名文件存在）。
set -u
REPO=/home/wangyuanshuo02/two-level-attention
MODEL_PATH=/home/wangyuanshuo02/Qwen3-32B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="${2:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt}"
TS=${1:-$(date +%Y%m%d-%H%M%S)}
OUT=$REPO/exp/results_ruler/Qwen3-32B/quest_vendor_1024_16
LOG_DIR=/tmp/qv_ruler_32b_128k_tp2_${TS}
mkdir -p "$OUT" "$LOG_DIR"
echo "TS=$TS" | tee "$LOG_DIR/ts.txt"
PAIRS=("0,1" "2,3" "4,5" "6,7")
FAILED=0

# 成对并发：一次最多 4 进程（8 卡 / 2）
run_batch() {
  local -a TA=("$@")
  local n=${#TA[@]}
  local start=0
  while [ "$start" -lt "$n" ]; do
    local end=$(( start + 4 ))
    [ "$end" -gt "$n" ] && end=$n
    local i=0
    for (( j=start; j<end; j++ )); do
      local T=${TA[$j]}
      local GPUS=${PAIRS[$i]}
      (
        echo "[$(date +%H:%M:%S)] L=131072 task=$T gpus=$GPUS start" >> "$LOG_DIR/pair$i.log"
        PYTHONPATH=~/local_pylibs:$REPO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          CUDA_VISIBLE_DEVICES=$GPUS python -u -m benchmark.RULER.pred_ruler \
          --model Qwen3-32B --model_path "$MODEL_PATH" \
          --task "$T" --context_length 131072 --method quest_vendor --t "$TS" \
          --data-root "$DATA" --output-dir "$OUT" \
          --quest_token_budget 1024 --quest_chunk_size 16 \
          --pred_postfix _1024 --max-num 100 --yarn \
          >> "$LOG_DIR/pair$i.log" 2>&1
        local rc=$?
        echo "[$(date +%H:%M:%S)] L=131072 task=$T gpus=$GPUS done rc=$rc" >> "$LOG_DIR/pair$i.log"
        if [ "$rc" -ne 0 ]; then echo "FAILED 131072 task=$T" >> "$LOG_DIR/failed.txt"; fi
      ) &
      i=$(( i + 1 ))
    done
    wait
    echo "[$(date +%H:%M:%S)] 128K round[$start/$n] done" | tee -a "$LOG_DIR/../qv_ruler_32b_128k_tp2_summary.txt"
    start=$end
  done
}

# 过滤：已存在同名完成 jsonl（ts 不同但任务已完成）则跳过
run_batch $TASKS
if [ -s "$LOG_DIR/failed.txt" ]; then
  echo "ALL DONE quest_32b_128k_tp2 (FAILED see $LOG_DIR/failed.txt)"
  exit 1
fi
echo "ALL DONE quest_32b_128k_tp2 TS=$TS"
