#!/bin/bash
# Quest vendor Qwen3-32B RULER 32K/64K/128K 派单（.187 8 卡，3 波并行）
# 用户指令 2026-10-20：「8B 补齐 + 32B Quest 并行」；Quest 官方免校准可直接跑 32B。
# 用法：bash benchmark/RULER/run_ruler_quest_32b.sh
# 口径：quest_token_budget=1024、chunk=16（对齐 Quest 8B quest_vendor_1024_16 与
#       论文 §4）；每任务 100 样本（对齐 RULER 基线）；64K/128K 自动 --yarn
#       （QWEN3_NATIVE_MPE=40960，pred_ruler.py L154 强制）。
# PYTHONPATH=~/local_pylibs（flash-attn 编译产物）必须，否则 import flash_attn 失败。
set -u
REPO=/home/wangyuanshuo02/two-level-attention
MODEL_PATH=/home/wangyuanshuo02/Qwen3-32B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"
OUT=$REPO/exp/results_ruler/Qwen3-32B/quest_vendor_1024_16
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/tmp/qv_ruler_32b_${TS}
mkdir -p "$OUT" "$LOG_DIR"
echo "TS=$TS" | tee "$LOG_DIR/ts.txt"
FAILED=0

# 【2026-10-21 修复】32B 单进程 ~67GB 权重（device_map auto 落在 CUDA_VISIBLE_DEVICES
# 指定的单卡）→ 必须每卡一进程。旧版 11 任务一轮全发 + card=i%8 轮转 → 卡 0-2
# 各并发 2 进程 → 2×67GB=134GB OOM（首派 33/33 全崩，根因）。改 chunk=8：
# 每轮最多 8 并发（一卡一进程），轮间 wait；128K 段 32B 单进程实测峰值 ~134GB
# （模型 67 + prefill 激活），expandable_segments 抗碎片，仍一卡一进程。
run_len() {
  local L=$1
  local YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  local -a TA=($TASKS)
  local n=${#TA[@]}
  local start=0
  while [ "$start" -lt "$n" ]; do
    local end=$(( start + 8 ))
    [ "$end" -gt "$n" ] && end=$n
    local i=0
    for (( j=start; j<end; j++ )); do
      local T=${TA[$j]}
      local card=$i
      (
        echo "[$(date +%H:%M:%S)] L=$L task=$T card=$card start" >> "$LOG_DIR/card$card.log"
        PYTHONPATH=~/local_pylibs:$REPO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          CUDA_VISIBLE_DEVICES=$card python -u -m benchmark.RULER.pred_ruler \
          --model Qwen3-32B --model_path "$MODEL_PATH" \
          --task "$T" --context_length "$L" --method quest_vendor --t "$TS" \
          --data-root "$DATA" --output-dir "$OUT" \
          --quest_token_budget 1024 --quest_chunk_size 16 \
          --pred_postfix _1024 --max-num 100 $YARN \
          >> "$LOG_DIR/card$card.log" 2>&1
        local rc=$?
        echo "[$(date +%H:%M:%S)] L=$L task=$T card=$card done rc=$rc" >> "$LOG_DIR/card$card.log"
        if [ "$rc" -ne 0 ]; then echo "FAILED L=$L task=$T" >> "$LOG_DIR/failed.txt"; fi
      ) &
      i=$(( i + 1 ))
    done
    wait
    echo "[$(date +%H:%M:%S)] L=$L round[$start/$n] done" | tee -a "$LOG_DIR/../qv_ruler_32b_summary.txt"
    start=$end
  done
  echo "[$(date +%H:%M:%S)] L=$L ALL DONE" | tee -a "$LOG_DIR/../qv_ruler_32b_summary.txt"
}

for L in 32768 65536 131072; do
  run_len "$L"
done
if [ -s "$LOG_DIR/failed.txt" ]; then
  echo "ALL DONE quest_32b (FAILED see $LOG_DIR/failed.txt)"
  exit 1
fi
echo "ALL DONE quest_32b"
