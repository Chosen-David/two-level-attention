#!/bin/bash
# 官方 Quest CUDA + 本地原生 GQA/接口适配的 Qwen3-8B RULER 派单。
# 用法：bash benchmark/RULER/run_ruler_quest.sh <gpu_id> <ts> <max_num> [lens] [tasks]
set -u
REPO=/home/wangyuanshuo02/two-level-attention
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
TS=$2
N=${3:-500}
LENS=${4:-"32768"}
TASKS=${5:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
OUT=${RULER_OUTPUT_ROOT:-$REPO/exp/results_ruler/Qwen3-8B}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T official_quest_b=1024 $YARN ==="
    PYTHONPATH=$REPO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.official_baselines.ruler_official \
      --method quest --model Qwen3-8B --model-path $MODEL \
      --task $T --context-length $L --t $TS \
      --data-root $DATA --output-dir $OUT \
      --budget 1024 --quest-page-size 16 --max-num $N $YARN
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T official_quest_b=1024 rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE quest_official (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE quest_official"
