#!/bin/bash
# FASA RULER 批量派单（B 轨官方源码复用，fasa_budget=1024 对齐主表 K2）
# 用法：bash benchmark/RULER/run_ruler_fasa.sh <gpu_id> <ts> <max_num> [lens...]
# 例：bash benchmark/RULER/run_ruler_fasa.sh 0 10181810 100 "65536"
# lens 超过原生 40960 自动加 --yarn（同 run_ruler_asynctls.sh 做法）。
# agreement 校准产物：dimension_rankings/Qwen3-8B/qasper_agreement.pkl（qasper 数据集）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
TS=$2
N=${3:-100}
LENS=${4:-"32768"}
TASKS=${5:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T fasa $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method fasa_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --fasa_budget 1024 --fasa_dataset qasper \
      --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T fasa rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE fasa (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE fasa"
