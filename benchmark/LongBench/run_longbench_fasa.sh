#!/bin/bash
# FASA LongBench 16 任务批量派单
# 用法：bash benchmark/LongBench/run_longbench_fasa.sh <gpu_id> <ts>
# FASA 官方 default: fasa_budget=1024, fasa_threshold=0.8, fasa_k=16
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
GPU=$1
TS=$2
TASKS=${3:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique multi_news gov_report qmsum trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T fasa_lb ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method fasa_vendor \
    --config-path benchmark/LongBench/config \
    --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B/pred_fasa_lb \
    --t $TS \
    --fasa_dataset qasper 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE fasa_lb (FAILED=$FAILED/16)"
  exit 1
fi
echo "ALL DONE fasa_lb"
