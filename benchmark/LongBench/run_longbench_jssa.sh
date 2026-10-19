#!/bin/bash
# JSSA LongBench 批量派单（#27 源码落地 + #28 接入完成，2026-10-18）
# 用法：bash benchmark/LongBench/run_longbench_jssa.sh <gpu_id> <ts> [tasks...]
# 预算口径：--jssa_budget 1024（主表 K2=1024，每 kv head top-B token）、rank 32。
# 校准基：bases/qwen3_8b.pt（非 yarn，LongBench 上下文不超 32K 原生）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
BASIS=${JSSA_BASIS_PATH:-/home/wangyuanshuo02/two-level-attention/bases/qwen3_8b.pt}
GPU=$1
TS=$2
TASKS=${3:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T jssa_vendor b=1024 ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method jssa_vendor --t $TS \
    --config-path benchmark/LongBench/config --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --jssa_budget 1024 --jssa_rank 32 --jssa_basis_path $BASIS 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T jssa_vendor rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE jssa LB (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE jssa LB"
