#!/bin/bash
# HISA LongBench 16 任务批量派单（#23 身份核验+接入完成，2026-10-17）
# 状态：同 run_ruler_hisa.sh——B 轨官方 kernel 语义 torch 等价（enable_hisa）；
#       块长 128 / 保留块 64 对齐官方默认，预算对齐 K2=1024。
# 用法：bash benchmark/LongBench/run_longbench_hisa.sh <gpu_id> <budget> <ts>
# 例：bash benchmark/LongBench/run_longbench_hisa.sh 0 1024 10160000
# 任务集与 MoBA/Quest/FASA/ClusterKV 主表 vendor 口径一致（16 任务，fasa_lb_16 同集）。
# 数据源：KVCache-Factory/data/LongBench（MoBA/FASA/Quest 同源，fingerprint 一致）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
GPU=$1
BUDGET=${2:-1024}
TS=$3
TASKS=${4:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique multi_news gov_report qmsum trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T budget=$BUDGET ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method hisa_vendor \
    --config-path benchmark/LongBench/config \
    --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --pred_postfix _hisa_b${BUDGET}_16 --t $TS \
    --hisa_budget $BUDGET --hisa_k_block_size 128 --hisa_block_topk 64 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T budget=$BUDGET rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE lb16_hisa_b${BUDGET} (FAILED=$FAILED/16)"
  exit 1
fi
echo "ALL DONE lb16_hisa_b${BUDGET}"
