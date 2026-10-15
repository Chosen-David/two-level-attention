#!/bin/bash
# MoBA LongBench 16 任务批量派单（主表 1024 / 附录A 官方默认）
# 用法：bash benchmark/LongBench/run_longbench_moba.sh <gpu_id> <chunk> <topk> <ts>
# 例：bash benchmark/LongBench/run_longbench_moba.sh 0 256 4 10150656
#   --moba_chunk_size 256 --moba_topk 4 → 预算 1024（主表口径，严格对齐 K2）
#   --moba_chunk_size 4096 --moba_topk 12 → 预算 49152（官方默认，附录 A）
# 任务集与 Quest/FASA 主表 vendor 口径一致（16 任务，fasa_lb_16 同集）：
#   narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique multi_news
#   gov_report qmsum trec triviaqa samsum passage_count passage_retrieval_en
#   lcc repobench-p
# 数据源：KVCache-Factory/data/LongBench（FASA/Quest 同源，fingerprint 一致）
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
GPU=$1
CHUNK=$2
TOPK=$3
TS=$4
TASKS=${5:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique multi_news gov_report qmsum trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
# 远程机默认 python 是 3.9，flash-attn 编译产物是 cpython-311 → 用
# PYTHON_BIN 显式指定（远程传 /home/fluentllmenv/bin/python，本机默认 python）
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T chunk=$CHUNK topk=$TOPK ==="
  # -m 方式运行（pred.py 顶部 `from benchmark.LongBench.lbv2_choice import ...`，
  # 须 repo 根在 sys.path；直接路径方式 sys.path[0]=脚本目录会 ModuleNotFoundError）
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method moba_vendor \
    --config-path benchmark/LongBench/config \
    --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --pred_postfix _moba_c${CHUNK}_k${TOPK}_16 --t $TS \
    --moba_chunk_size $CHUNK --moba_topk $TOPK 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T chunk=$CHUNK topk=$TOPK rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE lb16_moba_c${CHUNK}_k${TOPK} (FAILED=$FAILED/16)"
  exit 1
fi
echo "ALL DONE lb16_moba_c${CHUNK}_k${TOPK}"
