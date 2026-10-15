#!/bin/bash
# ClusterKV LongBench 16 任务批量派单（vendor 接入后使用；#16 未接入前勿跑）
# 用法：bash benchmark/LongBench/run_longbench_clusterkv.sh <gpu_id> <token_budget> <sink> <window> <ts>
# 例：bash benchmark/LongBench/run_longbench_clusterkv.sh 0 1024 16 320 10160000
#   --clusterkv_token_budget 1024 --clusterkv_sink 16 --clusterkv_window 320
#   = 官方默认口径（同 run_ruler_clusterkv.sh 说明，接入验收时按总预算换算对齐）。
# 任务集与 MoBA/Quest/FASA 主表 vendor 口径一致（16 任务，fasa_lb_16 同集）。
# 数据源：KVCache-Factory/data/LongBench（MoBA/FASA/Quest 同源，fingerprint 一致）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
GPU=$1
TB=$2
SINK=$3
WIN=$4
TS=$5
TASKS=${6:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique multi_news gov_report qmsum trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T tb=$TB sink=$SINK win=$WIN ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method clusterkv_vendor \
    --config-path benchmark/LongBench/config \
    --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --pred_postfix _clusterkv_tb${TB}_s${SINK}_w${WIN}_16 --t $TS \
    --clusterkv_token_budget $TB --clusterkv_sink $SINK --clusterkv_window $WIN \
    --clusterkv_nlist 400 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T tb=$TB sink=$SINK win=$WIN rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE lb16_clusterkv_tb${TB}_s${SINK}_w${WIN} (FAILED=$FAILED/16)"
  exit 1
fi
echo "ALL DONE lb16_clusterkv_tb${TB}_s${SINK}_w${WIN}"
