#!/bin/bash
# HISA LongBench 16 任务批量派单（通用 vendor 模板预留；#23 身份核验+接入完成前勿跑）
# 状态：同 run_ruler_hisa.sh——HISA 身份未核，接入后按官方真实接口绑定 enable_hisa。
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
  # TODO(#23)：接入后绑定 enable_hisa 真实参数（如 --hisa_budget 等，依官方接口定名）
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method hisa_vendor \
    --config-path benchmark/LongBench/config \
    --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --pred_postfix _hisa_b${BUDGET}_16 --t $TS \
    --hisa_budget $BUDGET 2>&1 | tail -4
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
