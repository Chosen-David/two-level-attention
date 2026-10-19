#!/bin/bash
# AsyncTLS LongBench 批量派单（C 轨语义复现，kt=1024 对齐论文 LongBench 口径）
# 用法：bash benchmark/LongBench/run_longbench_asynctls.sh <gpu_id> <ts>
# 预算口径：--asynctls_token_budget 1024（论文 LongBench kt=1024）；
#   --asynctls_block_size 64 --asynctls_top_kb 128（论文 §4 默认）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
CHAN=dimension_channels/qwen3-8b/qasper_channels.json
GPU=$1
TS=$2
TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T asynctls ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method asynctls_vendor --t $TS \
    --config-path benchmark/LongBench/config --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --asynctls_channel_file $CHAN --asynctls_token_budget 1024 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T asynctls rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE asynctls LB (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE asynctls LB"
