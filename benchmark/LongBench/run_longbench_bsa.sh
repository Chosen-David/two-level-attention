#!/bin/bash
# 单池 BSA（TLI 单池变体）LongBench 派单：单一 method（minmax/avg）两级筛选
# 用户指令 2026-10-21：「增加单池子 BSA（单一 method,minmax/avg 的两级筛选），出 Qwen8B 数据」。
# 预算与 TLI 主方法同款（bs=64、K1=128、K2=1024、sink=4、swa=128）；LongBench 不需要 YaRN。
# 用法：bash benchmark/LongBench/run_longbench_bsa.sh <gpu_id> <ts> [method]
# method 默认 minmax；avg 变体用第 3 参。单卡串行 16 任务。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench
GPU=$1
TS=$2
METHOD=${3:-minmax}
TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
FAILED=0
for T in $TASKS; do
  echo "=== [$(date +%H:%M:%S)] task=$T bsa-$METHOD ==="
  PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.LongBench.pred \
    --model Qwen3-8B --model_path $MODEL \
    --task $T --method tli --t $TS \
    --config-path benchmark/LongBench/config --dataset-path $DATA \
    --output-dir exp/results_longbench_vendor/Qwen3-8B \
    --tia_level1_topk 128 --tia_level2_topk 1024 \
    --tli_alpha 0 --tli_beta 0 --tli_gamma 0.625 \
    --tli_far_method $METHOD \
    --tli_enable_kmeans false --tli_enable_layer_skip false \
    --tli_subspace tail --tia_level2_cmp_ratio 4 \
    --tli_sink_tokens 4 --tli_swa_size 128 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "==== FAILED task=$T bsa-$METHOD rc=$rc ===="
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE bsa-$METHOD LB (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE bsa-$METHOD LB"
