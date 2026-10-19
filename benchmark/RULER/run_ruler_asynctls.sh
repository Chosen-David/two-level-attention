#!/bin/bash
# AsyncTLS RULER 批量派单（C 轨语义复现，kt=512 对齐论文 RULER 口径）
# 用法：bash benchmark/RULER/run_ruler_asynctls.sh <gpu_id> <ts> <max_num> [lens...]
# 例：bash benchmark/RULER/run_ruler_asynctls.sh 0 10171700 100 "32768"
# 预算口径：--asynctls_token_budget 512（论文 RULER kt=512）；
#   --asynctls_block_size 64 --asynctls_top_kb 128（论文 §4 默认）。
# lens 超过原生 40960 自动加 --yarn。运行环境：fluentllmenv + PYTHONPATH=~/local_pylibs。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
CHAN=dimension_channels/qwen3-8b/qasper_channels.json
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
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T asynctls $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method asynctls_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --asynctls_channel_file $CHAN --asynctls_token_budget 512 \
      --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T asynctls rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE asynctls (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE asynctls"
