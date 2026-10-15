#!/bin/bash
# MoBA RULER 批量派单（主表 1024 / 附录A 官方默认）
# 用法：bash benchmark/RULER/run_ruler_moba.sh <gpu_id> <chunk> <topk> <ts> <max_num> [lens...]
# 例：bash benchmark/RULER/run_ruler_moba.sh 0 256 4 10150000 500 "32768 65536 131072"
#   --moba_chunk_size 256 --moba_topk 4 → 预算 1024（主表口径，严格对齐 K2）
#   --moba_chunk_size 4096 --moba_topk 12 → 预算 49152（官方默认，附录 A）
# lens 超过原生 40960 自动加 --yarn（同 run_ruler_e109.sh 做法）。
# 注意：PYTHONPATH=~/local_pylibs 必须（本机 flash-attn 编译产物所在，
#   site-packages 归 root 无写权限故装到用户目录）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
CHUNK=$2
TOPK=$3
TS=$4
N=${5:-500}
LENS=${6:-"32768"}
TASKS=${7:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
# 远程机默认 python 是 3.9，flash-attn 编译产物是 cpython-311 → 用
# PYTHON_BIN 显式指定（远程传 /home/fluentllmenv/bin/python，本机默认 python）
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T chunk=$CHUNK topk=$TOPK $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method moba_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --moba_chunk_size $CHUNK --moba_topk $TOPK --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T chunk=$CHUNK topk=$TOPK rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE moba_c${CHUNK}_k${TOPK} (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE moba_c${CHUNK}_k${TOPK}"
