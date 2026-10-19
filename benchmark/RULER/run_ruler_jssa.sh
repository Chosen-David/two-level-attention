#!/bin/bash
# JSSA RULER 批量派单（#27 源码落地 + #28 接入完成，2026-10-18）
# 状态：JSSA=Joint Subspace Sparse Attention（arXiv:2610.19026v1）。B 轨官方 vLLM
#       patch 语义 torch 等价（qwen3_jssa_patch.py，enable_jssa）；官方校准基
#       M=TopEig_r(G_Qm/tr+G_K/tr) 由 jssa_calibrate.py 产出（bases/qwen3_8b.pt）。
#       主表预算对齐 K2=1024（--jssa_budget 每 kv head top-B token），rank 32。
# 用法：bash benchmark/RULER/run_ruler_jssa.sh <gpu_id> <ts> <max_num> [lens...] [tasks...]
# 例：bash benchmark/RULER/run_ruler_jssa.sh 0 10181900 500 "32768"
# lens 超过原生 40960 自动加 --yarn（同 run_ruler_moba.sh 做法）。
# 注意：PYTHONPATH=~/local_pylibs 必须（本机 flash-attn 编译产物所在）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
TS=$2
N=${3:-500}
LENS=${4:-"32768"}
TASKS=${5:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
BASIS=${JSSA_BASIS_PATH:-/home/wangyuanshuo02/two-level-attention/bases/qwen3_8b.pt}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T jssa_b=1024 $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method jssa_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --jssa_budget 1024 --jssa_rank 32 --jssa_basis_path $BASIS \
      --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T jssa_b=1024 rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE jssa_vendor (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE jssa_vendor"
