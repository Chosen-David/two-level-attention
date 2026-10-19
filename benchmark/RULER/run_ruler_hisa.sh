#!/bin/bash
# HISA RULER 批量派单（#23 身份核验+接入完成，2026-10-17）
# 状态：HISA=COLM 2026 / arXiv:2603.28458v4（Xu 等 14 人，MuLabPKU）。B 轨官方 kernel
#       语义 torch 等价（qwen3_hisa_patch.py，enable_hisa）；官方 TileLang kernels
#       在 third_party/tilelang_dsa_hisa/（MIT，pinned 7940102813）。主表预算对齐
#       K2=1024（--hisa_budget），块长 128 / 保留块 64 对齐官方默认。
# 用法：bash benchmark/RULER/run_ruler_hisa.sh <gpu_id> <budget> <ts> <max_num> [lens...] [tasks...]
# 例：bash benchmark/RULER/run_ruler_hisa.sh 0 1024 10160000 500 "32768 65536 131072"
# lens 超过原生 40960 自动加 --yarn（同 run_ruler_moba.sh 做法）。
# 注意：PYTHONPATH=~/local_pylibs 必须（本机 flash-attn 编译产物所在）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
BUDGET=${2:-1024}
TS=$3
N=${4:-500}
LENS=${5:-"32768"}
TASKS=${6:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T budget=$BUDGET $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method hisa_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --hisa_budget $BUDGET --hisa_k_block_size 128 --hisa_block_topk 64 \
      --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T budget=$BUDGET rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE hisa_b${BUDGET} (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE hisa_b${BUDGET}"
