#!/bin/bash
# SparQ RULER 批量派单（vendor：官方源码复用，decode |Q| 的 r 最大坐标近似→top-k 细筛）
# 用法：bash benchmark/RULER/run_ruler_sparq.sh <gpu_id> <ts> <max_num> [lens...]
# 例：bash benchmark/RULER/run_ruler_sparq.sh 6 10160020 50 "65536 131072"
# 预算口径：--sparq_k 1024（主表固定 token 预算，与 Quest/TLI K2 对齐）；
#   r = max(1, round(head_dim/(context_length/(2*k)))) 随序列长度变化
#   （Qwen3-8B head_dim=128 → 64K: r=4；128K: r=2），--sparq_context_length 默认
#   None = 跟随 --context_length（r 自适应）。
# lens 超过原生 40960 自动加 --yarn。运行环境：fluentllmenv（flash-attn 已装），
#   PYTHONPATH=~/local_pylibs。
cd /home/wangyuanshuo02/two-level-attention || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
TS=$2
N=${3:-50}
LENS=${4:-"32768"}
TASKS=${5:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
PYBIN=${PYTHON_BIN:-python}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T sparq $YARN ==="
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method sparq_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --sparq_k 1024 --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T sparq rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE sparq (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE sparq"
