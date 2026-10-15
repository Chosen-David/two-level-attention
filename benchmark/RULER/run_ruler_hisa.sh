#!/bin/bash
# HISA RULER 批量派单（通用 vendor 模板预留；#23 身份核验+接入完成前勿跑）
# 状态：evaluation.md:113「HISA 待唯一身份核验」——先填准确题名/作者/固定论文版/官方仓库，
#       不能根据缩写猜成 HiP 或其他方法；官方入口行明确「HISA 身份未核，暂不填猜测链接」。
#       接入后按官方真实接口把 --hisa_* 参数绑定到 enable_hisa；主表预算口径对齐 K2=1024。
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
    # TODO(#23)：接入后绑定 enable_hisa 真实参数（如 --hisa_budget 等，依官方接口定名）
    PYTHONPATH=~/local_pylibs CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method hisa_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --hisa_budget $BUDGET --max-num $N $YARN 2>&1 | tail -5
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
