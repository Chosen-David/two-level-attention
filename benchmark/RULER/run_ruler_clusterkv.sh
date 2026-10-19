#!/bin/bash
# ClusterKV RULER 批量派单（vendor 接入后使用；#16 未接入前勿跑）
# 用法：bash benchmark/RULER/run_ruler_clusterkv.sh <gpu_id> <token_budget> <sink> <window> <ts> <max_num> [lens...] [tasks...]
# 例：bash benchmark/RULER/run_ruler_clusterkv.sh 0 1024 16 320 10160000 500 "32768 65536 131072"
#   --clusterkv_token_budget 1024 --clusterkv_sink 16 --clusterkv_window 320
#   = 官方默认口径（ClusterKV accuracy/patch.py: token_budget=1024, sink=16,
#   nlist=400, fit_iter=20, head_sel=truc, balance=False, dist_t=cosine；
#   window 仅 statistics/换算用，accuracy selector 路径不施加 window）。
#   注意预算语义：官方 controller 的 kv_cache_size = sink + token_budget + window，
#   token_budget 是中间选择预算，sink/window 是额外的（不同于 TLI 的 K2 已含 sink/swa）。
#   接入验收时须按 evaluation.md「记录实际 unique token 数」口径换算对齐总预算 1024。
# 附录档：官方论文 LongBench/PG19 的预算扫描（256/512/1024/2048）另行派单，不入主表。
# lens 超过原生 40960 自动加 --yarn（同 run_ruler_moba.sh 做法）。
# 运行环境：source ckv_env.sh（LD_PRELOAD clusterkv cublas12.9 + pylibraft 24.10 完整
# 绑定 + fluentllmenv torch 共存；PYTHONPATH 已含 clusterkv site-packages 与 local_pylibs）。
cd /home/wangyuanshuo02/two-level-attention || exit 1
source exp/vendor_baselines/clusterkv_vendor/ckv_env.sh || exit 1
MODEL=/home/wangyuanshuo02/Qwen3-8B
DATA=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
GPU=$1
TB=$2
SINK=$3
WIN=$4
TS=$5
N=${6:-500}
LENS=${7:-"32768"}
TASKS=${8:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue cwe fwe vt"}
PYBIN=${PYTHON_BIN:-$CKV_PYTHON}
FAILED=0
for L in $LENS; do
  YARN=""
  if [ "$L" -gt 40960 ]; then YARN="--yarn"; fi
  for T in $TASKS; do
    echo "=== [$(date +%H:%M:%S)] L=$L task=$T tb=$TB sink=$SINK win=$WIN $YARN ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYBIN -u -m benchmark.RULER.pred_ruler \
      --model Qwen3-8B --model_path $MODEL \
      --task $T --context_length $L --method clusterkv_vendor --t $TS \
      --data-root $DATA \
      --output-dir exp/results_ruler/Qwen3-8B \
      --clusterkv_token_budget $TB --clusterkv_sink $SINK --clusterkv_window $WIN \
      --clusterkv_nlist 400 --max-num $N $YARN 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "==== FAILED L=$L task=$T tb=$TB sink=$SINK win=$WIN rc=$rc ===="
      FAILED=$((FAILED+1))
    fi
  done
done
if [ "$FAILED" -ne 0 ]; then
  echo "ALL DONE clusterkv_tb${TB}_s${SINK}_w${WIN} (FAILED=$FAILED)"
  exit 1
fi
echo "ALL DONE clusterkv_tb${TB}_s${SINK}_w${WIN}"
