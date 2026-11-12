#!/usr/bin/env bash
# Qwen3-8B × LongBench v1 英文16任务：三 baseline 同预算完整请求性能。
# 用法：bash benchmark/LongBench/run_longbench_v1_vendor_perf.sh \
#   <quest|jssa|asynctls> <gpu_id> <timestamp> [clean-e2e|operator-profile]
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <quest|jssa|asynctls> <gpu_id> <timestamp> [clean-e2e|operator-profile]" >&2
  exit 2
fi
ARM=$1
GPU=$2
TS=$3
MODE=${4:-clean-e2e}
case "$ARM" in
  quest) METHOD=quest_vendor_kernel ;;
  jssa) METHOD=jssa_official ;;
  asynctls) METHOD=asynctls_vendor ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
if [[ "$MODE" != clean-e2e && "$MODE" != operator-profile ]]; then
  echo "unknown measurement mode: $MODE" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
MODEL_PATH=${MODEL_PATH:-/home/wangyuanshuo02/Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/LongBench}
PYBIN=${PYTHON_BIN:-/home/fluentllmenv/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO_ROOT/agent_doc/results/qwen3_lbv1_vendor_perf_$TS}
TASKS=${TASKS:-"narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"}
OUT=$OUT_ROOT/$ARM/$MODE.json

# 计时进程必须独占目标卡；只拒绝点火，绝不清理其他进程。
BUSY=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
if [[ -n "$BUSY" ]]; then
  echo "GPU $GPU has compute processes; refusing non-exclusive timing: $BUSY" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$REPO_ROOT:$HOME/local_pylibs${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u -m benchmark.LongBench.perf_vendor \
  --method "$METHOD" --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --config-path "$REPO_ROOT/benchmark/LongBench/config" \
  --output "$OUT" --measurement-mode "$MODE" --tasks $TASKS
