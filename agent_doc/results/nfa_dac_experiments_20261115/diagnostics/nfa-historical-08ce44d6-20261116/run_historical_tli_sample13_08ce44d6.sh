#!/bin/bash
set -euo pipefail
ROOT=/tmp/nfa-historical-08ce44d6-20261116
OUT=${1:?usage: run_historical_tli_sample13_08ce44d6.sh OUT_DIR [GPU] [SUBSPACE]}
GPU=${2:-0}
SUBSPACE=${3:-tail}
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TLI_OPERATOR_MODE=off
export TLI_FINE_GATHER=0
export TLI_SPARSE_PREFILL=0
export TLI_DECODE_DENSE=0
if [[ "$SUBSPACE" == tail ]]; then
  ENABLE_SUBSPACE=true
elif [[ "$SUBSPACE" == full ]]; then
  ENABLE_SUBSPACE=false
else
  echo "unsupported SUBSPACE=$SUBSPACE" >&2
  exit 2
fi
exec env CUDA_VISIBLE_DEVICES="$GPU" python -u -m benchmark.RULER.pred_ruler \
  --model Qwen3-8B --model_path /home/wangyuanshuo02/Qwen3-8B \
  --task niah_multikey_2 --context_length 65536 --method tli \
  --t historical_sample13_mn128 \
  --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
  --output-dir "$OUT" --pred_postfix _historical_"$SUBSPACE" \
  --skip 13 --max-num 1 --yarn \
  --tia_block_size 64 --tia_level1_topk 128 --tia_level2_topk 1024 \
  --tia_level2_cmp_ratio 4 \
  --tli_enable_subspace "$ENABLE_SUBSPACE" --tli_subspace "$SUBSPACE" \
  --tli_l2_subspace same --tli_enable_layer_skip false \
  --tli_far_method minmax --tli_near_method avg \
  --tli_far_select 4bit --tli_near_select 4bit \
  --tli_near_pages 32 --tli_near_tokens off --tli_near_len 4096 \
  --tli_sink_tokens 4 --tli_swa_size 128 --tli_per_q_head false \
  --tli_l2_quant false \
  --tli_boundary_bypass true --tli_far_skip_backend legacy \
  --tli_far_skip_cache retain
