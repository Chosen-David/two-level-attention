#!/usr/bin/env bash
set -euo pipefail
repo="$1"
out="$2"
log_dir="$3"
python_bin=/home/fluentllmenv/bin/python3
model=/home/wangyuanshuo02/Qwen3-8B
data=/home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER
export PYTHONPATH="$repo:/tmp/nfa_quality_deps"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
mkdir -p "$log_dir"
for task in niah_multikey_1 cwe; do
  "$python_bin" "$repo/benchmark/RULER/pred_ruler.py" \
    --model Qwen3-8B --model_path "$model" \
    --task "$task" --context_length 32768 \
    --data-root "$data" --output-dir "$out" \
    --t strict-skip-quality --pred_postfix _strict_pair --max-num 5 \
    --method nfa --nfa_prefill_frozen_regions \
    --nfa_frozen_region_mode dual --nfa_frozen_region_epsilon 0.05 \
    --nfa_frozen_region_statistic mean --nfa_frozen_region_rows 256 \
    --tia_block_size 64 --tia_level1_topk 128 --tia_level2_topk 1024 \
    --tia_level2_cmp_ratio 4 --nfa_subspace tail --nfa_l2_subspace same \
    --nfa_far_method minmax --nfa_near_method avg \
    --nfa_far_select 4bit --nfa_near_select 4bit \
    --nfa_near_pages 32 --nfa_near_tokens off --nfa_near_len 4096 \
    --nfa_sink_tokens 4 --nfa_swa_size 128 \
    --nfa_enable_subspace true --nfa_enable_layer_skip false \
    --nfa_far_skip_backend legacy --nfa_far_skip_cache retain \
    --nfa_per_q_head false --nfa_l2_quant false --nfa_boundary_bypass true \
    > "$log_dir/$task.log" 2>&1
done
"$python_bin" /tmp/score_strict_ruler_tasks.py \
  --repo "$repo" --root "$out" --data-root "$data" --model-path "$model" \
  > "$log_dir/formal_score.log" 2>&1
