#!/usr/bin/env bash
set -euo pipefail
WT="$HOME/tla-lb-k2-20261117-v2"
EXPECTED_HEAD=8736a965e903fd461ef5591347933205e18adee0
ROOT="$HOME/ak_verify_results/qwen3_lb_k2_ablation_8736a965_20261117"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/LongBench"
mkdir -p "$ROOT/_logs" "$ROOT/_status"
if [[ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  printf 'unexpected source HEAD\n' >&2
  exit 3
fi
if [[ -n "$(git -C "$WT" status --short)" ]]; then
  printf 'source worktree is not clean\n' >&2
  exit 3
fi
run_cell() {
  local gpu="$1" k="$2"
  local out="$ROOT/k2_${k}"
  local log="$ROOT/_logs/k2_${k}_gpu${gpu}.log"
  local donef="$ROOT/_status/k2_${k}_gpu${gpu}.done"
  if [[ -e "$donef" ]]; then
    printf 'skip k2=%s gpu=%s already done\n' "$k" "$gpu" >&2
    return 0
  fi
  if [[ -e "$out/quality_manifest_s0of1.json" ]]; then
    printf 'skip k2=%s existing manifest\n' "$k" >&2
    return 0
  fi
  printf 'START k2=%s gpu=%s pid=%s time=%s\n' "$k" "$gpu" "$$" "$(date -Is)" >"$log"
  env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    /usr/bin/python -m benchmark.efficiency.benchmark_longbench_quality \
    --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 \
    --nfa-gate off --nfa-level2-topk "$k" \
    --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
    --data-root "$DATA" --output-dir "$out" \
    --tasks hotpotqa gov_report --device cuda:0 --visible-len 8192 >>"$log" 2>&1
  if [[ ! -f "$out/quality_manifest_s0of1.json" ]]; then
    printf 'k2=%s exited without manifest\n' "$k" >&2
    exit 6
  fi
  printf 'DONE k2=%s gpu=%s time=%s\n' "$k" "$gpu" "$(date -Is)" >"$donef"
}
run_cell 0 256 &
run_cell 0 512 &
run_cell 1 1024 &
run_cell 1 2048 &
wait
